"""Record the AutoScript calls of Thermo's milling methods, old and through the service.

Run as a script, in its own interpreter, for the same reason as
``autoscript_beam_parity.py``, whose fake SDK, microscope and recorder it reuses. It
writes JSON to the path it is given: ``cases``, each holding what a milling method
returned, the SDK calls and writes it made and the messages it logged, on a
microscope with its beams built and no milling service ("old") and on one with the
service built as connect builds it ("new"); and ``facts``, what the new API makes of
the service.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_beam_parity as B  # noqa: E402  (installs the fake SDK)

from fibsem.structures import (  # noqa: E402
    BeamType,
    CrossSectionPattern,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
)

S, LOG = B.S, B.LOG

APPLICATION_FILES = ["Si", "Si-ccs", "Si-multipass", "Al"]


class FakePattern(S.Node):
    """A pattern the SDK made: the attributes the drawing code reads."""


def _pattern(path):
    pattern = FakePattern(path)
    for name, value in (("dwell_time", 1e-6), ("pass_count", 10), ("time", 12.5)):
        object.__setattr__(pattern, name, value)
    return pattern


class FakePatterning(S.Node):
    """``connection.patterning``: records calls; makes patterns; reports a state."""

    def _call(self, name, args, kwargs):
        LOG.append(["call", f"{self._path}.{name}", S._plain(list(args)), kwargs])

    def list_all_application_files(self):
        self._call("list_all_application_files", (), {})
        return list(APPLICATION_FILES)

    def _create(self, name, *args, **kwargs):
        self._call(name, args, S._plain(kwargs))
        return _pattern(f"{self._path}.{name}()")

    def create_rectangle(self, *args, **kwargs):
        return self._create("create_rectangle", *args, **kwargs)

    def create_regular_cross_section(self, *args, **kwargs):
        return self._create("create_regular_cross_section", *args, **kwargs)

    def create_cleaning_cross_section(self, *args, **kwargs):
        return self._create("create_cleaning_cross_section", *args, **kwargs)

    def create_line(self, *args, **kwargs):
        return self._create("create_line", *args, **kwargs)

    def create_circle(self, *args, **kwargs):
        return self._create("create_circle", *args, **kwargs)

    @property
    def state(self):
        LOG.append(["get", f"{self._path}.state"])
        return self.__dict__.get("_state", "Idle")


def make(service):
    microscope = B.make(plasma=False)
    object.__setattr__(
        microscope.connection, "patterning", FakePatterning("connection.patterning")
    )
    microscope._patterns = []
    microscope._default_application_file = "Si"
    microscope._current_application_file = "Si"
    microscope.milling_channel = BeamType.ION
    microscope._build_beams()
    if service:
        microscope._build_milling()
        return microscope
    from milling_reads import own_milling_code

    return own_milling_code(microscope)


def _state(microscope, state):
    """The patterning state the next reads see, set without recording a write."""
    object.__setattr__(microscope.connection.patterning, "_state", state)


RECIPE = FibsemMillingSettings(
    milling_current=1e-9,
    milling_voltage=30e3,
    hfw=80e-6,
    application_file="Al",
    patterning_mode="Parallel",
)

PATTERNS = (
    ("rectangle", FibsemRectangleSettings(1e-5, 5e-6, 1e-6, 0, 0, passes=4, time=3)),
    (
        "regular cross-section",
        FibsemRectangleSettings(
            1e-5,
            5e-6,
            1e-6,
            0,
            0,
            cross_section=CrossSectionPattern.RegularCrossSection,
        ),
    ),
    (
        "cleaning cross-section",
        FibsemRectangleSettings(
            1e-5,
            5e-6,
            1e-6,
            0,
            0,
            cross_section=CrossSectionPattern.CleaningCrossSection,
            scan_direction="Nowhere",
        ),
    ),
    ("line", FibsemLineSettings(0, 0, 1e-6, 0, 1e-6)),
    ("circle", FibsemCircleSettings(0, 0, 1e-6, 1e-6, thickness=2e-7)),
)


def _drawn(microscope):
    return [p._path for p in microscope._patterns]


def _calls():
    calls = [
        ("setup", lambda m: m.setup_milling(RECIPE)),
        ("clear", lambda m: m.clear_patterns()),
        ("state idle", lambda m: m.get_milling_state().name),
        ("start", lambda m: m.start_milling()),
        ("start while running", lambda m: (_state(m, "Running"), m.start_milling())),
        ("pause", lambda m: (_state(m, "Running"), m.pause_milling())),
        ("resume", lambda m: (_state(m, "Paused"), m.resume_milling())),
        ("stop", lambda m: (_state(m, "Running"), m.stop_milling())),
        ("stop when idle", lambda m: m.stop_milling()),
        ("state running", lambda m: (_state(m, "Running"), m.get_milling_state().name)),
    ]
    for name, pattern in PATTERNS:
        calls.append(
            (f"draw {name}", lambda m, p=pattern: (m.draw_pattern(p), _drawn(m))[1])
        )
    calls.append(
        (
            "draw all and estimate",
            lambda m: (
                m.draw_patterns([p for _, p in PATTERNS]),
                m.estimate_milling_time(),
            )[1],
        )
    )
    return calls


def cases():
    out = []
    for name, call in _calls():
        old, new = make(service=False), make(service=True)
        out.append(
            {
                "key": name,
                "old": B.run(lambda: call(old)),
                "new": B.run(lambda: call(new)),
            }
        )
    return out


def _conditions(microscope):
    beam = microscope.connection.beams.ion_beam
    return [
        beam.high_voltage.value,
        beam.beam_current.value,
        beam.horizontal_field_width.value,
    ]


def facts():
    from fibsem.services.drivers.autoscript import AutoScriptMilling

    new = make(service=True)
    milling = new.milling
    before = _conditions(new)
    new.setup_milling(RECIPE)
    during = _conditions(new)
    finish = B.run(lambda: new.finish_milling())
    after = _conditions(new)

    given = make(service=True)
    given.setup_milling(RECIPE)
    given.finish_milling(imaging_current=1e-10, imaging_voltage=30e3)

    # an old finish: the caller's current and voltage, and Serial
    old = make(service=False)
    old_finish = B.run(lambda: old.finish_milling(1e-10, 30e3))

    no_ion = B.make(plasma=False, ion=False)
    no_ion._build_beams()
    no_ion._build_milling()

    used = []
    traced = make(service=True)
    for hook in ("_setup", "_draw", "_start", "_estimate", "_clear"):
        original = getattr(traced.milling, hook)

        def trace(*args, _hook=hook, _original=original, **kwargs):
            used.append(_hook)
            return _original(*args, **kwargs)

        object.__setattr__(traced.milling, hook, trace)
    traced.setup_milling(RECIPE)
    traced.draw_pattern(PATTERNS[0][1])
    traced.start_milling()
    traced.estimate_milling_time()
    traced.clear_patterns()

    from milling_reads import fields_setup_reads

    supported = make(service=True).milling.supported_settings()
    reads = fields_setup_reads(make(service=True).milling, RECIPE)

    return {
        "supported": sorted(supported),
        "application_files": list(supported["application_file"].choices),
        "setup_reads": sorted(reads),
        "type": type(milling).__name__,
        "is_autoscript": isinstance(milling, AutoScriptMilling),
        "roles": [milling.ion.name, milling.electron.name],
        "before": before,
        "during": during,
        "after": after,
        "finish": finish,
        "old_finish": old_finish,
        "given": _conditions(given),
        "no_ion": no_ion.milling is None,
        "used": used,
    }


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump({"cases": cases(), "facts": facts()}, f, indent=1, default=repr)
