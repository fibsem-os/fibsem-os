"""The Tescan beam driver makes the SDK calls Tescan's beam keys made before it.

``TescanBeam`` is ``TescanMicroscope``'s beam branches moved onto the ``Beam`` device
(FIB-1161). Each case runs an old ``get``/``set`` on a microscope connected as the app
connects it, whose beam keys are routed to the drivers, over a recording fake of the SDK
(``tests/fixtures/tescan_sdk.py``). It must give the result, the SDK calls in order and
the messages logged at info and above that the old branches gave, recorded in
``tests/fixtures/tescan_beam_calls.json`` before they were deleted. A ``set`` case reads
the key back after the write.

Cases: every beam key on both columns, including the ones a column does not have (the
old branches still answer them), the values the Tescan API refuses (the ion column's
current and voltage), the hfw clip, out-of-range detector levels, an unknown detector
and an unknown preset. Nothing here has run on an instrument.
"""

import json
import logging
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.beam import BEAM_ROUTES
from fibsem.devices.drivers.tescan import TescanBeam, bind_tescan_beams
from fibsem.structures import BeamType, Point
from tests.fixtures.tescan_sdk import connect

E, I = BeamType.ELECTRON, BeamType.ION

RECORDED = os.path.join(os.path.dirname(__file__), "fixtures", "tescan_beam_calls.json")


def _system(electron=True, ion=True):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.electron.enabled = electron
    system.ion.enabled = ion
    return system


def _connected(monkeypatch):
    microscope, fake = connect(monkeypatch, _system())
    assert set(microscope.beams) == {E, I}
    # what the last image reported: the API cannot read these
    for beam_type in (E, I):
        cache = microscope._beam_parameters[beam_type]
        cache.resolution = [1536, 1024]
        cache.dwell_time = 1e-6
        cache.stigmation = Point(0.1, -0.2)
        cache.preset = "30 keV; 1 nA" if beam_type is I else None
    return microscope, fake


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append([record.levelname, record.getMessage()])


def _run(microscope, fake, call):
    handler = _Messages()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    fake.log.clear()
    try:
        try:
            result = repr(call(microscope))
        except Exception as e:  # a refusal is behaviour too
            result = f"raises {type(e).__name__}: {e}"
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    # as the recording stored it: JSON, with anything else as its repr
    ran = {"result": result, "sdk": list(fake.log), "log": handler.messages}
    return json.loads(json.dumps(ran, default=repr))


GETS = sorted(BEAM_ROUTES)

SETS = [
    ("on", False),
    ("on", True),
    ("working_distance", 5e-3),
    ("current", 100e-12),
    ("voltage", 5000.0),
    ("hfw", 100e-6),
    ("hfw", 10.0),  # clipped to the column's limit
    ("scan_rotation", 3.141592653589793),
    ("shift", Point(1e-6, -2e-6)),
    ("resolution", [3072, 2048]),
    ("dwell_time", 2e-6),
    ("stigmation", Point(0.0, 0.0)),
    ("preset", "30 keV; 100 pA"),
    ("preset", "not a preset"),
    ("detector_type", "E-T"),
    ("detector_type", "SI"),
    ("detector_type", "not a detector"),
    ("detector_mode", "SE"),
    ("detector_contrast", 0.7),
    ("detector_brightness", 0.3),
    ("detector_brightness", 1.5),  # refused
]


def _cases():
    for beam_type in (E, I):
        for key in GETS:
            yield (
                f"{beam_type.name} get {key}",
                (lambda m, k=key, b=beam_type: m.get(k, b)),
            )
        for key, value in SETS:
            yield (
                f"{beam_type.name} set {key} {value!r}",
                (lambda m, k=key, v=value, b=beam_type: (m.set(k, v, b), m.get(k, b))),
            )


CASES = dict(_cases())


with open(RECORDED) as f:
    EXPECTED = json.load(f)


def test_every_case_was_recorded():
    assert sorted(CASES) == sorted(EXPECTED)


@pytest.mark.parametrize("case", list(CASES))
def test_a_routed_key_makes_the_same_calls_logs_and_result(monkeypatch, case):
    microscope, fake = _connected(monkeypatch)
    assert _run(microscope, fake, CASES[case]) == EXPECTED[case]
    assert fake.unlocked == []


def test_the_parity_cases_make_sdk_calls():
    """A guard on the guard: the cases above compare something."""
    assert sum(len(case["sdk"]) for case in EXPECTED.values()) > 200


@pytest.mark.parametrize(
    "beam_type, parameters",
    [
        (
            E,
            [
                "current",
                "detector_brightness",
                "detector_contrast",
                "detector_type",
                "dwell_time",
                "hfw",
                "on",
                "resolution",
                "scan_rotation",
                "shift",
                "stigmation",
                "voltage",
                "working_distance",
            ],
        ),
        (
            I,
            [
                "current",
                "detector_brightness",
                "detector_contrast",
                "detector_type",
                "dwell_time",
                "hfw",
                "on",
                "preset",
                "resolution",
                "scan_rotation",
                "shift",
                "stigmation",
                "voltage",
            ],
        ),
    ],
)
def test_each_column_has_what_its_api_has(monkeypatch, beam_type, parameters):
    microscope, _ = connect(monkeypatch, _system())
    beam = microscope.beams[beam_type]
    assert isinstance(beam, TescanBeam)
    assert sorted(beam.parameters) == parameters
    available = sorted(n for n, c in beam.commands.items() if c.available)
    assert available == ["acquire"]


def test_what_the_api_refuses_reads_as_not_settable(monkeypatch):
    """What the beam-settings widget can read instead of asking for the vendor."""
    microscope, _ = connect(monkeypatch, _system())
    settable = {
        beam_type: sorted(n for n, p in beam.parameters.items() if p.settable)
        for beam_type, beam in microscope.beams.items()
    }
    common = [
        "detector_brightness",
        "detector_contrast",
        "detector_type",
        "hfw",
        "on",
        "scan_rotation",
        "shift",
    ]
    assert settable[E] == sorted(common + ["voltage", "working_distance"])
    assert settable[I] == sorted(common + ["preset"])


@pytest.mark.parametrize("beam_type", [E, I])
@pytest.mark.parametrize("key", ["current", "detector_type", "preset"])
def test_the_choices_are_get_available_values(monkeypatch, beam_type, key):
    microscope, _ = connect(monkeypatch, _system())
    param = microscope.beams[beam_type].parameters.get(key)
    if param is None:  # the electron column's preset
        assert (beam_type, key) == (E, "preset")
        return
    assert list(param.choices) == microscope.get_available_values(key, beam_type)


def test_hfw_limits_are_the_columns(monkeypatch):
    microscope, _ = connect(monkeypatch, _system())
    for beam_type, high in ((E, 2580.0e-6), (I, 450.0e-6)):
        limits = microscope.beams[beam_type].hfw.limits
        assert (limits.min, limits.max) == (1.0e-6, high)


def test_connect_sets_the_default_detectors_through_the_devices(monkeypatch):
    seen = []
    write = TescanBeam.write_detector_type

    def recording(self, value):
        seen.append((self.beam_type, value))
        write(self, value)

    monkeypatch.setattr(TescanBeam, "write_detector_type", recording)
    microscope, fake = connect(monkeypatch, _system())
    assert seen == [(E, "SE"), (I, "SE")]
    assert microscope._active_detector[E].name == "SE"


def test_a_disabled_column_is_never_built_or_touched(monkeypatch):
    microscope, fake = connect(monkeypatch, _system(ion=False))
    assert set(microscope.beams) == {E}
    fake.log.clear()
    beams = bind_tescan_beams(microscope)
    assert set(beams) == {E}
    assert not [path for path, _, _ in fake.log if path.startswith("FIB.")]
