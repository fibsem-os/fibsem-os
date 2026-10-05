"""Record the AutoScript calls of Thermo's old beam keys and of the beam drivers.

Run as a script, in its own interpreter, for the same reason as
``autoscript_stage_parity.py``, whose fake SDK and recorder it reuses: the fake
``autoscript_sdb_microscope_client`` must be in ``sys.modules`` before
``fibsem.microscopes.autoscript`` is imported. It writes JSON to the path it is given:
``cases``, each holding what the old ``get``/``set`` returned, the SDK calls and writes
it made and the messages it logged, and the same for the microscope with its beam
keys routed to ``AutoscriptBeam``; and ``facts``, what the new API makes of each beam.
"""

import copy
import json
import logging
import os
import sys
from types import MappingProxyType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_stage_parity as S  # noqa: E402  (installs the fake SDK)

logging.disable(logging.NOTSET)

from fibsem.devices.beam import BEAM_ROUTES  # noqa: E402
from fibsem.devices.drivers.autoscript import bind_autoscript_beams  # noqa: E402
from fibsem.structures import BeamType, Point  # noqa: E402

A, LOG, STRUCTS, Node = S.A, S.LOG, S.STRUCTS, S.Node


class _Messages(logging.Handler):
    """The messages logged at INFO and above, which the old branches log."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append([record.levelname, record.getMessage()])


MESSAGES = _Messages()
logging.getLogger().addHandler(MESSAGES)
logging.getLogger().setLevel(logging.DEBUG)


def _preset(node, path, value):
    """Give a vendor attribute a value without recording it as a write."""
    *parents, name = path.split(".")
    for part in parents:
        node = getattr(node, part)
    object.__setattr__(node, name, value)


def _fake_beam(beam, beam_type):
    electron = beam_type is BeamType.ELECTRON
    values = {
        "is_on": True,
        "is_blanked": False,
        "working_distance.value": 4e-3 if electron else 16.5e-3,
        "beam_current.value": 1e-10 if electron else 2e-11,
        "beam_current.limits": STRUCTS.Limits(min=1e-12, max=1e-8),
        "beam_current.available_values": [1e-12, 2e-11, 1e-10, 1e-9],
        "high_voltage.value": 2000 if electron else 30000,
        "high_voltage.limits": STRUCTS.Limits(min=200, max=30000),
        "horizontal_field_width.value": 150e-6,
        "horizontal_field_width.limits": STRUCTS.Limits(min=1e-7, max=2e-3),
        "scanning.dwell_time.value": 1e-6,
        "scanning.rotation.value": 0.0,
        "scanning.resolution.value": "1536x1024",
        "beam_shift.value": STRUCTS.Point(x=1e-7, y=-2e-7),
        "stigmator.value": STRUCTS.Point(x=0.01, y=-0.02),
        "source.plasma_gas.value": "Xenon",
        "source.plasma_gas.available_values": ["Argon", "Oxygen", "Xenon"],
    }
    for path, value in values.items():
        _preset(beam, path, value)


def make(plasma, ion=True):
    """A ThermoMicroscope as connect leaves it, over the fake SDK."""
    microscope = S.make(compustage=False)
    microscope.system.electron.enabled = True
    microscope.system.ion.enabled = ion
    microscope.system.ion.plasma_gas = "Xenon" if plasma else None
    connection = microscope.connection
    _fake_beam(connection.beams.electron_beam, BeamType.ELECTRON)
    _fake_beam(connection.beams.ion_beam, BeamType.ION)
    _preset(connection, "detector.type.value", "ETD")
    _preset(connection, "detector.brightness.value", 0.5)
    return microscope


def routed(plasma):
    """The same microscope, its beam keys routed to the drivers."""
    microscope = make(plasma)
    microscope.beams = MappingProxyType(bind_autoscript_beams(microscope))
    microscope._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
    return microscope


def run(fn):
    """What *fn* returns (or raises), every SDK call and write, and the messages."""
    LOG.clear()
    MESSAGES.records.clear()
    try:
        result = S._plain(fn())
    except Exception as e:  # recorded, so a raise on one side only is a difference
        result = f"EXC {type(e).__name__}: {e}"
    return [result, copy.deepcopy(LOG), list(MESSAGES.records)]


GETS = (
    "on",
    "blanked",
    "working_distance",
    "current",
    "voltage",
    "hfw",
    "dwell_time",
    "scan_rotation",
    "shift",
    "stigmation",
    "resolution",
    "plasma_gas",
    # not moved: still the old branches on both sides
    "detector_type",
    "preset",
)

SETS = (
    ("on", True),
    ("on", False),
    ("blanked", True),
    ("blanked", False),
    ("working_distance", 5e-3),
    ("current", 1e-9),
    ("voltage", 5000),
    ("hfw", 100e-6),
    ("hfw", 10.0),  # clipped below the maximum
    ("hfw", 0.0),  # clipped to the minimum
    ("dwell_time", 2e-6),
    ("scan_rotation", 0.5),
    ("shift", Point(1e-6, -2e-6)),
    ("stigmation", Point(0.1, 0.2)),
    ("resolution", (3072, 2048)),
    ("resolution", [768, 512]),
    ("plasma_gas", "Argon"),
    ("plasma_gas", "Unobtainium"),  # warns and is still set
    # not moved: still the old branches on both sides
    ("detector_brightness", 0.5),
    ("preset", "anything"),
)


def cases():
    out = []
    for plasma in (False, True):
        for beam_type in (BeamType.ELECTRON, BeamType.ION):
            tag = f"plasma={plasma} {beam_type.name}"

            def add(name, call):
                old, new = make(plasma), routed(plasma)
                out.append(
                    {
                        "key": f"{tag} {name}",
                        "old": run(lambda: call(old)),
                        "new": run(lambda: call(new)),
                    }
                )

            for key in GETS:
                add(f"get {key}", lambda m, k=key, b=beam_type: m.get(k, b))
            for key, value in SETS:
                # the write, then a read of it back, so what was written is checked
                add(
                    f"set {key} {value!r}",
                    lambda m, k=key, v=value, b=beam_type: (
                        m.set(k, v, b),
                        m.get(k, b),
                    ),
                )
    return out


def _choices(beam, name):
    param = beam.parameters.get(name)
    return None if param is None else S._plain(param.choices)


def facts():
    """What the new API makes of each beam, beside the parity cases."""
    out = {}
    for plasma in (False, True):
        microscope = make(plasma)
        beams = bind_autoscript_beams(microscope)
        for beam_type, beam in beams.items():
            out[f"plasma={plasma} {beam_type.name}"] = {
                "parameters": sorted(beam.parameters),
                "commands": sorted(
                    name for name, info in beam.commands.items() if info.available
                ),
                "choices": {
                    name: _choices(beam, name)
                    for name in ("voltage", "current", "plasma_gas")
                },
                "old_choices": {
                    name: S._plain(microscope.get_available_values(name, beam_type))
                    for name in ("voltage", "current", "plasma_gas")
                },
                "hfw_limits": S._plain([beam.hfw.limits.min, beam.hfw.limits.max]),
            }

    # the keys the routed microscope sends to a driver, and the ones it leaves
    for plasma in (False, True):
        microscope = routed(plasma)
        for beam_type in (BeamType.ELECTRON, BeamType.ION):
            out[f"plasma={plasma} {beam_type.name}"]["routed"] = sorted(
                key for key in BEAM_ROUTES if microscope._route(key, beam_type)
            )

    # the new API checks a value; the old one passes it on as it is
    beam = bind_autoscript_beams(make(plasma=False))[BeamType.ELECTRON]
    LOG.clear()
    try:
        beam.voltage.set_value(1234)
        refused = None
    except ValueError as e:
        refused = str(e)
    out["voltage_off_the_list"] = {"refused": refused, "calls": copy.deepcopy(LOG)}

    # a disabled column is never built, and connect never touches it
    microscope = make(plasma=False, ion=False)
    LOG.clear()
    beams = bind_autoscript_beams(microscope)
    out["ion_disabled"] = {
        "beams": sorted(bt.name for bt in beams),
        "calls": copy.deepcopy(LOG),
    }
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump({"cases": cases(), "facts": facts()}, f, default=str)
