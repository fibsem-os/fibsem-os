"""Record the AutoScript calls of Thermo's chamber and manipulator code, routed
through the devices.

Run as a script, in its own interpreter, for the same reason as
``autoscript_stage_parity.py``, whose fake SDK and recorder it reuses: the fake
``autoscript_sdb_microscope_client`` must be in ``sys.modules`` before
``fibsem.drivers.autoscript.microscope`` is imported. It writes JSON to the path it is given:
``cases``, each holding what a call returned (or raised), the SDK calls and writes it
made and the messages it logged, on a microscope that built the devices as connect
does (the old code's are in ``autoscript_old_calls.json``, recorded over this fake
before it was deleted); and ``facts``, what the new API makes of them.
"""

import copy
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_stage_parity as S  # noqa: E402  (installs the fake SDK)

logging.disable(logging.NOTSET)

from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemManipulatorPosition,
)

LOG, STRUCTS, Node = S.LOG, S.STRUCTS, S.Node

# the waits (the needle's retract) are logged, not slept
time.sleep = lambda seconds: LOG.append(["sleep", seconds])


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append([record.levelname, record.getMessage()])


MESSAGES = _Messages()
logging.getLogger().addHandler(MESSAGES)
logging.getLogger().setLevel(logging.DEBUG)


# a repr without an address, so a position in a logged message compares
STRUCTS.ManipulatorPosition.__repr__ = lambda self: (
    f"ManipulatorPosition({self.x}, {self.y}, {self.z}, {self.coordinate_system})"
)


def _manipulator_position(x, y, z, cs="Raw"):
    return STRUCTS.ManipulatorPosition(x=x, y=y, z=z, r=None, coordinate_system=cs)


class FakeNeedle(Node):
    """``specimen.manipulator``: records calls, and moves."""

    def __init__(self, path):
        super().__init__(path)
        object.__setattr__(self, "_position", _manipulator_position(1e-4, 2e-4, 3e-4))
        object.__setattr__(self, "_state", "RETRACTED")

    def _log(self, name, *args):
        LOG.append(["call", f"{self._path}.{name}", S._plain(list(args)), "{}"])

    @property
    def current_position(self):
        return copy.deepcopy(self._position)

    @property
    def state(self):
        return self._state

    def get_saved_position(self, name, coordinate_system):
        self._log("get_saved_position", name, coordinate_system)
        z = 5e-4 if name == "PARK" else 0.0
        return _manipulator_position(0.0, 0.0, z, cs=coordinate_system)

    def insert(self, position=None):
        self._log("insert", position)
        object.__setattr__(self, "_state", "INSERTED")
        object.__setattr__(self, "_position", copy.deepcopy(position))

    def retract(self):
        self._log("retract")
        object.__setattr__(self, "_state", "RETRACTED")

    def absolute_move(self, position):
        self._log("absolute_move", position)
        object.__setattr__(self, "_position", copy.deepcopy(position))

    def relative_move(self, position):
        self._log("relative_move", position)
        moved = copy.deepcopy(self._position)
        for axis in ("x", "y", "z"):
            setattr(moved, axis, getattr(moved, axis) + getattr(position, axis))
        object.__setattr__(self, "_position", moved)


def make(build):
    """A ThermoMicroscope over the fake SDK, with its parts built as connect builds
    them when *build*, else as before."""
    microscope = S.make(compustage=False)
    microscope.system.manipulator.enabled = True
    connection = microscope.connection
    object.__setattr__(connection.vacuum, "chamber_state", "Pumped")
    object.__setattr__(connection.vacuum.chamber_pressure, "value", 2.5e-4)
    object.__setattr__(
        connection.specimen, "manipulator", FakeNeedle("specimen.manipulator")
    )
    microscope._build_stage()  # connect builds the stage first, and the moves read it
    if build:
        microscope._build_parts()
    return microscope


def run(fn):
    LOG.clear()
    MESSAGES.records.clear()
    try:
        result = S._plain(_value(fn()))
    except Exception as e:  # recorded, so a raise on one side only is a difference
        result = f"EXC {type(e).__name__}: {e}"
    return [result, copy.deepcopy(LOG), list(MESSAGES.records)]


def _value(value):
    if isinstance(value, FibsemManipulatorPosition):
        return [value.x, value.y, value.z, value.coordinate_system]
    return value


OFFSET = FibsemManipulatorPosition(x=1e-6, y=2e-6, z=3e-6)


CALLS = (
    ("get chamber_state", lambda m: m.get("chamber_state")),
    ("get chamber_pressure", lambda m: m.get("chamber_pressure")),
    ("pump", lambda m: m.pump()),
    ("vent", lambda m: m.vent()),
    ("set pump_chamber False", lambda m: m.set("pump_chamber", False)),
    ("get manipulator_position", lambda m: m.get("manipulator_position")),
    ("get manipulator_state", lambda m: m.get("manipulator_state")),
    ("manipulator state", lambda m: m.get_manipulator_state()),
    ("manipulator position", lambda m: m.get_manipulator_position()),
    ("insert PARK", lambda m: m.insert_manipulator("PARK")),
    ("insert EUCENTRIC", lambda m: m.insert_manipulator("EUCENTRIC")),
    ("insert BAD", lambda m: m.insert_manipulator("BAD")),
    (
        "insert then state",
        lambda m: (m.insert_manipulator(), m.get("manipulator_state")),
    ),
    ("retract", lambda m: m.retract_manipulator()),
    ("move relative", lambda m: m.move_manipulator_relative(OFFSET)),
    ("move absolute", lambda m: m.move_manipulator_absolute(OFFSET)),
    (
        "move corrected ELECTRON",
        lambda m: m.move_manipulator_corrected(1e-6, 2e-6, BeamType.ELECTRON),
    ),
    (
        "move corrected ION",
        lambda m: m.move_manipulator_corrected(1e-6, 2e-6, BeamType.ION),
    ),
    (
        "move to offset",
        lambda m: m.move_manipulator_to_position_offset(OFFSET, "EUCENTRIC"),
    ),
    ("saved PARK", lambda m: m._get_saved_manipulator_position("PARK")),
    ("saved BAD", lambda m: m._get_saved_manipulator_position("BAD")),
    ("named positions", lambda m: m.manipulator_named_positions()),
)


def cases():
    # built before the run, so connect's calls are not part of the case
    return [
        {"key": key, "new": run(lambda m=make(True): call(m))} for key, call in CALLS
    ]


def facts():
    out = {}
    microscope = make(True)
    out["devices"] = {
        "chamber": type(microscope.chamber_device).__name__,
        "manipulator": type(microscope.manipulator_device).__name__,
        "chamber_parameters": sorted(microscope.chamber_device.parameters),
        "manipulator_parameters": sorted(microscope.manipulator_device.parameters),
    }

    # the routed calls go through the devices' hooks
    microscope = make(True)
    used = []
    for device, hooks in (
        (microscope.chamber_device, ("_pump", "_vent")),
        (
            microscope.manipulator_device,
            ("_insert", "_retract", "_move_relative", "_move_absolute"),
        ),
    ):
        for hook in hooks:
            original = getattr(device, hook)

            def counted(*args, _o=original, _n=f"{device.name}.{hook}", **kwargs):
                used.append(_n)
                return _o(*args, **kwargs)

            setattr(device, hook, counted)
    microscope.pump()
    microscope.vent()
    microscope.insert_manipulator()
    microscope.move_manipulator_corrected(1e-6, 2e-6, BeamType.ELECTRON)
    microscope.move_manipulator_to_position_offset(OFFSET, "EUCENTRIC")
    microscope.retract_manipulator()
    out["through_devices"] = used

    # without a manipulator, none is built and its keys stay with the old branches
    microscope = S.make(compustage=False)
    microscope.system.manipulator.enabled = False
    microscope._build_parts()
    out["none_fitted"] = {
        "manipulator": microscope.manipulator_device is None,
        "routed": microscope._route("manipulator_state", None) is not None,
    }

    # hardware.devices entries: one switched off, and an added device no driver builds
    from fibsem.devices.entries import DeviceBuildError
    from fibsem.structures import DeviceEntry

    def configured(*entries):
        microscope = make(False)
        microscope.system.other_devices = [DeviceEntry.from_dict(e) for e in entries]
        microscope._build_parts()
        return microscope

    microscope = configured(
        {"name": "manipulator", "enabled": False},
        {"name": "laser", "type": "laser", "required": False},
    )
    out["configured"] = {
        "devices": list(microscope.devices),
        "manipulator": microscope.manipulator_device is None,
        "routed": microscope._route("manipulator_state", None) is not None,
    }
    try:
        configured({"name": "laser", "type": "laser"})
        out["configured"]["required"] = None
    except DeviceBuildError as e:
        out["configured"]["required"] = str(e)
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump({"cases": cases(), "facts": facts()}, f, default=str)
