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
from fibsem.structures import BeamType, FibsemRectangle, Point  # noqa: E402

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
        "scanning.mode.value": "FullFrame",
        "beam_shift.value": STRUCTS.Point(x=1e-7, y=-2e-7),
        "stigmator.value": STRUCTS.Point(x=0.01, y=-0.02),
        "source.plasma_gas.value": "Xenon",
        "source.plasma_gas.available_values": ["Argon", "Oxygen", "Xenon"],
    }
    if electron:
        values["angular_correction.angle.value"] = 0.05
        values["angular_correction.tilt_correction.is_on"] = False
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
    _preset(connection, "detector.type.available_values", ["ETD", "TLD", "ICE"])
    _preset(connection, "detector.mode.value", "SecondaryElectrons")
    _preset(
        connection,
        "detector.mode.available_values",
        ["SecondaryElectrons", "BackscatterElectrons"],
    )
    _preset(connection, "detector.brightness.value", 0.5)
    _preset(connection, "detector.contrast.value", 0.6)
    return microscope


def old_branches(plasma):
    """A microscope whose beam keys are all still answered by ``_get``/``_set``."""
    microscope = make(plasma)
    microscope.beams = MappingProxyType({})
    microscope._beam_routes = MappingProxyType({})
    return microscope


def routed(plasma):
    """The same microscope, its beam keys routed as connect routes them."""
    microscope = make(plasma)
    microscope._build_beams()
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
    "detector_type",
    "detector_mode",
    "detector_brightness",
    "detector_contrast",
    "angular_correction_angle",
    # not moved: still the old branches on both sides
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
    ("detector_type", "TLD"),
    ("detector_type", "Unobtainium"),  # warns, not set
    ("detector_mode", "BackscatterElectrons"),
    ("detector_mode", "Unobtainium"),  # warns, not set
    ("detector_brightness", 0.3),
    ("detector_brightness", 0.0),  # warns, not set
    ("detector_contrast", 0.7),
    ("detector_contrast", 1.5),  # warns, not set
    ("angular_correction_angle", 0.2),
    # not moved: still the old branches on both sides
    ("preset", "anything"),
)

# Sets with no read-back: the old key could only be set, so its get is new.
SETS_ONLY = (
    ("angular_correction_tilt_correction", True),
    ("angular_correction_tilt_correction", False),
)

# The scan-mode methods, through the scan commands on one side and the old keys on
# the other. Nothing read the scan mode before, so these are not read back.
SCANS = (
    ("spot", lambda m, b: m.set_spot_scanning_mode(Point(0.25, 0.75), b)),
    (
        "reduced_area",
        lambda m, b: m.set_reduced_area_scanning_mode(
            FibsemRectangle(0.1, 0.2, 0.3, 0.4), b
        ),
    ),
    ("full_frame", lambda m, b: m.set_full_frame_scanning_mode(b)),
)


def cases():
    out = []
    for plasma in (False, True):
        for beam_type in (BeamType.ELECTRON, BeamType.ION):
            tag = f"plasma={plasma} {beam_type.name}"

            def add(name, call):
                old, new = old_branches(plasma), routed(plasma)
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
            for key, value in SETS_ONLY:
                add(
                    f"set {key} {value!r}",
                    lambda m, k=key, v=value, b=beam_type: m.set(k, v, b),
                )
            for name, scan in SCANS:
                add(f"scan {name}", lambda m, s=scan, b=beam_type: s(m, b))
    return out


CHOICE_KEYS = ("voltage", "current", "plasma_gas", "detector_type", "detector_mode")


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
                "choices": {name: _choices(beam, name) for name in CHOICE_KEYS},
                "old_choices": {
                    name: S._plain(microscope.get_available_values(name, beam_type))
                    for name in CHOICE_KEYS
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

    # connect builds the beams before it resets the beam shifts through them
    class _Stop(Exception):
        pass

    microscope = make(plasma=True)
    seen = {}

    def reset_beam_shifts():
        seen["beams"] = sorted(bt.name for bt in microscope.beams)
        seen["routed"] = microscope._route("shift", BeamType.ION) is not None
        raise _Stop

    microscope.reset_beam_shifts = reset_beam_shifts
    try:
        microscope.connect_to_microscope("localhost")
    except _Stop:
        pass
    out["connect"] = seen

    # the new API checks a value; the old one passes it on as it is
    beam = bind_autoscript_beams(make(plasma=False))[BeamType.ELECTRON]
    LOG.clear()
    try:
        beam.voltage.set_value(1234)
        refused = None
    except ValueError as e:
        refused = str(e)
    out["voltage_off_the_list"] = {"refused": refused, "calls": copy.deepcopy(LOG)}

    # the scan mode, which nothing read before, and one with no ScanMode
    microscope = routed(plasma=False)
    vendor = microscope.connection.beams.electron_beam.scanning.mode
    scan = {"full_frame": microscope.get("scanning_mode", BeamType.ELECTRON)}
    _preset(vendor, "value", "Line")
    scan["line"] = run(lambda: microscope.get("scanning_mode", BeamType.ELECTRON))
    out["scanning_mode"] = scan

    # a detector read selects its beam's channel under the imaging channel's lock
    microscope = routed(plasma=False)
    selected = []
    set_channel = microscope.set_channel

    def set_channel_recording(beam_type):
        selected.append([beam_type.name, microscope._threading_lock._is_owned()])
        set_channel(beam_type)

    microscope.set_channel = set_channel_recording
    result = run(lambda: microscope.get("detector_type", BeamType.ION))
    out["detector_read"] = {"selected": selected, "result": result[0]}

    # the tilt correction, which could only be set before, reads what was set
    microscope = routed(plasma=False)
    tilt = {
        "before": microscope.get(
            "angular_correction_tilt_correction", BeamType.ELECTRON
        )
    }
    vendor = (
        microscope.connection.beams.electron_beam.angular_correction.tilt_correction
    )
    _preset(vendor, "is_on", True)
    tilt["after"] = microscope.get(
        "angular_correction_tilt_correction", BeamType.ELECTRON
    )
    tilt["old"] = old_branches(plasma=False).get(
        "angular_correction_tilt_correction", BeamType.ELECTRON
    )
    tilt["ion"] = microscope.get("angular_correction_tilt_correction", BeamType.ION)
    out["tilt_correction"] = tilt

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
