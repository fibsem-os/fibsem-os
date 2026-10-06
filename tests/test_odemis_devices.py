"""The Odemis beams and stage as devices make the odemis calls the old code made.

``OdemisBeam``, ``OdemisStage`` and ``OdemisChamber`` are ``OdemisThermoMicroscope``'s
beam, stage and chamber keys moved onto the devices. Each case runs an old call on a
microscope created as usual, over a fake odemis client and stage that record every
call, and requires the result (or error), the calls in order and the logged messages
the old branches gave, recorded in ``tests/fixtures/odemis_device_calls.json`` before
they were deleted.

No odemis installation: odemis is replaced by the stub modules in
``tests/fm/_odemis_stubs.py``. Nothing here has run on an instrument.
"""

import json
import logging
import os
import sys

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from tests.fm import _odemis_stubs as stubs

ODEMIS_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "odemis-configuration.yaml")
RECORDED = os.path.join(
    os.path.dirname(__file__), "fixtures", "odemis_device_calls.json"
)

LOG = []


class _Future:
    def __init__(self, name):
        self.name = name

    def result(self):
        LOG.append([f"{self.name}.result"])


class FakeStage:
    """The ``stage-bare`` component: a position, and moves that record."""

    def __init__(self):
        self.position = stubs.FakeVA(
            {"x": 1e-3, "y": 2e-3, "z": 3e-3, "rz": 0.5, "rx": 0.3}
        )

    def moveAbs(self, pdict):
        LOG.append(["moveAbs", dict(pdict)])
        value = dict(self.position.value)
        value.update(pdict)
        self.position.value = value
        return _Future("moveAbs")

    def moveRel(self, pdict):
        LOG.append(["moveRel", dict(pdict)])
        value = dict(self.position.value)
        for axis, delta in pdict.items():
            value[axis] = value[axis] + delta
        self.position.value = value
        return _Future("moveRel")


# What the fake client's reads answer, by method; a channel picks a column's value.
READS = {
    "get_software_version": "1.0",
    "get_hardware_version": "Fake",
    "get_beam_is_on": True,
    "beam_is_blanked": False,
    "get_working_distance": {"electron": 4e-3, "ion": 16.5e-3},
    "get_beam_current": {"electron": 1e-10, "ion": 3e-11},
    "get_high_voltage": {"electron": 2000.0, "ion": 30000.0},
    "get_field_of_view": {"electron": 150e-6, "ion": 900e-6},
    "get_dwell_time": 1e-6,
    "get_scan_rotation": {"electron": 0.0, "ion": 3.14},
    "get_beam_shift": {"electron": (1e-6, -2e-6), "ion": (3e-6, 4e-6)},
    "get_stigmator": {"electron": (0.1, 0.2), "ion": (0.3, 0.4)},
    "get_resolution": (1536, 1024),
    "get_detector_type": {"electron": "ETD", "ion": "ICE"},
    "get_detector_mode": "SecondaryElectrons",
    "get_brightness": 0.4,
    "get_contrast": 0.6,
    "detector_type_info": {
        "electron": {"choices": {"ETD", "TLD"}},
        "ion": {"choices": {"ICE", "ETD"}},
    },
    "detector_mode_info": {"choices": {"SecondaryElectrons", "BackscatterElectrons"}},
    "beam_current_info": {
        "electron": {"range": (1e-12, 1e-8)},
        "ion": {"choices": [1e-12, 3e-11, 1e-9]},
    },
    "high_voltage_info": {
        "electron": {"range": (500.0, 30000.0)},
        "ion": {"range": (500.0, 30000.0)},
    },
    "is_homed": True,
    "is_linked": False,
    "get_chamber_state": "vacuum",
    "get_pressure": 1e-5,
}

# The frames the fake client's imaging calls return.
FRAME = np.full((4, 6), 7, dtype=np.uint8)
READS["acquire_image"] = (FRAME, {})
READS["get_last_image"] = FRAME


class FakeClient:
    """The ``fibsem`` component: records every call, and answers reads from READS."""

    def __getattr__(self, name):
        def call(*args, **kwargs):
            LOG.append([name, list(args), dict(kwargs)])
            value = READS.get(name)
            channel = kwargs.get("channel", args[-1] if args else None)
            if isinstance(value, dict) and channel in value:
                return value[channel]
            return value

        return call


@pytest.fixture(scope="module")
def odemis_cls():
    """Import OdemisThermoMicroscope against stub odemis modules."""
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.modules.pop("fibsem.devices.drivers.odemis", None)
    stubs.install_odemis_stubs()
    from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

    yield OdemisThermoMicroscope

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


def _system(ion=True):
    system = utils.load_microscope_configuration(ODEMIS_CONFIG_PATH).system
    system.fm.enabled = False
    system.ion.enabled = ion
    return system


def make(cls, ion=True):
    """An OdemisThermoMicroscope over the fake client, created as usual."""
    stubs.use_components({"fibsem": FakeClient(), "stage-bare": FakeStage()})
    return cls(_system(ion=ion))


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append([record.levelname, record.getMessage()])


def _plain(value):
    if isinstance(value, Point):
        return ["Point", value.x, value.y]
    if isinstance(value, (FibsemStagePosition, FibsemDetectorSettings)):
        return value.to_dict()
    if isinstance(value, ImageSettings):
        return value.to_dict()
    if isinstance(value, FibsemImage):
        return {
            "shape": list(value.data.shape),
            "settings": value.metadata.image_settings.to_dict(),
            "pixel_size": _plain(value.metadata.pixel_size),
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def run(microscope, call):
    """What *call* returns (or raises), the odemis calls it makes and what it logs."""
    messages = _Messages()
    root = logging.getLogger()
    level = root.level
    root.addHandler(messages)
    root.setLevel(logging.DEBUG)
    LOG.clear()
    try:
        result = _plain(call(microscope))
    except Exception as e:  # recorded, so a raise on one side only is a difference
        result = f"EXC {type(e).__name__}: {e}"
    finally:
        root.removeHandler(messages)
        root.setLevel(level)
    # as the recording stored it: JSON, with anything else as its repr
    ran = {"result": result, "calls": list(LOG), "log": messages.records}
    return json.loads(json.dumps(ran, default=repr))


BEAM_GETS = (
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
    "detector_type",
    "detector_mode",
    "detector_brightness",
    "detector_contrast",
    "plasma_gas",
    "preset",
    "scanning_mode",
)

BEAM_SETS = (
    ("on", True),
    ("on", False),
    ("blanked", True),
    ("blanked", False),
    ("working_distance", 5e-3),
    ("current", 1e-9),
    ("voltage", 5000.0),
    ("hfw", 80e-6),
    ("dwell_time", 3e-6),
    ("scan_rotation", 1.0),
    ("shift", Point(1e-7, 2e-7)),
    ("stigmation", Point(0.01, -0.02)),
    ("resolution", (3072, 2048)),
    ("detector_type", "ETD"),
    ("detector_type", "NOPE"),
    ("detector_mode", "BackscatterElectrons"),
    ("detector_mode", "NOPE"),
    ("detector_brightness", 0.5),
    ("detector_brightness", 0.0),
    ("detector_brightness", 1.5),
    ("detector_contrast", 0.7),
    ("detector_contrast", 0.0),
    ("detector_contrast", 2.0),
)


def _beam_cases():
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        name = beam_type.name
        for key in BEAM_GETS:
            yield f"get {key} {name}", lambda m, k=key, b=beam_type: m.get(k, b)
        for key, value in BEAM_SETS:
            yield (
                f"set {key} {value} {name}",
                lambda m, k=key, v=value, b=beam_type: m.set(k, v, b),
            )
        yield (
            f"detector settings {name}",
            lambda m, b=beam_type: m.get_detector_settings(b),
        )
        yield (
            f"set detector settings {name}",
            lambda m, b=beam_type: m.set_detector_settings(
                FibsemDetectorSettings(
                    type="TLD" if b is BeamType.ELECTRON else "ICE",
                    mode="BackscatterElectrons",
                    brightness=0.3,
                    contrast=0.8,
                ),
                b,
            ),
        )
        yield (
            f"imaging settings {name}",
            lambda m, b=beam_type: m.get_imaging_settings(b),
        )
        yield (
            f"spot {name}",
            lambda m, b=beam_type: m.set_spot_scanning_mode(Point(0.5, 0.5), b),
        )
        yield (
            f"reduced area {name}",
            lambda m, b=beam_type: m.set_reduced_area_scanning_mode(AREA, b),
        )
        yield (
            f"full frame {name}",
            lambda m, b=beam_type: m.set_full_frame_scanning_mode(b),
        )
        yield (
            f"set spot_mode {name}",
            lambda m, b=beam_type: m.set("spot_mode", Point(0.25, 0.75), b),
        )
        yield (
            f"set reduced_area {name}",
            lambda m, b=beam_type: m.set("reduced_area", AREA, b),
        )
        yield (
            f"set full_frame {name}",
            lambda m, b=beam_type: m.set("full_frame", None, b),
        )
        for key in ("current", "voltage", "detector_type", "detector_mode"):
            yield (
                f"available {key} {name}",
                lambda m, k=key, b=beam_type: sorted(m.get_available_values(k, b)),
            )


AREA = FibsemRectangle(0.25, 0.25, 0.5, 0.5)


def _image_settings(beam_type, square=False, reduced=False):
    return ImageSettings(
        beam_type=beam_type,
        resolution=(1024, 1024) if square else (1536, 1024),
        dwell_time=2e-7,
        hfw=80e-6,
        reduced_area=FibsemRectangle(0.1, 0.2, 0.3, 0.4) if reduced else None,
        path="/data",
        filename="img",
    )


def _imaging_cases():
    for b in (BeamType.ELECTRON, BeamType.ION):
        name = b.name
        yield (
            f"acquire settings {name}",
            lambda m, b=b: m.acquire_image(_image_settings(b)),
        )
        yield (
            f"acquire square {name}",
            lambda m, b=b: m.acquire_image(_image_settings(b, square=True)),
        )
        yield (
            f"acquire reduced {name}",
            lambda m, b=b: m.acquire_image(_image_settings(b, reduced=True)),
        )
        yield (f"acquire current {name}", lambda m, b=b: m.acquire_image(beam_type=b))
        yield (
            f"acquire both {name}",
            lambda m, b=b: m.acquire_image(_image_settings(b), beam_type=b),
        )
        yield (
            f"acquire last settings {name}",
            lambda m, b=b: (
                m.acquire_image(_image_settings(b, square=True)),
                m._last_imaging_settings,
            )[1],
        )
        yield (f"last image {name}", lambda m, b=b: m.last_image(b))
        yield (f"autocontrast {name}", lambda m, b=b: m.autocontrast(b))
        yield (f"autocontrast area {name}", lambda m, b=b: m.autocontrast(b, AREA))
    yield ("acquire nothing", lambda m: m.acquire_image())


IMAGING_CASES = tuple(_imaging_cases())

STAGE_CASES = (
    ("get stage_position", lambda m: m.get("stage_position")),
    ("get stage_homed", lambda m: m.get("stage_homed")),
    ("get stage_linked", lambda m: m.get("stage_linked")),
    ("set stage_home", lambda m: m.set("stage_home", True)),
    ("set stage_link True", lambda m: m.set("stage_link", True)),
    ("set stage_link False", lambda m: m.set("stage_link", False)),
    ("stage position", lambda m: m.get_stage_position()),
    (
        "move absolute",
        lambda m: m.move_stage_absolute(FibsemStagePosition(x=1e-4, y=2e-4, t=0.2)),
    ),
    (
        "move relative",
        lambda m: m.move_stage_relative(FibsemStagePosition(x=1e-5, z=-1e-5)),
    ),
    ("home", lambda m: m.home()),
    ("link", lambda m: m.link_stage()),
)


def _chamber_state(name):
    """The chamber state when the client names it *name*."""

    def read(m):
        READS["get_chamber_state"] = name
        return m.get("chamber_state")

    return read


CHAMBER_CASES = tuple(
    (f"get chamber_state {name}", _chamber_state(name))
    for name in ("vacuum", "Pumped", "vented", "Vented", "pumping", "venting")
    + ("vacuum_error",)
) + (
    ("get chamber_pressure", lambda m: m.get("chamber_pressure")),
    ("pump", lambda m: m.pump()),
    ("vent", lambda m: m.vent()),
    ("set pump_chamber True", lambda m: m.set("pump_chamber", True)),
    ("set pump_chamber False", lambda m: m.set("pump_chamber", False)),
    ("set vent_chamber True", lambda m: m.set("vent_chamber", True)),
    ("set vent_chamber False", lambda m: m.set("vent_chamber", False)),
)

CASES = tuple(_beam_cases()) + IMAGING_CASES + STAGE_CASES + CHAMBER_CASES
# The old code had no reduced_area key: the set, and the method that made it, warned
# and did nothing. The beam's reduced_area command makes the client call instead.
REDUCED_AREA = {
    key: call
    for key, call in CASES
    if key.startswith(("reduced area", "set reduced_area"))
}
SAME = tuple((key, call) for key, call in CASES if key not in REDUCED_AREA)

with open(RECORDED) as f:
    EXPECTED = json.load(f)

# The calls the devices add, after the old call's first: the home command reads
# back whether the stage is homed, as ``home()`` always has, so a bare
# ``set("stage_home")`` asks once more; and the chamber's pump and vent read the
# pressure back with the state.
_PRESSURE = ["get_pressure", [], {}]
EXTRA_READS = {
    "set stage_home": [["is_homed", [], {}]],
    "pump": [_PRESSURE],
    "vent": [_PRESSURE],
    "set pump_chamber True": [_PRESSURE, ["get_chamber_state", [], {}]],
    "set vent_chamber True": [_PRESSURE, ["get_chamber_state", [], {}]],
}


def test_every_case_was_recorded():
    assert sorted(key for key, _ in CASES) == sorted(EXPECTED)


@pytest.mark.parametrize("key,call", SAME, ids=[key for key, _ in SAME])
def test_the_devices_make_the_same_odemis_calls_logs_and_results(odemis_cls, key, call):
    new = run(make(odemis_cls), call)
    READS["get_chamber_state"] = "vacuum"
    old = EXPECTED[key]
    calls = old["calls"][:1] + EXTRA_READS.get(key, []) + old["calls"][1:]
    assert new == {**old, "calls": calls}


@pytest.mark.parametrize("key", sorted(REDUCED_AREA))
def test_the_reduced_area_is_set_where_the_old_code_warned(odemis_cls, key):
    old = EXPECTED[key]
    assert old["calls"] == []
    assert old["log"][0][0] == "WARNING"
    assert "Unknown key: reduced_area" in old["log"][0][1]
    new = run(make(odemis_cls), REDUCED_AREA[key])
    channel = "electron" if key.endswith("ELECTRON") else "ion"
    area = {"left": 0.25, "top": 0.25, "width": 0.5, "height": 0.5}
    assert new == {
        "result": None,
        "calls": [["set_reduced_area_scan_mode", [], {"channel": channel, **area}]],
        "log": [],
    }


def test_the_cases_make_odemis_calls():
    """A guard on the guard: the recorded cases compare something."""
    calls = [case["calls"] for case in EXPECTED.values()]
    assert sum(1 for c in calls if c) > len(CASES) * 0.8


def test_creating_the_microscope_builds_the_beams_and_stage(odemis_cls):
    from fibsem.devices.drivers.odemis import OdemisBeam, OdemisChamber, OdemisStage

    microscope = make(odemis_cls)
    assert set(microscope.beams) == {BeamType.ELECTRON, BeamType.ION}
    assert all(isinstance(b, OdemisBeam) for b in microscope.beams.values())
    assert isinstance(microscope.stage, OdemisStage)
    electron = microscope.beams[BeamType.ELECTRON]
    assert "plasma_gas" not in electron.parameters
    assert "preset" not in electron.parameters
    assert "scanning_mode" not in electron.parameters
    assert electron.commands["spot"].available  # with no read back
    assert electron.current.choices[0] == 1e-12
    assert electron.voltage.choices == sorted(electron.voltage.choices)
    assert sorted(electron.detector_type.choices) == ["ETD", "TLD"]
    assert sorted(microscope.stage.axes) == ["r", "t", "x", "y", "z"]
    assert microscope.stage.commands["link"].available
    assert isinstance(microscope.chamber_device, OdemisChamber)
    assert sorted(microscope.chamber_device.parameters) == ["pressure", "state"]


def test_the_calls_go_through_the_devices(odemis_cls):
    microscope = make(odemis_cls)
    used = []
    for device, hooks in (
        (microscope.stage, ("_move_absolute", "_move_relative", "_home")),
        (microscope.chamber_device, ("_pump", "_vent")),
        (microscope.beams[BeamType.ION], ("_spot", "_reduced_area", "_full_frame")),
    ):
        for hook in hooks:
            original = getattr(device, hook)

            def counted(*args, _o=original, _n=hook, **kwargs):
                used.append(_n)
                return _o(*args, **kwargs)

            setattr(device, hook, counted)
    microscope.move_stage_absolute(FibsemStagePosition(x=0.0))
    microscope.move_stage_relative(FibsemStagePosition(x=1e-6))
    microscope.home()
    microscope.pump()
    microscope.vent()
    assert used == ["_move_absolute", "_move_relative", "_home", "_pump", "_vent"]
    used.clear()
    microscope.set("spot_mode", Point(0.5, 0.5), BeamType.ION)
    microscope.set("reduced_area", AREA, BeamType.ION)
    microscope.set("full_frame", None, BeamType.ION)
    microscope.set_full_frame_scanning_mode(BeamType.ION)
    assert used == ["_spot", "_reduced_area", "_full_frame", "_full_frame"]
    beam = microscope.beams[BeamType.ION]
    microscope.set("hfw", 50e-6, BeamType.ION)
    assert beam.hfw.cached == 50e-6


def test_imaging_goes_through_the_beam_commands(odemis_cls):
    microscope = make(odemis_cls)
    used = []
    for beam in microscope.beams.values():
        for command in ("acquire", "last_image", "autocontrast"):
            original = getattr(beam, command)

            def wrapper(*args, _name=f"{beam.name}.{command}", _f=original, **kw):
                used.append(_name)
                return _f(*args, **kw)

            setattr(beam, command, wrapper)
    sem, fib = microscope.beams[BeamType.ELECTRON], microscope.beams[BeamType.ION]
    microscope.acquire_image(_image_settings(BeamType.ELECTRON))
    microscope.acquire_image(beam_type=BeamType.ION)
    microscope.last_image(BeamType.ION)
    microscope.autocontrast(BeamType.ELECTRON, AREA)
    assert used == [
        f"{sem.name}.acquire",
        f"{fib.name}.acquire",
        f"{fib.name}.last_image",
        f"{sem.name}.autocontrast",
    ]
    available = {n for n, info in sem.commands.items() if info.available}
    assert {"acquire", "last_image", "autocontrast"} <= available
    # the working-distance sweep is not the instrument's routine, and there is no
    # live view: both stay on the microscope
    assert not sem.commands["auto_focus"].available
    assert not sem.commands["start_live"].available
    assert microscope._live_beams() == []


def test_a_disabled_column_gets_no_device(odemis_cls):
    microscope = make(odemis_cls, ion=False)
    assert set(microscope.beams) == {BeamType.ELECTRON}


def test_a_device_that_cannot_be_built_fails_the_connection(odemis_cls):
    """There is no other code for the device keys to fall back to."""

    def broken(channel):
        raise RuntimeError("no detector")

    stubs.use_components({"fibsem": FakeClient(), "stage-bare": FakeStage()})
    client = stubs._get_component("fibsem")
    object.__setattr__(client, "detector_type_info", broken)
    with pytest.raises(RuntimeError, match="no detector"):
        odemis_cls(_system())


def test_a_disabled_column_cannot_image(odemis_cls):
    microscope = make(odemis_cls, ion=False)
    with pytest.raises(ValueError, match="ION beam is not enabled"):
        microscope.acquire_image(beam_type=BeamType.ION)
    with pytest.raises(ValueError, match="ION beam is not enabled"):
        microscope.autocontrast(BeamType.ION)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_choices_are_the_old_available_values(odemis_cls, beam_type):
    microscope = make(odemis_cls)
    beam = microscope.beams[beam_type]
    for key in ("current", "voltage", "detector_type"):
        old = microscope.get_available_values(key, beam_type)
        assert sorted(beam.parameters[key].choices) == sorted(old), key


def test_an_unlisted_chamber_state_reads_unknown(odemis_cls):
    """The old key passed a state it did not know through as the client named it
    ("Prevac"); the device reads it as UNKNOWN, as the AutoScript chamber does."""
    READS["get_chamber_state"] = "Prevac"
    try:
        assert make(odemis_cls).get("chamber_state") == "Unknown"
    finally:
        READS["get_chamber_state"] = "vacuum"


def test_setting_the_plasma_gas_reaches_the_not_implemented_write(odemis_cls):
    """It raised TypeError from the old one-argument check before getting there."""
    microscope = make(odemis_cls)
    microscope.system.ion.plasma_gas = "Xenon"
    with pytest.raises(NotImplementedError):
        microscope.set("plasma_gas", "Argon", BeamType.ION)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.insert_manipulator(),
        lambda m: m.retract_manipulator(),
        lambda m: m.move_manipulator_relative(FibsemManipulatorPosition()),
        lambda m: m.move_manipulator_absolute(FibsemManipulatorPosition()),
        lambda m: m.move_manipulator_corrected(1e-6, 1e-6, BeamType.ION),
        lambda m: m.move_manipulator_to_position_offset(
            FibsemManipulatorPosition(), "EUCENTRIC"
        ),
        lambda m: m._get_saved_manipulator_position("PARK"),
    ],
    ids=[
        "insert",
        "retract",
        "relative",
        "absolute",
        "corrected",
        "offset",
        "saved",
    ],
)
def test_there_is_no_manipulator(odemis_cls, call):
    """The base class's answer: the methods raise rather than silently do nothing."""
    microscope = make(odemis_cls)
    assert not microscope.is_available("manipulator")
    assert microscope.manipulator_device is None
    with pytest.raises(NotImplementedError, match="does not support"):
        call(microscope)
