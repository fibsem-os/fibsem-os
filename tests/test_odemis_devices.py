"""The Odemis beams and stage as devices make the odemis calls the old code makes.

``OdemisBeam`` and ``OdemisStage`` are ``OdemisThermoMicroscope``'s beam and stage
keys moved onto the devices. Each case runs an old call on a microscope without the
devices and on one that built them as it does when created, both over a fake odemis
client and stage that record every call, and requires the same result (or error), the
same calls in the same order and the same logged messages.

No odemis installation: odemis is replaced by the stub modules in
``tests/fm/_odemis_stubs.py``. Nothing here has run on an instrument.
"""

import logging
import os
import sys
from types import MappingProxyType

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    BeamType,
    FibsemDetectorSettings,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from tests.fm import _odemis_stubs as stubs

ODEMIS_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "odemis-configuration.yaml")

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


def make(cls, devices=True, ion=True):
    """An OdemisThermoMicroscope over the fake client, created as usual, and without
    its devices unless *devices*."""
    stubs.use_components({"fibsem": FakeClient(), "stage-bare": FakeStage()})
    microscope = cls(_system(ion=ion))
    if not devices:
        microscope.beams = MappingProxyType({})
        microscope._beam_routes = MappingProxyType({})
        microscope.stage = None
        microscope._device_routes = MappingProxyType({})
        microscope._command_routes = MappingProxyType({})
    return microscope


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
    return result, list(LOG), messages.records


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
            lambda m, b=beam_type: m.set_reduced_area_scanning_mode(
                FibsemRectangle(0.1, 0.1, 0.5, 0.5), b
            ),
        )
        yield (
            f"full frame {name}",
            lambda m, b=beam_type: m.set_full_frame_scanning_mode(b),
        )
        for key in ("current", "voltage", "detector_type", "detector_mode"):
            yield (
                f"available {key} {name}",
                lambda m, k=key, b=beam_type: sorted(m.get_available_values(k, b)),
            )


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

CASES = tuple(_beam_cases()) + STAGE_CASES

# The one call the devices add: the home command reads back whether the stage is
# homed, as ``home()`` always has, so a bare ``set("stage_home")`` asks once more.
EXTRA_READS = {"set stage_home": [["is_homed", [], {}]]}


@pytest.mark.parametrize("key,call", CASES, ids=[key for key, _ in CASES])
def test_the_devices_make_the_same_odemis_calls_logs_and_results(odemis_cls, key, call):
    old = run(make(odemis_cls, devices=False), call)
    new = run(make(odemis_cls), call)
    assert new == (old[0], old[1] + EXTRA_READS.get(key, []), old[2])


def test_the_cases_make_odemis_calls(odemis_cls):
    calls = [run(make(odemis_cls, devices=False), call)[1] for _, call in CASES]
    assert sum(1 for c in calls if c) > len(CASES) * 0.8


def test_creating_the_microscope_builds_the_beams_and_stage(odemis_cls):
    from fibsem.devices.drivers.odemis import OdemisBeam, OdemisStage

    microscope = make(odemis_cls)
    assert set(microscope.beams) == {BeamType.ELECTRON, BeamType.ION}
    assert all(isinstance(b, OdemisBeam) for b in microscope.beams.values())
    assert isinstance(microscope.stage, OdemisStage)
    electron = microscope.beams[BeamType.ELECTRON]
    assert "plasma_gas" not in electron.parameters
    assert "preset" not in electron.parameters
    assert "scanning_mode" not in electron.parameters
    assert not electron.commands["spot"].available
    assert electron.current.choices[0] == 1e-12
    assert electron.voltage.choices == sorted(electron.voltage.choices)
    assert sorted(electron.detector_type.choices) == ["ETD", "TLD"]
    assert sorted(microscope.stage.axes) == ["r", "t", "x", "y", "z"]
    assert microscope.stage.commands["link"].available


def test_the_calls_go_through_the_devices(odemis_cls):
    microscope = make(odemis_cls)
    used = []
    for device, hooks in (
        (microscope.stage, ("_move_absolute", "_move_relative", "_home")),
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
    assert used == ["_move_absolute", "_move_relative", "_home"]
    beam = microscope.beams[BeamType.ION]
    microscope.set("hfw", 50e-6, BeamType.ION)
    assert beam.hfw.cached == 50e-6


def test_a_disabled_column_gets_no_device(odemis_cls):
    microscope = make(odemis_cls, ion=False)
    assert set(microscope.beams) == {BeamType.ELECTRON}


def test_a_device_that_cannot_be_built_leaves_the_old_code(odemis_cls, caplog):
    def broken(channel):
        raise RuntimeError("no detector")

    stubs.use_components({"fibsem": FakeClient(), "stage-bare": FakeStage()})
    client = stubs._get_component("fibsem")
    object.__setattr__(client, "detector_type_info", broken)
    with caplog.at_level(logging.WARNING):
        microscope = odemis_cls(_system())
    assert dict(microscope.beams) == {}
    assert microscope.stage is None
    assert microscope._route("stage_position", None) is None
    assert "Could not build the beam and stage devices" in caplog.text
    assert microscope.get("hfw", BeamType.ELECTRON) == 150e-6


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_choices_are_the_old_available_values(odemis_cls, beam_type):
    microscope = make(odemis_cls)
    beam = microscope.beams[beam_type]
    for key in ("current", "voltage", "detector_type"):
        old = microscope.get_available_values(key, beam_type)
        assert sorted(beam.parameters[key].choices) == sorted(old), key
