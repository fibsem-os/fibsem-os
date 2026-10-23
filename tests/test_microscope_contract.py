"""Contract tests for the FibsemMicroscope API, run against every simulated backend.

The FluorescenceMicroscope has its contract written down (`tests/fm/test_fm_contract.py`);
the microscope did not, so what a backend promises was spread across every test that
happens to use Demo. This file writes it down: the beam parameters (read and written as
the named methods read and write them), the named methods, the stage, chamber and
manipulator devices, stage moves, state, and image acquisition.

The tests that take the `microscope` fixture run on every driver with a fake SDK that
connects a whole microscope (`tests/fixtures/fake_drivers.py`): the Demo, which stands
in for ThermoFisher, and Tescan. A driver joins by adding its fake there. Not every
instrument has every part: a test asks the microscope what it has (a beam parameter,
whether it is settable, a chamber or manipulator device) and skips with the reason, or
pins what the API does without it, rather than keeping a skip list per driver. The
tests named `test_demo_*` (and the FM ones) pin the Demo's own simulation.

Until the Demo built from devices replaced the one before them
(`LegacyDemoMicroscope`, since deleted), every test here ran against both, so what is
pinned on the Demo is what it has always done.

What is pinned is today's behaviour, including the parts nobody would design on
purpose (a beam parameter read without a beam type raises). Where a test pins such a
quirk, its docstring says so. The string-key `get`/`set` this suite was first written
over were removed in 0.6.0.
"""

import logging
import numbers
import os
import tempfile
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest
import yaml

from fibsem import config as cfg
from fibsem import utils
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    MicroscopeState,
    Point,
    ScanMode,
)
from tests.fixtures import fake_drivers
from tests.fixtures.fake_drivers import FAKE_DRIVERS

BEAMS = [BeamType.ELECTRON, BeamType.ION]


def _connect(backend: str, base: str = cfg.DEFAULT_CONFIGURATION_PATH):
    """The Demo, built the way a session builds it, through `utils.setup_session`."""
    assert backend == "Demo", "the other drivers connect through fake_drivers"
    microscope, _ = utils.setup_session(
        config_path=base, manufacturer="Demo", setup_logging=False
    )
    return microscope


def test_backends_are_what_they_say(monkeypatch):
    from fibsem.drivers.demo.microscope import DemoMicroscope
    from fibsem.drivers.registry import get_driver

    assert type(_connect("Demo")) is DemoMicroscope
    for name in FAKE_DRIVERS:
        microscope = fake_drivers.connect(name, monkeypatch)
        assert type(microscope) is get_driver(name).load()


def test_every_built_in_driver_has_a_fake():
    """A built-in driver runs this suite over its fake SDK, or says why not yet."""
    from fibsem.drivers.registry import _BUILT_IN

    assert set(_BUILT_IN) == set(FAKE_DRIVERS) | set(fake_drivers.PENDING)
    assert not set(FAKE_DRIVERS) & set(fake_drivers.PENDING)


# The parts not every system has, and the patterns not every vendor can draw: a
# backend without them does not implement them, and the base class raises
# NotImplementedError.
OPTIONAL_METHODS = {
    "insert_manipulator",
    "retract_manipulator",
    "move_manipulator_relative",
    "move_manipulator_absolute",
    "move_manipulator_corrected",
    "move_manipulator_to_position_offset",
    "_get_saved_manipulator_position",
}


def test_optional_methods_are_not_abstract():
    from fibsem.microscope import FibsemMicroscope

    assert not OPTIONAL_METHODS & FibsemMicroscope.__abstractmethods__


def test_a_backend_without_an_optional_part_says_so():
    from fibsem.microscope import FibsemMicroscope

    class Minimal(FibsemMicroscope):
        pass

    for name in FibsemMicroscope.__abstractmethods__:
        setattr(Minimal, name, lambda self, *a, **k: None)
    Minimal.__abstractmethods__ = frozenset()
    minimal = Minimal.__new__(Minimal)
    with pytest.raises(NotImplementedError, match="Minimal does not support"):
        minimal.insert_manipulator("PARK")
    # milling goes to the milling service, and there is none
    with pytest.raises(ValueError, match="There is no milling"):
        minimal.draw_pattern(None)
    # the raw moves go through the devices, so a backend with none says so too
    minimal.stage_device = minimal.manipulator_device = None
    with pytest.raises(NotImplementedError, match="move_stage_absolute"):
        minimal.move_stage_absolute(FibsemStagePosition(x=0))
    with pytest.raises(NotImplementedError, match="retract_manipulator"):
        minimal.retract_manipulator()


@pytest.fixture(params=sorted(FAKE_DRIVERS))
def microscope(request, monkeypatch):
    return fake_drivers.connect(request.param, monkeypatch)


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_pair_of_ints(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, numbers.Integral) for v in value)
    )


def _read(microscope, name: str, beam_type):
    """A beam parameter, as the named methods read it (an enum as its plain value)."""
    return microscope._read_beam(name, beam_type)


def _write(microscope, name: str, value, beam_type) -> None:
    """A beam parameter, written as the named methods write it (unvalidated)."""
    microscope._write_beam(name, value, beam_type)


def _choices(microscope, name: str, beam_type) -> list:
    """A beam parameter's choices, or none where the beam does not have it."""
    param = _parameter(microscope, name, beam_type)
    return [] if param is None or param.choices is None else list(param.choices)


def _parameter(microscope, name: str, beam_type):
    """The beam's parameter, or None where the beam does not have it."""
    return microscope.beams[beam_type].parameters.get(name)


def _require(microscope, name: str, beam_type, settable: bool = False) -> None:
    """Skip unless the beam has the parameter (and, if asked, it can be set)."""
    param = _parameter(microscope, name, beam_type)
    if param is None:
        pytest.skip(f"the {beam_type.name} beam has no {name}")
    if settable and not param.settable:
        pytest.skip(f"the {beam_type.name} beam's {name} is not settable")


def _inserted(microscope) -> bool:
    return (
        microscope.manipulator_device.state.get_value()
        is InsertableDeviceState.INSERTED
    )


def _chamber_state(microscope) -> str:
    from fibsem.microscope import _chamber_state_name

    return _chamber_state_name(microscope.chamber_device.state.get_value())


def test_demo_reads_the_manipulator_and_chamber_from_their_devices():
    microscope = _connect("Demo")
    assert not _inserted(microscope)
    microscope.insert_manipulator("PARK")
    assert _inserted(microscope)
    assert (
        microscope.get_manipulator_position()
        == microscope.manipulator_device.position.cached
    )
    assert microscope.vent() == "Vented"
    assert microscope.pump() == "Pumped"


def test_demo_pumps_and_vents_through_its_chamber_device():
    microscope = _connect("Demo")
    chamber = microscope.chamber_device
    calls = []
    for name in ("_pump", "_vent"):
        original = getattr(chamber, name)
        setattr(
            chamber,
            name,
            lambda name=name, original=original: (calls.append(name), original()),
        )
    assert microscope.vent() == "Vented"
    assert chamber.pressure.cached == chamber.pressure.get_value()
    assert microscope.pump() == "Pumped"
    assert calls == ["_vent", "_pump"]


def test_demo_moves_its_manipulator_device():
    microscope = _connect("Demo")
    manipulator = microscope.manipulator_device
    calls = []
    for name in ("_insert", "_retract", "_move_absolute", "_move_relative"):
        original = getattr(manipulator, name)
        setattr(
            manipulator,
            name,
            lambda *args, name=name, original=original: (
                calls.append(name),
                original(*args),
            ),
        )
    microscope.insert_manipulator("PARK")
    microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))
    microscope.move_manipulator_corrected(1e-6, 1e-6, BeamType.ELECTRON)
    microscope.move_manipulator_to_position_offset(FibsemManipulatorPosition())
    microscope.move_manipulator_absolute(FibsemManipulatorPosition(z=1e-6))
    microscope.retract_manipulator()
    # insert and retract move through _move_absolute, as Demo's do.
    assert calls == [
        "_insert",
        "_move_absolute",
        "_move_relative",
        "_move_relative",
        "_move_absolute",
        "_move_absolute",
        "_retract",
        "_move_absolute",
    ]
    assert manipulator.state.cached is InsertableDeviceState.RETRACTED


def test_demo_moves_its_stage_device():
    """The Demo's stage methods go through its stage device."""
    microscope = _connect("Demo")
    stage = microscope.stage_device
    assert stage is not None
    calls = []

    def counted(name):
        original = getattr(stage, name)

        def call(*args):
            calls.append(name)
            return original(*args)

        return call

    for name in ("_move_absolute", "_move_relative", "_home"):
        setattr(stage, name, counted(name))
    microscope.move_stage_absolute(FibsemStagePosition(x=1e-3))
    microscope.move_stage_relative(FibsemStagePosition(y=1e-3))
    microscope.stage.home()
    assert calls == ["_move_absolute", "_move_relative", "_home"]
    assert np.allclose(
        _xyzrt(microscope.get_stage_position()), _xyzrt(stage.position.cached)
    )


def test_demo_stage_keeps_its_own_state():
    """The stage device owns its state: homing, linking and moves read and drive it."""
    microscope = _connect("Demo")
    stage = microscope.stage_device
    stage.sim_homed = False
    microscope.stage.home()
    microscope.stage.link()
    microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, t=0.1))
    assert stage.homed.get_value() is True
    assert stage.linked.get_value() is True
    position = stage.position.get_value()
    assert np.allclose(_xyzrt(position), _xyzrt(stage.sim_position))
    assert position.x == pytest.approx(1e-3)


def test_demo_chamber_keeps_its_own_state():
    """The chamber device owns its state: pumping and venting read and drive it."""
    microscope = _connect("Demo")
    chamber = microscope.chamber_device
    microscope.vent()
    assert _chamber_state(microscope) == "Vented"
    assert chamber.pressure.get_value() == chamber.sim_pressure
    microscope.pump()
    assert _chamber_state(microscope) == "Pumped"


def test_demo_manipulator_keeps_its_own_state():
    """The manipulator device owns its state: its moves read and drive it."""
    microscope = _connect("Demo")
    manipulator = microscope.manipulator_device
    assert not _inserted(microscope)
    microscope.insert_manipulator("PARK")
    microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))
    assert _inserted(microscope)
    position = manipulator.position.get_value()
    assert _xyzrt(position) == _xyzrt(manipulator.sim_position)
    assert position.x == pytest.approx(1e-6)


def test_demo_beams_keep_their_own_state():
    """The beam devices own their state, whatever the named methods do to the beams."""
    microscope = _connect("Demo")
    for beam_type in BEAMS:
        microscope.set_field_of_view(80e-6, beam_type)
        microscope.set_detector_contrast(0.9, beam_type)
        microscope.beam_shift(1e-6, 1e-6, beam_type)
        microscope.beams[beam_type].spot(Point(0.5, 0.5))
        microscope.blank(beam_type)
        microscope.set_full_frame_scanning_mode(beam_type)
        assert microscope.get_field_of_view(beam_type) == 80e-6
        assert microscope.beams[beam_type].sim_beam.hfw == 80e-6


@pytest.mark.parametrize("beam_type", BEAMS)
def test_demo_images_through_its_beam_devices(beam_type):
    """Acquiring, autocontrast and autofocus read and change the beam devices."""
    microscope = _connect("Demo")
    settings = ImageSettings(
        beam_type=beam_type, hfw=50e-6, resolution=(64, 48), dwell_time=1e-9
    )
    image = microscope.acquire_image(settings)
    beam = microscope.beams[beam_type]
    assert beam.sim_beam.hfw == 50e-6
    state = image.metadata.microscope_state
    beam_state = (
        state.electron_beam if beam_type is BeamType.ELECTRON else state.ion_beam
    )
    assert beam_state.hfw == 50e-6
    microscope.autocontrast(beam_type)
    assert beam.sim_detector.contrast == microscope.get_detector_contrast(beam_type)
    microscope.auto_focus(beam_type)
    assert beam.sim_beam.working_distance == microscope.get_working_distance(beam_type)


@pytest.mark.parametrize("beam_type", BEAMS)
def test_demo_scans_through_its_beam_commands(beam_type):
    """The Demo's scan-mode methods call the beam's commands."""
    microscope = _connect("Demo")
    beam = microscope.beams[beam_type]
    calls = []

    def counted(name):
        original = getattr(beam, name)

        def call(*args):
            calls.append((name, *args))
            return original(*args)

        return call

    for name in ("_spot", "_reduced_area", "_full_frame"):
        setattr(beam, name, counted(name))
    point, area = Point(0.5, 0.5), FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    microscope.beams[beam_type].spot(point)
    assert beam.scanning_mode.cached is ScanMode.SPOT
    microscope.set_reduced_area_scanning_mode(area, beam_type)
    assert beam.scanning_mode.cached is ScanMode.REDUCED_AREA
    microscope.set_full_frame_scanning_mode(beam_type)
    assert beam.scanning_mode.cached is ScanMode.FULL_FRAME
    assert calls == [("_spot", point), ("_reduced_area", area), ("_full_frame",)]


# ---------------------------------------------------------------------------
# What takes no beam type
# ---------------------------------------------------------------------------


def _has_chamber(m) -> bool:
    return m.chamber_device is not None


def _has_manipulator(m) -> bool:
    return m.manipulator_device is not None


def _stage_has(name: str) -> Callable[[Any], bool]:
    return lambda m: name in m.stage.parameters


# name -> (whether the microscope has it, how to read it, what the read must answer
# with).
GLOBAL_READS: Dict[
    str, Tuple[Callable[[Any], bool], Callable[[Any], Any], Callable[[Any], bool]]
] = {
    "chamber_state": (_has_chamber, _chamber_state, lambda v: isinstance(v, str)),
    "chamber_pressure": (
        _has_chamber,
        lambda m: m.chamber_device.pressure.get_value(),
        _is_real,
    ),
    "stage_position": (
        _stage_has("position"),
        lambda m: m.stage.position.get_value(),
        lambda v: isinstance(v, FibsemStagePosition),
    ),
    "stage_homed": (
        _stage_has("homed"),
        lambda m: m.stage.homed.get_value(),
        lambda v: isinstance(v, bool),
    ),
    "stage_linked": (
        _stage_has("linked"),
        lambda m: m.stage.linked.get_value(),
        lambda v: isinstance(v, bool),
    ),
    "manipulator_position": (
        _has_manipulator,
        lambda m: m.manipulator_device.position.get_value(),
        lambda v: isinstance(v, FibsemManipulatorPosition),
    ),
    "manipulator_state": (
        _has_manipulator,
        lambda m: m.manipulator_device.state.get_value(),
        lambda v: isinstance(v, InsertableDeviceState),
    ),
}


@pytest.mark.parametrize("name", sorted(GLOBAL_READS))
def test_global_reads(microscope, name):
    has, read, check = GLOBAL_READS[name]
    if not has(microscope):
        pytest.skip(f"{type(microscope).__name__} has no {name}")
    assert check(read(microscope))


# ---------------------------------------------------------------------------
# Beam parameters
# ---------------------------------------------------------------------------

# parameter -> (check on the value read, a value to write that is not the default).
# A value of None means "take one from its choices".
BEAM_KEYS: Dict[str, Tuple[Callable[[Any], bool], Any]] = {
    "voltage": (_is_real, None),
    "current": (_is_real, None),
    "working_distance": (_is_real, 5.5e-3),
    "hfw": (_is_real, 320e-6),
    "resolution": (_is_pair_of_ints, (768, 512)),
    "dwell_time": (_is_real, 2e-6),
    "stigmation": (lambda v: isinstance(v, Point), Point(1e-6, -2e-6)),
    "shift": (lambda v: isinstance(v, Point), Point(-3e-6, 4e-6)),
    "scan_rotation": (_is_real, 1.25),
    "detector_type": (lambda v: isinstance(v, str), None),
    "detector_mode": (lambda v: isinstance(v, str), None),
    "detector_contrast": (_is_real, 0.3),
    "detector_brightness": (_is_real, 0.7),
    "on": (lambda v: isinstance(v, bool), False),
    "blanked": (lambda v: isinstance(v, bool), True),
    "scanning_mode": (lambda v: isinstance(v, str), None),
}

# Parameters that are read but not written (the scan commands change the mode).
BEAM_READ_ONLY_KEYS = {"scanning_mode"}

# A value to write where the parameter lists no choices to take one from (a Tescan's
# electron voltage is set freely).
UNLISTED_VALUES: Dict[str, Any] = {"voltage": 2000.0}


def _new_value(microscope, key: str, beam_type: BeamType):
    """A valid value for `key` that differs from the one it has now."""
    value = BEAM_KEYS[key][1]
    if value is not None:
        return value
    current = _read(microscope, key, beam_type)
    listed = _choices(microscope, key, beam_type)
    if not listed and key in UNLISTED_VALUES:
        assert UNLISTED_VALUES[key] != current
        return UNLISTED_VALUES[key]
    choices = [c for c in listed if c != current]
    assert choices, f"{key} offers nothing to change to on {beam_type.name}"
    return choices[0]


def _image_settings(beam_type: BeamType) -> ImageSettings:
    return ImageSettings(
        beam_type=beam_type, hfw=50e-6, resolution=(64, 48), dwell_time=1e-6
    )


@pytest.mark.parametrize("key", sorted(BEAM_KEYS))
@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_key_reads(microscope, key, beam_type):
    """Every parameter the beam has reads as its type. One the instrument cannot set
    or report (a Tescan's resolution, dwell time and stigmation) is the last image's,
    so it reads once an image has been taken."""
    _require(microscope, key, beam_type)
    value = _read(microscope, key, beam_type)
    if value is None and not _parameter(microscope, key, beam_type).settable:
        microscope.acquire_image(_image_settings(beam_type))
        value = _read(microscope, key, beam_type)
    assert BEAM_KEYS[key][0](value)


@pytest.mark.parametrize("beam_type", BEAMS)
def test_a_beam_key_the_beam_lacks_reads_none(microscope, beam_type):
    """Absent = unsupported: a parameter the beam does not have reads None, and
    writing it changes nothing."""
    lacks = [k for k in BEAM_KEYS if _parameter(microscope, k, beam_type) is None]
    if not lacks:
        pytest.skip(f"the {beam_type.name} beam has every parameter")
    before = _readable_state(microscope)
    for key in lacks:
        _write(microscope, key, BEAM_KEYS[key][1], beam_type)
        assert _read(microscope, key, beam_type) is None, key
    assert _readable_state(microscope) == before


@pytest.mark.parametrize("key", sorted(BEAM_KEYS))
def test_beam_key_without_beam_type_raises(microscope, key):
    """Pinned quirk: a beam parameter read with no beam type raises rather than
    choosing a beam."""
    with pytest.raises(ValueError, match="needs a beam type"):
        _read(microscope, key, None)


@pytest.mark.parametrize(
    "key",
    sorted(set(BEAM_KEYS) - BEAM_READ_ONLY_KEYS),
)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_key_round_trips(microscope, key, beam_type):
    _require(microscope, key, beam_type, settable=True)
    value = _new_value(microscope, key, beam_type)
    _write(microscope, key, value, beam_type)
    assert _read(microscope, key, beam_type) == value


@pytest.mark.parametrize(
    "beam_type,system",
    [(BeamType.ELECTRON, "electron_beam"), (BeamType.ION, "ion_beam")],
)
def test_each_beam_can_be_disabled_on_its_own(microscope, beam_type, system):
    """The beam devices are what `is_available` answers from: a column with no
    device (``enabled: false`` at connect) is not there."""
    other = BeamType.ION if beam_type is BeamType.ELECTRON else BeamType.ELECTRON
    beams = dict(microscope.beams)
    microscope.beams = {other: beams[other]}
    assert not microscope.is_available(system)
    microscope.beams = beams
    assert microscope.is_available(system)


@pytest.mark.parametrize("key", sorted(set(BEAM_KEYS) - BEAM_READ_ONLY_KEYS))
def test_beam_key_write_leaves_the_other_beam_alone(microscope, key):
    _require(microscope, key, BeamType.ELECTRON, settable=True)
    before = _read(microscope, key, BeamType.ION)
    _write(
        microscope,
        key,
        _new_value(microscope, key, BeamType.ELECTRON),
        BeamType.ELECTRON,
    )
    assert _read(microscope, key, BeamType.ION) == before


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------

CHOICE_KEYS = ["current", "voltage", "detector_type", "detector_mode"]


@pytest.mark.parametrize("key", CHOICE_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_current_value_is_one_of_the_choices(microscope, key, beam_type):
    """A settable parameter that lists choices is at one of them. A Tescan's
    voltage lists none (it is set freely); its current is not settable, and the list
    it offers is a fixed one the instrument's current need not be on."""
    _require(microscope, key, beam_type, settable=True)
    choices = _choices(microscope, key, beam_type)
    if not choices:
        pytest.skip(f"the {beam_type.name} beam's {key} lists no choices")
    assert _read(microscope, key, beam_type) in choices


@pytest.mark.parametrize("key", CHOICE_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_every_choice_can_be_set(microscope, key, beam_type):
    _require(microscope, key, beam_type, settable=True)
    for choice in _choices(microscope, key, beam_type):
        _write(microscope, key, choice, beam_type)
        assert _read(microscope, key, beam_type) == choice


def _plasma_configuration() -> str:
    """The default configuration on a plasma FIB (Xenon)."""
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        configuration = yaml.safe_load(f)
    configuration["sim"] = {**(configuration.get("sim") or {}), "plasma_gas": "Xenon"}
    path = os.path.join(tempfile.mkdtemp(), "plasma-configuration.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(configuration, f)
    return path


def test_demo_beams_own_their_choices():
    """The beam parameters' choices come from the beam devices."""
    microscope = _connect("Demo", _plasma_configuration())
    for beam_type in BEAMS:
        for key in ("current", "voltage", "detector_type", "detector_mode"):
            assert _choices(microscope, key, beam_type)
    _write(microscope, "plasma_gas", "Argon", BeamType.ION)
    assert _read(microscope, "plasma_gas", BeamType.ION) == "Argon"


def test_demo_ion_currents_follow_the_plasma_gas():
    from fibsem.drivers.demo.simulator import SIMULATOR_BEAM_CURRENTS

    currents = SIMULATOR_BEAM_CURRENTS[BeamType.ION]
    microscope = _connect("Demo", _plasma_configuration())
    assert _read(microscope, "plasma_gas", BeamType.ION) == "Xenon"
    assert _choices(microscope, "current", BeamType.ION) == currents["Xenon"]
    _write(microscope, "plasma_gas", "Argon", BeamType.ION)
    assert _choices(microscope, "current", BeamType.ION) == currents["Argon"]
    _write(microscope, "plasma_gas", "Helium", BeamType.ION)  # not offered: ignored
    assert _read(microscope, "plasma_gas", BeamType.ION) == "Argon"


def test_demo_reads_its_configuration():
    """The configured choices and capabilities read the configuration."""
    microscope = _connect("Demo", _plasma_configuration())
    assert _choices(microscope, "plasma_gas", BeamType.ION)
    assert microscope.stage.position.limits


def test_demo_sets_its_milling_recipe():
    microscope = _connect("Demo")
    files = microscope._connection.milling_system.application_files
    microscope.setup_milling(
        FibsemMillingSettings(
            milling_channel=BeamType.ELECTRON,
            application_file=files[-1],
            patterning_mode="Parallel",
        )
    )
    milling = microscope._connection.milling_system
    assert milling.default_application_file == files[-1]
    assert milling.patterning_mode == "Parallel"
    assert milling.default_beam_type is BeamType.ELECTRON


def test_demo_has_no_simulated_parts_beside_its_devices():
    """The Demo is devices and the shared demo code, with no simulated parts of its
    own."""
    microscope = _connect("Demo")
    for part in (
        "chamber",
        "stage_system",
        "manipulator_system",
        "electron_system",
        "ion_system",
    ):
        assert not hasattr(microscope, part)


def test_plasma_gases_list_strings():
    """Only a plasma ion column offers gases: the ion beam's plasma gas choices."""
    microscope = _connect("Demo", _plasma_configuration())
    choices = _choices(microscope, "plasma_gas", BeamType.ION)
    assert choices and all(isinstance(c, str) for c in choices)
    assert _choices(microscope, "plasma_gas", BeamType.ELECTRON) == []


@pytest.mark.parametrize("beam_type", BEAMS)
def test_a_ga_column_offers_no_plasma_gas(microscope, beam_type):
    assert _choices(microscope, "plasma_gas", beam_type) == []


@pytest.mark.parametrize("beam_type", BEAMS)
def test_milling_choices_are_the_milling_services(microscope, beam_type):
    """The milling service lists the application files (``supported_settings``) and
    scan directions (``supported_pattern_settings``) where the instrument has them (a
    Tescan has no application files); no beam parameter has them."""
    milling = microscope.milling
    offered = {
        "application_file": milling.supported_settings(),
        "scan_direction": milling.supported_pattern_settings(),
    }
    assert "scan_direction" in offered["scan_direction"]
    for key, settings in offered.items():
        assert _choices(microscope, key, beam_type) == []
        if key in settings:
            choices = settings[key].choices
            assert choices and all(isinstance(c, str) for c in choices)


# ---------------------------------------------------------------------------
# What a beam does not have
# ---------------------------------------------------------------------------


def _readable_state(microscope) -> Dict[str, Any]:
    """Everything the contract reads, for detecting a write that should change
    nothing."""
    state = {
        name: read(microscope)
        for name, (has, read, _) in GLOBAL_READS.items()
        if has(microscope)
    }
    for beam_type in BEAMS:
        for key in BEAM_KEYS:
            state[(key, beam_type.name)] = _read(microscope, key, beam_type)
    return state


@pytest.mark.parametrize("beam_type", BEAMS)
def test_preset_is_a_silent_no_op(microscope, beam_type, caplog):
    """A beam with no preset (the Demo's): reading it is None and writing it does
    nothing, without a warning (absent = unsupported)."""
    if _parameter(microscope, "preset", beam_type) is not None:
        pytest.skip(f"the {beam_type.name} beam has presets")
    before = _readable_state(microscope)
    with caplog.at_level(logging.WARNING):
        assert microscope.get_preset(beam_type) is None
        microscope.set_preset("anything", beam_type)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert _readable_state(microscope) == before


# ---------------------------------------------------------------------------
# The named methods over the beam parameters
# ---------------------------------------------------------------------------

# (getter, setter, parameter): the getter must answer what the parameter reads, and
# the setter must write it and return what it then reads.
BEAM_WRAPPERS = [
    ("get_beam_current", "set_beam_current", "current"),
    ("get_beam_voltage", "set_beam_voltage", "voltage"),
    ("get_resolution", "set_resolution", "resolution"),
    ("get_field_of_view", "set_field_of_view", "hfw"),
    ("get_working_distance", "set_working_distance", "working_distance"),
    ("get_dwell_time", "set_dwell_time", "dwell_time"),
    ("get_stigmation", "set_stigmation", "stigmation"),
    ("get_beam_shift", "set_beam_shift", "shift"),
    ("get_scan_rotation", "set_scan_rotation", "scan_rotation"),
    ("get_detector_type", "set_detector_type", "detector_type"),
    ("get_detector_mode", "set_detector_mode", "detector_mode"),
    ("get_detector_contrast", "set_detector_contrast", "detector_contrast"),
    ("get_detector_brightness", "set_detector_brightness", "detector_brightness"),
]


@pytest.mark.parametrize("getter,setter,key", BEAM_WRAPPERS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_wrappers_match_their_key(microscope, getter, setter, key, beam_type):
    assert getattr(microscope, getter)(beam_type) == _read(microscope, key, beam_type)
    _require(microscope, key, beam_type, settable=True)
    value = _new_value(microscope, key, beam_type)
    returned = getattr(microscope, setter)(value, beam_type)
    assert returned == value
    assert _read(microscope, key, beam_type) == value


@pytest.mark.parametrize("beam_type", BEAMS)
def test_on_off_and_blanking(microscope, beam_type):
    assert microscope.turn_off(beam_type) is False
    assert microscope.is_on(beam_type) is False
    assert microscope.turn_on(beam_type) is True
    assert microscope.is_on(beam_type) is True
    if _parameter(microscope, "blanked", beam_type) is None:
        # absent = unsupported: blanking does nothing and says nothing
        assert microscope.blank(beam_type) is None
        assert microscope.is_blanked(beam_type) is None
        return
    assert microscope.blank(beam_type) is True
    assert microscope.is_blanked(beam_type) is True
    assert microscope.unblank(beam_type) is False
    assert microscope.is_blanked(beam_type) is False


@pytest.mark.parametrize("beam_type", BEAMS)
def test_scanning_modes(microscope, beam_type):
    _require(microscope, "scanning_mode", beam_type)
    microscope.beams[beam_type].spot(Point(0.5, 0.5))
    assert _read(microscope, "scanning_mode", beam_type) == "spot"
    microscope.set_reduced_area_scanning_mode(
        FibsemRectangle(0.25, 0.25, 0.5, 0.5), beam_type
    )
    assert _read(microscope, "scanning_mode", beam_type) == "reduced_area"
    microscope.set_full_frame_scanning_mode(beam_type)
    assert _read(microscope, "scanning_mode", beam_type) == "full_frame"


@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_shift_adds_to_the_shift(microscope, beam_type):
    start = microscope.get_beam_shift(beam_type)
    microscope.beam_shift(1e-6, -2e-6, beam_type)
    microscope.beam_shift(1e-6, -2e-6, beam_type)
    shift = microscope.get_beam_shift(beam_type)
    assert (shift.x, shift.y) == pytest.approx((start.x + 2e-6, start.y - 4e-6))


@pytest.mark.parametrize("beam_type", BEAMS)
def test_demo_spot_burn_reads_where_the_beam_is_parked(beam_type):
    """A spot burn marks the scene at the beam's spot, at its current field."""
    microscope = _connect("Demo")
    point = Point(0.25, 0.75)
    microscope.set_field_of_view(100e-6, beam_type)
    microscope.beams[beam_type].spot(point)
    spot, beam = microscope._spot_and_beam(beam_type)
    assert spot == point
    assert beam.hfw == microscope.get_field_of_view(beam_type)
    assert beam.resolution == microscope.get_resolution(beam_type)


def test_vent_and_pump(microscope):
    """A microscope with no chamber device says so rather than doing nothing."""
    if not _has_chamber(microscope):
        for method in (microscope.vent, microscope.pump):
            with pytest.raises(NotImplementedError, match=method.__name__):
                method()
        return
    assert microscope.vent() == "Vented"
    assert _chamber_state(microscope) == "Vented"
    assert microscope.pump() == "Pumped"
    assert _chamber_state(microscope) == "Pumped"


def test_home_homes_and_says_so(microscope):
    """`stage.home` returns whether the stage is homed afterwards. A stage that cannot
    home from here (a Tescan's) has no `homed` and lists `home` as unavailable."""
    if not _stage_has("homed")(microscope):
        assert microscope.stage.commands["home"].available is False
        return
    assert microscope.stage.home() is True
    assert microscope.stage.homed.get_value() is True


def _xyzrt(position: FibsemStagePosition) -> List[float]:
    return [position.x, position.y, position.z, position.r, position.t]


def test_get_stage_position_matches_the_device(microscope):
    assert _xyzrt(microscope.get_stage_position()) == _xyzrt(
        microscope.stage.position.get_value()
    )


def test_move_stage_absolute_arrives_and_returns_the_position(microscope):
    target = FibsemStagePosition(x=1e-3, y=-2e-3, z=3e-3, r=0.1, t=0.2)
    returned = microscope.move_stage_absolute(target)
    assert np.allclose(_xyzrt(returned), _xyzrt(target))
    assert np.allclose(_xyzrt(microscope.get_stage_position()), _xyzrt(target))


def test_move_stage_absolute_leaves_unset_axes_alone(microscope):
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1e-3, y=-2e-3, z=3e-3, r=0.1, t=0.2)
    )
    microscope.move_stage_absolute(FibsemStagePosition(x=5e-3))
    assert np.allclose(
        _xyzrt(microscope.get_stage_position()), [5e-3, -2e-3, 3e-3, 0.1, 0.2]
    )


def test_move_stage_relative_adds(microscope):
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1e-3, y=-2e-3, z=3e-3, r=0.1, t=0.2)
    )
    returned = microscope.move_stage_relative(FibsemStagePosition(x=1e-3, y=1e-3))
    expected = [2e-3, -1e-3, 3e-3, 0.1, 0.2]
    assert np.allclose(_xyzrt(returned), expected)
    assert np.allclose(_xyzrt(microscope.get_stage_position()), expected)


def test_safe_absolute_stage_movement_arrives(microscope):
    target = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=np.radians(170), t=0.3)
    microscope.safe_absolute_stage_movement(target)
    assert np.allclose(_xyzrt(microscope.get_stage_position()), _xyzrt(target))


# ---------------------------------------------------------------------------
# Manipulator
# ---------------------------------------------------------------------------


@pytest.fixture
def manipulator_microscope(microscope):
    if not _has_manipulator(microscope):
        pytest.skip(f"{type(microscope).__name__} has no manipulator")
    return microscope


def test_no_manipulator_says_so(microscope):
    """A microscope with no manipulator device raises on its manipulator methods."""
    if _has_manipulator(microscope):
        pytest.skip(f"{type(microscope).__name__} has a manipulator")
    with pytest.raises(NotImplementedError, match="insert_manipulator"):
        microscope.insert_manipulator("PARK")
    with pytest.raises(NotImplementedError, match="move_manipulator_relative"):
        microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))


def test_manipulator_inserts_and_retracts(manipulator_microscope):
    microscope = manipulator_microscope
    inserted = microscope.insert_manipulator("PARK")
    assert isinstance(inserted, FibsemManipulatorPosition)
    assert _xyzrt(inserted) == _xyzrt(microscope.get_manipulator_position())
    assert _inserted(microscope)
    retracted = microscope.retract_manipulator()
    assert isinstance(retracted, FibsemManipulatorPosition)
    assert _xyzrt(retracted) == _xyzrt(microscope.get_manipulator_position())
    assert not _inserted(microscope)


def test_manipulator_moves_absolute_and_relative(manipulator_microscope):
    microscope = manipulator_microscope
    target = FibsemManipulatorPosition(x=1e-6, y=2e-6, z=3e-6)
    moved = microscope.move_manipulator_absolute(deepcopy(target))
    assert np.allclose(_xyzrt(moved), _xyzrt(target))
    moved = microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))
    assert np.allclose(_xyzrt(moved), [2e-6, 2e-6, 3e-6, 0, 0])
    assert _xyzrt(moved) == _xyzrt(microscope.get_manipulator_position())


@pytest.mark.parametrize("beam_type", BEAMS)
def test_manipulator_corrected_move_returns_where_it_is(
    manipulator_microscope, beam_type
):
    microscope = manipulator_microscope
    moved = microscope.move_manipulator_corrected(1e-6, -1e-6, beam_type)
    assert isinstance(moved, FibsemManipulatorPosition)
    assert _xyzrt(moved) == _xyzrt(microscope.get_manipulator_position())


def test_manipulator_moves_to_an_offset_from_a_saved_position(manipulator_microscope):
    microscope = manipulator_microscope
    eucentric = microscope._get_saved_manipulator_position("EUCENTRIC")
    offset = FibsemManipulatorPosition(x=1e-6, z=-2e-6)
    moved = microscope.move_manipulator_to_position_offset(
        deepcopy(offset), "EUCENTRIC"
    )
    assert np.allclose(_xyzrt(moved), _xyzrt(eucentric + offset))


def test_unknown_saved_manipulator_position_raises(manipulator_microscope):
    microscope = manipulator_microscope
    with pytest.raises(ValueError):
        microscope._get_saved_manipulator_position("NOWHERE")


# ---------------------------------------------------------------------------
# Fluorescence (on a configuration with an FM)
# ---------------------------------------------------------------------------

FM_CONFIGURATION = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


@pytest.fixture
def fm_microscope():
    return _connect("Demo", FM_CONFIGURATION)


def test_a_compustage_is_never_linked_and_cannot_link(fm_microscope, caplog):
    """The Arctis simulator is a compustage: its stage has no link, so linking is
    unavailable and whether it is linked is unsupported, without a warning."""
    assert fm_microscope._fm_is_a_pose()
    with caplog.at_level(logging.WARNING):
        assert not fm_microscope.stage.commands["link"].available
        assert fm_microscope.stage.parameters.get("linked") is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_fm_acquires_a_channel(fm_microscope):
    from fibsem.fm.structures import ChannelSettings

    channel = ChannelSettings(excitation_wavelength=450, power=0.3, exposure_time=0.02)
    image = fm_microscope.fm.acquire_image(channel)
    assert image.data.ndim == 2 and image.data.size > 0
    assert fm_microscope.fm.camera.exposure_time == 0.02
    assert fm_microscope.fm.light_source.power == 0.3
    assert fm_microscope.fm.filter_set.excitation_wavelength == 450


def test_fm_objective_inserts_and_retracts(fm_microscope):
    objective = fm_microscope.fm.objective
    objective.insert()
    assert objective.state is InsertableDeviceState.INSERTED
    objective.retract()
    assert objective.state is InsertableDeviceState.RETRACTED


def test_demo_builds_no_fm_devices_without_an_fm():
    microscope = _connect("Demo")
    assert microscope.fm is None
    assert dict(microscope.fm_devices) == {}


def test_demo_fm_devices_share_state_with_the_fm():
    """The devices drive the parts `fm` holds: a change on either side shows on both."""
    microscope = _connect("Demo", FM_CONFIGURATION)
    fm, devices = microscope.fm, microscope.fm_devices
    assert sorted(devices) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    devices["camera"].exposure_time.set_value(0.2)
    assert fm.camera.exposure_time == 0.2
    fm.light_source.power = 0.4
    assert devices["light_source"].power.get_value() == 0.4
    devices["objective"].insert()
    assert fm.objective.state is InsertableDeviceState.INSERTED
    assert devices["objective"].state.cached is InsertableDeviceState.INSERTED


def test_demo_fm_group_acquires_a_channel():
    from fibsem.fm.structures import ChannelSettings

    microscope = _connect("Demo", FM_CONFIGURATION)
    devices = microscope.fm_devices
    channel = ChannelSettings(excitation_wavelength=450, power=0.3, exposure_time=0.02)
    data = devices["fm"].acquire_channel(channel.to_dict())
    assert data.ndim == 2 and data.size > 0
    # The group sets the parts through their parameters, so each is cached.
    assert devices["camera"].exposure_time.cached == 0.02
    assert devices["light_source"].power.cached == 0.3
    assert devices["filter_set"].excitation_wavelength.cached == 450


def test_demo_fm_is_the_fm_api_over_devices():
    """`fm` is the same FM API over devices a remote FM is, over the Demo FM devices."""
    from fibsem.drivers.demo.devices import DemoCamera
    from fibsem.fm.microscope import FluorescenceMicroscope

    microscope = _connect("Demo", FM_CONFIGURATION)
    fm = microscope.fm
    assert isinstance(fm, FluorescenceMicroscope)
    assert fm.devices == dict(microscope.fm_devices)
    assert isinstance(microscope.fm_devices["camera"], DemoCamera)
    # the frame is the camera's, at its binned resolution
    width, height = fm.camera.resolution
    assert fm.acquire_image().data.shape == (height, width)


def test_demo_fm_snaps_excitation_to_its_bands(caplog):
    """As the hardware does: an excitation between bands selects the nearest, with a
    warning."""
    microscope = _connect("Demo", FM_CONFIGURATION)
    microscope.fm.filter_set.excitation_wavelength = 488
    assert microscope.fm.filter_set.excitation_wavelength == 450
    assert "set to the nearest, 450" in caplog.text


def test_demo_fm_names_a_banded_emission_as_its_multi_band_filter():
    """The simulator's filter set has no bands, so a banded filter means its
    multi-band filter, as Thermo reports one."""
    from fibsem.fm.structures import REFLECTION, EmissionFilter

    microscope = _connect("Demo", FM_CONFIGURATION)
    microscope.fm.filter_set.emission_filter = EmissionFilter.band("GFP", 510, 560)
    assert microscope.fm.filter_set.emission_filter == EmissionFilter(
        "Fluorescence", multi_band=True
    )
    microscope.fm.filter_set.emission_filter = REFLECTION
    assert microscope.fm.filter_set.emission_filter == REFLECTION


def test_demo_fm_objective_clips_to_its_limit():
    microscope = _connect("Demo", FM_CONFIGURATION)
    objective = microscope.fm.objective
    objective.limit_position = 5e-3
    objective.move_absolute(7e-3)
    assert objective.position == 5e-3
    assert microscope.fm_devices["objective"].limit_position.cached == 5e-3


# ---------------------------------------------------------------------------
# State and imaging
# ---------------------------------------------------------------------------


def test_microscope_state_restores(microscope):
    state = microscope.get_microscope_state()
    assert isinstance(state, MicroscopeState)
    before = _readable_state(microscope)

    microscope.move_stage_absolute(FibsemStagePosition(x=4e-3, y=4e-3))
    written = [
        (key, beam_type)
        for beam_type in BEAMS
        for key in ("working_distance", "hfw", "dwell_time", "detector_contrast")
        if (param := _parameter(microscope, key, beam_type)) is not None
        and param.settable
    ]
    for key, beam_type in written:
        _write(microscope, key, _new_value(microscope, key, beam_type), beam_type)
    assert _readable_state(microscope) != before

    microscope.set_microscope_state(state)
    after = _readable_state(microscope)
    for key, beam_type in written:
        assert np.isclose(
            after[(key, beam_type.name)], before[(key, beam_type.name)]
        ), key
    assert np.allclose(
        _xyzrt(after["stage_position"]), _xyzrt(before["stage_position"])
    )


@pytest.mark.parametrize("beam_type", BEAMS)
def test_acquire_image(microscope, beam_type):
    settings = ImageSettings(
        beam_type=beam_type,
        resolution=(768, 512),
        hfw=100e-6,
        dwell_time=1e-6,
        save=False,
    )
    image = microscope.acquire_image(image_settings=settings)
    assert isinstance(image, FibsemImage)
    assert image.data.shape == (512, 768)
    assert image.metadata.image_settings.beam_type is beam_type
    assert tuple(image.metadata.image_settings.resolution) == (768, 512)
    assert np.isclose(image.metadata.image_settings.hfw, 100e-6)
    assert np.isclose(microscope.get_field_of_view(beam_type), 100e-6)


@pytest.mark.parametrize("beam_type", BEAMS)
def test_finish_milling_puts_the_beam_back(microscope, beam_type):
    """What the first setup_milling found comes back; a second finish has nothing
    left to put back.

    Where the current is not settable (a Tescan's comes with the milling preset,
    which the fake SDK does not model), only the putting back is checked."""
    from fibsem.structures import FibsemMillingSettings

    channels = microscope.milling.supported_settings()["milling_channel"].choices
    if channels and beam_type not in channels:
        pytest.skip(f"{type(microscope).__name__} does not mill with {beam_type.name}")
    sets_current = _parameter(microscope, "current", beam_type).settable

    def conditions():
        return [
            _read(microscope, key, beam_type) for key in ("voltage", "current", "hfw")
        ]

    before = conditions()
    for current, hfw in ((7.6e-9, 80e-6), (2e-9, 50e-6)):
        microscope.setup_milling(
            FibsemMillingSettings(
                milling_channel=beam_type, milling_current=current, hfw=hfw
            )
        )
    if sets_current:
        assert conditions() != before
    microscope.finish_milling()
    assert np.allclose(conditions(), before)
    if not sets_current:
        return

    _write(microscope, "hfw", 40e-6, beam_type)
    microscope.finish_milling()
    assert _read(microscope, "hfw", beam_type) == 40e-6


def test_each_microscope_has_its_own_imaging_lock():
    first, second = _connect("Demo"), _connect("Demo")
    assert first._threading_lock is not second._threading_lock
    assert first.resources is not second.resources


def test_the_imaging_channel_resource_is_the_old_paths_lock():
    from fibsem.devices import IMAGING_CHANNEL
    from fibsem.devices.stage import STAGE_RESOURCE

    microscope = _connect("Demo")
    assert microscope.resources.lock(IMAGING_CHANNEL) is microscope._threading_lock
    # a stage move does not wait for a frame
    assert microscope.resources.lock(STAGE_RESOURCE) is not microscope._threading_lock


def test_demo_devices_claim_the_microscopes_resources():
    microscope = _connect("Demo", FM_CONFIGURATION)
    devices = [
        *microscope.beams.values(),
        microscope.stage_device,
        microscope.chamber_device,
        *microscope.fm_devices.values(),
    ]
    # the simulated Arctis has no manipulator (`sim.has_manipulator: false`)
    assert microscope.manipulator_device is None
    assert microscope.fm_devices
    assert all(device.resources is microscope.resources for device in devices)
