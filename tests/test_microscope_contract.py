"""Contract tests for the FibsemMicroscope API, run against every simulated backend.

The FluorescenceMicroscope has its contract written down (`tests/fm/test_fm_contract.py`);
the microscope did not, so what a backend promises was spread across every test that
happens to use Demo. This file writes it down, through the public API only: `get`/`set`
keys, their base-class wrappers, stage moves, state, and image acquisition.

It runs against `DemoMicroscope`, the demo backend built from devices. Until it
replaced the Demo before devices (`LegacyDemoMicroscope`, since deleted), every test
here ran against both, so what is pinned is what the Demo has always done.

What is pinned is today's behaviour, including the parts nobody would design on
purpose (a beam key read without a beam type raises; `stage_link` links whatever value
it is given; an unknown key is a silent `None`). The old API keeps every one of them.
Where a test pins such a quirk, its docstring says so.
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

# The backends this suite runs against. Each is built the way a session builds it,
# through `utils.setup_session`.
BACKENDS = ["Demo"]

BEAMS = [BeamType.ELECTRON, BeamType.ION]


def _connect(backend: str, base: str = cfg.DEFAULT_CONFIGURATION_PATH):
    microscope, _ = utils.setup_session(
        config_path=base, manufacturer="Demo", setup_logging=False
    )
    return microscope


def test_backends_are_what_they_say():
    from fibsem.microscopes.device_demo import DemoMicroscope

    assert type(_connect("Demo")) is DemoMicroscope


# The parts not every system has, and the patterns not every vendor can draw: a
# backend without them does not implement them, and the base class raises
# NotImplementedError.
OPTIONAL_METHODS = {
    "last_image",
    "acquire_chamber_image",
    "insert_manipulator",
    "retract_manipulator",
    "move_manipulator_relative",
    "move_manipulator_absolute",
    "move_manipulator_corrected",
    "move_manipulator_to_position_offset",
    "_get_saved_manipulator_position",
    "draw_bitmap_pattern",
    "draw_polygon",
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
    with pytest.raises(NotImplementedError, match="draw_polygon"):
        minimal.draw_polygon(None)
    # the raw moves go through the devices, so a backend with none says so too
    minimal.stage_device = minimal.manipulator_device = None
    with pytest.raises(NotImplementedError, match="move_stage_absolute"):
        minimal.move_stage_absolute(FibsemStagePosition(x=0))
    with pytest.raises(NotImplementedError, match="retract_manipulator"):
        minimal.retract_manipulator()


@pytest.fixture(params=BACKENDS)
def microscope(request):
    return _connect(request.param)


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_pair_of_ints(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, numbers.Integral) for v in value)
    )


# The keys the Demo serves from its devices. The
# contract above runs through them; this checks that it really does.
DEMO_ROUTED_BEAM_KEYS = [
    "voltage",
    "current",
    "working_distance",
    "hfw",
    "scan_rotation",
    "blanked",
    "detector_type",
    "detector_mode",
    "detector_contrast",
    "detector_brightness",
    "resolution",
    "dwell_time",
    "stigmation",
    "shift",
    "on",
    "scanning_mode",
]


@pytest.mark.parametrize("key", DEMO_ROUTED_BEAM_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_demo_serves_its_beam_keys_from_devices(key, beam_type):
    microscope = _connect("Demo")
    param = microscope._route(key, beam_type)
    assert param is not None
    assert param.device is microscope.beams[beam_type]


def test_demo_reads_the_manipulator_and_chamber_from_their_devices():
    """The wrappers over these keys never reach the _get/_set chain on the Demo."""
    microscope = _connect("Demo")
    chain_get = microscope._get
    keys = ("manipulator_position", "manipulator_state", "chamber_state")

    def refuse(key, beam_type=None):
        assert key not in keys, f"{key} went to the _get/_set chain"
        return chain_get(key, beam_type)

    microscope._get = refuse
    assert microscope.get_manipulator_state() is False
    microscope.insert_manipulator("PARK")
    assert microscope.get_manipulator_state() is True
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
    assert chamber.pressure.cached == microscope.get("chamber_pressure")
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
    """The Demo's stage methods go through its stage device, not _get/_set."""
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
    microscope.home()
    assert calls == ["_move_absolute", "_move_relative", "_home"]
    assert np.allclose(
        _xyzrt(microscope.get_stage_position()), _xyzrt(stage.position.cached)
    )


def test_demo_stage_keeps_its_own_state_and_serves_the_stage_keys():
    """The stage device owns its state: the stage keys read and drive it."""
    microscope = _connect("Demo")
    chain = []
    for name in ("_get", "_set"):
        original = getattr(microscope, name)
        setattr(
            microscope,
            name,
            lambda key, *args, original=original: (
                chain.append(key),
                original(key, *args),
            )[1],
        )
    microscope.stage_device.sim_homed = False
    microscope.set("stage_home", True)
    microscope.set("stage_link", True)
    microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, t=0.1))
    assert microscope.get("stage_homed") is True
    assert microscope.get("stage_linked") is True
    assert np.allclose(
        _xyzrt(microscope.get("stage_position")),
        _xyzrt(microscope.stage_device.sim_position),
    )
    assert microscope.get("stage_position").x == pytest.approx(1e-3)
    assert not [key for key in chain if key.startswith("stage_")]


def test_demo_chamber_keeps_its_own_state_and_serves_the_chamber_keys():
    """The chamber device owns its state: the chamber keys read and drive it."""
    microscope = _connect("Demo")
    chain = []
    for name in ("_get", "_set"):
        original = getattr(microscope, name)
        setattr(
            microscope,
            name,
            lambda key, *args, original=original: (
                chain.append(key),
                original(key, *args),
            )[1],
        )
    microscope.set("vent_chamber", True)
    assert microscope.get("chamber_state") == "Vented"
    assert microscope.get("chamber_pressure") == microscope.chamber_device.sim_pressure
    microscope.set("pump_chamber", True)
    assert microscope.get("chamber_state") == "Pumped"
    assert not chain


def test_demo_manipulator_keeps_its_own_state_and_serves_its_keys():
    """The manipulator device owns its state: the manipulator keys read it."""
    microscope = _connect("Demo")
    chain = []
    original = microscope._get
    microscope._get = lambda key, *args: (chain.append(key), original(key, *args))[1]
    assert microscope.get("manipulator_state") is False
    microscope.insert_manipulator("PARK")
    microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))
    assert microscope.get("manipulator_state") is True
    assert _xyzrt(microscope.get("manipulator_position")) == _xyzrt(
        microscope.manipulator_device.sim_position
    )
    assert microscope.get("manipulator_position").x == pytest.approx(1e-6)
    assert not chain


def test_demo_beams_keep_their_own_state():
    """The beam devices own their state, whatever the old API does to the beams."""
    microscope = _connect("Demo")
    for beam_type in BEAMS:
        microscope.set("hfw", 80e-6, beam_type)
        microscope.set("detector_contrast", 0.9, beam_type)
        microscope.beam_shift(1e-6, 1e-6, beam_type)
        microscope.set("spot_mode", Point(0.5, 0.5), beam_type)
        microscope.blank(beam_type)
        microscope.set("full_frame", None, beam_type)
        assert microscope.get("hfw", beam_type) == 80e-6
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
    assert microscope.last_image(beam_type) is image
    microscope.autocontrast(beam_type)
    assert beam.sim_detector.contrast == microscope.get("detector_contrast", beam_type)
    microscope.auto_focus(beam_type)
    assert beam.sim_beam.working_distance == microscope.get(
        "working_distance", beam_type
    )


@pytest.mark.parametrize("beam_type", BEAMS)
def test_demo_scans_through_its_beam_commands(beam_type):
    """The Demo's scan-mode methods and keys call the beam's commands, not _set."""
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
    microscope.set_spot_scanning_mode(point, beam_type)
    assert beam.scanning_mode.cached is ScanMode.SPOT
    microscope.set_reduced_area_scanning_mode(area, beam_type)
    assert beam.scanning_mode.cached is ScanMode.REDUCED_AREA
    microscope.set_full_frame_scanning_mode(beam_type)
    assert beam.scanning_mode.cached is ScanMode.FULL_FRAME
    assert calls == [("_spot", point), ("_reduced_area", area), ("_full_frame",)]
    # and so do the old keys, which the shared set routes to the commands
    calls.clear()
    microscope.set("spot_mode", point, beam_type)
    microscope.set("reduced_area", area, beam_type)
    microscope.set("full_frame", None, beam_type)
    assert calls == [("_spot", point), ("_reduced_area", area), ("_full_frame",)]


# ---------------------------------------------------------------------------
# Keys that take no beam type
# ---------------------------------------------------------------------------

# key -> what `get(key)` must answer with.
GLOBAL_READ_KEYS: Dict[str, Callable[[Any], bool]] = {
    "manufacturer": lambda v: isinstance(v, str),
    "model": lambda v: isinstance(v, str),
    "serial_number": lambda v: isinstance(v, str),
    "software_version": lambda v: isinstance(v, str),
    "hardware_version": lambda v: isinstance(v, str),
    "chamber_state": lambda v: isinstance(v, str),
    "chamber_pressure": _is_real,
    "stage_position": lambda v: isinstance(v, FibsemStagePosition),
    "stage_homed": lambda v: isinstance(v, bool),
    "stage_linked": lambda v: isinstance(v, bool),
    "manipulator_position": lambda v: isinstance(v, FibsemManipulatorPosition),
    "manipulator_state": lambda v: isinstance(v, bool),
}


@pytest.mark.parametrize("key", sorted(GLOBAL_READ_KEYS))
def test_global_key_reads(microscope, key):
    assert GLOBAL_READ_KEYS[key](microscope.get(key))


@pytest.mark.parametrize("key", sorted(GLOBAL_READ_KEYS))
@pytest.mark.parametrize("beam_type", BEAMS)
def test_global_key_ignores_beam_type(microscope, key, beam_type):
    """A key with no beam gives the same answer whatever beam type is passed."""
    assert microscope.get(key, beam_type) == microscope.get(key)


INFO_KEYS = (
    "manufacturer",
    "model",
    "serial_number",
    "software_version",
    "hardware_version",
)


@pytest.mark.parametrize("key", INFO_KEYS)
def test_info_keys_are_the_configured_ones(microscope, key):
    assert microscope.get(key) == getattr(microscope.system.info, key)


# ---------------------------------------------------------------------------
# Keys that take a beam type
# ---------------------------------------------------------------------------

# key -> (check on the value read, a value to write that is not the default).
# A value of None means "take one from get_available_values".
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
    "beam_enabled": (lambda v: isinstance(v, bool), False),
    "eucentric_height": (_is_real, 9.5e-3),
    "column_tilt": (_is_real, 45),
    "scanning_mode": (lambda v: isinstance(v, str), None),
}

# Keys that are read with a beam type but not written through `set`.
BEAM_READ_ONLY_KEYS = {"scanning_mode"}


def _new_value(microscope, key: str, beam_type: BeamType):
    """A valid value for `key` that differs from the one it has now."""
    value = BEAM_KEYS[key][1]
    if value is not None:
        return value
    current = microscope.get(key, beam_type)
    choices = [
        c for c in microscope.get_available_values(key, beam_type) if c != current
    ]
    assert choices, f"{key} offers nothing to change to on {beam_type.name}"
    return choices[0]


@pytest.mark.parametrize("key", sorted(BEAM_KEYS))
@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_key_reads(microscope, key, beam_type):
    assert BEAM_KEYS[key][0](microscope.get(key, beam_type))


@pytest.mark.parametrize("key", sorted(BEAM_KEYS))
def test_beam_key_without_beam_type_raises(microscope, key):
    """Pinned quirk: a beam key read with no beam type raises rather than choosing one.

    The exception type is not part of the contract (Demo raises ValueError for some
    keys and UnboundLocalError for others); raising is.
    """
    with pytest.raises(Exception):
        microscope.get(key)


@pytest.mark.parametrize(
    "key",
    sorted(set(BEAM_KEYS) - BEAM_READ_ONLY_KEYS),
)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_key_round_trips(microscope, key, beam_type):
    value = _new_value(microscope, key, beam_type)
    microscope.set(key, value, beam_type)
    assert microscope.get(key, beam_type) == value


@pytest.mark.parametrize(
    "beam_type,system",
    [(BeamType.ELECTRON, "electron_beam"), (BeamType.ION, "ion_beam")],
)
def test_each_beam_can_be_disabled_on_its_own(microscope, beam_type, system):
    """`beam_enabled` is what `is_available` and `get_available_beams` answer from."""
    other = BeamType.ION if beam_type is BeamType.ELECTRON else BeamType.ELECTRON
    microscope.set("beam_enabled", False, beam_type)
    assert microscope.get("beam_enabled", beam_type) is False
    assert not microscope.is_available(system)
    assert microscope.get_available_beams() == [other]
    microscope.set("beam_enabled", True, beam_type)
    assert microscope.is_available(system)
    assert microscope.get_available_beams() == BEAMS


# key -> the field it is on the beam's system settings.
BEAM_CONFIG_KEYS = {
    "beam_enabled": "enabled",
    "eucentric_height": "eucentric_height",
    "column_tilt": "column_tilt",
}


@pytest.mark.parametrize("key", sorted(BEAM_CONFIG_KEYS))
@pytest.mark.parametrize("beam_type", BEAMS)
def test_config_keys_are_answered_before_the_backend(
    microscope, monkeypatch, key, beam_type
):
    """The base class answers the configuration keys from `system`, so no backend
    can answer them differently: `_get` and `_set` are never asked."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"{key} reached the backend")

    monkeypatch.setattr(microscope, "_get", refuse)
    monkeypatch.setattr(microscope, "_set", refuse)
    settings = (
        microscope.system.electron
        if beam_type is BeamType.ELECTRON
        else microscope.system.ion
    )
    value = not microscope.get(key, beam_type) if key == "beam_enabled" else 0.123
    microscope.set(key, value, beam_type)
    assert microscope.get(key, beam_type) == value
    assert getattr(settings, BEAM_CONFIG_KEYS[key]) == value


@pytest.mark.parametrize("key", INFO_KEYS)
def test_info_keys_are_answered_before_the_backend(microscope, monkeypatch, key):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{key} reached the backend")

    monkeypatch.setattr(microscope, "_get", refuse)
    assert microscope.get(key) == getattr(microscope.system.info, key)


@pytest.mark.parametrize("key", sorted(set(BEAM_KEYS) - BEAM_READ_ONLY_KEYS))
def test_beam_key_write_leaves_the_other_beam_alone(microscope, key):
    before = microscope.get(key, BeamType.ION)
    microscope.set(
        key, _new_value(microscope, key, BeamType.ELECTRON), BeamType.ELECTRON
    )
    assert microscope.get(key, BeamType.ION) == before


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------

CHOICE_KEYS = ["current", "voltage", "detector_type", "detector_mode"]


@pytest.mark.parametrize("key", CHOICE_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_current_value_is_one_of_the_choices(microscope, key, beam_type):
    choices = microscope.get_available_values(key, beam_type)
    assert isinstance(choices, list) and choices
    assert microscope.get(key, beam_type) in choices


@pytest.mark.parametrize("key", CHOICE_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_every_choice_can_be_set(microscope, key, beam_type):
    for choice in microscope.get_available_values(key, beam_type):
        microscope.set(key, choice, beam_type)
        assert microscope.get(key, beam_type) == choice


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
    """The beam keys' values come from the beam devices."""
    microscope = _connect("Demo", _plasma_configuration())
    for beam_type in BEAMS:
        for key in ("current", "voltage", "detector_type", "detector_mode"):
            assert microscope.get_available_values(key, beam_type)
    microscope.set("plasma_gas", "Argon", BeamType.ION)
    assert microscope.get("plasma_gas", BeamType.ION) == "Argon"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ion_currents_follow_the_plasma_gas(backend):
    from fibsem.microscopes.simulator import SIMULATOR_BEAM_CURRENTS

    currents = SIMULATOR_BEAM_CURRENTS[BeamType.ION]
    microscope = _connect(backend, _plasma_configuration())
    assert microscope.get("plasma_gas", BeamType.ION) == "Xenon"
    assert microscope.get_available_values("current", BeamType.ION) == currents["Xenon"]
    microscope.set("plasma_gas", "Argon", BeamType.ION)
    assert microscope.get_available_values("current", BeamType.ION) == currents["Argon"]
    microscope.set("plasma_gas", "Helium", BeamType.ION)  # not offered: ignored
    assert microscope.get("plasma_gas", BeamType.ION) == "Argon"


def test_demo_reads_its_configuration():
    """The configured keys and capabilities read the configuration."""
    microscope = _connect("Demo", _plasma_configuration())
    for key in ("plasma_gas", "scan_direction"):
        assert microscope.get_available_values(key)
    assert microscope._get_axis_limits()


def test_demo_sets_its_milling_recipe():
    microscope = _connect("Demo")
    files = microscope.get_available_values("application_file")
    microscope.set_milling_settings(
        FibsemMillingSettings(
            milling_channel=BeamType.ELECTRON,
            application_file=files[-1],
            patterning_mode="Parallel",
        )
    )
    milling = microscope.milling_system
    assert milling.default_application_file == files[-1]
    assert milling.patterning_mode == "Parallel"
    assert milling.default_beam_type is BeamType.ELECTRON


@pytest.mark.parametrize(
    "key", ["patterning_mode", "application_file", "default_patterning_beam_type"]
)
def test_the_milling_keys_are_gone(microscope, key, caplog):
    """A recipe sets them (`set_milling_settings`); as keys they are unknown."""
    with caplog.at_level("WARNING"):
        microscope.set(key, "Parallel")
    assert f"Unknown key: {key}" in caplog.text


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.set("active_view", BeamType.ION),
        lambda m: m.set("active_device", BeamType.ION),
        lambda m: m.get("plasma", BeamType.ION),
    ],
    ids=["active_view", "active_device", "plasma"],
)
def test_the_unused_keys_are_gone(call, caplog):
    """Nothing called them; the channel is `set_channel`, the column `system.ion`."""
    with caplog.at_level("WARNING"):
        call(_connect("Demo"))
    assert "Unknown key" in caplog.text


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


@pytest.mark.parametrize("key", ["plasma_gas", "application_file", "scan_direction"])
@pytest.mark.parametrize("beam_type", BEAMS)
def test_choice_only_keys_list_strings(microscope, key, beam_type):
    choices = microscope.get_available_values(key, beam_type)
    assert choices and all(isinstance(c, str) for c in choices)


@pytest.mark.parametrize("beam_type", BEAMS)
def test_unknown_choice_key_is_an_empty_list(microscope, beam_type):
    assert microscope.get_available_values("not_a_key", beam_type) == []


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def _readable_state(microscope) -> Dict[str, Any]:
    """Every key the contract reads, for detecting a write that should change nothing."""
    state = {key: microscope.get(key) for key in GLOBAL_READ_KEYS}
    for beam_type in BEAMS:
        for key in BEAM_KEYS:
            state[(key, beam_type.name)] = microscope.get(key, beam_type)
    return state


@pytest.mark.parametrize("beam_type", [None, *BEAMS])
def test_unknown_key_reads_none_and_warns(microscope, beam_type, caplog):
    """Pinned quirk: a typo in a key is a warning and a None, not an error."""
    with caplog.at_level(logging.WARNING):
        assert microscope.get("not_a_key", beam_type) is None
    assert any("not_a_key" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("beam_type", [None, *BEAMS])
def test_unknown_key_write_changes_nothing_and_warns(microscope, beam_type, caplog):
    before = _readable_state(microscope)
    with caplog.at_level(logging.WARNING):
        assert microscope.set("not_a_key", 1, beam_type) is None
    assert any("not_a_key" in r.getMessage() for r in caplog.records)
    assert _readable_state(microscope) == before


@pytest.mark.parametrize("beam_type", BEAMS)
def test_preset_is_a_silent_no_op(microscope, beam_type, caplog):
    """Pinned quirk: `preset` is accepted on Demo and does nothing, without a warning."""
    before = _readable_state(microscope)
    with caplog.at_level(logging.WARNING):
        assert microscope.get("preset", beam_type) is None
        microscope.set("preset", "anything", beam_type)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert _readable_state(microscope) == before


# ---------------------------------------------------------------------------
# Base-class wrappers over keys
# ---------------------------------------------------------------------------

# (getter, setter, key): the getter must answer what `get(key)` answers, and the
# setter must write `key` and return what it then reads.
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
    assert getattr(microscope, getter)(beam_type) == microscope.get(key, beam_type)
    value = _new_value(microscope, key, beam_type)
    returned = getattr(microscope, setter)(value, beam_type)
    assert returned == value
    assert microscope.get(key, beam_type) == value


@pytest.mark.parametrize("beam_type", BEAMS)
def test_on_off_and_blanking(microscope, beam_type):
    assert microscope.turn_off(beam_type) is False
    assert microscope.is_on(beam_type) is False
    assert microscope.turn_on(beam_type) is True
    assert microscope.is_on(beam_type) is True
    assert microscope.blank(beam_type) is True
    assert microscope.is_blanked(beam_type) is True
    assert microscope.unblank(beam_type) is False
    assert microscope.is_blanked(beam_type) is False


@pytest.mark.parametrize("beam_type", BEAMS)
def test_scanning_modes(microscope, beam_type):
    microscope.set_spot_scanning_mode(Point(0.5, 0.5), beam_type)
    assert microscope.get("scanning_mode", beam_type) == "spot"
    microscope.set_reduced_area_scanning_mode(
        FibsemRectangle(0.25, 0.25, 0.5, 0.5), beam_type
    )
    assert microscope.get("scanning_mode", beam_type) == "reduced_area"
    microscope.set_full_frame_scanning_mode(beam_type)
    assert microscope.get("scanning_mode", beam_type) == "full_frame"


@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_shift_adds_to_the_shift(microscope, beam_type):
    start = microscope.get("shift", beam_type)
    microscope.beam_shift(1e-6, -2e-6, beam_type)
    microscope.beam_shift(1e-6, -2e-6, beam_type)
    shift = microscope.get("shift", beam_type)
    assert (shift.x, shift.y) == pytest.approx((start.x + 2e-6, start.y - 4e-6))


@pytest.mark.parametrize("beam_type", BEAMS)
def test_scanning_mode_keys(microscope, beam_type):
    point = Point(0.25, 0.75)
    microscope.set("spot_mode", point, beam_type)
    assert microscope.get("scanning_mode", beam_type) == "spot"
    microscope.set("reduced_area", FibsemRectangle(0.25, 0.25, 0.5, 0.5), beam_type)
    assert microscope.get("scanning_mode", beam_type) == "reduced_area"
    microscope.set("full_frame", None, beam_type)
    assert microscope.get("scanning_mode", beam_type) == "full_frame"


@pytest.mark.parametrize("beam_type", BEAMS)
def test_the_spot_burn_reads_where_the_beam_is_parked(microscope, beam_type):
    """A spot burn marks the scene at the beam's spot, at its current field."""
    point = Point(0.25, 0.75)
    microscope.set("hfw", 100e-6, beam_type)
    microscope.set_spot_scanning_mode(point, beam_type)
    spot, beam = microscope._spot_and_beam(beam_type)
    assert spot == point
    assert beam.hfw == microscope.get("hfw", beam_type)
    assert beam.resolution == microscope.get("resolution", beam_type)


def test_vent_and_pump(microscope):
    assert microscope.vent() == "Vented"
    assert microscope.get("chamber_state") == "Vented"
    assert microscope.pump() == "Pumped"
    assert microscope.get("chamber_state") == "Pumped"


@pytest.mark.parametrize("key", ["pump_chamber", "vent_chamber"])
def test_a_false_pump_or_vent_does_nothing(microscope, key):
    """Pinned quirk: the old keys pump or vent only for a true value."""
    # start in the state the key would leave, so a pump or vent would show
    microscope.vent() if key == "pump_chamber" else microscope.pump()
    before = microscope.get("chamber_state")
    microscope.set(key, False)
    assert microscope.get("chamber_state") == before


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.get("plasma_gas", BeamType.ION),
        lambda m: m.set("plasma_gas", "Argon", BeamType.ION),
        lambda m: m.set("pump_chamber", False),
        lambda m: m.set("vent_chamber", False),
    ],
    ids=["read-gas", "set-gas", "pump-false", "vent-false"],
)
def test_keys_without_an_effect_here_are_not_unknown(microscope, call, caplog):
    """No plasma gas on a Ga column, and a false pump or vent: known keys that do
    nothing here, so they are not reported as unknown."""
    assert not microscope.system.ion.plasma
    with caplog.at_level(logging.WARNING):
        call(microscope)
    assert not [r for r in caplog.records if "Unknown key" in r.getMessage()]


def test_home_homes_and_says_so(microscope):
    """`home` returns whether the stage is homed afterwards, on every backend."""
    assert microscope.home() is True
    assert microscope.get("stage_homed") is True


def test_link_stage(microscope):
    assert microscope.link_stage() is True


def test_stage_link_links_whatever_it_is_given(microscope):
    """Pinned quirk: `set("stage_link", False)` links the stage on Demo."""
    microscope.set("stage_link", False)
    assert microscope.get("stage_linked") is True


@pytest.mark.parametrize("beam_type", BEAMS)
def test_beam_system_settings_match_their_keys(microscope, beam_type):
    settings = microscope.get_beam_system_settings(beam_type)
    assert settings.beam_type is beam_type
    assert settings.enabled == microscope.get("beam_enabled", beam_type)
    assert settings.eucentric_height == microscope.get("eucentric_height", beam_type)
    assert settings.column_tilt == microscope.get("column_tilt", beam_type)
    assert settings.plasma_gas == microscope.get("plasma_gas", beam_type)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _xyzrt(position: FibsemStagePosition) -> List[float]:
    return [position.x, position.y, position.z, position.r, position.t]


def test_get_stage_position_matches_the_key(microscope):
    assert _xyzrt(microscope.get_stage_position()) == _xyzrt(
        microscope.get("stage_position")
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


def test_manipulator_inserts_and_retracts(microscope):
    inserted = microscope.insert_manipulator("PARK")
    assert isinstance(inserted, FibsemManipulatorPosition)
    assert _xyzrt(inserted) == _xyzrt(microscope.get_manipulator_position())
    assert microscope.get_manipulator_state() is True
    retracted = microscope.retract_manipulator()
    assert isinstance(retracted, FibsemManipulatorPosition)
    assert _xyzrt(retracted) == _xyzrt(microscope.get_manipulator_position())
    assert microscope.get_manipulator_state() is False


def test_manipulator_moves_absolute_and_relative(microscope):
    target = FibsemManipulatorPosition(x=1e-6, y=2e-6, z=3e-6)
    moved = microscope.move_manipulator_absolute(deepcopy(target))
    assert np.allclose(_xyzrt(moved), _xyzrt(target))
    moved = microscope.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6))
    assert np.allclose(_xyzrt(moved), [2e-6, 2e-6, 3e-6, 0, 0])
    assert _xyzrt(moved) == _xyzrt(microscope.get_manipulator_position())


@pytest.mark.parametrize("beam_type", BEAMS)
def test_manipulator_corrected_move_returns_where_it_is(microscope, beam_type):
    moved = microscope.move_manipulator_corrected(1e-6, -1e-6, beam_type)
    assert isinstance(moved, FibsemManipulatorPosition)
    assert _xyzrt(moved) == _xyzrt(microscope.get_manipulator_position())


def test_manipulator_moves_to_an_offset_from_a_saved_position(microscope):
    eucentric = microscope._get_saved_manipulator_position("EUCENTRIC")
    offset = FibsemManipulatorPosition(x=1e-6, z=-2e-6)
    moved = microscope.move_manipulator_to_position_offset(
        deepcopy(offset), "EUCENTRIC"
    )
    assert np.allclose(_xyzrt(moved), _xyzrt(eucentric + offset))


def test_unknown_saved_manipulator_position_raises(microscope):
    with pytest.raises(ValueError):
        microscope._get_saved_manipulator_position("NOWHERE")


# ---------------------------------------------------------------------------
# Fluorescence (on a configuration with an FM)
# ---------------------------------------------------------------------------

FM_CONFIGURATION = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


@pytest.fixture(params=BACKENDS)
def fm_microscope(request):
    return _connect(request.param, FM_CONFIGURATION)


def test_a_compustage_is_never_linked_and_cannot_link(fm_microscope, caplog):
    """The Arctis simulator is a compustage: its stage has no link, so
    ``stage_linked`` is unsupported (None) and ``stage_link`` does nothing, without
    either being an unknown key."""
    assert fm_microscope.stage_is_compustage
    with caplog.at_level(logging.WARNING):
        fm_microscope.set("stage_link", True)
        assert fm_microscope.get("stage_linked") is None
    assert not [r for r in caplog.records if "Unknown key" in r.getMessage()]


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
    assert objective.state == "Inserted"
    objective.retract()
    assert objective.state == "Retracted"


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
    assert fm.objective.state == "Inserted"
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
    from fibsem.devices.drivers.demo import DemoCamera
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


def test_demo_fm_names_a_numeric_emission_as_its_multi_band_filter():
    """The simulator's filter set has no bands, so a wavelength means its multi-band
    filter, as Thermo reports one."""
    microscope = _connect("Demo", FM_CONFIGURATION)
    microscope.fm.filter_set.emission_wavelength = 520.0
    assert microscope.fm.filter_set.emission_wavelength == "Fluorescence"
    microscope.fm.filter_set.emission_wavelength = None
    assert microscope.fm.filter_set.emission_wavelength is None


def test_fm_image_metadata_is_the_state_it_was_taken_in(fm_microscope):
    """The devices report their state with the frame, and the image carries it."""
    from fibsem.fm.structures import ChannelSettings

    fm = fm_microscope.fm
    channel = ChannelSettings(
        name="GFP", excitation_wavelength=450, power=0.3, exposure_time=0.02
    )
    md = fm.acquire_image(channel).metadata
    now = fm.get_metadata()
    assert md.channels == now.channels
    assert (md.pixel_size_x, md.pixel_size_y) == (now.pixel_size_x, now.pixel_size_y)
    assert md.resolution == now.resolution
    assert md.stage_position == now.stage_position


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
    for beam_type in BEAMS:
        for key in ("working_distance", "hfw", "dwell_time", "detector_contrast"):
            microscope.set(key, _new_value(microscope, key, beam_type), beam_type)
    assert _readable_state(microscope) != before

    microscope.set_microscope_state(state)
    after = _readable_state(microscope)
    for key in ("working_distance", "hfw", "dwell_time", "detector_contrast"):
        for beam_type in BEAMS:
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
    assert np.isclose(microscope.get("hfw", beam_type), 100e-6)


@pytest.mark.parametrize("beam_type", BEAMS)
def test_finish_milling_puts_the_beam_back(microscope, beam_type):
    """What the first setup_milling found comes back, with an imaging current
    given still winning; a second finish has nothing left to put back."""
    from fibsem.structures import FibsemMillingSettings

    def conditions():
        return [microscope.get(key, beam_type) for key in ("voltage", "current", "hfw")]

    before = conditions()
    for current, hfw in ((7.6e-9, 80e-6), (2e-9, 50e-6)):
        microscope.setup_milling(
            FibsemMillingSettings(
                milling_channel=beam_type, milling_current=current, hfw=hfw
            )
        )
    assert conditions() != before
    microscope.finish_milling()
    assert np.allclose(conditions(), before)

    microscope.setup_milling(FibsemMillingSettings(milling_channel=beam_type))
    microscope.finish_milling(imaging_current=3e-10)
    assert np.allclose(conditions(), [before[0], 3e-10, before[2]])
    microscope.set("hfw", 40e-6, beam_type)
    microscope.finish_milling()
    assert microscope.get("hfw", beam_type) == 40e-6


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
        microscope.manipulator_device,
        *microscope.fm_devices.values(),
    ]
    assert microscope.fm_devices
    assert all(device.resources is microscope.resources for device in devices)
