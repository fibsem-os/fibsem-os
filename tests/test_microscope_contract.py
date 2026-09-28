"""Contract tests for the FibsemMicroscope API, run against every simulated backend.

The FluorescenceMicroscope has its contract written down (`tests/fm/test_fm_contract.py`);
the microscope did not, so what a backend promises was spread across every test that
happens to use Demo. This file writes it down, through the public API only: `get`/`set`
keys, their base-class wrappers, stage moves, state, and image acquisition.

It holds `DeviceDemoMicroscope`, the demo backend being rebuilt from devices, to
exactly what `DemoMicroscope` does today: every test runs against both, and the
differential test at the bottom compares them call by call.

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
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

# The backends this suite runs against. Each is built the way a session builds it,
# through `utils.setup_session`, so it is chosen exactly as a configuration file
# would choose it. "DeviceDemo" is the Demo backend rebuilt from devices
# (`fibsem.microscopes.device_demo`), selected by `sim: {devices: true}`.
BACKENDS = ["Demo", "DeviceDemo"]

# The backend every other one is compared against in the differential test.
REFERENCE_BACKEND = "Demo"

BEAMS = [BeamType.ELECTRON, BeamType.ION]


def _device_demo_configuration() -> str:
    """The default configuration with the device-built Demo selected."""
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        configuration = yaml.safe_load(f)
    configuration.setdefault("sim", {})
    configuration["sim"] = {**(configuration["sim"] or {}), "devices": True}
    path = os.path.join(tempfile.mkdtemp(), "device-demo-configuration.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(configuration, f)
    return path


def _connect(backend: str):
    config_path = _device_demo_configuration() if backend == "DeviceDemo" else None
    microscope, _ = utils.setup_session(
        config_path=config_path, manufacturer="Demo", setup_logging=False
    )
    return microscope


def test_backends_are_what_they_say():
    from fibsem.microscopes.device_demo import DeviceDemoMicroscope
    from fibsem.microscopes.simulator import DemoMicroscope

    assert type(_connect("Demo")) is DemoMicroscope
    assert type(_connect("DeviceDemo")) is DeviceDemoMicroscope


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


# The keys DeviceDemo serves from its devices rather than the Demo chain. The
# contract above runs through them; this checks that it really does.
DEVICE_DEMO_ROUTED_BEAM_KEYS = [
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


@pytest.mark.parametrize("key", DEVICE_DEMO_ROUTED_BEAM_KEYS)
@pytest.mark.parametrize("beam_type", BEAMS)
def test_device_demo_serves_its_beam_keys_from_devices(key, beam_type):
    microscope = _connect("DeviceDemo")
    param = microscope._route(key, beam_type)
    assert param is not None
    assert param.device is microscope.beams[beam_type]


def test_device_demo_moves_its_stage_device():
    """DeviceDemo's stage methods go through its stage device, not the Demo chain."""
    microscope = _connect("DeviceDemo")
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
    "plasma": lambda v: isinstance(v, bool),
}


@pytest.mark.parametrize("key", sorted(GLOBAL_READ_KEYS))
def test_global_key_reads(microscope, key):
    assert GLOBAL_READ_KEYS[key](microscope.get(key))


@pytest.mark.parametrize("key", sorted(GLOBAL_READ_KEYS))
@pytest.mark.parametrize("beam_type", BEAMS)
def test_global_key_ignores_beam_type(microscope, key, beam_type):
    """A key with no beam gives the same answer whatever beam type is passed."""
    assert microscope.get(key, beam_type) == microscope.get(key)


def test_manufacturer_is_the_configured_one(microscope):
    assert microscope.get("manufacturer") == microscope.system.info.manufacturer


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


@pytest.mark.parametrize(
    "key", ["plasma_gas", "application_file", "gis_ports", "scan_direction"]
)
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
    microscope.set_full_frame_scanning_mode(beam_type)
    assert microscope.get("scanning_mode", beam_type) == "full_frame"


def test_vent_and_pump(microscope):
    assert microscope.vent() == "Vented"
    assert microscope.get("chamber_state") == "Vented"
    assert microscope.pump() == "Pumped"
    assert microscope.get("chamber_state") == "Pumped"


def test_home_homes_and_returns_none(microscope):
    """Pinned quirk: Demo overrides `home` and returns None, where the base class
    returns `get("stage_homed")`.
    """
    assert microscope.home() is None
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


# ---------------------------------------------------------------------------
# Differential check
# ---------------------------------------------------------------------------

# A fixed call sequence. Each entry is (method, args, kwargs); every call is
# followed by a snapshot of every key the contract reads. Two backends that agree
# on every snapshot agree on everything this sequence exercises, including the
# interactions between calls that the per-key tests above do not reach.
CALL_SEQUENCE = [
    ("set", ("working_distance", 6e-3, BeamType.ELECTRON), {}),
    ("set", ("hfw", 250e-6, BeamType.ION), {}),
    ("set_beam_current", (2e-10, BeamType.ION), {}),
    ("set_beam_voltage", (5000, BeamType.ELECTRON), {}),
    ("set_resolution", ((3072, 2048), BeamType.ELECTRON), {}),
    ("set_detector_type", ("TLD", BeamType.ELECTRON), {}),
    ("set_detector_mode", ("BackscatteredElectrons", BeamType.ION), {}),
    ("set_beam_shift", (Point(1e-6, 1e-6), BeamType.ION), {}),
    ("blank", (BeamType.ION,), {}),
    ("unblank", (BeamType.ION,), {}),
    ("turn_off", (BeamType.ELECTRON,), {}),
    ("turn_on", (BeamType.ELECTRON,), {}),
    ("move_stage_absolute", (FibsemStagePosition(x=1e-3, y=1e-3, z=2e-3),), {}),
    ("move_stage_relative", (FibsemStagePosition(x=5e-4, t=0.1),), {}),
    (
        "safe_absolute_stage_movement",
        (FibsemStagePosition(x=0, y=0, z=1e-3, r=np.radians(180), t=0.2),),
        {},
    ),
    ("vent", (), {}),
    ("pump", (), {}),
    ("set_spot_scanning_mode", (Point(0.25, 0.75), BeamType.ION), {}),
    ("set_full_frame_scanning_mode", (BeamType.ION,), {}),
    ("set", ("not_a_key", 1), {}),
]


def _snapshot(microscope) -> Dict[str, Any]:
    state = _readable_state(microscope)
    # Positions compare by value; the objects themselves carry names and identities.
    for key in ("stage_position", "manipulator_position"):
        state[key] = [round(float(v), 12) for v in _xyzrt(state[key])]
    for key, value in list(state.items()):
        if isinstance(value, Point):
            state[key] = (value.x, value.y)
        elif isinstance(value, list) and key not in (
            "stage_position",
            "manipulator_position",
        ):
            state[key] = tuple(value)
    return state


def _record(microscope) -> List[Tuple[str, Any, Dict[str, Any]]]:
    record = [("start", None, _snapshot(microscope))]
    for method, args, kwargs in CALL_SEQUENCE:
        returned = getattr(microscope, method)(*deepcopy(args), **deepcopy(kwargs))
        if isinstance(returned, FibsemStagePosition):
            returned = [round(float(v), 12) for v in _xyzrt(returned)]
        elif isinstance(returned, Point):
            returned = (returned.x, returned.y)
        record.append((method, returned, _snapshot(microscope)))
    return record


def _first_difference(a, b) -> str:
    for (method, ret_a, snap_a), (_, ret_b, snap_b) in zip(a, b):
        if ret_a != ret_b:
            return f"{method} returned {ret_a!r} vs {ret_b!r}"
        for key in snap_a:
            if snap_a[key] != snap_b.get(key):
                return (
                    f"after {method}: {key} is {snap_a[key]!r} vs {snap_b.get(key)!r}"
                )
    return ""


def test_call_sequence_is_deterministic():
    """The differential check only means something if one backend agrees with itself."""
    a = _record(_connect(REFERENCE_BACKEND))
    b = _record(_connect(REFERENCE_BACKEND))
    assert not _first_difference(a, b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_sequence_matches_the_reference(backend):
    reference = _record(_connect(REFERENCE_BACKEND))
    candidate = _record(_connect(backend))
    difference = _first_difference(reference, candidate)
    assert not difference, difference
