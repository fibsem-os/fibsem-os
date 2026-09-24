"""What the instrument *is*, and what a session *starts from*, are different claims.

The model already drew this line and the file hid it. `BeamSystemSettings.from_dict`
hands the same flat `electron:` block to `BeamSettings`, to `FibsemDetectorSettings`
and to its own `column_tilt`/`eucentric_height` -- three objects built from one block,
with nothing in the file saying which key feeds which.

That is how `electron.hfw` and `imaging.hfw` came to look alike: one is pushed to the
column on Apply, the other only seeds the acquire tab, both were tagged `[USER]`, and
they diverge silently. A `defaults:` block names the difference.

The split is at the file boundary only. The records are unchanged, so the readers of
`system.electron.beam` and `system.ion.detector` do not move.
"""

import copy
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemStagePosition,
    MicroscopeSettings,
    SystemSettings,
)
from tests.test_configuration_schema import SHIPPED

HARDWARE_KEYS = set(SystemSettings.HARDWARE_BEAM_KEYS)
DEFAULT_KEYS = {
    "voltage",
    "current",
    "resolution",
    "hfw",
    "dwell_time",
    "detector_mode",
    "detector_type",
}


def _load(filename: str) -> dict:
    return utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))


# ---------------------------------------------------------------------------
# The shipped files say the same thing they did before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", SHIPPED)
def test_the_split_changed_no_configuration_s_meaning(filename: str):
    """Flatten each shipped file back to its pre-split shape and load both.

    A key that moved to the wrong side, or was dropped on the way, shows up here as a
    difference, and nothing else in this change would catch it. Reconstructed rather
    than read out of git, so the test still means something after this lands.
    """
    split = _load(filename)
    flat = copy.deepcopy(split)
    flat.update(flat.pop("hardware"))
    calibration = flat.pop("calibration")
    flat["stage"].update(calibration)
    defaults = flat.pop("defaults")
    for block in ("electron", "ion"):
        flat[block] = {**(flat.get(block) or {}), **(defaults.get(block) or {})}
    flat["imaging"] = defaults["imaging"]

    old = MicroscopeSettings.from_dict(flat)
    new = MicroscopeSettings.from_dict(split)

    assert new.system == old.system
    assert new.image == old.image


@pytest.mark.parametrize("filename", SHIPPED)
def test_the_column_blocks_hold_only_hardware(filename: str):
    """`electron:` and `ion:` describe what the column is. Nothing in them should be a
    thing an operator picks for a session."""
    config = _load(filename)
    for block in ("electron", "ion"):
        keys = set(config["hardware"][block])
        assert not (keys & DEFAULT_KEYS), f"{block} still states session state"
        assert keys <= HARDWARE_KEYS, (
            f"{block} has an unexpected key: {keys - HARDWARE_KEYS}"
        )


@pytest.mark.parametrize("filename", SHIPPED)
def test_the_defaults_block_holds_the_session_state(filename: str):
    config = _load(filename)
    defaults = config["defaults"]

    assert set(defaults["electron"]) == DEFAULT_KEYS
    assert set(defaults["ion"]) == DEFAULT_KEYS
    assert "beam_type" in defaults["imaging"]
    assert defaults["apply_on_connect"] is False
    assert "imaging" not in config, "the top-level imaging block should have moved"


# ---------------------------------------------------------------------------
# Old files keep loading
# ---------------------------------------------------------------------------


def test_a_file_written_before_the_split_still_loads():
    """Every configuration in the field is flat. The merge is a no-op for them, which
    is the whole reason it is a merge and not a move."""
    flat = {
        "electron": {
            "enabled": True,
            "column_tilt": 0,
            "eucentric_height": 7.0e-3,
            "voltage": 2000,
            "current": 50.0e-12,
            "hfw": 150.0e-6,
            "resolution": [1536, 1024],
            "dwell_time": 1.0e-6,
        },
        "imaging": {"beam_type": "ION", "hfw": 80.0e-6},
    }

    settings = MicroscopeSettings.from_dict(flat)

    assert settings.system.electron.beam.voltage == 2000
    assert settings.system.electron.column_tilt == 0
    assert settings.image.hfw == 80.0e-6
    assert settings.image.beam_type is BeamType.ION


def test_the_old_spellings_are_not_reported_as_unrecognised():
    """Read for migration, so a file in the field is not told its settings are
    unsupported -- but a typo under the old block still is."""
    flat = {
        "electron": {"voltage": 2000, "hfw": 150.0e-6},
        "imaging": {"beam_type": "ION", "hfw": 80.0e-6, "nonsense": 1},
    }
    assert utils.unrecognised_configuration_keys(flat) == ["imaging.nonsense"]


def test_defaults_win_over_the_flat_block():
    """A file may carry both -- hand-edited, or partially migrated. The more specific
    location wins, and this pins which that is."""
    settings = SystemSettings.from_dict(
        {"electron": {"voltage": 1000}, "defaults": {"electron": {"voltage": 5000}}}
    )
    assert settings.electron.beam.voltage == 5000


def test_a_typo_inside_the_defaults_block_is_reported():
    unknown = utils.unrecognised_configuration_keys(
        {"defaults": {"electron": {"voltage": 1, "volts": 2}, "imagin": {}}}
    )
    assert unknown == ["defaults.electron.volts", "defaults.imagin"]


# ---------------------------------------------------------------------------
# The round trip writes the split shape
# ---------------------------------------------------------------------------


def test_writing_a_configuration_produces_the_split_shape():
    settings = MicroscopeSettings.from_dict(_load("microscope-configuration.yaml"))

    written = settings.to_dict()

    assert set(written["defaults"]) == {
        "apply_on_connect",
        "electron",
        "ion",
        "imaging",
    }
    assert "imaging" not in written
    assert set(written["hardware"]["electron"]) <= HARDWARE_KEYS
    assert written["defaults"]["electron"]["voltage"] == 2000


# ---------------------------------------------------------------------------
# A default that is not stated is not pushed
# ---------------------------------------------------------------------------


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def test_a_default_that_is_not_stated_is_not_pushed(microscope):
    """A `defaults:` block that states only the voltage is a configuration.

    The readers default everything else to None, and before this a None reached
    `set_beam_voltage`, `set_field_of_view` and the rest as a value. Previously a
    file like this raised `KeyError` at load, which was at least loud.
    """
    before = microscope.get_beam_settings(BeamType.ELECTRON)
    sparse = BeamSettings(beam_type=BeamType.ELECTRON, voltage=before.voltage)
    assert sparse.hfw is None and sparse.resolution is None

    microscope.set_beam_settings(sparse)

    after = microscope.get_beam_settings(BeamType.ELECTRON)
    assert after.hfw == before.hfw
    assert after.resolution == before.resolution
    assert after.dwell_time == before.dwell_time


def test_apply_on_connect_is_read_and_written_and_does_nothing(monkeypatch):
    """Groundwork, disabled. The key is in the structure and the file so a later
    change is one `if`; until then connecting with it set pushes nothing."""
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config["defaults"]["apply_on_connect"] = True
    settings = MicroscopeSettings.from_dict(config)
    assert settings.system.apply_defaults_on_connect is True
    assert settings.to_dict()["defaults"]["apply_on_connect"] is True

    from fibsem.microscope import FibsemMicroscope

    def refuse(self, *args, **kwargs):
        raise AssertionError("apply_configuration ran at connect")

    monkeypatch.setattr(FibsemMicroscope, "apply_configuration", refuse)
    monkeypatch.setattr(FibsemMicroscope, "set_beam_system_settings", refuse)
    microscope, _ = utils.setup_session(manufacturer="Demo")
    try:
        assert microscope.system.apply_defaults_on_connect is False
    finally:
        microscope.disconnect()


# ---------------------------------------------------------------------------
# Remember current state as the defaults
# ---------------------------------------------------------------------------


def test_capturing_the_defaults_reads_the_instrument(microscope):
    """The gesture the block exists for: "it is set up how I like it, remember this"."""
    live = microscope.get_microscope_state().electron_beam.voltage
    microscope.system.electron.beam.voltage = (live or 0) + 4321  # a stale value

    microscope.capture_defaults()

    assert microscope.system.electron.beam.voltage == live


def test_capturing_does_not_record_the_stage_position(microscope):
    """`get_microscope_state` returns the stage position alongside the beams, and it
    has no business in the defaults.

    Applying one would move the stage, and where the stage happened to be sitting when
    somebody pressed Save is session state, not a preference.

    Asserted against the whole stage record rather than against the serialised output.
    `SystemSettings.to_dict` has nowhere to put a stage position, so a test that only
    reads the written file passes whatever `capture_defaults` does.
    """
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1.0e-3, y=-2.0e-3, r=0.5, t=0.1)
    )
    before = microscope.system.stage.to_dict()

    microscope.capture_defaults()

    assert microscope.system.stage.to_dict() == before
    written = microscope.system.to_dict()
    assert "stage" not in written["defaults"]


def test_capturing_snapshots_rather_than_aliasing(microscope):
    """The captured record must not be the live one, or the next beam change would
    silently rewrite the saved defaults."""
    microscope.capture_defaults()
    captured = microscope.system.electron.beam.voltage

    live = microscope.get_microscope_state().electron_beam
    live.voltage = (captured or 0) + 1234

    assert microscope.system.electron.beam.voltage == captured


def test_capturing_takes_only_the_defaults(microscope):
    """Not the beam shift, stigmation, scan rotation or working distance.

    Those are alignment state. Captured, they would land in the file on Save and
    be pushed back by Apply -- a working distance from one sample refocusing the
    column on the next. The first version copied the whole record and moved the
    simulated working distance from 7 mm to 4 mm on capture.
    """
    before = copy.deepcopy(microscope.system.electron.beam)
    live = microscope.get_microscope_state().electron_beam
    assert live.working_distance != before.working_distance, "fixture is blind"

    microscope.capture_defaults()

    after = microscope.system.electron.beam
    assert after.working_distance == before.working_distance
    assert after.shift == before.shift
    assert after.stigmation == before.stigmation
    assert after.scan_rotation == before.scan_rotation
    assert after.voltage == live.voltage


def test_capturing_leaves_the_hardware_description_alone(microscope):
    """Save reads the beams; it must not touch what the column *is*."""
    before_tilt = microscope.system.ion.column_tilt
    before_eucentric = microscope.system.electron.eucentric_height

    microscope.capture_defaults()

    assert microscope.system.ion.column_tilt == before_tilt
    assert microscope.system.electron.eucentric_height == before_eucentric


def test_the_detector_keys_in_every_shipped_file_are_read():
    """A defect older than the schema work, fixed here because the Defaults panel
    made it destructive.

    Every shipped configuration states `detector_type: ETD` and
    `detector_mode: SecondaryElectrons`. `BeamSystemSettings.to_dict` writes those
    names, but `FibsemDetectorSettings.from_dict` read `type` and `mode`, so nothing
    ever read them back: the detector loaded as "Unknown" on every system, and a
    file saved from the application lost its detector on the next load. With a
    panel that reads the record and writes it back, "Unknown" would have replaced
    every site's ETD on the first Save.

    What changes on hardware: Apply now pushes the detector the file names rather
    than "Unknown". That is what the file always intended.
    """
    config = _load("microscope-configuration.yaml")
    defaults = config["defaults"]["electron"]
    assert defaults["detector_type"] == "ETD"
    assert defaults["detector_mode"] == "SecondaryElectrons"

    settings = MicroscopeSettings.from_dict(config)

    assert settings.system.electron.detector.type == "ETD"
    assert settings.system.electron.detector.mode == "SecondaryElectrons"


def test_the_detector_round_trips_through_a_saved_file():
    """Brightness and contrast too: written as `detector_brightness`, they were read
    as `brightness` and came back as the default."""
    settings = MicroscopeSettings.from_dict(_load("microscope-configuration.yaml"))
    settings.system.ion.detector.brightness = 0.37
    settings.system.ion.detector.contrast = 0.62

    reloaded = MicroscopeSettings.from_dict(copy.deepcopy(settings.to_dict()))

    assert reloaded.system.ion.detector == settings.system.ion.detector
