"""What a refusal says, now that it can say something true (FIB-858).

Every refusal string in the FM tree used to be phrased as if the system had no
fluorescence microscope, or as if the stage were merely "not in a valid orientation"
-- a false generality when it is 48.8 mm from the instrument. The message helper
speaks the two axes' vocabulary: the stage travels between *devices* and is re-posed
between *orientations*, and the failing term names the verb.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import DeviceImagingState

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


# ── the failing term names the verb ──────────────────────────────────


def test_a_wrong_place_says_travel():
    microscope = _microscope()
    microscope.move_to_orientation("FIB")

    message = microscope.describe_device_imaging_state("FM")

    assert "travel" in message
    assert "move_to_device('FM')" in message
    assert "re-pose" not in message.lower()


def test_a_wrong_pose_says_re_pose():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_orientation("SEM")

    message = microscope.describe_device_imaging_state("FM")

    assert "Re-pose" in message
    assert "held in the SEM orientation" in message
    assert "travel there" not in message


def test_wrong_on_both_axes_says_both_in_the_bracketing_order():
    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    message = microscope.describe_device_imaging_state("FM")

    assert message.index("Re-pose") < message.index("travel")


def test_no_device_is_terminal_and_says_so():
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)

    message = microscope.describe_device_imaging_state("FM")

    assert message == "This system has no fluorescence microscope."


def test_an_unrecognised_pose_is_not_called_the_none_orientation():
    """`get_stage_orientation` returns "NONE" for a pose it cannot name, and "held in
    the NONE orientation" is not a thing to say to anyone."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    position = microscope.get_stage_position()
    position.t = position.t + 0.5  # ~29 degrees: nothing classifies this
    microscope.move_stage_absolute(position)

    message = microscope.describe_device_imaging_state("FM")

    assert "NONE" not in message
    assert "unrecognised" in message


def test_a_passed_state_is_the_one_described():
    """A gate asks once and describes what it asked -- the message and the decision
    it explains must be about the same moment, even if the stage has since moved."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")

    message = microscope.describe_device_imaging_state(
        "FM", DeviceImagingState.NEEDS_REPOSE
    )

    assert "Re-pose" in message


# ── the lifted two-question guard ────────────────────────────────────


def test_refusal_to_start_asks_when_before_where():
    """A busy instrument is the answer wherever the stage is."""
    microscope = _microscope()
    microscope.move_to_orientation("SEM")  # the where-answer would also refuse
    microscope.fm.set_acquiring(True, "overview acquisition")
    try:
        refusal = microscope.fm.refusal_to_start("a z-stack")
    finally:
        microscope.fm.set_acquiring(False)

    assert refusal is not None
    assert "in use" in refusal
    assert "overview acquisition" in refusal


def test_refusal_to_start_then_asks_where():
    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    refusal = microscope.fm.refusal_to_start("a z-stack")

    assert refusal is not None
    assert refusal.startswith("Cannot start a z-stack.")
    assert "Re-pose" in refusal and "travel" in refusal


def test_nothing_to_refuse_is_none():
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_to_device("FM")

    assert microscope.fm.refusal_to_start("a z-stack") is None


def test_a_compustage_at_a_beam_pose_is_not_refused():
    """The allowance, surviving the message layer: acquiring in place is permitted
    from any pose there, so no refusal exists to phrase."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_orientation("MILLING")

    assert microscope.fm.refusal_to_start("an image") is None


# ── the raised messages carry the same sentences ─────────────────────


def test_the_tiled_runner_raises_the_route():
    from fibsem.fm.acquisition import acquire_tileset
    from fibsem.fm.structures import (
        AutoFocusMode,
        ChannelSettings,
        OverviewParameters,
    )

    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    with pytest.raises(ValueError, match="Re-pose at the beams and travel out"):
        acquire_tileset(
            microscope=microscope,
            channel_settings=ChannelSettings(
                name="DAPI",
                excitation_wavelength=358,
                emission_wavelength=461,
                power=0.1,
                exposure_time=0.001,
            ),
            overview_parameters=OverviewParameters(
                rows=1, cols=1, overlap=0.1, autofocus_mode=AutoFocusMode.NONE
            ),
        )
