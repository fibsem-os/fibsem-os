import numpy as np
import pytest

from fibsem import utils
from fibsem.structures import BeamType, FibsemStagePosition


def test_microscope():
    """Test get/set microscope functions."""

    microscope, settings = utils.setup_session(manufacturer="Demo")

    hfw = 150e-6
    microscope.set_field_of_view(hfw, BeamType.ELECTRON)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == hfw

    beam_current = 1e-9
    microscope.set_beam_current(beam_current, BeamType.ION)
    assert microscope.get_beam_current(BeamType.ION) == beam_current


@pytest.mark.parametrize("tilt_deg,expected", [
    (0,    "SEM"),      # exact SEM position
    (-23,  "MILLING"),  # standard milling tilt (below SEM)
    (-10,  "MILLING"),  # arbitrary tilt between SEM and -45°
    (15,   "MILLING"),  # above SEM tilt
    (-128, "FIB"),      # compustage FIB position
    (-50,  "NONE"),     # below -45° floor
])
def test_get_stage_orientation_compustage(tilt_deg, expected):
    """Test get_stage_orientation for compustage (pretilt=0)."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()

    pos = FibsemStagePosition(r=np.radians(0), t=np.radians(tilt_deg))
    assert microscope.get_stage_orientation(pos) == expected


@pytest.mark.parametrize("rotation_deg,tilt_deg,expected", [
    (0,   35,  "SEM"),      # exact SEM position
    (0,   12,  "MILLING"),  # standard milling tilt (below SEM)
    (0,   20,  "MILLING"),  # arbitrary tilt between MILLING and SEM
    (0,   42,  "MILLING"),  # above SEM tilt
    (180, 17,  "FIB"),      # FIB position (rotation_180=180, col_tilt-pretilt=17)
    (0,   -50, "NONE"),     # below -45° floor
])
def test_get_stage_orientation_non_compustage(rotation_deg, tilt_deg, expected):
    """Test get_stage_orientation for non-compustage (pretilt=35°)."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope.system.stage.shuttle_pre_tilt = 35
    microscope.system.stage.rotation_180 = 180
    microscope._update_orientations()

    pos = FibsemStagePosition(r=np.radians(rotation_deg), t=np.radians(tilt_deg))
    assert microscope.get_stage_orientation(pos) == expected


@pytest.mark.parametrize("orientation", ["SEM", "FIB", "MILLING"])
def test_move_to_orientation(orientation):
    """Test that move_to_orientation moves the stage to the correct r, t."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    expected = microscope.get_orientation(orientation)
    microscope.move_to_orientation(orientation)
    pos = microscope.get_stage_position()
    assert np.isclose(pos.r, expected.r, atol=1e-6), f"{orientation}: r mismatch"
    assert np.isclose(pos.t, expected.t, atol=1e-6), f"{orientation}: t mismatch"


def test_move_to_orientation_invalid():
    """Test that an invalid orientation name raises ValueError."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    with pytest.raises(ValueError):
        microscope.move_to_orientation("INVALID")


@pytest.mark.parametrize("orientation", ["SEM", "FIB", "MILLING"])
def test_move_to_orientation_round_trip(orientation):
    """Test that move_to_orientation and get_stage_orientation are consistent."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.move_to_orientation(orientation)
    assert microscope.get_stage_orientation() == orientation