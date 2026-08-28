"""Rotating the stage while it is parked at the fluorescence microscope.

The objective is inserted over the sample there, and the rotation is compucentric
about a centre back at the beams -- some 48.8 mm away -- so a half turn swings the
sample most of the width of the chamber, underneath the objective.

`get_target_position` already computes targets that keep the re-pose at the beams: the
device legs bracket the orientation leg. But it computes a target, it does not order
the moves, and `safe_absolute_stage_movement` rotates the stage *where it stands* --
the right order leaving the beams and the wrong one coming back. This file is the
refusal that closes that.

See FIB-841.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _parked_at_the_fm():
    """An offset system at the FM. `move_to_microscope` accepts only FIB (FIB-832)."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_to_microscope("FM")
    assert microscope.get_current_device() == "FM"
    return microscope


# ── the refusal ──────────────────────────────────────────────────────


def test_rotating_at_the_fm_is_refused():
    """MILLING is the case that matters: it is a half turn from where the FM sits.

    Measured on the offset simulator -- FM at r = 180, MILLING at r = 0 -- so this is
    not an exotic request, it is the pose FM-MILLING wants (FIB-833).
    """
    microscope = _parked_at_the_fm()

    with pytest.raises(ValueError, match="Cannot rotate the stage while it is at the"):
        microscope.move_to_orientation("MILLING")


def test_the_refusal_happens_before_anything_moves():
    """A guard that fires after the first leg would leave the stage mid-manoeuvre.

    `safe_absolute_stage_movement` tilts flat and then rotates, both real moves, so
    the check has to come ahead of them rather than anywhere inside.
    """
    microscope = _parked_at_the_fm()
    before = microscope.get_stage_position()

    with pytest.raises(ValueError):
        microscope.move_to_orientation("MILLING")

    after = microscope.get_stage_position()
    assert after.x == pytest.approx(before.x)
    assert np.isclose(after.r, before.r)
    assert np.isclose(after.t, before.t)


def test_the_refusal_names_the_way_out():
    """The route exists; a caller that hits this needs to be told it, not just stopped."""
    microscope = _parked_at_the_fm()

    with pytest.raises(ValueError, match=r"move_to_microscope\('FIBSEM'\)"):
        microscope.move_to_orientation("MILLING")


def test_the_way_out_actually_works():
    """Traverse back to the beams, re-pose there. What the refusal tells you to do."""
    microscope = _parked_at_the_fm()

    microscope.move_to_microscope("FIBSEM")
    microscope.move_to_orientation("MILLING")

    assert microscope.get_current_device() == "FIBSEM"
    assert microscope.get_stage_orientation() == "MILLING"


def test_the_guard_is_on_the_movement_path_not_just_the_named_one():
    """`move_to_orientation` is one caller of ten; the guard sits under all of them."""
    microscope = _parked_at_the_fm()
    milling = microscope.get_orientation("MILLING")

    with pytest.raises(ValueError, match="Cannot rotate the stage"):
        microscope.safe_absolute_stage_movement(milling)


# ── what it does not refuse ──────────────────────────────────────────


def test_the_pose_it_is_already_in_is_not_a_rotation():
    """A real stage never sits at exactly 180.000, so the check has a tolerance.

    The same five degrees `get_stage_orientation` classifies within, so "the pose I am
    already in" and "the pose this position reads as" cannot disagree.
    """
    microscope = _parked_at_the_fm()

    microscope.move_to_orientation("FIB")  # where it already is

    assert microscope.get_current_device() == "FM"


def test_a_tilt_alone_is_not_refused():
    """A tilt pivots about an axis through the sample; it does not swing it.

    Where the objective does restrict tilt, the microscope refuses it itself -- FIB-640
    measured z and t. This guard does not duplicate that, and must not, or it would be
    stricter than the hazard.
    """
    microscope = _parked_at_the_fm()
    position = microscope.get_stage_position()
    position.t = position.t + np.radians(3)

    microscope.safe_absolute_stage_movement(position)

    assert microscope.get_current_device() == "FM"


def test_re_posing_at_the_beams_is_untouched():
    """Every existing caller is at the beams, and none of them changes behaviour."""
    microscope = _microscope()

    for orientation in ("SEM", "FIB", "MILLING", "SEM"):
        microscope.move_to_orientation(orientation)
        assert microscope.get_stage_orientation() == orientation

    assert microscope.get_current_device() == "FIBSEM"


def test_a_compustage_is_not_affected():
    """Its objective is under the grid: it reaches the FM by flipping, so "parked at
    the FM" is not a state it can be in -- and it has no rotation axis to swing about."""
    microscope = _microscope(ARCTIS_CONFIG)

    microscope.move_to_microscope("FM")
    microscope.move_to_orientation("SEM")

    assert microscope.get_stage_orientation() == "SEM"


def test_a_system_with_no_fluorescence_microscope_is_not_affected():
    """Dormant until the connection gate opens.

    `microscope.fm` is `None` on every non-compustage system today, so nothing can be
    parked at the FM to begin with -- and a beam-only system must not be told it
    cannot rotate somewhere it has every right to be.
    """
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)
    assert microscope.fm is None

    microscope.move_to_orientation("FIB")
    microscope.move_stage_relative(
        FibsemStagePosition(x=48.8e-3, y=0.0, z=0.0, r=0.0, t=0.0)
    )
    assert microscope.get_current_device() == "FM"

    microscope.move_to_orientation("MILLING")

    assert microscope.get_stage_orientation() == "MILLING"
