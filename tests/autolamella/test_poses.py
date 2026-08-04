"""Deriving a lamella's two poses from wherever it was marked.

The property under test is a round trip: a lamella marked on the beam side gets a
fluorescence pose derived from it, and feeding *that* back in has to describe the same
lamella. It did not, before the orientation was read off the position — a fluorescence
position was taken as somewhere to mill, giving a milling pose at t = -180, which is not
an orientation anything mills at. Nothing rejected it; it would have failed later,
somewhere else.

No Qt and no experiment here: this is the arithmetic, and it is worth being able to test
it without either.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    MILLING_ORIENTATION,
    build_lamella_poses,
)
from fibsem.structures import FibsemStagePosition


def _microscope(compustage: bool = True, with_fm: bool = True):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = compustage
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()
    if with_fm and microscope.fm is None:
        from fibsem.fm.microscope import FluorescenceMicroscope

        microscope.fm = FluorescenceMicroscope(parent=microscope)
    if not with_fm:
        microscope.fm = None
    return microscope


def _at(microscope, orientation: str, x: float = 100e-6, y: float = 50e-6):
    """A stage position in a named orientation, off-centre so a lost x/y shows up."""
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=x, y=y, z=0.0, r=pose.r, t=pose.t)


# ── the round trip ───────────────────────────────────────────────────────


def test_a_fluorescence_pose_fed_back_describes_the_same_lamella():
    """The invariant. Mark a lamella on the beam side, take the fluorescence pose it
    derives, hand that back — both poses must come out the same."""
    microscope = _microscope()
    marked = _at(microscope, MILLING_ORIENTATION)

    first = build_lamella_poses(microscope, marked)
    again = build_lamella_poses(microscope, first.fluorescence.stage_position)

    assert again.milling.stage_position.x == pytest.approx(
        first.milling.stage_position.x, abs=1e-12
    )
    assert again.milling.stage_position.y == pytest.approx(
        first.milling.stage_position.y, abs=1e-12
    )
    assert again.milling.stage_position.t == pytest.approx(
        first.milling.stage_position.t, abs=1e-9
    )
    assert again.fluorescence.stage_position.t == pytest.approx(
        first.fluorescence.stage_position.t, abs=1e-9
    )


def test_a_fluorescence_position_is_not_taken_as_somewhere_to_mill():
    """The failure this exists to stop. Handed a fluorescence position, the old path
    set it as the milling pose verbatim — a milling pose at t = -180."""
    microscope = _microscope()
    fm_position = _at(microscope, FLUORESCENCE_ORIENTATION)

    poses = build_lamella_poses(microscope, fm_position)

    assert (
        microscope.get_stage_orientation(poses.milling.stage_position)
        == MILLING_ORIENTATION
    )
    assert np.rad2deg(poses.milling.stage_position.t) != pytest.approx(-180.0)


def test_a_fluorescence_position_is_kept_as_the_fluorescence_pose():
    """It is what the user actually picked — deriving it back would be a round trip
    through an orientation for no reason, and any error in the transform would land on
    the one position that was known exactly."""
    microscope = _microscope()
    fm_position = _at(microscope, FLUORESCENCE_ORIENTATION)

    poses = build_lamella_poses(microscope, fm_position)

    assert poses.fluorescence.stage_position.x == pytest.approx(fm_position.x)
    assert poses.fluorescence.stage_position.y == pytest.approx(fm_position.y)
    assert poses.fluorescence.stage_position.t == pytest.approx(fm_position.t)


def test_marking_moves_neither_pose_laterally():
    """Only the orientation is rewritten. A transform that shifted x or y would put the
    lamella somewhere nobody pointed at, which is the whole family of bug this comes
    from."""
    microscope = _microscope()
    fm_position = _at(microscope, FLUORESCENCE_ORIENTATION, x=321e-6, y=-123e-6)

    poses = build_lamella_poses(microscope, fm_position)

    for pose in (poses.milling, poses.fluorescence):
        assert pose.stage_position.x == pytest.approx(321e-6, abs=1e-12)
        assert pose.stage_position.y == pytest.approx(-123e-6, abs=1e-12)


# ── beam-side behaviour must not change ──────────────────────────────────


@pytest.mark.parametrize("orientation", ["SEM", "MILLING"])
def test_a_beam_position_is_still_the_milling_pose_verbatim(orientation):
    """Every caller before this marked positions on the beam side, and the position
    they gave became the milling pose unchanged — including its tilt, which is not
    always the milling orientation. `update_milling_angle` reads the angle off that
    tilt, so re-posing it here would quietly change every existing lamella."""
    microscope = _microscope()
    marked = _at(microscope, orientation)

    poses = build_lamella_poses(microscope, marked)

    assert poses.milling.stage_position.t == pytest.approx(marked.t)
    assert poses.milling.stage_position.r == pytest.approx(marked.r)


def test_a_beam_position_still_derives_a_fluorescence_pose():
    microscope = _microscope()
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked)

    assert (
        microscope.get_stage_orientation(poses.fluorescence.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


# ── the objective ────────────────────────────────────────────────────────


def test_the_fluorescence_pose_carries_an_objective_position():
    """Without one the pose does not count as selected — `fluorescence_selected` checks
    exactly this — so a lamella could be marked and still read as unmarked."""
    microscope = _microscope()

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION))

    assert poses.fluorescence.objective_position is not None
    assert poses.fluorescence.objective_position == pytest.approx(
        microscope.fm.objective.focus_position
    )


def test_a_given_objective_position_wins():
    """The FM tab knows where the objective actually was; the focus position is only
    the fallback for a caller that cannot know."""
    microscope = _microscope()

    poses = build_lamella_poses(
        microscope, _at(microscope, MILLING_ORIENTATION), objective_position=4.2e-3
    )

    assert poses.fluorescence.objective_position == pytest.approx(4.2e-3)


# ── systems that cannot do it ────────────────────────────────────────────


def test_a_microscope_without_fluorescence_gets_no_fluorescence_pose():
    """None rather than an invented one: a pose for an instrument that does not exist
    is worse than its absence."""
    microscope = _microscope(with_fm=False)

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION))

    assert poses.fluorescence is None
    assert poses.milling is not None


def test_an_offset_mount_cannot_tell_a_fluorescence_position_apart():
    """Why `marked_at` exists at all.

    Deriving the orientation from the position works on a compustage, where each
    orientation has its own tilt. On an offset mount the fluorescence position is
    distinguished by travelling ~48 mm in x, and the tilt is no help: measured on the
    simulator, FM and FIB share a tilt of 17 degrees and both come back as MILLING.

    So a caller there cannot rely on the derivation, and this test says so rather than
    leaving the next person to discover it.
    """
    microscope = _microscope(compustage=False)

    fm_orientation = microscope.get_orientation(FLUORESCENCE_ORIENTATION)

    assert microscope.get_stage_orientation(fm_orientation) != FLUORESCENCE_ORIENTATION


def test_marking_from_fluorescence_is_refused_on_an_offset_mount():
    """There is no transform between the fluorescence and beam positions there — the
    ~48 mm shuttle is not modelled (FIB-93). Refused, because the alternative is a
    lamella with a milling pose nothing can mill at.

    Declared rather than derived, because on this system it cannot be derived."""
    microscope = _microscope(compustage=False)
    fm_position = FibsemStagePosition(
        x=48.8e-3, y=50e-6, z=0.0, r=0.0, t=np.deg2rad(17)
    )

    with pytest.raises(ValueError, match="FIB-93"):
        build_lamella_poses(
            microscope, fm_position, marked_at=FLUORESCENCE_ORIENTATION
        )


def test_a_declared_orientation_wins_over_the_derived_one():
    """The FM tab knows which side it is marking from; the position may not say."""
    microscope = _microscope()
    # A beam-side position by its tilt, declared as fluorescence anyway.
    beam_position = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(
        microscope, beam_position, marked_at=FLUORESCENCE_ORIENTATION
    )

    # Treated as the fluorescence pose: kept as given, and a milling pose derived.
    assert poses.fluorescence.stage_position.t == pytest.approx(beam_position.t)
    assert (
        microscope.get_stage_orientation(poses.milling.stage_position)
        == MILLING_ORIENTATION
    )


def test_marking_from_the_beam_side_still_works_on_an_offset_mount():
    """The other direction is a convenience, not a requirement. Refusing the whole
    lamella because no fluorescence pose could be worked out would stop offset systems
    marking lamellae at all."""
    microscope = _microscope(compustage=False)
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked)

    assert poses.milling.stage_position.x == pytest.approx(marked.x)
    assert poses.fluorescence is None
