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

from copy import deepcopy

import numpy as np
import pytest

from fibsem import utils
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    MILLING_ORIENTATION,
    build_lamella_poses,
    sync_fluorescence_pose,
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
        build_lamella_poses(microscope, fm_position, marked_at=FLUORESCENCE_ORIENTATION)


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


# ── keeping the two in step when only one of them moves ──────────────────


def _lamella(microscope, x=100e-6, y=50e-6, with_fluorescence=True):
    """A stand-in lamella with both poses, built the way the app builds them."""
    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION, x, y))

    class _Lamella:
        name = "Lamella-01"

    lamella = _Lamella()
    lamella.milling_pose = poses.milling
    lamella.fluorescence_pose = poses.fluorescence if with_fluorescence else None
    return lamella


def _move_milling_to(microscope, lamella, x, y):
    """Move only the milling pose, as every beam-side caller has always done."""
    lamella.milling_pose.stage_position = _at(microscope, MILLING_ORIENTATION, x, y)


def test_the_fluorescence_pose_follows_a_milling_pose_that_moved():
    """The bug this exists for. A lamella moved on the beam side kept a fluorescence
    pose describing where it used to be -- and nothing about a stale pose looks wrong."""
    microscope = _microscope()
    lamella = _lamella(microscope)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is True

    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.y == pytest.approx(-200e-6)


def test_the_synced_pose_is_still_in_the_fluorescence_orientation():
    """Only the place moves. A pose that came back carrying the milling tilt would put
    the stage 157 degrees from where the objective is."""
    microscope = _microscope()
    lamella = _lamella(microscope)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    sync_fluorescence_pose(microscope, lamella)

    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_syncing_keeps_the_objective_position():
    """Someone focused on this lamella by hand. Moving it sideways is not a reason to
    throw that away, and re-deriving the whole pose would."""
    microscope = _microscope()
    lamella = _lamella(microscope)
    lamella.fluorescence_pose.objective_position = 7.7e-3
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    sync_fluorescence_pose(microscope, lamella)

    assert lamella.fluorescence_pose.objective_position == pytest.approx(7.7e-3)


def test_a_lamella_with_no_fluorescence_pose_is_not_given_one():
    """It has never been marked under fluorescence, and `fluorescence_selected` asks
    only whether an objective position exists -- so conjuring one here would make a
    lamella nobody has looked at report itself as focused."""
    microscope = _microscope()
    lamella = _lamella(microscope, with_fluorescence=False)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is False
    assert lamella.fluorescence_pose is None


def test_an_offset_mount_says_so_rather_than_inventing_a_pose():
    """There is no transform between the two sides there (FIB-93). The existing pose is
    left alone -- it was put there deliberately and is the better of two bad answers --
    and the caller is told the sync did not happen."""
    microscope = _microscope(compustage=False)
    lamella = _lamella(microscope)
    lamella.fluorescence_pose = deepcopy(lamella.milling_pose)
    lamella.fluorescence_pose.stage_position = _at(
        microscope, FLUORESCENCE_ORIENTATION, 1e-6, 2e-6
    )
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is False
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(1e-6)


def test_syncing_agrees_with_marking_the_same_position_afresh():
    """A moved lamella and a newly marked one at the same place have to describe the
    same thing. Two routes to a fluorescence pose that disagreed would be a bug that
    only showed up on lamellae with a history."""
    microscope = _microscope()
    lamella = _lamella(microscope)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)
    sync_fluorescence_pose(microscope, lamella)

    fresh = build_lamella_poses(
        microscope, _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )

    moved = lamella.fluorescence_pose.stage_position
    assert moved.x == pytest.approx(fresh.fluorescence.stage_position.x, abs=1e-12)
    assert moved.y == pytest.approx(fresh.fluorescence.stage_position.y, abs=1e-12)
    assert moved.t == pytest.approx(fresh.fluorescence.stage_position.t, abs=1e-9)
