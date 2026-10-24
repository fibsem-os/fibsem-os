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
    sync_fluorescence_pose,
)
from fibsem.structures.settings import FibsemStagePosition
from tests.fixtures.demo_stage import demo_session


def _microscope(compustage: bool = True, with_fm: bool = True):
    microscope, _ = demo_session(compustage=compustage)
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope.stage_movement.update_orientations()
    if with_fm and microscope.fm is None:
        from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope

        microscope.fm = DemoFluorescenceMicroscope(parent=microscope)
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

    assert again.beam.stage_position.x == pytest.approx(
        first.beam.stage_position.x, abs=1e-12
    )
    assert again.beam.stage_position.y == pytest.approx(
        first.beam.stage_position.y, abs=1e-12
    )
    assert again.beam.stage_position.t == pytest.approx(
        first.beam.stage_position.t, abs=1e-9
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
        microscope.get_stage_orientation(poses.beam.stage_position)
        == MILLING_ORIENTATION
    )
    assert np.rad2deg(poses.beam.stage_position.t) != pytest.approx(-180.0)


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

    for pose in (poses.beam, poses.fluorescence):
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

    assert poses.beam.stage_position.t == pytest.approx(marked.t)
    assert poses.beam.stage_position.r == pytest.approx(marked.r)


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
    """Without one, restoring the pose leaves the objective wherever it is, so the
    lamella would come back out of focus."""
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
    assert poses.beam is not None


def test_a_declared_side_the_geometry_contradicts_is_refused():
    """`marked_at` is a check, not the decision. A beam position declared as
    fluorescence used to be taken as the fluorescence pose; the FM cannot see the
    sample from there, so it is refused."""
    microscope = _microscope()
    beam_position = _at(microscope, MILLING_ORIENTATION)

    with pytest.raises(ValueError, match="fluorescence position"):
        build_lamella_poses(
            microscope, beam_position, marked_at=FLUORESCENCE_ORIENTATION
        )


@pytest.mark.parametrize("x", [24e-3, -30e-3])
def test_without_fluorescence_every_position_is_a_milling_pose(x):
    """No FM, no other side: a position anywhere along the stage is the milling pose
    verbatim, not refused for want of a fluorescence position."""
    microscope = _iflm()
    microscope.fm = None
    marked = _at(microscope, MILLING_ORIENTATION, x=x)

    poses = build_lamella_poses(microscope, marked)

    assert poses.beam.stage_position == marked
    assert poses.fluorescence is None


# ── an offset mount: the side is a place, not a pose ────────────────────


def _iflm():
    import os

    import fibsem.config as cfg

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    return microscope


def test_an_offset_mount_cannot_tell_a_fluorescence_position_by_its_pose():
    """Why the side is read off the *place* there. The stage at the FM holds the FIB
    pose it was carried out in, and no r/t derivation can see 48 mm of x."""
    microscope = _iflm()
    at_the_fm = microscope.to_device(_at(microscope, "FIB"), "FM")

    assert microscope.get_stage_orientation(at_the_fm) == "FIB"
    assert microscope.is_at_device("FM", at_the_fm)


def test_a_position_at_an_offset_fm_is_the_fluorescence_pose():
    """It used to be refused (FIB-93). A target found in fluorescence now gets a
    milling pose under the beams."""
    microscope = _iflm()
    at_the_fm = microscope.to_device(_at(microscope, "FIB"), "FM")

    poses = build_lamella_poses(
        microscope, at_the_fm, marked_at=FLUORESCENCE_ORIENTATION
    )

    assert poses.fluorescence.stage_position.x == pytest.approx(at_the_fm.x)
    assert microscope.is_at_device("FIBSEM", poses.beam.stage_position)
    assert (
        microscope.get_stage_orientation(poses.beam.stage_position)
        == MILLING_ORIENTATION
    )


def test_a_beam_position_on_an_offset_mount_gets_a_fluorescence_pose():
    """It used to get none. The lamella is still the milling pose verbatim."""
    from fibsem.structures.system import DeviceImagingState

    microscope = _iflm()
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked)

    assert poses.beam.stage_position.x == pytest.approx(marked.x)
    assert (
        microscope.get_device_imaging_state("FM", poses.fluorescence.stage_position)
        is DeviceImagingState.READY
    )


def test_the_two_directions_agree_on_an_offset_mount():
    """Marked at the beams, its fluorescence pose fed back names the same x/y."""
    microscope = _iflm()
    marked = _at(microscope, MILLING_ORIENTATION)

    out = build_lamella_poses(microscope, marked)
    back = build_lamella_poses(microscope, out.fluorescence.stage_position)

    assert back.beam.stage_position.x == pytest.approx(marked.x, abs=1e-9)
    assert back.beam.stage_position.y == pytest.approx(marked.y, abs=1e-9)


def test_a_beam_position_marked_through_an_offset_fm_is_refused():
    """The FM tab hands over positions at the FM. One at the beams is a mistake, and
    taking it as a fluorescence pose would put the milling pose somewhere else."""
    microscope = _iflm()

    with pytest.raises(ValueError, match="fluorescence position"):
        build_lamella_poses(
            microscope,
            _at(microscope, MILLING_ORIENTATION),
            marked_at=FLUORESCENCE_ORIENTATION,
        )


# ── an unsupported pose ─────────────────────────────────────────────────


def test_an_unsupported_pose_is_refused_where_a_fluorescence_pose_is_owed():
    microscope = _microscope()
    marked = _at(microscope, "SEM")
    marked.t = np.radians(-90)
    assert microscope.get_stage_orientation(marked) == "NONE"

    with pytest.raises(ValueError, match="not a supported orientation"):
        build_lamella_poses(microscope, marked)


def test_an_unsupported_pose_is_still_a_milling_pose_without_an_fm():
    """Nothing is owed there but the milling pose, which is the position verbatim."""
    microscope = _microscope(with_fm=False)
    marked = _at(microscope, "SEM")
    marked.t = np.radians(-90)

    poses = build_lamella_poses(microscope, marked)

    assert poses.beam.stage_position.t == pytest.approx(marked.t)
    assert poses.fluorescence is None


# ── keeping the two in step when only one of them moves ──────────────────


def _lamella(microscope, x=100e-6, y=50e-6, with_fluorescence=True):
    """A stand-in lamella with both poses, built the way the app builds them."""
    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION, x, y))

    class _Lamella:
        name = "Lamella-01"

    lamella = _Lamella()
    lamella.milling_pose = poses.beam
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
    """It has never been marked under fluorescence, so conjuring a pose here would
    give a lamella nobody has looked at a focus to restore."""
    microscope = _microscope()
    lamella = _lamella(microscope, with_fluorescence=False)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is False
    assert lamella.fluorescence_pose is None


def test_the_fluorescence_pose_follows_on_an_offset_mount():
    """It used to be left behind with a warning (FIB-93): the moved lamella's
    fluorescence pose follows it to the FM, keeping its pose there."""
    microscope = _iflm()
    lamella = _lamella(microscope)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is True

    expected = microscope.to_device(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6), "FM"
    )
    moved = lamella.fluorescence_pose.stage_position
    assert moved.x == pytest.approx(expected.x, abs=1e-12)
    assert moved.y == pytest.approx(expected.y, abs=1e-12)
    assert moved.t == pytest.approx(expected.t, abs=1e-9)


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


# ── storing the objective position ───────────────────────────────────────


class _Experiment:
    def __init__(self, positions):
        self.positions = positions
        self.saves = 0

    def save(self):
        self.saves += 1


def test_an_objective_position_is_stored_and_saved():
    from fibsem.applications.autolamella.poses import set_objective_position

    lamella = _lamella(_microscope())
    experiment = _Experiment([lamella])

    set_objective_position(experiment, lamella, 1.5e-3)

    assert lamella.fluorescence_pose.objective_position == 1.5e-3
    assert experiment.saves == 1


def test_apply_to_all_skips_a_lamella_with_no_fluorescence_pose():
    from fibsem.applications.autolamella.poses import set_objective_position_on_all

    microscope = _microscope()
    with_pose = _lamella(microscope)
    without = _lamella(microscope, with_fluorescence=False)
    experiment = _Experiment([with_pose, without])

    assert set_objective_position_on_all(experiment, 2e-3) == 1
    assert with_pose.fluorescence_pose.objective_position == 2e-3
    assert without.fluorescence_pose is None
    assert experiment.saves == 1
    assert set_objective_position_on_all(_Experiment([without]), 2e-3) == 0
