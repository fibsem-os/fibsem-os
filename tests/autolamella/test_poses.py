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
    FLUORESCENCE_POSE,
    MILLING_ORIENTATION,
    MILLING_POSE,
    Followed,
    PoseProvenance,
    build_lamella_poses,
    derive_pose,
    move_pose,
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


# ── an offset mount: the side is a place, not a pose ────────────────────


def _iflm():
    import os

    import fibsem.config as cfg

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
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
    """It used to be refused: there was no conversion across the traverse. There is
    now, so a target found in fluorescence gets a milling pose under the beams."""
    from fibsem.applications.autolamella.poses import FLUORESCENCE_POSE

    microscope = _iflm()
    at_the_fm = microscope.to_device(_at(microscope, "FIB"), "FM")

    poses = build_lamella_poses(microscope, at_the_fm)

    assert poses.observed == FLUORESCENCE_POSE
    assert poses.fluorescence.stage_position.x == pytest.approx(at_the_fm.x)
    assert microscope.is_at_device("FIBSEM", poses.milling.stage_position)
    assert (
        microscope.get_stage_orientation(poses.milling.stage_position)
        == MILLING_ORIENTATION
    )


def test_a_beam_position_on_an_offset_mount_gets_a_fluorescence_pose():
    from fibsem.applications.autolamella.poses import MILLING_POSE
    from fibsem.structures import DeviceImagingState

    microscope = _iflm()
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked)

    assert poses.observed == MILLING_POSE
    assert poses.milling.stage_position.x == pytest.approx(marked.x)
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

    assert back.milling.stage_position.x == pytest.approx(marked.x, abs=1e-9)
    assert back.milling.stage_position.y == pytest.approx(marked.y, abs=1e-9)


def test_mid_traverse_is_refused():
    """Neither somewhere to mill nor somewhere the objective sees the sample from."""
    microscope = _iflm()
    between = _at(microscope, "FIB", x=24e-3)
    assert microscope.get_current_device(between) is None

    with pytest.raises(ValueError, match="fluorescence position"):
        build_lamella_poses(microscope, between)


def test_a_declared_side_cannot_override_the_geometry():
    """`observed` breaks a tie and nothing else. A beam position declared as
    fluorescence is still the milling pose -- declaring the side outright is how a
    milling pose 48 mm from the beams used to be possible."""
    from fibsem.applications.autolamella.poses import FLUORESCENCE_POSE, MILLING_POSE

    microscope = _iflm()
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked, observed=FLUORESCENCE_POSE)

    assert poses.observed == MILLING_POSE
    assert poses.milling.stage_position.x == pytest.approx(marked.x)


# ── a position both instruments can use ─────────────────────────────────


def _both_can_use_sem(microscope):
    """A compustage whose objective also images from the SEM pose."""
    microscope.system.stage.devices["FM"].acquisition_orientations = ["FM", "SEM"]
    return microscope


def test_a_position_both_can_use_is_both_poses():
    """The fluorescence pose is a copy, not a flip: the person was looking at the
    sample from here, and t = -180 is somewhere else."""
    microscope = _both_can_use_sem(_microscope())
    marked = _at(microscope, "SEM")

    poses = build_lamella_poses(microscope, marked)

    assert poses.milling.stage_position.t == pytest.approx(marked.t)
    assert poses.fluorescence.stage_position.t == pytest.approx(marked.t)
    assert poses.fluorescence.stage_position.x == pytest.approx(marked.x)


def test_the_caller_says_which_instrument_a_shared_position_was_marked_through():
    from fibsem.applications.autolamella.poses import FLUORESCENCE_POSE, MILLING_POSE

    microscope = _both_can_use_sem(_microscope())
    marked = _at(microscope, "SEM")

    assert build_lamella_poses(microscope, marked).observed == MILLING_POSE
    assert (
        build_lamella_poses(microscope, marked, observed=FLUORESCENCE_POSE).observed
        == FLUORESCENCE_POSE
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
    """Nothing is owed there but the milling pose, which is the position verbatim --
    what a beam-only system has always got."""
    microscope = _microscope(with_fm=False)
    marked = _at(microscope, "SEM")
    marked.t = np.radians(-90)

    poses = build_lamella_poses(microscope, marked)

    assert poses.milling.stage_position.t == pytest.approx(marked.t)
    assert poses.fluorescence is None


# ── moving one pose: what the other one does ─────────────────────────────
#
# The rule (`move_pose`): a pose that is still derived follows; one somebody observed
# stays until `derive_pose` is asked for by name; a missing one is not invented.


def _lamella(microscope, tmp_path, x=100e-6, y=50e-6, with_fluorescence=True):
    """A real lamella with both poses, built and stamped the way the app does it."""
    from fibsem.applications.autolamella.structures import Lamella

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION, x, y))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "L01"), number=1)
    lamella.milling_pose = poses.milling
    if with_fluorescence:
        lamella.fluorescence_pose = poses.fluorescence
    lamella.pose_provenance.update(
        {k: v for k, v in poses.provenance.items() if k in lamella.poses}
    )
    return lamella


def test_a_new_lamella_says_which_pose_was_marked(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.OBSERVED
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED


def test_a_derived_pose_follows_the_one_that_moved(tmp_path):
    """The bug the old sync existed for. A lamella moved on the beam side kept a
    fluorescence pose describing where it used to be -- and nothing about a pose
    pointing at the old place looks wrong."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    followed = move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert followed is Followed.DERIVED
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.y == pytest.approx(-200e-6)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_an_observed_pose_stays_where_somebody_put_it(tmp_path):
    """The bug the old sync *had*: it rewrote a pose somebody had centred by hand."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    centred = _at(microscope, FLUORESCENCE_ORIENTATION, 103e-6, 48e-6)
    move_pose(microscope, lamella, FLUORESCENCE_POSE, position=centred)

    followed = move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert followed is Followed.KEPT
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(103e-6)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED


def test_moving_a_pose_makes_it_observed_and_leaves_an_observed_other_alone(tmp_path):
    """Centring the fluorescence pose of a lamella marked at the beams: both are now
    somebody's, and neither moves the other."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.milling_pose.stage_position.x

    followed = move_pose(
        microscope,
        lamella,
        FLUORESCENCE_POSE,
        position=_at(microscope, FLUORESCENCE_ORIENTATION, 103e-6, 48e-6),
    )

    assert followed is Followed.KEPT
    assert lamella.milling_pose.stage_position.x == pytest.approx(before)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED


def test_a_derived_milling_pose_follows_a_fluorescence_move(tmp_path):
    """A target found in fluorescence: its milling pose is the guess, and follows --
    with the milling angle, which no caller has to remember."""
    from fibsem.applications.autolamella.structures import Lamella

    microscope = _microscope()
    poses = build_lamella_poses(microscope, _at(microscope, FLUORESCENCE_ORIENTATION))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "L01"), number=1)
    lamella.milling_pose = poses.milling
    lamella.fluorescence_pose = poses.fluorescence
    lamella.pose_provenance.update(poses.provenance)
    lamella.milling_angle = 999.0

    followed = move_pose(
        microscope,
        lamella,
        FLUORESCENCE_POSE,
        position=_at(microscope, FLUORESCENCE_ORIENTATION, 400e-6, -200e-6),
    )

    assert followed is Followed.DERIVED
    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.milling_angle != 999.0


def test_derive_overwrites_an_observed_pose_because_it_was_asked_to(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    move_pose(
        microscope,
        lamella,
        FLUORESCENCE_POSE,
        position=_at(microscope, FLUORESCENCE_ORIENTATION, 103e-6, 48e-6),
    )

    assert derive_pose(microscope, lamella, FLUORESCENCE_POSE) is True

    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(100e-6)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED


def test_a_move_keeps_the_objective_position(tmp_path):
    """Someone focused on this lamella by hand; moving it sideways is not a reason to
    throw that away."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.objective_position = 7.7e-3

    move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert lamella.fluorescence_pose.objective_position == pytest.approx(7.7e-3)


def test_recording_a_state_keeps_the_objective_position(tmp_path):
    """A recorded microscope state does not capture the objective."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.objective_position = 7.7e-3
    state = deepcopy(lamella.fluorescence_pose)
    state.objective_position = None

    move_pose(microscope, lamella, FLUORESCENCE_POSE, state=state)

    assert lamella.fluorescence_pose.objective_position == pytest.approx(7.7e-3)


def test_a_move_does_not_invent_a_fluorescence_pose(tmp_path):
    """A lamella with none has never been looked at under fluorescence, and
    `fluorescence_selected` asks only whether an objective position exists."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path, with_fluorescence=False)

    followed = move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert followed is Followed.MISSING
    assert lamella.fluorescence_pose is None


def test_a_derived_pose_follows_on_an_offset_mount_too(tmp_path):
    microscope = _iflm()
    lamella = _lamella(microscope, tmp_path)

    followed = move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert followed is Followed.DERIVED
    expected = microscope.to_device(lamella.milling_pose.stage_position, "FM")
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(expected.x)
    assert lamella.fluorescence_pose.stage_position.y == pytest.approx(expected.y)


def test_a_derived_pose_stays_in_the_orientation_it_was_put_in(tmp_path):
    """A fluorescence pose derived at the SEM tilt is not flipped to t = -180
    because its lamella moved."""
    microscope = _both_can_use_sem(_microscope())
    lamella = _lamella(microscope, tmp_path)
    assert derive_pose(microscope, lamella, FLUORESCENCE_POSE, orientation="SEM")

    move_pose(
        microscope,
        lamella,
        MILLING_POSE,
        position=_at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6),
    )

    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == "SEM"
    )


def test_a_milling_pose_is_not_derived_from_somewhere_the_fm_cannot_see(tmp_path):
    """The dangerous direction refuses, and leaves what was there."""
    microscope = _iflm()
    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.stage_position = _at(microscope, "FIB")  # at the beams
    before = lamella.milling_pose.stage_position.x

    assert derive_pose(microscope, lamella, MILLING_POSE) is False
    assert lamella.milling_pose.stage_position.x == pytest.approx(before)


def test_following_agrees_with_marking_the_same_position_afresh(tmp_path):
    """A moved lamella and a newly marked one at the same place have to describe the
    same fluorescence pose, or the two paths have drifted apart."""
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    target = _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)

    move_pose(microscope, lamella, MILLING_POSE, position=target)
    fresh = build_lamella_poses(microscope, target)

    moved = lamella.fluorescence_pose.stage_position
    assert moved.x == pytest.approx(fresh.fluorescence.stage_position.x, abs=1e-12)
    assert moved.y == pytest.approx(fresh.fluorescence.stage_position.y, abs=1e-12)
    assert moved.t == pytest.approx(fresh.fluorescence.stage_position.t, abs=1e-9)


# ── provenance on disk ──────────────────────────────────────────────────


def test_provenance_round_trips(tmp_path):
    from fibsem.applications.autolamella.structures import Lamella

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    loaded = Lamella.from_dict(lamella.to_dict())

    assert loaded.provenance_of(MILLING_POSE) is PoseProvenance.OBSERVED
    assert loaded.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED


def test_a_file_from_before_provenance_reads_as_observed(tmp_path):
    """Anything saved was somebody's decision, so nothing of theirs starts following."""
    from fibsem.applications.autolamella.structures import Lamella

    microscope = _microscope()
    data = _lamella(microscope, tmp_path).to_dict()
    del data["pose_provenance"]

    loaded = Lamella.from_dict(data)

    assert loaded.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED


def test_an_unknown_provenance_reads_as_observed(tmp_path):
    from fibsem.applications.autolamella.structures import Lamella

    microscope = _microscope()
    data = _lamella(microscope, tmp_path).to_dict()
    data["pose_provenance"]["FLUORESCENCE"] = "stale"

    assert (
        Lamella.from_dict(data).provenance_of(FLUORESCENCE_POSE)
        is PoseProvenance.OBSERVED
    )
