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

import os
from copy import deepcopy

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    MILLING_ORIENTATION,
    build_lamella_poses,
    sync_fluorescence_pose,
)
from fibsem.structures import DeviceImagingState, FibsemStagePosition

# The offset mounting, as shipped: the FM 48.8 mm out along x, imaging in the FIB
# pose. Loaded rather than faked by flipping `stage_is_compustage` on the Demo, so the
# device declaration the derivation now reads is the real one.
IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
FM_X = 48.8e-3


def _microscope(compustage: bool = True, with_fm: bool = True):
    if compustage:
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        microscope.system.stage.shuttle_pre_tilt = 0
        microscope._update_orientations()
    else:
        microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
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


def _at_the_fm(microscope, x: float = 0.0, y: float = 50e-6):
    """A position on an offset mount where the objective sees the sample: parked at
    the FM, in the pose the sample was carried out in."""
    fib = microscope.get_orientation("FIB")
    return FibsemStagePosition(x=FM_X + x, y=y, z=0.0, r=fib.r, t=fib.t)


def test_an_offset_mount_tells_a_fluorescence_position_apart_by_place():
    """Why the side is a device question, not an orientation one.

    On an offset mount the fluorescence position is distinguished by travelling ~48 mm
    in x, which no r/t derivation can see: the stage at the FM holds the FIB pose it
    was carried out in. The same pose under the beams is somewhere to mill.
    """
    microscope = _microscope(compustage=False)
    at_the_fm = _at_the_fm(microscope)
    under_the_beams = _at(microscope, "FIB")

    assert microscope.get_stage_orientation(
        at_the_fm
    ) == microscope.get_stage_orientation(under_the_beams)

    from_fm = build_lamella_poses(microscope, at_the_fm)
    from_beams = build_lamella_poses(microscope, under_the_beams)

    assert from_fm.fluorescence.stage_position.x == pytest.approx(at_the_fm.x)
    assert microscope.is_at_device("FIBSEM", from_fm.milling.stage_position)
    assert from_beams.milling.stage_position.x == pytest.approx(under_the_beams.x)


def test_marking_from_fluorescence_derives_a_milling_pose_on_an_offset_mount():
    """The traverse is the device leg of the transform, so a position found under the
    objective comes back under the beams in the milling orientation -- which is what
    the FM overview's *Add Position Here* needs."""
    microscope = _microscope(compustage=False)
    fm_position = _at_the_fm(microscope)

    poses = build_lamella_poses(
        microscope, fm_position, marked_at=FLUORESCENCE_ORIENTATION
    )

    milling = poses.milling.stage_position
    assert microscope.get_stage_orientation(milling) == MILLING_ORIENTATION
    assert microscope.is_at_device("FIBSEM", milling)
    assert poses.fluorescence.stage_position.x == pytest.approx(fm_position.x)


def test_a_declared_side_the_geometry_contradicts_is_refused():
    """The FM tab says which side it is marking from; the position is still checked.
    A beam position declared as fluorescence would otherwise be converted into a
    milling pose somewhere the lamella is not."""
    microscope = _microscope()
    beam_position = _at(microscope, MILLING_ORIENTATION)

    with pytest.raises(ValueError, match="Cannot take this as a fluorescence position"):
        build_lamella_poses(
            microscope, beam_position, marked_at=FLUORESCENCE_ORIENTATION
        )


def test_marking_from_the_beam_side_derives_a_fluorescence_pose_on_an_offset_mount():
    """The pose is at the FM, in the orientation it images in: somewhere the
    objective can see the sample from without moving anything first."""
    microscope = _microscope(compustage=False)
    marked = _at(microscope, MILLING_ORIENTATION)

    poses = build_lamella_poses(microscope, marked)

    assert poses.milling.stage_position.x == pytest.approx(marked.x)
    fluorescence = poses.fluorescence.stage_position
    assert (
        microscope.get_device_imaging_state("FM", fluorescence)
        is DeviceImagingState.READY
    )
    assert fluorescence.x == pytest.approx(FM_X, abs=1e-3)


def test_a_default_orientation_the_objective_cannot_image_from_is_not_used():
    """The Default Orientation control offers a fixed pair of names, and loading a
    saved fluorescence configuration writes its own back -- so "FM" can land on an
    offset mount, where it names no pose at all. The derivation falls back to what the
    device declares rather than quietly producing no fluorescence pose."""
    microscope = _microscope(compustage=False)
    microscope.fm.default_orientation = "FM"

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION))

    assert poses.fluorescence is not None
    assert (
        microscope.get_device_imaging_state("FM", poses.fluorescence.stage_position)
        is DeviceImagingState.READY
    )


def test_the_round_trip_holds_on_an_offset_mount():
    """Same property as the compustage round trip, across the traverse."""
    microscope = _microscope(compustage=False)
    marked = _at(microscope, MILLING_ORIENTATION)
    first = build_lamella_poses(microscope, marked)

    second = build_lamella_poses(microscope, first.fluorescence.stage_position)

    for axis in ("x", "y", "z"):
        assert getattr(second.milling.stage_position, axis) == pytest.approx(
            getattr(first.milling.stage_position, axis), abs=1e-12
        )
    assert second.milling.stage_position.t == pytest.approx(
        first.milling.stage_position.t, abs=1e-9
    )


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


def test_the_fluorescence_pose_follows_on_an_offset_mount_too():
    """It used to warn and leave the pose stale there, for want of a transform across
    the traverse. Now it is the same sync as on a compustage."""
    microscope = _microscope(compustage=False)
    lamella = _lamella(microscope)
    _move_milling_to(microscope, lamella, 400e-6, -200e-6)

    assert sync_fluorescence_pose(microscope, lamella) is True

    moved = lamella.fluorescence_pose.stage_position
    assert microscope.get_device_imaging_state("FM", moved) is DeviceImagingState.READY
    fresh = build_lamella_poses(
        microscope, _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    assert moved.x == pytest.approx(fresh.fluorescence.stage_position.x, abs=1e-12)
    assert moved.y == pytest.approx(fresh.fluorescence.stage_position.y, abs=1e-12)


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
