"""A simulated microscope with an OFFSET fluorescence microscope.

Until now the only simulator configuration with an FM was the Arctis, which is a
compustage: the objective sits under the grid and the stage turns over to face it.
Every offset-mount code path -- the traverse, the device transform, the fluorescence
pose derivation -- was unreachable without hardware, because `microscope.fm` is
refused on any non-compustage system.

`sim-iflm-configuration.yaml` is the other mounting, and this file is what says it
works. It also pins the things that are *known broken* on an offset mount, so that
fixing them is noticed rather than silently absorbed. Those tests are marked below;
each one failing is the good outcome, not a regression.

See FIB-830 for the device axis that fixes them.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import _to_fluorescence, _to_milling
from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import InsertableDeviceState

ARCTIS_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")
IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


# ── the capability itself ────────────────────────────────────────────


def test_the_offset_configuration_has_a_fluorescence_microscope():
    """The point of the file: an FM on a stage that does not flip.

    Before this, `microscope.fm` was `None` on every non-compustage system, so there
    was nowhere to exercise the offset paths at all.
    """
    microscope = _microscope(IFLM_CONFIG)

    assert microscope._fm_is_a_pose() is False
    assert microscope.fm is not None


def test_the_compustage_configuration_still_has_one():
    """The Arctis sim is unchanged, and did not have to gain a key to stay that way."""
    microscope = _microscope(ARCTIS_CONFIG)

    assert microscope._fm_is_a_pose() is True
    assert microscope.fm is not None


def test_has_fm_defaults_to_whether_the_stage_is_a_compustage():
    """The default is the old branch, so no existing configuration changes behaviour.

    `sim-arctis-configuration.yaml` does not set `has_fm` and still gets an FM; a
    non-compustage configuration that does not set it still gets none.
    """
    assert "has_fm" not in _microscope(ARCTIS_CONFIG).system.sim
    assert _microscope(ARCTIS_CONFIG).fm is not None

    # The shipped default configuration is non-compustage and says nothing about an FM.
    default = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)
    assert default._fm_is_a_pose() is False
    assert default.fm is None


def test_the_offset_fm_looks_along_the_ion_column():
    """`camera_tilt` on an offset mount, and where the number comes from.

    Not a derivation -- the FM was *designed* to share the ion column's line of sight
    on TFS, so this is a coupling to a per-site editable value that happens to be
    right here. Pinned because FIB-335 changes where it comes from, not what it is.
    """
    microscope = _microscope(IFLM_CONFIG)

    assert microscope.fm.camera_tilt == microscope.system.ion.column_tilt == 52


# ── what is known broken on an offset mount ──────────────────────────
#
# Each of these passes *because* the offset support is incomplete. When FIB-830 lands
# they should start failing, and that failure is how it gets noticed. Do not "fix"
# them by loosening the assertion.


def test_there_is_no_fm_orientation_on_an_offset_mount():
    """The root of it: on an offset mount the FM is a place, not a pose.

    `_update_orientations` used to copy the FIB entry -- a second name for a pose
    that already had one, which `get_stage_orientation` (FIB matches first) could
    never return. The copy is gone: asking for the FM *orientation* here is a
    category error, and the position at the FM classifies as the pose it really is.
    Everything below is downstream of this one fact.
    """
    microscope = _microscope(IFLM_CONFIG)

    with pytest.raises(ValueError, match="not supported"):
        microscope.get_orientation("FM")

    fib_pose = microscope.get_orientation("FIB")
    assert microscope.get_stage_orientation(fib_pose) == "FIB"


# ── fixed by FIB-831: the lamella poses convert across the traverse ───


def test_marking_from_the_fluorescence_view_derives_a_milling_pose():
    """Refused until FIB-831: a target found at the FM now gets a milling pose under
    the beams, the traverse undone."""
    microscope = _microscope(IFLM_CONFIG)
    at_the_fm = microscope.to_device(_at(microscope, "FIB"), "FM")

    milling = _to_milling(microscope, at_the_fm)

    assert microscope.is_at_device("FIBSEM", milling)
    assert microscope.get_stage_orientation(milling) == "MILLING"


def test_marking_from_the_beam_side_yields_a_fluorescence_pose():
    """The lamella marked on the beam overview gets somewhere to go under the FM."""
    microscope = _microscope(IFLM_CONFIG)
    fluorescence = _to_fluorescence(microscope, _at(microscope, "MILLING"))

    assert fluorescence is not None
    assert microscope.is_at_device("FM", fluorescence)


def _at(microscope, orientation):
    """A position in a named orientation, off centre and with every axis set."""
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=100e-6, y=50e-6, z=0.0, r=pose.r, t=pose.t)


def test_the_acquisition_guard_can_answer_now():
    """The placeholder guard used to say yes unconditionally off a compustage,
    because neither axis alone could tell an FM position from a beam one. The
    predicate asks both axes, so the offset mount finally gets a real answer:
    not at the FM, and SEM is not a pose the objective images from."""
    from fibsem.structures.system import DeviceImagingState

    microscope = _microscope(IFLM_CONFIG)

    microscope.move_to_orientation("SEM")
    assert (
        microscope.get_device_imaging_state("FM")
        is DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL
    )


def test_the_traverse_re_poses_instead_of_refusing():
    """Step 2 of the workflow leaves the stage at SEM; step 3 accepts that now.

    This used to raise "Cannot move to FM from SEM or MILLING orientation" -- a
    refusal on the ordinary path, since a real workflow acquires the beam-side
    overview at SEM and then moves to the FM. `move_to_device` plans the route the
    message told the user to take by hand: re-pose to FIB at the beams, then travel
    (FIB-832).
    """
    microscope = _microscope(IFLM_CONFIG)

    microscope.move_to_orientation("SEM")
    microscope.move_to_device("FM")

    assert microscope.stage_movement.current_device() == "FM"
    assert microscope.get_stage_orientation() == "FIB"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED


# ── a saved position at the other device goes by the traverse ────────


def _objective_at_moves(microscope) -> list:
    """The objective's state and the stage's device at every leg the stage drives."""
    seen = []
    stage = microscope._movement_service()
    real = stage._move_stage

    def record(position, *args, **kwargs):
        seen.append(
            (microscope.fm.objective.state, microscope.stage_movement.current_device())
        )
        return real(position, *args, **kwargs)

    stage._move_stage = record
    return seen


def test_a_fluorescence_pose_is_reached_with_the_objective_out_for_the_traverse():
    """A task going from a milling pose to a fluorescence pose used to cross the
    chamber in one absolute move, re-posing at the far end."""
    microscope = _microscope(IFLM_CONFIG)
    milling = _at(microscope, "MILLING")
    fluorescence = microscope.to_device(_at(microscope, "FIB"), "FM")
    microscope.safe_absolute_stage_movement(milling)
    microscope.fm.objective.insert()
    seen = _objective_at_moves(microscope)

    microscope.safe_absolute_stage_movement(fluorescence)

    assert microscope.stage_movement.current_device() == "FM"
    assert microscope.get_stage_orientation() == "FIB"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED
    # Every leg driven from the beams, the re-pose and the traverse, with it out.
    assert seen
    assert all(
        state is not InsertableDeviceState.INSERTED
        for state, device in seen
        if device == "FIBSEM"
    )


def test_a_milling_pose_is_reached_from_the_fm_with_the_objective_out():
    """The way back: a task leaving the FM pose with the objective inserted."""
    microscope = _microscope(IFLM_CONFIG)
    milling = _at(microscope, "MILLING")
    microscope.safe_absolute_stage_movement(_at(microscope, "FIB"))
    microscope.move_to_device("FM")
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED
    seen = _objective_at_moves(microscope)

    microscope.safe_absolute_stage_movement(milling)

    assert microscope.stage_movement.current_device() == "FIBSEM"
    assert microscope.get_stage_orientation() == "MILLING"
    assert seen
    assert all(state is not InsertableDeviceState.INSERTED for state, _ in seen)


def test_a_holder_slot_is_reached_from_the_fm_with_the_objective_out():
    """Moving to a slot used to be a raw absolute move, so from the FM it crossed
    the chamber with the objective inserted."""
    microscope = _microscope(IFLM_CONFIG)
    slot = microscope.sample_handling.holder.slots["Slot-01"]
    slot.position = _at(microscope, "SEM")
    microscope.safe_absolute_stage_movement(_at(microscope, "FIB"))
    microscope.move_to_device("FM")
    seen = _objective_at_moves(microscope)

    microscope.sample_handling.move_to_slot("Slot-01")

    assert microscope.stage_movement.current_device() == "FIBSEM"
    assert seen
    assert all(state is not InsertableDeviceState.INSERTED for state, _ in seen)
