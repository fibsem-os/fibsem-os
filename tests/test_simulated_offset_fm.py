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

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")


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

    assert microscope.stage_is_compustage is False
    assert microscope.fm is not None


def test_the_compustage_configuration_still_has_one():
    """The Arctis sim is unchanged, and did not have to gain a key to stay that way."""
    microscope = _microscope(ARCTIS_CONFIG)

    assert microscope.stage_is_compustage is True
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
    assert default.stage_is_compustage is False
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


def test_the_fm_orientation_is_indistinguishable_from_the_fib_one():
    """The root of it: on an offset mount the FM is a place, not a pose.

    `_update_orientations` copies the FIB entry, so the two are identical and
    `get_stage_orientation` -- which checks FIB first -- can never answer "FM".
    Everything below is downstream of this one fact.
    """
    microscope = _microscope(IFLM_CONFIG)

    fm_pose = microscope.get_orientation("FM")
    fib_pose = microscope.get_orientation("FIB")
    assert (fm_pose.r, fm_pose.t) == (fib_pose.r, fib_pose.t)

    assert microscope.get_stage_orientation(fm_pose) == "FIB"


def test_marking_from_the_fluorescence_view_is_refused():
    """`_to_milling` declines rather than returning a plausible wrong pose.

    The alternative would be a milling pose 48 mm off the beam axis that nothing
    rejects until something tries to mill it.
    """
    microscope = _microscope(IFLM_CONFIG)

    with pytest.raises(ValueError, match="offset mount"):
        _to_milling(microscope, microscope.get_orientation("FM"))


def test_marking_from_the_beam_side_yields_no_fluorescence_pose():
    """The quieter half: the lamella is created, with only one of its two poses.

    `_to_fluorescence` catches the transform's refusal and returns `None`, so a
    lamella marked on the beam overview simply has nowhere to go under the FM.
    """
    microscope = _microscope(IFLM_CONFIG)

    assert _to_fluorescence(microscope, microscope.get_orientation("MILLING")) is None


def test_the_acquisition_guard_can_answer_now():
    """The placeholder guard used to say yes unconditionally off a compustage,
    because neither axis alone could tell an FM position from a beam one. The
    predicate asks both axes, so the offset mount finally gets a real answer:
    not at the FM, and SEM is not a pose the objective images from."""
    from fibsem.structures import DeviceImagingState

    microscope = _microscope(IFLM_CONFIG)

    microscope.move_to_orientation("SEM")
    assert (
        microscope.get_device_imaging_state("FM")
        is DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL
    )


def test_the_traverse_refuses_unless_the_stage_is_already_at_fib():
    """Step 2 of the workflow leaves the stage at SEM; step 3 will not accept that.

    A real workflow acquires the beam-side overview at the SEM orientation and then
    moves to the FM, so this refusal is on the ordinary path, not an edge case. The
    reorientation is something the software could do itself -- see FIB-832.
    """
    microscope = _microscope(IFLM_CONFIG)

    microscope.move_to_orientation("SEM")
    with pytest.raises(ValueError, match="Cannot move to FM"):
        microscope.move_to_microscope("FM")
