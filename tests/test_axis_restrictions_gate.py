"""Which stage types drop z and rotation from an absolute move.

`ThermoMicroscope.move_stage_absolute` blanks two axes when it believes the
microscope will refuse them. That guard has only ever run on a compustage, because
`microscope.fm` is None on every other system -- and opening the connection gate is
what makes it reachable elsewhere.

It is not safe to let that happen. The axes are not equivalent across stage types:
`stage_position_to_autoscript` returns a `CompustagePosition(x, y, z, a)` with no `r`
field at all, so dropping `r` there has never done anything, while on an offset mount
it would drop a real rotation axis. Every absolute move would silently half-succeed --
land at x and y, no z, no rotation.

So the objective half is compustage-gated **temporarily**, and FIB-640 owns removing
it. That issue is explicit that the branch should *not* stay gated once it is correct
-- an iFLM has an objective too -- and it is also where the axis pair gets settled: it
measured z and t, not z and r.

The predicate is tested rather than the move: AutoScript is not installed in CI, so
`ThermoMicroscope.move_stage_absolute` cannot be executed at all.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def test_a_compustage_with_the_objective_in_still_restricts():
    """Unchanged. This is the only place the guard has ever run."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.insert()

    assert microscope._axis_restrictions_apply() is True


def test_a_compustage_with_the_objective_out_does_not():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.retract()
    microscope.move_to_orientation("SEM")

    assert microscope._axis_restrictions_apply() is False


def test_an_offset_mount_with_the_objective_in_does_not_restrict():
    """The gate. Without it, every absolute move on an offset system with an FM would
    silently lose its z and its rotation -- behaviour that has executed on no
    instrument. Remove this with FIB-640, once the axis pair is settled on hardware."""
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.insert()

    assert microscope.stage_is_compustage is False
    assert microscope.fm.objective.state == "Inserted"
    assert microscope._axis_restrictions_apply() is False


def test_a_system_with_no_fluorescence_microscope_does_not_restrict():
    """Where every offset system sits today, gate closed."""
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)

    assert microscope.fm is None
    assert microscope._axis_restrictions_apply() is False


def test_the_orientation_half_needs_no_gate_of_its_own():
    """It is confined to the compustage by the orientations themselves.

    `get_stage_orientation` can never return "FM" on an offset mount -- the FM is a
    device there, and `orientations["FM"]` is a byte-identical copy of the FIB entry,
    which is matched first. So that half is dead off a compustage without anything
    saying so, and gating it would be describing the same fact twice.
    """
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.move_to_microscope("FM")

    assert microscope.get_current_device() == "FM"
    assert microscope.get_stage_orientation() != "FM"


@pytest.mark.parametrize("config_path", [ARCTIS_CONFIG, IFLM_CONFIG])
def test_the_predicate_reads_the_microscope_rather_than_being_told(config_path: str):
    """It answers from live objective state, so inserting flips it where it applies."""
    microscope = _microscope(config_path)
    microscope.move_to_orientation("SEM")
    microscope.fm.objective.retract()
    before = microscope._axis_restrictions_apply()

    microscope.fm.objective.insert()
    after = microscope._axis_restrictions_apply()

    assert before is False
    assert after is microscope.stage_is_compustage


def _at_fm_pose(microscope, z: float = 5.0e-3) -> None:
    fm = microscope.get_orientation("FM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=z, r=fm.r, t=fm.t, coordinate_system="RAW")
    )
    microscope.fm.objective.retract()
    assert microscope.get_stage_orientation() == "FM"


def _pose(microscope, orientation: str, z: float) -> FibsemStagePosition:
    o = microscope.get_orientation(orientation)
    return FibsemStagePosition(
        x=100e-6, y=50e-6, z=z, r=o.r, t=o.t, coordinate_system="RAW"
    )


def test_leaving_the_fluorescence_pose_with_the_objective_out_is_not_restricted():
    """The reported bug (FIB-640). Measured: z and t are both available at t = -180
    once the objective is retracted, so the move out carries its z and lands on the
    first press. Asking the stage's current pose here is what dropped it."""
    microscope = _microscope(ARCTIS_CONFIG)
    _at_fm_pose(microscope, z=5.0e-3)

    target = _pose(microscope, "MILLING", z=12.0e-3)
    assert microscope._axis_restrictions_apply(target) is False


def test_moving_into_the_fluorescence_pose_is_still_restricted():
    """More conservative than the measurement requires, and free: the one path that
    enters the pose sends r and t only."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.retract()
    microscope.move_to_orientation("SEM")

    target = _pose(microscope, "FM", z=12.0e-3)
    assert microscope._axis_restrictions_apply(target) is True


def test_a_partial_pose_falls_back_to_where_the_stage_is():
    """`_safe_rotation_movement` sends a bare tilt with no rotation: nothing to read a
    destination from, so the stage's own pose stands in -- what such a move got before."""
    microscope = _microscope(ARCTIS_CONFIG)
    _at_fm_pose(microscope)

    tilt_only = FibsemStagePosition(t=0.0, coordinate_system="RAW")
    assert microscope._axis_restrictions_apply(tilt_only) is True
