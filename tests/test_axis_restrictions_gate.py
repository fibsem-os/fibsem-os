"""Which stage axes an absolute move leaves alone, and on which stages.

`ThermoMicroscope.move_stage_absolute` blanks the axes it believes the microscope
will refuse. The objective half used to be compustage-gated, because it dropped z and
r, and r is a real rotation axis on an offset mount. FIB-640 measured the inserted
objective blocking z and t, and that pair was settled (2026-10-05), so the half now
applies on every mount and names z and t (`ObjectiveLens.blocked_axes`).

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
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_orientation("SEM")
    microscope.fm.objective.insert()

    assert microscope._axis_restrictions_apply() is True
    assert microscope._blocked_axes() == ("z", "t")


def test_a_compustage_with_the_objective_out_does_not():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.retract()
    microscope.move_to_orientation("SEM")

    assert microscope._axis_restrictions_apply() is False


def test_an_offset_mount_with_the_objective_in_blocks_z_and_t():
    """An iFLM has an objective too. It blocks z and t, and leaves the rotation, which
    is a real axis here, to the move."""
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.insert()

    assert microscope._fm_is_a_pose() is False
    assert microscope._blocked_axes() == ("z", "t")


def test_a_move_leaves_the_blocked_axes_alone():
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.insert()
    target = FibsemStagePosition(
        x=1e-3, y=2e-3, z=3e-3, r=0.5, t=0.2, coordinate_system="RAW"
    )

    sent = microscope._without_blocked_axes(target)

    assert (sent.x, sent.y, sent.z, sent.r, sent.t) == (1e-3, 2e-3, None, 0.5, None)
    assert target.z == 3e-3, "the caller's position is not changed"


def test_with_the_objective_out_nothing_is_blocked():
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.retract()
    target = FibsemStagePosition(x=1e-3, z=3e-3, t=0.2, coordinate_system="RAW")

    assert microscope._without_blocked_axes(target) is target


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
    assert after is True


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
    """The safe move's rotation step sends a bare tilt with no rotation: nothing to read a
    destination from, so the stage's own pose stands in -- what such a move got before."""
    microscope = _microscope(ARCTIS_CONFIG)
    _at_fm_pose(microscope)

    tilt_only = FibsemStagePosition(t=0.0, coordinate_system="RAW")
    assert microscope._axis_restrictions_apply(tilt_only) is True
