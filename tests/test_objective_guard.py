"""No stage move in the movement service carries the inserted FM objective away.

On an offset mount the objective is inserted over the sample at the FM, 48.8 mm from
the beams. A move that leaves the FM, or rotates there, with it still in would drag
the sample under it across the chamber. `move_to_device` and the safe move retract it
first; every other move through the service is refused while it is in. The raw
`move_stage_*` calls are left alone.

Every test starts at the FM with the objective inserted, records each stage move the
service makes, and checks the objective was out whenever a move changed device.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures.settings import BeamType, FibsemStagePosition
from fibsem.structures.system import InsertableDeviceState

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")

REFUSED = "Retract the objective, or use move_to_device"


@pytest.fixture
def at_the_fm():
    """An offset system at the FM with the objective in, and a log of its moves:
    (device before, device after, objective state) for each stage move."""
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")
    movement.move_to_device("FM")
    assert movement.current_device() == "FM"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED

    moves = []
    stage = microscope.stage
    for name in ("move_absolute", "move_relative"):
        original = getattr(stage, name)

        def spy(position, _original=original):
            before = movement.current_device()
            state = microscope.fm.objective.state
            result = _original(position)
            moves.append((before, movement.current_device(), state))
            return result

        setattr(stage, name, spy)
    return microscope, moves


def _objective_out_whenever_the_device_changed(moves):
    for before, after, state in moves:
        if before != after:
            assert state is not InsertableDeviceState.INSERTED, (before, after)


def _refused_without_moving(microscope, moves, move):
    before = microscope.get_stage_position()
    with pytest.raises(ValueError, match=REFUSED):
        move()
    assert moves == []
    after = microscope.get_stage_position()
    assert after.x == pytest.approx(before.x)
    assert after.r == pytest.approx(before.r)


# ── refused: they would leave the FM or rotate with the objective in ─────


def test_an_absolute_move_to_the_beams_is_refused(at_the_fm):
    microscope, moves = at_the_fm
    _refused_without_moving(
        microscope,
        moves,
        lambda: microscope.stage_movement.move_absolute(FibsemStagePosition(x=0.0)),
    )


def test_a_relative_move_off_the_fm_is_refused(at_the_fm):
    microscope, moves = at_the_fm
    _refused_without_moving(
        microscope,
        moves,
        lambda: microscope.stage_movement.move_relative(
            FibsemStagePosition(x=-48.8e-3)
        ),
    )


def test_an_absolute_rotation_at_the_fm_is_refused(at_the_fm):
    microscope, moves = at_the_fm
    _refused_without_moving(
        microscope,
        moves,
        lambda: microscope.stage_movement.move_absolute(FibsemStagePosition(r=0.0)),
    )


def test_a_stable_move_off_the_fm_is_refused(at_the_fm):
    microscope, moves = at_the_fm
    _refused_without_moving(
        microscope,
        moves,
        lambda: microscope.stage_movement.stable_move(
            dx=48.8e-3, dy=0.0, beam_type=BeamType.ELECTRON
        ),
    )


# ── allowed: they stay at the FM ─────────────────────────────────────────


def test_moves_within_the_fm_pass(at_the_fm):
    microscope, moves = at_the_fm
    movement = microscope.stage_movement
    movement.move_relative(FibsemStagePosition(x=0.5e-3))
    movement.move_absolute(FibsemStagePosition(y=0.5e-3))
    movement.stable_move(dx=10e-6, dy=10e-6, beam_type=BeamType.ELECTRON)
    movement.vertical_move(dy=5e-6, beam_type=BeamType.ION)
    movement.fm_stable_move(dx=10e-6, dy=10e-6)
    assert len(moves) >= 5
    assert movement.current_device() == "FM"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED


# ── the routes that retract first ────────────────────────────────────────


def test_moving_to_the_beams_and_back_retracts(at_the_fm):
    microscope, moves = at_the_fm
    movement = microscope.stage_movement
    movement.move_to_device("FIBSEM")
    assert movement.current_device() == "FIBSEM"
    movement.move_to_orientation("SEM")
    movement.move_to_device("FM")
    assert movement.current_device() == "FM"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED
    _objective_out_whenever_the_device_changed(moves)


def test_a_new_orientation_at_the_fm_is_refused_before_moving(at_the_fm):
    """The rotation guard's refusal: re-posing goes by move_to_device."""
    microscope, moves = at_the_fm
    with pytest.raises(ValueError, match="Cannot rotate the stage"):
        microscope.stage_movement.move_to_orientation("SEM")
    assert moves == []


def test_a_saved_position_at_the_beams_is_reached_with_the_objective_out(at_the_fm):
    microscope, moves = at_the_fm
    movement = microscope.stage_movement
    at_the_fm_position = microscope.get_stage_position()
    beams = movement.target_position(at_the_fm_position, target_device="FIBSEM")
    movement.save_position("at the beams", beams)
    movement.move_to_saved_position("at the beams")
    assert movement.current_device() == "FIBSEM"
    _objective_out_whenever_the_device_changed(moves)


def test_the_safe_move_to_the_beams_retracts(at_the_fm):
    microscope, moves = at_the_fm
    movement = microscope.stage_movement
    beams = movement.target_position(
        microscope.get_stage_position(), target_device="FIBSEM"
    )
    movement.safe_absolute_stage_movement(beams)
    assert movement.current_device() == "FIBSEM"
    _objective_out_whenever_the_device_changed(moves)


def test_a_milling_angle_at_the_fm_never_moves_with_the_objective_in(at_the_fm):
    microscope, moves = at_the_fm
    try:
        microscope.stage_movement.move_to_milling_angle(15)
    except ValueError:
        pass
    _objective_out_whenever_the_device_changed(moves)
    for before, after, state in moves:
        if state is InsertableDeviceState.INSERTED:
            assert before == after == "FM"


def test_inserting_at_a_tilt_retracts_before_moving(at_the_fm):
    microscope, moves = at_the_fm
    position = microscope.get_stage_position()
    microscope.stage_movement.insert_objective(tilt=position.t + 0.02)
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED
    for _, _, state in moves:
        assert state is not InsertableDeviceState.INSERTED


# ── what is left alone ───────────────────────────────────────────────────


def test_the_raw_move_is_left_alone(at_the_fm):
    microscope, _ = at_the_fm
    microscope.move_stage_relative(FibsemStagePosition(x=-48.8e-3))
    assert microscope.stage_movement.current_device() == "FIBSEM"


def test_a_compustage_is_not_affected():
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    microscope.stage_movement.move_to_device("FM")
    microscope.stage_movement.move_relative(FibsemStagePosition(x=0.5e-3))
