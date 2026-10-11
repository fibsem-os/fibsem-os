"""The stage movement service: built where there is a stage, and what its own
commands add over the microscope methods that route to it.

What the moves command is pinned in ``test_stage_movement_pins.py``, through the old
methods; this covers the service called directly.
"""

import os

import pytest

from fibsem import utils
from fibsem.devices.stage import StageLimitError
from fibsem.services.stage_movement import StageMovement
from fibsem.structures import BeamType, FibsemStagePosition


@pytest.fixture
def microscope():
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(
        manufacturer="Demo", ip_address="localhost", setup_logging=False
    )
    return microscope


def test_the_demo_builds_one_over_its_stage_and_beams(microscope):
    movement = microscope.stage_movement
    assert isinstance(movement, StageMovement)
    assert movement.stage is microscope.stage
    assert movement.electron is microscope.beams[BeamType.ELECTRON]
    assert movement.ion is microscope.beams[BeamType.ION]
    assert movement not in microscope.devices.values()


def test_without_a_stage_there_is_none_and_the_old_moves_stay(microscope):
    from fibsem.services.stage_movement import bind_stage_movement

    microscope.stage = None
    assert bind_stage_movement(StageMovement, microscope) is None


def test_its_views_are_the_backends(microscope):
    assert microscope.stage_movement.vertical_move_views == tuple(
        microscope.vertical_move_views
    )
    microscope.vertical_move_views = (BeamType.ION,)
    assert not microscope.stage_movement.supports_vertical_move(BeamType.ELECTRON)


def test_its_plain_moves_check_the_stage_limits(microscope):
    movement = microscope.stage_movement
    start = microscope.get_stage_position()

    end = movement.move_relative(FibsemStagePosition(x=10e-6))
    assert end.x == pytest.approx(start.x + 10e-6)

    end = movement.move_absolute(FibsemStagePosition(x=0.0, y=0.0))
    assert (end.x, end.y) == (0.0, 0.0)

    with pytest.raises(StageLimitError):
        movement.move_absolute(FibsemStagePosition(x=10.0))


def test_a_move_called_on_it_is_recorded_once_under_its_name(microscope):
    moves = []
    microscope.record_signal.connect(
        lambda kind, payload: moves.append(payload) if kind == "stage_moved" else None
    )

    microscope.stage_movement.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ION)
    microscope.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ION)

    assert [m["move"] for m in moves] == ["stable_move", "stable_move"]
    assert moves[0]["request"] == moves[1]["request"]
