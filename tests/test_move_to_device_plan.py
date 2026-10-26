"""`plan_move_to_device`: the route `move_to_device` takes, with where each step
ends, listed without moving; and the objective's insert and retract positions it
names. On the offset simulator."""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.fm.microscope import ObjectiveLens
from fibsem.structures.system import InsertableDeviceState

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    microscope.move_to_orientation("SEM")
    return microscope


def _close(position, expected):
    for axis in ("x", "y", "z", "r", "t"):
        assert getattr(position, axis) == pytest.approx(expected[axis], abs=1e-9), axis


def test_planning_moves_nothing(microscope):
    before = microscope.get_stage_position()
    microscope.stage_movement.plan_move_to_device("FM", "FIB")
    _close(microscope.get_stage_position(), before.to_dict())
    assert microscope.fm.objective.state is InsertableDeviceState.RETRACTED


def test_the_stage_ends_where_the_plan_says(microscope):
    movement = microscope.stage_movement
    objective = microscope.fm.objective

    out = movement.plan_move_to_device("FM", "FIB")
    assert [s["step"] for s in out] == ["retract", "turn", "travel", "insert"]
    movement.move_to_device("FM", "FIB")
    _close(microscope.get_stage_position(), out[2]["position"])
    assert objective.position == pytest.approx(out[3]["objective_z"])
    assert out[3]["objective_z"] == objective.insert_position

    back = movement.plan_move_to_device("FIBSEM", "SEM")
    assert [s["step"] for s in back] == ["retract", "travel", "turn"]
    assert back[0]["objective_z"] == objective.retract_position
    movement.move_to_device("FIBSEM", "SEM")
    _close(microscope.get_stage_position(), back[2]["position"])
    assert objective.position == pytest.approx(objective.retract_position)


def test_each_step_reads_with_its_values(microscope):
    lines = [s["text"] for s in microscope.stage_movement.plan_move_to_device("FM")]
    assert lines == [
        "Retract the objective to z = -10.000 mm",
        "Turn to the FIB orientation at the beams (r = 180.0°, t = 17.0°)",
        "Travel 48.8 mm to the FM, to x = 48.800 mm, y = 0.000 mm, z = 0.000 mm",
        "Insert the objective to z = 6.000 mm",
    ]


def test_a_turn_at_the_beams_has_no_travel(microscope):
    steps = microscope.stage_movement.plan_move_to_device("FIBSEM", "FIB")
    assert [s["step"] for s in steps] == ["retract", "turn"]
    assert np.degrees(steps[1]["t"]) == pytest.approx(17.0)


def test_the_plan_refuses_what_the_move_refuses(microscope):
    microscope.system.stage.device_calibration = None
    movement = microscope.stage_movement
    with pytest.raises(ValueError) as planned:
        movement.plan_move_to_device("FM", "FIB")
    with pytest.raises(ValueError) as moved:
        movement.move_to_device("FM", "FIB")
    assert str(planned.value) == str(moved.value)


def test_an_objective_that_cannot_say_where_it_goes_reads_none():
    assert ObjectiveLens().insert_position is None
    assert ObjectiveLens().retract_position is None
