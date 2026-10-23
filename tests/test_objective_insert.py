"""Inserting the FM objective through the stage movement service: where the stage
stands, or at a tilt the stage is moved to first, with the objective out for the
tilt. What the FM tab's Insert and the agent's fm_insert_objective call."""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import InsertableDeviceState

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


@pytest.fixture
def arctis():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.retract()
    return microscope


def _tilted(microscope, degrees: float):
    position = microscope.get_stage_position()
    position.t = np.radians(degrees)
    microscope.move_stage_absolute(position)
    return microscope.get_stage_position()


def test_without_a_tilt_it_inserts_where_the_stage_stands(arctis):
    before = _tilted(arctis, 5.0)

    arctis.stage_movement.insert_objective()

    assert arctis.fm.objective.state is InsertableDeviceState.INSERTED
    assert arctis.get_stage_position().t == pytest.approx(before.t)


def test_a_tilt_moves_the_stage_with_the_objective_out_then_inserts(arctis):
    _tilted(arctis, 5.0)
    arctis.fm.objective.insert()
    states = []
    original = arctis.move_stage_absolute

    def record(position):
        states.append(arctis.fm.objective.state)
        return original(position)

    arctis.move_stage_absolute = record
    arctis.stage_movement.insert_objective(tilt=0.0)

    assert states == [InsertableDeviceState.RETRACTED]
    assert arctis.get_stage_position().t == pytest.approx(0.0)
    assert arctis.fm.objective.state is InsertableDeviceState.INSERTED


def test_a_stage_within_half_a_degree_stays_put(arctis):
    _tilted(arctis, 0.3)
    moved = []
    arctis.move_stage_absolute = lambda p: moved.append(p)

    arctis.stage_movement.insert_objective(tilt=0.0)

    assert moved == []
    assert arctis.fm.objective.state is InsertableDeviceState.INSERTED


def test_a_tilt_away_from_the_fm_is_refused():
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.retract()
    before = microscope.get_stage_position()

    with pytest.raises(ValueError, match="not at the FM"):
        microscope.stage_movement.insert_objective(tilt=0.0)

    assert microscope.fm.objective.state is InsertableDeviceState.RETRACTED
    assert microscope.get_stage_position().t == pytest.approx(before.t)


def test_retract_takes_the_objective_out(arctis):
    arctis.fm.objective.insert()

    arctis.stage_movement.retract_objective()

    assert arctis.fm.objective.state is InsertableDeviceState.RETRACTED
