"""A fluorescence pose set at the FM stays when the milling pose moves; one worked
out from the milling pose follows it. On the offset (iFLM) simulator."""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import (
    add_lamella,
    fluorescence_move,
    move_lamella,
    set_lamella_pose,
    sync_fluorescence_pose,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.experiment import EXPERIMENT_FILENAME

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    microscope.move_to_orientation("SEM")
    return microscope


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment(path=tmp_path, name="fm-pose")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    return experiment


def _beam_point(microscope, dx: float) -> FibsemStagePosition:
    position = microscope.get_stage_position()
    position.x += dx
    return position


def _marked_at_the_fm(experiment, microscope):
    microscope.stage_movement.move_to_device("FM", "FIB")
    lamella = add_lamella(experiment, microscope)
    microscope.stage_movement.move_to_device("FIBSEM", "SEM")
    return lamella


def test_a_pose_derived_at_the_beams_follows_the_milling_pose(experiment, microscope):
    lamella = add_lamella(experiment, microscope)
    assert lamella.fluorescence_pose_set_at_fm is False
    before = lamella.fluorescence_pose.stage_position.x

    move_lamella(experiment, microscope, lamella, _beam_point(microscope, 200e-6))

    # The FM images in the FIB pose, a half turn from SEM: x runs the other way.
    moved = lamella.fluorescence_pose.stage_position.x - before
    assert abs(moved) == pytest.approx(200e-6)


def test_a_pose_marked_at_the_fm_stays_when_the_milling_pose_moves(
    experiment, microscope
):
    lamella = _marked_at_the_fm(experiment, microscope)
    assert lamella.fluorescence_pose_set_at_fm is True
    kept = lamella.fluorescence_pose.stage_position
    milling = lamella.milling_pose.stage_position.x

    move_lamella(experiment, microscope, lamella, _beam_point(microscope, 200e-6))

    assert lamella.milling_pose.stage_position.x != pytest.approx(milling)
    assert lamella.fluorescence_pose.stage_position == kept


def test_set_pose_at_the_fm_keeps_it_through_a_milling_set_pose(experiment, microscope):
    lamella = add_lamella(experiment, microscope)
    microscope.stage_movement.move_to_device("FM", "FIB")
    set_lamella_pose(
        experiment,
        microscope,
        lamella,
        "FLUORESCENCE",
        microscope.get_microscope_state(),
    )
    kept = lamella.fluorescence_pose.stage_position
    microscope.stage_movement.move_to_device("FIBSEM", "SEM")
    state = microscope.get_microscope_state()
    state.stage_position.x += 200e-6

    followed = set_lamella_pose(experiment, microscope, lamella, "MILLING", state)

    assert followed is False
    assert lamella.fluorescence_pose.stage_position == kept


def test_a_drag_on_the_fm_overview_sets_it_at_the_fm(experiment, microscope):
    lamella = add_lamella(experiment, microscope)
    target = lamella.fluorescence_pose.stage_position
    target.x += 100e-6

    move_lamella(
        experiment,
        microscope,
        lamella,
        poses=fluorescence_move(microscope, lamella, target),
    )

    assert lamella.fluorescence_pose_set_at_fm is True


def test_forced_sync_follows_and_marks_it_derived(experiment, microscope):
    lamella = _marked_at_the_fm(experiment, microscope)
    lamella.milling_pose.stage_position.x += 200e-6

    assert sync_fluorescence_pose(microscope, lamella) is False
    assert sync_fluorescence_pose(microscope, lamella, force=True) is True
    assert lamella.fluorescence_pose_set_at_fm is False


def test_it_is_saved_and_an_older_file_reads_derived(experiment, microscope):
    _marked_at_the_fm(experiment, microscope)
    loaded = Experiment.load(os.path.join(experiment.path, EXPERIMENT_FILENAME))
    assert loaded.positions[0].fluorescence_pose_set_at_fm is True

    saved = loaded.positions[0].to_dict()
    del saved["fluorescence_pose_set_at_fm"]
    assert Lamella.from_dict(saved).fluorescence_pose_set_at_fm is False
