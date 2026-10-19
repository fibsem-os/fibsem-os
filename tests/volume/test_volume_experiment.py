"""The volume experiment's saved form: positions, protocol, and which app reads it."""

import pytest
import yaml

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.volume.experiment import PROTOCOL_FILENAME, VolumeExperiment
from fibsem.applications.volume.position import MILL_POSE, SEM_POSE, VolumePosition
from fibsem.applications.volume.protocol import VolumeProtocol
from fibsem.applications.volume.tasks import VolumeFIBMillingTaskConfig
from fibsem.structures import FibsemStagePosition, MicroscopeState, Point
from fibsem.workflows.experiment import load_experiment
from fibsem.workflows.saved_format import WrongApplication
from fibsem.workflows.status import TaskState, TaskStatus

TASKS = ["Volume FIB Milling", "Volume SEM Acquisition"]


def _state(x: float = 0.0) -> MicroscopeState:
    return MicroscopeState(stage_position=FibsemStagePosition(x=x, y=0, z=0, r=0, t=0))


def _ran(position: VolumePosition, name: str, iteration: int, status: TaskStatus):
    position.task_history.append(
        TaskState(name=name, iteration=iteration, status=status)
    )


@pytest.fixture
def experiment(tmp_path):
    return VolumeExperiment.create(tmp_path, "vol")


def test_a_saved_experiment_reads_back_as_volume(experiment):
    position = experiment.add_position(
        "aspen", _state(1e-3), Point(0, 2e-6), iterations=4
    )
    position.poses[SEM_POSE] = _state(2e-3)
    _ran(position, TASKS[0], 0, TaskStatus.Completed)
    experiment.task_protocol.task_config[TASKS[0]].slice_thickness = 50e-9
    experiment.save(save_protocol=True)

    loaded = load_experiment(experiment.path)

    assert isinstance(loaded, VolumeExperiment)
    assert loaded.id == experiment.id
    (back,) = loaded.positions
    assert back.to_dict() == position.to_dict()
    assert back.pose(SEM_POSE).stage_position.x == pytest.approx(2e-3)
    assert loaded.task_protocol.ordered_task_names == TASKS
    assert loaded.task_protocol.task_config[TASKS[0]].slice_thickness == 50e-9


def test_it_saves_as_app_volume_with_the_protocol_beside_it(experiment, tmp_path):
    saved = yaml.safe_load((tmp_path / "vol" / "experiment.yaml").read_text())
    protocol = yaml.safe_load((tmp_path / "vol" / PROTOCOL_FILENAME).read_text())

    assert saved["app"] == "volume"
    assert list(protocol["tasks"]) == TASKS


def test_an_autolamella_file_is_refused_and_left_alone(tmp_path):
    lamella_experiment = Experiment(path=tmp_path, name="al")
    (tmp_path / "al").mkdir()
    lamella_experiment.save()
    saved = tmp_path / "al" / "experiment.yaml"
    before = saved.read_text()

    with pytest.raises(WrongApplication):
        VolumeExperiment.load(saved)

    assert saved.read_text() == before


def test_a_missing_protocol_reads_as_the_default(experiment, tmp_path):
    (tmp_path / "vol" / PROTOCOL_FILENAME).unlink()

    loaded = VolumeExperiment.load(tmp_path / "vol")

    assert loaded.task_protocol.ordered_task_names == TASKS


def test_position_names_are_unique(experiment):
    experiment.add_position("aspen", _state())

    with pytest.raises(ValueError, match="aspen"):
        experiment.add_position("aspen", _state())


def test_a_position_saves_under_positions(experiment, tmp_path):
    position = experiment.add_position("aspen", _state())

    assert experiment.item_path(position) == tmp_path / "vol" / "positions" / "aspen"


def test_the_imaging_pose_falls_back_to_the_milling_pose():
    position = VolumePosition(name="aspen", poses={MILL_POSE: _state(1e-3)})

    assert position.pose(SEM_POSE) is position.poses[MILL_POSE]
    with pytest.raises(KeyError, match="MILL"):
        VolumePosition(name="birch").pose(MILL_POSE)


def test_each_slice_moves_the_poi_down_the_face():
    position = VolumePosition(name="aspen", initial_poi=Point(1e-6, 2e-6))

    poi = position.slice_poi(3, 100e-9)

    assert (poi.x, poi.y) == pytest.approx((1e-6, 2e-6 - 300e-9))
    assert (position.initial_poi.x, position.initial_poi.y) == (1e-6, 2e-6)


def test_a_slice_is_done_when_every_task_completed_for_it():
    position = VolumePosition(name="aspen")
    _ran(position, TASKS[0], 0, TaskStatus.Completed)
    _ran(position, TASKS[1], 0, TaskStatus.Failed)
    _ran(position, TASKS[0], 1, TaskStatus.Completed)
    _ran(position, TASKS[1], 1, TaskStatus.Completed)

    assert not position.iteration_completed(0, TASKS)
    assert position.iteration_completed(1, TASKS)
    assert not position.iteration_completed(2, TASKS)


def test_the_protocol_round_trips_its_configs():
    protocol = VolumeProtocol.default()
    protocol.task_config[TASKS[0]].milling.stages[0].pattern.width = 20e-6

    back = VolumeProtocol.from_dict(protocol.to_dict())

    assert back.to_dict() == protocol.to_dict()
    assert isinstance(back.task_config[TASKS[0]], VolumeFIBMillingTaskConfig)
