"""The volume tasks run one slice of one position on the demo microscope."""

import os
import threading

import pytest

from fibsem import acquire, utils
from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.position import FIB_REFERENCE, SEM_REFERENCE
from fibsem.applications.volume.tasks import (
    VolumeFIBMillingTask,
    run_volume_task,
)
from fibsem.applications.volume.tasks import milling as milling_module
from fibsem.structures import BeamType, ImageSettings, Point
from fibsem.workflows.status import TaskStatus

MILL = "Volume FIB Milling"
IMAGE = "Volume SEM Acquisition"


class FakeManager:
    """What a task reads off the manager: the token, should_abort, hooks."""

    def __init__(self) -> None:
        self.abort_token = threading.Event()
        self.hook_manager = None

    @property
    def should_abort(self) -> bool:
        return self.abort_token.is_set()


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    return microscope


@pytest.fixture
def experiment(tmp_path):
    return VolumeExperiment.create(tmp_path, "vol")


@pytest.fixture
def position(experiment, microscope):
    position = experiment.add_position(
        "aspen", microscope.get_microscope_state(), Point(0, 1e-6), iterations=3
    )
    folder = experiment.position_path(position)
    folder.mkdir(parents=True)
    for beam, filename in (
        (BeamType.ION, FIB_REFERENCE),
        (BeamType.ELECTRON, SEM_REFERENCE),
    ):
        settings = ImageSettings(beam_type=beam, hfw=50e-6, resolution=(256, 170))
        acquire.acquire_image(microscope, settings).save(str(folder / filename))
    return position


@pytest.fixture
def milled(monkeypatch):
    """The configs the milling task hands to the mill."""
    configs = []
    real = milling_module.run_milling_task

    def run(microscope, config, **kwargs):
        configs.append(config)
        return real(microscope, config, **kwargs)

    monkeypatch.setattr(milling_module, "run_milling_task", run)
    return configs


def test_a_slice_is_milled_at_its_point_of_interest(
    microscope, experiment, position, milled
):
    experiment.task_protocol.task_config[MILL].slice_thickness = 100e-9

    run_volume_task(microscope, MILL, experiment, position, iteration=2)

    (config,) = milled
    point = config.stages[0].pattern.point
    assert (point.x, point.y) == pytest.approx((0, 1e-6 - 200e-9))
    saved = experiment.task_protocol.task_config[MILL].milling.stages[0].pattern.point
    assert (saved.x, saved.y) == (0, 0)
    state = position.task_history[-1]
    assert (state.name, state.iteration, state.status) == (
        MILL,
        2,
        TaskStatus.Completed,
    )


def test_each_slice_writes_into_its_own_folder(microscope, experiment, position):
    for iteration in (0, 1):
        run_volume_task(microscope, IMAGE, experiment, position, iteration=iteration)

    folder = experiment.position_path(position) / IMAGE
    assert sorted(os.listdir(folder)) == ["0000", "0001"]
    outputs = position.task_history[-1].outputs["slice"]
    assert outputs == [f"{IMAGE}/0001/slice_0001_eb.tif"]
    assert (experiment.position_path(position) / outputs[0]).exists()


def test_autofocus_runs_on_the_slices_it_is_due(
    microscope, experiment, position, monkeypatch
):
    experiment.task_protocol.task_config[IMAGE].autofocus_every = 2
    focused = []
    monkeypatch.setattr(
        microscope, "auto_focus", lambda beam, reduced_area=None: focused.append(beam)
    )

    for iteration in range(3):
        run_volume_task(microscope, IMAGE, experiment, position, iteration=iteration)

    assert focused == [BeamType.ELECTRON, BeamType.ELECTRON]


def test_a_position_with_no_reference_fails_before_milling(
    microscope, experiment, position, milled
):
    os.remove(experiment.position_path(position) / FIB_REFERENCE)

    with pytest.raises(RuntimeError, match="no alignment reference"):
        run_volume_task(microscope, MILL, experiment, position, iteration=0)

    assert milled == []
    assert position.task_history[-1].status is TaskStatus.Failed


def test_a_task_needs_a_slice(microscope, experiment, position):
    with pytest.raises(ValueError, match="iteration"):
        run_volume_task(microscope, MILL, experiment, position)


def test_a_stop_is_recorded_as_cancelled(microscope, experiment, position, milled):
    manager = FakeManager()
    manager.abort_token.set()
    task = VolumeFIBMillingTask(
        microscope,
        experiment.task_protocol.task_config[MILL],
        position,
        experiment,
        task_manager=manager,
    )
    task.iteration = 0

    with pytest.raises(InterruptedError):
        task.run()

    assert milled == []
    assert position.task_history[-1].status is TaskStatus.Cancelled
