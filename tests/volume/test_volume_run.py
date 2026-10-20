"""A volume run with no window: each position cuts and images its slices in turn,
and a stopped run resumes where it stopped."""

import os

import pytest

from fibsem.applications.volume import manager as manager_module
from fibsem.applications.volume.manager import VolumeTaskManager
from fibsem.applications.volume.position import FIB_REFERENCE
from fibsem.applications.volume.runs import check_volume_run
from fibsem.workflows.runner import ITEMS, RunRequest, app_runs
from fibsem.workflows.status import TaskStatus

from .conftest import set_up_position

MILL = "Volume FIB Milling"
IMAGE = "Volume SEM Acquisition"
TASKS = [MILL, IMAGE]


@pytest.fixture
def positions(experiment, microscope):
    return [
        set_up_position(experiment, microscope, name, iterations=3)
        for name in ("aspen", "birch")
    ]


def _run(microscope, experiment) -> VolumeTaskManager:
    manager = VolumeTaskManager(microscope=microscope, experiment=experiment)
    manager.execute(TASKS)
    return manager


def _stop_after(monkeypatch, position_name: str, task_name: str, iteration: int):
    """Stop the run once ``task_name`` finishes slice ``iteration`` of the
    position: between milling and imaging, say."""
    real = manager_module.run_volume_task

    def run(microscope, name, experiment, position, task_manager=None):
        task = real(microscope, name, experiment, position, task_manager=task_manager)
        if (position.name, name, task.iteration) == (
            position_name,
            task_name,
            iteration,
        ):
            task_manager.stop()
        return task

    monkeypatch.setattr(manager_module, "run_volume_task", run)
    return real


def _slices(position):
    return [(s.name, s.iteration, s.status) for s in position.task_history]


def test_each_position_cuts_and_images_every_slice(
    microscope, experiment, positions, milled
):
    manager = _run(microscope, experiment)

    for position in positions:
        assert _slices(position) == [
            (task, n, TaskStatus.Completed) for n in range(3) for task in TASKS
        ]
    # the positions take turns, a slice each
    order = [(i.item_name, i.iteration) for i in manager.queue.items]
    assert order[:4] == [("aspen", 0), ("aspen", 0), ("birch", 0), ("birch", 0)]
    assert len(milled) == 6
    assert manager._completion_message() == (
        "Volume run complete: 6 slices on 2 positions."
    )


def test_a_stopped_run_resumes_without_cutting_a_slice_twice(
    microscope, experiment, positions, milled, monkeypatch
):
    real = _stop_after(monkeypatch, "aspen", MILL, 1)

    first = _run(microscope, experiment)

    assert first.is_stopped
    aspen = positions[0]
    assert aspen.iteration_completed(0, TASKS)
    assert aspen.iteration_completed(1, [MILL])
    assert not aspen.iteration_completed(1, [IMAGE])

    monkeypatch.setattr(manager_module, "run_volume_task", real)
    second = _run(microscope, experiment)

    # the resume images aspen's cut face first, then goes on
    assert (second.queue.items[0].item_name, second.queue.items[0].task_name) == (
        "aspen",
        IMAGE,
    )
    assert second.queue.items[0].iteration == 1
    for position in positions:
        assert all(position.iteration_completed(n, TASKS) for n in range(3))
        folder = experiment.position_path(position) / MILL
        assert sorted(os.listdir(folder)) == ["0000", "0001", "0002"]
    assert len(milled) == 6


def test_a_failed_slice_stops_its_position_and_not_the_others(
    microscope, experiment, positions, milled
):
    os.remove(experiment.position_path(positions[1]) / FIB_REFERENCE)

    manager = _run(microscope, experiment)

    aspen, birch = positions
    assert all(aspen.iteration_completed(n, TASKS) for n in range(3))
    # the uncut face is not imaged: the image needs the cut
    assert _slices(birch) == [(MILL, 0, TaskStatus.Failed)]
    assert [
        (i.task_name, i.status) for i in manager.queue.items if i.item_name == "birch"
    ] == [
        (MILL, TaskStatus.Failed),
        (IMAGE, TaskStatus.Skipped),
    ]
    assert "1 task failed" in manager._completion_message()


def test_the_runner_runs_volume_experiments(experiment, positions):
    rules = app_runs("volume").items

    ok = rules.check(experiment, None, RunRequest(kind=ITEMS, task_names=tuple(TASKS)))
    unknown = check_volume_run(
        experiment, None, RunRequest(kind=ITEMS, task_names=(MILL,), names=("oak",))
    )

    assert ok.started and ok.request.names == ("aspen", "birch")
    assert not unknown.started and "oak" in unknown.reason
