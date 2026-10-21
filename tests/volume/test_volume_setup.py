"""A position is set up once, through the shared proposals and milling loop,
before its slices: the setup leaves the poses and references they align to."""

import pytest

from fibsem.applications.volume.manager import VolumeTaskManager
from fibsem.applications.volume.position import (
    FIB_REFERENCE,
    SEM_POSE,
    SEM_REFERENCE,
)
from fibsem.applications.volume.tasks import VolumeSetupMillingTask
from fibsem.structures import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    MILLING,
    POINT_OF_INTEREST,
    STATE,
    DecisionOutcome,
)
from fibsem.workflows.status import TaskStatus

SET_UP_MILLING = "Setup FIB Milling Position"
SET_UP_IMAGING = "Setup SEM Acquisition"
MILL = "Volume FIB Milling"
IMAGE = "Volume SEM Acquisition"
TASKS = [SET_UP_MILLING, SET_UP_IMAGING, MILL, IMAGE]


@pytest.fixture
def unset(experiment, microscope):
    """Positions as added: a milling pose and a point, no references."""
    for name in (SET_UP_MILLING, SET_UP_IMAGING):
        experiment.task_protocol.task_config[name].attention = Attention.automated
    return [
        experiment.add_position(
            name, microscope.get_microscope_state(), Point(0, 1e-6), iterations=2
        )
        for name in ("aspen", "birch")
    ]


def _run(microscope, experiment) -> VolumeTaskManager:
    manager = VolumeTaskManager(microscope=microscope, experiment=experiment)
    manager.execute(TASKS)
    return manager


def _history(position):
    return [(s.name, s.iteration, s.status) for s in position.task_history]


def test_a_position_is_set_up_once_and_then_sliced(
    microscope, experiment, unset, milled
):
    manager = _run(microscope, experiment)

    aspen = unset[0]
    done = TaskStatus.Completed
    assert _history(aspen) == [
        (SET_UP_MILLING, None, done),
        (SET_UP_IMAGING, None, done),
        (MILL, 0, done),
        (IMAGE, 0, done),
        (MILL, 1, done),
        (IMAGE, 1, done),
    ]
    folder = experiment.position_path(aspen)
    assert (folder / FIB_REFERENCE).exists() and (folder / SEM_REFERENCE).exists()
    assert {MILLING_POSE, SEM_POSE} <= set(aspen.poses)
    # trench, fiducial and face, then a slice each, on both positions
    assert len(milled) == 2 * (3 + 2)
    face = milled[2].stages[0].pattern.point
    assert (face.x, face.y) == (aspen.poi.x, aspen.poi.y)
    kinds = {p.kind for p in aspen.proposals}
    assert kinds == {STATE, MILLING, POINT_OF_INTEREST, ALIGNMENT_AREA}
    assert all(p.current.outcome is DecisionOutcome.Unreviewed for p in aspen.proposals)
    assert manager._completion_message() == (
        "Volume run complete: 2 positions set up, 4 slices on 2 positions."
    )


def test_a_run_again_does_not_set_a_position_up_twice(
    microscope, experiment, unset, milled
):
    _run(microscope, experiment)
    unset[0].iterations = 3

    _run(microscope, experiment)

    names = [s.name for s in unset[0].task_history]
    assert names.count(SET_UP_MILLING) == 1
    assert _history(unset[0])[-2:] == [
        (MILL, 2, TaskStatus.Completed),
        (IMAGE, 2, TaskStatus.Completed),
    ]


def test_a_failed_setup_stops_its_position_and_not_the_others(
    microscope, experiment, unset, milled, monkeypatch
):
    real = VolumeSetupMillingTask._place_point_of_interest

    def place(task):
        if task.position.name == "birch":
            raise RuntimeError("no face here")
        real(task)

    monkeypatch.setattr(VolumeSetupMillingTask, "_place_point_of_interest", place)

    manager = _run(microscope, experiment)

    aspen, birch = unset
    assert aspen.iteration_completed(1, [MILL, IMAGE])
    assert _history(birch)[0] == (SET_UP_MILLING, None, TaskStatus.Failed)
    assert not any(
        i.item_name == "birch" and i.iteration is not None for i in manager.queue.items
    )
