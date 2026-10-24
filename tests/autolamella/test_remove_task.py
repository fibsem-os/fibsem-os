"""Removing a task removes it from the whole experiment (FIB-1109).

The protocol editor's Remove Task used to delete only the protocol's config, so
the workflow still listed the task, still required it, and each lamella still
held a copy a run would execute.
"""

from pathlib import Path

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTaskConfig,
)
from fibsem.structures.settings import MicroscopeState
from fibsem.workflows.status import TaskState, TaskStatus
from tests.protocols import planned, task_protocol

SETUP = "Setup"
FIDUCIAL = "Fiducial"
ROUGH = "Rough"


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    """Setup -> Fiducial -> Rough, two lamellae; the first has done Fiducial."""
    exp = Experiment(path=tmp_path, name="remove-task")
    exp.task_protocol = task_protocol(
        planned(name=SETUP),
        planned(name=FIDUCIAL, requires=[SETUP]),
        planned(name=ROUGH, requires=[SETUP, FIDUCIAL]),
        task_config=EventedDict(
            {
                name: MillRoughTaskConfig(task_name=name)
                for name in (SETUP, FIDUCIAL, ROUGH)
            }
        ),
    )
    for _ in range(2):
        exp.add_new_lamella(MicroscopeState(), exp.task_protocol.task_config)
    exp.positions[0].task_history.append(
        TaskState(name=FIDUCIAL, status=TaskStatus.Completed)
    )
    return exp


def test_the_task_leaves_the_protocol_the_workflow_and_every_lamella(experiment):
    experiment.remove_task(FIDUCIAL)

    assert FIDUCIAL not in experiment.task_protocol.task_config
    assert experiment.task_protocol.workflow_config.workflow == [SETUP, ROUGH]
    for lamella in experiment.positions:
        assert list(lamella.task_config) == [SETUP, ROUGH]


def test_a_task_that_required_it_no_longer_does(experiment):
    experiment.remove_task(FIDUCIAL)

    protocol = experiment.task_protocol
    assert protocol.requirements(ROUGH) == [SETUP]
    assert protocol.validate() == []


def test_task_history_is_kept(experiment):
    experiment.remove_task(FIDUCIAL)

    assert [t.name for t in experiment.positions[0].task_history] == [FIDUCIAL]


def test_the_removal_survives_a_save_and_load(experiment):
    experiment.remove_task(FIDUCIAL)
    experiment.save(save_protocol=True)

    loaded = Experiment.load(Path(experiment.path) / "experiment.yaml")

    assert FIDUCIAL not in loaded.task_protocol.task_config
    assert loaded.task_protocol.workflow_config.workflow == [SETUP, ROUGH]
    assert all(FIDUCIAL not in p.task_config for p in loaded.positions)
    assert [t.name for t in loaded.positions[0].task_history] == [FIDUCIAL]


def test_removing_a_task_the_experiment_does_not_have_changes_nothing(experiment):
    before = experiment.task_protocol.to_dict()

    experiment.remove_task("Not A Task")

    assert experiment.task_protocol.to_dict() == before
