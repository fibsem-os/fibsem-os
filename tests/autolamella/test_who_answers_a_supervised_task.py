"""A supervised task asks a person as it runs only while the operator is
attached to answer (a window). Its supervision is the run's experiment's, so a
run with no window reads it the same way and asks nobody."""

import types

import pytest

from fibsem.acting import AGENT, OPERATOR
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    AutoLamellaTask,
    get_task_supervision,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.workflows.attention import Attention

TASK = "Rough Milling"


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment(path=tmp_path, name="who-answers")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name=TASK, required=True, attention=Attention.supervised
        )
    )
    return experiment


def _validate(experiment, deciders):
    manager = TaskManager(
        microscope=None, experiment=experiment, deciders=lambda: deciders
    )
    task = types.SimpleNamespace(task_manager=manager, task_name=TASK)
    return AutoLamellaTask.validate.fget(task)


def test_supervision_is_the_experiments():
    assert get_task_supervision(TASK, None) is False


def test_a_supervised_task_asks_the_attached_operator(experiment):
    assert get_task_supervision(TASK, experiment) is True
    assert _validate(experiment, (OPERATOR,)) is True


@pytest.mark.parametrize("deciders", [(), (AGENT,)], ids=["nobody", "agent"])
def test_without_the_operator_it_asks_nobody_as_it_runs(experiment, deciders):
    # The milling and spot burn loops are asked on a window until they are
    # proposals; an agent answers the task's proposals, not these.
    assert _validate(experiment, deciders) is False


def test_a_task_outside_a_run_asks_nobody():
    task = types.SimpleNamespace(task_manager=None, task_name=TASK)
    assert AutoLamellaTask.validate.fget(task) is False
