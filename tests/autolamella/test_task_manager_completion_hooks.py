"""ITEM_COMPLETED and EXPERIMENT_COMPLETED: when a unit of work is finished.

AutoLamella's item is a lamella.

Both fire on the *transition* into completeness, so a run that re-touches finished
work stays quiet. Completion is driven through the real predicate — a workflow config
with required tasks, and lamellae whose task_history satisfies it — rather than by
stubbing, so these break if the predicate's inputs change.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookContext, HookEvent, HookManager

REQUIRED_TASKS = ["MillTrench", "MillUndercut"]


@pytest.fixture
def recorder() -> "tuple[HookManager, List[HookContext]]":
    fired: List[HookContext] = []
    manager = HookManager()
    manager.register(
        FunctionHook(name="recorder", events=list(HookEvent), callback=fired.append)
    )
    return manager, fired


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    """An experiment whose workflow requires two tasks of every lamella."""
    exp = Experiment(path=tmp_path, name="test-exp")
    # task_protocol is None until one is loaded, so build one
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=name, supervise=False, required=True)
                for name in REQUIRED_TASKS
            ]
        )
    )
    return exp


def _manager(experiment: Experiment, hook_manager: HookManager) -> TaskManager:
    class _NoMicroscope:
        fm = None

    return TaskManager(
        microscope=_NoMicroscope(),
        experiment=experiment,
        parent_ui=None,
        hook_manager=hook_manager,
    )


def _add_lamella(experiment: Experiment, name: str, completed: List[str]) -> Lamella:
    lamella = Lamella(path=experiment.path, number=len(experiment.positions) + 1, petname=name)
    for task_name in completed:
        lamella.task_history.append(AutoLamellaTaskState(name=task_name))
    experiment.positions.append(lamella)
    return lamella


def _events(fired: List[HookContext]) -> List[str]:
    return [c.event for c in fired]


# ---------------------------------------------------------------------------
# item_completed (an item is a lamella here)
# ---------------------------------------------------------------------------

def test_fires_when_the_last_required_task_lands(experiment, recorder):
    hook_manager, fired = recorder
    lamella = _add_lamella(experiment, "lam-1", completed=["MillTrench"])
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    # not finished yet: one required task outstanding
    manager._maybe_fire_lamella_completed(lamella, "MillTrench")
    assert _events(fired) == []

    lamella.task_history.append(AutoLamellaTaskState(name="MillUndercut"))
    manager._maybe_fire_lamella_completed(lamella, "MillUndercut")

    assert _events(fired) == ["item_completed"]
    assert fired[0].item_name == lamella.name
    assert fired[0].task_name == "MillUndercut"


def test_fires_once_not_on_every_later_task(experiment, recorder):
    hook_manager, fired = recorder
    lamella = _add_lamella(experiment, "lam-1", completed=REQUIRED_TASKS)
    manager = _manager(experiment, hook_manager)
    manager._completed_lamella = set()  # pretend it finished during this run

    manager._maybe_fire_lamella_completed(lamella, "MillUndercut")
    manager._maybe_fire_lamella_completed(lamella, "SomeExtraTask")

    assert _events(fired) == ["item_completed"]


def test_a_lamella_already_finished_before_the_run_stays_quiet(experiment, recorder):
    """Re-running a task on finished work must not re-announce it."""
    hook_manager, fired = recorder
    lamella = _add_lamella(experiment, "lam-1", completed=REQUIRED_TASKS)
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    manager._maybe_fire_lamella_completed(lamella, "MillTrench")

    assert _events(fired) == []


def test_each_lamella_fires_for_itself(experiment, recorder):
    hook_manager, fired = recorder
    lam1 = _add_lamella(experiment, "lam-1", completed=["MillTrench"])
    lam2 = _add_lamella(experiment, "lam-2", completed=["MillTrench"])
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    for lamella in (lam1, lam2):
        lamella.task_history.append(AutoLamellaTaskState(name="MillUndercut"))
        manager._maybe_fire_lamella_completed(lamella, "MillUndercut")

    assert [c.item_name for c in fired] == [lam1.name, lam2.name]


# ---------------------------------------------------------------------------
# experiment_completed
# ---------------------------------------------------------------------------

def test_fires_when_the_last_outstanding_lamella_finishes(experiment, recorder):
    hook_manager, fired = recorder
    _add_lamella(experiment, "lam-1", completed=REQUIRED_TASKS)
    lam2 = _add_lamella(experiment, "lam-2", completed=["MillTrench"])
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    manager._maybe_fire_experiment_completed()
    assert _events(fired) == []  # lam-2 still outstanding

    lam2.task_history.append(AutoLamellaTaskState(name="MillUndercut"))
    manager._maybe_fire_experiment_completed()

    assert _events(fired) == ["experiment_completed"]
    assert fired[0].item_name == experiment.name


def test_does_not_refire_for_an_already_complete_experiment(experiment, recorder):
    hook_manager, fired = recorder
    _add_lamella(experiment, "lam-1", completed=REQUIRED_TASKS)
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    manager._maybe_fire_experiment_completed()
    manager._maybe_fire_experiment_completed()

    assert _events(fired) == []


def test_an_empty_experiment_is_not_a_completed_one(experiment, recorder):
    """all([]) is True, which would declare an experiment with no lamellae finished."""
    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)
    manager._snapshot_completion()

    manager._maybe_fire_experiment_completed()

    assert _events(fired) == []


def test_a_workflow_requiring_nothing_completes_nothing(tmp_path, recorder):
    """is_completed() is vacuously True against an empty required-task list, so a
    protocol that failed to load would otherwise announce every lamella as finished."""
    hook_manager, fired = recorder
    exp = Experiment(path=tmp_path, name="no-protocol")  # task_protocol is None
    lamella = _add_lamella(exp, "lam-1", completed=[])
    manager = _manager(exp, hook_manager)
    manager._snapshot_completion()

    manager._maybe_fire_lamella_completed(lamella, "MillTrench")
    manager._maybe_fire_experiment_completed()

    assert _events(fired) == []


# ---------------------------------------------------------------------------
# through a real run
# ---------------------------------------------------------------------------

def test_a_cancelled_run_does_not_complete_the_experiment(experiment, recorder):
    """An aborted run establishes nothing about the experiment, even if the work
    happens to be finished."""
    hook_manager, fired = recorder
    _add_lamella(experiment, "lam-1", completed=REQUIRED_TASKS)
    manager = _manager(experiment, hook_manager)
    manager._completed_lamella = set()
    manager.stop()

    manager.run(task_names=["MillTrench"], required_lamella=[])

    assert "experiment_completed" not in _events(fired)
    assert _events(fired)[-1] == "workflow_cancelled"
