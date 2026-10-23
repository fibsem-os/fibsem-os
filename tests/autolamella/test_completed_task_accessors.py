"""A task in the history is not necessarily a task that succeeded.

FIB-490 put failed and cancelled outcomes into ``task_history``, which is what makes
failures durable. That changes what the history *contains*, so the accessors that
answer "what has this lamella completed?" have to filter on status -- otherwise a
failed task silently counts as done, and the prerequisite gate opens on work that
never succeeded.
"""

from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Lamella,
)
from fibsem.workflows.status import TaskState, TaskStatus
from tests.protocols import planned, task_protocol


@pytest.fixture
def lamella(tmp_path: Path) -> Lamella:
    return Lamella(path=tmp_path / "lam", number=0, petname="test")


@pytest.fixture
def workflow() -> AutoLamellaTaskProtocol:
    """MillUndercut requires MillTrench."""
    return task_protocol(
        planned(name="MillTrench", required=True),
        planned(name="MillUndercut", required=True, requires=["MillTrench"]),
    )


def _finished(name: str, status: TaskStatus) -> TaskState:
    return TaskState(name=name, status=status)


def test_a_failed_task_is_not_a_completed_task(lamella: Lamella) -> None:
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Failed))

    assert lamella.completed_tasks == []
    assert not lamella.has_completed_task("MillTrench")


def test_a_cancelled_task_is_not_a_completed_task(lamella: Lamella) -> None:
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Cancelled))

    assert not lamella.has_completed_task("MillTrench")


def test_a_completed_task_still_counts(lamella: Lamella) -> None:
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Completed))

    assert lamella.completed_tasks == ["MillTrench"]
    assert lamella.has_completed_task("MillTrench")


def test_a_failed_prerequisite_does_not_satisfy_the_gate(
    lamella: Lamella, workflow: AutoLamellaTaskProtocol
) -> None:
    """The blocking case: TaskManager._should_skip gates on has_completed_task, and
    nothing else stops it -- is_failure reads defect.verdict, which only a human
    clicking in the UI ever sets. A failed trench must not license an undercut."""
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Failed))

    requirements = workflow.requirements("MillUndercut")
    assert requirements == ["MillTrench"]
    assert not all(lamella.has_completed_task(req) for req in requirements)


def test_a_failed_required_task_leaves_the_workflow_incomplete(
    lamella: Lamella, workflow: AutoLamellaTaskProtocol
) -> None:
    """Feeds ITEM_COMPLETED / EXPERIMENT_COMPLETED and the is_completed report column."""
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Failed))
    lamella.task_history.append(_finished("MillUndercut", TaskStatus.Completed))

    assert not workflow.workflow_config.is_completed(lamella)
    assert lamella.completed_tasks == ["MillUndercut"]


def test_last_completed_task_skips_a_failed_entry(lamella: Lamella) -> None:
    """Reported as last_completed / last_completed_at in the experiment summary, so a
    failure landing last would otherwise be presented as the lamella's latest progress."""
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Completed))
    lamella.task_history.append(_finished("MillUndercut", TaskStatus.Failed))

    last = lamella.last_completed_task
    assert last is not None
    assert last.name == "MillTrench"


def test_the_history_still_holds_the_failure(lamella: Lamella) -> None:
    """Filtering the accessors must not undo FIB-490 -- the record itself stays."""
    lamella.task_history.append(_finished("MillTrench", TaskStatus.Failed))

    assert len(lamella.task_history) == 1
    assert lamella.task_history[0].status is TaskStatus.Failed
