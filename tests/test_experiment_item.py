"""What the engine reads from any item, on an item that is not a lamella.

A volume position or a grid answers the same questions as a lamella from the
same fields, so these run against a plain item with nothing else on it.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

from fibsem.workflows.item import ExperimentItem
from fibsem.workflows.status import TaskStatus


def _run(name: str, status: TaskStatus, task_id: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, status=status, status_message="", task_id=task_id)


@dataclass
class PlainItem(ExperimentItem):
    name: str
    id: str = "item-1"
    description: str = ""
    task_state: Any = None
    task_history: List[Any] = field(default_factory=list)
    proposals: list = field(default_factory=list)


def test_only_completed_runs_count_as_done():
    item = PlainItem(
        "p1",
        task_history=[
            _run("Mill", TaskStatus.Completed, "a"),
            _run("Image", TaskStatus.Failed, "b"),
        ],
    )

    assert item.completed_tasks == ["Mill"]
    assert item.has_completed_task("Mill")
    assert not item.has_completed_task("Image")


def test_a_failed_rerun_is_not_the_latest_completion():
    item = PlainItem(
        "p1",
        task_history=[
            _run("Mill", TaskStatus.Completed, "a"),
            _run("Mill", TaskStatus.Failed, "b"),
        ],
    )

    assert item.has_completed_task("Mill")
    assert not item.latest_run_completed("Mill")


def test_a_decision_moves_the_frozen_run_and_the_live_one():
    live = _run("Mill", TaskStatus.AwaitingDecision, "a")
    item = PlainItem(
        "p1",
        task_state=live,
        task_history=[_run("Mill", TaskStatus.AwaitingDecision, "a")],
    )
    assert item.is_awaiting_decision("Mill")

    item.set_task_status("Mill", TaskStatus.Completed, "accepted")

    assert item.task_history[-1].status is TaskStatus.Completed
    assert live.status is TaskStatus.Completed
    assert live.status_message == "accepted"
    assert not item.is_awaiting_decision("Mill")
