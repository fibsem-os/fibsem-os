"""What a task runs on: one item of an experiment.

A lamella, a grid and a volume position are items. Each has an ``id``, a
``name``, a ``description``, a ``task_state`` (the run in hand, mutated in
place, never replaced), a ``task_history`` (each run's state, frozen when it
ended) and ``proposals`` (what its tasks proposed and what was decided, keyed
by the producing task's name). :class:`ExperimentItem` answers what the
engine asks of any of them from those fields; an application's item classes
declare the fields and add their own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fibsem.workflows.proposals import (
    Proposal,
    current_proposal,
    current_proposals,
    record,
)
from fibsem.workflows.status import TaskStatus


class ExperimentItem:
    """The task record every item keeps, and what the engine reads from it.

    A mixin, not a dataclass: an item's own dataclass declares the fields, in
    the order its saved form and constructor already use.
    """

    id: str
    name: str
    description: str
    task_state: Any
    task_history: List[Any]
    proposals: Dict[str, List[Proposal]]

    @property
    def completed_tasks(self) -> List[str]:
        """The names of the tasks that have completed on this item.

        Filtered on status: task_history records every terminal outcome, not
        just successes (FIB-490), so an unfiltered read would count a failed
        task as done -- and a failed prerequisite would license the task that
        requires it.
        """
        return [
            task.name
            for task in self.task_history
            if task.status is TaskStatus.Completed
        ]

    def has_completed_task(self, task_name: str) -> bool:
        """Whether any run of ``task_name`` completed."""
        return task_name in self.completed_tasks

    def latest_status(self, task_name: str) -> Optional[TaskStatus]:
        """How the latest run of ``task_name`` ended; None if it never ran."""
        for task in reversed(self.task_history):
            if task.name == task_name:
                return task.status
        return None

    def latest_run_completed(self, task_name: str) -> bool:
        """Whether the latest run of ``task_name`` completed: what a task that
        requires it needs. Not ``has_completed_task``, which an old success
        satisfies after a rerun failed or was rejected."""
        return self.latest_status(task_name) is TaskStatus.Completed

    def is_awaiting_decision(self, task_name: str) -> bool:
        """Whether the latest run of ``task_name`` ended waiting on a decision."""
        return self.latest_status(task_name) is TaskStatus.AwaitingDecision

    def set_task_status(
        self, task_name: str, status: TaskStatus, message: str = ""
    ) -> None:
        """Move the latest run of ``task_name`` to ``status``, in the history
        and on the live task_state when that is the same run.

        The two are separate objects (task_history holds a copy frozen at the
        end of the run), so a status that changes after the run -- a decision
        landing on a task that was awaiting one -- has to be written to both.
        """
        for task in reversed(self.task_history):
            if task.name == task_name:
                task.status = status
                task.status_message = message
                state = self.task_state
                if state is not None and state.task_id == task.task_id:
                    state.status = status
                    state.status_message = message
                return

    def proposal(
        self, task_name: str, kind: Optional[str] = None
    ) -> Optional[Proposal]:
        """The current proposal from ``task_name``: the last one it made, or
        the last of ``kind``. None when it never proposed."""
        return current_proposal(self.proposals.get(task_name), kind)

    def current_proposals(self, task_name: str) -> List[Proposal]:
        """Every current proposal from ``task_name``: the last of each kind."""
        return current_proposals(self.proposals.get(task_name))

    def record_proposal(self, task_name: str, proposal: Proposal) -> Proposal:
        """Put ``proposal`` on the record as the current one from ``task_name``."""
        return record(self.proposals.setdefault(task_name, []), proposal)
