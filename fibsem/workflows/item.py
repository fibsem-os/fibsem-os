"""What a task runs on: one item of an experiment.

A lamella, a grid and a volume position are items. Each has an ``id``, a
``name``, a ``description``, a ``task_state`` (the run in hand, mutated in
place, never replaced), a ``task_history`` (each run's state, frozen when it
ended) and ``proposals`` (what was proposed for it and what was decided, oldest
first, each saying who asked it). :class:`ExperimentItem` answers what the
engine asks of any of them from those fields; an application's item classes
declare the fields and add their own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fibsem.acting import Actor, ActorKind
from fibsem.structures import FibsemRectangle, MicroscopeState, Point
from fibsem.workflows.proposals import (
    Proposal,
    current_proposal,
    current_proposals,
    record,
)
from fibsem.workflows.status import TaskStatus

# The pose an item is milled at.
MILLING_POSE = "MILLING"


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
    proposals: List[Proposal]

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

    def set_task_status(
        self, task_name: str, status: TaskStatus, message: str = ""
    ) -> None:
        """Move the latest run of ``task_name`` to ``status``, in the history
        and on the live task_state when that is the same run.

        The two are separate objects (task_history holds a copy frozen at the
        end of the run), so a status that changes after the run has to be
        written to both.
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

    def proposals_from(self, task_name: str) -> List[Proposal]:
        """Everything ``task_name`` asked about this item, oldest first."""
        return [p for p in self.proposals if p.asked_by_task(task_name)]

    def asking_tasks(self) -> List[str]:
        """The tasks that asked something about this item, in the order they
        first did."""
        names: List[str] = []
        for p in self.proposals:
            asked_by = p.asked_by
            if asked_by is not None and asked_by.kind is ActorKind.task:
                if asked_by.name not in names:
                    names.append(asked_by.name)
        return names

    def proposal(
        self, task_name: str, kind: Optional[str] = None
    ) -> Optional[Proposal]:
        """The current proposal from ``task_name``: the last one it made, or
        the last of ``kind``. None when it never proposed."""
        return current_proposal(self.proposals_from(task_name), kind)

    def current_proposals(self, task_name: str) -> List[Proposal]:
        """Every current proposal from ``task_name``: the last of each kind."""
        return current_proposals(self.proposals_from(task_name))

    def record_proposal(self, task_name: str, proposal: Proposal) -> Proposal:
        """Put ``proposal`` on the record as the current one from ``task_name``."""
        proposal.asked_by = Actor.task(task_name)
        return record(self.proposals, proposal)


class SiteItem(ExperimentItem):
    """An item at a site on the sample: a lamella or a volume position. Its
    ``poses`` by name, a point of interest ``poi`` and an ``alignment_area``, in
    fractions of the frame. What a task that works at a site reads."""

    poses: Dict[str, MicroscopeState]
    poi: Point
    alignment_area: FibsemRectangle

    def pose(self, name: str) -> MicroscopeState:
        """The pose ``name``. Raises KeyError when the item has none."""
        if self.poses.get(name) is None:
            raise KeyError(f"{self.name} has no {name} pose.")
        return self.poses[name]

    @property
    def milling_pose(self) -> Optional[MicroscopeState]:
        return self.poses.get(MILLING_POSE)

    @milling_pose.setter
    def milling_pose(self, value: MicroscopeState) -> None:
        if not isinstance(value, MicroscopeState):
            raise TypeError("Milling pose must be a MicroscopeState instance.")
        self.poses[MILLING_POSE] = value
