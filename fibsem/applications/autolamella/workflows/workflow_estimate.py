"""How long a workflow will take, and when it will finish.

The arithmetic behind the pre-flight dialog, kept out of it: everything here is plain
data, so it can be checked against a real experiment without a microscope or a widget.

Per-task durations come from each config's ``estimated_duration`` (see
:mod:`fibsem.timing`). This module's job is the workflow-level assembly -- which is not a
sum, because a scheduled task makes the workflow *hold* until its start time, and quoting
``now + Σ`` would be wrong by the length of that wait.

See FIB-666.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment


@dataclass(frozen=True)
class TaskEstimate:
    """One row of the breakdown: a task, across every lamella it will run on."""

    name: str
    lamella_count: int
    seconds: float
    supervised: bool
    scheduled_at: Optional[datetime] = None
    hold_seconds: float = 0.0
    """How long the workflow waits before this task, for its own schedule.

    Per task rather than only in the total: with two scheduled tasks, quoting the
    workflow's whole hold against each of them says the same wrong number twice.
    """


@dataclass(frozen=True)
class WorkflowEstimate:
    """What a workflow will cost, and what it will wait for.

    ``work_seconds`` is machine time and includes the supervised tasks: a supervised
    task still mills, images and moves, and only the pause for a human is unbounded.
    Dropping the whole task would quietly understate the total by the work it does --
    so the wait is reported separately, as a count, rather than by omitting the task.
    """

    tasks: List[TaskEstimate] = field(default_factory=list)
    lamella_names: List[str] = field(default_factory=list)
    work_seconds: float = 0.0
    hold_seconds: float = 0.0
    started_at: Optional[datetime] = None
    expected_finish: Optional[datetime] = None

    @property
    def supervised_tasks(self) -> List[TaskEstimate]:
        """Rows that will pause for a human, in workflow order."""
        return [t for t in self.tasks if t.supervised]

    @property
    def scheduled_tasks(self) -> List[TaskEstimate]:
        """Rows that will hold until a wall-clock time, in workflow order."""
        return [t for t in self.tasks if t.scheduled_at is not None]

    @property
    def step_count(self) -> int:
        """Queue items: one per (task, lamella) pair, matching build_from_matrix."""
        return sum(t.lamella_count for t in self.tasks)

    @property
    def waits_for_a_human(self) -> bool:
        return any(t.supervised for t in self.tasks)


def _as_naive_local(when: Optional[datetime]) -> Optional[datetime]:
    """Normalise to naive-local, the way the workflow loop does.

    Times are naive-local throughout, but a hand-edited or externally-produced protocol
    may carry a tz-aware ``scheduled_at``; ``_wait_until_scheduled`` normalises it for
    the same reason, so that comparing against ``datetime.now()`` cannot raise.
    """
    if when is None or when.tzinfo is None:
        return when
    return when.astimezone().replace(tzinfo=None)


def estimate_workflow(
    experiment: "Experiment",
    task_names: List[str],
    lamella_names: List[str],
    now: Optional[datetime] = None,
) -> WorkflowEstimate:
    """Estimate a workflow of ``task_names`` over ``lamella_names``.

    Walks the queue in the order it will actually run -- task-outer, lamella-inner,
    matching ``TaskQueue.build_from_matrix`` -- carrying a clock, so a scheduled task
    pushes everything after it later rather than being averaged into a total.

    A lamella with no configuration for a task contributes nothing and is not counted
    in that row: the queue would still hold the item, but it has no duration to offer
    and inventing one would be worse than omitting it.

    Args:
        experiment: the experiment holding the per-lamella task configs and the protocol
        task_names: tasks to run, in queue order
        lamella_names: lamellae to run them on
        now: the clock to start from; defaults to the current time

    Returns:
        WorkflowEstimate: the per-task breakdown and the workflow totals.
    """
    clock = now if now is not None else datetime.now()
    started_at = clock

    workflow_config = experiment.task_protocol.workflow_config
    lamellae = [
        lam for lam in experiment.positions if lam.name in set(lamella_names)
    ]

    rows: List[TaskEstimate] = []
    work_seconds = 0.0
    hold_seconds = 0.0

    for task_name in task_names:
        seconds = 0.0
        count = 0
        for lamella in lamellae:
            config = lamella.task_config.get(task_name)
            if config is None:
                continue
            seconds += config.estimated_duration
            count += 1

        scheduled_at = _as_naive_local(workflow_config.get_scheduled_at(task_name))
        held = 0.0
        if scheduled_at is not None and scheduled_at > clock:
            held = (scheduled_at - clock).total_seconds()
            hold_seconds += held
            clock = scheduled_at
        clock += timedelta(seconds=seconds)
        work_seconds += seconds

        rows.append(
            TaskEstimate(
                name=task_name,
                lamella_count=count,
                seconds=seconds,
                supervised=workflow_config.get_supervision(task_name),
                scheduled_at=scheduled_at,
                hold_seconds=held,
            )
        )

    return WorkflowEstimate(
        tasks=rows,
        lamella_names=list(lamella_names),
        work_seconds=work_seconds,
        hold_seconds=hold_seconds,
        started_at=started_at,
        expected_finish=clock,
    )
