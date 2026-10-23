"""How long a run will take, and when it will finish.

One walk over queue items, carrying a clock: what the pre-flight quotes before a
run, what the timeline shows during one, and what adding work to it costs. It is
not a sum, because a scheduled task makes the run *hold* until its start time,
and quoting ``now + Σ`` would be wrong by the length of that wait.

Everything here is plain data, so it can be checked against a real experiment
without a microscope or a widget. Per-step durations come from each config's
``estimated_duration`` (see :mod:`fibsem.timing`); who has to be there while the
run goes is not an estimate and is not here (``fibsem/workflows/attendance.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from fibsem.workflows.queue import WorkItem

# what a step costs, by (item name, task name); None for a step with nothing to offer
StepSeconds = Callable[[str, str], Optional[float]]


@dataclass(frozen=True)
class TaskEstimate:
    """One row of the breakdown: a task, across every item it will run on."""

    name: str
    count: int
    seconds: float
    scheduled_at: Optional[datetime] = None
    hold_seconds: float = 0.0
    """How long the run waits before this task, for its own schedule.

    Per task rather than only in the total: with two scheduled tasks, quoting the
    run's whole hold against each of them says the same wrong number twice.
    """


@dataclass(frozen=True)
class RunEstimate:
    """What a run will cost, and what it will wait for.

    ``work_seconds`` is machine time and includes the supervised tasks: a supervised
    task still mills, images and moves, and only the pause for a human is unbounded.
    """

    tasks: List[TaskEstimate] = field(default_factory=list)
    item_names: List[str] = field(default_factory=list)
    work_seconds: float = 0.0
    hold_seconds: float = 0.0
    started_at: Optional[datetime] = None
    expected_finish: Optional[datetime] = None

    @property
    def scheduled_tasks(self) -> List[TaskEstimate]:
        """Rows that will hold until a wall-clock time, in run order."""
        return [t for t in self.tasks if t.scheduled_at is not None]

    @property
    def step_count(self) -> int:
        """Queue items: one per (task, item) pair."""
        return sum(t.count for t in self.tasks)


def _as_naive_local(when: Optional[datetime]) -> Optional[datetime]:
    """Normalise to naive-local, the way the workflow loop does.

    Times are naive-local throughout, but a hand-edited or externally-produced protocol
    may carry a tz-aware ``scheduled_at``; ``_wait_until_scheduled`` normalises it for
    the same reason, so that comparing against ``datetime.now()`` cannot raise.
    """
    if when is None or when.tzinfo is None:
        return when
    return when.astimezone().replace(tzinfo=None)


def _step(
    item: "WorkItem",
    clock: datetime,
    seconds_for: "Callable[[WorkItem], Optional[float]]",
    schedule: Dict[str, datetime],
) -> Tuple[datetime, float, Optional[float]]:
    """One pending item on the clock: the clock after it, the hold before it,
    and its seconds (None when it has nothing to offer).

    The hold fires per item, as ``_wait_until_scheduled`` does: a scheduled task's
    hold lands on the first of its items and the rest find the time passed.
    """
    held = 0.0
    scheduled_at = _as_naive_local(schedule.get(item.task_name))
    if scheduled_at is not None and scheduled_at > clock:
        held = (scheduled_at - clock).total_seconds()
        clock = scheduled_at
    seconds = seconds_for(item)
    if seconds is not None:
        clock += timedelta(seconds=seconds)
    return clock, held, seconds


def estimate_run(
    pairs: Sequence[Tuple[str, str]],
    step_seconds: StepSeconds,
    schedule: Optional[Dict[str, datetime]] = None,
    now: Optional[datetime] = None,
) -> RunEstimate:
    """Estimate a run of ``pairs``, ``(item, task)`` in the order the queue
    will hold them, before it starts.

    The queue walk over the planned items, rolled up into a row per task in the
    order the tasks first appear. A step with no estimate contributes nothing and
    is not counted in its row: the queue would still hold the item, but it has no
    duration to offer and inventing one would be worse than omitting it.
    """
    from fibsem.workflows.queue import WorkItem

    clock = now if now is not None else datetime.now()
    started_at = clock
    schedule = schedule or {}

    def seconds_for(item: "WorkItem") -> Optional[float]:
        return step_seconds(item.item_name, item.task_name)

    rows: Dict[str, Dict[str, float]] = {}
    items: List[str] = []
    for item_name, task_name in pairs:
        if item_name not in items:
            items.append(item_name)
        row = rows.setdefault(task_name, {"count": 0, "seconds": 0.0, "hold": 0.0})
        clock, held, seconds = _step(
            WorkItem(item_name=item_name, task_name=task_name),
            clock,
            seconds_for,
            schedule,
        )
        row["hold"] += held
        if seconds is not None:
            row["count"] += 1
            row["seconds"] += seconds

    tasks = [
        TaskEstimate(
            name=name,
            count=int(row["count"]),
            seconds=row["seconds"],
            scheduled_at=_as_naive_local(schedule.get(name)),
            hold_seconds=row["hold"],
        )
        for name, row in rows.items()
    ]
    return RunEstimate(
        tasks=tasks,
        item_names=items,
        work_seconds=sum(t.seconds for t in tasks),
        hold_seconds=sum(t.hold_seconds for t in tasks),
        started_at=started_at,
        expected_finish=clock,
    )


@dataclass(frozen=True)
class QueueEstimate:
    """What is left of a live queue, and when it will be done.

    The in-run counterpart of :class:`RunEstimate`: the same walk, over what remains
    rather than over what was planned. Completed items contribute nothing -- their
    cost has already been paid, and the finish time therefore converges on its own as
    the run proceeds, without anything being re-learned.
    """

    remaining_seconds: float = 0.0
    """Wall-clock to the finish: the pending work plus any scheduled holds ahead."""

    work_seconds: float = 0.0
    """Machine time only, holds excluded.

    Quoted instead of ``remaining_seconds`` while a supervised task waits for a human:
    that wait is unbounded, so a finish time would be a guess dressed as a fact, while
    the work still to be done is perfectly knowable.
    """

    hold_seconds: float = 0.0
    expected_finish: Optional[datetime] = None

    active_remaining: Optional[float] = None
    """Seconds left on the running task, or None if nothing is running."""

    active_estimate_spent: bool = False
    """Whether the running task has already used up its estimate.

    Not a warning and never rendered as one: a task running long is ordinary variance,
    not an error. It is only how the row knows to stop predicting and report elapsed
    instead --.
    """


def estimate_queue(
    items: "Sequence[WorkItem]",
    seconds_for: "Callable[[WorkItem], Optional[float]]",
    schedule: Optional[Dict[str, datetime]] = None,
    now: Optional[datetime] = None,
    active_elapsed: Optional[float] = None,
) -> QueueEstimate:
    """Estimate what is left of a running queue.

    Walks the items in queue order carrying a clock. Tracked by walking in order
    rather than by task, so an interleaved queue -- which a mid-run "run next" can
    produce -- holds wherever the schedule actually catches it.

    An overrunning task contributes **zero** to the remaining total rather than a
    negative number, so the total can neither go negative nor stall: it slides later by a
    second per second, which is the honest answer once the task's end is unknown.

    Args:
        items: the queue snapshot, in order, including items already run
        seconds_for: per-item estimate; None for an item with nothing to offer
        schedule: ``scheduled_at`` per task name, for the tasks that have one
        now: the clock to walk from; defaults to the current time
        active_elapsed: seconds the running task has been going, excluding any pause for
            a human -- on why the countdown freezes but the elapsed shown on
            the row does not

    Returns:
        QueueEstimate: what remains, and when it lands.
    """
    from fibsem.workflows.status import TaskStatus

    clock = now if now is not None else datetime.now()
    schedule = schedule or {}

    work_seconds = 0.0
    hold_seconds = 0.0
    active_remaining: Optional[float] = None
    active_estimate_spent = False
    for item in items:
        if item.status is TaskStatus.InProgress:
            estimate = seconds_for(item)
            if estimate is not None and active_elapsed is not None:
                active_remaining = max(0.0, estimate - active_elapsed)
                active_estimate_spent = active_elapsed >= estimate
                work_seconds += active_remaining
                clock += timedelta(seconds=active_remaining)
            continue

        if not item.is_pending:
            continue

        clock, held, seconds = _step(item, clock, seconds_for, schedule)
        hold_seconds += held
        if seconds is not None:
            work_seconds += seconds

    return QueueEstimate(
        remaining_seconds=work_seconds + hold_seconds,
        work_seconds=work_seconds,
        hold_seconds=hold_seconds,
        expected_finish=clock,
        active_remaining=active_remaining,
        active_estimate_spent=active_estimate_spent,
    )


@dataclass(frozen=True)
class AdditionEstimate:
    """What appending work to a running queue costs, and what it moves.

    Two figures rather than one, because they can disagree and the disagreement is the
    point: work added ahead of a scheduled hold is absorbed into dead time the run
    was going to spend anyway, so ``work_seconds`` can be an hour while ``delay_seconds``
    is zero. Quoting only the first would overstate the cost; quoting only the second
    would hide that the machine is now busy for an hour it was not.
    """

    work_seconds: float = 0.0
    """Machine time the addition adds. Independent of where it lands in the queue."""

    delay_seconds: float = 0.0
    """How much later the run now finishes. Zero when a hold absorbs the work."""

    finish_before: Optional[datetime] = None
    finish_after: Optional[datetime] = None

    hold_seconds: float = 0.0
    """Scheduled dead time in the queue once the addition is in.

    Zero when nothing is scheduled, which is what lets a caller tell "this work costs no
    delay because a wait absorbed it" from "this work costs no delay because there is
    barely any of it" -- two sentences that must not be swapped.
    """

    priced_count: int = 0
    """How many of the added items had an estimate to offer."""

    total_count: int = 0

    @property
    def is_priced(self) -> bool:
        """Whether there is a number worth showing at all.

        False when nothing added could be priced -- a protocol whose configs cannot be
        estimated, say. The caller shows the addition without the figures rather than
        quoting a confident zero.
        """
        return self.priced_count > 0


def _with_addition(
    items: "Sequence[WorkItem]",
    added: "Sequence[WorkItem]",
    run_next: bool,
) -> "List[WorkItem]":
    """The queue as it would be with ``added`` spliced in.

    Mirrors where ``TaskQueue`` actually puts them: ``run_next`` lands the batch at the
    first pending slot -- ``_pending_start()``, so never ahead of the running item or
    inside history -- and otherwise at the end. The batch keeps the order it is given,
    which is what ``WorkflowRunner.add`` produces by anchoring each item after the last.
    """
    if not run_next:
        return list(items) + list(added)
    start = next((n for n, i in enumerate(items) if i.is_pending), len(items))
    return list(items[:start]) + list(added) + list(items[start:])


def estimate_addition(
    items: "Sequence[WorkItem]",
    pairs: "Sequence",
    seconds_for: "Callable[[WorkItem], Optional[float]]",
    run_next: bool = False,
    schedule: Optional[Dict[str, datetime]] = None,
    now: Optional[datetime] = None,
    active_elapsed: Optional[float] = None,
) -> AdditionEstimate:
    """Estimate what adding ``pairs`` to a running queue would cost.

    The same queue walk twice, over the snapshot as it is and as it would be. Doing it
    by difference rather than by summing the addition is what makes the scheduled-hold
    case come out right: a hold is not a property of the added work, it is a property of
    where the work lands relative to it.

    It also makes the *delay* exact even when the absolute finish is not. Both walks
    treat the running task identically, so whatever ``active_elapsed`` cannot account for
    -- a supervised pause, most of all -- cancels in the subtraction.

    Args:
        items: the live queue snapshot, in order
        pairs: (item_name, task_name) to add, in the order they will be inserted
        seconds_for: per-item estimate; None for an item with nothing to offer
        run_next: True to insert at the front of the pending region, False for the end
        schedule: ``scheduled_at`` per task name, for the tasks that have one
        now: the clock to walk from; defaults to the current time
        active_elapsed: seconds the running task has been going

    Returns:
        AdditionEstimate: the work added, and the delay it causes.
    """
    from fibsem.workflows.queue import WorkItem

    clock = now if now is not None else datetime.now()
    added = [WorkItem(item_name=ln, task_name=tn) for ln, tn in pairs]
    proposed = _with_addition(items, added, run_next)

    before = estimate_queue(items, seconds_for, schedule, clock, active_elapsed)
    after = estimate_queue(proposed, seconds_for, schedule, clock, active_elapsed)

    priced = sum(1 for item in added if seconds_for(item) is not None)
    return AdditionEstimate(
        work_seconds=after.work_seconds - before.work_seconds,
        # Never negative: adding work cannot bring a finish forward, and floating-point
        # dust on two large sums should not render as "-0s".
        delay_seconds=max(
            0.0, (after.expected_finish - before.expected_finish).total_seconds()
        ),
        finish_before=before.expected_finish,
        finish_after=after.expected_finish,
        hold_seconds=after.hold_seconds,
        priced_count=priced,
        total_count=len(pairs),
    )
