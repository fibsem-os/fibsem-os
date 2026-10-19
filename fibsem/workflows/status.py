"""The status a task runs through, in any application's workflow, and the typed
payloads on ``workflow_status_signal`` that report it.

``WorkflowStatusUpdate`` began as the record nested inside the dict
``workflow_update_signal``'s ``status`` key; that signal is gone (FIB-826), and
the record now rides inside :class:`WorkflowStatusEvent`.

Unlike the other progress signals, there was no vocabulary to design here. ``status``
has always been a real ``TaskStatus`` on the wire; what was missing was a
declared shape around it.
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from psygnal import evented

from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.util.durations import format_duration_precise
from fibsem.util.timestamps import format_time, now, to_datetime

if TYPE_CHECKING:
    from fibsem.workflows.queue import WorkItem


class TaskStatus(Enum):
    NotStarted = auto()
    InProgress = auto()
    Completed = auto()
    Failed = auto()
    Skipped = auto()
    Cancelled = auto()  # aborted by the user (Stop), distinct from a genuine Failure
    Removed = auto()  # pulled from the queue by the user before it ran


@evented
@dataclass
class TaskState:
    """One run of a task on one item: the item's live ``task_state`` while it
    runs, frozen into ``task_history`` when it ends."""

    name: str = ""
    step: str = ""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    # The item the run is on (a lamella, a grid, a volume position).
    item_id: str = ""
    # Which iteration of the item's tasks this run belongs to, for an item that
    # repeats its tasks; None for one that does not.
    iteration: Optional[int] = None
    # When the run started and ended: aware datetimes, written as ISO 8601 with
    # their offset (FIB-1197). Files written before that hold POSIX floats, which
    # read as this machine's zone -- the clock time they always displayed.
    start_timestamp: Optional[datetime] = field(default_factory=now)
    end_timestamp: Optional[datetime] = None
    status: TaskStatus = TaskStatus.NotStarted
    status_message: str = ""
    # files this run produced, keyed by role, as paths relative to the item's path.
    # roles mirror the naming convention the files already carry: phase x modality,
    # i.e. final_sem / final_fib / start_sem / start_fib, plus fluorescence.
    # paths only -- the experiment is written with yaml.safe_dump, which refuses
    # numpy scalars and enums and would fail the whole save. measured values belong
    # in a separate field, not here.
    outputs: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A caller that still passes a POSIX float gets it read as one.
        self.start_timestamp = to_datetime(self.start_timestamp)
        self.end_timestamp = to_datetime(self.end_timestamp)

    @property
    def completed(self) -> str:
        return f"{self.name} ({self.completed_at})"

    @property
    def completed_at(self) -> str:
        if self.end_timestamp is None:
            return "in progress"
        return format_time(self.end_timestamp, TIME_DISPLAY_AMPM_SHORT) or ""

    @property
    def started_at(self) -> str:
        return format_time(self.start_timestamp, TIME_DISPLAY_AMPM_SHORT) or ""

    @property
    def duration(self) -> float:
        """Seconds from start to end; 0 while it runs or when either is unknown."""
        start, end = to_datetime(self.start_timestamp), to_datetime(self.end_timestamp)
        if start is None or end is None:
            return 0
        return (end - start).total_seconds()

    @property
    def duration_str(self) -> str:
        return format_duration_precise(self.duration)

    def to_dict(self) -> dict:
        """Convert the task state to a dictionary."""
        ddict = asdict(self)
        ddict["status"] = self.status.name
        for key in ("start_timestamp", "end_timestamp"):
            when = to_datetime(ddict[key])
            ddict[key] = when.isoformat() if when is not None else None
        return ddict

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """Create a task state from a dictionary."""
        if data is None:
            return cls()
        data = data.copy()
        # written as lamella_id before the item was any item
        if "lamella_id" in data:
            data.setdefault("item_id", data.pop("lamella_id"))
        status = data.get("status", "NotStarted")
        # Written while a supervised result waited on a decision after the
        # task ended (0.6 development builds): the task had finished.
        data["status"] = TaskStatus[
            "Completed" if status == "AwaitingDecision" else status
        ]
        # Either form: ISO with offset, or the POSIX float written before FIB-1197.
        for key in ("start_timestamp", "end_timestamp"):
            if key in data:
                data[key] = to_datetime(data[key])
        # drop keys this build doesn't know about: an experiment written by a newer
        # version must still load in an older one rather than raising TypeError.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True, eq=False)
class WorkflowStatusUpdate:
    """One report on a task's lifecycle, from ``TaskManager._emit_status``.

    ``eq=False`` is load-bearing, for a different reason than ``TiledProgress``'s.
    Three fields hold lists, and ``WorkItem`` is a *mutable* dataclass, so it is
    unhashable. ``frozen=True`` with the default ``eq=True`` generates a ``__hash__``
    over every field, which raises ``TypeError: unhashable type: 'list'`` the moment
    anything puts a report in a set or uses it as a key. There is no numpy here --
    if you came looking for the tiled contract's reason, this is not it.
    """

    task_name: str = "Unknown Task"
    item_name: str = "Unknown Lamella"
    status: TaskStatus = TaskStatus.NotStarted
    timestamp: Optional[float] = None
    error_message: Optional[str] = None
    task_duration: Optional[float] = None
    skip_reason: Optional[str] = None
    # Position in the *live* queue, and ALREADY 1-based: the producer writes
    # `position + 1`. Deliberately unlike milling progress's 0-based
    # `current_stage`, which seven consumers each increment. There is nothing to
    # correct here, so do not port a `display_*` property over -- it would
    # double-count.
    queue_position: Optional[int] = None
    queue_total: int = 0
    # A snapshot of copies, not the live queue: `Queue.items` returns
    # `[copy.copy(i) for i in self._items]` under a lock. Safe to read from the GUI
    # thread, and no further defensive copying is wanted.
    #
    # `None` rather than `[]` when absent, and the difference is load-bearing: the
    # timeline treats "this payload carries no snapshot" as "leave the rows alone",
    # and "the snapshot is empty" as "reconcile to zero rows". Defaulting to `[]`
    # would collapse the two and leave stale rows on screen when a queue empties.
    queue_items: Optional[List["WorkItem"]] = None
    # The plan this run was launched with. Informational only -- the live queue may
    # since have diverged from it. No in-repo production consumer reads either of
    # these; they are kept for the same reason `lamella_name` is, and should be
    # dropped together with it.
    task_names: List[str] = field(default_factory=list)
    lamella_names: List[str] = field(default_factory=list)
    # The queue entry this report is about, so a timeline matches its row by id
    # rather than by (item, task), which repeats when items run in iterations.
    work_item_id: str = ""
    # Its iteration (WorkItem.iteration): None for items that run their tasks once.
    iteration: Optional[int] = None

    @property
    def lamella_name(self) -> str:
        """Deprecated alias for :attr:`item_name`.

        A property rather than a field so there is one source of truth and the two
        cannot drift. Drop it alongside the ``HookContext`` shims after v0.6.
        """
        return self.item_name

    @classmethod
    def from_payload(
        cls, payload: Union["WorkflowStatusUpdate", Dict[str, Any], None]
    ) -> "WorkflowStatusUpdate":
        """Accept either the typed report or the dict the signal still carries.

        Total by construction, and that is the point rather than politeness. This
        runs inside a queued Qt slot, where PyQt5 turns any escaping exception into
        ``qFatal`` -- the process aborts mid-run and the abort reaches no logfile
        (FIB-329). This signal has already killed the app that way once, over a
        missing ``msg`` key. So every field has a default and nothing here indexes.

        Deleted once the producer flips; until then the dict is the live path.
        """
        if isinstance(payload, cls):
            return payload
        if not payload:
            return cls()

        status = payload.get("status")
        if not isinstance(status, TaskStatus):
            # Not a lie worth telling loudly: `NotStarted` fires no consumer branch,
            # so an unrecognised value renders as "nothing in particular happened"
            # rather than crashing or picking a wrong outcome.
            status = TaskStatus.NotStarted

        return cls(
            task_name=payload.get("task_name") or "Unknown Task",
            item_name=payload.get("item_name") or "Unknown Lamella",
            status=status,
            timestamp=payload.get("timestamp"),
            error_message=payload.get("error_message"),
            task_duration=payload.get("task_duration"),
            skip_reason=payload.get("skip_reason"),
            queue_position=payload.get("queue_position"),
            queue_total=payload.get("queue_total") or 0,
            queue_items=(
                list(payload["queue_items"])
                if payload.get("queue_items") is not None
                else None
            ),
            task_names=list(payload.get("task_names") or []),
            lamella_names=list(payload.get("lamella_names") or []),
        )


@dataclass(frozen=True)
class WorkflowStatusEvent:
    """One fire-and-forget status update, on ``workflow_status_signal``.

    Everything the workflow says without needing an answer travels as one of
    these — its own channel, so that saying something can never release a
    blocked waiter (the defect the deleted dict ``workflow_update_signal`` had:
    its handler freed *every* waiter on *any* emission). Instructions and
    questions do not ride a signal at all; they are requests to the responder
    (``fibsem/workflows/interaction.py``).

    Three text destinations and one structured report, and every field optional
    because the emit sites say different subsets:

    * ``message`` — the instruction label. ``None`` says nothing about the prompt
      and leaves it alone — the responder's chrome-refresh events must not take
      down the question they just put up. ``""`` clears it (how a prompt comes
      down); any other text replaces it.
    * ``workflow_info`` — the workflow-information label. ``None`` leaves its text
      alone (it still shows the label; pinned behaviour).
    * ``status_bar`` — transient main-window status-bar text.
    * ``report`` — a task-lifecycle report, for the timeline and run controls.
    """

    message: Optional[str] = None
    workflow_info: Optional[str] = None
    status_bar: Optional[str] = None
    report: Optional[WorkflowStatusUpdate] = None


class HoldKind(str, Enum):
    """Who is holding the run."""

    question = "question"  # a supervised question is up, for the operator


@dataclass(frozen=True)
class Hold:
    """The run is active and nothing is executing because someone has to act.

    One value in place of the flags it replaced (``WAITING_FOR_USER_INTERACTION``
    and ``WAITING_FOR_REVIEW``): who holds the run and what releases it.
    ``BaseTaskManager.held`` says it and a window's ``hold`` keeps it; None means
    the run is not held. Taken by whoever holds the run -- the responder for a
    prompt it shows, the task manager for a question -- and read by
    the window chrome: border colour, attention button, status-bar sentence.
    """

    kind: HoldKind
    # what releases it, as a sentence fragment: "answer the question on the
    # Microscope tab"
    releases: str
    # the lamella/task pairs waiting, when there are some
    items: Tuple[str, ...] = ()


def take_back_finish(item: Any, task_id: str, status: TaskStatus, message: str) -> None:
    """Record that a run which finished did not end Completed after all: what
    came after its work -- settling its result, saving the experiment -- failed
    (FIB-1374). The item's task_state says so, and so does the run's history
    entry, which post_task froze as Completed: replaced, so the run has one
    entry and a later save writes the failure, not the success. A run with no
    entry yet gets one.

    ``item`` is a lamella or a grid record: anything with task_state and
    task_history. The task_state object is set field by field, never
    replaced, because the UI holds on to it.
    """
    state = item.task_state
    state.status = status
    state.status_message = message
    if state.end_timestamp is None:
        state.end_timestamp = now()
    history = item.task_history
    if history and history[-1].task_id == task_id:
        history[-1] = deepcopy(state)
    else:
        history.append(deepcopy(state))


def post_status(
    run: Any,
    msg: Optional[str],
    workflow_info: Optional[str] = None,
    status_bar: Optional[str] = None,
) -> None:
    """Post a line on the run's ``status`` (``BaseTaskManager``), which a window
    draws on its workflow-information label or status bar; in the log when there
    is no run."""
    status = getattr(run, "status", None)
    if status is None:
        logging.info(msg or status_bar or "")
        return
    status.emit(
        WorkflowStatusEvent(
            message=msg, workflow_info=workflow_info, status_bar=status_bar
        )
    )
