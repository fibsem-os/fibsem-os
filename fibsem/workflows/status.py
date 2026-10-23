"""The status a task runs through, in any application's workflow, and the typed
payloads on the runner's ``status`` that report it.

``WorkflowStatusUpdate`` began as the record nested inside the dict
``workflow_update_signal``'s ``status`` key; that signal is gone, and
the record now rides inside :class:`WorkflowStatusEvent`.

Unlike the other progress signals, there was no vocabulary to design here. ``status``
has always been a real ``TaskStatus`` on the wire; what was missing was a
declared shape around it. A task reports through :func:`update_status_ui` and
:func:`set_images_ui`, each also a point where a Stop interrupts it.
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from psygnal import evented

from fibsem.codec import NamedEnum, Timestamp, decode, encode
from fibsem.codec.upgrades import records as upgrade
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.structures import BeamType, FibsemImage
from fibsem.util.timestamps import format_time, now, to_datetime
from fibsem.workflows.interaction import (
    INSTRUCTION_TIMEOUT_S,
    SetImages,
    _abort_requested,
    ask,
)

if TYPE_CHECKING:
    from fibsem.workflows.queue import WorkItem


class TaskStatus(NamedEnum, Enum):
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
    # their offset. Files written before that hold POSIX floats, which
    # read as this machine's zone -- the clock time they always displayed.
    start_timestamp: Optional[Timestamp] = field(default_factory=now)
    end_timestamp: Optional[Timestamp] = None
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

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TaskState":
        """Read a run saved by any version (`fibsem.codec.upgrades.records`)."""
        return decode(cls, upgrade.task_state(data))


@dataclass(frozen=True, eq=False)
class WorkflowStatusUpdate:
    """One report on a task's lifecycle, from ``BaseTaskManager._emit_report``.

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
    # The queue entry this report is about, so a timeline matches its row by id
    # rather than by (item, task), which repeats when items run in iterations.
    work_item_id: str = ""
    # Its iteration (WorkItem.iteration): None for items that run their tasks once.
    iteration: Optional[int] = None

    @classmethod
    def from_payload(
        cls, payload: Union["WorkflowStatusUpdate", Dict[str, Any], None]
    ) -> "WorkflowStatusUpdate":
        """Accept either the typed report or the dict the signal still carries.

        Total by construction, and that is the point rather than politeness. This
        runs inside a queued Qt slot, where PyQt5 turns any escaping exception into
        ``qFatal`` -- the process aborts mid-run and the abort reaches no logfile.
        This signal has already killed the app that way once, over a
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
        )


@dataclass(frozen=True)
class WorkflowStatusEvent:
    """One fire-and-forget status update, on the runner's ``status``.

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

    proposal = "proposal"  # a supervised proposal is up, for the operator


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
    # an agent at Drive is listening and expected to decide it: the operator
    # is not needed unless it goes quiet or is stopped
    agent_deciding: bool = False


def take_back_finish(item: Any, task_id: str, status: TaskStatus, message: str) -> None:
    """Record that a run which finished did not end Completed after all: what
    came after its work -- settling its result, saving the experiment -- failed.
    The item's task_state says so, and so does the run's history
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


def _check_for_abort(
    run: Optional[Any], msg: str = "Workflow aborted by user."
) -> bool:
    if _abort_requested(run):
        raise InterruptedError(msg)
    return False


def set_images_ui(
    run: Optional[Any],
    eb_image: Optional[FibsemImage] = None,
    ib_image: Optional[FibsemImage] = None,
    check_abort: bool = True,
):
    """Keep a task's images as the latest of each beam (the run's ``images``),
    then have the window, when one answers, show them. ``check_abort=False``
    keeps them after a Stop without asking the window, as `update_status_ui`."""
    if run is None:
        return

    if check_abort:
        _check_for_abort(run)

    images = getattr(run, "images", None)
    if images is not None:
        if eb_image is not None:
            images.set(eb_image, BeamType.ELECTRON)
        if ib_image is not None:
            images.set(ib_image, BeamType.ION)
    responder = getattr(run, "responder", None)
    if responder is None or _abort_requested(run):
        return

    # TMP: prevent milling images overwriting existing
    if eb_image is not None:
        eb_image.metadata.image_settings.save = False
    if ib_image is not None:
        ib_image.metadata.image_settings.save = False

    # One future per call, owned by this caller — unlike WAITING_FOR_UI_UPDATE,
    # no other emitter can release it. A GUI-side failure re-raises here, on the
    # workflow thread, instead of aborting the process out of a queued slot.
    ask(
        responder,
        SetImages(sem_image=eb_image, fib_image=ib_image),
        abort=lambda: _abort_requested(run),
        timeout=INSTRUCTION_TIMEOUT_S,
    )


def update_status_ui(
    run: Optional[Any],
    msg: Optional[str],
    workflow_info: Optional[str] = None,
    status_bar: Optional[str] = None,
    check_abort: bool = True,
) -> None:
    """Put a line on the workflow-information label or the status bar.

    A status point doubles as a cancellation point: a task that reports progress
    is interrupted here once Stop has been pressed. ``check_abort=False`` is for
    the manager's closing line after a Stop -- "Workflow cancelled by user." -- which is
    the one status that has to get out precisely because the run was aborted.
    """
    if check_abort:
        _check_for_abort(run)
    post_status(run, msg, workflow_info=workflow_info, status_bar=status_bar)
