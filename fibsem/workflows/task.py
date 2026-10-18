"""One run of a task on one item, whatever the item is.

``pre_task -> _run -> post_task``. A failed or cancelled ``_run`` is recorded on
the item's history before the exception goes on, and a run that finished but
whose result could not be recorded is taken back. Then the result is proposed
on the item and decided (``proposing.settle``).

An application's task base subclasses ``BaseTask``: AutoLamella's lamella and
grid tasks are two. The item is anything with a ``name``, an ``id``, a
``task_state``, a ``task_history`` and ``proposals``.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from fibsem.acting import TASK, acting
from fibsem.cancellation import OperationCancelledError
from fibsem.util.timestamps import now, now_iso
from fibsem.workflows.proposals import Decision, Proposer, TaskResultProposer
from fibsem.workflows.proposing import settle
from fibsem.workflows.status import TaskStatus, take_back_finish

# Characters a folder name cannot hold on Windows (and ``/`` anywhere).
_UNSAFE_IN_FOLDER_NAMES = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def iteration_folder(iteration: int) -> str:
    """The folder name of one iteration of an item's tasks: ``0003``."""
    return f"{iteration:04d}"


def run_folder(
    item_dir: Any, task_name: str, started: datetime, iteration: Optional[int] = None
) -> str:
    """Where one run of ``task_name`` on an item saves what it makes:
    ``<item>/<task name>/<YYYYMMDD-HHMMSS>/`` (FIB-1275), so a re-run
    overwrites nothing; ``<item>/<task name>/<iteration:04d>/`` for an item that
    runs in iterations, one folder per iteration. The item's shared state
    (``ref_alignment``, the coincidence setup reference) is not a run's, and
    stays in the item's folder (FIB-1258).

    The task name is made safe for a folder name; a second run of the same
    second, or the same iteration, as one already saved gets ``-2``. Not created
    here: a run that saves nothing leaves no folder. Readers find a run's files
    from its record (``outputs``), not from this name.
    """
    task_folder = _UNSAFE_IN_FOLDER_NAMES.sub("-", task_name).strip().rstrip(".")
    leaf = (
        iteration_folder(iteration)
        if iteration is not None
        else f"{started:%Y%m%d-%H%M%S}"
    )
    base = os.path.join(str(item_dir), task_folder or "task", leaf)
    folder, n = base, 1
    while os.path.exists(folder):
        n += 1
        folder = f"{base}-{n}"
    return folder


# Steps recorded as the task's own lifecycle events (task_started,
# task_completed), not as task_step.
LIFECYCLE_STEPS = {"STARTED", "FINISHED"}


class BaseTask(ABC):
    """The lifecycle every task shares. Subclasses implement ``_run``,
    ``pre_task`` and ``post_task``, and say whether the run is supervised."""

    config: Any
    # What this task type proposes for the Review tab. The default proposes the
    # task's result ("here is what I did, look"); a task whose output is a
    # value someone might change swaps in a proposer for that kind; None only
    # for a type with nothing to look at.
    proposer: ClassVar[Optional[Proposer]] = TaskResultProposer()
    # Which recorded output roles are the result images a proposal points at,
    # by provenance key; the last file under each role is the one shown.
    result_images: ClassVar[Dict[str, str]] = {}
    # What the item is called in the log and in task_step events.
    item_type: ClassVar[str] = "item"

    def __init__(
        self,
        microscope: Any,
        config: Any,
        item: Any,
        experiment: Optional[Any] = None,
        parent_ui: Optional[Any] = None,
        task_manager: Optional[Any] = None,
    ) -> None:
        self.microscope = microscope
        self.config = config
        self.item = item
        self.experiment = (
            experiment
            if experiment is not None
            else getattr(task_manager, "experiment", None)
        )
        self.parent_ui = parent_ui
        self.task_manager = task_manager
        self.task_id = str(uuid.uuid4())
        # The iteration of the item's tasks this run is, when the item runs in
        # iterations: the queue entry the manager is running.
        self.iteration: Optional[int] = getattr(task_manager, "active_iteration", None)
        # The manager's token, not its raw event: what counts as "cancelled" is
        # the manager's to decide, and a task should not have to know.
        self._stop_event = task_manager.abort_token if task_manager else None
        # A decision the task took in its own run -- the operator's inline
        # answer to a supervised question -- recorded on the proposal once it
        # is made, after the run, through Experiment.decide like every other.
        self.inline_decision: Optional[Decision] = None

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @property
    def display_name(self) -> str:
        return self.config.display_name

    @property
    @abstractmethod
    def supervised(self) -> bool:
        """Whether this task's attention is supervised: then a result nobody
        decided in the run waits on a decision in the Review tab; otherwise
        its values are used as proposed."""

    # -- lifecycle -------------------------------------------------------------

    # Everything the task does, on its own thread, is the task's (FIB-1062).
    @acting(TASK)
    def run(self) -> None:
        self.pre_task()
        self._fire_hook("task_started")
        try:
            self._run()
        except Exception as e:
            # A user Stop is a cancellation, not a failure — fire the distinct hook so it
            # notifies as "cancelled" (warning) rather than "FAILED" (error).
            cancelled = self._is_cancellation(e)
            # Record before announcing. The hook payload carries task_state, which
            # still reads InProgress from pre_task until this runs.
            #
            # Guarded: anything raising in here would replace the task's own
            # exception with this one, losing what actually went wrong.
            try:
                self.item.task_state.status = (
                    TaskStatus.Cancelled if cancelled else TaskStatus.Failed
                )
                self.item.task_state.status_message = (
                    "Cancelled by user." if cancelled else str(e)
                )
                self._record_outcome()
                # A failure is exactly when someone wants to look; a Stop is
                # not, whoever pressed it already knows.
                if not cancelled:
                    self._settle(failure=str(e))
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook(
                "task_cancelled" if cancelled else "task_failed", error=str(e)
            )
            raise
        finally:
            # In a finally, not in post_task, which never runs when _run() raises. Left
            # set, the next thing acquired -- an operator looking at what went wrong,
            # a supervision step -- would be stamped with the task that just failed.
            # That is silently wrong, and wrong exactly when the record matters most.
            self._clear_workflow_metadata()
        try:
            self.post_task()
            self._settle()
        except Exception as e:
            # The work finished, but what follows it failed: settling its result
            # can save the experiment, and a save that fails leaves nothing on
            # disk to say so. It did not complete (FIB-1374).
            cancelled = self._is_cancellation(e)
            try:
                take_back_finish(
                    self.item,
                    self.task_id,
                    TaskStatus.Cancelled if cancelled else TaskStatus.Failed,
                    "Cancelled by user."
                    if cancelled
                    else f"Finished, but its result could not be recorded: {e}",
                )
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook(
                "task_cancelled" if cancelled else "task_failed", error=str(e)
            )
            raise
        self._fire_hook("task_completed")

    @abstractmethod
    def _run(self) -> None: ...

    @abstractmethod
    def pre_task(self) -> None: ...

    @abstractmethod
    def post_task(self) -> None: ...

    def _start_task_state(self) -> None:
        """Mark the item's task_state as this run, InProgress.

        task_state is a single object reused across every run, so each per-run
        field is cleared here or the next run inherits the previous one's value.
        Never replace the object: the item list and card widgets connect psygnal
        handlers to this instance and use them as their only refresh trigger;
        replacing it orphans those connections and they silently stop updating
        mid-workflow (FIB-325)."""
        state = self.item.task_state
        state.name = self.task_name
        state.start_timestamp = now()
        state.end_timestamp = None
        state.task_id = self.task_id
        state.task_type = self.task_type
        state.iteration = self.iteration
        state.status = TaskStatus.InProgress
        state.status_message = ""
        state.outputs = {}

    def _record_outcome(self) -> None:
        """Freeze the finished task_state into task_history.

        Runs on every terminal path, not just success. task_state is a single object
        that the next task's pre_task overwrites, so an outcome that isn't frozen here
        is gone from the experiment as soon as that item starts its next task --
        leaving the logfile as the only record that the task ever failed. See FIB-490.
        """
        if self.item.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.item.task_state.end_timestamp = now()
        self.item.task_history.append(deepcopy(self.item.task_state))

    def _settle(self, failure: str = "") -> None:
        """Propose this run's result on the item and decide it; see
        ``proposing.settle``."""
        settle(
            self,
            self.item,
            proposer=type(self).proposer,
            result_images=self.result_images,
            supervised=self.supervised,
            inline_decision=self.inline_decision,
            experiment=self.experiment,
            failure=failure,
        )

    def _is_cancellation(self, exc: Exception) -> bool:
        """Whether ``exc`` is a user Stop rather than a genuine failure: it surfaces as
        OperationCancelledError (milling / autofocus) or InterruptedError (_check_for_abort),
        or the task manager says this task should be unwinding.

        should_abort, not is_stopped: abandoning one task is as much a cancellation
        as ending the run, and an exception thrown on the way out of one should not
        be recorded as a failure."""
        if isinstance(exc, (OperationCancelledError, InterruptedError)):
            return True
        return bool(getattr(self.task_manager, "should_abort", False))

    def _check_for_abort(self) -> None:
        """Raise InterruptedError if this task should stop. Call between steps.

        Polls the manager's token, which already answers for every kind of stop —
        so this needs no second way to ask, and neither do the inline token checks
        in the tasks that poll it themselves.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            raise InterruptedError("Workflow aborted by user.")

    # -- status, metadata, hooks -----------------------------------------------

    def log_status_message(
        self, message: str, display_message: Optional[str] = None
    ) -> None:
        """Record a step of this run: in the log, on the item's task_state, as a
        task_step event and on the window's step line."""
        logging.debug(
            {
                "msg": "status",
                "timestamp": now_iso(),
                self.item_type: self.item.name,
                f"{self.item_type}_id": self.item.id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "task_step": message,
            }
        )
        if self.item.task_state is not None:
            self.item.task_state.step = message
            self.item.task_state.status_message = display_message or ""
        if message in LIFECYCLE_STEPS:
            return
        record = getattr(self.microscope, "record_event", None)
        if record is not None:
            record(
                "task_step",
                {
                    "step": message,
                    "display_message": display_message,
                    "item_type": self.item_type,
                    "task_type": self.task_type,
                },
            )
        signal = getattr(self.parent_ui, "step_update_signal", None)
        if signal is not None:
            signal.emit(display_message or message)

    def _set_workflow_metadata(self) -> None:
        """Stamp everything acquired from here until the run ends with the item
        and task that produced it, rather than relying on which folder it
        happens to sit in (FIB-466)."""
        ref = getattr(self.microscope, "experiment", None)
        setter = getattr(ref, "set_workflow_metadata", None)
        if setter is not None:
            setter(
                item_id=self.item.id,
                item_name=self.item.name,
                task_id=self.task_id,
                task_name=self.task_name,
                iteration=self.iteration,
            )

    def _clear_workflow_metadata(self) -> None:
        ref = getattr(self.microscope, "experiment", None)
        clear = getattr(ref, "clear_workflow_metadata", None)
        if clear is not None:
            clear()

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        from fibsem.hooks import fire_event

        # Which experiment, and how much of the queue is left. Only the manager knows
        # either, and a task can run without one -- a script driving a task directly, or
        # a stand-in -- so those fields are simply absent then.
        get_run_context = getattr(self.task_manager, "hook_run_context", None)
        run_context = get_run_context() if callable(get_run_context) else {}
        fire_event(
            getattr(self.task_manager, "hook_manager", None),
            event,
            task_name=self.task_name,
            task_type=self.task_type,
            item_name=self.item.name,
            item_id=self.item.id,
            task_id=self.task_id,
            task_state=self.item.task_state,
            error=error,
            **run_context,
        )
