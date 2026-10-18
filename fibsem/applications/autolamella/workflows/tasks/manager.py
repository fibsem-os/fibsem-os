"""AutoLamella's lamella workflow: its task manager and how a run is started."""

import logging
import time
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import pandas as pd

from fibsem.acting import WORKFLOW
from fibsem.applications.autolamella.workflows.ui import update_status_ui
from fibsem.cancellation import OperationCancelledError
from fibsem.constants import DATETIME_DISPLAY_AMPM
from fibsem.hooks import HookEvent, HookManager, fire_event
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import InsertableDeviceState
from fibsem.util.durations import format_duration_rounded
from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.queue import WorkItem
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


def run_task(
    microscope: FibsemMicroscope,
    task_name: str,
    lamella: "Lamella",
    parent_ui: Optional["AutoLamellaUI"] = None,
    task_manager: Optional["TaskManager"] = None,
) -> None:
    """Run a specific AutoLamella task."""

    task_config = lamella.task_config.get(task_name)
    if task_config is None:
        raise ValueError(
            f"Task configuration for {task_name} not found in lamella tasks."
        )

    from fibsem.applications.autolamella.workflows.tasks import get_tasks

    task_cls = get_tasks().get(task_config.task_type)
    if task_cls is None:
        raise ValueError(f"Task {task_config.task_type} is not registered.")

    task = task_cls(
        microscope=microscope,
        config=task_config,
        lamella=lamella,
        parent_ui=parent_ui,
        task_manager=task_manager,
    )
    task.run()


class TaskManager(BaseTaskManager):
    """Manages execution of autolamella tasks across lamellas."""

    def __init__(
        self,
        microscope: FibsemMicroscope,
        experiment: "Experiment",
        parent_ui: Optional["AutoLamellaUI"] = None,
        hook_manager: Optional[HookManager] = None,
    ):
        super().__init__(microscope, experiment, parent_ui, hook_manager)
        # Lamellae already finished when the run started, so re-running a task on
        # finished work does not re-announce it. Seeded in _run_queue.
        self._completed_lamella: Set[str] = set()
        self._experiment_was_complete = False

    def run(
        self,
        task_names: List[str],
        required_lamella: Optional[List[str]] = None,
    ) -> None:
        """Run the specified tasks for all lamellas in the experiment.
        Args:
            task_names: List of task names to run.
            required_lamella: List of lamella names to run tasks on. If None, all lamellas are processed.

        Every selected (lamella, task) pair runs, completed ones included:
        that is how a task is re-run. The one exception is a task that is
        AwaitingDecision -- its run is over and its record waits in the Review
        tab -- which is left out: running it again would re-move the stage
        and supersede the proposal someone is about to decide, or just did.
        Re-running it is a deliberate act on that task, not a side effect of
        pressing Run. With the Review surface off there is no way to decide,
        so the exception is not made: Run re-runs it and the producer confirms
        its own record, as it does for every task with the flag off.
        """
        if required_lamella is None:
            required_lamella = [p.name for p in self.experiment.positions]
        pairs = [
            (name, task)
            for task in task_names
            for name in required_lamella
            if not (self.review_enabled and self._awaiting_decision(name, task))
        ]
        self.queue.build_from_pairs(
            pairs, task_names=task_names, item_names=required_lamella
        )
        self._run_queue()

    def _awaiting_decision(self, lamella_name: str, task_name: str) -> bool:
        lamella = self.experiment.get_lamella_by_name(lamella_name)
        return lamella is not None and lamella.is_awaiting_decision(task_name)

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        lamella = self.experiment.get_lamella_by_name(item.item_name)
        if lamella is None or lamella.is_failure:
            return None  # let the loop retire it with a reason
        return self._defer_reason(lamella, item.task_name)

    def _run_queue(self) -> None:
        """Process queue items until empty or stopped."""
        self._snapshot_completion()
        with self._recording():
            self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
            self.experiment.decided.connect(self._on_decided)
            try:
                self._prepare()
                self._run_items()
            except Exception as e:
                self._end_escaped_run(e)
                raise
            finally:
                self.experiment.decided.disconnect(self._on_decided)

    def _run_items(self) -> None:
        while not self.is_stopped:
            # Cleared before the scan, so a decision that lands between a scan
            # finding nothing and the wait starting is not lost.
            self._decision_event.clear()
            item = self._next_item()
            if item is None:
                if self.queue.is_empty:
                    break
                if self._wait_for_a_decision():
                    continue
                break

            # A stop_task click that landed between two tasks was aimed at the one
            # that has just finished, not at this one.
            self._task_stop_event.clear()

            lamella = self.experiment.get_lamella_by_name(item.item_name)
            if lamella is None:
                # No lamella to build a status dict from, so this used to leave no trace
                # anywhere — not in the log, the UI, or a hook.
                logging.warning(
                    f"Skipping {item.task_name}: no lamella named {item.item_name} in the experiment."
                )
                self.queue.mark_done(item, TaskStatus.Skipped)
                self._fire_skipped_hook(
                    item.task_name, item.item_name, "lamella_not_found"
                )
                continue

            skip_reason = self._should_skip(lamella, item.task_name)
            if skip_reason is not None:
                msg = f"Skipping {lamella.name} for {item.task_name}: {skip_reason}."
                self.queue.mark_done(item, TaskStatus.Skipped)
                self._emit_status(
                    item=item,
                    lamella=lamella,
                    status=TaskStatus.Skipped,
                    msg=msg,
                    skip_reason=skip_reason,
                )
                task_config = lamella.task_config.get(item.task_name)
                self._fire_skipped_hook(
                    item.task_name,
                    lamella.name,
                    skip_reason,
                    task_type=task_config.task_type if task_config else "",
                    item_id=lamella.id,
                    missing_prereqs=self._missing_prereqs(lamella, item.task_name)
                    if skip_reason == "missing_prereqs"
                    else None,
                )
                continue

            # Wait until the task's scheduled start time, if set and in the future.
            scheduled_at = (
                self.experiment.task_protocol.workflow_config.get_scheduled_at(
                    item.task_name
                )
            )
            if scheduled_at is not None:
                self._wait_until_scheduled(scheduled_at, item.task_name, lamella)
                if self.is_stopped:
                    break

            self._expire_what_this_consumes(lamella.id, item.task_name)

            # Emit InProgress status
            self._emit_status(
                item=item,
                lamella=lamella,
                status=TaskStatus.InProgress,
                msg=f"Starting task {item.task_name} for Lamella {lamella.name}.",
            )

            # Execute the task
            err = self._run_single_task(item.task_name, lamella)
            final_status = lamella.task_state.status
            self.queue.mark_done(item, final_status)

            # Emit Completed/Failed status
            if err is None:
                msg = f"Completed task {item.task_name} for Lamella {lamella.name}."
            else:
                msg = f"Error in task {item.task_name} for Lamella {lamella.name}."
            self._emit_status(
                item=item,
                lamella=lamella,
                status=final_status,
                error_message=lamella.task_state.status_message,
                task_duration=lamella.task_state.duration,
                msg=msg,
            )

            self._maybe_fire_lamella_completed(lamella, item.task_name)

        # if the objective is inserted, retract for safety
        if (
            self.microscope.fm is not None
            and self.microscope.fm.objective.state is InsertableDeviceState.INSERTED
        ):
            self.microscope.fm.objective.retract()

        # The loop exits when the queue drains *or* when the user presses Stop. Reporting
        # completion for both is what made an abort look like a finished workflow.
        if self.is_stopped:
            self._fire_workflow_hook(HookEvent.WORKFLOW_CANCELLED)
            # After a Stop the abort predicate is true, so this line must not
            # be a cancellation point itself: it is the report of one.
            update_status_ui(
                self.parent_ui,
                "",
                workflow_info="Workflow cancelled by user.",
                check_abort=False,
            )
        elif self.stalled:
            # Drained but not done. Not completed -- work remains -- and not
            # cancelled -- nobody pressed Stop. The experiment is not finishable
            # from here either. Items that never ran stay NotStarted; the tasks
            # awaiting a decision stay so; the next Run picks up from there.
            self._report_stall()
        else:
            self._fire_workflow_hook(
                HookEvent.WORKFLOW_COMPLETED, task_counts=self._task_counts()
            )
            update_status_ui(
                self.parent_ui, "", workflow_info=self._completion_message()
            )
            # After the run event, since finishing the run is what makes the experiment
            # finishable. Not fired on the cancelled path: an aborted run has not
            # established anything about the experiment.
            self._maybe_fire_experiment_completed()

    def build_run_summary_dataframe(self) -> pd.DataFrame:
        """Build a per-run summary of attempted tasks from the queue snapshot.

        One row per (lamella, task) attempted in this run, including skipped
        tasks (which are absent from the experiment task history). The
        completed_at and duration values are pulled from the lamella's task
        history where available, and left blank for skipped tasks.
        """
        rows: List[dict] = []
        for item in self.queue.items:
            lamella = self.experiment.get_lamella_by_name(item.item_name)
            completed_at = ""
            duration = None
            if lamella is not None:
                # most recent matching history entry for this task
                for task in reversed(lamella.task_history):
                    if task.name == item.task_name:
                        completed_at = task.completed_at
                        duration = task.duration
                        break
            rows.append(
                {
                    "lamella_name": item.item_name,
                    "task_name": item.task_name,
                    "task_status": item.status.name,
                    "completed_at": completed_at,
                    "duration": duration,
                }
            )
        return pd.DataFrame(rows)

    def _is_complete(self, lamella: "Lamella") -> bool:
        """Whether a lamella has completed every task the workflow requires of it."""
        # task_protocol is None until one is loaded ("must be set externally").
        if self.experiment.task_protocol is None:
            return False
        workflow_config = self.experiment.task_protocol.workflow_config
        # A workflow that requires nothing completes nothing. is_completed() is
        # vacuously True against an empty required-task list, which would announce
        # every lamella as finished the moment a protocol failed to load — and with
        # FIB-461 that means generating artifacts for work that never happened.
        if not workflow_config.required_tasks:
            return False
        return workflow_config.is_completed(lamella)

    def _snapshot_completion(self) -> None:
        """Record what was already finished before this run, so the completion events
        fire on the transition rather than on every run that ends in a finished state."""
        self._completed_lamella = {
            p.name for p in self.experiment.positions if self._is_complete(p)
        }
        self._experiment_was_complete = self._all_lamella_complete()

    def _all_lamella_complete(self) -> bool:
        """Whether every lamella still in scope has finished.

        Lamellae a human has marked defective are out of scope: _should_skip skips them
        for every remaining task, so they can never satisfy the completion predicate.
        Counting them meant a single abandoned lamella permanently prevented its
        experiment from ever reporting complete — and in a real session most experiments
        have one. The completion predicate and the skip predicate have to agree on which
        lamellae they are talking about.
        """
        active = [p for p in self.experiment.positions if not p.is_failure]
        # An experiment with nothing left in scope — empty, or every lamella abandoned —
        # produced nothing, and must not announce success. all([]) would say otherwise.
        return bool(active) and all(self._is_complete(p) for p in active)

    def _maybe_fire_lamella_completed(self, lamella: "Lamella", task_name: str) -> None:
        """Fire ITEM_COMPLETED the first time a lamella satisfies the workflow's
        completion predicate during this run. The lamella is AutoLamella's item.

        A lamella is the unit that gets delivered to a TEM, so this is the event
        downstream work hangs off — it lets a per-lamella artifact be produced the
        moment that lamella is finished, rather than waiting for the whole run.
        """
        if lamella.name in self._completed_lamella or not self._is_complete(lamella):
            return
        self._completed_lamella.add(lamella.name)
        fire_event(
            self.hook_manager,
            HookEvent.ITEM_COMPLETED,
            task_name=task_name,
            task_type=lamella.task_state.task_type,
            item_name=lamella.name,
            item_id=lamella.id,
            task_id=lamella.task_state.task_id,
            task_state=lamella.task_state,
            **self.hook_run_context(),
        )

    def _maybe_fire_experiment_completed(self) -> None:
        """Fire EXPERIMENT_COMPLETED once, when the last outstanding lamella finishes.

        The roll-up of ITEM_COMPLETED: same predicate, applied to every position.
        """
        if self._experiment_was_complete or not self._all_lamella_complete():
            return
        self._experiment_was_complete = True
        logging.info(f"Experiment {self.experiment.name} is complete.")
        fire_event(
            self.hook_manager,
            HookEvent.EXPERIMENT_COMPLETED,
            item_name=self.experiment.name,
            item_id=self.experiment.id,
            **self.hook_run_context(),
        )

    def _completion_message(self) -> str:
        """Closing status line for a run that was not cancelled.

        A run where everything was skipped did no work, so it must not report as a
        completed workflow — that is the same misreport as an abort claiming
        completion, only quieter. Warns as well as returning, because update_status_ui
        does not log workflow_info, so a headless run would see nothing.
        """
        items = self.queue.items
        if not items:
            return "No tasks to run."
        skipped = self._task_counts()["skipped"]
        if skipped == len(items):
            logging.warning(f"No tasks ran: all {skipped} were skipped.")
            return f"No tasks ran: all {skipped} skipped."
        if skipped:
            return f"All tasks completed ({skipped} skipped)."
        return "All tasks completed."

    def _task_counts(self) -> Dict[str, int]:
        """How the run's tasks ended: completed, failed and skipped. A task that
        ended any other way (cancelled, awaiting a decision) is in none of them."""
        counts = Counter(item.status for item in self.queue.items)
        return {
            "completed": counts[TaskStatus.Completed],
            "failed": counts[TaskStatus.Failed],
            "skipped": counts[TaskStatus.Skipped],
        }

    def _wait_until_scheduled(
        self, scheduled_at: datetime, task_name: str, lamella: "Lamella"
    ) -> None:
        """Block until ``scheduled_at`` is reached, emitting a periodic countdown.

        The wait is interruptible by either kind of stop. The item is already
        InProgress by this point, so it is the row the user sees running and
        stop_task has to reach it — otherwise a cancel would sit unanswered until
        the scheduled time arrived. On a task stop the loop falls through and the
        task aborts at its first checkpoint, taking the same path as any other
        mid-task cancel rather than needing its own.
        """
        # Times are naive-local throughout, but a hand-edited/externally-produced
        # protocol may carry a tz-aware scheduled_at. Normalize to naive-local so
        # the subtraction against datetime.now() below doesn't raise TypeError.
        if scheduled_at.tzinfo is not None:
            scheduled_at = scheduled_at.astimezone().replace(tzinfo=None)
        target_str = scheduled_at.strftime(DATETIME_DISPLAY_AMPM)
        next_update = 0.0  # force an immediate first message
        self._set_workflow_pending(True)
        try:
            while not self.should_abort:
                remaining = (scheduled_at - datetime.now()).total_seconds()
                if remaining <= 0:
                    break

                now = time.monotonic()
                if now >= next_update:
                    msg = (
                        f"Waiting until {target_str} to start {task_name} on "
                        f"{lamella.name} ({format_duration_rounded(remaining)} remaining)."
                    )
                    update_status_ui(self.parent_ui, "", status_bar=msg)
                    next_update = now + 15.0

                time.sleep(min(1.0, remaining))
        finally:
            # The time arriving, a stop, and an exception all have to clear this.
            # A stranded flag would leave the border reading "nothing is running"
            # for the rest of the run.
            self._set_workflow_pending(False)

    def _requirements_of(self, task_name: str) -> List[str]:
        return list(
            self.experiment.task_protocol.workflow_config.requirements(task_name)
        )

    def _defer_reason(self, lamella: "Lamella", task_name: str) -> Optional[str]:
        """Why this task cannot run *yet* -- as opposed to _should_skip, which
        says why it never will this run.

        The prerequisite check used to be terminal: unmet meant Skipped, done for
        the run. That was right only because ordering guaranteed the prerequisite
        had already had its turn. Two things break the guarantee, and both are
        answered by looking at the queue and the item rather than at history:

        - ``awaiting_decision``: a required task ran and is not finished: its
          record waits in the Review tab. The decision makes this runnable.
        - ``prereq_pending``: a required task is still in the queue ahead or
          behind (a re-run, a mid-run insertion). It will get its turn.

        Deferred items stay pending and are offered again on the next scan.
        Nothing is logged per scan: a long stall would flood the log with one
        line per pass.
        """
        for req in self.experiment.task_protocol.workflow_config.requirements(
            task_name
        ):
            # a rerun still queued is the attempt that counts, whatever an
            # earlier run of it did (FIB-1006)
            if self.queue.has_pending_pair(lamella.name, req):
                return "prereq_pending"
            if lamella.is_awaiting_decision(req):
                return "awaiting_decision"
        return None

    def _should_skip(self, lamella: "Lamella", task_name: str) -> Optional[str]:
        """Return skip reason string, or None if task should run. Terminal: a
        reason here means Skipped, finished for the run. "Not yet" is
        _defer_reason's answer and is asked first, by the queue scan.

        Deliberately does not re-check the run's lamella selection: that filter
        is applied when the queue is built, so anything that reaches here is
        wanted — including items the user added mid-run, which are not in the
        original selection and were previously skipped as "not_required".
        """
        if lamella.is_failure:
            logging.info(
                f"Skipping lamella {lamella.name} for task {task_name}. Marked as failure or has defect."
            )
            return "failure"

        missing = self._missing_prereqs(lamella, task_name)
        if missing:
            logging.info(
                f"Skipping lamella {lamella.name} for task {task_name}. Required tasks {missing} not completed."
            )
            return "missing_prereqs"

        return None

    def _missing_prereqs(self, lamella: "Lamella", task_name: str) -> List[str]:
        """The tasks *task_name* requires whose latest run on *lamella* did not
        complete: the latest run, not any past success, as a rerun that failed or
        was rejected is the answer (FIB-1006)."""
        requirements = self.experiment.task_protocol.workflow_config.requirements(
            task_name
        )
        return [
            req for req in requirements or [] if not lamella.latest_run_completed(req)
        ]

    def _emit_status(
        self,
        item: WorkItem,
        lamella: "Lamella",
        status: "TaskStatus",
        msg: str = "",
        error_message: Optional[str] = None,
        task_duration: Optional[float] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        """The lamella spelling of `_emit_report`: the item is the lamella."""
        self._emit_report(
            item=item,
            item_name=lamella.name,
            status=status,
            msg=msg,
            error_message=error_message,
            task_duration=task_duration,
            skip_reason=skip_reason,
        )

    def _run_single_task(
        self, task_name: str, lamella: "Lamella"
    ) -> Optional[Exception]:
        """Execute one task for one lamella. Returns exception or None."""
        try:
            run_task(
                microscope=self.microscope,
                task_name=task_name,
                lamella=lamella,
                parent_ui=self.parent_ui,
                task_manager=self,
            )
        except Exception as e:
            # A user Stop surfaces as OperationCancelledError (milling) or InterruptedError
            # (_check_for_abort); either way self.is_stopped is set by TaskManager.stop().
            # Mark it Cancelled, not Failed — it isn't an error.
            #
            # The task itself now records the outcome and freezes it into task_history
            # before re-raising (FIB-490), so this repeats what it already set. It stays
            # as the fallback for exceptions raised before the task's run() is entered --
            # an unknown task name, or a construction failure -- where nothing else has.
            # The predicate matches AutoLamellaTaskBase._is_cancellation so the frozen
            # history entry and the live task_state cannot disagree.
            if self.should_abort or isinstance(
                e, (OperationCancelledError, InterruptedError)
            ):
                logging.info(
                    f"Task {task_name} for lamella {lamella.name} cancelled by user."
                )
                lamella.task_state.status = TaskStatus.Cancelled
                lamella.task_state.status_message = "Cancelled by user."
            else:
                logging.warning(
                    f"Error running task {task_name} for lamella {lamella.name}: {e}",
                    exc_info=True,
                )
                lamella.task_state.status = TaskStatus.Failed
                lamella.task_state.status_message = str(e)
            self.experiment.save()
            return e
        try:
            self.experiment.save()
        except Exception as e:
            self._result_not_saved(lamella, task_name, e)
            raise
        return None


def run_tasks(
    microscope: FibsemMicroscope,
    experiment: "Experiment",
    task_names: List[str],
    required_lamella: Optional[List[str]] = None,
    parent_ui: Optional["AutoLamellaUI"] = None,
    hook_manager: Optional[HookManager] = None,
) -> None:
    """Run the specified tasks for all lamellas in the experiment.
    Thin wrapper around TaskManager for backward compatibility and headless usage.
    In control of the microscope for the run (FIB-1270): refused with
    ``MicroscopeBusy`` if someone else is.
    """
    manager = TaskManager(microscope, experiment, parent_ui, hook_manager=hook_manager)
    with microscope.control.take(WORKFLOW, "a lamella run"):
        manager.run(task_names, required_lamella)
