"""Task execution manager for autolamella workflows."""

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Set

import fibsem.config as fcfg
from fibsem.constants import DATETIME_DISPLAY_AMPM
import pandas as pd

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.cancellation import OperationCancelledError
from fibsem.hooks import HookEvent, HookManager, fire_event
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue, WorkItem
from fibsem.applications.autolamella.workflows.ui import update_status_ui
from fibsem.microscope import FibsemMicroscope
from fibsem.utils import format_time_remaining

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui import AutoLamellaUI


def run_task(microscope: FibsemMicroscope,
          task_name: str,
          lamella: 'Lamella',
          parent_ui: Optional['AutoLamellaUI'] = None,
          task_manager: Optional['TaskManager'] = None) -> None:
    """Run a specific AutoLamella task."""

    task_config = lamella.task_config.get(task_name)
    if task_config is None:
        raise ValueError(f"Task configuration for {task_name} not found in lamella tasks.")

    from fibsem.applications.autolamella.workflows.tasks import get_tasks
    task_cls = get_tasks().get(task_config.task_type)
    if task_cls is None:
        raise ValueError(f"Task {task_config.task_type} is not registered.")

    task = task_cls(microscope=microscope,
                    config=task_config,
                    lamella=lamella,
                    parent_ui=parent_ui,
                    task_manager=task_manager)
    task.run()


class TaskManager:
    """Manages execution of autolamella tasks across lamellas."""

    def __init__(self,
                 microscope: FibsemMicroscope,
                 experiment: 'Experiment',
                 parent_ui: Optional['AutoLamellaUI'] = None,
                 hook_manager: Optional[HookManager] = None):
        self.microscope = microscope
        self.experiment = experiment
        self.parent_ui = parent_ui
        self.hook_manager = hook_manager
        self._stop_event = threading.Event()
        self.queue = TaskQueue()

        # Stamp the experiment onto the images this run acquires. Done here rather
        # than only in the UI so a headless run through run_tasks() records it too;
        # the app also calls it when it adopts an experiment, and repeating it just
        # re-sets the same two attributes. See FIB-449.
        self.experiment.register_metadata(self.microscope)
        # Lamellae already finished when the run started, so re-running a task on
        # finished work does not re-announce it. Seeded in _run_queue.
        self._completed_lamella: Set[str] = set()
        self._experiment_was_complete = False

    # --- Public API ---

    def run(self, task_names: List[str],
            required_lamella: Optional[List[str]] = None) -> None:
        """Run the specified tasks for all lamellas in the experiment.
        Args:
            task_names: List of task names to run.
            required_lamella: List of lamella names to run tasks on. If None, all lamellas are processed.
        """
        if required_lamella is None:
            required_lamella = [p.name for p in self.experiment.positions]

        self.queue.build_from_matrix(task_names, required_lamella)
        self._run_queue()

    def _run_queue(self) -> None:
        """Process queue items until empty or stopped."""
        self._snapshot_completion()
        self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            lamella = self.experiment.get_lamella_by_name(item.lamella_name)
            if lamella is None:
                # No lamella to build a status dict from, so this used to leave no trace
                # anywhere — not in the log, the UI, or a hook.
                logging.warning(
                    f"Skipping {item.task_name}: no lamella named {item.lamella_name} in the experiment."
                )
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._fire_skipped_hook(item.task_name, item.lamella_name, "lamella_not_found")
                continue

            skip_reason = self._should_skip(lamella, item.task_name)
            if skip_reason is not None:
                msg = f"Skipping {lamella.name} for {item.task_name}: {skip_reason}."
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_status(
                    item=item,
                    lamella=lamella,
                    status=AutoLamellaTaskStatus.Skipped,
                    msg=msg,
                    skip_reason=skip_reason,
                )
                task_config = lamella.task_config.get(item.task_name)
                self._fire_skipped_hook(
                    item.task_name, lamella.name, skip_reason,
                    task_type=task_config.task_type if task_config else "",
                    item_id=lamella.id,
                )
                continue

            # Wait until the task's scheduled start time, if set and in the future
            # (only when the scheduled-tasks feature is enabled; otherwise any
            # saved scheduled_at is left dormant and the task runs immediately).
            scheduled_at = self.experiment.task_protocol.workflow_config.get_scheduled_at(item.task_name)
            if scheduled_at is not None and fcfg.FEATURE_SCHEDULED_TASKS_ENABLED:
                self._wait_until_scheduled(scheduled_at, item.task_name, lamella)
                if self.is_stopped:
                    break

            # Emit InProgress status
            self._emit_status(
                item=item,
                lamella=lamella,
                status=AutoLamellaTaskStatus.InProgress,
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
        if self.microscope.fm is not None and self.microscope.fm.objective.state == "Inserted":
            self.microscope.fm.objective.retract()

        # The loop exits when the queue drains *or* when the user presses Stop. Reporting
        # completion for both is what made an abort look like a finished workflow.
        if self.is_stopped:
            self._fire_workflow_hook(HookEvent.WORKFLOW_CANCELLED)
            update_status_ui(self.parent_ui, "", workflow_info="Workflow cancelled.")
        else:
            self._fire_workflow_hook(HookEvent.WORKFLOW_COMPLETED)
            update_status_ui(self.parent_ui, "", workflow_info=self._completion_message())
            # After the run event, since finishing the run is what makes the experiment
            # finishable. Not fired on the cancelled path: an aborted run has not
            # established anything about the experiment.
            self._maybe_fire_experiment_completed()
        print(self.experiment.task_history_dataframe())

    def build_run_summary_dataframe(self) -> pd.DataFrame:
        """Build a per-run summary of attempted tasks from the queue snapshot.

        One row per (lamella, task) attempted in this run, including skipped
        tasks (which are absent from the experiment task history). The
        completed_at and duration values are pulled from the lamella's task
        history where available, and left blank for skipped tasks.
        """
        rows: List[dict] = []
        for item in self.queue.items:
            lamella = self.experiment.get_lamella_by_name(item.lamella_name)
            completed_at = ""
            duration = None
            if lamella is not None:
                # most recent matching history entry for this task
                for task in reversed(lamella.task_history):
                    if task.name == item.task_name:
                        completed_at = task.completed_at
                        duration = task.duration
                        break
            rows.append({
                "lamella_name": item.lamella_name,
                "task_name": item.task_name,
                "task_status": item.status.name,
                "completed_at": completed_at,
                "duration": duration,
            })
        return pd.DataFrame(rows)

    def stop(self) -> None:
        """Signal the manager to stop after current task completes."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def notify_queue_changed(self) -> None:
        """Tell the UI the queue's contents changed, outside the task lifecycle.

        Status updates only fire when a task starts, finishes or is skipped, so
        an edit made between two tasks would otherwise stay invisible until the
        next one began. Routing it through the same signal keeps the UI's view
        of the queue single-sourced, and makes an edit from the worker thread
        (a script, say) as safe as one from the UI thread.
        """
        if self.parent_ui is None:
            return
        self.parent_ui.workflow_update_signal.emit({
            "queue_changed": {
                "queue_items": self.queue.items,  # thread-safe snapshot
                "version": self.queue.version,
            }
        })

    def hook_run_context(self) -> dict:
        """The fields every hook event from this run carries: which experiment, and
        how much of the queue is left.

        Public because AutoLamellaTask fires task events itself and needs the same
        fields; a task reaches this through its task_manager.

        tasks_remaining excludes the task the event is about: queue.next() marks an
        item InProgress before the task runs, so the one in flight is already out of
        the pending set on every path, skips included.
        """
        pending, total = self.queue.counts
        return {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "tasks_remaining": pending,
            "tasks_total": total,
        }

    def _fire_workflow_hook(self, event: HookEvent) -> None:
        # Workflow events used to carry nothing at all, so workflow_completed reached a
        # webhook with no way to tell which experiment had finished.
        fire_event(self.hook_manager, event, **self.hook_run_context())

    def _is_complete(self, lamella: 'Lamella') -> bool:
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
        positions = self.experiment.positions
        # An experiment with no lamellae has not been completed, it is empty. all([])
        # would say otherwise.
        return bool(positions) and all(self._is_complete(p) for p in positions)

    def _maybe_fire_lamella_completed(self, lamella: 'Lamella', task_name: str) -> None:
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
        skipped = sum(1 for i in items if i.status is AutoLamellaTaskStatus.Skipped)
        if skipped == len(items):
            logging.warning(f"No tasks ran: all {skipped} were skipped.")
            return f"No tasks ran: all {skipped} skipped."
        if skipped:
            return f"All tasks completed ({skipped} skipped)."
        return "All tasks completed."

    def _fire_skipped_hook(self, task_name: str, item_name: str,
                           skip_reason: str, task_type: str = "",
                           item_id: str = "") -> None:
        """Fire TASK_SKIPPED. Unlike the other task events this cannot come from
        AutoLamellaTask, because the skip is decided before any task object exists —
        so there is no task_state to attach, and no task_id: nothing ran to have one.
        item_id is empty on the lamella-not-found path, where there is no lamella."""
        fire_event(
            self.hook_manager,
            HookEvent.TASK_SKIPPED,
            task_name=task_name,
            task_type=task_type,
            item_name=item_name,
            item_id=item_id,
            skip_reason=skip_reason,
            **self.hook_run_context(),
        )

    # --- Internal helpers ---

    def _wait_until_scheduled(self, scheduled_at: datetime, task_name: str,
                              lamella: 'Lamella') -> None:
        """Block until ``scheduled_at`` is reached, emitting a periodic countdown.

        The wait is interruptible: if ``stop()`` is called the loop exits early
        and the caller skips running the task.
        """
        # Times are naive-local throughout, but a hand-edited/externally-produced
        # protocol may carry a tz-aware scheduled_at. Normalize to naive-local so
        # the subtraction against datetime.now() below doesn't raise TypeError.
        if scheduled_at.tzinfo is not None:
            scheduled_at = scheduled_at.astimezone().replace(tzinfo=None)
        target_str = scheduled_at.strftime(DATETIME_DISPLAY_AMPM)
        next_update = 0.0  # force an immediate first message
        while not self.is_stopped:
            remaining = (scheduled_at - datetime.now()).total_seconds()
            if remaining <= 0:
                break

            now = time.monotonic()
            if now >= next_update:
                msg = (f"Waiting until {target_str} to start {task_name} on "
                       f"{lamella.name} ({format_time_remaining(remaining)} remaining).")
                update_status_ui(self.parent_ui, "", status_bar=msg)
                next_update = now + 15.0

            time.sleep(min(1.0, remaining))

    def _should_skip(self, lamella: 'Lamella', task_name: str) -> Optional[str]:
        """Return skip reason string, or None if task should run.

        Deliberately does not re-check the run's lamella selection: that filter
        is applied when the queue is built, so anything that reaches here is
        wanted — including items the user added mid-run, which are not in the
        original selection and were previously skipped as "not_required".
        """
        if lamella.is_failure:
            logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Marked as failure or has defect.")
            return "failure"

        task_requirements = self.experiment.task_protocol.workflow_config.requirements(task_name)
        if task_requirements and not all(lamella.has_completed_task(req) for req in task_requirements):
            logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Required tasks {task_requirements} not completed.")
            return "missing_prereqs"

        return None

    def _emit_status(self, item: WorkItem, lamella: 'Lamella',
                     status: 'AutoLamellaTaskStatus',
                     msg: str = "",
                     error_message: Optional[str] = None,
                     task_duration: Optional[float] = None,
                     skip_reason: Optional[str] = None) -> None:
        """Emit workflow_update_signal with the standard status dict.

        This stream and the hook stream describe the same lifecycle from two different
        brackets — the manager brackets the task *call*, the task brackets its own
        *execution* — and they are deliberately not merged. Hooks run user-configurable
        code synchronously on the worker thread and HookManager.fire swallows what it
        raises; the UI needs neither of those properties. See FIB-464.

        What the two must agree on is vocabulary: the item is item_name here as it is in
        HookContext. The queue position fields below have no hook counterpart and stay —
        they are timeline positioning, not lifecycle facts.
        """
        if self.parent_ui is None:
            return

        # Progress is measured against the live queue rather than the launch
        # plan. A position in the original task x lamella matrix has no answer
        # for an item the user added mid-run, and stops being meaningful for
        # everything else as soon as the queue is reordered.
        items = self.queue.items
        position = next((i for i, it in enumerate(items) if it.id == item.id), None)

        status_dict = {
            "task_name": item.task_name,
            "item_name": lamella.name,
            # Deprecated alias for item_name, kept for consumers outside this repo.
            # Drop alongside the HookContext shims after v0.6.
            "lamella_name": lamella.name,
            "status": status,
            "timestamp": time.time(),
            "error_message": error_message,
            "task_duration": task_duration,
            "skip_reason": skip_reason,
            "queue_position": position + 1 if position is not None else None,
            "queue_total": len(items),
            "queue_items": items,  # thread-safe snapshot
            # The plan this run was launched with. Informational only — the live
            # queue may since have diverged from it.
            "task_names": self.queue.task_names,
            "lamella_names": self.queue.lamella_names,
        }

        self.parent_ui.workflow_update_signal.emit({"msg": msg, "status": status_dict})

    def _run_single_task(self, task_name: str, lamella: 'Lamella') -> Optional[Exception]:
        """Execute one task for one lamella. Returns exception or None."""
        try:
            run_task(microscope=self.microscope,
                     task_name=task_name,
                     lamella=lamella,
                     parent_ui=self.parent_ui,
                     task_manager=self)
            self.experiment.save()
            return None
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
            if self.is_stopped or isinstance(e, (OperationCancelledError, InterruptedError)):
                logging.info(f"Task {task_name} for lamella {lamella.name} cancelled by user.")
                lamella.task_state.status = AutoLamellaTaskStatus.Cancelled
                lamella.task_state.status_message = "Cancelled by user."
            else:
                logging.warning(f"Error running task {task_name} for lamella {lamella.name}: {e}")
                lamella.task_state.status = AutoLamellaTaskStatus.Failed
                lamella.task_state.status_message = str(e)
            self.experiment.save()
            return e


def run_tasks(microscope: FibsemMicroscope,
            experiment: 'Experiment',
            task_names: List[str],
            required_lamella: Optional[List[str]] = None,
            parent_ui: Optional['AutoLamellaUI'] = None,
            hook_manager: Optional[HookManager] = None) -> None:
    """Run the specified tasks for all lamellas in the experiment.
    Thin wrapper around TaskManager for backward compatibility and headless usage.
    """
    manager = TaskManager(microscope, experiment, parent_ui, hook_manager=hook_manager)
    manager.run(task_names, required_lamella)
