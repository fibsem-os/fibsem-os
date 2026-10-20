"""AutoLamella's lamella workflow: its task manager and how a run is started."""

import logging
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from fibsem.acting import ActorKind
from fibsem.constants import DATETIME_DISPLAY_AMPM
from fibsem.hooks import HookEvent, HookManager, fire_event
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import InsertableDeviceState
from fibsem.util.durations import format_duration_rounded
from fibsem.workflows.interaction import Responder
from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.queue import WorkItem
from fibsem.workflows.status import TaskStatus, update_status_ui

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella


def run_task(
    microscope: FibsemMicroscope,
    task_name: str,
    lamella: "Lamella",
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
        task_manager=task_manager,
    )
    task.run()


class TaskManager(BaseTaskManager):
    """Manages execution of autolamella tasks across lamellas."""

    RUN_DESCRIPTION = "a lamella run"

    def __init__(
        self,
        microscope: FibsemMicroscope,
        experiment: "Experiment",
        hook_manager: Optional[HookManager] = None,
        responder: Optional[Responder] = None,
        deciders: Optional[Callable[[], Tuple[ActorKind, ...]]] = None,
        images: Optional[Any] = None,
    ):
        super().__init__(
            microscope,
            experiment,
            hook_manager,
            responder,
            deciders,
            images,
        )
        # Lamellae already finished when the run started, so re-running a task on
        # finished work does not re-announce it. Seeded in _before_run.
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
        that is how a task is re-run.
        """
        if required_lamella is None:
            required_lamella = [p.name for p in self.experiment.positions]
        pairs = [(name, task) for task in task_names for name in required_lamella]
        self.queue.build_from_pairs(pairs)
        self._run_queue()

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        lamella = self.experiment.get_lamella_by_name(item.item_name)
        if lamella is None or lamella.is_failure:
            return None  # let the loop retire it with a reason
        return self._defer_reason(lamella, item.task_name)

    TASK_LINES = (
        "Starting task {task} for Lamella {item}.",
        "Completed task {task} for Lamella {item}.",
        "Error in task {task} for Lamella {item}.",
    )

    def _before_run(self) -> None:
        self._snapshot_completion()

    def _record_for(self, item: WorkItem) -> Optional["Lamella"]:
        lamella = self.experiment.get_lamella_by_name(item.item_name)
        if lamella is None:
            logging.warning(
                f"Skipping {item.task_name}: no lamella named {item.item_name} in the experiment."
            )
            self.queue.mark_done(item, TaskStatus.Skipped)
            self._fire_skipped_hook(item.task_name, item.item_name, "lamella_not_found")
        return lamella

    def _ready(self, item: WorkItem, lamella: "Lamella") -> bool:
        skip_reason = self._should_skip(lamella, item.task_name)
        if skip_reason is not None:
            self.queue.mark_done(item, TaskStatus.Skipped)
            self._emit_report(
                item=item,
                item_name=lamella.name,
                status=TaskStatus.Skipped,
                msg=f"Skipping {lamella.name} for {item.task_name}: {skip_reason}.",
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
            return False

        # Wait until the task's scheduled start time, if set and in the future.
        scheduled_at = self.experiment.task_protocol.workflow_config.get_scheduled_at(
            item.task_name
        )
        if scheduled_at is not None:
            self._wait_until_scheduled(scheduled_at, item.task_name, lamella)
        return not self.is_stopped

    def _run_task(self, task_name: str, lamella: "Lamella") -> None:
        run_task(
            microscope=self.microscope,
            task_name=task_name,
            lamella=lamella,
            task_manager=self,
        )

    def _after_task(self, item: WorkItem, lamella: "Lamella") -> None:
        self._maybe_fire_lamella_completed(lamella, item.task_name)

    def _end_run(self) -> None:
        # if the objective is inserted, retract for safety
        if (
            self.microscope.fm is not None
            and self.microscope.fm.objective.state is InsertableDeviceState.INSERTED
        ):
            self.microscope.fm.objective.retract()
        super()._end_run()

    def _complete(self) -> None:
        self._fire_workflow_hook(
            HookEvent.WORKFLOW_COMPLETED, task_counts=self._task_counts()
        )
        self._say(workflow_info=self._completion_message())
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
        completion, only quieter. Warns as well as returning, because the status line
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
        ended any other way (cancelled) is in none of them."""
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
        self._set_workflow_pending(True)
        try:
            while not self.should_abort:
                remaining = (scheduled_at - datetime.now()).total_seconds()
                if remaining <= 0:
                    break
                msg = (
                    f"Waiting until {target_str} to start {task_name} on "
                    f"{lamella.name} ({format_duration_rounded(remaining)} remaining)."
                )
                update_status_ui(self, "", status_bar=msg)
                # Either stop wakes this at once.
                self.abort_token.wait(min(15.0, remaining))
        finally:
            # The time arriving, a stop, and an exception all have to clear this.
            # A stranded flag would leave the border reading "nothing is running"
            # for the rest of the run.
            self._set_workflow_pending(False)

    def _defer_reason(self, lamella: "Lamella", task_name: str) -> Optional[str]:
        """Why this task cannot run *yet* -- as opposed to _should_skip, which
        says why it never will this run.

        The prerequisite check used to be terminal: unmet meant Skipped, done for
        the run. That was right only because ordering guaranteed the prerequisite
        had already had its turn. A re-run or a mid-run insertion breaks the
        guarantee, and is answered by looking at the queue rather than at
        history: ``prereq_pending``, a required task is still in the queue
        ahead or behind. It will get its turn.

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
