"""The grid workflow's run loop: grid-outer, and a grid that will not load is skipped.

Grid-outer and dumb. For each grid in turn: make it reachable, run the selected
tasks on it in order, save, next grid. Reaching a grid is the one expensive step
-- on an autoloader it is a magazine exchange -- so the queue is built item-outer
and a grid is exchanged once for all of its tasks.

Three questions with three separate answers, never collapsed into one:

* *Did the grid load?* A ``load`` entry on the grid's history, Completed or Failed
  with the hardware's message, written here per attempt.
* *Did each task complete?* One history entry per task, written by the task's own
  lifecycle: Completed, Failed, Cancelled, or (from here) Skipped.
* *Is the grid any good?* ``GridRecord.quality``, set by a person, never here.

A failed load records the failure, skips the grid's remaining tasks as "grid not
loaded", and the run continues with the next grid: an overnight run must not stop
on grid 3 of 12. A failed task fails only itself; the next task on the same grid
still runs, since an SEM overview failing says nothing about the FM one.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    GridRecord,
)
from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
    run_grid_task,
)
from fibsem.applications.autolamella.workflows.tasks.manager import BaseTaskManager
from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem
from fibsem.applications.autolamella.workflows.tasks.status import WorkflowStatusEvent
from fibsem.cancellation import OperationCancelledError
from fibsem.hooks import HookEvent, HookManager
from fibsem.microscopes._stage import GridExchangeError

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.microscope import FibsemMicroscope

# The load step: a queue item ahead of each grid's tasks, so the plan shows where
# the exchanges fall and the timeline shows how each one went, and the history
# entry the exchange leaves on the grid. Not a task in the protocol: loading is
# the manager's job. It is in the queue so it can be *seen*, not so it can be
# switched off -- a task whose grid is not in the beam loads it anyway, so
# removing or reordering a load item changes what is shown, never what runs.
LOAD_ENTRY_NAME = "load"
LOAD_TASK_TYPE = "LOAD_GRID"

# Skip reasons, in the vocabulary TASK_SKIPPED hooks and status reports carry.
SKIP_GRID_NOT_FOUND = "grid_not_found"
SKIP_GRID_NOT_LOADED = "grid_not_loaded"


def plan_grid_run(
    task_names: List[str], grid_names: List[str]
) -> List[Tuple[str, str]]:
    """The ``(grid, step)`` sequence a run would execute: grid-outer, each grid's
    load first, then its tasks in order. What a run preview shows."""
    return [
        (grid, step) for grid in grid_names for step in [LOAD_ENTRY_NAME, *task_names]
    ]


class GridTaskManager(BaseTaskManager):
    """Runs grid tasks over the experiment's grids, one grid at a time."""

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        experiment: "Experiment",
        parent_ui: Optional["AutoLamellaUI"] = None,
        hook_manager: Optional[HookManager] = None,
    ):
        super().__init__(microscope, experiment, parent_ui, hook_manager)
        # Grids this run could not bring into the beam, and why. One attempt per
        # grid per run: an exchange that failed once is not retried on the next
        # task, which would only re-run the same failure in front of a queue of
        # grids that might load fine.
        self._not_loaded: Dict[str, str] = {}

    # --- Public API ---

    def run(
        self, task_names: List[str], grid_names: Optional[List[str]] = None
    ) -> None:
        """Run ``task_names``, in order, on each of ``grid_names`` (all grids if None)."""
        if grid_names is None:
            grid_names = [g.name for g in self.experiment.grids]
        self.queue.build_from_pairs(
            plan_grid_run(task_names, grid_names),
            task_names=task_names,
            item_names=grid_names,
        )
        self._run_queue()

    def build_run_summary_dataframe(self) -> pd.DataFrame:
        """One row per (grid, task) attempted in this run, skipped tasks included.

        ``loaded`` is the answer to the first question: whether the grid was in
        the beam for this row's task. Completion time and duration come from the
        grid's history where a task ran, and are blank where it did not.
        """
        rows: List[dict] = []
        for item in self.queue.items:
            grid = self.experiment.get_grid_by_name(item.item_name)
            completed_at = ""
            duration = None
            if grid is not None and item.status not in (
                AutoLamellaTaskStatus.Skipped,
                AutoLamellaTaskStatus.NotStarted,
            ):
                for task in reversed(grid.task_history):
                    if task.name == item.task_name:
                        completed_at = task.completed_at
                        duration = task.duration
                        break
            rows.append(
                {
                    "grid_name": item.item_name,
                    "task_name": item.task_name,
                    "task_status": item.status.name,
                    "loaded": item.item_name not in self._not_loaded,
                    "completed_at": completed_at,
                    "duration": duration,
                }
            )
        return pd.DataFrame(rows)

    # --- The run loop ---

    def _run_queue(self) -> None:
        self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            # A stop_task click that landed between two tasks was aimed at the one
            # that has just finished, not at this one.
            self._task_stop_event.clear()

            grid = self.experiment.get_grid_by_name(item.item_name)
            if grid is None:
                msg = (
                    f"Skipping {item.task_name}: no grid named {item.item_name} "
                    "in the experiment."
                )
                logging.warning(msg)
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_report(
                    item=item,
                    item_name=item.item_name,
                    status=AutoLamellaTaskStatus.Skipped,
                    msg=msg,
                    skip_reason=SKIP_GRID_NOT_FOUND,
                )
                self._fire_skipped_hook(
                    item.task_name, item.item_name, SKIP_GRID_NOT_FOUND
                )
                continue

            if item.task_name == LOAD_ENTRY_NAME:
                self._run_load_step(item, grid)
                continue

            if not self._ensure_loaded(grid):
                reason = self._not_loaded[grid.name]
                msg = f"Skipping {item.task_name} on {grid.name}: grid not loaded."
                logging.info(f"{msg} {reason}")
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_report(
                    item=item,
                    item_name=grid.name,
                    status=AutoLamellaTaskStatus.Skipped,
                    msg=msg,
                    error_message=reason,
                    skip_reason=SKIP_GRID_NOT_LOADED,
                )
                self._fire_skipped_hook(
                    item.task_name,
                    grid.name,
                    SKIP_GRID_NOT_LOADED,
                    task_type=self._task_type(item.task_name),
                    item_id=grid.id,
                )
                continue

            self._emit_report(
                item=item,
                item_name=grid.name,
                status=AutoLamellaTaskStatus.InProgress,
                msg=f"Starting {item.task_name} on grid {grid.name}.",
            )
            err = self._run_single_task(item.task_name, grid)
            final_status = grid.task_state.status
            self.queue.mark_done(item, final_status)
            if err is None:
                msg = f"Completed {item.task_name} on grid {grid.name}."
            else:
                msg = f"Error in {item.task_name} on grid {grid.name}."
            self._emit_report(
                item=item,
                item_name=grid.name,
                status=final_status,
                error_message=grid.task_state.status_message,
                task_duration=grid.task_state.duration,
                msg=msg,
            )

        # The loop exits when the queue drains *or* on Stop; only one of those is a
        # finished workflow.
        if self.is_stopped:
            self._fire_workflow_hook(HookEvent.WORKFLOW_CANCELLED)
            self._say(workflow_info="Grid workflow cancelled.")
        else:
            self._fire_workflow_hook(HookEvent.WORKFLOW_COMPLETED)
            self._say(workflow_info=self._completion_message())
        for line in self._grid_summary_lines():
            logging.info(line)

    def _run_load_step(self, item: WorkItem, grid: GridRecord) -> None:
        """The planned exchange. Its outcome is the queue item's status, so the
        timeline shows a grid that would not load where it failed."""
        self._emit_report(
            item=item,
            item_name=grid.name,
            status=AutoLamellaTaskStatus.InProgress,
            msg=f"Loading grid {grid.name}.",
        )
        loaded = self._ensure_loaded(grid)
        status = (
            AutoLamellaTaskStatus.Completed if loaded else AutoLamellaTaskStatus.Failed
        )
        self.queue.mark_done(item, status)
        self._emit_report(
            item=item,
            item_name=grid.name,
            status=status,
            error_message=None if loaded else self._not_loaded[grid.name],
            msg=(
                f"Grid {grid.name} is in the beam."
                if loaded
                else f"Grid {grid.name} could not be loaded."
            ),
        )

    def _ensure_loaded(self, grid: GridRecord) -> bool:
        """Bring the grid into the beam, recording the attempt when it costs one.

        A grid already in a holder slot is confirmed, not exchanged, and leaves no
        entry: the second task on a grid is not a second load. An exchange, or a
        refusal, is recorded on the grid's history as a ``load`` entry with how
        long it took or why it did not happen. A failure is remembered for the
        rest of the run so the grid's other tasks skip without retrying it.
        """
        if grid.name in self._not_loaded:
            return False
        stage = self.microscope._stage
        in_beam = stage.holder.find_slot_by_grid_name(grid.name) is not None
        entry = AutoLamellaTaskState(
            name=LOAD_ENTRY_NAME,
            task_type=LOAD_TASK_TYPE,
            status=AutoLamellaTaskStatus.InProgress,
        )
        if not in_beam:
            logging.info(f"Loading grid {grid.name}.")
            self._say(status_bar=f"Loading grid {grid.name}...")
        try:
            slot = stage.ensure_loaded(grid.name)
        except GridExchangeError as e:
            self._not_loaded[grid.name] = str(e)
            entry.status = AutoLamellaTaskStatus.Failed
            entry.status_message = str(e)
            entry.end_timestamp = datetime.timestamp(datetime.now())
            grid.task_history.append(entry)
            self.experiment.save()
            logging.warning(
                f"Grid {grid.name} could not be loaded: {e} "
                "Its remaining tasks in this run are skipped."
            )
            return False
        if not in_beam:
            entry.status = AutoLamellaTaskStatus.Completed
            entry.status_message = f"Loaded into {slot.name}."
            entry.end_timestamp = datetime.timestamp(datetime.now())
            grid.task_history.append(entry)
            self.experiment.save()
            logging.info(
                f"Grid {grid.name} loaded into {slot.name} in {entry.duration:.1f} s."
            )
        return True

    def _run_single_task(self, task_name: str, grid: GridRecord) -> Optional[Exception]:
        """Execute one task on one grid. Returns the exception, or None."""
        try:
            run_grid_task(
                self.microscope,
                task_name,
                self.experiment,
                grid,
                parent_ui=self.parent_ui,
                task_manager=self,
            )
            self.experiment.save()
            return None
        except Exception as e:
            # The task records its own outcome before re-raising; this is the
            # fallback for what fails before a task exists -- an unknown task name,
            # a construction failure. Same cancellation predicate as GridTask, so
            # the live state and the frozen history entry cannot disagree.
            if self.should_abort or isinstance(
                e, (OperationCancelledError, InterruptedError)
            ):
                logging.info(f"Task {task_name} on grid {grid.name} cancelled by user.")
                grid.task_state.status = AutoLamellaTaskStatus.Cancelled
                grid.task_state.status_message = "Cancelled by user."
            else:
                logging.warning(f"Error running {task_name} on grid {grid.name}: {e}")
                grid.task_state.status = AutoLamellaTaskStatus.Failed
                grid.task_state.status_message = str(e)
            self.experiment.save()
            return e

    # --- Reporting ---

    def _say(
        self, workflow_info: Optional[str] = None, status_bar: Optional[str] = None
    ) -> None:
        """A line for the workflow-information label or the status bar.

        Straight onto the status channel rather than through `update_status_ui`:
        that helper re-checks for abort and raises once Stop has been pressed,
        which is exactly when the closing "cancelled" line has to get out.
        """
        text = workflow_info or status_bar or ""
        if self.parent_ui is None:
            logging.info(text)
            return
        self.parent_ui.workflow_status_signal.emit(
            WorkflowStatusEvent(workflow_info=workflow_info, status_bar=status_bar)
        )

    def _task_type(self, task_name: str) -> str:
        try:
            config = self.experiment.grid_protocol.task_config.get(task_name)
        except ValueError:  # no task protocol on this experiment
            return ""
        return config.task_type if config is not None else ""

    def _grids_in_run(self) -> List[str]:
        """Every grid the queue holds, in run order: the launch plan plus anything
        added mid-run."""
        return list(dict.fromkeys(i.item_name for i in self.queue.items))

    def _completion_message(self) -> str:
        items = self.queue.items
        if not items:
            return "No tasks to run."
        grids = self._grids_in_run()
        not_loaded = [g for g in grids if g in self._not_loaded]
        failed = sum(
            1
            for i in items
            if i.status is AutoLamellaTaskStatus.Failed
            and i.task_name != LOAD_ENTRY_NAME
        )
        parts = [f"{len(grids) - len(not_loaded)} of {len(grids)} grids run"]
        if not_loaded:
            parts.append(f"{len(not_loaded)} could not be loaded")
        if failed:
            parts.append(f"{failed} task{'s' if failed != 1 else ''} failed")
        return "Grid workflow complete: " + ", ".join(parts) + "."

    def _grid_summary_lines(self) -> List[str]:
        """One line per grid: whether it loaded, and how its tasks ended."""
        lines = []
        items = self.queue.items
        for name in self._grids_in_run():
            if name in self._not_loaded:
                lines.append(f"{name}: not loaded ({self._not_loaded[name]})")
                continue
            outcomes = [
                i.status.name
                for i in items
                if i.item_name == name and i.task_name != LOAD_ENTRY_NAME
            ]
            counts = {s: outcomes.count(s) for s in dict.fromkeys(outcomes)}
            summary = ", ".join(f"{n} {s.lower()}" for s, n in counts.items())
            lines.append(f"{name}: {summary}")
        return lines


def run_grid_tasks(
    microscope: "FibsemMicroscope",
    experiment: "Experiment",
    task_names: Optional[List[str]] = None,
    grid_names: Optional[List[str]] = None,
    parent_ui: Optional["AutoLamellaUI"] = None,
    hook_manager: Optional[HookManager] = None,
) -> GridTaskManager:
    """Run grid tasks headless: the protocol's tasks in its order, on every grid,
    unless told otherwise. Returns the manager, for its queue and run summary."""
    if task_names is None:
        task_names = experiment.grid_protocol.ordered_task_names
    manager = GridTaskManager(microscope, experiment, parent_ui, hook_manager)
    manager.run(task_names, grid_names)
    return manager
