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
on grid 3 of 12. A failed task fails only itself and what ``requires`` it; the
next task on the same grid still runs, since an SEM overview failing says nothing
about the FM one.

Decisions follow the lamella workflow's rules. A task waits while a task it
requires awaits a decision in the Review tab or is still queued for the grid,
and is skipped when one did not complete. While a grid waits, the run moves on
to the next grid and comes back once the decision lands.

Where Stop lands
----------------
The hardware calls are atomic; Stop is honoured at the checkpoints between them.

* Between queue items: the loop asks ``is_stopped`` before taking the next one.
* Inside a task: at the task's own checkpoints -- after its stage move, between
  tiles, inside a focus sweep. The task ends Cancelled and the loop ends the run.
* Inside an exchange: before the unload, and between the unload and the load.
  A Stop that lands during the unload leaves the working slot empty and the
  next grid in the magazine; the load entry says so. A Stop that lands during
  the load is honoured once the grid is in, the last checkpoint an exchange has.
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
from fibsem.applications.autolamella.workflows.ui import update_status_ui
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
# switched off -- a task whose grid is not loaded loads it anyway, so
# removing or reordering a load item changes what is shown, never what runs.
# Readable where it shows: the queue, the timeline, the summary, the history.
LOAD_ENTRY_NAME = "Load grid"
LOAD_TASK_TYPE = "LOAD_GRID"

# Skip reasons, in the vocabulary TASK_SKIPPED hooks and status reports carry.
SKIP_GRID_NOT_FOUND = "grid_not_found"
SKIP_GRID_NOT_LOADED = "grid_not_loaded"
SKIP_MISSING_PREREQS = "missing_prereqs"  # the lamella manager's word for it
SKIP_NOTHING_TO_RUN = "nothing_to_run"  # a load with no runnable task behind it


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

    ITEM_NOUN = "grids"

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        experiment: "Experiment",
        parent_ui: Optional["AutoLamellaUI"] = None,
        hook_manager: Optional[HookManager] = None,
    ):
        super().__init__(microscope, experiment, parent_ui, hook_manager)
        # Grids this run could not bring onto the stage, and why. One attempt per
        # grid per run: an exchange that failed once is not retried on the next
        # task, which would only re-run the same failure in front of a queue of
        # grids that might load fine.
        self._not_loaded: Dict[str, str] = {}

    # --- Public API ---

    def run(
        self, task_names: List[str], grid_names: Optional[List[str]] = None
    ) -> None:
        """Run ``task_names``, in order, on each of ``grid_names`` (all grids if None).

        A task that is AwaitingDecision on a grid is left out for that grid, as
        on the lamella side: its run is over and its record waits in the Review
        tab, and running it again would supersede the proposal someone is about
        to decide. With the Review surface off there is no way to decide, so
        Run re-runs it."""
        if grid_names is None:
            grid_names = [g.name for g in self.experiment.grids]
        self.queue.build_from_pairs(
            [
                (grid, step)
                for grid, step in plan_grid_run(task_names, grid_names)
                if not (self.review_enabled and self._awaiting_decision(grid, step))
            ],
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
                    # A task that never ran has no answer to "was the grid in
                    # the beam for it": blank, not the last load's outcome.
                    "loaded": (
                        None
                        if item.status is AutoLamellaTaskStatus.NotStarted
                        else item.item_name not in self._not_loaded
                    ),
                    "completed_at": completed_at,
                    "duration": duration,
                }
            )
        return pd.DataFrame(rows)

    # --- The run loop ---

    def _awaiting_decision(self, grid_name: str, task_name: str) -> bool:
        grid = self.experiment.get_grid_by_name(grid_name)
        return grid is not None and grid.is_awaiting_decision(task_name)

    def _requirements(self, task_name: str) -> List[str]:
        try:
            return self.experiment.grid_protocol.requirements(task_name)
        except ValueError:  # no task protocol on this experiment
            return []

    def _defer_reason(self, grid: GridRecord, task_name: str) -> Optional[str]:
        """Why this task cannot run on this grid *yet*, as on the lamella side:
        a task it requires awaits a decision (``awaiting_decision``), or is still
        queued for this grid (``prereq_pending``). The run moves on to the next
        grid meanwhile, and comes back -- an exchange -- when it can run."""
        for req in self._requirements(task_name):
            # a rerun still queued is the attempt that counts, whatever an
            # earlier run of it did
            if self.queue.has_pending_pair(grid.name, req):
                return "prereq_pending"
            if grid.is_awaiting_decision(req):
                return "awaiting_decision"
        return None

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        grid = self.experiment.get_grid_by_name(item.item_name)
        if grid is None:
            return None  # let the loop retire it with a reason
        if item.task_name == LOAD_ENTRY_NAME:
            return self._load_defer_reason(grid)
        return self._defer_reason(grid, item.task_name)

    def _pending_tasks(self, grid: GridRecord) -> List[WorkItem]:
        return [
            i
            for i in self.queue.pending
            if i.item_name == grid.name and i.task_name != LOAD_ENTRY_NAME
        ]

    def _runnable_now(self, grid: GridRecord, task_name: str) -> bool:
        return self._defer_reason(
            grid, task_name
        ) is None and not self._missing_requirements(grid, task_name)

    def _load_defer_reason(self, grid: GridRecord) -> Optional[str]:
        """An exchange is the expensive step, so a grid is loaded for work that
        can run now, not for work still waiting (FIB-1005). The load waits while
        every task queued for the grid waits on a decision or a queued
        requirement, and goes ahead once one can run. A load with no tasks
        queued behind it is a load someone asked for, and runs."""
        pending = self._pending_tasks(grid)
        if not pending or any(self._runnable_now(grid, i.task_name) for i in pending):
            return None
        if any(self._defer_reason(grid, i.task_name) for i in pending):
            return "waiting_for_work"
        return None  # every task will be skipped: the load step retires itself

    def _missing_requirements(self, grid: GridRecord, task_name: str) -> List[str]:
        """Required tasks whose latest run on this grid did not complete: failed,
        rejected, cancelled, or never run. An older success does not count.
        Terminal for this run, as on the lamella side."""
        return [
            req
            for req in self._requirements(task_name)
            if not grid.latest_run_completed(req)
        ]

    def _run_queue(self) -> None:
        self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
        self.experiment.decided.connect(self._on_decided)
        try:
            self._run_items()
        finally:
            self.experiment.decided.disconnect(self._on_decided)

    def _requirements_of(self, task_name: str) -> List[str]:
        return self._requirements(task_name)

    def _run_items(self) -> None:
        while not self.is_stopped:
            # Cleared before the scan, so a decision that lands between a scan
            # finding nothing and the wait starting is not lost.
            self._decision_event.clear()
            item = self.queue.next(skip=self._is_deferred)
            if item is None:
                if self.queue.is_empty:
                    break
                if self._wait_for_a_decision():
                    continue
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

            # Before the load: a task that cannot use what it requires is not
            # worth an exchange.
            missing = self._missing_requirements(grid, item.task_name)
            if missing:
                msg = (
                    f"Skipping {item.task_name} on {grid.name}: required "
                    f"{', '.join(missing)} did not complete."
                )
                logging.info(msg)
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_report(
                    item=item,
                    item_name=grid.name,
                    status=AutoLamellaTaskStatus.Skipped,
                    msg=msg,
                    skip_reason=SKIP_MISSING_PREREQS,
                )
                self._fire_skipped_hook(
                    item.task_name,
                    grid.name,
                    SKIP_MISSING_PREREQS,
                    task_type=self._task_type(item.task_name),
                    item_id=grid.id,
                )
                continue

            self._expire_what_this_consumes(grid.id, item.task_name)

            try:
                loaded = self._ensure_loaded(grid)
            except OperationCancelledError as e:
                self._cancel_load(item, grid, str(e))
                continue
            if not loaded:
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
            if err is None and final_status is AutoLamellaTaskStatus.AwaitingDecision:
                msg = (
                    f"{item.task_name} on grid {grid.name} awaits a decision "
                    "in the Review tab."
                )
            elif err is None:
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
            self._say(workflow_info="Grid workflow cancelled by user.")
        elif self.stalled:
            # Drained with work waiting on decisions, and the wait ran out: not
            # completed, not cancelled. The next Run picks up from there.
            self._report_stall()
        else:
            self._fire_workflow_hook(HookEvent.WORKFLOW_COMPLETED)
            self._say(workflow_info=self._completion_message())
        for line in self._grid_summary_lines():
            logging.info(line)

    def _run_load_step(self, item: WorkItem, grid: GridRecord) -> None:
        """The planned exchange. Its outcome is the queue item's status, so the
        timeline shows a grid that would not load where it failed. Skipped,
        with no exchange, when every task queued for the grid is going to be
        skipped for a requirement that did not complete (FIB-1005)."""
        pending = self._pending_tasks(grid)
        if pending and not any(self._runnable_now(grid, i.task_name) for i in pending):
            msg = (
                f"Not loading grid {grid.name}: none of its selected tasks can run "
                "(a task they require did not complete)."
            )
            logging.info(msg)
            self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
            self._emit_report(
                item=item,
                item_name=grid.name,
                status=AutoLamellaTaskStatus.Skipped,
                msg=msg,
                skip_reason=SKIP_NOTHING_TO_RUN,
            )
            return
        self._emit_report(
            item=item,
            item_name=grid.name,
            status=AutoLamellaTaskStatus.InProgress,
            msg=f"Loading grid {grid.name}.",
        )
        try:
            loaded = self._ensure_loaded(grid)
        except OperationCancelledError as e:
            self._cancel_load(item, grid, str(e))
            return
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
                f"Grid {grid.name} is loaded."
                if loaded
                else f"Grid {grid.name} could not be loaded."
            ),
        )

    def _cancel_load(self, item: WorkItem, grid: GridRecord, msg: str) -> None:
        """A Stop landed inside the exchange: the queue item ends Cancelled, not
        Failed, and the loop ends the run at its next check."""
        logging.info(msg)
        self.queue.mark_done(item, AutoLamellaTaskStatus.Cancelled)
        self._emit_report(
            item=item,
            item_name=grid.name,
            status=AutoLamellaTaskStatus.Cancelled,
            msg=msg,
        )

    def _ensure_loaded(self, grid: GridRecord) -> bool:
        """Bring the grid onto the stage, recording the attempt when it costs one.

        A grid already in a holder slot is confirmed, not exchanged, and leaves no
        entry: the second task on a grid is not a second load. An exchange, or a
        refusal, is recorded on the grid's history as a ``load`` entry with how
        long it took or why it did not happen. A failure is remembered for the
        rest of the run so the grid's other tasks skip without retrying it.

        An exchange is two hardware calls, the unload and the load, and Stop is
        honoured before each: a Stop that arrives while the working slot is being
        emptied ends the run with the slot empty, rather than loading a grid the
        operator has just asked not to have loaded. Raises
        ``OperationCancelledError`` at either checkpoint, with the entry recorded.
        """
        if grid.name in self._not_loaded:
            return False
        stage = self.microscope._stage
        loaded = stage.holder.find_slot_by_grid_name(grid.name) is not None
        entry = AutoLamellaTaskState(
            name=LOAD_ENTRY_NAME,
            task_type=LOAD_TASK_TYPE,
            status=AutoLamellaTaskStatus.InProgress,
        )
        try:
            if not loaded:
                self._stop_exchange_if_asked(
                    grid, entry, f"Stopped before loading {grid.name}."
                )
                occupant = self._working_slot_occupant(stage)
                if occupant is not None:
                    logging.info(f"Unloading grid {occupant} for {grid.name}.")
                    self._say(status_bar=f"Unloading grid {occupant}...")
                    stage.unload()
                    self._stop_exchange_if_asked(
                        grid,
                        entry,
                        f"Stopped after unloading {occupant}; "
                        f"{grid.name} was not loaded.",
                    )
                logging.info(f"Loading grid {grid.name}.")
                self._say(status_bar=f"Loading grid {grid.name}...")
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
        if not loaded:
            entry.status = AutoLamellaTaskStatus.Completed
            entry.status_message = f"Loaded into {slot.name}."
            entry.end_timestamp = datetime.timestamp(datetime.now())
            grid.task_history.append(entry)
            self.experiment.save()
            logging.info(
                f"Grid {grid.name} loaded into {slot.name} in {entry.duration:.1f} s."
            )
        return True

    @staticmethod
    def _working_slot_occupant(stage) -> Optional[str]:
        """The name of the grid an exchange would have to retract first, or None.
        Only a loader retracts anything; a fixed holder has nowhere to put it."""
        if stage.loader is None:
            return None
        occupant = stage.loader.working_slot.loaded_grid
        return occupant.name if occupant is not None else None

    def _stop_exchange_if_asked(
        self, grid: GridRecord, entry: AutoLamellaTaskState, msg: str
    ) -> None:
        """A checkpoint inside the exchange: record the entry and raise if Stop
        has been asked for since the loop last looked."""
        if not self.is_stopped:
            return
        entry.status = AutoLamellaTaskStatus.Cancelled
        entry.status_message = msg
        entry.end_timestamp = datetime.timestamp(datetime.now())
        grid.task_history.append(entry)
        self.experiment.save()
        raise OperationCancelledError(msg)

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
        """A line for the workflow-information label or the status bar, that
        gets out after a Stop too: the closing "cancelled" line is the point."""
        update_status_ui(
            self.parent_ui,
            "",
            workflow_info=workflow_info,
            status_bar=status_bar,
            check_abort=False,
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
