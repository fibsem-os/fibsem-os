"""Grid task execution manager.

Drives a (grid x task) work matrix in **grid-outer** order: all of a grid's
tasks run before moving to the next grid, so a robotic grid exchange is
amortised across the grid's whole task group. Before each grid's tasks, the
grid is brought into the holder's working slot via ``ensure_loaded`` — a no-op
on a static shuttle (the grid is already in a slot), or a loader exchange on an
autoloader. An exchange failure halts the workflow and raises.

This mirrors the lamella ``TaskManager`` and reuses the shared ``TaskQueue``.
The two managers are kept parallel for now; a shared ``BaseTaskManager`` is to
be extracted once both are concrete (see the design doc, §8 Decisions).
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, GridRecord
from fibsem.applications.autolamella.workflows.tasks.grid import run_grid_task
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import GridSlot, SampleGrid

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.hooks import HookManager


class GridExchangeError(RuntimeError):
    """Raised when a grid cannot be brought into the working slot."""


class GridTaskManager:
    """Manages execution of grid tasks across the grids in an experiment."""

    def __init__(self,
                 microscope: FibsemMicroscope,
                 experiment: 'Experiment',
                 parent_ui: Optional['AutoLamellaUI'] = None,
                 hook_manager: Optional['HookManager'] = None):
        self.microscope = microscope
        self.experiment = experiment
        self.parent_ui = parent_ui
        self.hook_manager = hook_manager
        self._stop_event = threading.Event()
        self.queue = TaskQueue()
        self._task_names: List[str] = []
        self._grid_names: List[str] = []

    # --- public API ---

    def run(self, task_names: List[str],
            grid_names: Optional[List[str]] = None) -> None:
        """Run the given tasks for the given grids (all grids if None).

        Grid-outer ordering: each grid is loaded once and runs its full task
        group before the next grid.
        """
        if grid_names is None:
            grid_names = [g.name for g in self.experiment.grids]

        self._task_names = list(task_names)
        self._grid_names = list(grid_names)
        self.queue.build_from_matrix(task_names, grid_names, unit_outer=True)
        self._run_queue(required=grid_names)

    def stop(self) -> None:
        """Signal the manager to stop after the current task completes."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # --- hardware: bring a grid into the working slot ---

    def ensure_loaded(self, record: GridRecord) -> GridSlot:
        """Ensure ``record``'s grid sits in the holder working slot.

        No-op (returns the slot) when the grid is already loaded — the static
        shuttle case. On an autoloader, unloads whatever occupies the working
        slot and loads the requested grid. Raises ``GridExchangeError`` on
        failure, which halts the workflow.
        """
        holder = self.microscope._stage.holder
        slot = holder.find_slot_by_grid_name(record.name)
        if slot is not None:
            return slot

        loader = self.microscope._stage.loader
        if loader is None:
            raise GridExchangeError(
                f"Grid '{record.name}' is not loaded and there is no loader to load it."
            )
        # source the real grid (with geometry) from the magazine; fall back to a
        # bare grid only if the record has no magazine entry
        magazine_slot = loader.find_grid(record.name)
        grid = magazine_slot.loaded_grid if magazine_slot is not None else SampleGrid(name=record.name)
        try:
            for loaded in loader.loaded_slots:
                loader.unload_grid(loaded.name)
            target = next(iter(holder.slots))
            loader.load_grid(target, grid)
        except Exception as e:  # noqa: BLE001 - re-raised as a halt
            raise GridExchangeError(
                f"Failed to exchange grid '{record.name}' into the working slot: {e}"
            ) from e
        return holder.slots[target]

    # --- internals ---

    def _run_queue(self, required: List[str]) -> None:
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            record = self.experiment.get_grid_by_name(item.item_name)
            if record is None:
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue

            skip_reason = self._should_skip(record, item.task_name, required)
            if skip_reason is not None:
                logging.info(
                    f"Skipping grid {record.name} for {item.task_name}: {skip_reason}."
                )
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_status(record.name, item.task_name, "Skipped",
                                  skip_reason=skip_reason)
                continue

            # bring the grid into the working slot (halts the workflow on failure)
            self.ensure_loaded(record)

            self._emit_status(record.name, item.task_name, "InProgress",
                              msg=f"Running {item.task_name} on {record.name}")
            self._run_single_task(item.task_name, record)
            self.queue.mark_done(item, record.task_state.status)
            self._emit_status(record.name, item.task_name,
                              record.task_state.status.name,
                              error_message=record.task_state.status_message or None)

        logging.info("Grid workflow complete.\n%s", self.queue.items)
        self._emit_status("", "", "Completed", msg="Grid workflow complete.")
        print(self.experiment.grid_task_history_dataframe())

    def _emit_status(self, grid_name: str, task_name: str, status: str, *,
                     msg: Optional[str] = None, error_message: Optional[str] = None,
                     skip_reason: Optional[str] = None) -> None:
        """Emit grid_workflow_update_signal (thread → GUI). No-op without a UI."""
        if self.parent_ui is None:
            return
        tasks, grids = self._task_names, self._grid_names
        status_dict = {
            "item_name": grid_name,   # neutral key shared with the timeline
            "grid_name": grid_name,   # grid-specific alias for grid consumers
            "task_name": task_name,
            "status": status,
            "timestamp": time.time(),
            "error_message": error_message,
            "skip_reason": skip_reason,
            "task_names": tasks,
            "total_tasks": len(tasks),
            "current_task_index": tasks.index(task_name) if task_name in tasks else None,
            "item_names": grids,
            "total_items": len(grids),
            "current_item_index": grids.index(grid_name) if grid_name in grids else None,
            "queue_items": self.queue.items,
        }
        self.parent_ui.grid_workflow_update_signal.emit({"msg": msg or "", "status": status_dict})

    def _should_skip(self, record: GridRecord, task_name: str,
                     required: List[str]) -> Optional[str]:
        if required and record.name not in required:
            return "not_required"
        # NOTE: a failed task does NOT fail the grid — grid tasks are independent,
        # so a failure is recorded per-task (task_state/history) but the grid's
        # other tasks (and other grids) still run.
        return None

    def _run_single_task(self, task_name: str, record: GridRecord) -> Optional[Exception]:
        try:
            run_grid_task(self.microscope, task_name, self.experiment, record,
                          parent_ui=self.parent_ui, task_manager=self)
            return None
        except Exception as e:
            logging.error(
                f"Error running grid task {task_name} for grid {record.name}: {e}",
                exc_info=True,
            )
            # task.on_failure already recorded Failed state; persist it
            self.experiment.save()
            return e


def run_grid_workflow(microscope: FibsemMicroscope,
                      experiment: 'Experiment',
                      task_names: List[str],
                      grid_names: Optional[List[str]] = None,
                      parent_ui: Optional['AutoLamellaUI'] = None) -> None:
    """Thin wrapper around GridTaskManager for headless / scripted use."""
    GridTaskManager(microscope, experiment, parent_ui).run(task_names, grid_names)
