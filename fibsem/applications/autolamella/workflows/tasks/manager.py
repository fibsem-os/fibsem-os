"""Task execution manager for autolamella workflows."""

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.workflows.tasks.hooks import HookContext, HookEvent, HookManager
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.applications.autolamella.workflows.tasks.scheduling import (
    plan_for_run,
    should_skip,
)
from fibsem.applications.autolamella.workflows.ui import update_status_ui
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import GridExchangeError

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
        self.plan = None  # the baked schedule for the current run (set in run())

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

        # bake the schedule once, up front (the planner is the single ordering
        # authority; the same build_plan is what the UI previews)
        self.plan = self._build_plan(task_names, required_lamella)
        self.queue.build_from_plan(self.plan.pairs)
        self._run_queue()

    def _build_plan(self, task_names: List[str], required_lamella: List[str]):
        """Build the load-aware plan for this run (the shared planner entry point)."""
        return plan_for_run(self.experiment, self.microscope,
                            task_names, required_lamella)

    def _run_queue(self) -> None:
        """Walk the baked plan, skipping in place.

        The order is already decided (see :func:`build_plan`). Per item we
        re-validate the shared :func:`should_skip` against live state, then bring
        its grid into the working slot on demand. Grids whose exchange fails are
        isolated — their items are skipped and the run continues. No re-ordering:
        grid-greedy contiguity keeps skip-in-place optimal.
        """
        self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
        failed_grids: set = set()
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            # re-derive the loaded set from reality each item: ensure_loaded is
            # idempotent, so a manual mid-run exchange can never desync us.
            loaded = self._loaded_grid_ids()
            lamella = self.experiment.get_lamella_by_name(item.item_name)
            if lamella is None:
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue

            task_names = self.queue.task_names
            lamella_names = self.queue.item_names

            # re-validate skip against live state before touching hardware
            skip_reason = should_skip(self.experiment, lamella, item.task_name, lamella_names)
            if skip_reason is not None:
                msg = f"Skipping {lamella.name} for {item.task_name}: {skip_reason}."
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                self._emit_status(
                    task_name=item.task_name,
                    task_names=task_names,
                    lamella=lamella,
                    required_lamella=lamella_names,
                    status=AutoLamellaTaskStatus.Skipped,
                    msg=msg,
                    skip_reason=skip_reason,
                )
                continue

            # bring this lamella's grid into the working slot if it isn't already
            grid = self.experiment.get_grid_for_lamella(lamella)
            if grid is not None and grid._id not in loaded:
                if grid._id in failed_grids or not self._ensure_grid_loaded(grid):
                    failed_grids.add(grid._id)
                    self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                    self._emit_status(
                        task_name=item.task_name,
                        task_names=task_names,
                        lamella=lamella,
                        required_lamella=lamella_names,
                        status=AutoLamellaTaskStatus.Skipped,
                        msg=f"Skipping {lamella.name}: grid '{grid.name}' could not be loaded.",
                        skip_reason="grid_exchange_failed",
                    )
                    continue
                self._realign_after_load(grid)

            # Emit InProgress status
            self._emit_status(
                task_name=item.task_name,
                task_names=task_names,
                lamella=lamella,
                required_lamella=lamella_names,
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
                task_name=item.task_name,
                task_names=task_names,
                lamella=lamella,
                required_lamella=lamella_names,
                status=final_status,
                error_message=lamella.task_state.status_message,
                task_duration=lamella.task_state.duration,
                msg=msg,
            )

        # if the objective is inserted, retract for safety
        if self.microscope.fm is not None and self.microscope.fm.objective.state == "Inserted":
            self.microscope.fm.objective.retract()

        self._fire_workflow_hook(HookEvent.WORKFLOW_COMPLETED)
        update_status_ui(self.parent_ui, "", workflow_info="All tasks completed.")
        print(self.experiment.task_history_dataframe())

    def stop(self) -> None:
        """Signal the manager to stop after current task completes."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _fire_workflow_hook(self, event: HookEvent) -> None:
        if self.hook_manager is None:
            return
        self.hook_manager.fire(HookContext(event=event))

    # --- Grid loading (multi-grid execution) ---

    def _loaded_grid_ids(self) -> set:
        """Grid ids currently in the working slot(s) — re-derived from the holder.

        Always read from reality (never a cached set) so a manual exchange by an
        operator mid-run cannot desync the scheduler.
        """
        try:
            return {g._id for g in self.experiment.get_loaded_grids(self.microscope)}
        except Exception:  # noqa: BLE001 - holder may be absent (legacy backends)
            return set()

    def _ensure_grid_loaded(self, grid) -> bool:
        """Bring ``grid`` into the working slot. Returns False on exchange failure.

        Delegates to the single loading authority ``Stage.ensure_loaded`` (no-op
        if already loaded). Legacy backends without a stage abstraction are
        treated as always-reachable.
        """
        stage = getattr(self.microscope, "_stage", None)
        if stage is None:
            return True
        try:
            stage.ensure_loaded(grid.name)
            return True
        except GridExchangeError as e:
            logging.error(f"Grid exchange failed for '{grid.name}': {e}")
            return False

    def _realign_after_load(self, grid) -> None:
        """Coarse realignment after a real exchange (Phase 3 seam; stub for now).

        On a static holder or the simulator no exchange happens, so this never
        fires. The body (overview register -> correction) is intentionally not
        built yet — see docs/design/multi-grid-lamella-execution.md.
        """
        logging.info(f"[realign] grid '{grid.name}' loaded — realignment stubbed (no-op).")

    # --- Internal helpers ---

    def _emit_status(self, task_name: str, task_names: List[str],
                     lamella: 'Lamella', required_lamella: List[str],
                     status: 'AutoLamellaTaskStatus',
                     msg: str = "",
                     error_message: Optional[str] = None,
                     task_duration: Optional[float] = None,
                     skip_reason: Optional[str] = None) -> None:
        """Emit workflow_update_signal with standard status dict."""
        if self.parent_ui is None:
            return

        status_dict = {
            "task_name": task_name,
            "item_name": lamella.name,    # neutral key shared with the timeline
            "lamella_name": lamella.name,  # lamella-specific alias (status bar)
            "status": status,
            "timestamp": time.time(),
            "error_message": error_message,
            "task_duration": task_duration,
            "skip_reason": skip_reason,
        }

        # Include full context when task_names list is available
        status_dict["task_names"] = task_names
        status_dict["total_tasks"] = len(task_names)
        status_dict["current_task_index"] = task_names.index(task_name)
        status_dict["lamella_names"] = required_lamella
        status_dict["current_lamella_index"] = (
            required_lamella.index(lamella.name) if lamella.name in required_lamella else None
        )
        status_dict["total_lamellas"] = len(required_lamella)
        status_dict["queue_items"] = self.queue.items  # thread-safe snapshot

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
