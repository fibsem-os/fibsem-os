"""Task execution manager for autolamella workflows."""

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.workflows.tasks.hooks import HookContext, HookEvent, HookManager
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.applications.autolamella.workflows.ui import update_status_ui
from fibsem.microscope import FibsemMicroscope

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
        self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            lamella = self.experiment.get_lamella_by_name(item.lamella_name)
            if lamella is None:
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue

            task_names = self.queue.task_names
            lamella_names = self.queue.lamella_names

            skip_reason = self._should_skip(lamella, item.task_name, lamella_names)
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

    # --- Internal helpers ---

    def _should_skip(self, lamella: 'Lamella', task_name: str,
                     required_lamella: List[str]) -> Optional[str]:
        """Return skip reason string, or None if task should run."""
        if required_lamella and lamella.name not in required_lamella:
            logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Not in required lamella list.")
            return "not_required"

        if lamella.is_failure:
            logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Marked as failure or has defect.")
            return "failure"

        task_requirements = self.experiment.task_protocol.workflow_config.requirements(task_name)
        if task_requirements and not all(lamella.has_completed_task(req) for req in task_requirements):
            logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Required tasks {task_requirements} not completed.")
            return "missing_prereqs"

        return None

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
            "lamella_name": lamella.name,
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
