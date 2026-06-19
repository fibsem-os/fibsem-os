"""Base classes for grid-level workflow tasks: GridTaskConfig + GridTask."""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Optional,
    Tuple,
    get_type_hints,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, GridRecord
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import GridSlot
from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager


@dataclass
class GridTaskConfig(ABC):
    """Base configuration for grid tasks.

    Grid configs are intentionally lean: they carry only ``task_name`` plus
    each task's own fields (no shared milling / reference-imaging blocks).
    Serialization is **flat** — each task-specific field is written at the top
    level, and nested dataclasses serialize via their own ``to_dict`` /
    ``from_dict`` (rather than a generic ``parameters`` subdict). The
    ``parameters`` property is retained for generic UI form generation.
    """
    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = ""

    @property
    def parameters(self) -> Tuple[str, ...]:
        """Names of this task's own fields (everything except the core fields)."""
        core = {f.name for f in fields(GridTaskConfig)}
        return tuple(f.name for f in fields(self) if f.name not in core)

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, Enum):
            return value.value
        return value

    def to_dict(self) -> dict:
        ddict: Dict[str, Any] = {"task_type": self.task_type, "task_name": self.task_name}
        for name in self.parameters:
            ddict[name] = self._serialize(getattr(self, name))
        return ddict

    @classmethod
    def from_dict(cls, ddict: Dict[str, Any]) -> "GridTaskConfig":
        hints = get_type_hints(cls)
        core = {f.name for f in fields(GridTaskConfig)}
        kwargs: Dict[str, Any] = {}
        if "task_name" in ddict:
            kwargs["task_name"] = ddict["task_name"]
        for f in fields(cls):
            if f.name in core or f.name not in ddict:
                continue
            typ = hints.get(f.name)
            value = ddict[f.name]
            if value is not None and isinstance(typ, type) and hasattr(typ, "from_dict"):
                kwargs[f.name] = typ.from_dict(value)
            elif value is not None and isinstance(typ, type) and issubclass(typ, Enum):
                kwargs[f.name] = typ(value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)


class GridTask(ABC):
    """Base class for grid-level workflow tasks.

    A GridTask operates on a :class:`GridRecord` (the workflow unit), mirroring
    the ``AutoLamellaTask`` lifecycle: ``pre_task() -> _run() -> post_task()``,
    writing progress into the record's ``task_state`` / ``task_history``. The
    hardware slot is resolved live from the holder by the grid's name.
    """
    config_cls: ClassVar[GridTaskConfig]
    config: GridTaskConfig

    def __init__(self,
                 microscope: FibsemMicroscope,
                 config: GridTaskConfig,
                 grid: 'GridRecord',
                 experiment: 'Experiment',
                 parent_ui: Optional['AutoLamellaUI'] = None,
                 task_manager: Optional['TaskManager'] = None):
        self.microscope = microscope
        self.config = config
        self.experiment = experiment
        self.grid = grid
        self.parent_ui = parent_ui
        self.task_manager = task_manager
        self.task_id = str(uuid.uuid4())
        self._stop_event = task_manager._stop_event if task_manager else None

    @property
    def slot(self) -> Optional[GridSlot]:
        """Resolve the hardware slot this grid is loaded in, live, by name."""
        return self.microscope._stage.holder.find_slot_by_grid_name(self.grid.name)

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @abstractmethod
    def _run(self):
        """Run the task. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def run(self):
        """Execute the task lifecycle: pre_task -> _run -> post_task, firing
        lifecycle hooks (logging / notifications / webhooks)."""
        self.pre_task()
        self._fire_hook("task_started")
        try:
            self._run()
        except Exception as error:
            self.on_failure(error)
            self._fire_hook("task_failed", error=str(error))
            raise
        self.post_task()
        self._fire_hook("task_completed")

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        hook_manager = getattr(self.task_manager, "hook_manager", None)
        if hook_manager is None:
            return
        from fibsem.applications.autolamella.workflows.tasks.hooks import HookContext
        hook_manager.fire(HookContext(
            event=event,
            task_name=self.task_name,
            task_type=self.task_type,
            item_name=self.grid.name,
            task_state=self.grid.task_state,
            error=error,
        ))

    def pre_task(self) -> None:
        """Mark the grid's task_state as InProgress and stamp the start time."""
        logging.info(
            f"Running grid task {self.task_name} ({self.task_type}, {self.task_id}) "
            f"for grid {self.grid.name} ({self.grid._id})"
        )
        ts = self.grid.task_state
        ts.name = self.task_name
        ts.task_id = self.task_id
        ts.task_type = self.task_type
        ts.start_timestamp = datetime.timestamp(datetime.now())
        ts.end_timestamp = None
        ts.status = AutoLamellaTaskStatus.InProgress
        ts.status_message = ""

    def post_task(self) -> None:
        """Mark the grid's task_state Completed and append it to history."""
        ts = self.grid.task_state
        ts.end_timestamp = datetime.timestamp(datetime.now())
        ts.status = AutoLamellaTaskStatus.Completed
        ts.status_message = ""
        self.grid.task_history.append(deepcopy(ts))
        logging.info(f"Completed grid task {self.task_name} for grid {self.grid.name}")

    def on_failure(self, error: Exception) -> None:
        """Record a task failure on the grid's task_state and history."""
        ts = self.grid.task_state
        ts.end_timestamp = datetime.timestamp(datetime.now())
        ts.status = AutoLamellaTaskStatus.Failed
        ts.status_message = str(error)
        self.grid.task_history.append(deepcopy(ts))
        logging.error(
            f"Grid task {self.task_name} failed for grid {self.grid.name}: {error}",
            exc_info=True,
        )

    def record_result(self, **artifacts: Any) -> None:
        """Record this task's output artifacts/metadata on the grid record, keyed
        by task_name, for the Results UI (e.g. record_result(overview=path))."""
        self.grid.results[self.task_name] = dict(artifacts)

    def grid_dir(self) -> str:
        """Per-grid output directory: ``<experiment>/grids/<grid>``.

        Grids live under a ``grids/`` subdir to keep them separate from the
        top-level (per-lamella) experiment layout.
        """
        return os.path.join(self.experiment.path, "grids", self.grid.name)

    def task_dir(self) -> str:
        """Per-task output directory (``<grid_dir>/<task_name>``), created if needed."""
        path = os.path.join(self.grid_dir(), self.task_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_stage_position_for_orientation(
        self,
        stage_position: FibsemStagePosition,
        orientation: Optional[str],
    ) -> FibsemStagePosition:
        """Return target position for orientation, or stage_position unchanged if orientation is None."""
        if orientation is None:
            return stage_position
        return self.microscope.get_target_position(stage_position, orientation)

    def _move_to_grid_slot_position(self, orientation: str):
        """Move to the grid slot position in the target orientation"""
        target_position = self._get_stage_position_for_orientation(self.slot.position, orientation=orientation)
        self.microscope._stage.move_absolute(target_position)
