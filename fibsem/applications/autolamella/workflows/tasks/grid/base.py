"""Base classes for grid-level workflow tasks: GridTaskConfig + GridTask."""

from __future__ import annotations

import logging
import os
import time
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
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

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

    # --- user interaction (mirrors the lamella task primitives) ---

    @property
    def validate(self) -> bool:
        """Whether this task is supervised (pauses for user confirmation).

        Read live from the grid workflow config's per-task supervise flag (the
        grid analogue of the lamella ``get_task_supervision``).
        """
        protocol = getattr(self.experiment, "grid_protocol", None)
        wf = getattr(protocol, "workflow_config", None)
        if wf is None:
            return False
        desc = wf.get(self.task_name)
        return bool(desc.supervise) if desc is not None else False

    def update_status_ui(self, message: str, workflow_info: Optional[str] = None) -> None:
        """Push a status message to the shared workflow status bar (prefixed with
        the grid + task). No-op (logs) in headless mode."""
        # lazy import to avoid pulling the heavy workflows.ui module at import time
        from fibsem.applications.autolamella.workflows.ui import update_status_ui
        update_status_ui(
            parent_ui=self.parent_ui,
            msg=f"{self.grid.name} [{self.task_name}] {message}",
            workflow_info=workflow_info,
        )

    def ask_user(self, msg: str, pos: str = "Continue", neg: Optional[str] = "Cancel") -> bool:
        """Pause for user confirmation when supervised; auto-continue otherwise.

        Returns True to proceed. When the task is not supervised (or headless),
        returns True immediately without prompting.
        """
        if not self.validate:
            return True
        from fibsem.applications.autolamella.workflows.ui import ask_user
        return ask_user(parent_ui=self.parent_ui, msg=msg, pos=pos, neg=neg)

    # --- progress (drives the shared status-bar progress widget) ---

    def progress_countdown(self, remaining: float, total: float, message: str = "") -> None:
        """Show a countdown progress bar (remaining/total seconds)."""
        from fibsem.applications.autolamella.workflows.ui import update_progress_ui
        update_progress_ui(self.parent_ui, remaining=remaining, total=total,
                           message=f"{self.grid.name} [{self.task_name}] {message}")

    def progress_indeterminate(self, message: str = "") -> None:
        """Show an indeterminate progress bar (unknown duration / blocking op)."""
        from fibsem.applications.autolamella.workflows.ui import update_progress_ui
        update_progress_ui(self.parent_ui, indeterminate=True,
                           message=f"{self.grid.name} [{self.task_name}] {message}")

    def progress_done(self) -> None:
        """Clear / complete the progress bar."""
        from fibsem.applications.autolamella.workflows.ui import update_progress_ui
        update_progress_ui(self.parent_ui, done=True)

    def wait_with_progress(self, duration: float, message: str = "Working") -> bool:
        """Wait ``duration`` seconds, emitting a per-second countdown to the
        status bar and honouring the stop event.

        The generic "wait while hardware does its thing" loop shared by timed
        grid tasks (FIB cleaning, GIS deposition). Returns True if it ran to
        completion, or False if interrupted by the stop event.
        """
        start = time.time()
        while (time.time() - start) < duration:
            if self._stop_event is not None and self._stop_event.is_set():
                logging.info(f"{self.task_name}: wait interrupted by stop event.")
                return False
            time.sleep(1)
            remaining = duration - (time.time() - start)
            self.update_status_ui(f"{message}... {remaining:.0f}s remaining")
        return True

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

    # --- reference imaging ---

    def _save_grid_thumbnail(self, image, width: int = 400) -> str:
        """Save a small resized PNG thumbnail of an image for the grid cards."""
        import numpy as np
        from PIL import Image as PILImage

        data = image.data
        if data.ndim == 3:
            data = data[..., :3]
        else:
            data = np.stack([data, data, data], axis=2)
        thumb = PILImage.fromarray(data.astype(np.uint8))
        thumb.thumbnail((width, width))  # preserves aspect ratio
        thumb_path = os.path.join(self.grid_dir(), "thumbnail.png")
        thumb.save(thumb_path)
        return thumb_path

    def acquire_grid_reference_image(
        self,
        orientation: str = "SEM",
        hfw: float = 2000e-6,
        beam_type: BeamType = BeamType.ELECTRON,
        filename: str = "reference",
    ) -> Tuple[str, str]:
        """Acquire + save a low-mag reference image of the grid (e.g. after a
        treatment step), plus a thumbnail. Re-asserts the grid orientation first,
        since the preceding step may have moved the stage. Returns
        ``(image_path, thumbnail_path)``; the caller records the result.

        The file is suffixed with the beam abbreviation (``_eb``/``_ib``) to
        match the repo convention and stay correct across beam types.
        """
        self.update_status_ui("Acquiring reference image...")
        self._move_to_grid_slot_position(orientation)

        image_settings = ImageSettings(
            resolution=(1536, 1024), hfw=hfw, dwell_time=1e-6,
            beam_type=beam_type, save=False,
        )
        image = self.microscope.acquire_image(image_settings=image_settings)
        suffix = "ib" if beam_type is BeamType.ION else "eb"
        image_path = os.path.join(self.task_dir(), f"{filename}_{suffix}.tif")
        image.save(image_path)
        thumb_path = self._save_grid_thumbnail(image)
        return image_path, thumb_path
