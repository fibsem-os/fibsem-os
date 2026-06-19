from __future__ import annotations

import os
import uuid
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
    get_type_hints,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, GridRecord
from fibsem.imaging.tiled import tiled_image_acquisition_and_stitch
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleHolder
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings, OverviewAcquisitionSettings

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager

import logging


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


def _default_overview_settings() -> OverviewAcquisitionSettings:
    """Default tiled-overview settings for the overview grid task."""
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            resolution=(1024, 1024),
            hfw=500e-6,
            dwell_time=1e-6,
            beam_type=BeamType.ION,
            path=None,
            filename="overview-image",
        ),
        nrows=1,
        ncols=2,
    )


@dataclass
class AcquireOverviewImageGridTaskConfig(GridTaskConfig):
    """Configuration for acquiring overview image grid task."""
    task_type: ClassVar[str] = "ACQUIRE_OVERVIEW_IMAGE_GRID"
    display_name: ClassVar[str] = "Acquire Overview Image"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    settings: OverviewAcquisitionSettings = field(default_factory=_default_overview_settings)


class AcquireOverviewImageGridTask(GridTask):
    """Task to acquire an overview image of the sample grid."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AcquireOverviewImageGridTaskConfig
    config: AcquireOverviewImageGridTaskConfig

    def _run(self):
        """Acquire an overview image of the sample grid."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        test_path = os.path.join(self.experiment.path, self.grid.name, self.task_name)
        os.makedirs(test_path, exist_ok=True)
        self.config.settings.image_settings.path = test_path

        logging.info(f"Path: {test_path}")
        logging.info(f"Moving to grid {self.grid.name} at slot {slot}")

        self._move_to_grid_slot_position(self.config.orientation)

        image = tiled_image_acquisition_and_stitch(
            microscope=self.microscope,
            settings=self.config.settings
        )

        # save the stitched overview (full) + a small thumbnail (for grid cards)
        overview_path = os.path.join(test_path, "overview.tif")
        image.save(overview_path)
        thumb_path = self._save_thumbnail(image)
        self.record_result(overview=overview_path, thumbnail=thumb_path)

        logging.info(f"Acquired overview image for grid {self.grid.name}")

    def _save_thumbnail(self, image, width: int = 400) -> str:
        """Save a small resized PNG thumbnail of the overview for grid cards."""
        import numpy as np
        from PIL import Image as PILImage

        data = image.data
        if data.ndim == 3:
            data = data[..., :3]
        else:
            data = np.stack([data, data, data], axis=2)
        thumb = PILImage.fromarray(data.astype(np.uint8))
        thumb.thumbnail((width, width))  # preserves aspect ratio
        thumb_path = os.path.join(self.experiment.path, self.grid.name, "thumbnail.png")
        thumb.save(thumb_path)
        return thumb_path



def _default_acquire_image_settings() -> ImageSettings:
    """Default image settings for the single-image grid task."""
    return ImageSettings(
        resolution=(4096, 4096),
        hfw=2000e-6,
        dwell_time=1e-6,
        beam_type=BeamType.ELECTRON,
    )


@dataclass
class AcquireImageGridTaskConfig(GridTaskConfig):
    """Configuration for acquiring overview image grid task."""
    task_type: ClassVar[str] = "ACQUIRE_IMAGE_GRID"
    display_name: ClassVar[str] = "Acquire Image"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    voltage: float = field(default=5_000, metadata={"label": "Imaging Voltage" })
    beam_current: float = field(default=1e-9, metadata={"label": "Beam Current"})
    image_settings: ImageSettings = field(default_factory=_default_acquire_image_settings)


class AcquireImageTask(GridTask):
    """Task to acquire an image of the sample grid at a specified voltage."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AcquireImageGridTaskConfig
    config: AcquireImageGridTaskConfig

    def _run(self):
        """Acquire an overview image of the sample grid."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        test_path = os.path.join(self.experiment.path, self.grid.name, self.task_name)
        os.makedirs(test_path, exist_ok=True)
        # self.config.settings.image_settings.path = test_path

        logging.info(f"Path: {test_path}")
        logging.info(f"Moving to grid {self.grid.name} at slot {slot}")

        self._move_to_grid_slot_position(self.config.orientation)

        image_settings = self.config.image_settings  # per-run copy (run_grid_task deepcopies)
        image_settings.save = False

        # apply voltage/current to the beam the image is acquired on (not always ELECTRON)
        beam_type = image_settings.beam_type
        inital_state = self.microscope.get_microscope_state()
        self.microscope.set_beam_voltage(self.config.voltage, beam_type=beam_type)
        self.microscope.set_beam_current(self.config.beam_current, beam_type=beam_type)

        from fibsem import utils
        image = self.microscope.acquire_image(image_settings=image_settings)
        image_path = os.path.join(test_path, f"grid-image-{utils.current_timestamp_v3()}.tif")
        image.save(image_path)
        self.record_result(image=image_path)

        self.microscope.set_microscope_state(inital_state)

        logging.info(f"Acquired image for grid {self.grid.name}")



@dataclass
class CryoDepositionGridTaskConfig(GridTaskConfig):
    """Configuration for cryo deposition task."""
    task_type: ClassVar[str] = "CRYO_DEPOSITION_GRID"
    display_name: ClassVar[str] = "Cryo Deposition"
    deposition_time: float = 30.0  # seconds


@dataclass
class CryoSputterGridTaskConfig(GridTaskConfig):
    """Configuration for cryo sputter task."""
    task_type: ClassVar[str] = "CRYO_SPUTTER_GRID"
    display_name: ClassVar[str] = "Cryo Sputter"
    sputter_time: float = 60.0  # seconds
    sputter_voltage: float = 5.0  # kV
    sputter_current: float = 0.1  # nA


@dataclass
class CryoCleaningGridTaskConfig(GridTaskConfig):
    """Configuration for cryo cleaning milling task."""
    task_type: ClassVar[str] = "CRYO_CLEANING_GRID"
    display_name: ClassVar[str] = "Cryo Cleaning Milling"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    milling_angle: float = 38.0 # degrees
    field_of_view: float = 900e-6  # meters
    duration: float = 10.0  # seconds
    current: float = 15e-9  # A


class CryoCleaningGridTask(GridTask):
    """Task to perform cryo cleaning on the sample grid."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoCleaningGridTaskConfig
    config: CryoCleaningGridTaskConfig

    # ref: https://www.nature.com/articles/s41467-025-57493-3

    def _run(self):
        """Perform cryo cleaning on the sample grid using FIB."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        logging.info(f"Starting cryo cleaning for grid {self.grid.name}")

        # move to grid position
        self._move_to_grid_slot_position(self.config.orientation)

        # set beam parameters
        self.microscope.set_beam_current(self.config.current, beam_type=BeamType.ION)
        self.microscope.set_field_of_view(self.config.field_of_view, beam_type=BeamType.ION)

        # start timer for duration
        start_time = time.time()
        self.microscope.start_acquisition(beam_type=BeamType.ION)
        while (time.time() - start_time) < self.config.duration:
            if self._stop_event and self._stop_event.is_set():
                logging.info("Cryo cleaning task stopped.")
                break
            time.sleep(1)  # wait for 1 second before checking again
            remaining_time = self.config.duration - (time.time() - start_time)
            logging.info(f"Cryo cleaning in progress... {remaining_time:.1f} seconds remaining.")
        self.microscope.stop_acquisition()

        # restore previous settings if needed
        self.microscope.set_beam_current(self.microscope.system.ion.beam.beam_current, beam_type=BeamType.ION)

        # Implement the cryo cleaning logic here
        logging.info(f"Completed cryo cleaning for grid {self.grid.name}")
        # acquire image
        image = self.microscope.acquire_image(beam_type=BeamType.ION)
        path = os.path.join(self.experiment.path, self.grid.name, self.task_name)
        os.makedirs(path, exist_ok=True)
        fib_path = os.path.join(path, "post-grid-cleaining_ib.tif")
        image.save(fib_path)
        self.record_result(fib=fib_path)

@dataclass
class ParallelTrenchMillingGridTaskConfig(GridTaskConfig):
    """Configuration for parallel trench milling task."""
    task_type: ClassVar[str] = "PARALLEL_TRENCH_MILLING_GRID"
    display_name: ClassVar[str] = "Parallel Trench Milling"



GRID_TASK_REGISTRY: Dict[str, Type[GridTask]] = {
    AcquireOverviewImageGridTaskConfig.task_type: AcquireOverviewImageGridTask,
    AcquireImageGridTaskConfig.task_type: AcquireImageTask,
    CryoCleaningGridTaskConfig.task_type: CryoCleaningGridTask,
    # Add other tasks here as needed
}


def get_grid_task_config_cls(task_type: str) -> Type[GridTaskConfig]:
    """Return the GridTaskConfig subclass registered for a task_type."""
    task_cls = GRID_TASK_REGISTRY.get(task_type)
    if task_cls is None:
        raise ValueError(f"Grid task type '{task_type}' is not registered.")
    return task_cls.config_cls


def load_grid_task_config(ddict: Dict[str, Any]) -> Optional[GridTaskConfig]:
    """Reconstruct a typed GridTaskConfig from a serialized dict via its task_type.

    Returns None (with a warning) if the task_type is not registered, so an
    unknown task in a saved protocol is skipped rather than fatal.
    """
    task_type = ddict.get("task_type")
    if task_type not in GRID_TASK_REGISTRY:
        logging.warning(f"Grid task type '{task_type}' is not registered. Skipping.")
        return None
    return get_grid_task_config_cls(task_type).from_dict(ddict)


def run_grid_task(microscope: FibsemMicroscope,
          task_name: str,
          experiment: 'Experiment',
          grid: 'GridRecord',
          parent_ui: Optional['AutoLamellaUI'] = None,
          task_manager: Optional['GridTaskManager'] = None) -> None:
    """Run a single grid task against a GridRecord, persisting state afterwards.

    The config is read from ``experiment.grid_protocol`` (keyed by task_name);
    if no saved config exists, ``task_name`` is treated as a task_type and a
    default config is instantiated. ``task_manager`` (when supplied) provides
    the stop event so long-running tasks can be interrupted.
    """
    config = experiment.grid_protocol.task_config.get(task_name)
    if config is not None:
        task_cls = GRID_TASK_REGISTRY.get(config.task_type)
    else:
        # fallback: task_name is a task_type → run with defaults
        task_cls = GRID_TASK_REGISTRY.get(task_name)
        if task_cls is not None:
            config = task_cls.config_cls(task_name=task_name)

    if task_cls is None or config is None:
        raise ValueError(f"No registered grid task for '{task_name}'.")

    # the protocol config is a shared template — run on a copy so per-run
    # mutation (e.g. tiled acquisition writing total_fov back into hfw, paths,
    # filenames) does not leak across grids or corrupt the saved protocol.
    config = deepcopy(config)

    task = task_cls(microscope=microscope,
                    config=config,
                    experiment=experiment,
                    grid=grid,
                    parent_ui=parent_ui,
                    task_manager=task_manager)
    task.run()
    experiment.save()  # persist the grid's updated task_state / history


def run_grid_tasks(microscope: FibsemMicroscope,
                   experiment: 'Experiment',
                   grid_names: list[str],
                   task_names: list[str]) -> None:
    """Run tasks for the named grids, creating GridRecords on demand."""
    for grid_name in grid_names:
        record = experiment.get_grid_by_name(grid_name)
        if record is None:
            slot = microscope._stage.holder.find_slot_by_grid_name(grid_name)
            if slot is None or slot.loaded_grid is None:
                logging.warning(f"Grid '{grid_name}' not found in any loaded slot.")
                continue
            record = GridRecord(name=grid_name)
            experiment.add_grid(record)

        for task_name in task_names:
            logging.info(f"Running task {task_name} on grid {grid_name}.")
            run_grid_task(microscope, task_name, experiment=experiment, grid=record)
