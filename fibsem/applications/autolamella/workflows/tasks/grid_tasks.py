import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, Literal, Optional, Type

from fibsem.imaging.tiled import ImageSettings, tiled_image_acquisition_and_stitch
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import SampleGrid, SampleHolder
from fibsem.structures import BeamType

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

import logging


@dataclass
class GridTaskConfig(ABC):
    """Configuration for AutoLamella tasks."""
    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = ""


class GridTask(ABC):
    """Base class for AutoLamella tasks."""
    config_cls: ClassVar[GridTaskConfig]
    config: GridTaskConfig

    def __init__(self,
                 microscope: FibsemMicroscope,
                 config: GridTaskConfig,
                 grid: SampleGrid,
                 experiment: 'Experiment',
                 parent_ui: Optional['AutoLamellaUI'] = None):
        self.microscope = microscope
        self.config = config
        self.experiment = experiment
        self.grid = grid
        self.parent_ui = parent_ui
        self.task_id = str(uuid.uuid4())
        self._stop_event = self.parent_ui._workflow_stop_event if self.parent_ui else None

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @abstractmethod
    def _run(self):
        """Run the task. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def run(self):
        """Public method to run the task."""
        self._run()


@dataclass
class AcquireOverviewImageGridTaskConfig(GridTaskConfig):
    """Configuration for acquiring overview image grid task."""
    task_type: ClassVar[str] = "ACQUIRE_OVERVIEW_IMAGE_GRID"
    display_name: ClassVar[str] = "Acquire Overview Image"
    orientation: Literal["SEM", "FIB", "MILLING"] = "SEM"


class AcquireOverviewImageGridTask(GridTask):
    """Task to acquire an overview image of the sample grid."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AcquireOverviewImageGridTaskConfig
    config: AcquireOverviewImageGridTaskConfig

    def _run(self):
        """Acquire an overview image of the sample grid."""

        microscope = self.microscope
        test_path = os.path.join(self.experiment.path, self.grid.name, self.task_name)
        os.makedirs(test_path, exist_ok=True)

        logging.info(f"Path: {test_path}")
        logging.info(f"Moving to grid {self.grid.name} at position {self.grid.position}")

        # self.microscope._stage.move_to_grid(self.grid.name)

        target_position = self.microscope.get_target_position(self.grid.position, self.config.orientation)
        self.microscope._stage.move_absolute(target_position)

        image_settings = ImageSettings(
                resolution=(1024, 1024),
                hfw=500e-6,
                dwell_time=1e-6,
                beam_type=BeamType.ELECTRON,
                path=test_path,
                filename="overview-image"
            )
        tiled_image_acquisition_and_stitch(
            microscope=microscope,
            image_settings=image_settings,
            nrows=3, ncols=3, tile_size=image_settings.hfw,
            cryo=False,
        )

        logging.info(f"Acquired overview image for grid {self.grid.name}")



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
class CryoCleaningMillingGridTaskConfig(GridTaskConfig):
    """Configuration for cryo cleaning milling task."""
    task_type: ClassVar[str] = "CRYO_CLEANING_MILLING_GRID"
    display_name: ClassVar[str] = "Cryo Cleaning Milling"
    milling_angle: float = 10.0  # degrees
    milling_time: float = 120.0  # seconds
    milling_current: float = 0.5  # nA


@dataclass
class ParallelTrenchMIllingGridTaskConfig(GridTaskConfig):
    """Configuration for parallel trench milling task."""
    task_type: ClassVar[str] = "PARALLEL_TRENCH_MILLING_GRID"
    display_name: ClassVar[str] = "Parallel Trench Milling"





GRID_TASK_REGISTRY: Dict[str, Type[GridTask]] = {
    AcquireOverviewImageGridTaskConfig.task_type: AcquireOverviewImageGridTask,
    # Add other tasks here as needed
}   

def run_grid_task(microscope: FibsemMicroscope, 
          task_name: str, 
          experiment: 'Experiment',
          grid: SampleGrid, 
          parent_ui: Optional['AutoLamellaUI'] = None) -> None:
    """Run a specific AutoLamella task."""

    # task_config = experiment.task_protocol.task_config.get(task_name)
    # if task_config is None:
        # raise ValueError(f"Task configuration for {task_name} not found in lamella tasks.")

    task_cls = GRID_TASK_REGISTRY.get(task_name)
    if task_cls is None:
        raise ValueError(f"Task {task_name} is not registered.")

    config = task_cls.config_cls()

    task = task_cls(microscope=microscope,
                    config=config,
                    experiment=experiment,
                    grid=grid,
                    parent_ui=parent_ui)
    task.run()
    # TODO: add task config to experiment, integrate into runner


def run_grid_tasks(microscope: FibsemMicroscope, 
                   experiment: 'Experiment', 
                   grid_names: list[str],
                   task_names: list[str]) -> None:
    """Run tasks for specified grids."""
    for task_name in task_names:
        for grid_name in grid_names:
            grid = microscope._stage.holder.grids.get(grid_name)
            if grid is None:
                logging.warning(f"Grid {grid_name} not found in holder.")
                continue

            logging.info(f"Running task {task_name} on grid {grid_name}.")
            run_grid_task(microscope, task_name, experiment=experiment, grid=grid)