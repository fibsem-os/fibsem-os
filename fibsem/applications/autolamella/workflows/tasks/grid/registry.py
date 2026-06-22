"""Grid task registry + runners.

Maps task_type -> GridTask class, reconstructs configs from serialized dicts,
and runs tasks against GridRecords (reading the saved protocol config).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from fibsem.applications.autolamella.structures import GridRecord
from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.cryo import (
    CryoCleaningGridTask,
    CryoCleaningGridTaskConfig,
    CryoDepositionGridTask,
    CryoDepositionGridTaskConfig,
    CryoSputterGridTask,
    CryoSputterGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    AcquireFluorescenceOverviewImageTask,
    AcquireFluorescenceOverviewImageTaskConfig,
    AcquireImageGridTaskConfig,
    AcquireImageTask,
    AcquireOverviewImageGridTask,
    AcquireOverviewImageGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.milling import (
    ParallelTrenchMillingGridTask,
    ParallelTrenchMillingGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.targeting import (
    AutoLamellaTargetingGridTask,
    AutoLamellaTargetingGridTaskConfig,
)
from fibsem.microscope import FibsemMicroscope

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager


GRID_TASK_REGISTRY: Dict[str, Type[GridTask]] = {
    AcquireOverviewImageGridTaskConfig.task_type: AcquireOverviewImageGridTask,
    AcquireFluorescenceOverviewImageTaskConfig.task_type: AcquireFluorescenceOverviewImageTask,
    AcquireImageGridTaskConfig.task_type: AcquireImageTask,
    CryoCleaningGridTaskConfig.task_type: CryoCleaningGridTask,
    CryoDepositionGridTaskConfig.task_type: CryoDepositionGridTask,
    CryoSputterGridTaskConfig.task_type: CryoSputterGridTask,
    ParallelTrenchMillingGridTaskConfig.task_type: ParallelTrenchMillingGridTask,
    AutoLamellaTargetingGridTaskConfig.task_type: AutoLamellaTargetingGridTask,
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
