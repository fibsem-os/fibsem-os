"""Deprecated shim — grid tasks moved to the ``tasks.grid`` package.

Import from ``fibsem.applications.autolamella.workflows.tasks.grid`` instead.
This module re-exports the public API for backward compatibility.
"""

from fibsem.applications.autolamella.workflows.tasks.grid import (  # noqa: F401
    GRID_TASK_REGISTRY,
    AcquireImageGridTaskConfig,
    AcquireImageTask,
    AcquireOverviewImageGridTask,
    AcquireOverviewImageGridTaskConfig,
    CryoCleaningGridTask,
    CryoCleaningGridTaskConfig,
    CryoDepositionGridTaskConfig,
    CryoSputterGridTaskConfig,
    GridTask,
    GridTaskConfig,
    ParallelTrenchMillingGridTaskConfig,
    get_grid_task_config_cls,
    load_grid_task_config,
    run_grid_task,
    run_grid_tasks,
)

__all__ = [
    "GridTask",
    "GridTaskConfig",
    "AcquireOverviewImageGridTaskConfig",
    "AcquireOverviewImageGridTask",
    "AcquireImageGridTaskConfig",
    "AcquireImageTask",
    "CryoDepositionGridTaskConfig",
    "CryoSputterGridTaskConfig",
    "CryoCleaningGridTaskConfig",
    "CryoCleaningGridTask",
    "ParallelTrenchMillingGridTaskConfig",
    "GRID_TASK_REGISTRY",
    "get_grid_task_config_cls",
    "load_grid_task_config",
    "run_grid_task",
    "run_grid_tasks",
]
