"""Grid tasks: operations on a whole grid, run as workflow tasks.

A grid task is a thin lifecycle wrapper around a plain operation. The operation --
"acquire a tiled overview here with these settings" -- is a standalone function the
Overview tab can call too; the task only adds what makes it a workflow step:
state and history on the `GridRecord`, outputs recorded by role, hooks, and the
stop token. That split is what lets a lamella workflow call the same operation
later without going through the grid manager.
"""

from fibsem.workflows.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.workflows.grid.fluorescence import (  # noqa: E402
    FluorescenceOverviewGridTask,
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.workflows.grid.imaging import (  # noqa: E402
    BeamOverviewGridTask,
    BeamOverviewGridTaskConfig,
)
from fibsem.workflows.grid.registry import (
    BUILTIN_GRID_TASKS,
    GRID_TASK_ENTRY_POINT_GROUP,
    GRID_TASK_PLUGINS,
    get_grid_tasks,
    load_grid_task_config,
    load_grid_task_configs,
    register_grid_task,
    run_grid_task,
)

__all__ = [
    "BUILTIN_GRID_TASKS",
    "GRID_TASK_ENTRY_POINT_GROUP",
    "GRID_TASK_PLUGINS",
    "BeamOverviewGridTask",
    "BeamOverviewGridTaskConfig",
    "FluorescenceOverviewGridTask",
    "FluorescenceOverviewGridTaskConfig",
    "GridTask",
    "GridTaskConfig",
    "get_grid_tasks",
    "load_grid_task_config",
    "load_grid_task_configs",
    "register_grid_task",
    "run_grid_task",
]
