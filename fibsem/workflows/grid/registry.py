"""Which grid tasks exist, and how a saved configuration becomes one."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Type

from fibsem.plugins.loader import PluginRegistry, subclass_of
from fibsem.workflows.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.workflows.grid.fluorescence import (
    FluorescenceOverviewGridTask,
)
from fibsem.workflows.grid.imaging import (
    BeamOverviewGridTask,
)
from fibsem.workflows.task_config import load_task_config, load_task_configs

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.workflows.grid.protocol import GridExperiment
    from fibsem.workflows.grids import GridRecord

BUILTIN_GRID_TASKS: Dict[str, Type[GridTask]] = {
    cls.config_cls.task_type: cls
    for cls in (BeamOverviewGridTask, FluorescenceOverviewGridTask)
}

GRID_TASK_ENTRY_POINT_GROUP = "fibsem.grid.tasks"

# Built-ins, runtime registrations and the fibsem.grid.tasks entry points. To
# add a plugin grid task, add to your package's pyproject.toml:
#
#     [project.entry-points.'fibsem.grid.tasks']
#     my_task = "my_package.tasks:MyGridTask"
GRID_TASK_PLUGINS: PluginRegistry[Type[GridTask]] = PluginRegistry(
    group=GRID_TASK_ENTRY_POINT_GROUP,
    kind="grid task",
    resolve=subclass_of(GridTask, lambda cls: cls.config_cls.task_type),
    builtins=BUILTIN_GRID_TASKS,
)


def get_grid_tasks() -> Dict[str, Type[GridTask]]:
    """Every grid task, by task_type: built-ins, then runtime registrations,
    then plugins."""
    return GRID_TASK_PLUGINS.all()


def load_grid_task_config(data: Dict[str, Any]) -> GridTaskConfig:
    """A typed config from its saved form, via the registry."""
    return load_task_config(GRID_TASK_PLUGINS, data, kind="grid task")


def load_grid_task_configs(
    ddict: Dict[str, Dict[str, Any]],
) -> Dict[str, GridTaskConfig]:
    """name -> config, skipping (with a warning) any type this build does not know."""
    return load_task_configs(GRID_TASK_PLUGINS, ddict, kind="grid task")


def run_grid_task(
    microscope: "FibsemMicroscope",
    task_name: str,
    experiment: "GridExperiment",
    grid: "GridRecord",
    task_manager=None,
) -> GridTask:
    """Run the named task from the experiment's grid protocol on one grid.

    The saved config is deep-copied per run, so a task that mutates its config
    (a per-run path, say) does not write back into the protocol.
    """
    saved = experiment.grid_protocol.task_config.get(task_name)
    if saved is None:
        raise KeyError(f"Task '{task_name}' is not in the grid protocol.")
    task_cls = GRID_TASK_PLUGINS.get(saved.task_type)
    if task_cls is None:
        raise KeyError(f"Grid task type '{saved.task_type}' is not registered.")
    task = task_cls(
        microscope=microscope,
        config=deepcopy(saved),
        grid=grid,
        experiment=experiment,
        task_manager=task_manager,
    )
    task.run()
    return task
