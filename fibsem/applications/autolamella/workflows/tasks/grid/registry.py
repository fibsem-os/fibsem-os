"""Which grid tasks exist, and how a saved configuration becomes one."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, GridRecord
    from fibsem.microscope import FibsemMicroscope

# task_type -> task class. Filled by `register_grid_task`; the built-in tasks
# register themselves on import.
GRID_TASK_REGISTRY: Dict[str, Type[GridTask]] = {}


def register_grid_task(task_cls: Type[GridTask]) -> Type[GridTask]:
    """Class decorator: make a task available by its config's `task_type`."""
    task_type = task_cls.config_cls.task_type
    if (
        task_type in GRID_TASK_REGISTRY
        and GRID_TASK_REGISTRY[task_type] is not task_cls
    ):
        logging.warning(f"Grid task type '{task_type}' registered twice; replacing.")
    GRID_TASK_REGISTRY[task_type] = task_cls
    return task_cls


def get_grid_tasks() -> Dict[str, Type[GridTask]]:
    return dict(GRID_TASK_REGISTRY)


def load_grid_task_config(data: Dict[str, Any]) -> GridTaskConfig:
    """A typed config from its saved form, via the registry."""
    task_type = data.get("task_type")
    if task_type not in GRID_TASK_REGISTRY:
        raise KeyError(f"Grid task type '{task_type}' is not registered.")
    return GRID_TASK_REGISTRY[task_type].config_cls.from_dict(data)


def load_grid_task_configs(
    ddict: Dict[str, Dict[str, Any]],
) -> Dict[str, GridTaskConfig]:
    """name -> config, skipping (with a warning) any type this build does not know."""
    configs: Dict[str, GridTaskConfig] = {}
    for name, data in (ddict or {}).items():
        try:
            config = load_grid_task_config(data)
        except KeyError as e:
            logging.warning(f"Grid task '{name}' skipped: {e}")
            continue
        config.task_name = name
        configs[name] = config
    return configs


def run_grid_task(
    microscope: "FibsemMicroscope",
    task_name: str,
    experiment: "Experiment",
    grid: "GridRecord",
    parent_ui=None,
    task_manager=None,
) -> GridTask:
    """Run the named task from the experiment's grid protocol on one grid.

    The saved config is deep-copied per run, so a task that mutates its config
    (a per-run path, say) does not write back into the protocol.
    """
    saved = experiment.grid_protocol.task_config.get(task_name)
    if saved is None:
        raise KeyError(f"Task '{task_name}' is not in the grid protocol.")
    task_cls = GRID_TASK_REGISTRY.get(saved.task_type)
    if task_cls is None:
        raise KeyError(f"Grid task type '{saved.task_type}' is not registered.")
    task = task_cls(
        microscope=microscope,
        config=deepcopy(saved),
        grid=grid,
        experiment=experiment,
        parent_ui=parent_ui,
        task_manager=task_manager,
    )
    task.run()
    return task
