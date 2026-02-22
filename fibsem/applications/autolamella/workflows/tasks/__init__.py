"""
AutoLamellaTask plugin system for fibsem autolamella tasks.

This module provides a plugin architecture for AutoLamellaTask implementations,
similar to the BasePattern plugin system in fibsem.milling.patterning.
"""

import logging
import typing

try:
    from functools import cache
except ImportError:  # Python < 3.9 fallback
    from functools import lru_cache as cache
from typing import Dict, List, Type

from fibsem.applications.autolamella.workflows.tasks.tasks import AutoLamellaTask

# Built-in task classes
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    MillTrenchTask,
    MillUndercutTask,
    MillRoughTask,
    MillPolishingTask,
    SpotBurnFiducialTask,
    MillFiducialTask,
    AcquireReferenceImageTask,
    BasicMillingTask,
    SelectMillingPositionTask,
)

# Built-in task config classes
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    MillTrenchTaskConfig,
    MillUndercutTaskConfig,
    MillRoughTaskConfig,
    MillPolishingTaskConfig,
    SpotBurnFiducialTaskConfig,
    MillFiducialTaskConfig,
    AcquireReferenceImageConfig,
    BasicMillingTaskConfig,
    SelectMillingPositionTaskConfig,
)

# Helper functions and exceptions
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    TaskNotRegisteredError,
    get_task_supervision,
    load_task_config,
    load_config,
    get_task_config,
    run_task,
    run_tasks,
)

# Built-in tasks registry
BUILTIN_TASKS: Dict[str, Type[AutoLamellaTask]] = {
    MillTrenchTaskConfig.task_type: MillTrenchTask,
    MillUndercutTaskConfig.task_type: MillUndercutTask,
    MillRoughTaskConfig.task_type: MillRoughTask,
    MillPolishingTaskConfig.task_type: MillPolishingTask,
    SpotBurnFiducialTaskConfig.task_type: SpotBurnFiducialTask,
    MillFiducialTaskConfig.task_type: MillFiducialTask,
    AcquireReferenceImageConfig.task_type: AcquireReferenceImageTask,
    BasicMillingTaskConfig.task_type: BasicMillingTask,
    SelectMillingPositionTaskConfig.task_type: SelectMillingPositionTask,
    "SETUP_LAMELLA": MillFiducialTask,  # BACKWARDS_COMPATIBILITY
}

# Runtime registered tasks
REGISTERED_TASKS: Dict[str, Type[AutoLamellaTask]] = {}


def register_task(task_cls: Type[AutoLamellaTask]) -> None:
    """Register a task class at runtime.

    Args:
        task_cls: The task class to register. Must be a subclass of AutoLamellaTask
                  with a config_cls ClassVar that has a task_type ClassVar.

    Example:
        >>> from fibsem.applications.autolamella.workflows.tasks import register_task
        >>> register_task(CustomTask)
    """
    global REGISTERED_TASKS
    task_type = task_cls.config_cls.task_type
    REGISTERED_TASKS[task_type] = task_cls
    logging.info("Registered task '%s'", task_type)


@cache
def _get_plugin_tasks() -> Dict[str, Type[AutoLamellaTask]]:
    """
    Discover and import task plugins via entry points.

    The plugin logic is based on:
    https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata

    To add a plugin task, add to your package's pyproject.toml:

    [project.entry-points.'fibsem.tasks']
    my_task = "my_package.tasks:MyCustomTask"
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    tasks: Dict[str, Type[AutoLamellaTask]] = {}

    for task_entry_point in entry_points(group="fibsem.tasks"):
        try:
            task = task_entry_point.load()
            if not issubclass(task, AutoLamellaTask):
                raise TypeError(
                    f"'{task_entry_point.value}' is not a subclass of AutoLamellaTask"
                )
            task_type = task.config_cls.task_type
            logging.info("Loaded task plugin '%s'", task_type)
            tasks[task_type] = task
        except TypeError as e:
            logging.warning("Invalid task plugin found: %s", str(e))
        except Exception:
            logging.error(
                "Unexpected error raised while attempting to import task from '%s'",
                task_entry_point.value,
                exc_info=True,
            )

    return tasks


def get_tasks() -> Dict[str, Type[AutoLamellaTask]]:
    """Get all available tasks.

    Returns tasks in priority order (highest to lowest):
    1. Built-in tasks
    2. Runtime registered tasks
    3. Plugin tasks

    Returns:
        Dictionary mapping task type strings to task classes
    """
    # This order means that builtins > registered > plugins if there are any name clashes
    return {**_get_plugin_tasks(), **REGISTERED_TASKS, **BUILTIN_TASKS}


def get_task_names() -> typing.List[str]:
    """Get list of all available task type names."""
    return list(get_tasks().keys())


# Legacy support - maintain backward compatibility
TASK_REGISTRY: Dict[str, Type[AutoLamellaTask]] = get_tasks()
