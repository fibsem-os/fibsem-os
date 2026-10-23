"""
AutoLamellaTask plugin system for fibsem autolamella tasks.

This module provides a plugin architecture for AutoLamellaTask implementations,
similar to the BasePattern plugin system in fibsem.milling.patterning.
"""

from typing import TYPE_CHECKING, Any, Dict, Type

from fibsem.plugins.loader import PluginRegistry, subclass_of
from fibsem.workflows.site_task import SiteTask
from fibsem.workflows.site_tasks import BUILTIN_SITE_TASKS

if TYPE_CHECKING:
    from psygnal.containers import EventedDict

    from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig

from fibsem.applications.autolamella.workflows.tasks.manager import (
    TaskManager,
    run_task,
)

# Built-in task classes
# Built-in task config classes
# Helper functions and exceptions
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
    AcquireFluorescenceImageTask,
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
    AutoLamellaTask,
    MillCoincidentTask,
    MillCoincidentTaskConfig,
    MillFiducialTask,
    MillFiducialTaskConfig,
    MillPerforationTask,
    MillPerforationTaskConfig,
    MillPolishingTask,
    MillPolishingTaskConfig,
    MillRoughTask,
    MillRoughTaskConfig,
    MillTrenchTask,
    MillTrenchTaskConfig,
    MillUndercutTask,
    MillUndercutTaskConfig,
    SelectFluorescencePositionConfig,
    SelectFluorescencePositionTask,
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
    SetupCoincidenceMillingTask,
    SetupCoincidenceMillingTaskConfig,
    SpotBurnFiducialTask,
    SpotBurnFiducialTaskConfig,
    get_task_supervision,
)
from fibsem.workflows.queue import (
    QueueOp,
    QueueResult,
    TaskQueue,
    WorkItem,
)

# Built-in tasks registry
BUILTIN_TASKS: Dict[str, Type[SiteTask]] = {
    **BUILTIN_SITE_TASKS,
    MillTrenchTaskConfig.task_type: MillTrenchTask,
    MillUndercutTaskConfig.task_type: MillUndercutTask,
    MillRoughTaskConfig.task_type: MillRoughTask,
    MillPolishingTaskConfig.task_type: MillPolishingTask,
    MillPerforationTaskConfig.task_type: MillPerforationTask,
    MillFiducialTaskConfig.task_type: MillFiducialTask,
    SelectMillingPositionTaskConfig.task_type: SelectMillingPositionTask,
    "SETUP_LAMELLA": MillFiducialTask,  # BACKWARDS_COMPATIBILITY,
    SelectFluorescencePositionConfig.task_type: SelectFluorescencePositionTask,
    AcquireFluorescenceImageConfig.task_type: AcquireFluorescenceImageTask,
    SetupCoincidenceMillingTaskConfig.task_type: SetupCoincidenceMillingTask,
    MillCoincidentTaskConfig.task_type: MillCoincidentTask,
}

TASK_ENTRY_POINT_GROUP = "fibsem.autolamella.tasks"

# Built-ins, runtime registrations and the fibsem.autolamella.tasks entry points. To add a
# plugin task, add to your package's pyproject.toml:
#
#     [project.entry-points.'fibsem.autolamella.tasks']
#     my_task = "my_package.tasks:MyCustomTask"
TASK_PLUGINS: PluginRegistry[Type[AutoLamellaTask]] = PluginRegistry(
    group=TASK_ENTRY_POINT_GROUP,
    kind="task",
    # A task registers under its config's task_type, not its own name.
    resolve=subclass_of(AutoLamellaTask, lambda cls: cls.config_cls.task_type),
    builtins=BUILTIN_TASKS,
)


def get_tasks() -> Dict[str, Type[AutoLamellaTask]]:
    """Get all available tasks.

    Returns tasks in priority order (highest to lowest):
    1. Built-in tasks
    2. Runtime registered tasks
    3. Plugin tasks

    Returns:
        Dictionary mapping task type strings to task classes
    """
    return TASK_PLUGINS.all()


def load_task_config(
    ddict: Dict[str, Any],
) -> "EventedDict[str, AutoLamellaTaskConfig]":
    """name -> typed config, skipping (with a warning) any type this build
    does not know."""
    from psygnal.containers import EventedDict

    from fibsem.workflows.task_config import load_task_configs

    return EventedDict(load_task_configs(TASK_PLUGINS, ddict))
