"""The tasks any app whose items are site items runs, by task type. Each app's
task registry includes them as built-ins."""

from typing import Dict, Type

from fibsem.workflows.site_task import SiteTask
from fibsem.workflows.site_tasks.mill import MillTask, MillTaskConfig
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.workflows.site_tasks.spot_burn import (
    SpotBurnFiducialTask,
    SpotBurnFiducialTaskConfig,
)

__all__ = [
    "BUILTIN_SITE_TASKS",
    "AcquireReferenceImageConfig",
    "AcquireReferenceImageTask",
    "MillTask",
    "MillTaskConfig",
    "SpotBurnFiducialTask",
    "SpotBurnFiducialTaskConfig",
]

BUILTIN_SITE_TASKS: Dict[str, Type[SiteTask]] = {
    cls.config_cls.task_type: cls
    for cls in (AcquireReferenceImageTask, MillTask, SpotBurnFiducialTask)
}
