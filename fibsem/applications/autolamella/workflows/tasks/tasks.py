
# Backwards-compatible re-exports — do not remove
# This file re-exports all public symbols from the split task modules so that
# existing callers importing from this module continue to work without changes.

__all__ = [
    # base
    "AutoLamellaTask",
    "get_task_supervision",
    "MAX_ALIGNMENT_ATTEMPTS",
    "ALIGNMENT_REFERENCE_IMAGE_FILENAME",
    "TAutoLamellaTaskConfig",
    "_LIFECYCLE_STEPS",
    # trench
    "MillTrenchTaskConfig",
    "MillTrenchTask",
    # undercut
    "MillUndercutTaskConfig",
    "MillUndercutTask",
    # rough
    "MillRoughTaskConfig",
    "MillRoughTask",
    # polishing
    "MillPolishingTaskConfig",
    "MillPolishingTask",
    # fiducial
    "MillFiducialTaskConfig",
    "MillFiducialTask",
    # spot burn
    "SpotBurnFiducialTaskConfig",
    "SpotBurnFiducialTask",
    # reference image
    "AcquireReferenceImageConfig",
    "AcquireReferenceImageTask",
    # select position
    "SelectMillingPositionTaskConfig",
    "SelectMillingPositionTask",
    # basic milling
    "BasicMillingTaskConfig",
    "BasicMillingTask",
]

from fibsem.applications.autolamella.workflows.tasks.base import (
    AutoLamellaTask,
    get_task_supervision,
    MAX_ALIGNMENT_ATTEMPTS,
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    TAutoLamellaTaskConfig,
    _LIFECYCLE_STEPS,
)
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTaskConfig,
    MillTrenchTask,
)
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTaskConfig,
    MillUndercutTask,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTaskConfig,
    MillRoughTask,
)
from fibsem.applications.autolamella.workflows.tasks.polishing import (
    MillPolishingTaskConfig,
    MillPolishingTask,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTaskConfig,
    MillFiducialTask,
)
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
    SpotBurnFiducialTask,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
    SelectMillingPositionTask,
)
from fibsem.applications.autolamella.workflows.tasks.basic_milling import (
    BasicMillingTaskConfig,
    BasicMillingTask,
)

# related tasks (must be defined after task definitions, due to circular nature)
MillFiducialTaskConfig.related_tasks = [MillRoughTaskConfig, MillPolishingTaskConfig]
MillRoughTaskConfig.related_tasks = [MillFiducialTaskConfig, MillPolishingTaskConfig]
MillPolishingTaskConfig.related_tasks = [MillFiducialTaskConfig, MillRoughTaskConfig]
