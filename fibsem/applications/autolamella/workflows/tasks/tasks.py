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
    "SetupCoincidenceMillingTask",
    "SetupCoincidenceMillingTaskConfig",
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
    # acquire fluorescence image
    "AcquireFluorescenceImageConfig",
    "AcquireFluorescenceImageTask",
    # select fluorescence position
    "SelectFluorescencePositionConfig",
    "SelectFluorescencePositionTask",
]

from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
    AcquireFluorescenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    _LIFECYCLE_STEPS,
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    MAX_ALIGNMENT_ATTEMPTS,
    AutoLamellaTask,
    TAutoLamellaTaskConfig,
    get_task_supervision,
)
from fibsem.applications.autolamella.workflows.tasks.basic_milling import (
    BasicMillingTask,
    BasicMillingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTask,
    MillFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.polishing import (
    MillPolishingTask,
    MillPolishingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.select_fluorescence_position import (
    SelectFluorescencePositionConfig,
    SelectFluorescencePositionTask,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    SetupCoincidenceMillingTask,
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTask,
    SpotBurnFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTask,
    MillTrenchTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
)

# related tasks (must be defined after task definitions, due to circular nature)
MillFiducialTaskConfig.related_tasks = [MillRoughTaskConfig, MillPolishingTaskConfig]
MillRoughTaskConfig.related_tasks = [MillFiducialTaskConfig, MillPolishingTaskConfig]
MillPolishingTaskConfig.related_tasks = [MillFiducialTaskConfig, MillRoughTaskConfig]
