"""Mill at a site: move to its milling pose, run the task's milling, and take
the reference images."""

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.workflows.proposals import MILLING
from fibsem.workflows.site_task import SiteTask, SiteTaskConfig


def default_milling() -> FibsemMillingTaskConfig:
    """One rectangle, at the milling defaults."""
    return FibsemMillingTaskConfig(
        name="Mill",
        field_of_view=80e-6,
        stages=[
            FibsemMillingStage(
                name="Mill",
                pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
            )
        ],
    )


@dataclass
class MillTaskConfig(SiteTaskConfig):
    task_type: ClassVar[str] = "MILL"
    display_name: ClassVar[str] = "Mill"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = {"milling": default_milling()}


class MillTask(SiteTask):
    """Move to the milling pose, run the milling, and take reference images."""

    # Work at the microscope with the tools, then Continue: the milling
    # session. What needs a person present when the task is supervised.
    asks = (MILLING,)

    config: MillTaskConfig
    config_cls: ClassVar[Type[MillTaskConfig]] = MillTaskConfig

    def _run(self) -> None:
        self._move_to_pose()
        self.log_status_message("RUN_MILLING", "Milling...")
        for key, milling_task_config in self.config.milling.items():
            milling_task_config = self.update_milling_config_ui(milling_task_config)
            self.config.milling[key] = deepcopy(milling_task_config)
        self._acquire_set_of_reference_images(self.config.imaging)
