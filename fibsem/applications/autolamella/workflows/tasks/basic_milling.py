
######## BASIC MILLING TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.applications.autolamella.protocol.constants import TRENCH_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask


@dataclass
class BasicMillingTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the BasicMillingTask."""
    task_type: ClassVar[str] = "BASIC_MILLING"
    display_name: ClassVar[str] = "Basic Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({"milling": DEFAULT_MILLING_CONFIG[TRENCH_KEY]})


class BasicMillingTask(AutoLamellaTask):
    """A simple milling task that moves to the lamella position, runs milling, and takes reference images."""
    config: BasicMillingTaskConfig
    config_cls: ClassVar[Type[BasicMillingTaskConfig]] = BasicMillingTaskConfig

    def _run(self) -> None:
        """Run the basic milling task."""

        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        self.microscope.safe_absolute_stage_movement(self.lamella.stage_position)

        self.log_status_message("RUN_MILLING", "Milling...")

        for key, milling_task_config in self.config.milling.items():
            milling_task_config.acquisition.imaging.path = self.lamella.path
            milling_task_config = self.update_milling_config_ui( milling_task_config)
            self.config.milling[key] = deepcopy(milling_task_config)

        self.log_status_message("ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images...")
        self._acquire_set_of_reference_images(image_settings)
