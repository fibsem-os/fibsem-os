
######## POLISHING TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Type

from fibsem.applications.autolamella.protocol.constants import MILL_POLISHING_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)


@dataclass
class MillPolishingTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillPolishingTask."""
    sync_to_poi: bool = field(
        default=True,
        metadata={
            "label": "Link to Point of Interest",
            "help": "Link the milling pattern positions to the point of interest. Pattern positions will update when the POI is updated."},
    )
    task_type: ClassVar[str] = "MILL_POLISHING"
    display_name: ClassVar[str] = "Polishing"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({MILL_POLISHING_KEY: DEFAULT_MILLING_CONFIG[MILL_POLISHING_KEY]})


class MillPolishingTask(AutoLamellaTask):
    """Task to mill the polishing trench for a lamella."""
    config: MillPolishingTaskConfig
    config_cls: ClassVar[Type[MillPolishingTaskConfig]] = MillPolishingTaskConfig

    def _run(self) -> None:

        """Run the task to mill the polishing trenches for a lamella."""
        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self._move_to_milling_pose()

        # beam_shift alignment
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        # reference images
        self._acquire_reference_image(image_settings, field_of_view=self.config.milling[MILL_POLISHING_KEY].field_of_view)

        # mill polishing
        self.log_status_message("MILL_LAMELLA", "Milling Polishing Lamella...")
        milling_task_config = self.config.milling[MILL_POLISHING_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area
        milling_task_config.acquisition.imaging.path = self.lamella.path

        msg = f"Press Run Milling to mill the polishing for {self.lamella.name}. Press Continue when done."
        milling_task_config = self.update_milling_config_ui(milling_task_config, msg=msg)
        self.config.milling[MILL_POLISHING_KEY] = deepcopy(milling_task_config)

        # reference images
        self._acquire_set_of_reference_images(image_settings)
