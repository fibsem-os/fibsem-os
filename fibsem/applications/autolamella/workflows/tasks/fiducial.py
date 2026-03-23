
######## FIDUCIAL TASK DEFINITIONS ########

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.milling.patterning.utils import get_pattern_reduced_area
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage


@dataclass
class MillFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillFiducialTask."""

    alignment_expansion: float = field(
        default=100.0,
        metadata={
            "help": "The percentage to expand the alignment area around the fiducial",
            "units": "%",
        },
    )
    align_to_reference: bool = field(
        default=True,
        metadata={
            "help": "Align to the reference image before milling fiducial (if available)"
        }

    )
    task_type: ClassVar[str] = "MILL_FIDUCIAL"
    display_name: ClassVar[str] = "Mill Fiducial"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({FIDUCIAL_KEY: DEFAULT_MILLING_CONFIG[FIDUCIAL_KEY]})


class MillFiducialTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""
    config: MillFiducialTaskConfig
    config_cls: ClassVar[Type[MillFiducialTaskConfig]] = MillFiducialTaskConfig

    def _run(self) -> None:
        """Run the task to setup the lamella for milling."""

        # bookkeeping
        from fibsem.structures import ImageSettings
        image_settings: ImageSettings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self._move_to_milling_pose()

        # beam_shift alignment
        if self.config.align_to_reference:
            self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        fiducial_task_config = self.config.milling[FIDUCIAL_KEY]

        self._acquire_reference_image(image_settings, field_of_view=fiducial_task_config.field_of_view)

        # fiducial
        self.log_status_message("MILL_FIDUCIAL", "Milling Fiducial...")
        msg = f"Press Run Milling to mill the Fiducial for {self.lamella.name}. Press Continue when done."
        fiducial_task_config.alignment.rect = self.lamella.alignment_area
        fiducial_task_config.acquisition.imaging.path = self.lamella.path
        milling_task_config = self.update_milling_config_ui(fiducial_task_config, msg=msg)
        self.config.milling[FIDUCIAL_KEY] = deepcopy(milling_task_config)

        alignment_hfw = milling_task_config.field_of_view
        # get alignment area based on fiducial bounding box
        self.lamella.alignment_area = get_pattern_reduced_area(pattern=milling_task_config.stages[0].pattern,
                                                        image=FibsemImage.generate_blank_image(hfw=alignment_hfw),
                                                        expand_percent=int(self.config.alignment_expansion))

        if not self.lamella.alignment_area.is_valid_reduced_area:
            raise ValueError(f"Invalid alignment area: {self.lamella.alignment_area}, check the field of view for the fiducial milling pattern.")

        # validate alignment area
        self._validate_alignment_area()

        # # acquire alignment reference image
        self._acquire_alignment_reference_image(image_settings=image_settings,
                                      reduced_area=self.lamella.alignment_area,
                                      field_of_view=alignment_hfw)

        # sync alignment area to rough and polishing milling tasks (QUERY: should we sync all tasks?)
        rough_milling_task_config: Optional[FibsemMillingTaskConfig] = None
        rough_milling_name = None
        polishing_milling_task_config: Optional[FibsemMillingTaskConfig] = None
        polishing_milling_name = None
        try:
            # find MILL_ROUGH and MILL_POLISHING task configs
            # we need to store these task names, so we can then update them if they are changed in the gui
            for task_name, task_config in self.lamella.task_config.items():
                if task_config.task_type == "MILL_ROUGH":
                    rough_milling_task_config = task_config.milling[MILL_ROUGH_KEY]
                    rough_milling_name = task_name
                elif task_config.task_type == "MILL_POLISHING":
                    polishing_milling_task_config = task_config.milling[MILL_POLISHING_KEY]
                    polishing_milling_name = task_name
        except Exception as e:
            logging.warning(f"Unable to find MillRoughTaskConfig or MillPolishingTaskConfig in lamella task config: {e}")

        if rough_milling_task_config is not None and rough_milling_name is not None:
            self.lamella.task_config[rough_milling_name].milling[MILL_ROUGH_KEY].alignment.rect = deepcopy(self.lamella.alignment_area)
        if polishing_milling_task_config is not None and polishing_milling_name is not None:
            self.lamella.task_config[polishing_milling_name].milling[MILL_POLISHING_KEY].alignment.rect = deepcopy(self.lamella.alignment_area)

        # reference images
        self._acquire_set_of_reference_images(image_settings)

        # store milling angle and pose
        self.lamella.milling_angle = self.microscope.get_current_milling_angle()
        self.lamella.milling_pose = self.microscope.get_microscope_state()
