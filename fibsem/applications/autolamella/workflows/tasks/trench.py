
######## TRENCH TASK DEFINITIONS ########

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem import alignment, calibration
from fibsem.applications.autolamella.protocol.constants import TRENCH_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
    MAX_ALIGNMENT_ATTEMPTS,
)
from fibsem.structures import BeamType, FibsemImage


@dataclass
class MillTrenchTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillTrenchTask."""
    align_reference: bool = field(
        default=False,  # whether to align to a trench reference image
        metadata={"help": "Whether to align to a trench reference image"},
    )
    charge_neutralisation: bool = field(
        default=True,  # whether to perform charge neutralisation
        metadata={"help": "Whether to perform charge neutralisation"},
    )
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = field(
        default=None,
        metadata={"help": "The orientation to perform trench milling in", "items": ("SEM", "FIB", "MILLING", None)},
    )
    task_type: ClassVar[str] = "MILL_TRENCH"
    display_name: ClassVar[str] = "Trench Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({TRENCH_KEY: DEFAULT_MILLING_CONFIG[TRENCH_KEY]})


class MillTrenchTask(AutoLamellaTask):
    """Task to mill the trench for a lamella."""
    config_cls: ClassVar[Type[MillTrenchTaskConfig]] = MillTrenchTaskConfig
    config: MillTrenchTaskConfig

    def _run(self) -> None:
        """Run the task to mill the trench for a lamella."""

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_TRENCH", "Moving to Trench Position...")
        trench_position = self._get_stage_position_for_orientation(self.lamella.stage_position,
                                                                   self.config.orientation)
        self.microscope.safe_absolute_stage_movement(trench_position)

        # align to reference image
        # TODO: support saving a reference image when selecting the trench from minimap
        reference_image_path = os.path.join(self.lamella.path, "ref_PositionReady.tif")
        if os.path.exists(reference_image_path) and self.config.align_reference:
            self.log_status_message("ALIGN_TRENCH_REFERENCE", "Aligning Trench Reference...")
            ref_image = FibsemImage.load(reference_image_path)
            alignment.multi_step_alignment_v2(microscope=self.microscope,
                                            ref_image=ref_image,
                                            beam_type=BeamType.ION,
                                            alignment_current=None,
                                            steps=1, subsystem="stage")

        # get trench milling stages
        milling_task_config = self.config.milling[TRENCH_KEY]

        # acquire reference images
        self._acquire_reference_image(image_settings, field_of_view=milling_task_config.field_of_view)

        # log the task configuration
        self.log_status_message("MILL_TRENCH", "Milling Trench...")
        msg = f"Press Run Milling to mill the Trench for {self.lamella.name}. Press Continue when done."
        milling_task_config.acquisition.imaging.path = self.lamella.path
        milling_task_config = self.update_milling_config_ui(milling_task_config,
                                                          msg=msg,
                                                          )
        self.config.milling[TRENCH_KEY] = deepcopy(milling_task_config)

        # charge neutralisation
        if self.config.charge_neutralisation:
            self.log_status_message("CHARGE_NEUTRALISATION", "Neutralising Sample Charge...")
            image_settings.beam_type = BeamType.ELECTRON
            calibration.auto_charge_neutralisation(self.microscope, image_settings)

        # reference images
        self._acquire_set_of_reference_images(image_settings)
