
######## SELECT MILLING POSITION TASK DEFINITIONS ########

from dataclasses import dataclass, field
from typing import ClassVar, Type

import numpy as np

from fibsem import constants
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import ask_user
from fibsem.structures import BeamType, ImageSettings


@dataclass
class SelectMillingPositionTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SelectMillingPositionTask."""

    milling_angle: float = field(
        default=15,
        metadata={
            "help": "The angle between the FIB and sample used for milling",
            "units": constants.DEGREE_SYMBOL,
        },)
    auto_milling_alignment: bool = field(
        default=False,
        metadata={
            "label": "Auto Milling Angle Alignment",
            "help": "Whether to automatically align for a milling position"},
    )
    use_autofocus: bool = field(
        default=True,
        metadata={
            "label": "Use Autofocus",
            "help": "Whether to autofocus before moving to the milling position"},
    )
    task_type: ClassVar[str] = "SELECT_MILLING_POSITION"
    display_name: ClassVar[str] = "Select Milling Position"


class SelectMillingPositionTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""
    config: SelectMillingPositionTaskConfig
    config_cls: ClassVar[Type[SelectMillingPositionTaskConfig]] = SelectMillingPositionTaskConfig

    def _run(self) -> None:
        """Run the task to select the milling position for the lamella for milling."""

        # bookkeeping
        self.image_settings: ImageSettings = self.config.imaging
        self.image_settings.path = self.lamella.path

        # move to lamella milling position
        self._move_to_milling_pose()

        self.log_status_message("SELECT_POSITION", "Selecting Position...")
        milling_angle = self.config.milling_angle
        is_close = self.microscope.is_close_to_milling_angle(milling_angle=milling_angle)

        # acquire an image at the milling position
        if self.config.use_autofocus:
            self.microscope.auto_focus(beam_type=BeamType.ION)
        self._acquire_reference_image(image_settings=self.image_settings,
                                      filename=f"ref_{self.task_name}_start",
                                      field_of_view=self.config.reference_imaging.field_of_view1)

        if not is_close:
            if self.config.auto_milling_alignment:
                from fibsem.transformations import get_stage_tilt_from_milling_angle
                from fibsem import alignment
                target_stage_tilt_degrees = np.degrees(get_stage_tilt_from_milling_angle(self.microscope,
                                                                                 np.radians(milling_angle)))
                alignment._eucentric_tilt_alignment(microscope=self.microscope,
                                                    image_settings=self.image_settings,
                                                    target_angle=float(target_stage_tilt_degrees),
                                                    step_size=3,
                                                    )

            elif self.validate:
                current_milling_angle = self.microscope.get_current_milling_angle()
                ret = ask_user(parent_ui=self.parent_ui,
                            msg=f"Tilt to specified milling angle ({milling_angle:.1f}{constants.DEGREE_SYMBOL})? "
                            f"Current milling angle is {current_milling_angle:.1f}{constants.DEGREE_SYMBOL}.",
                            pos="Tilt", neg="Skip")
                if ret:
                    self.microscope.move_to_milling_angle(milling_angle=np.radians(milling_angle))
            else:
                self.microscope.move_to_milling_angle(milling_angle=np.radians(milling_angle))

            if self.config.use_autofocus:
                self.microscope.auto_focus(beam_type=BeamType.ION)

            # reacquire image at milling angle
            self._acquire_reference_image(image_settings=self.image_settings,
                                        filename=f"ref_{self.task_name}_post_tilt",
                                        field_of_view=self.config.reference_imaging.field_of_view1)

        # confirm with user to move to milling position
        if self.validate:
            ask_user(parent_ui=self.parent_ui,
                    msg=f"Double click the image to move to the milling position for {self.lamella.name}. "
                        f"Press Continue when done.",
                    pos="Continue")

        # validate alignment area
        self._validate_alignment_area()

        # acquire alignment reference image
        self._acquire_alignment_reference_image(image_settings=self.image_settings,
                                      reduced_area=self.lamella.alignment_area,
                                      field_of_view=self.config.reference_imaging.field_of_view1)

        # reference images
        self._acquire_set_of_reference_images(self.image_settings)

        # store milling angle and pose
        self.lamella.milling_angle = self.microscope.get_current_milling_angle()
        self.lamella.milling_pose = self.microscope.get_microscope_state()
