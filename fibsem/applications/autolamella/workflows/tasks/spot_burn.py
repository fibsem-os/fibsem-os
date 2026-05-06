
######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem import config as fcfg
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
    clear_spot_burn_ui,
    update_spot_burn_parameters,
)


@dataclass
class SpotBurnFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SpotBurnFiducialTask."""
    task_type: ClassVar[str] = "SPOT_BURN_FIDUCIAL"
    display_name: ClassVar[str] = "Spot Burn Fiducial"
    milling_current: float = field(
        default=60.0e-12,  # in Amperes
        metadata={
            'help': 'Milling current in Amperes',
            'units': 'A',
            'scale': 1e12
        }
    )
    exposure_time: int = field(
        default=10,
        metadata={
            'help': 'Exposure time in seconds',
            'units': 's',
            'scale': 1
        }
    )
    orientation: Literal["SEM", "FIB", "FM", "MILLING", None] = field(
        default="MILLING",
        metadata={"help": "The orientation to perform spot burning in", "items": ("SEM", "FIB", "MILLING")},
    )


class SpotBurnFiducialTask(AutoLamellaTask):
    """Task to mill spot fiducial markers for correlation."""
    config: SpotBurnFiducialTaskConfig
    config_cls: ClassVar[Type[SpotBurnFiducialTaskConfig]] = SpotBurnFiducialTaskConfig

    def _run(self) -> None:
        """Run the task to mill spot fiducial markers for correlation."""
        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to the target position at the FIB orientation
        self.log_status_message("MOVE_TO_SPOT_BURN", "Moving to Spot Burn Position...")
        stage_position = self.lamella.stage_position
        target_position = self._get_stage_position_for_orientation(stage_position,
                                                                   self.config.orientation)
        self.microscope.safe_absolute_stage_movement(target_position)

        # acquire images, set ui
        self._acquire_reference_image(image_settings, field_of_view=fcfg.REFERENCE_HFW_HIGH)


        # update the spot burn parameters in the UI # TODO: allow user to store spot positions?
        params = deepcopy({"milling_current": self.config.milling_current,
                           "exposure_time": self.config.exposure_time})
        self.update_spot_burn_parameters_ui(params)

        # acquire final reference images
        self._acquire_set_of_reference_images(image_settings)

    def update_spot_burn_parameters_ui(self, parameters: dict):
        """Update the spot burn parameters in the UI."""
        update_spot_burn_parameters(parent_ui=self.parent_ui, parameters=parameters)

        # ask the user to select the position/parameters for spot burns
        msg = f"Run the spot burn workflow for {self.lamella.name}. Press continue when finished."
        ask_user(self.parent_ui, msg=msg, pos="Continue", spot_burn=True)

        # clear the spot burn parameters from the UI
        clear_spot_burn_ui(self.parent_ui)
