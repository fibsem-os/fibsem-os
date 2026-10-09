######## TRENCH TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem import calibration
from fibsem.applications.autolamella.protocol.constants import TRENCH_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    MAX_ALIGNMENT_ATTEMPTS,
    AutoLamellaTask,
)
from fibsem.autofunctions.charge_neutralisation import auto_charge_neutralisation
from fibsem.structures import BeamType, field_meta


@dataclass
class MillTrenchTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillTrenchTask."""

    # align_reference aligned to ref_PositionReady.tif, which only the legacy
    # workflow wrote.
    retired_parameters: ClassVar[frozenset] = frozenset({"align_reference"})
    charge_neutralisation: bool = field(
        default=True,  # whether to perform charge neutralisation
        metadata=field_meta(tooltip="Whether to perform charge neutralisation"),
    )
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = field(
        default=None,
        metadata=field_meta(
            tooltip="The orientation to perform trench milling in",
            items=("SEM", "FIB", "MILLING", None),
        ),
    )
    task_type: ClassVar[str] = "MILL_TRENCH"
    display_name: ClassVar[str] = "Trench Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({TRENCH_KEY: DEFAULT_MILLING_CONFIG[TRENCH_KEY]})


class MillTrenchTask(AutoLamellaTask):
    """Task to mill the trench for a lamella."""

    # Work at the microscope with the tools, then Continue: the milling
    # session. What needs a person present when the task is supervised.
    sessions = ("milling",)

    config_cls: ClassVar[Type[MillTrenchTaskConfig]] = MillTrenchTaskConfig
    config: MillTrenchTaskConfig

    def _run(self) -> None:
        """Run the task to mill the trench for a lamella."""

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_TRENCH", "Moving to Trench Position...")
        trench_position = self._get_stage_position_for_orientation(
            self.lamella.stage_position, self.config.orientation
        )
        self.microscope.safe_absolute_stage_movement(trench_position)

        # get trench milling stages
        milling_task_config = self.config.milling[TRENCH_KEY]

        # acquire reference images
        self._acquire_reference_image(
            image_settings, field_of_view=milling_task_config.field_of_view
        )

        # log the task configuration
        self.log_status_message("MILL_TRENCH", "Milling Trench...")
        msg = f"Press Run Milling to mill the Trench for {self.lamella.name}. Press Continue when done."
        milling_task_config.acquisition.imaging.path = self.lamella.path
        milling_task_config = self.update_milling_config_ui(
            milling_task_config,
            msg=msg,
        )
        self.config.milling[TRENCH_KEY] = deepcopy(milling_task_config)

        # charge neutralisation
        if self.config.charge_neutralisation:
            self.log_status_message(
                "CHARGE_NEUTRALISATION", "Neutralising Sample Charge..."
            )
            image_settings.beam_type = BeamType.ELECTRON
            auto_charge_neutralisation(self.microscope, image_settings)

        # reference images
        self._acquire_set_of_reference_images(image_settings)
