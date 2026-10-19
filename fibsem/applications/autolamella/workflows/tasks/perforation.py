######## PERFORATION TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Literal, Type

from fibsem.applications.autolamella.protocol.constants import PERFORATION_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.structures import BEAM_ORIENTATIONS, param
from fibsem.workflows.proposals import MILLING


@dataclass
class MillPerforationTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillPerforationTask."""

    orientation: Literal["SEM", "FIB", "MILLING"] = param(
        "SEM",
        choices=BEAM_ORIENTATIONS,
        doc="The orientation to perform perforation milling in",
    )
    task_type: ClassVar[str] = "MILL_PERFORATION"
    display_name: ClassVar[str] = "Perforation Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy(
                {PERFORATION_KEY: DEFAULT_MILLING_CONFIG[PERFORATION_KEY]}
            )


class MillPerforationTask(AutoLamellaTask):
    """A milling task that creates perforations in the lamella."""

    # Work at the microscope with the tools, then Continue: the milling
    # session. What needs a person present when the task is supervised.
    asks = (MILLING,)

    config: MillPerforationTaskConfig
    config_cls: ClassVar[Type[MillPerforationTaskConfig]] = MillPerforationTaskConfig

    def _run(self) -> None:
        """Run the perforation milling task."""

        # bookkeeping
        image_settings = self.config.imaging

        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        target_position = self._get_stage_position_for_orientation(
            deepcopy(self.lamella.stage_position), self.config.orientation
        )
        self.microscope.safe_absolute_stage_movement(target_position)

        self.log_status_message("RUN_MILLING", "Milling...")

        key = next(iter(self.config.milling))
        milling_task_config = self.config.milling[key]
        milling_task_config.alignment.rect = self.lamella.alignment_area
        self._acquire_reference_image(
            image_settings, field_of_view=milling_task_config.field_of_view
        )

        msg = f"Mill the perforations for {self.lamella.name}."
        milling_task_config = self.update_milling_config_ui(
            milling_task_config, msg=msg
        )
        self.config.milling[key] = deepcopy(milling_task_config)

        self.log_status_message(
            "ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images..."
        )
        self._acquire_set_of_reference_images(image_settings)
