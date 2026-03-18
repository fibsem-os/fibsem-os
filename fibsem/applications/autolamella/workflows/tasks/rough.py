
######## ROUGH MILLING TASK DEFINITIONS ########

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem.applications.autolamella.protocol.constants import (
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    STRESS_RELIEF_KEY,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import Point


@dataclass
class MillRoughTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillRoughTask."""
    sync_polishing_position: bool = field(
        default=True,
        metadata={
            "label": "Synchronize Polishing Position",
            "help": "Whether to synchronize the polishing position with the rough milling position (recommended.)"},
    )
    sync_to_poi: bool = field(
        default=True,
        metadata={
            "label": "Link to Point of Interest",
            "help": "Link the milling pattern positions to the point of interest. Pattern positions will update when the POI is updated."},
    )
    reacquire_alignment_reference: bool = field(
        default=False,
        metadata={
            "label": "Reacquire Alignment Reference",
            "help": "Whether to reacquire the alignment reference after milling"},
    )
    task_type: ClassVar[str] = "MILL_ROUGH"
    display_name: ClassVar[str] = "Rough Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({MILL_ROUGH_KEY: DEFAULT_MILLING_CONFIG[MILL_ROUGH_KEY]})


class MillRoughTask(AutoLamellaTask):
    """Task to mill the rough trench for a lamella."""
    config: MillRoughTaskConfig
    config_cls: ClassVar[Type[MillRoughTaskConfig]] = MillRoughTaskConfig

    def _run(self) -> None:
        """Run the task to mill the rough trenches for a lamella."""

        # bookkeeping
        self.image_settings = self.config.imaging
        self.image_settings.path = self.lamella.path

        # move to lamella milling position
        self._move_to_milling_pose()

        # beam_shift alignment
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        # take reference images
        self._acquire_reference_image(self.image_settings,
                                      field_of_view=self.config.milling[MILL_ROUGH_KEY].field_of_view)

        # mill stress relief features # QUERY: should stress relief be a separate task, or just part of mill rough
        # PRO: allows it to be 'separate'
        # CON: doesn't allow for easy management of related tasks, re-ordering
        if STRESS_RELIEF_KEY in self.config.milling:
            self.log_status_message("MILL_STRESS_RELIEF", "Milling Stress Relief Features...")
            milling_task_config = self.config.milling[STRESS_RELIEF_KEY]
            milling_task_config.alignment.rect = self.lamella.alignment_area
            milling_task_config.acquisition.imaging.path = self.lamella.path

            msg=f"Press Run Milling to mill the stress relief features for {self.lamella.name}. Press Continue when done."
            milling_task_config = self.update_milling_config_ui(milling_task_config, msg=msg)
            self.config.milling[STRESS_RELIEF_KEY] = deepcopy(milling_task_config)

        # mill rough trench
        self.log_status_message("MILL_LAMELLA", "Milling Rough Lamella...")
        milling_task_config = self.config.milling[MILL_ROUGH_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area
        milling_task_config.acquisition.imaging.path = self.lamella.path # TODO: move into update_milling_config_ui

        msg=f"Press Run Milling to mill the lamella for {self.lamella.name}. Press Continue when done."
        milling_task_config = self.update_milling_config_ui(milling_task_config, msg=msg)
        self.config.milling[MILL_ROUGH_KEY] = deepcopy(milling_task_config)

        # sync polishing milling task position
        self.sync_polishing_milling_task_position(milling_task_config.stages[0].pattern.point)

        # acquire alignment reference image
        if self.config.reacquire_alignment_reference:
            self._acquire_alignment_reference_image(image_settings=self.image_settings,
                                        reduced_area=self.lamella.alignment_area,
                                        field_of_view=milling_task_config.field_of_view)

        # reference images
        self._acquire_set_of_reference_images(self.image_settings)

    def sync_polishing_milling_task_position(self, rough_milling_point: Point) -> None:
        """Sync the polishing milling task position to the rough milling task position."""
        if not self.config.sync_polishing_position:
            return

        # if polishing task exists, we want to sync the position of the milling patterns
        polishing_milling_task_config: Optional[FibsemMillingTaskConfig] = None
        polishing_task_name: Optional[str] = None
        try:
            for task_name, task_config in self.lamella.task_config.items():
                if task_config.task_type == "MILL_POLISHING":
                    polishing_milling_task_config = task_config.milling[MILL_POLISHING_KEY]
                    polishing_task_name = task_name
                    break
        except Exception as e:
            logging.warning(f"Unable to find MillPolishingTaskConfig in lamella task config: {e}")

        if polishing_milling_task_config is not None and polishing_task_name is not None:
            logging.info("Syncing polishing milling pattern positions with rough milling pattern positions...")
            for polishing_stage in polishing_milling_task_config.stages:
                polishing_stage.pattern.point = deepcopy(rough_milling_point)
            # update lamella task config
            self.lamella.task_config[polishing_task_name].milling[MILL_POLISHING_KEY] = deepcopy(polishing_milling_task_config)
