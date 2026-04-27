
######## UNDERCUT TASK DEFINITIONS ########

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, List, Literal, Optional, Type

import numpy as np

from fibsem import config as fcfg
from fibsem import constants
from fibsem.applications.autolamella.protocol.constants import UNDERCUT_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    update_detection_ui,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.detection.detection import LamellaCentre, LamellaBottomEdge, LamellaTopEdge
from fibsem.structures import BeamType


@dataclass
class MillUndercutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillUndercutTask."""
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = field(
        default="SEM",
        metadata={"help": "The orientation to perform undercut milling in", "items": ("SEM", "FIB", "MILLING", None)},
    )
    milling_angles: List[float] = field(
        default_factory=lambda: [25, 20],  # in degrees
        metadata={"help": "The angles to mill the undercuts at",
                  "units": constants.DEGREE_SYMBOL},
    )
    task_type: ClassVar[str] = "MILL_UNDERCUT"
    display_name: ClassVar[str] = "Undercut Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({UNDERCUT_KEY: DEFAULT_MILLING_CONFIG[UNDERCUT_KEY]})


class MillUndercutTask(AutoLamellaTask):
    """Task to mill the undercut for a lamella."""
    config: MillUndercutTaskConfig
    config_cls: ClassVar[Type[MillUndercutTaskConfig]] = MillUndercutTaskConfig

    def _run(self) -> None:

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        checkpoint = "autolamella-waffle-20240107.pt" # if self.lamella.protocol.options.checkpoint is None else self.lamella.protocol.options.checkpoint

        # move to sem orientation
        self.log_status_message("MOVE_TO_UNDERCUT", "Moving to Undercut Position...")
        undercut_position = self._get_stage_position_for_orientation(self.lamella.stage_position,
                                                                     self.config.orientation)
        self.microscope.safe_absolute_stage_movement(undercut_position)
        # TODO: support compucentric offset

        # align feature coincident
        feature = LamellaCentre()
        lamella = align_feature_coincident(
            microscope=self.microscope,
            image_settings=image_settings,
            lamella=self.lamella,
            checkpoint=checkpoint,
            parent_ui=self.parent_ui,
            validate=self.validate,
            feature=feature,
        )

        # mill under cut
        milling_task_config = self.config.milling[UNDERCUT_KEY]
        post_milled_undercut_stages = []
        undercut_milling_angles = self.config.milling_angles # deg

        # TODO: support multiple undercuts?

        if len(milling_task_config.stages) != len(undercut_milling_angles):
            raise ValueError(
                f"Number of undercut milling angles ({len(undercut_milling_angles)}) "
                f"does not match number of undercut milling stages ({len(milling_task_config.stages)})"
            )

        for i, undercut_milling_angle in enumerate(undercut_milling_angles):

            nid = f"{i+1:02d}" # helper

            # tilt down, align to trench
            self.log_status_message(f"TILT_UNDERCUT_{nid}", f"Tilting to Undercut Position {nid}...")
            self.microscope.move_to_milling_angle(milling_angle=np.radians(undercut_milling_angle))

            # detect
            self.log_status_message(f"ALIGN_UNDERCUT_{nid}", f"Aligning Undercut Position {nid}...")
            self._acquire_reference_image(image_settings,
                                          filename=f"ref_{self.task_name}_align_ml_{nid}",
                                          field_of_view=milling_task_config.field_of_view)

            # get pattern
            scan_rotation = self.microscope.get_scan_rotation(beam_type=BeamType.ION)
            features = [LamellaTopEdge() if np.isclose(scan_rotation, 0) else LamellaBottomEdge()]

            det = update_detection_ui(microscope=self.microscope,
                                    image_settings=image_settings,
                                    checkpoint=checkpoint,
                                    features=features,
                                    parent_ui=self.parent_ui,
                                    validate=self.validate,
                                    msg=lamella.status_info)

            # set pattern position
            offset = milling_task_config.stages[0].pattern.height / 2
            point = deepcopy(det.features[0].feature_m)
            point.y += offset if np.isclose(scan_rotation, 0) else -offset
            milling_task_config.stages[0].pattern.point = point

            # mill undercut
            self.log_status_message(f"MILL_UNDERCUT_{nid}")
            msg=f"Press Run Milling to mill the Undercut for {self.lamella.name}. Press Continue when done."
            milling_task_config = self.update_milling_config_ui(milling_task_config, msg=msg)

            # log the task configuration
            # post_milled_undercut_stages.extend(stages)

        # log undercut stages
        self.config.milling[UNDERCUT_KEY] = deepcopy(milling_task_config)

        # take reference images
        self._acquire_set_of_reference_images(image_settings, filename=f"ref_{self.task_name}_undercut")

        # re-align to lamella centre
        self.log_status_message("ALIGN_FINAL", "Aligning Final Position...")
        image_settings.beam_type = BeamType.ION
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

        features = [LamellaCentre()]
        det = update_detection_ui(microscope=self.microscope,
                                    image_settings=image_settings,
                                    checkpoint=checkpoint,
                                    features=features,
                                    parent_ui=self.parent_ui,
                                    validate=self.validate,
                                    msg=self.lamella.status_info)

        # align vertical
        self.microscope.vertical_move(
            dx=det.features[0].feature_m.x,
            dy=det.features[0].feature_m.y,
        )

        # acquire reference images
        self._acquire_set_of_reference_images(image_settings)
