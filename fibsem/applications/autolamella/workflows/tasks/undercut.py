######## UNDERCUT TASK DEFINITIONS ########

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, List, Literal, Optional, Type

import numpy as np

from fibsem import config as fcfg
from fibsem import constants
from fibsem.applications.autolamella.protocol.constants import TRENCH_KEY, UNDERCUT_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    update_detection_ui,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.tasks.trench import MillTrenchTaskConfig
from fibsem.applications.autolamella.workflows.ui import ask_user
from fibsem.autofunctions.charge_neutralisation import auto_charge_neutralisation
from fibsem.detection.detection import LamellaBottomEdge, LamellaCentre, LamellaTopEdge
from fibsem.structures import BeamType, ImageSettings, field_meta


@dataclass
class MillUndercutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillUndercutTask."""

    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = field(
        default="SEM",
        metadata=field_meta(
            tooltip="The orientation to perform undercut milling in",
            items=("SEM", "FIB", "MILLING", None),
        ),
    )
    milling_angles: List[float] = field(
        default_factory=lambda: [25, 20],  # in degrees
        metadata=field_meta(
            tooltip="The angles to mill the undercuts at", unit=constants.DEGREE_SYMBOL
        ),
    )

    model_checkpoint: str = field(
        default="autolamella-waffle-20240107.pt",
        metadata={"parameter": True, "help": "ML model checkpoint"},
    )

    for_liftout: bool = field(
        default=False,
        metadata={
            "help": "Whether the undercut task is being performed for a liftout protocol. Enabling this flag means this will run only for the lamella marked as for liftout block."
        },
    )

    auto_abort: bool = field(
        default=True,
        metadata={
            "help": "Auto abort the step and mark lamella as defect if ML detection for undercut fails. This allows overmilling to be avoided on lamellae where detections went wrong"
        },
    )

    gate_det_based_on_trench_milling: bool = field(
        default=False,
        metadata={
            "help": "WARNING: Experimental feature: Add measure for ML detection based on Trench milling step, estimates lamella size of undercut from known geometry from trench milling"
            + "Helps make ML detection more robust and makes sure bad detections do not ruin sample, Trench milling step must precede undercut. Built primarily for Waffle Method"
        },
    )

    task_type: ClassVar[str] = "MILL_UNDERCUT"
    display_name: ClassVar[str] = "Undercut Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy(
                {UNDERCUT_KEY: DEFAULT_MILLING_CONFIG[UNDERCUT_KEY]}
            )


class MillUndercutTask(AutoLamellaTask):
    """Task to mill the undercut for a lamella."""

    config: MillUndercutTaskConfig
    config_cls: ClassVar[Type[MillUndercutTaskConfig]] = MillUndercutTaskConfig

    def _run(self) -> None:

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        checkpoint = self.config.model_checkpoint

        # move to sem orientation
        self.log_status_message("MOVE_TO_UNDERCUT", "Moving to Undercut Position...")
        undercut_position = self._get_stage_position_for_orientation(
            self.lamella.stage_position, self.config.orientation
        )
        self.microscope.safe_absolute_stage_movement(undercut_position)
        # TODO: support compucentric offset
        #
        # change image hfw for aligning
        align_coincident_image_settings = deepcopy(image_settings)

        align_feature_hfw = fcfg.REFERENCE_HFW_HIGH
        # if lamella is used for liftout, we expect it to be large and require a larger hfw for alignment
        if self.config.for_liftout:
            align_feature_hfw = fcfg.REFERENCE_HFW_INIT_NEEDLE_VIEW_IB

        # align feature coincident
        feature = LamellaCentre()
        lamella = align_feature_coincident(
            microscope=self.microscope,
            image_settings=align_coincident_image_settings,
            lamella=self.lamella,
            checkpoint=checkpoint,
            parent_ui=self.parent_ui,
            validate=self.validate,
            feature=feature,
            hfw=align_feature_hfw,
        )

        self._save_images_for_ml_training(
            image_settings=align_coincident_image_settings,
            acquire_sem=True,
            acquire_fib=True,
        )

        # mill under cut
        milling_task_config = self.config.milling[UNDERCUT_KEY]
        undercut_milling_angles = self.config.milling_angles  # deg

        # TODO: support multiple undercuts?

        if len(milling_task_config.stages) != len(undercut_milling_angles):
            raise ValueError(
                f"Number of undercut milling angles ({len(undercut_milling_angles)}) "
                f"does not match number of undercut milling stages ({len(milling_task_config.stages)})"
            )

        baseline_point_y = [s.pattern.point.y for s in milling_task_config.stages]

        for i, undercut_milling_angle in enumerate(undercut_milling_angles):
            nid = f"{i + 1:02d}"  # helper

            # tilt down, align to trench
            self.log_status_message(
                f"TILT_UNDERCUT_{nid}", f"Tilting to Undercut Position {nid}..."
            )
            self.microscope.move_to_milling_angle(
                milling_angle=np.radians(undercut_milling_angle)
            )

            # detect
            self.log_status_message(
                f"ALIGN_UNDERCUT_{nid}", f"Aligning Undercut Position {nid}..."
            )
            # self._acquire_reference_image(image_settings,
            #                               filename=f"ref_{self.task_name}_align_ml_{nid}",
            #                               field_of_view=milling_task_config.field_of_view)

            # get pattern
            scan_rotation = self.microscope.get_scan_rotation(beam_type=BeamType.ION)

            # once tilted, realign to centre of lamella
            image_settings.hfw = self.config.milling[
                UNDERCUT_KEY
            ].field_of_view  # change hfw for detection to match milling fov
            image_settings.beam_type = BeamType.ION

            undercut_feature = LamellaCentre()

            if self.config.gate_det_based_on_trench_milling:
                lamella_pix = self._estimate_lamella_size(
                    image_settings=image_settings, tilt=undercut_milling_angle
                )
                if lamella_pix is not None:
                    undercut_feature.pix_threshold = lamella_pix
                    logging.info(
                        f"Setting Lamella Pixel Threshold for Undercut Detection to {lamella_pix} based on Trench Milling Step"
                    )

            detected_successfully = self._align_with_ml_locally(
                image_settings=image_settings,
                milling_key=UNDERCUT_KEY,
                feature=undercut_feature,
            )

            if not detected_successfully and self.config.auto_abort:
                # need to mark lamella as failed?
                # should not continue with the rest of the step
                # Only valid when unsupervised, when supervised, user will make decision
                if self.validate:
                    should_abort = ask_user(
                        parent_ui=self.parent_ui,
                        msg="ML Model has failed to detect feature, Abort Undercut and follwing workflow for Lamella? "
                        + "Workflow will attempt to continue onto next lamella",
                        pos="Abort",
                        neg="Continue with Lamella",
                    )
                else:
                    should_abort = True

                if should_abort:
                    logging.info(
                        f"Aborting Undercut Step for Lamella {self.lamella.name} "
                    )
                    logging.info("Marking Lamella as Defect")
                    self.lamella.defect.set_defect(
                        description="Undercut ML detection Failed"
                    )
                    break

            features = [
                LamellaTopEdge()
                if np.isclose(scan_rotation, 0)
                else LamellaBottomEdge()
            ]

            det_Edge = update_detection_ui(
                microscope=self.microscope,
                image_settings=image_settings,
                checkpoint=checkpoint,
                features=features,
                parent_ui=self.parent_ui,
                validate=self.validate,
                msg=lamella.status_info,
            )

            self._save_images_for_ml_training(fib_image=self._last_fib_image)

            # only want to run specific stage
            current_milling_config = deepcopy(milling_task_config)
            current_milling_config.stages = [deepcopy(milling_task_config.stages[i])]

            # set pattern position
            offset = (
                current_milling_config.stages[0].pattern.height / 2
            )  # pattern height
            # offset += current_milling_config.stages[0].pattern.point.y # pattern y point
            # offset += lamella_mid_height # add offset to middle of lamella height

            # top/bottom edge detection y point
            det_y = float(det_Edge.features[0].feature_m.y)

            det_y += offset if np.isclose(scan_rotation, 0) else -offset
            # current_milling_config.stages[0].pattern.point.y += det_y
            current_milling_config.stages[0].pattern.point.y = (
                deepcopy(milling_task_config.stages[i]).pattern.point.y + det_y
            )
            # current_milling_config.stages[0].pattern.point.y = baseline_point_y[i] + det_y

            # mill undercut
            self.log_status_message(f"MILL_UNDERCUT_{nid}")
            msg = f"Press Run Milling to mill the Undercut for {self.lamella.name}. Press Continue when done."
            current_milling_config = self.update_milling_config_ui(
                current_milling_config, msg=msg
            )

            # milling_task_config.stages[i] = current_milling_config.stages[0]

            ## charge neutralise (seems to build up a lot)

            self.log_status_message(
                "CHARGE_NEUTRALISATION", "Neutralising Sample Charge..."
            )
            image_settings.beam_type = BeamType.ELECTRON
            auto_charge_neutralisation(self.microscope, image_settings)

            self._acquire_reference_image(
                image_settings,
                field_of_view=self.config.milling[UNDERCUT_KEY].field_of_view,
                filename=f"ref_undercut_{nid}_final",
            )
        # log undercut stages
        self.config.milling[UNDERCUT_KEY] = deepcopy(milling_task_config)

        # write pose
        self.lamella.milling_pose = self.microscope.get_microscope_state()


        # acquire reference images
        self._acquire_set_of_reference_images(image_settings)

    def _estimate_lamella_size(self, image_settings: ImageSettings, tilt: int):
        """
        Based on Trench Milling Pattern, Lamella geometry can be calculated at different tilt angles.
        This gives an estimated pixel count for the lamella class for detection, which can help identify bad detections
        """

        # find trench key, if this returns error, return default

        trench_milling = None
        for task_config in self.lamella.task_config.values():
            if task_config.task_type == MillTrenchTaskConfig.task_type:
                trench_milling = task_config.milling.get(TRENCH_KEY)
        if trench_milling is None:
            logging.warning(
                "Trench milling configuration not found; "
                "cannot estimate lamella size for undercut detection."
            )
            return None

        try:
            lamella_height = trench_milling.stages[0].pattern.spacing
        except Exception as e:
            logging.warning(f"Error obtaining spacing from trench milling pattern: {e}")
            return None

        try:
            lamella_width = trench_milling.stages[0].pattern.width
        except Exception as e:
            logging.warning(f"Error obtaining width from trench milling pattern: {e}")
            return None

        # calculate px_size

        px_size = image_settings.hfw / image_settings.resolution[0]

        new_height = lamella_height * np.sin(np.radians(tilt))

        estimated_pix = new_height / px_size * lamella_width / px_size

        estimated_pix *= 0.8  # set lower to account for variability, will probably need to play with this number

        return int(estimated_pix)
