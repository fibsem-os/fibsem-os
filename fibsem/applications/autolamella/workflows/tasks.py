
######## TASK DEFINITIONS ########


import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from fibsem import acquire, alignment, calibration
from fibsem import config as fcfg
from fibsem.applications.autolamella.protocol.validation import (
    DEFAULT_ALIGNMENT_AREA,
    FIDUCIAL_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)

from fibsem.applications.autolamella.structures import (
    Experiment,
    Lamella,
    AutoLamellaTaskConfig,
    AutoLamellaTaskState,
)
from psygnal import Signal, evented

from fibsem.applications.autolamella.ui import AutoLamellaUI
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    get_supervision,
    log_status_message,
    set_images_ui,
    update_alignment_area_ui,
    update_detection_ui,
    update_experiment_ui,
    update_milling_ui,
    update_status_ui,
    ask_user,
)
from fibsem.constants import DEGREE_SYMBOL
from fibsem.detection.detection import (
    Feature,
    LamellaBottomEdge,
    LamellaCentre,
    LamellaTopEdge,
    VolumeBlockCentre,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.utils import get_pattern_reduced_area
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.transformations import is_close_to_milling_angle, move_to_milling_angle
from fibsem.utils import format_duration

TAutoLamellaTaskConfig = TypeVar(
    "TAutoLamellaTaskConfig", bound="AutoLamellaTaskConfig"
)

# TODO: create a ui for mill task, update_milling_ui doesnt work for this
MAX_ALIGNMENT_ATTEMPTS = 3
STRESS_RELIEF_KEY = "stress-relief"





class AutoLamellaTask(ABC):
    """Base class for AutoLamella tasks."""
    config_cls: ClassVar[AutoLamellaTaskConfig]
    config: AutoLamellaTaskConfig

    def __init__(self,
                 microscope: FibsemMicroscope,
                 config: AutoLamellaTaskConfig,
                 lamella: Lamella,
                 parent_ui: Optional['AutoLamellaUI'] = None):
        self.microscope = microscope
        self.config = config
        self.lamella = lamella
        self.parent_ui = parent_ui
        self.task_id = str(uuid.uuid4())

    @property
    def task_name(self) -> str:
        """Return the name of the task."""
        return self.config.task_name

    @property
    def display_name(self) -> str:
        """Return the display name of the task."""
        return self.config.display_name

    def run(self) -> None:
        self.pre_task()
        self._run()
        self.post_task()

    @abstractmethod
    def _run(self) -> None:
        pass

    def pre_task(self) -> None:
        logging.info(f"Running {self.task_name} ({self.task_id}) for {self.lamella.name} ({self.lamella._id})")

        # pre-task
        self.lamella.task.name = self.task_name
        self.lamella.task.start_timestamp = datetime.timestamp(datetime.now())
        self.lamella.task.task_id = self.task_id
        self.log_status_message("STARTED")

    def post_task(self) -> None:
        # post-task
        if self.lamella.task is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.lamella.state.microscope_state = self.microscope.get_microscope_state()
        self.lamella.task.end_timestamp = datetime.timestamp(datetime.now())
        self.log_status_message("FINISHED")
        self.update_status_ui("Finished")
        self.lamella.task_config[self.task_name] = deepcopy(self.config)
        self.lamella.task_history.append(deepcopy(self.lamella.task))

    def log_status_message(self, message: str) -> None:
        logging.debug({"msg": "status", 
                       "timestamp": datetime.now().isoformat(),
                       "lamella": self.lamella.name,
                       "lamella_id": self.lamella._id,
                       "task_id": self.task_id,
                       "task": self.task_name, 
                       "step": message})
        if self.lamella.task is not None:
            self.lamella.task.step = message

    def update_status_ui(self, message: str) -> None:
        update_status_ui(self.parent_ui, f"{self.lamella.name} [{self.task_name}] {message}")

@dataclass
class MillTrenchTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillTrenchTask."""
    align_reference: bool = False  # whether to align to a trench reference image
    charge_neutralisation: bool = True
    orientation: str = "FIB"
    task_name: ClassVar[str] = "MILL_TRENCH"
    display_name: ClassVar[str] = "Trench Milling"


class MillTrenchTask(AutoLamellaTask):
    """Task to mill the trench for a lamella."""
    config_cls: ClassVar[Type[MillTrenchTaskConfig]] = MillTrenchTaskConfig
    config: MillTrenchTaskConfig

    def _run(self) -> None:

        # TODO: make the pre-task and post-task updates more generic, so they can be reused in other tasks
        """Run the task to mill the trench for a lamella."""

        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_TRENCH")
        self.update_status_ui("Moving to Trench Position...")
        trench_position = self.microscope.get_target_position(self.lamella.stage_position, 
                                                              self.config.orientation)
        self.microscope.safe_absolute_stage_movement(trench_position)

        # align to reference image
        # TODO: support saving a reference image when selecting the trench from minimap
        reference_image_path = os.path.join(self.lamella.path, "ref_PositionReady.tif")
        if os.path.exists(reference_image_path) and self.config.align_reference:
            self.log_status_message("ALIGN_TRENCH_REFERENCE")
            self.update_status_ui("Aligning Trench Reference...")
            ref_image = FibsemImage.load(reference_image_path)
            alignment.multi_step_alignment_v2(microscope=self.microscope, 
                                            ref_image=ref_image, 
                                            beam_type=BeamType.ION, 
                                            alignment_current=None,
                                            steps=1, subsystem="stage")

        self.log_status_message("MILL_TRENCH")

        # get trench milling stages
        milling_task_config = self.config.milling[TRENCH_KEY]

        # acquire reference images
        image_settings.hfw = milling_task_config.field_of_view
        image_settings.filename = f"ref_{self.task_name}_start"
        image_settings.save = True
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)
        self.update_status_ui("Preparing Trench...")

        parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
        milling_task = run_milling_task(self.microscope, milling_task_config, parent_widget)

        # log the task configuration
        self.config.milling[TRENCH_KEY] = deepcopy(milling_task.config)

        # charge neutralisation
        if self.config.charge_neutralisation:
            self.log_status_message("CHARGE_NEUTRALISATION")
            self.update_status_ui("Neutralising Sample Charge...")
            image_settings.beam_type = BeamType.ELECTRON
            calibration.auto_charge_neutralisation(self.microscope, image_settings)

        # reference images
        self.log_status_message("REFERENCE_IMAGES")
        reference_images = acquire.take_set_of_reference_images(
            microscope=self.microscope,
            image_settings=image_settings,
            hfws=(fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH),
            filename=f"ref_{self.task_name}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

@dataclass
class MillUndercutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillUndercutTask."""
    orientation: str = "SEM"
    milling_angles: List[float] = field(default_factory=lambda: [25, 20])  # in degrees
    task_name: ClassVar[str] = "MILL_UNDERCUT"
    display_name: ClassVar[str] = "Undercut Milling"


class MillUndercutTask(AutoLamellaTask):
    """Task to mill the undercut for a lamella."""
    config: MillUndercutTaskConfig
    config_cls: ClassVar[Type[MillUndercutTaskConfig]] = MillUndercutTaskConfig

    def _run(self) -> None:

        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        checkpoint = "autolamella-waffle-20240107.pt" # if self.lamella.protocol.options.checkpoint is None else self.lamella.protocol.options.checkpoint

        # move to sem orientation
        self.log_status_message("MOVE_TO_UNDERCUT")
        self.update_status_ui("Moving to Undercut Position...")
        undercut_position = self.microscope.get_target_position(self.lamella.stage_position, 
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
            validate=validate,
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
            self.log_status_message(f"TILT_UNDERCUT_{nid}")
            self.update_status_ui("Tilting to Undercut Position...")
            move_to_milling_angle(microscope=self.microscope, milling_angle=np.radians(undercut_milling_angle))

            # detect
            self.log_status_message(f"ALIGN_UNDERCUT_{nid}")
            image_settings.beam_type = BeamType.ION
            image_settings.hfw = milling_task_config.field_of_view
            image_settings.filename = f"ref_{self.task_name}_align_ml_{nid}"
            image_settings.save = True
            sem_image, fib_image = acquire.take_reference_images(self.microscope, image_settings)
            set_images_ui(self.parent_ui, sem_image, fib_image)

            # get pattern
            scan_rotation = self.microscope.get_scan_rotation(beam_type=BeamType.ION)
            features = [LamellaTopEdge() if np.isclose(scan_rotation, 0) else LamellaBottomEdge()]

            det = update_detection_ui(microscope=self.microscope, 
                                    image_settings=image_settings, 
                                    checkpoint=checkpoint, 
                                    features=features, 
                                    parent_ui=self.parent_ui, 
                                    validate=validate, 
                                    msg=lamella.info)

            # set pattern position
            offset = milling_task_config.stages[0].pattern.height / 2
            point = deepcopy(det.features[0].feature_m)
            point.y += offset if np.isclose(scan_rotation, 0) else -offset
            milling_task_config.stages[0].pattern.point = point

            # mill undercut
            self.log_status_message(f"MILL_UNDERCUT_{nid}")
            # stages = update_milling_ui(self.microscope, [undercut_stage], self.parent_ui,
            #     msg=f"Press Run Milling to mill the Undercut {nid} for {self.lamella.name}. Press Continue when done.",
            #     validate=validate,
            # )
            parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
            milling_task = run_milling_task(self.microscope, milling_task_config, parent_widget)

            # post_milled_undercut_stages.extend(stages)

        # log undercut stages
        self.config.milling[UNDERCUT_KEY] = deepcopy(milling_task.config)

        # take reference images
        self.log_status_message("REFERENCE_IMAGES")
        self.update_status_ui("Acquiring Reference Images...")
        image_settings.beam_type = BeamType.ION
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
        image_settings.save = True
        image_settings.filename=f"ref_{self.task_name}_undercut"
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)

        self.log_status_message("ALIGN_FINAL")
        image_settings.beam_type = BeamType.ION
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

        features = [LamellaCentre()]
        det = update_detection_ui(microscope=self.microscope,
                                    image_settings=image_settings,
                                    checkpoint=checkpoint,
                                    features=features,
                                    parent_ui=self.parent_ui,
                                    validate=validate,
                                    msg=self.lamella.info)

        # align vertical
        self.microscope.vertical_move(
            dx=det.features[0].feature_m.x,
            dy=det.features[0].feature_m.y,
        )

        # take reference images
        self.log_status_message("REFERENCE_IMAGES")
        self.update_status_ui("Acquiring Reference Images...")

        reference_images = acquire.take_set_of_reference_images(
            microscope=self.microscope,
            image_settings=image_settings,
            hfws=(fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH),
            filename=f"ref_{self.task_name}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)


@dataclass
class MillRoughTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillRoughTask."""
    task_name: ClassVar[str] = "MILL_ROUGH"
    display_name: ClassVar[str] = "Rough Milling"


@dataclass
class MillPolishingTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillPolishingTask."""
    task_name: ClassVar[str] = "MILL_POLISHING"
    display_name: ClassVar[str] = "Polishing"


class MillRoughTask(AutoLamellaTask):
    """Task to mill the rough trench for a lamella."""
    config: MillRoughTaskConfig
    config_cls: ClassVar[Type[MillRoughTaskConfig]] = MillRoughTaskConfig

    def _run(self) -> None:
        """Run the task to mill the rough trenches for a lamella."""

        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self.log_status_message("MOVE_TO_LAMELLA")
        self.update_status_ui("Moving to Lamella Position...")

        if self.lamella.milling_pose is None or self.lamella.milling_pose.stage_position is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        milling_position = self.lamella.milling_pose.stage_position

        if self.microscope.get_stage_orientation(milling_position) != "MILLING":
            logging.warning(f"Stage position {milling_position} is not in MILLING orientation...")
            # QUERY: need to think how to handle this. need a defined position for rough milling as 'state' can be arbitarily changed. 
            # milling_position = self.microscope.get_target_position(
            #     stage_position=milling_position, target_orientation="MILLING")
            # self.lamella.state.microscope_state.stage_position = milling_position
        self.microscope.set_microscope_state(self.lamella.milling_pose)

        # TODO: how to ensure this is always the correct position/state?

        # beam_shift alignment
        self.log_status_message("ALIGN_LAMELLA")
        self.update_status_ui("Aligning Reference Images...")
        ref_image = FibsemImage.load(os.path.join(self.lamella.path, "ref_alignment_ib.tif"))
        alignment.multi_step_alignment_v2(microscope=self.microscope, 
                                        ref_image=ref_image, 
                                        beam_type=BeamType.ION, 
                                        alignment_current=None,
                                        steps=MAX_ALIGNMENT_ATTEMPTS)

        # take reference images
        self.update_status_ui("Acquiring Reference Images...")
        image_settings.filename = f"ref_{self.task_name}_start"
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)

        # mill stress relief features
        self.log_status_message("MILL_STRESS_RELIEF")
        milling_task_config = self.config.milling[STRESS_RELIEF_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area # TODO: this should be set in protocol

        parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
        milling_task = run_milling_task(self.microscope, milling_task_config, parent_widget)
        self.config.milling[STRESS_RELIEF_KEY] = deepcopy(milling_task.config)

        # mill rough trench
        self.log_status_message("MILL_LAMELLA")
        milling_task_config = self.config.milling[MILL_ROUGH_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area # TODO: this should be set in protocol

        parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
        milling_task = run_milling_task(self.microscope, milling_task_config, parent_widget)
        self.config.milling[MILL_ROUGH_KEY] = deepcopy(milling_task.config)

        # take reference images
        self.log_status_message("REFERENCE_IMAGES")
        self.update_status_ui("Acquiring Reference Images...")
        reference_images = acquire.take_set_of_reference_images(
            microscope=self.microscope,
            image_settings=image_settings,
            hfws=(fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER),
            filename=f"ref_{self.task_name}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

class MillPolishingTask(AutoLamellaTask):
    """Task to mill the polishing trench for a lamella."""
    config: MillPolishingTaskConfig
    config_cls: ClassVar[Type[MillPolishingTaskConfig]] = MillPolishingTaskConfig

    def _run(self) -> None:
        
        """Run the task to mill the polishing trenches for a lamella."""
        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self.log_status_message("MOVE_TO_LAMELLA")
        self.update_status_ui("Moving to Lamella Position...")

        if self.lamella.milling_pose is None or self.lamella.milling_pose.stage_position is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        milling_position = self.lamella.milling_pose.stage_position

        if self.microscope.get_stage_orientation(milling_position) != "MILLING":
            logging.warning(f"Stage position {milling_position} is not in MILLING orientation...")
            # QUERY: need to think how to handle this. need a defined position for rough milling as 'state' can be arbitarily changed. 
            # milling_position = self.microscope.get_target_position(
            #     stage_position=milling_position, target_orientation="MILLING")
            # self.lamella.state.microscope_state.stage_position = milling_position
        self.microscope.set_microscope_state(self.lamella.milling_pose)

        # TODO: how to ensure this is always the correct position/state?

        # beam_shift alignment
        self.log_status_message("ALIGN_LAMELLA")
        self.update_status_ui("Aligning Reference Images...")
        ref_image = FibsemImage.load(os.path.join(self.lamella.path, "ref_alignment_ib.tif"))
        alignment.multi_step_alignment_v2(microscope=self.microscope, 
                                        ref_image=ref_image, 
                                        beam_type=BeamType.ION, 
                                        alignment_current=None,
                                        steps=MAX_ALIGNMENT_ATTEMPTS)

        # take reference images
        self.update_status_ui("Acquiring Reference Images...")
        image_settings.filename = f"ref_{self.task_name}_start"
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)

        # mill rough trench
        self.log_status_message("MILL_LAMELLA")
        milling_task_config = self.config.milling[MILL_POLISHING_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area # TODO: this should be set in protocol

        parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
        milling_task = run_milling_task(self.microscope, milling_task_config, parent_widget)
        self.config.milling[MILL_POLISHING_KEY] = deepcopy(milling_task.config)

        # take reference images
        self.log_status_message("REFERENCE_IMAGES")
        self.update_status_ui("Acquiring Reference Images...")
        reference_images = acquire.take_set_of_reference_images(
            microscope=self.microscope,
            image_settings=image_settings,
            hfws=(fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER),
            filename=f"ref_{self.task_name}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)


@dataclass
class SpotBurnFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SpotBurnFiducialTask."""
    task_name: ClassVar[str] = "SPOT_BURN_FIDUCIAL"
    display_name: ClassVar[str] = "Spot Burn Fiducial"
    milling_current: float = 60.0e-12  # in Amperes
    exposure_time: int = 10  # in seconds
    orientation: str = "FIB"

class SpotBurnFiducialTask(AutoLamellaTask):
    """Task to mill spot fiducial markers for correlation."""
    config: SpotBurnFiducialTaskConfig
    config_cls: ClassVar[Type[SpotBurnFiducialTaskConfig]] = SpotBurnFiducialTaskConfig

    def _run(self) -> None:
        """Run the task to mill spot fiducial markers for correlation."""
        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to the target position at the FIB orientation
        self.log_status_message("MOVE_TO_SPOT_BURN")
        stage_position = self.lamella.stage_position
        target_position = self.microscope.get_target_position(stage_position=stage_position,
                                                         target_orientation="FIB")
        self.microscope.safe_absolute_stage_movement(target_position)

        # acquire images, set ui
        self.log_status_message("REFERENCE_IMAGES")
        image_settings.filename = f"ref_{self.task_name}_start"
        image_settings.save = True
        sem_image, fib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, sem_image, fib_image)

        self.log_status_message("SPOT_BURN_FIDUCIAL")
        # ask the user to select the position/parameters for spot burns
        msg = f"Run the spot burn workflow for {self.lamella.name}. Press continue when finished."
        ask_user(self.parent_ui, msg=msg, pos="Continue", spot_burn=True)

        # acquire final reference images
        self.log_status_message("REFERENCE_IMAGES")
        image_settings.filename = f"ref_{self.task_name}_end"
        image_settings.save = True
        sem_image, fib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, sem_image, fib_image)

# TODO: we need to split this into select position and setup lamell tasks:
# select position: move to milling angle, correct coincidence, acquire base image
# setup lamella: mill fiducial, acquire alignment image, set alignment area
# then allow the user to modify the other patterns (rough mill, polishing) asynchronously in gui

@dataclass
class SetupLamellaTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SetupLamellaTask."""
    milling_angle: float = 15 # in degrees
    use_fiducial: bool = True
    task_name: ClassVar[str] = "SETUP_LAMELLA"
    display_name: ClassVar[str] = "Setup Lamella"

ATOL_STAGE_TILT = np.radians(1)  # radians, acceptable tolerance for stage tilt in radians

class SetupLamellaTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""
    config: SetupLamellaTaskConfig
    config_cls: ClassVar[Type[SetupLamellaTaskConfig]] = SetupLamellaTaskConfig

    def _run(self) -> None:
        """Run the task to setup the lamella for milling."""

        # bookkeeping
        validate = self.config.supervise
        checkpoint = "autolamella-waffle-20240107.pt" 

        image_settings: ImageSettings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("SELECT_POSITION")
        self.update_status_ui("Aligning Lamella...")

        milling_angle = self.config.milling_angle
        is_close = is_close_to_milling_angle(microscope=self.microscope, 
                                            milling_angle=np.radians(milling_angle),
                                            atol=ATOL_STAGE_TILT * 2)

        if not is_close and validate:
            current_milling_angle = self.microscope.get_current_milling_angle()
            ret = ask_user(parent_ui=self.parent_ui,
                        msg=f"Tilt to specified milling angle ({milling_angle:.1f} {DEGREE_SYMBOL})? "
                        f"Current milling angle is {current_milling_angle:.1f} {DEGREE_SYMBOL}.",
                        pos="Tilt", neg="Skip")
            if ret:
                move_to_milling_angle(microscope=self.microscope, milling_angle=np.radians(milling_angle))

            # move_to_milling_angle(microscope=self.microscope, milling_angle=np.radians(milling_angle))
            # lamella = align_feature_coincident(microscope=microscope, 
            #                                 image_settings=image_settings, 
            #                                 lamella=lamella, 
            #                                 checkpoint=protocol.options.checkpoint, 
            #                                 parent_ui=parent_ui, 
            #                                 validate=validate)
        self.lamella.milling_pose = self.microscope.get_microscope_state()

        self.log_status_message("SETUP_PATTERNS")

        rough_milling_task_config = self.lamella.task_config[MillRoughTaskConfig.task_name].milling[MILL_ROUGH_KEY]
        polishing_milling_task_config = self.lamella.task_config[MillPolishingTaskConfig.task_name].milling[MILL_POLISHING_KEY]
        fiducial_task_config = self.config.milling[FIDUCIAL_KEY]

        assert rough_milling_task_config.field_of_view == polishing_milling_task_config.field_of_view, \
            "Rough and polishing milling tasks must have the same field of view."
        assert rough_milling_task_config.field_of_view == fiducial_task_config.field_of_view, \
            "Rough milling and fiducial tasks must have the same field of view."

        image_settings.hfw = rough_milling_task_config.field_of_view
        image_settings.filename = f"ref_{self.task_name}_start"
        image_settings.save = True
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)

        # fiducial
        if self.config.use_fiducial:
            
            # mill the fiducial
            self.log_status_message("MILL_FIDUCIAL")
            parent_widget = self.parent_ui.milling_widget if self.parent_ui else None
            milling_task = run_milling_task(self.microscope, fiducial_task_config, parent_widget)
            self.config.milling[MILL_POLISHING_KEY] = deepcopy(milling_task.config)

            alignment_hfw = milling_task.config.stages[0].milling.hfw

            # get alignment area based on fiducial bounding box
            self.lamella.alignment_area = get_pattern_reduced_area(stage=milling_task.config.stages[0],
                                                            image=FibsemImage.generate_blank_image(hfw=alignment_hfw),
                                                            expand_percent=20)
        else:
            # non-fiducial based alignment
            self.lamella.alignment_area = FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
            alignment_hfw = rough_milling_task_config.field_of_view

        # update alignment area
        self.log_status_message("ACQUIRE_ALIGNMENT_IMAGE")
        logging.debug(f"alignment_area: {self.lamella.alignment_area}")
        self.lamella.alignment_area = update_alignment_area_ui(alignment_area=self.lamella.alignment_area,
                                                parent_ui=self.parent_ui,
                                                msg="Edit Alignment Area. Press Continue when done.", 
                                                validate=validate)

        # set reduced area for fiducial alignment
        image_settings.reduced_area = self.lamella.alignment_area

        # acquire reference image for alignment
        image_settings.beam_type = BeamType.ION
        image_settings.save = True
        image_settings.hfw = alignment_hfw
        image_settings.filename = "ref_alignment"
        image_settings.autocontrast = False # disable autocontrast for alignment
        fib_image = acquire.acquire_image(self.microscope, image_settings)
        image_settings.reduced_area = None
        image_settings.autocontrast = True

        # take reference images
        log_status_message(self.lamella, "REFERENCE_IMAGES")
        self.update_status_ui("Acquiring Reference Images...")
        reference_images = acquire.take_set_of_reference_images(
            self.microscope,
            image_settings,
            hfws=(fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER),
            filename=f"ref_{self.task_name}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

        self.lamella.milling_pose = self.microscope.get_microscope_state()


class TaskNotRegisteredError(Exception):
    """Exception raised when a task is not registered in the TASK_REGISTRY."""
    def __init__(self, task_name: str):
        super().__init__(f"Task '{task_name}' is not registered in the TASK_REGISTRY.")
        self.task_name = task_name

    def __str__(self) -> str:
        return f"TaskNotRegisteredError: {self.task_name}"


def load_task_config(ddict: Dict[str, Any]) -> Dict[str, AutoLamellaTaskConfig]:
    """Load task configurations from a dictionary."""
    task_config = {}
    for name, v in ddict.items():
        if name not in TASK_REGISTRY:
            # raise ValueError(f"Task '{name}' is not registered.")
            logging.warning(f"Task '{name}' is not registered. Skipping.")
            continue
        config_class = TASK_REGISTRY[name].config_cls
        task_config[name] = config_class.from_dict(v)
    return task_config

def load_config(task_name: str, ddict: Dict[str, Any]) -> AutoLamellaTaskConfig:
    """Load a task configuration from a dictionary."""
    config_class = get_task_config(name=task_name)
    return config_class.from_dict(ddict)

def get_task_config(name: str) -> Type[AutoLamellaTaskConfig]:
    """Get the task configuration by name."""
    if name not in TASK_REGISTRY:
        raise TaskNotRegisteredError(name)
    return TASK_REGISTRY[name].config_cls  # type: ignore


TASK_REGISTRY: Dict[str, Type[AutoLamellaTask]] = {
    MillTrenchTaskConfig.task_name: MillTrenchTask,
    MillUndercutTaskConfig.task_name: MillUndercutTask,
    MillRoughTaskConfig.task_name: MillRoughTask,
    MillPolishingTaskConfig.task_name: MillPolishingTask,
    SpotBurnFiducialTaskConfig.task_name: SpotBurnFiducialTask,
    SetupLamellaTaskConfig.task_name: SetupLamellaTask,
    # Add other tasks here as needed
}

def run_task(microscope: FibsemMicroscope, 
          task_name: str, 
          lamella: 'Lamella', 
          parent_ui: Optional[AutoLamellaUI] = None) -> None:
    """Run a specific AutoLamella task."""
    task_cls = TASK_REGISTRY.get(task_name)
    if task_cls is None:
        raise ValueError(f"Task {task_name} is not registered.")

    task_config = lamella.task_config.get(task_name)
    if task_config is None:
        raise ValueError(f"Task configuration for {task_name} not found in lamella tasks.")


    task = task_cls(microscope=microscope, config=task_config, lamella=lamella, parent_ui=parent_ui)
    task.run()

def run_tasks(microscope: FibsemMicroscope, 
            experiment: 'Experiment', 
            task_names: List[str],
            required_lamella: Optional[List[str]] = None,
            parent_ui: Optional[AutoLamellaUI] = None) -> 'Experiment':
    """Run the specified tasks for all lamellas in the experiment.
    Args:
        microscope (FibsemMicroscope): The microscope instance.
        experiment (Experiment): The experiment containing lamellas.
        task_names (List[str]): List of task names to run.
        required_lamella (Optional[List[str]]): List of lamella names to run tasks on. If None, all lamellas are processed.
        parent_ui (Optional[AutoLamellaUI]): Parent UI for status updates.
    Returns:
        Experiment: The updated experiment with task results.
    """
    for task_name in task_names:
        for lamella in experiment.positions:
            if required_lamella and lamella.name not in required_lamella:
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Not in required lamella list.")
                continue

            # check if this lamella has already completed the task
            if lamella.has_completed_task(task_name):
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Already completed.")
                continue

            # check if this lamella has completed required tasks
            task_requires = experiment.task_protocol.workflow_config.requires(task_name)
            if task_requires and not all(lamella.has_completed_task(req) for req in task_requires):
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Required tasks {task_requires} not completed.")
                continue

            # TODO: how to handle:
            # - if the task is already completed
            # - if the task has not completed the required tasks
            # - if the lamella has a defect
            # - how to define the workflow and required tasks
            # - how to mark the workflow as 'completed'
            # - how to handle supervision: only enabled when parent_ui available

            run_task(microscope=microscope,
                     task_name=task_name,
                     lamella=lamella,
                     parent_ui=parent_ui)
            experiment.save()
    return experiment


# TODO: supervision should be handled globally, not per lamella