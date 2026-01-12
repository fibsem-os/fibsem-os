
######## TASK DEFINITIONS ########


import glob
import logging
import os
import random
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
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from psygnal.containers import EventedDict

from fibsem import acquire, alignment, calibration, constants, utils
from fibsem import config as fcfg
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
    STRESS_RELIEF_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    ask_user,
    set_images_ui,
    update_alignment_area_ui,
    update_detection_ui,
    update_status_ui,
)
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
    DEFAULT_ALIGNMENT_AREA,
)
from fibsem.applications.autolamella.workflows._default_milling_config import DEFAULT_MILLING_CONFIG

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI

TAutoLamellaTaskConfig = TypeVar(
    "TAutoLamellaTaskConfig", bound="AutoLamellaTaskConfig"
)

MAX_ALIGNMENT_ATTEMPTS = 3
ALIGNMENT_REFERENCE_IMAGE_FILENAME = "ref_alignment_ib.tif"

# feature flags

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
    orientation: str = field(
        default="FIB",
        metadata={"help": "The orientation to perform trench milling in"},
    )
    task_type: ClassVar[str] = "MILL_TRENCH"
    display_name: ClassVar[str] = "Trench Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({TRENCH_KEY: DEFAULT_MILLING_CONFIG[TRENCH_KEY]})


@dataclass
class MillUndercutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillUndercutTask."""
    orientation: str = field(
        default="SEM",
        metadata={"help": "The orientation to perform undercut milling in"},
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
    task_type: ClassVar[str] = "SELECT_MILLING_POSITION"
    display_name: ClassVar[str] = "Select Milling Position"


@dataclass
class SetupLamellaTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SetupLamellaTask."""

    milling_angle: float = field(
        default=15,
        metadata={
            "help": "The angle between the FIB and sample used for milling",
            "units": constants.DEGREE_SYMBOL,
        },
    )
    use_fiducial: bool = field(
        default=True,
        metadata={"help": "Whether to mill a fiducial marker for alignment"},
    )
    align_to_reference: bool = field(
        default=True,
        metadata={"help": "Whether to align to a reference image before milling the fiducial"},
    )
    alignment_expansion: float = field(
        default=30.0,
        metadata={
            "help": "The percentage to expand the alignment area around the fiducial",
            "units": "%",
        },
    )
    display_fluorescence: bool = field(
        default=True,
        metadata={"help": "Whether to display fluorescence images for lamella setup (if available)"},
    )
    task_type: ClassVar[str] = "SETUP_LAMELLA"
    display_name: ClassVar[str] = "Setup Lamella"

    def __post_init__(self):
        if self.milling == {} and self.use_fiducial:
            self.milling = deepcopy({FIDUCIAL_KEY: DEFAULT_MILLING_CONFIG[FIDUCIAL_KEY]})

@dataclass
class MillRoughTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillRoughTask."""
    sync_polishing_position: bool = field(
        default=True,
        metadata={"help": "Whether to synchronize the polishing position with the rough milling position (recommended.)"},
    )
    task_type: ClassVar[str] = "MILL_ROUGH"
    display_name: ClassVar[str] = "Rough Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({MILL_ROUGH_KEY: DEFAULT_MILLING_CONFIG[MILL_ROUGH_KEY]})

@dataclass
class MillPolishingTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillPolishingTask."""
    task_type: ClassVar[str] = "MILL_POLISHING"
    display_name: ClassVar[str] = "Polishing"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({MILL_POLISHING_KEY: DEFAULT_MILLING_CONFIG[MILL_POLISHING_KEY]})

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
    orientation: Literal["SEM", "FIB", "FM", None] = field(
        default="FIB",
        metadata={"help": "The orientation to perform spot burning in"},
    )

@dataclass
class AcquireReferenceImageConfig(AutoLamellaTaskConfig):
    """Configuration for the AcquireReferenceImageTask."""
    task_type: ClassVar[str] = "ACQUIRE_REFERENCE_IMAGE"
    display_name: ClassVar[str] = "Acquire Reference Image"
    orientation: Literal["SEM", "FIB", "MILLING"] = field(
        default="MILLING",
        metadata={"help": "The orientation to acquire reference images in (SEM, FIB, MILLING)"},
    ) # change to pose?


@dataclass
class BasicMillingTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the BasicMillingTask."""
    task_type: ClassVar[str] = "BASIC_MILLING"
    display_name: ClassVar[str] = "Basic Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy({"milling": DEFAULT_MILLING_CONFIG[TRENCH_KEY]})


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
        self._stop_event = self.parent_ui._workflow_stop_event if self.parent_ui else None

    @property
    def task_type(self) -> str:
        """Return the type of the task."""
        return self.config.task_type

    @property
    def task_name(self) -> str:
        """Return the name of the task."""
        return self.config.task_name

    @property
    def display_name(self) -> str:
        """Return the display name of the task type."""
        return self.config.display_name

    @property
    def validate(self) -> bool:
        """Return whether the task should be validated by the user."""
        return get_task_supervision(self.task_name, self.parent_ui)

    def run(self) -> None:
        self.pre_task()
        self._run()
        self.post_task()

    @abstractmethod
    def _run(self) -> None:
        pass

    def pre_task(self) -> None:
        logging.info(f"Running {self.task_name}, {self.task_type} ({self.task_id}) for {self.lamella.name} ({self.lamella._id})")

        # pre-task
        self.lamella.task_state.name = self.task_name
        self.lamella.task_state.start_timestamp = datetime.timestamp(datetime.now())
        self.lamella.task_state.task_id = self.task_id
        self.lamella.task_state.task_type = self.task_type
        self.lamella.task_state.status = AutoLamellaTaskStatus.InProgress
        self.lamella.task_state.status_message = ""
        self.log_status_message(message="STARTED", 
                                display_message="Started", 
                                workflow_display_message=f"{self.lamella.name} [{self.display_name}]")

    def post_task(self) -> None:
        # post-task
        if self.lamella.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.lamella.task_state.end_timestamp = datetime.timestamp(datetime.now())
        self.lamella.task_state.status = AutoLamellaTaskStatus.Completed
        self.lamella.task_state.status_message = ""
        self.log_status_message(message="FINISHED", display_message="Finished")
        self.log_task_config()
        self.lamella.task_config[self.task_name] = deepcopy(self.config)
        self.lamella.task_history.append(deepcopy(self.lamella.task_state)) # TODO: append to the history if task fails?

    def log_task_config(self) -> None:
        """Log the task configuration to the log file. This can be used for debugging or reporting."""
        logging.debug(
            {
                "msg": "task_config",
                "timestamp": datetime.now().isoformat(),
                "lamella": self.lamella.name,
                "lamella_id": self.lamella._id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "task_config": self.config.to_dict(),
            }
        )

    def log_status_message(self, message: str, 
                           display_message: Optional[str] = None, 
                           workflow_display_message: Optional[str] = None) -> None:
        logging.debug({"msg": "status", 
                       "timestamp": datetime.now().isoformat(),
                       "lamella": self.lamella.name,
                       "lamella_id": self.lamella._id,
                       "task_id": self.task_id,
                       "task_type": self.task_type,
                       "task_name": self.task_name, 
                       "task_step": message})
        if self.lamella.task_state is not None:
            self.lamella.task_state.step = message
            self.lamella.task_state.status_message = display_message if display_message is not None else ""

        if display_message is not None:
            self.update_status_ui(message = display_message, 
                                  workflow_info = workflow_display_message)

    def update_status_ui(self, message: str, workflow_info: Optional[str] = None) -> None:
        update_status_ui(parent_ui=self.parent_ui, 
                         msg=f"{self.lamella.name} [{self.task_name}] {message}", 
                         workflow_info=workflow_info)

    def _check_for_abort(self) -> None:
        """Check if the workflow has been aborted from the UI, and raise an InterruptedError if so."""
        from fibsem.applications.autolamella.workflows.ui import _check_for_abort
        _check_for_abort(self.parent_ui)

    def update_milling_config_ui(self, 
                                 milling_config: FibsemMillingTaskConfig, 
                                 msg: str = "Run Milling",
                                 milling_enabled: bool = True) -> FibsemMillingTaskConfig:
        """Update the milling config in the milling widget, and optionally run the milling task."""
        # headless mode
        if self.parent_ui is None:
            if milling_enabled:
                milling_task = run_milling_task(self.microscope, milling_config, None)
                milling_task_config = milling_task.config
            return milling_task_config

        if self.parent_ui.milling_task_config_widget is None:
            raise ValueError("Milling task config widget is not set in the parent UI.")

        # set milling config in milling widget
        self._set_milling_config_ui(milling_config)

        # ask user to confirm milling config
        pos, neg = "Run Milling", "Continue"

        # we only want the user to confirm the milling patterns, not acatually run them
        if milling_enabled is False:
            pos = "Continue"
            neg = None

        response = True
        if self.validate:
            response = ask_user(self.parent_ui, msg=msg, pos=pos, neg=neg, mill=milling_enabled)

        while response and milling_enabled:
            self.update_status_ui(f"Milling {milling_config.name}...")
            self.parent_ui.milling_task_config_widget.milling_widget.start_milling_signal.emit()

            # wait for milling to start
            wait_for_milling_timeout = 5  # seconds
            start_wait = time.time()
            while not self.parent_ui.milling_task_config_widget.milling_widget.is_milling:
                if time.time() - start_wait > wait_for_milling_timeout:
                    logging.warning(f"Timed out waiting for milling to start after {wait_for_milling_timeout}s.")
                    break
                self._check_for_abort()
                time.sleep(0.1)

            # wait for milling to finish
            logging.info("WAITING FOR MILLING TO FINISH... ")
            while self.parent_ui.milling_task_config_widget.milling_widget.is_milling:
                self._check_for_abort()
                time.sleep(1)

            self.update_status_ui(
                f"Milling {milling_config.name} Complete: {len(milling_config.stages)} stages completed."
            )

            response = False
            if self.validate:
                response = ask_user(self.parent_ui, msg=msg, pos=pos, neg=neg, mill=milling_enabled)

        # get milling config from milling widget
        milling_config = deepcopy(self.parent_ui.milling_task_config_widget.get_config())

        # clear milling config from milling widget
        self.clear_milling_config_ui()

        return milling_config

    def _set_milling_config_ui(self, milling_config: FibsemMillingTaskConfig):
        """Set the milling config in the milling widget."""
        if self.parent_ui is None:
            return

        self._check_for_abort()

        info = {
            "msg": "Updating Milling Config",
            "milling_config": deepcopy(milling_config),
        }

        self.parent_ui.WAITING_FOR_UI_UPDATE = True
        self.parent_ui.workflow_update_signal.emit(info) # type: ignore
        while self.parent_ui.WAITING_FOR_UI_UPDATE:
            time.sleep(0.5)

    def clear_milling_config_ui(self):
        """Clear the milling config from the milling widget."""
        if self.parent_ui is None:
            return

        info = {
            "msg": "Clearing Milling Config",
            "clear_milling_config": True,
        }

        self.parent_ui.WAITING_FOR_UI_UPDATE = True
        self.parent_ui.workflow_update_signal.emit(info) # type: ignore
        while self.parent_ui.WAITING_FOR_UI_UPDATE:
            time.sleep(0.5)

    def _align_reference_image(self, filename: str):
        """Align to a reference image."""
        # beam_shift alignment
        self.log_status_message("ALIGN_REFERENCE_IMAGE", "Aligning Reference Images...")
        ref_image = FibsemImage.load(os.path.join(self.lamella.path, filename))
        alignment.multi_step_alignment_v2(microscope=self.microscope, 
                                        ref_image=ref_image, 
                                        beam_type=BeamType.ION, 
                                        alignment_current=None,
                                        steps=MAX_ALIGNMENT_ATTEMPTS,
                                        stop_event=self._stop_event)

    def _acquire_reference_image(self, image_settings: ImageSettings, filename: Optional[str] = None, field_of_view: float = 150e-6) -> None:
        """Acquire a reference image with given field of view."""
        acquire_fib = self.config.reference_imaging.acquire_fib
        acquire_sem = self.config.reference_imaging.acquire_sem
        return self._acquire_channels(image_settings, 
                                        field_of_view=field_of_view, 
                                        filename=filename, 
                                        acquire_sem=acquire_sem,
                                        acquire_fib=acquire_fib)

    def _acquire_set_of_reference_images(self,
                                 image_settings: ImageSettings, 
                                 filename: Optional[str] = None, 
                                 field_of_views: Optional[Tuple[float, ...]] = None) -> None:
        """Acquire a set of reference images."""
        acquire_fib = self.config.reference_imaging.acquire_fib
        acquire_sem = self.config.reference_imaging.acquire_sem
        if field_of_views is None:
            field_of_views = self.config.reference_imaging.field_of_views
        image_settings = self.config.reference_imaging.imaging
        return self._acquire_set_of_channels(image_settings,
                                                field_of_views=field_of_views,
                                                filename=filename,
                                                acquire_sem=acquire_sem,
                                                acquire_fib=acquire_fib)

    def _acquire_channels(self, 
                          image_settings: ImageSettings, 
                          filename: Optional[str] = None, 
                          field_of_view: float = 150e-6,
                          acquire_sem: bool = True, 
                          acquire_fib: bool = True) -> None:
        """Acquire images for sem/fib channels at given field of view."""
        if filename is None:
            filename = f"ref_{self.task_name}_start"

        self.log_status_message("ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images...")
        image_settings.hfw = field_of_view
        image_settings.filename = filename
        image_settings.save = True
        sem_image, fib_image = acquire.acquire_channels(self.microscope,
                                                        image_settings,
                                                        acquire_sem=acquire_sem,
                                                        acquire_fib=acquire_fib)
        set_images_ui(self.parent_ui, sem_image, fib_image)

    def _acquire_set_of_channels(self, image_settings: ImageSettings, 
                                 field_of_views: Optional[Tuple[float, ...]] = None, 
                                 filename: Optional[str] = None,
                                 acquire_sem: bool = True,
                                 acquire_fib: bool = True) -> None:
        """Acquire a set of images for each sem/fib channel at given field of views."""
        
        if field_of_views is None:
            field_of_views = (fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER)
        if filename is None:
            filename = f"ref_{self.task_name}_final"

        self.log_status_message("ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images...")
        images = acquire.acquire_set_of_channels(
            self.microscope,
            image_settings,
            field_of_views,
            filename=filename,
            acquire_sem=acquire_sem,
            acquire_fib=acquire_fib,
        )

        sem_image, fib_image = images[-1] # last acquired image
        set_images_ui(self.parent_ui, sem_image, fib_image)  # show the last acquired image


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
        trench_position = self.microscope.get_target_position(self.lamella.stage_position, 
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

        self.log_status_message("MILL_TRENCH", "Preparing to Mill Trench...")

        # get trench milling stages
        milling_task_config = self.config.milling[TRENCH_KEY]

        # acquire reference images
        self._acquire_reference_image(image_settings, field_of_view=milling_task_config.field_of_view)

        # log the task configuration
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
        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        if self.lamella.milling_pose is None or self.lamella.milling_pose.stage_position is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

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
                if task_config.task_type == MillPolishingTaskConfig.task_type:
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


class MillPolishingTask(AutoLamellaTask):
    """Task to mill the polishing trench for a lamella."""
    config: MillPolishingTaskConfig
    config_cls: ClassVar[Type[MillPolishingTaskConfig]] = MillPolishingTaskConfig

    def _run(self) -> None:
        
        """Run the task to mill the polishing trenches for a lamella."""
        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        if self.lamella.milling_pose is None or self.lamella.milling_pose.stage_position is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

        # beam_shift alignment
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        # reference images
        self._acquire_reference_image(image_settings, field_of_view=self.config.milling[MILL_POLISHING_KEY].field_of_view)

        # mill polishing 
        self.log_status_message("MILL_LAMELLA", "Milling Polishing Lamella...")
        milling_task_config = self.config.milling[MILL_POLISHING_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area
        milling_task_config.acquisition.imaging.path = self.lamella.path

        msg = f"Press Run Milling to mill the polishing for {self.lamella.name}. Press Continue when done."
        milling_task_config = self.update_milling_config_ui(milling_task_config, msg=msg)
        self.config.milling[MILL_POLISHING_KEY] = deepcopy(milling_task_config)

        # reference images
        self._acquire_set_of_reference_images(image_settings)


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
        if self.config.orientation is None: # use current position
            target_position = stage_position
        else:
            target_position = self.microscope.get_target_position(stage_position=stage_position,
                                                         target_orientation=self.config.orientation)
        self.microscope.safe_absolute_stage_movement(target_position)

        # acquire images, set ui
        self._acquire_reference_image(image_settings, field_of_view=fcfg.REFERENCE_HFW_HIGH)

        self.log_status_message("SPOT_BURN_FIDUCIAL")
        # ask the user to select the position/parameters for spot burns
        msg = f"Run the spot burn workflow for {self.lamella.name}. Press continue when finished."
        ask_user(self.parent_ui, msg=msg, pos="Continue", spot_burn=True)

        # acquire final reference images
        self._acquire_set_of_reference_images(image_settings)


# TODO: we need to split this into select position and setup lamella tasks:
# select position: move to milling angle, correct coincidence, acquire base image
# setup lamella: mill fiducial, acquire alignment image, set alignment area
# then allow the user to modify the other patterns (rough mill, polishing) asynchronously in gui


class SelectMillingPositionTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""
    config: SelectMillingPositionTaskConfig
    config_cls: ClassVar[Type[SelectMillingPositionTaskConfig]] = SelectMillingPositionTaskConfig

    def _run(self) -> None:
        """Run the task to select the milling position for the lamella for milling."""

        # bookkeeping
        self.image_settings: ImageSettings = self.config.imaging
        self.image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        if self.lamella.milling_pose is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

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
                            msg=f"Tilt to specified milling angle ({milling_angle:.1f} {constants.DEGREE_SYMBOL})? "
                            f"Current milling angle is {current_milling_angle:.1f} {constants.DEGREE_SYMBOL}.",
                            pos="Tilt", neg="Skip")
                if ret:
                    self.microscope.move_to_milling_angle(milling_angle=np.radians(milling_angle))


        # acquire an image at the milling position
        self._acquire_reference_image(image_settings=self.image_settings,
                                      filename=f"ref_{self.task_name}_initial_position",
                                      field_of_view=self.config.reference_imaging.field_of_view1)

        # confirm with user to move to milling position
        if self.validate:
            ask_user(parent_ui=self.parent_ui,
                    msg=f"Move to the milling position for {self.lamella.name} in the microscope UI. "
                        f"Press Continue when done.",
                    pos="Continue")

        # reference images
        self._acquire_set_of_reference_images(self.image_settings)

        # store milling angle and pose
        self.lamella.milling_angle = self.microscope.get_current_milling_angle()
        self.lamella.milling_pose = self.microscope.get_microscope_state()


class SetupLamellaTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""
    config: SetupLamellaTaskConfig
    config_cls: ClassVar[Type[SetupLamellaTaskConfig]] = SetupLamellaTaskConfig

    def _run(self) -> None:
        """Run the task to setup the lamella for milling."""

        # bookkeeping
        checkpoint = "autolamella-waffle-20240107.pt" # TODO: where should this come from? .options?

        image_settings: ImageSettings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        stage_position = self.lamella.stage_position
        if self.lamella.milling_pose is not None and self.lamella.milling_pose.stage_position is not None:
            logging.info(f"Lamella {self.lamella.name} already has a milling pose set. Using existing milling pose.")
            stage_position = self.lamella.milling_pose.stage_position
        self.microscope.safe_absolute_stage_movement(stage_position)

        # beam_shift alignment
        filenames = sorted(glob.glob(os.path.join(self.lamella.path, "ref_reference_image*_ib.tif")))
        if filenames and self.config.align_to_reference:
            self.log_status_message("ALIGN_REFERENCE_IMAGE", "Aligning Reference Images...")
            ref_image = FibsemImage.load(filenames[-1])

            # confirm that we are close to the reference image stage position before aligning
            image_stage_position = ref_image.metadata.stage_position
            if image_stage_position.is_close2(self.microscope.get_stage_position()):
                alignment.multi_step_alignment_v2(microscope=self.microscope,
                                                ref_image=ref_image,
                                                beam_type=BeamType.ION,
                                                steps=MAX_ALIGNMENT_ATTEMPTS)

        self.log_status_message("SELECT_POSITION", "Selecting Position...")
        milling_angle = self.config.milling_angle
        is_close = self.microscope.is_close_to_milling_angle(milling_angle=milling_angle)

        if not is_close and self.validate:
            current_milling_angle = self.microscope.get_current_milling_angle()
            ret = ask_user(parent_ui=self.parent_ui,
                        msg=f"Tilt to specified milling angle ({milling_angle:.1f} {constants.DEGREE_SYMBOL})? "
                        f"Current milling angle is {current_milling_angle:.1f} {constants.DEGREE_SYMBOL}.",
                        pos="Tilt", neg="Skip")
            if ret:
                self.microscope.move_to_milling_angle(milling_angle=np.radians(milling_angle))
                # TODO: create an automated eucentric version of this...
                # alignment._eucentric_tilt_alignment(microscope=self.microscope,
                #                                     image_settings=image_settings,
                #                                     target_angle=milling_angle,
                #                                     step_size=3,
                #                                     )

            # move_to_milling_angle(microscope=self.microscope, milling_angle=np.radians(milling_angle))
            # lamella = align_feature_coincident(microscope=microscope, 
            #                                 image_settings=image_settings, 
            #                                 lamella=lamella, 
            #                                 checkpoint=protocol.options.checkpoint, 
            #                                 parent_ui=parent_ui, 
            #                                 validate=validate)
        self.lamella.milling_pose = self.microscope.get_microscope_state()

        # TODO: this assumes the lamella is always aligned coincidentally at the milling angle?

        self.log_status_message("SETUP_PATTERNS", "Setting up Lamella Patterns...")

        rough_milling_task_config: Optional[FibsemMillingTaskConfig] = None
        rough_milling_name = None
        polishing_milling_task_config: Optional[FibsemMillingTaskConfig] = None
        polishing_milling_name = None
        try:
            # TODO: we need to store these task names, so we can then update them if they are changed in the gui    
            for task_name, task_config in self.lamella.task_config.items():
                if task_config.task_type == MillRoughTaskConfig.task_type:
                    rough_milling_task_config = task_config.milling[MILL_ROUGH_KEY]
                    rough_milling_name = task_name
                elif task_config.task_type == MillPolishingTaskConfig.task_type:
                    polishing_milling_task_config = task_config.milling[MILL_POLISHING_KEY]
                    polishing_milling_name = task_name
        except Exception as e:
            logging.warning(f"Unable to find MillRoughTaskConfig or MillPolishingTaskConfig in lamella task config: {e}")
        # find MILL_ROUGH and MILL_POLISHING task configs

        fiducial_task_config = self.config.milling[FIDUCIAL_KEY]

        # assert np.isclose(rough_milling_task_config.field_of_view, polishing_milling_task_config.field_of_view, atol=1e-6), \
        #     "Rough and polishing milling tasks must have the same field of view."
        # assert np.isclose(rough_milling_task_config.field_of_view, fiducial_task_config.field_of_view, atol=1e-6), \
        #     "Rough milling and fiducial tasks must have the same field of view."

        self._acquire_reference_image(image_settings, field_of_view=fiducial_task_config.field_of_view)

        # review rough milling pattern
        if self.validate and rough_milling_task_config is not None and rough_milling_name is not None:
            milling_task_config = self.update_milling_config_ui(rough_milling_task_config,
                                                                msg=f"Review the rough milling pattern for {self.lamella.name}. Press Continue when done.",
                                                                milling_enabled=False)
            self.lamella.task_config[rough_milling_name].milling[MILL_ROUGH_KEY] = deepcopy(milling_task_config)

        # TODO: display milling task config to display lamella milling tasks

        # display max intensity projections of any fluorescence images for the lamella
        if self.config.display_fluorescence:
            self._display_fluorescence_images()

        # fiducial
        if self.config.use_fiducial:

            # mill the fiducial
            self.log_status_message("MILL_FIDUCIAL", "Milling Fiducial...")
            msg = f"Press Run Milling to mill the Fiducial for {self.lamella.name}. Press Continue when done."
            fiducial_task_config.acquisition.imaging.path = self.lamella.path
            milling_task_config = self.update_milling_config_ui(fiducial_task_config, msg=msg)
            self.config.milling[FIDUCIAL_KEY] = deepcopy(milling_task_config)

            alignment_hfw = milling_task_config.field_of_view
            # get alignment area based on fiducial bounding box
            self.lamella.alignment_area = get_pattern_reduced_area(pattern=milling_task_config.stages[0].pattern,
                                                            image=FibsemImage.generate_blank_image(hfw=alignment_hfw),
                                                            expand_percent=int(self.config.alignment_expansion))
        else:
            # non-fiducial based alignment
            self.lamella.alignment_area = FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
            alignment_hfw = 150e-6 #, #rough_milling_task_config.field_of_view

        if not self.lamella.alignment_area.is_valid_reduced_area:
            raise ValueError(f"Invalid alignment area: {self.lamella.alignment_area}, check the field of view for the fiducial milling pattern.")

        # update alignment area
        self.log_status_message("ACQUIRE_ALIGNMENT_IMAGE", "Acquiring Alignment Image...")
        logging.info(f"alignment_area: {self.lamella.alignment_area}")
        self.lamella.alignment_area = update_alignment_area_ui(alignment_area=self.lamella.alignment_area,
                                                parent_ui=self.parent_ui,
                                                msg="Edit Alignment Area. Press Continue when done.", 
                                                validate=self.validate)

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

        # sync alignment area to rough and polishing milling tasks (QUERY: should we sync all tasks?)
        if rough_milling_task_config is not None and rough_milling_name is not None:
            self.lamella.task_config[rough_milling_name].milling[MILL_ROUGH_KEY].alignment.rect = deepcopy(self.lamella.alignment_area)
        if polishing_milling_task_config is not None and polishing_milling_name is not None:
            self.lamella.task_config[polishing_milling_name].milling[MILL_POLISHING_KEY].alignment.rect = deepcopy(self.lamella.alignment_area)

        # reference images
        self._acquire_set_of_reference_images(image_settings)

        # store milling angle and pose
        self.lamella.milling_angle = self.microscope.get_current_milling_angle()
        self.lamella.milling_pose = self.microscope.get_microscope_state()

    def _display_fluorescence_images(self, latest_only: bool = True) -> None:
        """Display fluorescence images in the FM control widget."""
        return


class AcquireReferenceImageTask(AutoLamellaTask):
    """Task to acquire reference image with specified settings."""
    config: AcquireReferenceImageConfig
    config_cls: ClassVar[Type[AcquireReferenceImageConfig]] = AcquireReferenceImageConfig

    def _run(self) -> None:
        """Run the task to acquire reference image with the specified settings."""

        # move to position
        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        stage_position = self.lamella.stage_position
        self.microscope.safe_absolute_stage_movement(stage_position) # TODO: use orientation/pose instead

        if self.validate:
            ask_user(self.parent_ui,
                    msg=f"Acquire reference image for {self.lamella.name}. Press continue when ready.",
                    pos="Continue"
                    )

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("ACQUIRE_REFERENCE_IMAGE", "Acquiring Reference Image...")

        # add the last task completed to the reference image filename
        task_name = "Setup"
        if self.lamella.last_completed_task is not None:
            task_name = self.lamella.last_completed_task.name.replace(" ", "-")

        # acquire reference images
        filename = f"ref_reference_image-{task_name}-{utils.current_timestamp_v3()}"
        self._acquire_set_of_reference_images(image_settings=image_settings, filename=filename)


class BasicMillingTask(AutoLamellaTask):
    """A simple milling task that moves to the lamella position, runs milling, and takes reference images."""
    config: BasicMillingTaskConfig
    config_cls: ClassVar[Type[BasicMillingTaskConfig]] = BasicMillingTaskConfig

    def _run(self) -> None:
        """Run the basic milling task."""

        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        self.microscope.safe_absolute_stage_movement(self.lamella.stage_position)

        self.log_status_message("RUN_MILLING", "Milling...")

        for key, milling_task_config in self.config.milling.items():
            milling_task_config.acquisition.imaging.path = self.lamella.path
            milling_task_config = self.update_milling_config_ui( milling_task_config)
            self.config.milling[key] = deepcopy(milling_task_config)

        self.log_status_message("ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images...")
        self._acquire_set_of_reference_images(image_settings)


def get_task_supervision(task_name: str, 
                    parent_ui: Optional['AutoLamellaUI'] = None) -> bool:
    """Get supervision status for a task."""
    if parent_ui is None:
        return False
    if not hasattr(parent_ui, 'experiment') or not hasattr(parent_ui.experiment, 'task_protocol'):
        logging.warning("Parent UI does not have an experiment or task protocol.")
        return False
    if parent_ui.experiment is None or parent_ui.experiment.task_protocol is None:
        logging.warning("Parent UI experiment task protocol is None.")
        return False
    return parent_ui.experiment.task_protocol.get_supervision(task_name)


class TaskNotRegisteredError(Exception):
    """Exception raised when a task is not registered in the TASK_REGISTRY."""
    def __init__(self, task_type: str):
        super().__init__(f"Task '{task_type}' is not registered in the TASK_REGISTRY.")
        self.task_type = task_type

    def __str__(self) -> str:
        return f"TaskNotRegisteredError: {self.task_type}"


def load_task_config(ddict: Dict[str, Any]) -> EventedDict[str, AutoLamellaTaskConfig]:
    """Load task configurations from a dictionary."""
    task_config = EventedDict()
    for name, v in ddict.items():
        task_type = v.get("task_type")
        if task_type not in TASK_REGISTRY:
            # raise ValueError(f"Task '{name}' is not registered.")
            logging.warning(f"Task '{name}' is not registered. Skipping.")
            continue
        config_class = TASK_REGISTRY[task_type].config_cls
        task_config[name] = config_class.from_dict(v)
        task_config[name].task_name = name
    return task_config

def load_config(task_type: str, ddict: Dict[str, Any]) -> AutoLamellaTaskConfig:
    """Load a task configuration from a dictionary."""
    config_class = get_task_config(task_type=task_type)
    return config_class.from_dict(ddict)

def get_task_config(task_type: str) -> Type[AutoLamellaTaskConfig]:
    """Get the task configuration by name."""
    if task_type not in TASK_REGISTRY:
        raise TaskNotRegisteredError(task_type)
    return TASK_REGISTRY[task_type].config_cls  # type: ignore


TASK_REGISTRY: Dict[str, Type[AutoLamellaTask]] = {
    MillTrenchTaskConfig.task_type: MillTrenchTask,
    MillUndercutTaskConfig.task_type: MillUndercutTask,
    MillRoughTaskConfig.task_type: MillRoughTask,
    MillPolishingTaskConfig.task_type: MillPolishingTask,
    SpotBurnFiducialTaskConfig.task_type: SpotBurnFiducialTask,
    SetupLamellaTaskConfig.task_type: SetupLamellaTask,
    AcquireReferenceImageConfig.task_type: AcquireReferenceImageTask,
    BasicMillingTaskConfig.task_type: BasicMillingTask,
    SelectMillingPositionTaskConfig.task_type: SelectMillingPositionTask,
    # Add other tasks here as needed
}

def run_task(microscope: FibsemMicroscope, 
          task_name: str, 
          lamella: 'Lamella', 
          parent_ui: Optional['AutoLamellaUI'] = None) -> None:
    """Run a specific AutoLamella task."""

    task_config = lamella.task_config.get(task_name)
    if task_config is None:
        raise ValueError(f"Task configuration for {task_name} not found in lamella tasks.")

    task_cls = TASK_REGISTRY.get(task_config.task_type)
    if task_cls is None:
        raise ValueError(f"Task {task_config.task_type} is not registered.")

    task = task_cls(microscope=microscope,
                    config=task_config,
                    lamella=lamella,
                    parent_ui=parent_ui)
    task.run()

def sync_lamella_config_updates(lamella: 'Lamella', parent_ui: Optional['AutoLamellaUI'] = None) -> 'Lamella':
    """Sync config updates from GUI to lamella before processing.
    
    This is a placeholder implementation that can be extended to:
    - Check for pending config updates in the UI
    - Apply updates to milling parameters, imaging settings, etc.
    - Validate updates for safety during processing
    
    Args:
        lamella: The lamella to update
        parent_ui: Parent UI containing updated configurations
        
    Returns:
        Updated lamella with synced configuration
    """
    if parent_ui is None:
        return lamella
        
    # TODO: Implement actual config sync logic here
    # Example areas to sync:
    # - Milling currents and patterns from UI widgets
    # - Imaging parameters (HFW, resolution, etc.)
    # - Task-specific settings from protocol editor
    # - Lamella-specific overrides
    
    logging.debug(f"Config sync check for lamella {lamella.name} (placeholder)")
    return lamella


# TODO: create a TaskManager class to handle this?
def run_tasks(microscope: FibsemMicroscope, 
            experiment: 'Experiment', 
            task_names: List[str],
            required_lamella: Optional[List[str]] = None,
            parent_ui: Optional['AutoLamellaUI'] = None) -> 'Experiment':
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
    if required_lamella is None:
        required_lamella = [p.name for p in experiment.positions]

    # TODO: clear the task state for all required lamella, and mark as not started with

    for task_name in task_names:
        for lamella in experiment.positions:
            # Sync config updates from GUI before processing this lamella
            lamella = sync_lamella_config_updates(lamella, parent_ui)

            # if parent_ui:
            #     parent_ui.workflow_update_signal.emit({"msg": f"Starting task {task_name} for lamella {lamella.name}.",
            #         "status": {"task_name": task_name, 
            #                     "task_names": task_names,
            #                     "total_tasks": len(task_names),
            #                     "current_task_index": task_names.index(task_name),
            #                     "lamella_name": lamella.name,
            #                     "lamella_names": required_lamella,
            #                     "current_lamella_index": required_lamella.index(lamella.name),
            #                     "total_lamellas": len(required_lamella)
            #                     }
            #                 })

            if required_lamella and lamella.name not in required_lamella:
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Not in required lamella list.")
                continue
            if lamella.is_failure:
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Marked as failure or has defect.")
                continue

            # check if this lamella has already completed the task
            # if lamella.has_completed_task(task_name): # TODO: need to handle re-running tasks
                # logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Already completed.")
                # continue

            # check if this lamella has completed required tasks 
            task_requirements = experiment.task_protocol.workflow_config.requirements(task_name)
            if task_requirements and not all(lamella.has_completed_task(req) for req in task_requirements):
                logging.info(f"Skipping lamella {lamella.name} for task {task_name}. Required tasks {task_requirements} not completed.")
                continue

            # TODO: how to handle:
            # - if the task is already completed
            # - if the task has not completed the required tasks
            # - if the lamella has a defect
            # - how to define the workflow and required tasks
            # - how to mark the workflow as 'completed'
            # - how to handle supervision: only enabled when parent_ui available
            try:
                run_task(microscope=microscope,
                        task_name=task_name,
                        lamella=lamella,
                        parent_ui=parent_ui)
                experiment.save()
            except Exception as e:
                logging.warning(f"Error running task {task_name} for lamella {lamella.name}: {e}")
                lamella.task_state.status = AutoLamellaTaskStatus.Failed
                lamella.task_state.status_message = str(e)
                experiment.save()

    update_status_ui(parent_ui, "", workflow_info="All tasks completed.")

    print(experiment.task_history_dataframe())
    return experiment
