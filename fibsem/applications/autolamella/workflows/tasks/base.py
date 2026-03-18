
######## TASK BASE DEFINITIONS ########


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
    ClassVar,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from fibsem import acquire, alignment, calibration, constants, utils
from fibsem import config as fcfg
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    STRESS_RELIEF_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Lamella,
)
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    set_images_ui,
    update_detection_ui,
    update_status_ui,
)
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
    clear_spot_burn_ui,
    update_alignment_area_ui,
    update_spot_burn_parameters,
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
    DEFAULT_ALIGNMENT_AREA,
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

TAutoLamellaTaskConfig = TypeVar(
    "TAutoLamellaTaskConfig", bound="AutoLamellaTaskConfig"
)

MAX_ALIGNMENT_ATTEMPTS = 3
ALIGNMENT_REFERENCE_IMAGE_FILENAME = "ref_alignment_ib.tif"

_LIFECYCLE_STEPS = {"STARTED", "FINISHED"}


class AutoLamellaTask(ABC):
    """Base class for AutoLamella tasks."""
    config_cls: ClassVar[AutoLamellaTaskConfig]
    config: AutoLamellaTaskConfig

    def __init__(self,
                 microscope: FibsemMicroscope,
                 config: AutoLamellaTaskConfig,
                 lamella: Lamella,
                 parent_ui: Optional['AutoLamellaUI'] = None,
                 task_manager: Optional['TaskManager'] = None):
        self.microscope = microscope
        self.config = config
        self.lamella = lamella
        self.parent_ui = parent_ui
        self.task_manager = task_manager
        self.task_id = str(uuid.uuid4())
        self._stop_event = task_manager._stop_event if task_manager else None
        self._last_fib_image: Optional[FibsemImage] = None

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
        self._fire_hook("task_started")
        try:
            self._run()
        except Exception as e:
            self._fire_hook("task_failed", error=str(e))
            raise
        self.post_task()
        self._fire_hook("task_completed")

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        hook_manager = getattr(self.task_manager, "hook_manager", None)
        if hook_manager is None:
            return
        from fibsem.applications.autolamella.workflows.tasks.hooks import HookContext
        hook_manager.fire(HookContext(
            event=event,
            task_name=self.task_name,
            task_type=self.task_type,
            lamella_name=self.lamella.name,
            task_state=self.lamella.task_state,
            error=error,
        ))

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
        self.lamella.task_history.append(deepcopy(self.lamella.task_state))
        if self._last_fib_image is not None:
            self.lamella.save_thumbnail(self._last_fib_image) # TODO: append to the history if task fails?

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
                "supervised": self.validate,
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

        if message not in _LIFECYCLE_STEPS and self.parent_ui is not None:
            self.parent_ui.step_update_signal.emit(display_message or message)

        if display_message is not None:
            self.update_status_ui(message = display_message,
                                  workflow_info = workflow_display_message)

    def update_status_ui(self, message: str, workflow_info: Optional[str] = None) -> None:
        update_status_ui(parent_ui=self.parent_ui,
                         msg=f"{self.lamella.name} [{self.task_name}] {message}",
                         workflow_info=workflow_info)

    def _check_for_abort(self) -> None:
        """Check if the workflow has been aborted from the UI, and raise an InterruptedError if so."""
        if self._stop_event is not None and self._stop_event.is_set():
            raise InterruptedError("Workflow aborted by user.")

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
            wait_for_milling_timeout = 60  # seconds
            start_wait = time.time()
            while not self.parent_ui.milling_task_config_widget.milling_widget.is_milling:
                logging.info("Waiting for milling to start...")
                if time.time() - start_wait > wait_for_milling_timeout:
                    logging.warning(f"Timed out waiting for milling to start after {wait_for_milling_timeout}s.")
                    raise TimeoutError("Timed out waiting for milling to start.")
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
        full_filename = os.path.join(self.lamella.path, filename)

        # validate reference image exists
        if not os.path.exists(full_filename):
            logging.warning(f"Reference image {full_filename} for alignment does not exist" "" \
            f"but was requested by {self.task_name}. Skipping alignment.")
            return

        # load reference image, align
        ref_image = FibsemImage.load(full_filename)
        alignment.multi_step_alignment_v2(microscope=self.microscope,
                                        ref_image=ref_image,
                                        beam_type=BeamType.ION,
                                        alignment_current=None,
                                        use_autocontrast=True,
                                        steps=MAX_ALIGNMENT_ATTEMPTS,
                                        stop_event=self._stop_event,
                                        plot_title=f"{self.lamella.name} - {self.task_name}")

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
        if fib_image is not None:
            self._last_fib_image = fib_image
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
        if fib_image is not None:
            self._last_fib_image = fib_image
        set_images_ui(self.parent_ui, sem_image, fib_image)  # show the last acquired image

    def _move_to_milling_pose(self) -> None:
        """Move to the lamella milling pose."""
        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        if self.lamella.milling_pose is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

    def _acquire_alignment_reference_image(self,
                                            image_settings: ImageSettings,
                                            field_of_view: float,
                                            reduced_area: FibsemRectangle) -> FibsemImage:
        """Acquire alignment reference image with reduced area.
        Args:
            image_settings (ImageSettings): The image settings to use for acquisition.
            field_of_view (float): The field of view to use for acquisition.
            reduced_area (FibsemRectangle): The reduced area to use for acquisition.
        Returns:
            FibsemImage: The acquired alignment reference image.
        """
        self.log_status_message("ACQUIRE_ALIGNMENT_REFERENCE_IMAGE", "Acquiring Alignment Reference Image...")
        alignment_image_settings = deepcopy(image_settings)

        # set reduced area for fiducial alignment
        alignment_image_settings.reduced_area = reduced_area

        # acquire reference image for alignment
        alignment_image_settings.beam_type = BeamType.ION
        alignment_image_settings.save = True
        alignment_image_settings.hfw = field_of_view
        alignment_image_settings.filename = "ref_alignment"
        alignment_image_settings.resolution = (1536, 1024)
        alignment_image_settings.dwell_time = 1e-6
        alignment_image_settings.autocontrast = True # enable autocontrast for alignment
        fib_image = acquire.acquire_image(self.microscope, alignment_image_settings)

        return fib_image

    def _validate_alignment_area(self) -> None:
        """Validate the alignment area with the user."""
        self.log_status_message("VALIDATE_ALIGNMENT_AREA", "Validating Alignment Image...")
        self.lamella.alignment_area = update_alignment_area_ui(alignment_area=self.lamella.alignment_area,
                                                parent_ui=self.parent_ui,
                                                msg="Drag to edit the Alignment Area. Press Continue when done.",
                                                validate=self.validate)


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
