
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
from fibsem.cancellation import OperationCancelledError
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
from fibsem.fm.structures import ChannelSettings
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
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.fm.structures import FluorescenceImage

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
        # The manager's token, not its raw event: what counts as "cancelled" is
        # the manager's to decide, and a task should not have to know.
        self._stop_event = task_manager.abort_token if task_manager else None
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
            # A user Stop is a cancellation, not a failure — fire the distinct hook so it
            # notifies as "cancelled" (warning) rather than "FAILED" (error).
            cancelled = self._is_cancellation(e)
            # Record before announcing. The hook payload carries task_state, which
            # still reads InProgress from pre_task until this runs.
            #
            # Guarded: anything raising in here would replace the task's own
            # exception with this one, losing what actually went wrong.
            try:
                self.lamella.task_state.status = (
                    AutoLamellaTaskStatus.Cancelled if cancelled else AutoLamellaTaskStatus.Failed
                )
                self.lamella.task_state.status_message = (
                    "Cancelled by user." if cancelled else str(e)
                )
                self._record_outcome()
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook("task_cancelled" if cancelled else "task_failed", error=str(e))
            raise
        finally:
            # In a finally, not in post_task, which never runs when _run() raises. Left
            # set, the next thing acquired -- an operator looking at what went wrong,
            # a supervision step -- would be stamped with the task that just failed.
            # That is silently wrong, and wrong exactly when the record matters most.
            self.microscope.experiment.clear_workflow_metadata()
        self.post_task()
        self._fire_hook("task_completed")

    def _record_outcome(self) -> None:
        """Freeze the finished task_state into task_history.

        Runs on every terminal path, not just success. task_state is a single object
        that the next task's pre_task overwrites, so an outcome that isn't frozen here
        is gone from the experiment as soon as that lamella starts its next task --
        leaving the logfile as the only record that the task ever failed. See FIB-490.
        """
        if self.lamella.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.lamella.task_state.end_timestamp = datetime.timestamp(datetime.now())
        self.lamella.task_history.append(deepcopy(self.lamella.task_state))

    def _is_cancellation(self, exc: Exception) -> bool:
        """Whether ``exc`` is a user Stop rather than a genuine failure: it surfaces as
        OperationCancelledError (milling / autofocus) or InterruptedError (_check_for_abort),
        or the task manager says this task should be unwinding.

        should_abort, not is_stopped: abandoning one task is as much a cancellation
        as ending the run, and an exception thrown on the way out of one should not
        be recorded as a failure."""
        if isinstance(exc, (OperationCancelledError, InterruptedError)):
            return True
        return bool(getattr(getattr(self, "task_manager", None), "should_abort", False))

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        from fibsem.hooks import fire_event
        # Which experiment, and how much of the queue is left. Only the manager knows
        # either, and a task can run without one -- a script driving a task directly, or
        # a stand-in -- so those fields are simply absent then. Fetched the same
        # defensive way as hook_manager below, rather than requiring a real TaskManager.
        get_run_context = getattr(self.task_manager, "hook_run_context", None)
        run_context = get_run_context() if callable(get_run_context) else {}
        fire_event(
            getattr(self.task_manager, "hook_manager", None),
            event,
            task_name=self.task_name,
            task_type=self.task_type,
            item_name=self.lamella.name,
            item_id=self.lamella.id,
            task_id=self.task_id,
            task_state=self.lamella.task_state,
            error=error,
            **run_context,
        )

    @abstractmethod
    def _run(self) -> None:
        pass

    def pre_task(self) -> None:
        logging.info(f"Running {self.task_name}, {self.task_type} ({self.task_id}) for {self.lamella.name} ({self.lamella.id})")

        # pre-task
        # task_state is a single object reused across every run, so each per-run field
        # has to be cleared here or the next run inherits the previous one's value.
        # NOTE: do not "fix" this by assigning a fresh AutoLamellaTaskState. The lamella
        # list and card widgets connect psygnal handlers to this instance and use them as
        # their only refresh trigger; replacing the object orphans those connections and
        # they silently stop updating mid-workflow. See FIB-325 for the real fix.
        self.lamella.task_state.name = self.task_name
        self.lamella.task_state.start_timestamp = datetime.timestamp(datetime.now())
        self.lamella.task_state.end_timestamp = None
        self.lamella.task_state.task_id = self.task_id
        self.lamella.task_state.task_type = self.task_type
        self.lamella.task_state.status = AutoLamellaTaskStatus.InProgress
        self.lamella.task_state.status_message = ""
        self.lamella.task_state.outputs = {}

        # Stamp onto everything acquired from here until run() clears it, so an image
        # says which lamella and task produced it rather than relying on which folder
        # it happens to sit in (FIB-466).
        self.microscope.experiment.set_workflow_metadata(
            item_id=self.lamella.id,
            item_name=self.lamella.name,
            task_id=self.task_id,
            task_name=self.task_name,
        )
        self.log_status_message(message="STARTED",
                                display_message="Started",
                                workflow_display_message=f"{self.lamella.name} [{self.display_name}]")

    def post_task(self) -> None:
        # post-task
        if self.lamella.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.lamella.task_state.status = AutoLamellaTaskStatus.Completed
        self.lamella.task_state.status_message = ""
        self.log_status_message(message="FINISHED", display_message="Finished")
        self.log_task_config()
        self.lamella.task_config[self.task_name] = deepcopy(self.config)
        self._record_outcome()
        if self._last_fib_image is not None:
            self.lamella.save_thumbnail(self._last_fib_image)

    def log_task_config(self) -> None:
        """Log the task configuration to the log file. This can be used for debugging or reporting."""
        logging.debug(
            {
                "msg": "task_config",
                "timestamp": datetime.now().isoformat(),
                "lamella": self.lamella.name,
                "lamella_id": self.lamella.id,
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
                       "lamella_id": self.lamella.id,
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
        """Raise InterruptedError if this task should stop.

        Polls the manager's token, which already answers for every kind of stop —
        so this needs no second way to ask, and neither do the inline token checks
        in the fluorescence, coincidence-milling and grid tasks.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            raise InterruptedError("Workflow aborted by user.")

    def _update_fluorescence_pose(self) -> None:
        """Refresh the lamella's recorded fluorescence pose from the current microscope state.

        The configured objective (focus) position is preserved: get_microscope_state()
        does not capture the objective position, so overwriting the pose here (and
        re-reading the live objective position) previously wiped the user's configured
        focus.
        """
        configured_objective_position = (
            self.lamella.fluorescence_pose.objective_position
            if self.lamella.fluorescence_pose is not None
            else None
        )
        self.lamella.fluorescence_pose = self.microscope.get_microscope_state()
        self.lamella.fluorescence_pose.objective_position = configured_objective_position

    def update_milling_config_ui(self,
                                 milling_config: FibsemMillingTaskConfig,
                                 msg: str = "Run Milling",
                                 milling_enabled: bool = True) -> FibsemMillingTaskConfig:
        """Update the milling config in the milling widget, and optionally run the milling task."""
        # headless mode
        if self.parent_ui is None:
            if milling_enabled:
                milling_task = run_milling_task(self.microscope, milling_config, None)
                return milling_task.config
            return milling_config

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
            # BlockingQueuedConnection guarantees run_milling() has returned and
            # _milling_thread.start() has been called before emit() unblocks.
            # No need to poll for is_milling to become True — it already is (or
            # milling finished before we got here, in which case the loop below
            # exits immediately, which is correct).
            self.parent_ui.milling_task_config_widget.milling_widget.start_milling_signal.emit()

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
                                        use_autocontrast=True,
                                        steps=MAX_ALIGNMENT_ATTEMPTS,
                                        stop_event=self._stop_event,
                                        run_name=f"{self.lamella.name} - {self.task_name}")

    def _run_autofocus(self, beam_type: BeamType, hfw: Optional[float] = None) -> None:
        """Run the image-based autofocus sweep, saving diagnostics to the lamella path.

        Args:
            beam_type: which beam to focus.
            hfw: field width for the probe images. Defaults to the task's configured
                imaging hfw. Pass the field of view the task actually images at, so the
                sweep is scored on the same view the task goes on to use.

        A user Stop raises OperationCancelledError out of run_auto_focus (which restores
        the starting working distance first); _is_cancellation treats that as a
        cancellation rather than a task failure, so it is deliberately not caught here.
        """
        from fibsem.autofunctions.autofocus import run_auto_focus, AutoFocusSettings, FocusSweepPass
        settings = AutoFocusSettings(
            method="tenengrad",
            passes=[
                FocusSweepPass(search_range=1e-3, step_size=100e-6),
                FocusSweepPass(search_range=100e-6, step_size=10e-6),
            ],
            reduced_area=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
            use_autocontrast=True)
        # NOTE: config.imaging, not self.image_settings -- only two subclasses define
        # that attribute, and only partway through their own _run().
        # `is None` rather than `or`, so an explicit hfw=0 is not silently replaced.
        if hfw is None:
            hfw = self.config.imaging.hfw
        self.log_status_message("AUTOFOCUS", f"Running autofocus ({beam_type.name})...")
        result = run_auto_focus(
            self.microscope,
            beam_type=beam_type,
            hfw=hfw,
            settings=settings,
            stop_event=self._stop_event,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save(path=os.path.join(self.lamella.path, "autofunctions"), name=f"{self.task_name}_autofocus_{ts}")
        self.log_status_message(
            "AUTOFOCUS",
            f"Autofocus (image-based): WD={result.working_distance*1e3:.3f}mm score={result.focus_score:.2f}",
            f"Autofocus complete: WD={result.working_distance*1e3:.3f}mm",
        )

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

    def _record_output(self, role: str, image: Optional[Union[FibsemImage, "FluorescenceImage"]]) -> None:
        """Record where an acquired image was written, under the given role.

        Stored relative to the lamella directory so the record survives the
        experiment being copied off the microscope. Images that were never
        written to disk carry no filepath and are skipped.
        """
        if image is None or image.filepath is None:
            return
        path = os.path.relpath(image.filepath, self.lamella.path)
        paths = self.lamella.task_state.outputs.setdefault(role, [])
        # a run can acquire the same set more than once -- MillCoincidentTask does,
        # before and after milling -- and the default filename means the second write
        # overwrites the first. The record describes files, not writes, so the same
        # path recorded twice is still one file.
        if path not in paths:
            paths.append(path)

    def _record_channel_outputs(self, phase: str, images: List[Tuple[Optional[FibsemImage], Optional[FibsemImage]]]) -> None:
        """Record a set of (sem, fib) pairs under `{phase}_sem` / `{phase}_fib`."""
        for sem_image, fib_image in images:
            self._record_output(f"{phase}_sem", sem_image)
            self._record_output(f"{phase}_fib", fib_image)

    def _acquire_channels(self,
                          image_settings: ImageSettings,
                          filename: Optional[str] = None,
                          field_of_view: float = 150e-6,
                          acquire_sem: bool = True,
                          acquire_fib: bool = True) -> None:
        """Acquire images for sem/fib channels at given field of view."""
        # only the default filename is the conventional start-of-task reference set.
        # tasks that pass their own name are recorded separately, so consumers asking
        # for the conventional sets don't silently pick up one-off acquisitions.
        phase = "start" if filename is None else "other"
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
        self._record_channel_outputs(phase, [(sem_image, fib_image)])
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
        # only the default filename is the conventional final reference set; see
        # _acquire_channels for why one-off acquisitions are recorded separately.
        phase = "final" if filename is None else "other"
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
        self._record_channel_outputs(phase, images)

        sem_image, fib_image = images[-1] # last acquired image
        if fib_image is not None:
            self._last_fib_image = fib_image
        set_images_ui(self.parent_ui, sem_image, fib_image)  # show the last acquired image

    def _retract_objective(self) -> None:
        """Retract the FM objective if it is inserted.

        Called at the end of tasks that insert the objective, so it is clear of the
        stage before any subsequent move or tilt. The task manager also retracts at
        the end of a run, but that is once for the whole queue -- without this the
        objective stays inserted across every task in between.

        Failures are logged, not raised: a task that has otherwise completed should
        not be reported as failed because the retract did not take.
        """
        if self.microscope.fm is None:
            return
        try:
            if self.microscope.fm.objective.state != "Inserted":
                return
            self.log_status_message("RETRACT_OBJECTIVE", "Retracting Objective...")
            self.microscope.fm.objective.retract()
        except Exception as e:
            logging.warning(f"Failed to retract the objective after {self.task_name}: {e}",
                            exc_info=True)

    def _move_to_milling_pose(self) -> None:
        """Move to the lamella milling pose."""
        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        if self.lamella.milling_pose is None:
            raise ValueError(f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella.")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

    def _get_stage_position_for_orientation(
        self,
        stage_position: FibsemStagePosition,
        orientation: Optional[str],
    ) -> FibsemStagePosition:
        """Return target position for orientation, or stage_position unchanged if orientation is None."""
        if orientation is None:
            return stage_position
        return self.microscope.get_target_position(stage_position, orientation)

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

    def set_fluorescence_channels_ui(self, channel_settings: List[ChannelSettings]) -> None:
        """Set the fluorescence channel settings in the fluorescence widget."""
        if self.parent_ui is None:
            return

        info = {
            "msg": "Updating Fluorescence Channel Settings",
            "fluorescence_channel_settings": deepcopy(channel_settings),
        }

        self.parent_ui.WAITING_FOR_UI_UPDATE = True
        self.parent_ui.workflow_update_signal.emit(info) # type: ignore
        while self.parent_ui.WAITING_FOR_UI_UPDATE:
            time.sleep(0.5)

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
