######## TASK BASE DEFINITIONS ########


import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from fibsem import acquire, alignment, constants, utils
from fibsem import config as fcfg
from fibsem.acting import AGENT, OPERATOR
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
    Lamella,
)
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    set_images_ui,
    update_status_ui,
)
from fibsem.cancellation import SKIPPED
from fibsem.detection import detection as detection_module
from fibsem.detection.decided import (
    decided_features,
    detection_image_file,
    detection_provenance,
    detection_values,
    record_training_data,
)
from fibsem.detection.detection import (
    DetectedFeatures,
    Feature,
    LamellaBottomEdge,
    LamellaCentre,
    LamellaTopEdge,
    VolumeBlockCentre,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.utils import get_pattern_reduced_area
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    Point,
)
from fibsem.util.timestamps import now, now_iso
from fibsem.workflows.attention import Attention
from fibsem.workflows.interaction import (
    INSTRUCTION_TIMEOUT_S,
    SetFluorescenceChannels,
    ShowDetection,
    _abort_requested,
    ask,
)
from fibsem.workflows.proposals import (
    DETECTION,
    DecisionOutcome,
)
from fibsem.workflows.status import TaskState, TaskStatus
from fibsem.workflows.task import BaseTask, run_folder

if TYPE_CHECKING:
    from fibsem.alignment import AlignmentResult
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.autofunctions.autofocus import AutoFocusResult
    from fibsem.fm.structures import FluorescenceImage

TAutoLamellaTaskConfig = TypeVar(
    "TAutoLamellaTaskConfig", bound="AutoLamellaTaskConfig"
)

MAX_ALIGNMENT_ATTEMPTS = 3
ALIGNMENT_REFERENCE_IMAGE_FILENAME = "ref_alignment_ib.tif"


class AutoLamellaTask(BaseTask):
    """Base class for AutoLamella tasks: a task on a lamella."""

    config_cls: ClassVar[AutoLamellaTaskConfig]
    config: AutoLamellaTaskConfig
    item_type: ClassVar[str] = "lamella"

    def __init__(
        self,
        microscope: FibsemMicroscope,
        config: AutoLamellaTaskConfig,
        lamella: Lamella,
        task_manager: Optional["TaskManager"] = None,
    ):
        super().__init__(microscope, config, lamella, task_manager=task_manager)
        # This task object is one run: its folder is fixed as it starts (FIB-1275).
        self._output_dir = run_folder(
            lamella.path, config.task_name, now(), iteration=self.iteration
        )
        # Every task's reference images are acquired with its config's imaging
        # settings; they are saved under its output folder. Set here, once, for
        # the run this task object is, rather than by hand at the top of each
        # task's _run (FIB-1253). In __init__ rather than pre_task, so a task
        # whose _run is called directly still writes there.
        self.config.imaging.path = self.output_dir

    @property
    def lamella(self) -> Lamella:
        """The lamella this task runs on: its item."""
        return self.item

    @lamella.setter
    def lamella(self, lamella: Lamella) -> None:
        self.item = lamella

    @property
    def path(self) -> Path:
        """The lamella's folder, which it knows without an experiment."""
        return Path(self.lamella.path)

    @property
    def output_dir(self) -> str:
        """Where this task's run writes what it makes: its reference images, its
        alignment and autofocus runs, its milling. Every save path a run hands on
        starts here, so where a task writes is decided once rather than at each
        call (FIB-1253): the run's own folder, ``<lamella>/<task name>/
        <YYYYMMDD-HHMMSS>/`` (FIB-1275), so a re-run overwrites nothing. The
        lamella's shared state is written to the lamella's folder instead."""
        return self._output_dir

    @property
    def validate(self) -> bool:
        """Whether a person answers this task's questions as it runs: the
        protocol supervises it and the operator is attached (a window) to
        answer. The run's experiment says the first, the session the second."""
        manager = self.task_manager
        if manager is None or OPERATOR not in manager.deciders():
            return False
        return get_task_supervision(self.task_name, manager.experiment)

    @property
    def supervised(self) -> bool:
        """Whether the protocol's attention for this task is Supervised, read
        through the manager rather than the UI so a headless run behaves the
        same: then its questions are asked of whoever can answer."""
        manager = self.task_manager
        if manager is None:
            return False
        return get_task_supervision(self.task_name, manager.experiment)

    def pre_task(self) -> None:
        logging.info(
            f"Running {self.task_name}, {self.task_type} ({self.task_id}) for {self.lamella.name} ({self.lamella.id})"
        )

        self._start_task_state()
        self._set_workflow_metadata()
        self.log_status_message(
            message="STARTED",
            display_message="Started",
            workflow_display_message=f"{self.lamella.name} [{self.display_name}]",
        )

    def post_task(self) -> None:
        # post-task
        if self.lamella.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        # Before the status, not after. Setting the status fires `task_state.events.status`,
        # which the lamella card subscribes to; its handler is `@ensure_main_thread`, so the
        # read is queued to the GUI thread while this one carries straight on. Writing
        # afterwards meant the card read the file at the moment it was being written --
        # `OSError: image file is truncated`, reported from the microscope -- and, even
        # when it survived that, read the *previous* thumbnail, so the picture from the
        # task that had just finished was never the one shown.
        #
        # Guarded, because ordering it first would otherwise let a thumbnail failure stop
        # a finished task being marked Completed. A missing picture is worth far less than
        # an accurate task state.
        if self._last_fib_image is not None:
            try:
                self.lamella.save_thumbnail(self._last_fib_image)
            except Exception as e:
                logging.warning(
                    f"Could not save the thumbnail for {self.lamella.name}: {e}"
                )
        self.lamella.task_state.status = TaskStatus.Completed
        self.lamella.task_state.status_message = ""
        self.log_status_message(message="FINISHED", display_message="Finished")
        self.log_task_config()
        self.lamella.task_config[self.task_name] = deepcopy(self.config)
        self._record_outcome()

    def log_task_config(self) -> None:
        """Log the task configuration to the log file. This can be used for debugging or reporting."""
        logging.debug(
            {
                "msg": "task_config",
                "timestamp": now_iso(),
                "lamella": self.lamella.name,
                "lamella_id": self.lamella.id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "task_config": self.config.to_dict(),
                "supervised": self.validate,
            }
        )

    def log_status_message(
        self,
        message: str,
        display_message: Optional[str] = None,
        workflow_display_message: Optional[str] = None,
    ) -> None:
        super().log_status_message(message, display_message)
        if display_message is not None:
            self.update_status_ui(
                message=display_message, workflow_info=workflow_display_message
            )

    def update_status_ui(
        self, message: str, workflow_info: Optional[str] = None
    ) -> None:
        update_status_ui(
            run=self.task_manager,
            msg=f"{self.lamella.name} [{self.task_name}] {message}",
            workflow_info=workflow_info,
        )

    def detect(
        self,
        image_settings: ImageSettings,
        checkpoint: str,
        features: Sequence[Feature],
        *,
        position: Optional[FibsemStagePosition] = None,
        message: str = "",
        enabled: bool = True,
    ) -> DetectedFeatures:
        """Take an image, run the model on it, and ask about what it found.

        The detection is a ``detection`` question through ``ask``: recorded
        on the lamella against the image the model ran on, answered on the
        Detection tab (the features shown there first, a marker dragged,
        then Continue) or by an agent, and the decided points are put back
        on the detection with ``feature_m`` recomputed, so the stage moves by
        the corrected value. A rejection (an agent's) fails the task. The
        training data is written here, against the decided points.
        """
        message = message or self.lamella.status_info
        names = ", ".join(f.name for f in features)
        if len(names) > 15:
            names = names[:15] + "..."
        update_status_ui(
            self.task_manager, f"{message}: Detecting Features ({names})..."
        )
        detection = detection_module.take_image_and_detect_features(
            microscope=self.microscope,
            image_settings=image_settings,
            features=features,
            point=position,
            checkpoint=checkpoint,
        )
        if self.responder is not None and not self.unasked_reason(
            DETECTION, enabled=enabled
        ):
            # The Detection tab shows what the model found, as a copy: the
            # operator drags its markers, and the original is what the
            # correction is measured against.
            ask(
                self.responder,
                ShowDetection(detection=deepcopy(detection)),
                abort=lambda: _abort_requested(self.task_manager),
                timeout=INSTRUCTION_TIMEOUT_S,
            )
        decision = self.ask(
            DETECTION,
            detection_values(detection),
            image=detection_image_file(detection, str(self.lamella.path)),
            provenance=detection_provenance(detection),
            message=message,
            enabled=enabled,
        )
        if decision.outcome is DecisionOutcome.Rejected:
            raise RuntimeError(
                f"{self.task_name} was rejected: {decision.reason or 'no reason given'}"
            )
        answer = decided_features(detection, decision.values)
        record_training_data(detection, answer)
        return answer

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
        self.lamella.fluorescence_pose.objective_position = (
            configured_objective_position
        )

    def _align_reference_image(self, filename: str) -> Optional["AlignmentResult"]:
        """Align to a reference image in the lamella's folder, saving the run
        under the task's output folder. Returns what the alignment measured, or
        None when the reference image does not exist."""
        # beam_shift alignment
        self.log_status_message("ALIGN_REFERENCE_IMAGE", "Aligning Reference Images...")
        full_filename = os.path.join(self.lamella.path, filename)

        # validate reference image exists
        if not os.path.exists(full_filename):
            logging.warning(
                f"Reference image {full_filename} for alignment does not exist, "
                f"but was requested by {self.task_name}. Skipping alignment."
            )
            # Recorded, not only logged: the alignment the task asked for did
            # not happen, and the record should say so beside the ones that did.
            self.microscope.record_event(
                "alignment",
                {
                    "status": SKIPPED,
                    "reason": f"reference image {filename} does not exist",
                    "reference": full_filename,
                },
            )
            return None

        # load reference image, align
        ref_image = FibsemImage.load(full_filename)
        return alignment.multi_step_alignment_v2(
            microscope=self.microscope,
            ref_image=ref_image,
            use_autocontrast=True,
            steps=MAX_ALIGNMENT_ATTEMPTS,
            stop_event=self._stop_event,
            run_name=f"{self.lamella.name} - {self.task_name}",
            path=os.path.join(self.output_dir, alignment.ALIGNMENT_SUBDIR),
        )

    def _run_autofocus(
        self, beam_type: BeamType, hfw: Optional[float] = None
    ) -> Optional["AutoFocusResult"]:
        """Run the image-based autofocus sweep, saving diagnostics under the task's
        output folder. Returns what it measured, or None when the sweep was skipped.

        Args:
            beam_type: which beam to focus.
            hfw: field width for the probe images. Defaults to the task's configured
                imaging hfw. Pass the field of view the task actually images at, so the
                sweep is scored on the same view the task goes on to use.

        A user Stop raises OperationCancelledError out of run_auto_focus (which restores
        the starting working distance first); _is_cancellation treats that as a
        cancellation rather than a task failure, so it is deliberately not caught here.
        """
        from fibsem.autofunctions.autofocus import (
            AutoFocusSettings,
            FocusSweepPass,
            run_auto_focus,
        )

        settings = AutoFocusSettings(
            method="tenengrad",
            passes=[
                FocusSweepPass(search_range=1e-3, step_size=100e-6),
                FocusSweepPass(search_range=100e-6, step_size=10e-6),
            ],
            reduced_area=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
            use_autocontrast=True,
        )
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
            path=os.path.join(self.output_dir, "autofunctions"),
            name=f"{self.task_name}_autofocus_{datetime.now():%Y%m%d_%H%M%S}",
        )
        if result is None:
            # run_auto_focus declined the sweep (the working distance is not settable
            # for this beam on this backend -- TESCAN ION, see FIB-508), and recorded
            # it skipped. Nothing is saved: a placeholder result would write a
            # completed-looking artifact directory and log a working distance that
            # was never applied.
            self.log_status_message(
                "AUTOFOCUS",
                f"Autofocus skipped: the {beam_type.name} working distance is not settable on this system.",
                f"Autofocus skipped ({beam_type.name})",
            )
            return None
        self.log_status_message(
            "AUTOFOCUS",
            f"Autofocus (image-based): WD={result.working_distance * 1e3:.3f}mm score={result.focus_score:.2f}",
            f"Autofocus complete: WD={result.working_distance * 1e3:.3f}mm",
        )
        return result

    def _acquire_reference_image(
        self,
        image_settings: ImageSettings,
        filename: Optional[str] = None,
        field_of_view: float = 150e-6,
    ) -> None:
        """Acquire a reference image with given field of view."""
        acquire_fib = self.config.reference_imaging.acquire_fib
        acquire_sem = self.config.reference_imaging.acquire_sem
        return self._acquire_channels(
            image_settings,
            field_of_view=field_of_view,
            filename=filename,
            acquire_sem=acquire_sem,
            acquire_fib=acquire_fib,
        )

    def _acquire_set_of_reference_images(
        self,
        image_settings: ImageSettings,
        filename: Optional[str] = None,
        field_of_views: Optional[Tuple[float, ...]] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Acquire a set of reference images.

        `phase` names the role the images are recorded under; see
        :meth:`_acquire_set_of_channels` for when to pass one.
        """
        acquire_fib = self.config.reference_imaging.acquire_fib
        acquire_sem = self.config.reference_imaging.acquire_sem
        if field_of_views is None:
            field_of_views = self.config.reference_imaging.field_of_views
        image_settings = self.config.reference_imaging.imaging
        return self._acquire_set_of_channels(
            image_settings,
            field_of_views=field_of_views,
            filename=filename,
            acquire_sem=acquire_sem,
            acquire_fib=acquire_fib,
            phase=phase,
        )

    def _record_output(
        self, role: str, image: Optional[Union[FibsemImage, "FluorescenceImage"]]
    ) -> None:
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

    def _record_channel_outputs(
        self,
        phase: str,
        images: List[Tuple[Optional[FibsemImage], Optional[FibsemImage]]],
    ) -> None:
        """Record a set of (sem, fib) pairs under `{phase}_sem` / `{phase}_fib`."""
        for sem_image, fib_image in images:
            self._record_output(f"{phase}_sem", sem_image)
            self._record_output(f"{phase}_fib", fib_image)

    def _acquire_channels(
        self,
        image_settings: ImageSettings,
        filename: Optional[str] = None,
        field_of_view: float = 150e-6,
        acquire_sem: bool = True,
        acquire_fib: bool = True,
    ) -> None:
        """Acquire images for sem/fib channels at given field of view."""
        # only the default filename is the conventional start-of-task reference set.
        # tasks that pass their own name are recorded separately, so consumers asking
        # for the conventional sets don't silently pick up one-off acquisitions.
        phase = "start" if filename is None else "other"
        if filename is None:
            filename = f"ref_{self.task_name}_start"

        self.log_status_message(
            "ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images..."
        )
        image_settings.hfw = field_of_view
        image_settings.filename = filename
        image_settings.save = True
        sem_image, fib_image = acquire.acquire_channels(
            self.microscope,
            image_settings,
            acquire_sem=acquire_sem,
            acquire_fib=acquire_fib,
        )
        self._record_channel_outputs(phase, [(sem_image, fib_image)])
        if fib_image is not None:
            self._last_fib_image = fib_image
        set_images_ui(self.task_manager, sem_image, fib_image)

    def _acquire_set_of_channels(
        self,
        image_settings: ImageSettings,
        field_of_views: Optional[Tuple[float, ...]] = None,
        filename: Optional[str] = None,
        acquire_sem: bool = True,
        acquire_fib: bool = True,
        phase: Optional[str] = None,
    ) -> None:
        """Acquire a set of images for each sem/fib channel at given field of views.

        `phase` is the role the images are recorded under, and defaults to reading it
        off the filename. Pass it when a task names its own files but the set *is* the
        task's reference set rather than an extra one -- AcquireReferenceImageTask
        timestamps its filenames, so the default rule filed its only product under
        "other" and the History panel, which asks for "final", never saw it (FIB-579).
        """

        if field_of_views is None:
            field_of_views = (fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER)
        # only the default filename is the conventional final reference set; see
        # _acquire_channels for why one-off acquisitions are recorded separately.
        if phase is None:
            phase = "final" if filename is None else "other"
        if filename is None:
            filename = f"ref_{self.task_name}_final"

        self.log_status_message(
            "ACQUIRE_REFERENCE_IMAGES", "Acquiring Reference Images..."
        )
        images = acquire.acquire_set_of_channels(
            self.microscope,
            image_settings,
            field_of_views,
            filename=filename,
            acquire_sem=acquire_sem,
            acquire_fib=acquire_fib,
        )
        self._record_channel_outputs(phase, images)

        sem_image, fib_image = images[-1]  # last acquired image
        if fib_image is not None:
            self._last_fib_image = fib_image
        set_images_ui(
            self.task_manager, sem_image, fib_image
        )  # show the last acquired image

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
            if self.microscope.fm.objective.state is not InsertableDeviceState.INSERTED:
                return
            self.log_status_message("RETRACT_OBJECTIVE", "Retracting Objective...")
            self.microscope.fm.objective.retract()
        except Exception as e:
            logging.warning(
                f"Failed to retract the objective after {self.task_name}: {e}",
                exc_info=True,
            )

    def _move_to_milling_pose(self) -> None:
        """Move to the lamella milling pose."""
        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        if self.lamella.milling_pose is None:
            raise ValueError(
                f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella."
            )
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

    def set_fluorescence_channels_ui(
        self, channel_settings: List[ChannelSettings]
    ) -> None:
        """Set the fluorescence channel settings in the fluorescence widget."""
        if self.responder is None:
            return

        ask(
            self.responder,
            SetFluorescenceChannels(channels=deepcopy(channel_settings)),
            abort=lambda: _abort_requested(self.task_manager),
            timeout=INSTRUCTION_TIMEOUT_S,
        )


def get_task_supervision(task_name: str, experiment: Optional[Any] = None) -> bool:
    """Whether ``experiment``'s protocol asks for ``task_name`` to be supervised;
    False with no experiment or protocol."""
    protocol = getattr(experiment, "task_protocol", None)
    if protocol is None:
        return False
    return protocol.get_attention(task_name) is Attention.supervised
