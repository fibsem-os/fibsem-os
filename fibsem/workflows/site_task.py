"""A task at a site on the sample (a lamella, a volume position) and its
config: the task's milling and the reference images it takes, the run folder
it writes to, and how it moves, focuses and images there. Any app whose items
are site items can run these tasks."""

from __future__ import annotations

import logging
import os
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
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
    Union,
)

from fibsem import acquire, timing
from fibsem import config as fcfg
from fibsem.acting import OPERATOR
from fibsem.codec.upgrades import protocol as upgrade
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    ReferenceImageParameters,
    param,
)
from fibsem.util.timestamps import now, now_iso
from fibsem.workflows.attention import Attention
from fibsem.workflows.item import SiteItem
from fibsem.workflows.status import TaskStatus, set_images_ui, update_status_ui
from fibsem.workflows.task import BaseTask, run_folder
from fibsem.workflows.task_config import TaskConfig

if TYPE_CHECKING:
    from fibsem.autofunctions.autofocus import AutoFocusResult
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.microscope import FibsemMicroscope


@dataclass
class SiteTaskConfig(TaskConfig, ABC):
    """Configuration for a task at a site: the engine's, with the task's milling
    and the reference images it acquires when it finishes."""

    milling: Dict[str, FibsemMillingTaskConfig] = param(
        doc="The task's milling, by key (trench, rough, polishing, ...): each one's patterns, strategy and alignment.",
        default_factory=dict,
    )
    reference_imaging: ReferenceImageParameters = param(
        doc="The reference images the task acquires when it finishes.",
        default_factory=ReferenceImageParameters,
    )

    @property
    def parameters(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in super().parameters
            if name not in ("milling", "reference_imaging")
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteTaskConfig":
        return super().from_dict(upgrade.task_config(data))

    @property
    def estimated_time(self) -> float:
        """Estimate the total time for this task (milling + reference imaging)."""
        milling_time = sum(t.estimated_time for t in self.milling.values())
        imaging_time = self.reference_imaging.estimated_time
        return milling_time + imaging_time

    @property
    def opens_with_stage_move(self) -> bool:
        """Whether ``_run`` begins by moving the stage onto the task's pose.

        True for nearly every task, and the exceptions say so. Charged even when the
        stage is already in place -- a task following another on the same lamella pays
        0.02 s for the move rather than 7.6 s -- because which case applies is runtime
        state the config cannot see, and over-charging errs long.
        """
        return True

    @property
    def opens_with_reference_alignment(self) -> bool:
        """Whether ``_run`` aligns against the stored reference before its own work.

        Off by default: only some tasks do it, and a task that claims it when it does
        not adds 5 s of fiction to every estimate that includes it.
        """
        return False

    @property
    def estimated_duration(self) -> float:
        """Conservative forward estimate of this task's wall-clock, in seconds.

        Unlike :attr:`estimated_time`, which counts only the scan arithmetic and the
        milling, this adds the measured per-operation costs the task actually pays --
        see :mod:`fibsem.timing`. On the run it was calibrated against, that moved
        Setup Lamella Position from 0.22x of its real duration to 0.99x.

        A task whose cost is dominated by something this base cannot see -- spot burn
        exposures, fluorescence channels -- overrides this and adds its own term. The
        estimate lives on the config rather than on the task because the callers that
        need it (the pre-run dialog, the queue) hold configs and have no microscope to
        construct a task with.
        """
        # Reference images at both ends, but not the same number: a task opens with
        # _acquire_reference_image at one field of view and closes with
        # _acquire_set_of_reference_images over all of them.
        total = timing.reference_image_cost(self.reference_imaging, fovs=1)
        total += timing.reference_image_cost(self.reference_imaging)
        # what _run does before its own work
        if self.opens_with_stage_move:
            total += timing.stage_move_cost(1)
        if self.opens_with_reference_alignment:
            total += timing.REFERENCE_ALIGNMENT_S
        for milling_task in self.milling.values():
            total += timing.milling_task_cost(milling_task)
        return total

    @property
    def imaging(self) -> ImageSettings:
        """Get the imaging settings from the reference imaging parameters."""
        return self.reference_imaging.imaging

    @imaging.setter
    def imaging(self, value: ImageSettings):
        """Set the imaging settings in the reference imaging parameters."""
        self.reference_imaging.imaging = value


def get_task_supervision(task_name: str, experiment: Optional[Any] = None) -> bool:
    """Whether ``experiment``'s protocol asks for ``task_name`` to be supervised;
    False with no experiment or protocol."""
    protocol = getattr(experiment, "task_protocol", None)
    if protocol is None:
        return False
    return protocol.get_attention(task_name) is Attention.supervised


class SiteTask(BaseTask):
    """Base class for a task at a site: a lamella's, a volume position's."""

    config_cls: ClassVar[Type[SiteTaskConfig]]
    config: SiteTaskConfig

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: SiteTaskConfig,
        item: SiteItem,
        experiment: Optional[Any] = None,
        task_manager: Optional[Any] = None,
    ) -> None:
        super().__init__(
            microscope, config, item, experiment=experiment, task_manager=task_manager
        )
        # This task object is one run: its folder is fixed as it starts.
        self._output_dir = run_folder(
            self.path, config.task_name, now(), iteration=self.iteration
        )
        # Every task's reference images are acquired with its config's imaging
        # settings; they are saved under its output folder. Set here, once, for
        # the run this task object is, rather than by hand at the top of each
        # task's _run. In __init__ rather than pre_task, so a task
        # whose _run is called directly still writes there.
        self.config.imaging.path = self.output_dir
        # Set once only the closing reference images are left: the work is done,
        # so a Stop lets the task finish rather than record it Cancelled.
        self._closing = False

    @property
    def item_type(self) -> str:
        return self.item.item_type

    @property
    def output_dir(self) -> str:
        """Where this task's run writes what it makes: its reference images, its
        alignment and autofocus runs, its milling. Every save path a run hands on
        starts here, so where a task writes is decided once rather than at each
        call: the run's own folder, ``<item>/<task name>/
        <YYYYMMDD-HHMMSS>/``, so a re-run overwrites nothing. The
        item's shared state is written to the item's folder instead."""
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
            f"Running {self.task_name}, {self.task_type} ({self.task_id}) "
            f"for {self.item_type} {self.item.name} ({self.item.id})"
        )

        self._start_task_state()
        self._set_workflow_metadata()
        self.log_status_message(
            message="STARTED",
            display_message="Started",
            workflow_display_message=f"{self.item.name} [{self.display_name}]",
        )

    def post_task(self) -> None:
        if self.item.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        # Before the status, not after. Setting the status fires `task_state.events.status`,
        # which the item's card subscribes to; its handler is `@ensure_main_thread`, so the
        # read is queued to the GUI thread while this one carries straight on. Writing
        # the thumbnail afterwards meant the card read the file at the moment it was being
        # written -- `OSError: image file is truncated`, reported from the microscope --
        # and, even when it survived that, read the *previous* thumbnail.
        self.item.finishing_task(self.task_name, self.config, self._last_fib_image)
        self.item.task_state.status = TaskStatus.Completed
        self.item.task_state.status_message = ""
        self.log_status_message(message="FINISHED", display_message="Finished")
        self.log_task_config()
        self._record_outcome()

    def log_task_config(self) -> None:
        """Log the task configuration to the log file. This can be used for debugging or reporting."""
        logging.debug(
            {
                "msg": "task_config",
                "timestamp": now_iso(),
                self.item_type: self.item.name,
                f"{self.item_type}_id": self.item.id,
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
            msg=f"{self.item.name} [{self.task_name}] {message}",
            workflow_info=workflow_info,
            check_abort=not self._closing,
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
            # for this beam on this backend -- TESCAN ION, and recorded
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

        Stored relative to the item's directory so the record survives the
        experiment being copied off the microscope. Images that were never
        written to disk carry no filepath and are skipped.
        """
        if image is None or image.filepath is None:
            return
        path = os.path.relpath(image.filepath, self.path)
        paths = self.item.task_state.outputs.setdefault(role, [])
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

    def _image_after_pass(
        self, config: "FibsemMillingTaskConfig", passes: int, take: bool = True
    ) -> str:
        """The mill's own finished frame, else one FIB frame at the milling
        field of view when the proposal will be asked: the decision on a pass
        is made on what it milled."""
        image = super()._image_after_pass(config, passes, take)
        if image or not take:
            return image
        self._acquire_channels(
            deepcopy(self.config.imaging),
            filename=f"ref_{self.task_name}_pass_{passes}",
            field_of_view=config.field_of_view,
            acquire_sem=False,
            acquire_fib=True,
        )
        return self._last_fib_image_file()

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
        "other" and the History panel, which asks for "final", never saw it.
        """

        if field_of_views is None:
            field_of_views = (fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER)
        # only the default filename is the conventional final reference set; see
        # _acquire_channels for why one-off acquisitions are recorded separately.
        if phase is None:
            phase = "final" if filename is None else "other"
        if filename is None:
            filename = f"ref_{self.task_name}_final"
            self._closing = True

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
            self.task_manager, sem_image, fib_image, check_abort=not self._closing
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

    def _get_stage_position_for_orientation(
        self,
        stage_position: FibsemStagePosition,
        orientation: Optional[str],
    ) -> FibsemStagePosition:
        """Return target position for orientation, or stage_position unchanged if orientation is None."""
        if orientation is None:
            return stage_position
        return self.microscope.get_target_position(stage_position, orientation)
