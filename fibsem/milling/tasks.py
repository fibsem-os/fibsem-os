from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fibsem import acquire
from fibsem import config as fcfg
from fibsem.cancellation import OperationCancelledError, raise_if_cancelled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.structures import (
    BeamType,
    FibsemImage,
    ImageSettings,
    MillingAlignment,
    Point,
)
from fibsem.utils import current_timestamp_v3

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2


@dataclass
class MillingTaskAcquisitionSettings:
    """Settings for the acquisition of images during a milling task."""

    acquire_sem: bool = field(
        default=False,
        metadata={
            "label": "Acquire SEM Image",
            "tooltip": "Whether to acquire SEM images between the milling task stages.",
        },
    )
    acquire_fib: bool = field(
        default=False,
        metadata={
            "label": "Acquire FIB Image",
            "tooltip": "Whether to acquire FIB images between the milling task stages.",
        },
    )
    acquire_final_image: bool = field(
        default=True,
        metadata={
            "label": "Acquire Final Image",
            "tooltip": "Refresh the FIB view with a single image once the task finishes. "
            "Disable for low-kV polishing, where imaging the lamella with a "
            "higher-voltage beam undoes the polish.",
        },
    )
    imaging: ImageSettings = field(default_factory=ImageSettings)

    @property
    def enabled(self) -> bool:
        return self.acquire_sem or self.acquire_fib

    @property
    def estimated_time(self) -> float:
        if not self.enabled:
            return 0.0
        n = sum([self.acquire_sem, self.acquire_fib])
        return self.imaging.estimated_time * n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acquire_sem": self.acquire_sem,
            "acquire_fib": self.acquire_fib,
            "acquire_final_image": self.acquire_final_image,
            "imaging": self.imaging.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MillingTaskAcquisitionSettings":
        imaging = data.get("imaging", {})
        if imaging == {} or imaging.get("path", None) is None:
            imaging["path"] = None
        return cls(
            acquire_sem=data.get("acquire_sem", False),
            acquire_fib=data.get("acquire_fib", False),
            # default True so protocols written before this flag existed keep the
            # post-task FIB refresh they have always had
            acquire_final_image=data.get("acquire_final_image", True),
            imaging=ImageSettings.from_dict(imaging),
        )


@dataclass
class FibsemMillingTaskConfig:
    """Configuration for a milling task."""

    name: str = "Milling Task"
    field_of_view: float = 150e-6
    channel: BeamType = BeamType.ION
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    acquisition: MillingTaskAcquisitionSettings = field(
        default_factory=MillingTaskAcquisitionSettings
    )
    stages: List[FibsemMillingStage] = field(default_factory=list)

    @property
    def enabled_stages(self) -> List[FibsemMillingStage]:
        """Return only stages where enabled is True."""
        return [s for s in self.stages if s.enabled]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field_of_view": self.field_of_view,
            "channel": self.channel.name,
            "alignment": self.alignment.to_dict(),
            "acquisition": self.acquisition.to_dict(),
            "stages": [stage.to_dict(short=True) for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FibsemMillingTaskConfig":
        alignment = data.get("alignment", {})
        acquisition = data.get("acquisition", {})
        return cls(
            name=data.get("name", "Milling Task"),
            field_of_view=data.get("field_of_view", 150e-6),
            channel=BeamType[data.get("channel", BeamType.ION.name)],
            alignment=MillingAlignment.from_dict(alignment),
            acquisition=MillingTaskAcquisitionSettings.from_dict(acquisition),
            stages=[
                FibsemMillingStage.from_dict(stage) for stage in data.get("stages", [])
            ],
        )

    @classmethod
    def from_stages(
        cls, stages: List[FibsemMillingStage], name: str = "Milling Task"
    ) -> "FibsemMillingTaskConfig":
        """Create a FibsemMillingTaskConfig from a list of FibsemMillingStage."""

        if not stages:
            raise ValueError("No milling stages provided to create task config.")

        # Use the first stage's properties as defaults
        first_stage = stages[0]
        return FibsemMillingTaskConfig(
            name=name,
            field_of_view=first_stage.milling.hfw,
            channel=first_stage.milling.milling_channel,
            alignment=first_stage.alignment,
            acquisition=MillingTaskAcquisitionSettings(
                acquire_sem=first_stage.milling.acquire_images,
                acquire_fib=first_stage.milling.acquire_images,
                imaging=first_stage.imaging,
            ),
            stages=stages,
        )

    @property
    def estimated_time(self) -> float:
        """Estimate the total milling time for a list of milling stages.

        Over the enabled stages only, matching what run() will actually mill.
        Summing every stage counted work that was switched off: a real task with
        two of its three stages disabled estimated 845 s against ~102 s milled.
        """
        milling_time = sum(stage.estimated_time for stage in self.enabled_stages)
        return milling_time + self.acquisition.estimated_time

    def compatible_stages(
        self, reference_idx: int = 0
    ) -> List[Tuple[int, FibsemMillingStage]]:
        """Return stages whose milling settings & strategy match the stage at reference_idx."""
        if not self.stages:
            return []

        if reference_idx < 0 or reference_idx >= len(self.stages):
            raise IndexError(
                f"reference_idx {reference_idx} out of range for {len(self.stages)} stages."
            )

        reference_stage = self.stages[reference_idx]
        compatible: List[Tuple[int, FibsemMillingStage]] = []

        for idx, stage in enumerate(self.stages):
            if idx != reference_idx and stage.is_compatible_with(reference_stage):
                compatible.append((idx, stage))

        return compatible

    def merge_compatible_stages(self, reference_idx: int = 0) -> "FibsemMillingStage":
        compat_stages = self.compatible_stages(reference_idx=reference_idx)
        logging.info(
            f"Compatible Stages: {[(idx, stage.name) for idx, stage in compat_stages]}"
        )

        reference_stage = self.stages[reference_idx]
        if compat_stages:
            reference_stage.patterns = [reference_stage.pattern]
            for idx, stage in compat_stages:
                reference_stage.patterns.append(self.stages[idx].pattern)

        return reference_stage


# TODO: remove parent_ui arg, use microscope signal only, and stop_event -> need to migrate
# TODO: restore current to initial current, rather than system.ion.beam.beam_current
# TODO: support customising alignment imaging parameters?
@dataclass
class FibsemMillingTask:
    config: FibsemMillingTaskConfig = field(default_factory=FibsemMillingTaskConfig)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reference_image: Optional[FibsemImage] = None

    def __init__(
        self,
        microscope: FibsemMicroscope,
        config: FibsemMillingTaskConfig,
        parent_ui: Optional["FibsemMillingWidget2"] = None,
    ):
        self.config = config
        self.microscope = microscope
        self.parent_ui = parent_ui
        self.task_id = str(uuid.uuid4())
        self.initial_beam_shift: Optional[Point] = None
        # Imaging current/voltage captured before milling starts, so cleanup restores the
        # exact pre-milling state (not a config default) even if the task is cancelled.
        self.initial_imaging_current: Optional[float] = None
        self.initial_imaging_voltage: Optional[float] = None
        self._stop_event: Optional[threading.Event] = None
        if self.parent_ui and hasattr(self.parent_ui, "_milling_stop_event"):
            self._stop_event = self.parent_ui._milling_stop_event

    @property
    def name(self) -> str:
        """Return the name of the milling task."""
        return self.config.name

    @property
    def stages(self) -> List[FibsemMillingStage]:
        """Return the list of milling stages."""
        return self.config.enabled_stages

    def _emit(self, status: MillingProgressStatus, **fields) -> None:
        """Report where this task has got to.

        Replaces `_handle_progress`, which was a pass-through: it emitted whatever it
        was handed, handled nothing, and described the traffic backwards -- the updates
        go *to* the microscope's signal rather than from it. Residue of an unfinished
        migration.

        Its one real effect was hiding two emit sites from any scan keyed on the signal
        name, which is a blind spot rather than a feature. This version earns the
        indirection instead: the task-identity stamp lives here rather than being
        repeated at every call site, and the guard in
        tests/milling/test_progress_producers.py knows to follow it.
        """
        self.microscope.milling_progress_signal.emit(
            MillingProgress(
                status=status, task_id=self.task_id, task_name=self.name, **fields
            )
        )

    def _imaging_conditions(self) -> Tuple[float, float]:
        """The (current, voltage) the user was imaging at before milling started.

        Falls back to the system defaults only if the task failed before it could
        capture them. Every restore path must go through here: reading
        ``system.ion.beam.*`` directly picks up the *config* default, which bounces
        the column back to 30 kV whenever a stage mills at some other voltage.
        """
        current = self.initial_imaging_current
        voltage = self.initial_imaging_voltage
        return (
            current
            if current is not None
            else self.microscope.system.ion.beam.beam_current,
            voltage if voltage is not None else self.microscope.system.ion.beam.voltage,
        )

    def _configure_path(self) -> None:
        """Configure the acquisition path for the milling task.

        Everything the task saves hangs off this path: the alignment reference image,
        the drift-correction run, and the per-stage images. The path is unset by
        default — a config built from stages, or loaded from a protocol that doesn't
        name one, carries ``None`` — so fall back to the cross-correlation data
        directory, as every other acquisition in this path already does. Stringifying
        the ``None`` instead put the whole task under a directory literally named
        "None", beside the current working directory.
        """
        path = self.config.acquisition.imaging.path
        if path is None:
            path = fcfg.DATA_CC_PATH
        self.config.acquisition.imaging.path = os.path.join(
            str(path),
            "Milling",
            self.name.replace(" ", "-"),
        )

    def run(self) -> None:
        """Run a list of milling stages, with a progress bar and notifications."""

        logging.info(f"Running milling task: {self.name} with ID: {self.task_id}")

        # Read by the `finally` below, which runs whichever way the task ends. Seeded
        # with the failure case because `except Exception` does not catch everything:
        # a KeyboardInterrupt unwinds straight through to the `finally`, and reporting
        # that as a completed mill is the defect this whole block exists to fix.
        outcome = MillingProgressStatus.TASK_FAILED
        error: Optional[str] = None

        self._emit(MillingProgressStatus.TASK_STARTED, total_stages=len(self.stages))

        try:
            self.initial_beam_shift = self.microscope.get_beam_shift(
                beam_type=self.config.channel
            )
            # Capture the live imaging current/voltage BEFORE setup_milling switches to the
            # milling current, so cleanup restores exactly what the user was imaging at.
            self.initial_imaging_current = self.microscope.get_beam_current(
                self.config.channel
            )
            self.initial_imaging_voltage = self.microscope.get_beam_voltage(
                self.config.channel
            )

            # configure acquisition filepaths
            self._configure_path()

            # acquire a reference image for alignment
            if self.config.alignment.enabled:
                self._acquire_reference_image()

            # acquire an image before starting the task
            if self.config.acquisition.enabled:
                self._acquire_milling_task_images(stage_name=self.name, tag="pre-task")

            for idx, stage in enumerate(self.stages):
                self._mill_stage(stage, idx)

            outcome = MillingProgressStatus.TASK_FINISHED
        except OperationCancelledError as e:
            logging.info(f"Milling task '{self.name}' cancelled by user: {e}")
            outcome = MillingProgressStatus.TASK_CANCELLED
        except Exception as e:
            logging.error(e)
            outcome = MillingProgressStatus.TASK_FAILED
            error = str(e)
        finally:
            # The outcome, not just "it stopped". Both `except` blocks used to log and
            # fall through to a `finally` that emitted `finished` regardless, so a mill
            # the user cancelled and a mill that crashed both told the UI they had
            # finished -- the status bar rendered "Done" either way and the exception
            # text reached only the logfile.
            self._emit(outcome, error=error)
            # restore the captured pre-milling imaging current/voltage
            imaging_current, imaging_voltage = self._imaging_conditions()
            self.microscope.finish_milling(
                imaging_current=imaging_current,
                imaging_voltage=imaging_voltage,
            )
            # restore initial beam shift
            if self.initial_beam_shift is not None:
                self.microscope.set_beam_shift(
                    self.initial_beam_shift, beam_type=self.config.channel
                )

            self._post_task_acquisition()

    def _post_task_acquisition(self) -> None:
        """Acquire an image after finishing the milling task."""
        try:
            # refresh the view with a single image if the task didn't already acquire one.
            # NB: acquisition.enabled is (acquire_sem or acquire_fib), so it subsumes the
            # acquire_fib check this condition used to carry.
            if (
                self.config.acquisition.acquire_final_image
                and not self.config.acquisition.enabled
            ):
                self.microscope.autocontrast(beam_type=self.config.channel)
                fib_image = self.microscope.acquire_image(
                    image_settings=None, beam_type=self.config.channel
                )
                self.microscope.fib_acquisition_signal.emit(fib_image)
        except Exception as e:
            logging.error(f"Error acquiring image after milling task: {e}")

    def _mill_stage(self, stage: FibsemMillingStage, idx: int) -> None:
        """Run a single milling stage with progress updates.
        Args:
            stage (FibsemMillingStage): The milling stage to run.
            idx (int): The index of the milling stage.
        """

        start_time = time.time()
        raise_if_cancelled(self._stop_event)

        self._emit(
            MillingProgressStatus.STAGE_STARTED,
            start_time=start_time,
            current_stage=idx,
            total_stages=len(self.stages),
            stage_name=stage.name,
        )

        try:
            # if self.config.acquisition.enabled:
            # self._acquire_milling_task_images(stage_name=stage.name, tag="start")

            # Set up the stage with the task configuration
            stage.reference_image = self.reference_image
            stage.milling.hfw = self.config.field_of_view
            stage.milling.milling_channel = self.config.channel
            stage.milling.acquire_images = self.config.acquisition.enabled
            stage.imaging.path = self.config.acquisition.imaging.path
            stage.imaging = self.config.acquisition.imaging
            stage.alignment = self.config.alignment
            stage.strategy.run(
                microscope=self.microscope,
                stage=stage,
                asynch=False,
                parent_ui=self.parent_ui,
                stop_event=self._stop_event,
            )
            # TODO: pass task as parent into strategy.run()?, allow logging from strategy?
            # performance logging
            msgd = {
                "msg": "milling_task",
                "milling_task_id": self.task_id,
                "milling_task_name": self.name,
                "idx": idx,
                "stage": stage.to_dict(),
                "start_time": start_time,
                "end_time": time.time(),
                "timestamp": datetime.now().isoformat(),
            }
            logging.debug(f"{msgd}")

            # optionally acquire images after milling
            if self.config.acquisition.enabled:
                self._acquire_milling_task_images(
                    stage_name=f"{self.name}-{stage.name}", tag="finished"
                )

            self._emit(
                MillingProgressStatus.STAGE_FINISHED,
                start_time=start_time,
                current_stage=idx,
                total_stages=len(self.stages),
                stage_name=stage.name,
            )

        except OperationCancelledError:
            raise  # unwind to run() so the whole task aborts + restores conditions
        except Exception as e:
            logging.error(
                f"Error running milling stage: {stage.name}, {e}", exc_info=True
            )

    def _acquire_reference_image(self) -> Optional[FibsemImage]:
        """Acquire a reference image for the milling task."""

        if self.reference_image is not None:
            return self.reference_image

        path = self.config.acquisition.imaging.path
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)

        filename = f"{self.name}_{fcfg.REFERENCE_FILENAME}_{current_timestamp_v3(timeonly=True)}".replace(
            " ", "-"
        )
        image_settings = ImageSettings(
            hfw=self.config.field_of_view,
            dwell_time=self.config.alignment.imaging.dwell_time,
            resolution=self.config.alignment.imaging.resolution,
            beam_type=self.config.channel,
            reduced_area=self.config.alignment.rect,
            save=True,
            path=path,
            filename=filename,
        )
        self.reference_image = acquire.acquire_image(
            microscope=self.microscope, settings=image_settings
        )

    def _acquire_milling_task_images(
        self,
        stage_name: str,
        tag: str = "finished",
    ) -> None:
        """Acquire images after milling for reference.
        Args:
            stage_name (str): Name of the milling stage
            tag (str): Tag to append to the filename
        """
        imaging_current, imaging_voltage = self._imaging_conditions()
        self.microscope.finish_milling(
            imaging_current=imaging_current, imaging_voltage=imaging_voltage
        )

        acq_date = current_timestamp_v3(timeonly=True)
        self.config.acquisition.imaging.filename = (
            f"{stage_name}_{tag}_{acq_date}".replace(" ", "-")
        )
        self.config.acquisition.imaging.save = True
        self.config.acquisition.imaging.hfw = (
            self.config.field_of_view
        )  # force field of view to match task

        if self.config.acquisition.imaging.path is None:
            self.config.acquisition.imaging.path = (
                self.microscope._last_imaging_settings.path
            )

        # support specifying acquiring sem and/or fib images only
        sem_image, fib_image = acquire.acquire_channels(
            microscope=self.microscope,
            image_settings=self.config.acquisition.imaging,
            acquire_sem=self.config.acquisition.acquire_sem,
            acquire_fib=self.config.acquisition.acquire_fib,
        )

        try:
            if sem_image is not None:
                self.microscope.sem_acquisition_signal.emit(sem_image)  # sem image
            if fib_image is not None:
                self.microscope.fib_acquisition_signal.emit(fib_image)  # ion image
        except Exception as e:
            logging.error(f"Error emitting acquisition signals: {e}")


def run_milling_task(
    microscope: FibsemMicroscope,
    config: FibsemMillingTaskConfig,
    parent_ui: Optional["FibsemMillingWidget2"] = None,
) -> FibsemMillingTask:
    """Run a milling task with the given configuration.
    Args:
        microscope (FibsemMicroscope): The microscope to use for milling.
        config (FibsemMillingTaskConfig): The configuration for the milling task.
        parent_ui (Optional[FibsemMillingWidget2]): The parent UI widget for progress updates.
    Returns:
        FibsemMillingTask: The milling task that was run.
    """
    task = FibsemMillingTask(microscope=microscope, config=config, parent_ui=parent_ui)
    task.run()
    return task
