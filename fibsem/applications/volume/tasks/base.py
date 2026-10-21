"""The volume task lifecycle: a task on one position, for one slice."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Type

from fibsem import acquire
from fibsem.alignment import ALIGNMENT_SUBDIR, multi_step_alignment_v2
from fibsem.applications.volume.position import VolumePosition
from fibsem.structures import BeamType, FibsemImage, ImageSettings
from fibsem.util.timestamps import now
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import STATE
from fibsem.workflows.task import BaseTask, run_folder
from fibsem.workflows.task_config import TaskConfig

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.microscope import FibsemMicroscope

# The output role a task's reference images are recorded under.
REFERENCE_ROLE = "reference"


class VolumeTask(BaseTask):
    """Base class for volume tasks: a task on a position. Subclasses implement
    `_run`; ``iteration`` is the slice it works on."""

    config_cls: ClassVar[Type[TaskConfig]]
    config: TaskConfig
    item_type: ClassVar[str] = "position"
    # False for a task run once on a position, before its slices (its setup).
    per_slice: ClassVar[bool] = True

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: TaskConfig,
        position: VolumePosition,
        experiment: "VolumeExperiment",
        task_manager: Optional[Any] = None,
    ) -> None:
        super().__init__(
            microscope,
            config,
            position,
            experiment=experiment,
            task_manager=task_manager,
        )
        self._output_dir: Optional[str] = None

    @property
    def position(self) -> VolumePosition:
        """The position this task runs on: its item."""
        return self.item

    @property
    def supervised(self) -> bool:
        return self.config.attention is Attention.supervised

    @property
    def slice(self) -> int:
        """The slice this run works on, from 0."""
        if self.iteration is None:
            raise ValueError(
                f"{self.task_name} runs once per slice: it needs an iteration."
            )
        return self.iteration

    @property
    def output_dir(self) -> str:
        """This run's folder, ``<position>/<task name>/<slice:04d>/``, created on
        first use."""
        if self._output_dir is None:
            self._output_dir = run_folder(
                self.path, self.task_name, now(), iteration=self.iteration
            )
        os.makedirs(self._output_dir, exist_ok=True)
        return self._output_dir

    def _acquire_reference(
        self, beam_type: BeamType, filename: str, field_of_view: float
    ) -> FibsemImage:
        """Save a reference image of the position into this run's folder."""
        settings = ImageSettings(
            beam_type=beam_type,
            hfw=field_of_view,
            path=self.output_dir,
            filename=filename,
            save=True,
            autocontrast=True,
        )
        image = acquire.acquire_image(self.microscope, settings)
        if beam_type is BeamType.ION:
            self._last_fib_image = image
        self.record_output(REFERENCE_ROLE, image)
        return image

    def _confirm_position(self, message: str) -> None:
        """Ask the position to be confirmed as it stands, or moved first."""
        self.ask(
            STATE,
            {"stage_position": self.microscope.get_stage_position()},
            message=message,
            decided=lambda: {"stage_position": self.microscope.get_stage_position()},
        )

    def _align_to_reference(self, filename: str) -> None:
        """Align to the position's reference ``filename``. A position without
        one has not been set up, and a slice cut unaligned is worse than none,
        so the task fails."""
        reference = os.path.join(self.path, filename)
        if not os.path.exists(reference):
            raise RuntimeError(
                f"{self.position.name} has no alignment reference {filename}; "
                "set the position up first."
            )
        self.log_status_message("ALIGN_REFERENCE_IMAGE", "Aligning to the reference.")
        multi_step_alignment_v2(
            microscope=self.microscope,
            ref_image=FibsemImage.load(reference),
            stop_event=self._stop_event,
            run_name=f"{self.position.name} - {self.task_name}",
            path=os.path.join(self.output_dir, ALIGNMENT_SUBDIR),
        )
        self._check_for_abort()
