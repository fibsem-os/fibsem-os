from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    ClassVar,
    Type,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
)

@dataclass
class SelectFluorescencePositionConfig(AutoLamellaTaskConfig):
    """Configuration for the SelectFluorescencePositionTask."""
    task_type: ClassVar[str] = "SELECT_FLUORESCENCE_POSITION"
    display_name: ClassVar[str] = "Select Fluorescence Position"


class SelectFluorescencePositionTask(AutoLamellaTask):
    """Task to select fluorescence stage position and objective position."""
    config: SelectFluorescencePositionConfig
    config_cls: ClassVar[Type[SelectFluorescencePositionConfig]] = SelectFluorescencePositionConfig

    def _run(self) -> None:
        """Run the task to select fluorescence stage position and objective position."""

        if self.microscope.fm is None:
            raise ValueError("Fluorescence microscope not initialized in the FibsemMicroscope instance")
        if self.lamella.objective_position is None:
            raise ValueError(f"Objective position for {self.lamella.name} is not set. Please set the objective position before acquiring fluorescence images.")
        if self.lamella.stage_position is None:
            raise ValueError(f"Stage position for {self.lamella.name} is not set. Please set the stage position before acquiring fluorescence images.")

        if self.lamella.fluorescence_pose is not None and self.lamella.fluorescence_pose.stage_position is not None:
            stage_position = self.lamella.fluorescence_pose.stage_position
        else:
            stage_position = self.lamella.stage_position

        if not self.microscope.fm.has_valid_orientation(stage_position):
            raise ValueError(f"Stage Position {self.lamella.name} is not in a valid orientation: {stage_position.pretty}, {self.microscope.fm.valid_orientations}")

        # Check for cancellation before each position
        if self._stop_event and self._stop_event.is_set():
            logging.info(f"{self.task_name}: {self.lamella.name} - Acquisition cancelled")
            return

        # Move stage to the saved stage position and objective position
        self.log_status_message("MOVE_TO_POSITION", "Moving to Position...")
        self.microscope.fm.acquisition_progress_signal.emit({"state": "moving", "task": f"{self.task_name}"})
        self.microscope.safe_absolute_stage_movement(stage_position)

        # move objective to saved position
        self.log_status_message("MOVE_OBJECTIVE", "Moving Objective to position...")
        if not self.microscope.fm.objective.state == "Inserted":
            logging.info("Objective is not inserted. Inserting the objective before acquiring fluorescence images.")
            self.microscope.fm.objective.insert()
        self.microscope.fm.objective.move_absolute(self.lamella.objective_position)

        # Acquire image
        self.log_status_message("ACQUIRE_FLUORESCENCE_IMAGE", "Acquiring Fluorescence Image...")
        self.microscope.fm.acquisition_progress_signal.emit({"state": "acquiring", "task": f"{self.task_name}"})
        self.microscope.fm.acquire_image()

        if self.validate:
            ask_user(self.parent_ui,
                msg=f"Move to fluorescence position (stage and objective)for {self.lamella.name}. Press Continue to proceed.",
                pos="Continue"
                )
            self.microscope.fm.stop_acquisition()

        # store the fluorescence pose
        self.lamella.fluorescence_pose = self.microscope.get_microscope_state()
        self.lamella.objective_position = self.microscope.fm.objective.position

