from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Type,
)

import fibsem.utils as utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
)
from fibsem.fm.acquisition import acquire_image, run_autofocus
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, ZParameters


@dataclass
class AcquireFluorescenceImageConfig(AutoLamellaTaskConfig):
    """Configuration for the AcquireFluorescenceImageTask."""
    task_type: ClassVar[str] = "ACQUIRE_FLUORESCENCE_IMAGE"
    display_name: ClassVar[str] = "Acquire Fluorescence Image"
    channel_settings: list[ChannelSettings] = field(default_factory=list,
                                                    metadata={"help": "Settings for each fluorescence channel",
                                                             "label": "Channel Settings"})
    zparams: ZParameters = field(default_factory=ZParameters, 
                                  metadata={"help": "Z-stack acquisition parameters",
                                            "label": "Z-Stack Parameters"})
    autofocus_settings: AutoFocusSettings = field(default_factory=AutoFocusSettings,
                                                  metadata={"help": "Settings for autofocus before acquiring fluorescence images",
                                                            "label": "Autofocus Settings"})

    def to_dict(self) -> dict:
        ddict = {}
        ddict["task_type"] = self.task_type
        ddict["channel_settings"] = [cs.to_dict() for cs in self.channel_settings]
        ddict["autofocus_settings"] = self.autofocus_settings.to_dict()
        ddict["zparams"] = self.zparams.to_dict()
        return ddict

    @classmethod
    def from_dict(cls, data: dict) -> 'AcquireFluorescenceImageConfig':
        channel_settings = [ChannelSettings.from_dict(cs) for cs in data.get("channel_settings", [])]
        autofocus_settings = AutoFocusSettings.from_dict(data.get("autofocus_settings", {}))
        zparams = ZParameters.from_dict(data.get("zparams", {}))
        return cls(
            task_name=data.get("task_name", ""),
            milling=data.get("milling", {}) ,
            channel_settings=channel_settings,
            autofocus_settings=autofocus_settings,
            zparams=zparams
        )

# TODO: implement time estimates...
class AcquireFluorescenceImageTask(AutoLamellaTask):
    """Task to acquire fluorescence image with specified settings."""
    config: AcquireFluorescenceImageConfig
    config_cls: ClassVar[Type[AcquireFluorescenceImageConfig]] = AcquireFluorescenceImageConfig

    def _run(self) -> None:
        """Run the task to acquire fluorescence image with the specified settings."""


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

        if self.microscope.get_stage_orientation(stage_position) not in self.microscope.fm.valid_orientations:
            logging.warning(f"Stage Position {self.lamella.name} is not in a valid orientation: {stage_position}, moving to SEM orientation...")
            stage_position = self.microscope.get_target_position(stage_position=stage_position, target_orientation="SEM")

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
            logging.warning("Objective is not inserted. Inserting the objective before acquiring fluorescence images.")
            self.microscope.fm.objective.insert()
        self.microscope.fm.objective.move_absolute(self.lamella.objective_position)

        # Run autofocus if requested
        if self.config.autofocus_settings.fine_enabled:
            self._run_autofocus()
            # TODO: store the focused objective position?

        # Generate timestamp-based filename
        timestamp = utils.current_timestamp_v3(timeonly=True)
        basename = f"{self.lamella.name}-zstack-{timestamp}.ome.tiff"
        filename = os.path.join(self.lamella.path, basename)

        # Acquire image
        self.log_status_message("ACQUIRE_FLUORESCENCE_IMAGE", "Acquiring Fluorescence Image...")
        self.microscope.fm.acquisition_progress_signal.emit({"state": "acquiring", "task": f"{self.task_name}"})
        image = acquire_image(microscope=self.microscope.fm,
                                channel_settings=self.config.channel_settings,
                                zparams=self.config.zparams,
                                stop_event=self._stop_event,
                                filename=filename)

        # store the fluorescence pose
        self.lamella.fluorescence_pose = self.microscope.get_microscope_state()

        # QUERY: update objective position to use new focused position?
        try:
            if self.parent_ui is not None and hasattr(self.parent_ui, 'fm_widget') and self.parent_ui.fm_widget is not None:
                self.parent_ui.fm_widget.update_persistent_image_signal.emit(image)
        except Exception as e:
            logging.error(f"Error setting FM image: {e}")

    def _run_autofocus(self,):
        """Run autofocus with the specified settings."""
        # Find autofocus channel by name if specified in autofocus_settings
        if self.config.autofocus_settings.channel_name:
            for ch in self.config.channel_settings:
                if ch.name == self.config.autofocus_settings.channel_name:
                    autofocus_channel = ch
                    break

        # Fall back to first channel if no specific channel found
        if autofocus_channel is None:
            logging.warning(f"Autofocus channel '{self.config.autofocus_settings.channel_name}' not found, using first channel")
            autofocus_channel = self.config.channel_settings[0]

        # Set up Z parameters for autofocus
        autofocus_method = self.config.autofocus_settings.method.value
        autofocus_zparams = ZParameters(
            zmin=-self.config.autofocus_settings.fine_range/2,
            zmax=self.config.autofocus_settings.fine_range/2, 
            zstep=self.config.autofocus_settings.fine_step
        )

        logging.info(f"Running autofocus for {self.lamella.name} using channel '{autofocus_channel.name}' with method '{autofocus_method}' and zparams: {autofocus_zparams}")
        if self.validate:
            ask_user(self.parent_ui,
                msg=f"Run autofocus for {self.lamella.name} using channel '{autofocus_channel.name}'. Press continue when ready.",
                pos="Continue"
                )

        self.log_status_message("AUTOFOCUS", "Running Autofocus...")
        self.microscope.fm.acquisition_progress_signal.emit({"state": "autofocusing", "task": f"{self.task_name}"}) # type: ignore
        result = run_autofocus(microscope=self.microscope.fm,                                                       # type: ignore
                                channel_settings=autofocus_channel,
                                z_parameters=autofocus_zparams, 
                                method=autofocus_method,
                                stop_event=self._stop_event)
        if result is None:
            logging.info("Autofocus cancelled")
            raise InterruptedError(f"Task {self.task_name} for {self.lamella.name} cancelled during autofocus.")
