from __future__ import annotations

from datetime import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fibsem import acquire
from fibsem import config as fcfg
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import BeamType, FibsemImage, ImageSettings, MillingAlignment
from fibsem.utils import current_timestamp_v3

if TYPE_CHECKING:
    from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget

@dataclass
class MillingTaskAcquisitionSettings:
    """Settings for the acquisition of images during a milling task."""
    enabled: bool = True
    imaging: ImageSettings = field(default_factory=ImageSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "imaging": self.imaging.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MillingTaskAcquisitionSettings":
        imaging = data.get("imaging", {})
        if imaging == {} or imaging.get("path", None) is None:
            imaging["path"] = None
        return cls(
            enabled=data.get("enabled", True),
            imaging=ImageSettings.from_dict(imaging),
        )


@dataclass
class FibsemMillingTaskConfig:
    """Configuration for a milling task."""
    name: str = "Milling Task"
    field_of_view: float = 150e-6
    channel: BeamType = BeamType.ION
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    acquisition: MillingTaskAcquisitionSettings = field(default_factory=MillingTaskAcquisitionSettings)
    stages: List[FibsemMillingStage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field_of_view": self.field_of_view,
            "channel": self.channel.value,
            "alignment": self.alignment.to_dict(),
            "acquisition": self.acquisition.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FibsemMillingTaskConfig":
        alignment = data.get("alignment", {})
        acquisition = data.get("acquisition", {})
        return cls(
            name=data.get("name", "Milling Task"),
            field_of_view=data.get("field_of_view", 150e-6),
            channel=BeamType(data.get("channel", BeamType.ION.value)),
            alignment=MillingAlignment.from_dict(alignment),
            acquisition=MillingTaskAcquisitionSettings.from_dict(acquisition),
            stages=[FibsemMillingStage.from_dict(stage) for stage in data.get("stages", [])],
        )

    @classmethod
    def from_stages(cls, stages: List[FibsemMillingStage], name: str = "Milling Task") -> "FibsemMillingTaskConfig":
        """Create a FibsemMillingTaskConfig from a list of FibsemMillingStage."""

        if not stages:
            raise ValueError("No milling stages provided to create task config.")

        # Use the first stage's properties as defaults
        first_stage = stages[0]
        return FibsemMillingTaskConfig(
            name=first_stage.name,
            field_of_view=first_stage.milling.hfw,
            channel=first_stage.milling.milling_channel,
            alignment=first_stage.alignment,
            acquisition=MillingTaskAcquisitionSettings(
                enabled=first_stage.milling.acquire_images,
                imaging=first_stage.imaging,
            ),
            stages=stages,
        )

    @property
    def estimated_time(self) -> float:
        """Estimate the total milling time for a list of milling stages"""
        return sum([stage.estimated_time for stage in self.stages])

@dataclass
class FibsemMillingTask:
    config: FibsemMillingTaskConfig = field(default_factory=FibsemMillingTaskConfig)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reference_image: Optional[FibsemImage] = None

    def __init__(self, microscope: FibsemMicroscope, config: FibsemMillingTaskConfig, parent_ui: Optional['FibsemMillingWidget'] = None):
        self.config = config
        self.microscope = microscope
        self.parent_ui = parent_ui
        self.task_id = str(uuid.uuid4())

    @property
    def name(self) -> str:
        """Return the name of the milling task."""
        return self.config.name

    @property
    def stages(self) -> List[FibsemMillingStage]:
        """Return the list of milling stages."""
        return self.config.stages

    def _handle_progress(self, ddict: dict) -> None:
        """Handle progress updates from the microscope."""
        if self.parent_ui:
            self.parent_ui.milling_progress_signal.emit(ddict)
        else:
            logging.info(ddict)

    def run(self) -> None:
        """Run a list of milling stages, with a progress bar and notifications."""

        logging.info(f"Running milling task: {self.name} with ID: {self.task_id}")

        try:
            self.microscope.milling_progress_signal.connect(self._handle_progress)
            initial_beam_shift = self.microscope.get_beam_shift(beam_type=self.config.channel)

            # acquire a reference image for alignment
            if self.config.alignment.enabled:
                self._acquire_reference_image()

            for idx, stage in enumerate(self.stages):
                start_time = time.time()
                if self.parent_ui:
                    if self.parent_ui.STOP_MILLING:
                        raise Exception("Milling stopped by user.")

                msgd =  {"msg": f"Preparing: {stage.name}",
                        "progress": {"state": "start", 
                                    "start_time": start_time,
                                    "current_stage": idx, 
                                    "total_stages": len(self.stages),
                                    "task_id": self.task_id,
                                    "task_name": self.name
                                    }}
                self._handle_progress(msgd)

                try:
                    if self.config.acquisition.enabled:
                        self._acquire_milling_task_images(stage_name=stage.name, tag="start")

                    # Set up the stage with the task configuration
                    stage.reference_image = self.reference_image
                    stage.milling.hfw = self.config.field_of_view
                    stage.milling.milling_channel = self.config.channel
                    stage.milling.acquire_images = self.config.acquisition.enabled
                    stage.alignment = self.config.alignment
                    stage.strategy.run(
                        microscope=self.microscope,
                        stage=stage,
                        asynch=False,
                        parent_ui=self.parent_ui,
                    )
                    # TODO: pass task as parent into strategy.run()?, allow logging from strategy?
                    # performance logging
                    msgd = {"msg": "milling_task",
                            "task_id": self.task_id,
                            "task_name": self.name,
                            "idx": idx,
                            "stage": stage.to_dict(),
                            "start_time": start_time,
                            "end_time": time.time(),
                            "timestamp": datetime.now().isoformat()}
                    logging.debug(f"{msgd}")

                    # optionally acquire images after milling
                    if self.config.acquisition.enabled:
                        self._acquire_milling_task_images(stage_name=stage.name, tag="finished")

                    if self.parent_ui:
                        self.parent_ui.milling_progress_signal.emit({"msg": f"Finished: {stage.name}"})
                except Exception as e:
                    logging.error(f"Error running milling stage: {stage.name}, {e}")

            self._handle_progress({"msg": f"Finished Milling Task: {self.name}. Restoring Imaging Conditions..."})

        except Exception as e:
            logging.error(e)
        finally:
            self.microscope.finish_milling(
                imaging_current=self.microscope.system.ion.beam.beam_current,
                imaging_voltage=self.microscope.system.ion.beam.voltage,
            )
            # restore initial beam shift
            if initial_beam_shift:
                self.microscope.set_beam_shift(initial_beam_shift, beam_type=self.config.channel)
            self.microscope.milling_progress_signal.disconnect(self._handle_progress)

    def _acquire_reference_image(self) -> Optional[FibsemImage]:
        """Acquire a reference image for the milling task."""

        if self.reference_image is not None:
            return self.reference_image

        path = self.config.acquisition.imaging.path
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)

        filename = f"ref_milling_{self.name}_{fcfg.REFERENCE_FILENAME}_{current_timestamp_v3(timeonly=True)}".replace(' ', '-')
        image_settings = ImageSettings(
            hfw=self.config.field_of_view,
            dwell_time=1e-6,
            resolution=(1536, 1024),
            beam_type=self.config.channel,
            reduced_area=self.config.alignment.rect,
            save=True,
            path=path,
            filename=filename,
        )
        self.reference_image =  acquire.acquire_image(microscope=self.microscope, settings=image_settings)

    def _acquire_milling_task_images(
        self,
        stage_name: str,
        tag: str = "finished",
    ) -> Tuple[FibsemImage, FibsemImage]:
        """Acquire images after milling for reference.
        Args:
            stage_name (str): Name of the milling stage
            tag (str): Tag to append to the filename
        """
        self.microscope.finish_milling(self.microscope.system.ion.beam.beam_current, self.microscope.system.ion.beam.voltage)

        # TODO: migrate to path = path/task_name/stage_name?
        acq_date = current_timestamp_v3(timeonly=True)
        self.config.acquisition.imaging.filename = f"ref_milling_{self.name}_{stage_name}_{tag}_{acq_date}".replace(' ', '-')
        self.config.acquisition.imaging.save = True
        self.config.acquisition.imaging.hfw = self.config.field_of_view

        if self.config.acquisition.imaging.path is None:
            self.config.acquisition.imaging.path = self.microscope._last_imaging_settings.path

        # acquire images
        images = acquire.take_reference_images(self.microscope, self.config.acquisition.imaging)

        return images

def run_milling_task(microscope: FibsemMicroscope, 
                     config: FibsemMillingTaskConfig, 
                     parent_ui: Optional['FibsemMillingWidget'] = None) -> FibsemMillingTask:
    """Run a milling task with the given configuration."""
    task = FibsemMillingTask(microscope=microscope, 
                             config=config, 
                             parent_ui=parent_ui)
    task.run()
    return task
