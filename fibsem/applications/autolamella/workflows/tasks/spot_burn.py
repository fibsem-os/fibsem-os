
######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem import config as fcfg
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
    clear_spot_burn_ui,
    update_spot_burn_parameters,
)
from fibsem.structures import BeamType, Point


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
    orientation: Literal["SEM", "FIB", "FM", "MILLING"] = field(
        default="MILLING",
        metadata={"help": "The orientation to perform spot burning in", 
                  "items": ["SEM", "FIB", "FM", "MILLING"]},
    )
    coordinates: list[Point] = field(
        default_factory=list,
        metadata={"help": "Spot burn positions in normalised image coordinates (0-1)"},
    )

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(p for p in super().parameters if p != "coordinates")

    def to_dict(self) -> dict:
        ddict = {}
        ddict["task_type"] = self.task_type
        ddict["parameters"] = {
            "milling_current": self.milling_current,
            "exposure_time": self.exposure_time,
            "orientation": self.orientation,
        }
        ddict["milling"] = {k: v.to_dict() for k, v in self.milling.items()}
        if self.reference_imaging is not None:
            ddict["reference_imaging"] = self.reference_imaging.to_dict()
        ddict["coordinates"] = [pt.to_dict() for pt in self.coordinates]
        return ddict

    @classmethod
    def from_dict(cls, ddict: dict) -> 'SpotBurnFiducialTaskConfig':
        cfg = AutoLamellaTaskConfig.from_dict(ddict)
        params = ddict.get("parameters", {})
        coordinates = [Point.from_dict(pt) for pt in ddict.get("coordinates", [])]
        return cls(
            task_name=cfg.task_name,
            milling=cfg.milling,
            reference_imaging=cfg.reference_imaging,
            milling_current=params.get("milling_current", 60.0e-12),
            exposure_time=params.get("exposure_time", 10),
            orientation=params.get("orientation", "MILLING"),
            coordinates=coordinates,
        )

    def to_settings(self) -> SpotBurnSettings:
        """The run payload (coordinates + current + exposure) for this task."""
        from fibsem.imaging.spot import SpotBurnSettings
        return SpotBurnSettings(
            coordinates=list(self.coordinates),
            milling_current=self.milling_current,
            exposure_time=float(self.exposure_time),
        )

    def apply_settings(self, settings: SpotBurnSettings) -> None:
        """Apply a run payload back onto this task config (coordinates + current + exposure)."""
        self.coordinates = list(settings.coordinates)
        self.milling_current = settings.milling_current
        self.exposure_time = settings.exposure_time


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
        target_position = self._get_stage_position_for_orientation(stage_position,
                                                                   self.config.orientation)
        self.microscope.safe_absolute_stage_movement(target_position)

        self.config.exposure_time = float(self.config.exposure_time)

        # acquire images, set ui
        self._acquire_reference_image(image_settings, field_of_view=self.config.reference_imaging.field_of_view1)

        self.log_status_message("SPOT_BURN_FIDUCIAL")

        # update the spot burn parameters in the UI
        self.update_spot_burn_parameters_ui()

        # acquire final reference images
        self._acquire_set_of_reference_images(image_settings)

    def update_spot_burn_parameters_ui(self):
        """Update the spot burn parameters in the UI, or run automatically if unsupervised."""
        if (not self.validate and self.config.coordinates) or self.parent_ui is None:
            # run spot burn automatically with the stored settings
            # (no parent_ui -> no progress signals; headless)
            self.config.to_settings().run(
                microscope=self.microscope,
                beam_type=BeamType.ION,
                stop_event=self._stop_event,
            )
            return

        if self.parent_ui is None or self.parent_ui.spot_burn_widget is None:
            logging.warning("Spot burn widget not available in UI.")
            return

        update_spot_burn_parameters(
            parent_ui=self.parent_ui, settings=self.config.to_settings()
        )

        # ask the user to place/adjust the spots and burn
        msg = f"Run the spot burn workflow for {self.lamella.name}. Press continue when finished."
        ask_user(self.parent_ui, msg=msg, pos="Continue", spot_burn=True)

        # store the user's settings (coordinates + current/exposure) back to the config
        self.config.apply_settings(self.parent_ui.spot_burn_widget.get_settings())

        # clear the spot burn UI
        clear_spot_burn_ui(self.parent_ui)
