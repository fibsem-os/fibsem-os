
######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########
from __future__ import annotations

import logging
import time
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
            # coerce numeric params: older protocols may have stored these as strings
            milling_current=float(params.get("milling_current", 60.0e-12)),
            exposure_time=int(float(params.get("exposure_time", 10))),
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

        self.log_status_message("SPOT_BURN_FIDUCIAL", "Running Spot Burn...")

        # update the spot burn parameters in the UI
        self.update_spot_burn_parameters_ui()

        # acquire final reference images
        self._acquire_set_of_reference_images(image_settings)

    def update_spot_burn_parameters_ui(self):
        """Run the spot burn automatically (unsupervised/headless), or hand off to the UI.

        Supervised runs let the user place/adjust points and run the burn in the spot
        burn widget; unsupervised/headless runs burn the stored coordinates directly.
        """
        # automatic path: no user in the loop (unsupervised or headless). Skip (rather than
        # block on the interactive prompt) when there are no coordinates to burn.
        if not self.validate or self.parent_ui is None:
            if not self.config.coordinates:
                logging.warning(
                    f"No spot burn coordinates set for {self.lamella.name}; skipping spot burn."
                )
                return
            # burn the stored coordinates directly (progress via the microscope signal)
            self.config.to_settings().run(
                microscope=self.microscope,
                beam_type=BeamType.ION,
                stop_event=self._stop_event,
            )
            return

        # supervised path: task-orchestrated run/wait/re-prompt loop (mirrors milling).
        # The user runs the burn via the workflow "Run Spot Burn" button; the task waits
        # for each burn to finish before continuing so the workflow can't advance mid-burn.
        if self.parent_ui.spot_burn_widget is None:
            logging.warning("Spot burn widget not available in UI.")
            return

        update_spot_burn_parameters(
            parent_ui=self.parent_ui, settings=self.config.to_settings()
        )

        # supervised run/wait/re-prompt loop (mirrors milling): the user runs each burn
        # via the workflow "Run Spot Burn" control, and the task waits for it to finish
        # before continuing so the workflow can't advance mid-burn.
        spot_burn_widget = self.parent_ui.spot_burn_widget
        msg = f"Place points and run the spot burn for {self.lamella.name}. Press Continue when finished."
        response = ask_user(self.parent_ui, msg=msg, pos="Run Spot Burn", neg="Continue", spot_burn=True)
        while response:
            self.update_status_ui("Running Spot Burn...")
            spot_burn_widget.start_spot_burn_signal.emit()
            # BlockingQueuedConnection: on return from emit the burn is either running
            # (is_burning=True) or was refused (no in-bounds points), in which case the
            # wait loop exits immediately and the user is re-prompted.
            try:
                while spot_burn_widget.is_burning:
                    self._check_for_abort()
                    time.sleep(1)
            except InterruptedError:
                # workflow stopped: take the burn down with the task. Covers the race
                # where the burn starts after the Stop click already ran cancel (the
                # worker clears its stop_event on start). cancel_spot_burn only sets
                # a threading.Event, so it is safe to call from the task thread.
                spot_burn_widget.cancel_spot_burn()
                raise
            response = ask_user(self.parent_ui, msg=msg, pos="Run Spot Burn", neg="Continue", spot_burn=True)

        # store the user's settings (coordinates + current/exposure) back to the config
        self.config.apply_settings(spot_burn_widget.get_settings())

        # clear the spot burn UI
        clear_spot_burn_ui(self.parent_ui)
