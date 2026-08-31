######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem import config as fcfg
from fibsem import timing
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.interaction import RunSpotBurn, ask
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.applications.autolamella.workflows.ui import _abort_requested
from fibsem.imaging.spot import SpotBurnSettings, run_spot_burn
from fibsem.structures import BeamType, Point, field_meta


@dataclass
class SpotBurnFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SpotBurnFiducialTask."""

    task_type: ClassVar[str] = "SPOT_BURN_FIDUCIAL"
    display_name: ClassVar[str] = "Spot Burn Fiducial"
    milling_current: float = field(
        default=60.0e-12,  # in Amperes
        metadata=field_meta(tooltip="Milling current in Amperes", unit="A", scale=1e12),
    )
    exposure_time: int = field(
        default=10,
        metadata=field_meta(tooltip="Exposure time in seconds", unit="s", scale=1),
    )
    autofocus: bool = field(
        default=False,
        metadata=field_meta(
            tooltip="Run a FIB autofocus before acquiring the reference image, so the "
            "points are placed on (and burned into) a focused image",
            label="Autofocus",
        ),
    )
    coordinates: list[Point] = field(
        default_factory=list,
        metadata=field_meta(
            tooltip="Spot burn positions in normalised image coordinates (0-1)"
        ),
    )

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(p for p in super().parameters if p != "coordinates")

    @property
    def opens_with_reference_alignment(self) -> bool:
        return True

    @property
    def estimated_duration(self) -> float:
        """The base estimate plus everything ``_run`` does, in the order it does it.

        The burn is the dominant term and the base cannot see it: on the measured run,
        11 points at 10 s was 110 s of the task. But the burn alone left the estimate
        at 0.71x of the machine time, and the remainder is all here -- the move onto
        the milling pose, the alignment against the stored reference, and the beam
        current switching into the burn current and back out.

        Each point costs its exposure plus a blank/park/unblank cycle.
        """
        total = super().estimated_duration
        total += 2 * timing.BEAM_CURRENT_CHANGE_S  # into the burn current and back
        total += len(self.coordinates) * (
            self.exposure_time + timing.SPOT_BURN_POINT_OVERHEAD_S
        )
        return total

    def to_dict(self) -> dict:
        ddict = {}
        ddict["task_type"] = self.task_type
        ddict["parameters"] = {
            "milling_current": self.milling_current,
            "exposure_time": self.exposure_time,
            "autofocus": self.autofocus,
        }
        ddict["milling"] = {k: v.to_dict() for k, v in self.milling.items()}
        if self.reference_imaging is not None:
            ddict["reference_imaging"] = self.reference_imaging.to_dict()
        ddict["coordinates"] = [pt.to_dict() for pt in self.coordinates]
        return ddict

    @classmethod
    def from_dict(cls, ddict: dict) -> "SpotBurnFiducialTaskConfig":
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
            autofocus=bool(params.get("autofocus", False)),
            coordinates=coordinates,
        )

    def to_settings(self) -> SpotBurnSettings:
        """The run payload (coordinates + current + exposure) for this task."""
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

        # restore the full milling-pose state, then align to the stored reference so
        # the burn coordinates land on target (mirrors rough/polishing)
        self._move_to_milling_pose()
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        self.config.exposure_time = float(self.config.exposure_time)

        # focus before the reference image, not after: the points are placed on that
        # image, and the burn lands where they were placed. Same field of view, so the
        # sweep is scored on the view the user actually works in.
        field_of_view = self.config.reference_imaging.field_of_view1
        if self.config.autofocus:
            self._run_autofocus(BeamType.ION, hfw=field_of_view)

        # acquire images, set ui
        self._acquire_reference_image(image_settings, field_of_view=field_of_view)

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
            run_spot_burn(
                microscope=self.microscope,
                settings=self.config.to_settings(),
                beam_type=BeamType.ION,
                stop_event=self._stop_event,
            )
            return

        # supervised path: one question over the Responder, mirroring milling. The
        # whole run/wait/re-prompt loop lives on the GUI thread that owns the
        # widget; this thread blocks on the answer — the settings as actually used
        # (coordinates + current/exposure), with the widget already cleared. None
        # means no spot-burn widget: optional hardware must not fail a workflow.
        msg = f"Place points and run the spot burn for {self.lamella.name}. Press Continue when finished."
        try:
            settings = ask(
                self.parent_ui.ui_responder,
                RunSpotBurn(settings=self.config.to_settings(), message=msg),
                abort=lambda: _abort_requested(self.parent_ui),
            )
        except InterruptedError:
            # workflow stopped: take a running burn down with the task. Covers the
            # race where the burn starts after the Stop click already ran cancel
            # (the worker clears its stop_event on start). cancel_spot_burn only
            # sets a threading.Event, so it is safe to call from the task thread.
            spot_burn_widget = self.parent_ui.spot_burn_widget
            if spot_burn_widget is not None:
                spot_burn_widget.cancel_spot_burn()
            raise

        if settings is not None:
            # store the user's settings back to the config
            self.config.apply_settings(settings)
