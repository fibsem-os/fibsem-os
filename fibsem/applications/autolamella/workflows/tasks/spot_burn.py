######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem import config as fcfg
from fibsem import timing
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.interaction import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    ClearSpotBurn,
    RunSpotBurn,
    ask,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.applications.autolamella.workflows.ui import (
    INSTRUCTION_TIMEOUT_S,
    _abort_requested,
)
from fibsem.imaging.spot import SpotBurnSettings, run_spot_burn, show_after_burn
from fibsem.structures import BeamType, Display, Point, field_meta


@dataclass
class SpotBurnFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SpotBurnFiducialTask."""

    task_type: ClassVar[str] = "SPOT_BURN_FIDUCIAL"
    display_name: ClassVar[str] = "Spot Burn Fiducial"
    milling_current: float = field(
        default=100.0e-12,  # in Amperes; below this the marks are too faint to find
        metadata=field_meta(
            unit="A", doc="Milling current in Amperes", display=Display(scale=1e12)
        ),
    )
    exposure_time: int = field(
        default=10,
        metadata=field_meta(unit="s", doc="Exposure time in seconds"),
    )
    autofocus: bool = field(
        default=False,
        metadata=field_meta(
            doc="Run a FIB autofocus before acquiring the reference image, so the "
            "points are placed on (and burned into) a focused image",
            display=Display("Autofocus"),
        ),
    )
    coordinates: list[Point] = field(
        default_factory=list,
        metadata=field_meta(
            doc="Spot burn positions in normalised image coordinates (0-1)"
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
        params = ddict.get("parameters") or {}
        cls._warn_unknown_parameters(params)
        coordinates = [Point.from_dict(pt) for pt in ddict.get("coordinates", [])]
        return cls(
            **cls._load_core(ddict),
            # coerce numeric params: older protocols may have stored these as strings
            milling_current=float(params.get("milling_current", 100.0e-12)),
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

    sessions = ("the spot burn",)

    config: SpotBurnFiducialTaskConfig
    config_cls: ClassVar[Type[SpotBurnFiducialTaskConfig]] = SpotBurnFiducialTaskConfig

    def _run(self) -> None:
        """Run the task to mill spot fiducial markers for correlation."""
        # bookkeeping
        image_settings = self.config.imaging

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
        """Burn the points, asking first when supervised, and keep the settings
        as last used.

        The loop is the task's, on this thread, as the mill's is (FIB-1236).
        Supervised, it asks :class:`RunSpotBurn` what to do: run burns the
        settings as answered and asks again with the pass counted (and the
        error, if the burn failed); edit asks again with the settings as
        answered; skip and accept end the loop. A run with no points burns
        nothing and asks again, saying so. Unsupervised or headless, it
        burns the stored points once, and a failure fails the task; with no
        points it skips. Supervision is read at each decision.
        """
        responder = getattr(self.parent_ui, "ui_responder", None)
        settings = self.config.to_settings()
        msg = f"Place points for the spot burn of {self.lamella.name}."
        burned, passes, error, refused = settings, 0, None, None
        try:
            while True:
                if responder is not None and self.validate:
                    decision = ask(
                        responder,
                        RunSpotBurn(
                            settings=deepcopy(settings),
                            message=msg,
                            passes=passes,
                            last_error=refused
                            or (None if error is None else str(error)),
                        ),
                        abort=lambda: _abort_requested(self.parent_ui),
                    )
                    if decision is None:
                        # No spot-burn widget: optional hardware must not fail
                        # a workflow.
                        return
                    if decision.action == ACTION_ACCEPT and passes:
                        settings = burned  # not an edit made after the burn
                        break
                    settings, refused = decision.settings, None
                    if decision.action == ACTION_EDIT:
                        continue
                    if decision.action != ACTION_RUN:
                        break
                    if not settings.coordinates:
                        refused = "No points to burn: place some first."
                        continue
                elif passes:
                    if error is not None:
                        raise error
                    break
                elif not settings.coordinates:
                    logging.warning(
                        f"No spot burn coordinates set for {self.lamella.name}; "
                        "skipping spot burn."
                    )
                    return
                error = self._burn(settings, show=responder is not None)
                burned, passes = settings, passes + 1
        finally:
            # After a Stop the workflow's end clears the widget instead.
            if responder is not None and not _abort_requested(self.parent_ui):
                ask(
                    responder,
                    ClearSpotBurn(),
                    abort=lambda: _abort_requested(self.parent_ui),
                    timeout=INSTRUCTION_TIMEOUT_S,
                )
        self.config.apply_settings(settings)

    def _burn(self, settings: SpotBurnSettings, show: bool) -> Optional[Exception]:
        """Burn once, under the task's stop; the error if it failed. With a
        window, the FIB view then shows what the burn left. A Stop of the
        workflow during it raises once it has ended."""
        error: Optional[Exception] = None
        try:
            run_spot_burn(
                microscope=self.microscope,
                settings=settings,
                beam_type=BeamType.ION,
                stop_event=self._stop_event,
            )
        except Exception as exc:  # noqa: BLE001 - asked again, or raised, by the loop
            logging.error(f"Spot burn failed: {exc}", exc_info=True)
            error = exc
        self._check_for_abort()
        if error is None and show:
            show_after_burn(self.microscope)
        return error
