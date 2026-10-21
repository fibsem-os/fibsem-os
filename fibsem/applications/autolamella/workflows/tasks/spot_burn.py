######## SPOT BURN FIDUCIAL TASK DEFINITIONS ########
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem import timing
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.imaging.spot import SpotBurnSettings, run_spot_burn, show_after_burn
from fibsem.structures import BeamType, Display, Point, param
from fibsem.workflows.interaction import (
    INSTRUCTION_TIMEOUT_S,
    ClearSpotBurn,
    _abort_requested,
    ask,
)
from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    ACTION_SKIP,
    SPOT_BURN,
    DecisionOutcome,
)


@dataclass
class SpotBurnFiducialTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SpotBurnFiducialTask."""

    task_type: ClassVar[str] = "SPOT_BURN_FIDUCIAL"
    display_name: ClassVar[str] = "Spot Burn Fiducial"
    milling_current: float = param(
        100.0e-12,
        unit="A",
        doc="Milling current in Amperes",
        display=Display(scale=1e12),
    )
    exposure_time: int = param(10, unit="s", doc="Exposure time in seconds")
    autofocus: bool = param(
        False,
        doc="Run a FIB autofocus before acquiring the reference image, so the "
        "points are placed on (and burned into) a focused image",
        display=Display("Autofocus"),
    )
    coordinates: list[Point] = param(
        doc="Spot burn positions in normalised image coordinates (0-1)",
        default_factory=list,
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

    asks = (SPOT_BURN,)

    config: SpotBurnFiducialTaskConfig
    config_cls: ClassVar[Type[SpotBurnFiducialTaskConfig]] = SpotBurnFiducialTaskConfig

    def _run(self) -> None:
        """Run the task to mill spot fiducial markers for correlation."""
        # bookkeeping
        image_settings = self.config.imaging

        # restore the full milling-pose state, then align to the stored reference so
        # the burn coordinates land on target (mirrors rough/polishing)
        self._move_to_pose()
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

        The loop is the task's, on this thread, as the mill's is (FIB-1236):
        each pass a ``spot_burn`` question through ``ask``, on the record,
        decided as run (burn the settings as decided and ask again with the
        pass counted, and the error if the burn failed), edit (ask again with
        the settings as decided), skip or accept (end the loop). A run with no
        points burns nothing and asks again, saying so. Asked of nobody, the
        first pass burns the stored points once and the next accepts, and a
        failure fails the task; with no points it skips.
        """
        responder = self.responder
        settings = self.config.to_settings()
        msg = f"Place points for the spot burn of {self.lamella.name}."
        burned, passes, error, refused, watched = settings, 0, None, None, False
        if not settings.coordinates and self.unasked_reason(SPOT_BURN):
            logging.warning(
                f"No spot burn coordinates set for {self.lamella.name}; "
                "skipping spot burn."
            )
            return
        try:
            while True:
                actions = (
                    (ACTION_RUN, ACTION_EDIT, ACTION_SKIP)
                    if passes == 0
                    else (ACTION_RUN, ACTION_EDIT, ACTION_ACCEPT)
                )
                asked = deepcopy(settings)
                decision = self.ask(
                    SPOT_BURN,
                    {"spot_burn": asked},
                    image=self._last_fib_image_file(),
                    provenance={
                        "actions": list(actions),
                        "passes": passes,
                        "last_error": refused
                        or (None if error is None else str(error)),
                    },
                    message=msg,
                    decided=lambda: {"spot_burn": asked},
                )
                if decision.outcome is DecisionOutcome.Rejected:
                    raise RuntimeError(
                        f"{self.task_name} was rejected: "
                        f"{decision.reason or 'no reason given'}"
                    )
                if decision.outcome is not DecisionOutcome.Unreviewed:
                    watched = True
                if decision.action == ACTION_ACCEPT and passes:
                    if error is not None and not watched:
                        raise error
                    settings = burned  # not an edit made after the burn
                    break
                settings, refused = decision.values.get("spot_burn", settings), None
                if decision.action == ACTION_EDIT:
                    continue
                if decision.action != ACTION_RUN:
                    break
                if not settings.coordinates:
                    if decision.outcome is DecisionOutcome.Unreviewed:
                        break  # nobody here to place them
                    refused = "No points to burn: place some first."
                    continue
                error = self._burn(settings, show=responder is not None)
                burned, passes = settings, passes + 1
        finally:
            # After a Stop the workflow's end clears the widget instead.
            if responder is not None and not _abort_requested(self.task_manager):
                ask(
                    responder,
                    ClearSpotBurn(),
                    abort=lambda: _abort_requested(self.task_manager),
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
