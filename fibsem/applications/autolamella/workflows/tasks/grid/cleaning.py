"""The cryo cleaning grid task: scan the ion beam over the grid to clean its surface.

The operation -- `cryo_clean` -- is a plain function, callable from a script or,
later, from a lamella workflow mid-run. It scans frames of the ion beam over a
field at a high current for a fixed time, stopping between frames when asked, and
puts the current back however it ends. The task adds what makes it a workflow
step: the grid's slot at the requested orientation, and an ion reference image of
the cleaned field, recorded by role with a thumbnail beside it.

The defaults are the prototype's (15 nA over 900 um for 10 s, at the SEM
orientation); they have not been checked on an instrument.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Optional, Type

from fibsem.acquire import new_image
from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
    register_grid_task,
)
from fibsem.cancellation import raise_if_cancelled
from fibsem.imaging.thumbnail import write_thumbnail
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.microscopes._stage import uncalibrated_message
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# The frames the clean is scanned as. What cleans is the dose -- the current, over
# the field, for the time -- not the pixels, so a small fast frame: it is also how
# often a Stop is seen.
CLEANING_RESOLUTION = (768, 512)
CLEANING_DWELL_TIME = 0.2e-6

# The reference image of the cleaned field, at the beam's imaging current.
REFERENCE_RESOLUTION = (1536, 1024)
REFERENCE_DWELL_TIME = 1e-6

ROLE = "cleaning_fib"


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def cryo_clean(
    microscope: "FibsemMicroscope",
    current: float,
    field_of_view: float,
    duration: float,
    stop_event=None,
) -> int:
    """Scan the ion beam over `field_of_view` at `current` for `duration` seconds.

    Frames are scanned until the time is up -- at least one, and the last may run
    past it by up to one frame -- and `stop_event` is checked between them, raising
    `OperationCancelledError` when set. The ion current is restored to what it was
    on every path, a failure or a Stop included: leaving the beam at a cleaning
    current would mill whatever is imaged next. The field of view is left at the
    cleaned field. Returns the number of frames scanned.
    """
    settings = ImageSettings(
        resolution=CLEANING_RESOLUTION,
        dwell_time=CLEANING_DWELL_TIME,
        hfw=field_of_view,
        beam_type=BeamType.ION,
        autocontrast=False,
        save=False,
    )
    previous = microscope.get_beam_current(BeamType.ION)
    frames = 0
    try:
        actual = microscope.set_beam_current(current, BeamType.ION)
        logging.info(
            f"Cryo cleaning at {actual * 1e9:.2f} nA over {field_of_view * 1e6:.0f} um "
            f"for {duration:.1f} s"
        )
        start = time.monotonic()
        while frames == 0 or time.monotonic() - start < duration:
            raise_if_cancelled(stop_event, "Cryo cleaning cancelled by user.")
            microscope.acquire_image(image_settings=settings)
            frames += 1
    finally:
        try:
            microscope.set_beam_current(previous, BeamType.ION)
        except Exception:  # noqa: BLE001 - say so, without hiding why the clean ended
            logging.exception(
                f"Could not restore the ion current to {previous * 1e9:.2f} nA "
                "after cryo cleaning. Check it before imaging."
            )
    return frames


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


@dataclass
class CryoCleaningGridTaskConfig(GridTaskConfig):
    """Clean the grid's surface with the ion beam, then image it with the ion beam."""

    task_type: ClassVar[str] = "CRYO_CLEANING_GRID"
    display_name: ClassVar[str] = "Cryo cleaning"
    orientation: Literal["SEM", "FIB", "MILLING"] = field(
        default="SEM",
        metadata={
            "tooltip": "The pose to clean at; the grid's slot position is "
            "re-expressed for it",
        },
    )
    current: float = field(
        default=15e-9,
        metadata={
            "label": "Ion current",
            "unit": "A",
            "scale": 1e9,
            "minimum": 0.0,
            "decimals": 2,
            "tooltip": "The ion current to clean at; the instrument uses its "
            "nearest. Restored afterwards.",
        },
    )
    field_of_view: float = field(
        default=900e-6,
        metadata={
            "label": "Field of view",
            "unit": "m",
            "scale": 1e6,
            "minimum": 1.0,
            "decimals": 0,
            "step": 50.0,
            "tooltip": "The width of the field the ion beam scans",
        },
    )
    duration: float = field(
        default=10.0,
        metadata={
            "unit": "s",
            "minimum": 0.0,
            "decimals": 1,
            "tooltip": "How long to scan; Stop ends it between frames",
        },
    )
    acquire_reference: bool = field(
        default=True,
        metadata={
            "label": "Reference image",
            "tooltip": "Take an ion image of the cleaned field afterwards",
        },
    )
    filename: str = field(
        default="cleaned",
        metadata={
            "tooltip": "The stem of the reference image's name; a time stamp is "
            "added per run",
        },
    )

    @property
    def role(self) -> str:
        return ROLE


@register_grid_task
class CryoCleaningGridTask(GridTask):
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoCleaningGridTaskConfig
    config: CryoCleaningGridTaskConfig

    @property
    def result_images(self):
        return {"reference_image": ROLE} if self.config.acquire_reference else {}

    def grid_centre(self) -> FibsemStagePosition:
        """The grid's calibrated slot position, in the requested orientation.

        Refuses a grid that is not in a holder slot, and a slot with no calibrated
        position, rather than cleaning wherever the stage happens to be.
        """
        slot = self.slot
        if slot is None:
            raise RuntimeError(
                f"Grid '{self.grid.name}' is not in a holder slot. Load it first."
            )
        if slot.position is None:
            raise RuntimeError(uncalibrated_message(slot.name))
        return self.microscope.get_target_position(
            slot.position, self.config.orientation
        )

    def _run(self) -> None:
        centre = self.grid_centre()
        self.log_status_message(
            "MOVE_TO_GRID",
            f"Moving to {self.grid.name} at the {self.config.orientation} orientation",
        )
        self.microscope.safe_absolute_stage_movement(centre)
        self._check_for_abort()

        self.log_status_message(
            "CLEAN",
            f"Cleaning for {self.config.duration:.0f} s at "
            f"{self.config.current * 1e9:.1f} nA",
        )
        frames = cryo_clean(
            self.microscope,
            self.config.current,
            self.config.field_of_view,
            self.config.duration,
            stop_event=self._stop_event,
        )
        self.log_status_message("CLEANED", f"Cleaned ({frames} frames)")
        if not self.config.acquire_reference:
            return
        self._check_for_abort()

        self.log_status_message("REFERENCE", "Acquiring the ion reference image")
        image = new_image(
            self.microscope,
            ImageSettings(
                resolution=REFERENCE_RESOLUTION,
                dwell_time=REFERENCE_DWELL_TIME,
                hfw=self.config.field_of_view,
                beam_type=BeamType.ION,
                autocontrast=True,
                save=True,
                path=self.output_dir,
                filename=stamped_overview_name(self.config.filename),
            ),
        )
        self.record_output(ROLE, image)
        # As the overview does: a missing thumbnail is worth far less than the
        # recorded image, so a failure here is logged, not raised.
        try:
            stem = Path(image.filepath).stem if image.filepath else self.config.filename
            thumbnail = write_thumbnail(
                image.filtered_data, self.output_dir / f"{stem}-thumbnail.png"
            )
            self.record_output(f"{ROLE}_thumbnail", thumbnail)
        except Exception as e:  # noqa: BLE001 - the reference is already recorded
            logging.warning(f"Could not write the cleaning thumbnail: {e}")
