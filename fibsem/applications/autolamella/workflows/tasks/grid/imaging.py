"""Overview grid tasks: thin wrappers over the tiled runners on main.

The operation -- `acquire_beam_overview` -- is a plain function over
`TiledAcquisitionRunner`, callable from the Overview tab or a script. The task
adds only what makes it a workflow step: where the grid is (its calibrated slot,
re-expressed for the requested orientation), where the files go (the grid's own
directory), and what to record (the stitched image and a thumbnail, by role).
"""

from __future__ import annotations

import logging
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Type, Union

import numpy as np

from fibsem import utils
from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
    register_grid_task,
)
from fibsem.imaging.tiled import TiledAcquisitionRunner
from fibsem.microscopes._stage import uncalibrated_message
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

THUMBNAIL_MAX_EDGE = 512

# The output roles a results card, the Overview tab and the agent server read
# (FIB-876). One per beam, plus the thumbnail beside it.
ROLE_BY_BEAM = {BeamType.ELECTRON: "overview_sem", BeamType.ION: "overview_fib"}


def stamped_name(stem: str) -> str:
    """`overview` -> `overview-14-23-05`: the name is a location, and two runs
    called the same thing land on each other (the Overview tab does the same)."""
    return f"{stem}-{utils.current_timestamp_v3(timeonly=True)}"


def write_thumbnail(data: np.ndarray, destination: Union[str, Path]) -> str:
    """Write `data` as a display-sized PNG, atomically. Returns the path written.

    Same contract as a lamella's thumbnail: staged in the destination directory
    and moved into place, so a reader on another thread never sees a partial file.
    """
    from PIL import Image

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=2)
    handle, staged = tempfile.mkstemp(
        dir=str(destination.parent), prefix=".thumbnail-", suffix=".png"
    )
    os.close(handle)
    try:
        thumbnail = Image.fromarray(np.asarray(data).astype(np.uint8))
        thumbnail.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE), Image.LANCZOS)
        thumbnail.save(staged)
        os.replace(staged, str(destination))
    except BaseException:
        try:
            os.remove(staged)
        except OSError:
            pass
        raise
    return str(destination)


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def acquire_beam_overview(
    microscope: "FibsemMicroscope",
    settings: OverviewAcquisitionSettings,
    centre: FibsemStagePosition,
    directory: Union[str, Path],
    stem: str = "overview",
    stop_event=None,
) -> FibsemImage:
    """Acquire a tiled SEM or FIB overview centred on `centre`, saved under `directory`.

    A copy of `settings` is used, so the caller's object is not rewritten with the
    per-run path and stamped name. The beam is `settings.image_settings.beam_type`.
    The runner restores the stage to where it started; `centre` is the grid's
    centre, not a new home.
    """
    settings = deepcopy(settings)
    settings.image_settings.save = True
    settings.image_settings.path = str(directory)
    settings.image_settings.filename = stamped_name(stem)
    Path(directory).mkdir(parents=True, exist_ok=True)
    runner = TiledAcquisitionRunner(
        microscope, settings, stop_event=stop_event, centre_position=centre
    )
    return runner.run_and_stitch()


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


@dataclass
class BeamOverviewGridTaskConfig(GridTaskConfig):
    """A tiled SEM or FIB overview of the grid.

    One task for both beams: the beam is `settings.image_settings.beam_type`, as it
    is everywhere else the overview settings are used, so the Overview tab's settings
    form edits this unchanged. `orientation` is the pose to acquire in; the grid's
    calibrated slot position is re-expressed for it.
    """

    task_type: ClassVar[str] = "BEAM_OVERVIEW_GRID"
    display_name: ClassVar[str] = "Beam overview"
    orientation: str = "SEM"
    settings: OverviewAcquisitionSettings = field(
        default_factory=OverviewAcquisitionSettings
    )
    filename: str = "overview"

    @property
    def beam_type(self) -> BeamType:
        return self.settings.image_settings.beam_type

    @property
    def role(self) -> str:
        return ROLE_BY_BEAM.get(self.beam_type, "overview")


@register_grid_task
class BeamOverviewGridTask(GridTask):
    config_cls: ClassVar[Type[GridTaskConfig]] = BeamOverviewGridTaskConfig
    config: BeamOverviewGridTaskConfig

    def grid_centre(self) -> FibsemStagePosition:
        """The grid's calibrated slot position, in the requested orientation.

        Refuses a grid that is not in a holder slot (nothing has made it reachable)
        and a slot with no calibrated position, with the same message the stage
        gives, rather than acquiring an overview of wherever the stage happens to be.
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

        beam = self.config.beam_type.name
        self.log_status_message(
            "ACQUIRE",
            f"Acquiring {beam} overview: {self.config.settings.nrows} x "
            f"{self.config.settings.ncols} tiles",
        )
        image = acquire_beam_overview(
            self.microscope,
            self.config.settings,
            centre,
            self.output_dir,
            stem=self.config.filename,
            stop_event=self._stop_event,
        )
        self.record_output(self.config.role, image)

        # Beside the stitched image, named after it, so a card can show it without
        # decoding a full-resolution overview. A missing thumbnail is worth far less
        # than a recorded overview, so a failure here is logged, not raised.
        try:
            stem = Path(image.filepath).stem if image.filepath else self.config.filename
            thumbnail = write_thumbnail(
                image.filtered_data, self.output_dir / f"{stem}-thumbnail.png"
            )
            self.record_output(f"{self.config.role}_thumbnail", thumbnail)
        except Exception as e:  # noqa: BLE001 - the overview is already recorded
            logging.warning(f"Could not write the overview thumbnail: {e}")
        self.log_status_message("ACQUIRED", f"{beam} overview recorded")
