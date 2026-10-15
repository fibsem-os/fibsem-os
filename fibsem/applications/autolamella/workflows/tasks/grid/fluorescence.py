"""The fluorescence overview grid task: a thin wrapper over the FM tiled runner.

The operation is `fibsem.fm.acquisition.acquire_fm_tiled_overview`, which the FM
Overview tab and the imaging service call too. The task adds the travel to the
FM, the objective, the grid's own directory, and the record: the mosaic and a
channel-composite thumbnail.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type

from fibsem import timing
from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    disable_tiles,
)
from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
    register_grid_task,
)
from fibsem.autofunctions.autofocus import AutoFocusSettings
from fibsem.fm.acquisition import OverviewDestination, acquire_fm_tiled_overview
from fibsem.fm.preview import composite_projection
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.fm.timing import estimate_fm_overview_time
from fibsem.geometry.projection import FMStageProjection
from fibsem.imaging.thumbnail import write_thumbnail
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.imaging.tiling.geometry import (
    compute_tile_grid_from_fov,
    unreachable_tiles,
)
from fibsem.services.sample_handling import uncalibrated_message
from fibsem.structures import FibsemStagePosition, InsertableDeviceState

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# ---------------------------------------------------------------------------


def reachable_fluorescence_overview(
    microscope: "FibsemMicroscope",
    parameters: OverviewParameters,
    centre: FibsemStagePosition,
    projection: Optional[FMStageProjection] = None,
) -> Tuple[OverviewParameters, List[Tuple[int, int]]]:
    """*parameters* with the tiles the stage cannot reach from *centre* turned off,
    and which tiles they were: `reachable_overview` for the fluorescence tiler.

    The tiles are laid out as `FMTiledAcquisitionRunner` lays them out, from the
    camera's field of view, and judged through the projection the FM Overview tab
    draws unreachable tiles with. Read after the move to the FM, with the objective
    in, when the camera's pixel size is the one the run will use.

    *projection* is one already built, by a caller asking repeatedly (the Grid
    page, on every edit). It carries the camera's pixel size and shape, so the
    camera is not read again; building one reads it, which is slow on hardware.
    """
    stage = microscope.stage
    limits = stage.position.limits if stage is not None else None
    if projection is None:
        projection = FMStageProjection.from_microscope(microscope)
    if not limits or projection is None:
        return parameters, []
    height, width = projection.shape
    tiles = compute_tile_grid_from_fov(
        nrows=parameters.rows,
        ncols=parameters.cols,
        fov_x=width * projection.pixel_size,
        fov_y=height * projection.pixel_size,
        image_width=width,
        image_height=height,
        overlap=parameters.overlap,
        mask=parameters.tile_mask,
    )
    skipped = unreachable_tiles(
        tiles,
        parameters.tile_order,
        lambda dx, dy: projection.from_plane(dx, dy, centre),
        limits,
    )
    if not skipped:
        return parameters, []
    parameters = deepcopy(parameters)
    parameters.tile_mask = disable_tiles(
        parameters.tile_mask, parameters.rows, parameters.cols, skipped
    )
    return parameters, skipped


@dataclass
class FluorescenceOverviewGridTaskConfig(GridTaskConfig):
    """A tiled fluorescence overview of the grid: the FM Overview tab's inputs."""

    task_type: ClassVar[str] = "FM_OVERVIEW_GRID"
    display_name: ClassVar[str] = "Fluorescence overview"
    channels: List[ChannelSettings] = field(
        default_factory=lambda: [ChannelSettings(name="Channel-01")]
    )
    # `overview`, not `parameters`: the base config's `parameters` is the list
    # of a form's fields, and a field of that name would shadow it.
    overview: OverviewParameters = field(default_factory=OverviewParameters)
    zparams: ZParameters = field(default_factory=ZParameters)
    autofocus_settings: Optional[AutoFocusSettings] = None
    filename: str = "overview"

    role: ClassVar[str] = "overview_fm"

    @property
    def estimated_duration(self) -> float:
        """What ``_run`` does, in order: the move to the FM, the objective in, the
        overview (`estimate_fm_overview_time`, which counts the runner's return of
        the stage and objective), the objective out. Inserting and retracting are
        always counted, though the task skips both when the objective is already
        in: which it is, is runtime state. Errs long.
        """
        if self.overview.n_enabled_tiles <= 0:
            return 0.0
        total = estimate_fm_overview_time(
            self.channels,
            self.overview,
            zparams=self.zparams,
            autofocus_settings=self.autofocus_settings,
        )
        total += timing.stage_move_cost(1) + timing.OBJECTIVE_INSERT_S
        total += timing.OBJECTIVE_RETRACT_S
        return total


@register_grid_task
class FluorescenceOverviewGridTask(GridTask):
    """Move to the FM, put the objective in, acquire, record, and put things back.

    The objective is inserted for the run if it was not already, and returned to
    how it was found afterwards: a grid exchange with the objective in is not a
    thing to leave possible by accident.
    """

    config_cls: ClassVar[Type[GridTaskConfig]] = FluorescenceOverviewGridTaskConfig
    config: FluorescenceOverviewGridTaskConfig

    def grid_centre(self) -> FibsemStagePosition:
        """The grid's calibrated slot position, as the FM sees it.

        One call on either mounting (`to_device`): a compustage gets the flip, an
        offset mount gets the traverse, and the pose is kept where the objective
        images from it and otherwise put into the first orientation it declares.
        """
        slot = self.slot
        if slot is None:
            raise RuntimeError(
                f"Grid '{self.grid.name}' is not in a holder slot. Load it first."
            )
        if slot.position is None:
            raise RuntimeError(uncalibrated_message(slot.name))
        return self.microscope.to_device(slot.position, "FM")

    def _run(self) -> None:
        fm = getattr(self.microscope, "fm", None)
        if fm is None:
            raise RuntimeError("This system has no fluorescence microscope.")
        centre = self.grid_centre()

        # Read before the travel: on a compustage `move_to_device("FM")` inserts the
        # objective itself, and "how it was found" means before any of this ran.
        objective = fm.objective
        was_inserted = objective.state is InsertableDeviceState.INSERTED

        self.log_status_message("MOVE_TO_FM", f"Moving {self.grid.name} to the FM")
        self.microscope.move_to_device("FM")
        self._check_for_abort()

        if objective.state is not InsertableDeviceState.INSERTED:
            self.log_status_message("INSERT_OBJECTIVE", "Inserting the objective")
            objective.insert()
        try:
            overview, skipped = reachable_fluorescence_overview(
                self.microscope, self.config.overview, centre
            )
            enabled = self.config.overview.n_enabled_tiles
            if skipped and overview.n_enabled_tiles == 0:
                raise RuntimeError(
                    f"None of the {enabled} tiles of this overview is within the "
                    f"stage's reach from the centre of {self.grid.name}. Make it "
                    "smaller."
                )
            self.note_skipped_tiles(skipped, enabled)
            out_of_reach = f", {len(skipped)} out of reach" if skipped else ""
            self.log_status_message(
                "ACQUIRE",
                f"Acquiring fluorescence overview: {overview.rows} x "
                f"{overview.cols} tiles{out_of_reach}, "
                f"{len(self.config.channels)} channel(s)",
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            mosaic, saved = acquire_fm_tiled_overview(
                self.microscope,
                self.config.channels,
                overview,
                OverviewDestination.create(
                    str(self.output_dir), stamped_overview_name(self.config.filename)
                ),
                zparams=self.config.zparams if overview.use_zstack else None,
                autofocus_settings=self.config.autofocus_settings,
                centre_position=centre,
                stop_event=self._stop_event,
            )
            if saved is None:
                raise RuntimeError("The fluorescence overview could not be saved.")
            self.record_output(self.config.role, saved)
            try:
                thumbnail = write_thumbnail(
                    composite_projection(mosaic),
                    self.output_dir / f"{Path(saved).name.split('.')[0]}-thumbnail.png",
                )
                self.record_output(f"{self.config.role}_thumbnail", thumbnail)
            except Exception as e:  # noqa: BLE001 - the overview is already recorded
                logging.warning(f"Could not write the fluorescence thumbnail: {e}")
        finally:
            if not was_inserted:
                self.log_status_message("RETRACT_OBJECTIVE", "Retracting the objective")
                objective.retract()
        self.log_status_message("ACQUIRED", "Fluorescence overview recorded")
