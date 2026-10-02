"""The fluorescence overview grid task: a thin wrapper over the FM tiled runner.

The operation -- `acquire_fluorescence_overview` -- is a plain function over
`FMTiledAcquisitionRunner` and `OverviewDestination`, laying files out as the FM
Overview tab does. The task adds the travel to the FM, the objective, the grid's
own directory, and the record: the mosaic and a channel-composite thumbnail.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type, Union

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
from fibsem.cancellation import OperationCancelledError
from fibsem.fm.acquisition import (
    FMTiledAcquisitionRunner,
    OverviewDestination,
    record_fluorescence_image,
)
from fibsem.fm.preview import composite_projection
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.imaging.thumbnail import write_thumbnail
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.imaging.tiling.geometry import (
    compute_tile_grid_from_fov,
    unreachable_tiles,
)
from fibsem.imaging.tiling.progress import (
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.microscopes._stage import uncalibrated_message
from fibsem.projection import FMStageProjection
from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage
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
    limits = getattr(microscope._stage, "limits", None)
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


def acquire_fluorescence_overview(
    microscope: "FibsemMicroscope",
    channels: List[ChannelSettings],
    parameters: OverviewParameters,
    centre: FibsemStagePosition,
    directory: Union[str, Path],
    stem: str = "overview",
    zparams: Optional[ZParameters] = None,
    autofocus_settings: Optional[AutoFocusSettings] = None,
    stop_event=None,
) -> Tuple["FluorescenceImage", Optional[str]]:
    """Acquire a fluorescence tileset centred on `centre`, saved under `directory`.

    Returns the stitched mosaic and where it was written (None if the write
    failed; the mosaic is still returned). The tiles land in a directory of the
    run's name beside the mosaic, as the FM Overview tab lays them out.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    destination = OverviewDestination.create(
        str(directory), stamped_overview_name(stem)
    )
    runner = FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=list(channels),
        overview_parameters=parameters,
        zparams=zparams if parameters.use_zstack else None,
        autofocus_settings=autofocus_settings,
        save_directory=destination.tiles_directory,
        stop_event=stop_event,
        centre_position=centre,
    )

    # The runner reports up to the stitch and leaves the save and the ending to
    # whoever does the save: on the FM Overview tab that is the widget, here it
    # is this function. Without the terminal report the window's status bar
    # stays on "Stitching tiles" after the run has finished.
    def report(status: TiledStatus, error: Optional[str] = None) -> None:
        microscope.tiled_acquisition_signal.emit(
            TiledProgress(status=status, modality=MODALITY_FLUORESCENCE, error=error)
        )

    try:
        mosaic = runner.run_and_stitch()
        report(TiledStatus.SAVING)
        path = destination.save_mosaic(mosaic)
        record_fluorescence_image(microscope, mosaic, overview=parameters)
    except OperationCancelledError:
        report(TiledStatus.CANCELLED)
        raise
    except Exception as e:
        report(TiledStatus.FAILED, error=str(e))
        raise
    report(TiledStatus.FINISHED)
    return mosaic, path


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
        was_inserted = objective.state == "Inserted"

        self.log_status_message("MOVE_TO_FM", f"Moving {self.grid.name} to the FM")
        self.microscope.move_to_device("FM")
        self._check_for_abort()

        if objective.state != "Inserted":
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
            mosaic, saved = acquire_fluorescence_overview(
                self.microscope,
                self.config.channels,
                overview,
                centre,
                self.output_dir,
                stem=self.config.filename,
                zparams=self.config.zparams,
                autofocus_settings=self.config.autofocus_settings,
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
