"""Which tiles of a planned overview the stage can reach, asked before it runs.

The runners refuse a grid with any tile past the stage's travel, but only once the
run has started (`raise_if_outside_stage_limits`). These ask the same question from a
centre and a projection beforehand, through `unreachable_tiles`, for the grid tasks
(which turn the unreachable tiles off and acquire the rest), the Protocol
tab's Grid page and the imaging service's overview estimates. Reach is judged by tile
centre, as the Overview tabs draw it.

Apart from `geometry`, which is kept free of any microscope import: these read the
stage's limits and build a projection from the microscope.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from fibsem.geometry.projection import BeamStageProjection, FMStageProjection
from fibsem.imaging.tiling.geometry import (
    compute_tile_grid,
    compute_tile_grid_from_fov,
    unreachable_tiles,
)
from fibsem.structures.settings import FibsemStagePosition, OverviewAcquisitionSettings

if TYPE_CHECKING:
    from fibsem.fm.structures import OverviewParameters
    from fibsem.microscope import FibsemMicroscope


def disable_tiles(
    mask: Optional[List[List[bool]]],
    nrows: int,
    ncols: int,
    tiles: Iterable[Tuple[int, int]],
) -> List[List[bool]]:
    """A new tile mask: *mask* (None: every tile on) with each (row, col) in *tiles*
    turned off. How the tiles `unreachable_tiles` names are dropped, so that the
    runner's `raise_if_outside_stage_limits` then passes on what is left."""
    out = (
        [[True] * ncols for _ in range(nrows)]
        if mask is None
        else [list(row) for row in mask]
    )
    for row, col in tiles:
        out[row][col] = False
    return out


def reachable_overview(
    microscope: "FibsemMicroscope",
    settings: OverviewAcquisitionSettings,
    centre: FibsemStagePosition,
    projection: Optional[BeamStageProjection] = None,
) -> Tuple[OverviewAcquisitionSettings, List[Tuple[int, int]]]:
    """*settings* with the tiles the stage cannot reach from *centre* turned off, and
    which tiles they were.

    The tile runner refuses a grid with any tile past the stage's travel, but only at
    acquire time (`raise_if_outside_stage_limits`). For a grid overview that is after
    the exchange, and again on every grid, so the tiles out of reach are
    turned off first and the rest acquired. It is the question the Overview tab asks
    before it draws a tile as unreachable, through the same helper and projection, so
    the runner's own check passes on what is left. Reach is judged by tile centre, as
    there. Returns *settings* unchanged when every tile is reachable or the limits
    cannot be read.

    *projection* is one already built for this beam, by a caller asking repeatedly
    (the Grid page, on every edit): building it reads the scan rotation.
    """
    stage = microscope.stage
    limits = stage.position.limits if stage is not None else None
    if projection is None:
        projection = BeamStageProjection.from_microscope(
            microscope, settings.image_settings.beam_type
        )
    if not limits or projection is None:
        return settings, []
    skipped = unreachable_tiles(
        compute_tile_grid(settings, mask=settings.tile_mask),
        settings.tile_order,
        lambda dx, dy: projection.from_plane(dx, dy, centre),
        limits,
    )
    if not skipped:
        return settings, []
    settings = deepcopy(settings)
    settings.tile_mask = disable_tiles(
        settings.tile_mask, settings.nrows, settings.ncols, skipped
    )
    return settings, skipped


def reachable_fluorescence_overview(
    microscope: "FibsemMicroscope",
    parameters: "OverviewParameters",
    centre: FibsemStagePosition,
    projection: Optional[FMStageProjection] = None,
) -> Tuple["OverviewParameters", List[Tuple[int, int]]]:
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
