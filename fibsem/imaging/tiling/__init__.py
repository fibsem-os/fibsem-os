"""Tiled acquisition: shared geometry, reprojection, and plotting.

The pure layers live here. The acquisition runners, which need a live microscope,
remain in `fibsem.imaging.tiled` for now.
"""

from fibsem.imaging.tiling.geometry import (
    TilePosition,
    compute_tile_grid,
    grid_centre_offset,
    order_tiles,
    raise_if_outside_stage_limits,
    unreachable_tiles,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling.plotting import (
    DEFECT_FAILURE_COLOUR,
    DEFECT_REWORK_COLOUR,
    plot_minimap,
    plot_stage_positions_on_image,
    plot_tile_grid,
    plot_tile_positions,
)
from fibsem.imaging.tiling.reprojection import (
    calculate_reprojected_stage_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image,
    reproject_stage_positions_onto_image2,
)

__all__ = [
    "DEFECT_FAILURE_COLOUR",
    "DEFECT_REWORK_COLOUR",
    "TilePosition",
    "calculate_reprojected_stage_position",
    "calculate_reprojected_stage_position2",
    "compute_tile_grid",
    "grid_centre_offset",
    "order_tiles",
    "plot_minimap",
    "plot_stage_positions_on_image",
    "plot_tile_grid",
    "plot_tile_positions",
    "raise_if_outside_stage_limits",
    "reproject_stage_positions_onto_image",
    "reproject_stage_positions_onto_image2",
    "unreachable_tiles",
    "validate_tile_stage_positions",
]
