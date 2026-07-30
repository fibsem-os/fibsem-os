"""Tiled acquisition: shared geometry, reprojection, and plotting.

The pure layers live here. The acquisition runners, which need a live microscope,
remain in `fibsem.imaging.tiled` for now.
"""

from fibsem.imaging.tiling.geometry import (
    TilePosition,
    compute_tile_grid,
    order_tiles,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling.plotting import (
    plot_minimap,
    plot_stage_positions_on_image,
    plot_tile_positions,
)
from fibsem.imaging.tiling.reprojection import (
    calculate_reprojected_stage_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image,
    reproject_stage_positions_onto_image2,
)

__all__ = [
    "TilePosition",
    "calculate_reprojected_stage_position",
    "calculate_reprojected_stage_position2",
    "compute_tile_grid",
    "order_tiles",
    "plot_minimap",
    "plot_stage_positions_on_image",
    "plot_tile_positions",
    "reproject_stage_positions_onto_image",
    "reproject_stage_positions_onto_image2",
    "validate_tile_stage_positions",
]
