"""Tile grid geometry: layout, ordering, and canvas placement.

Deliberately free of any microscope import. Both the beam tiler and the fluorescence
tiler consume this, and they have already drifted once -- the FM tiler stepped rows in
the opposite direction to the beam tiler, reversing mosaic row order (#226). One
definition is the fix.

Importing this module must not pull in `fibsem.microscope`; that is enforced by test,
not convention, because it is a single convenient import away from being broken.
"""

from __future__ import annotations

from dataclasses import dataclass

from fibsem.structures import (
    FibsemStagePosition,
    OverviewAcquisitionSettings,
    TileOrderStrategy,
)


@dataclass
class TilePosition:
    """Physical and canvas coordinates for one tile in a tiled acquisition grid.

    Attributes:
        row: Grid row index (0 = top).
        col: Grid column index (0 = left).
        dx: X offset from start_position in metres; positive = right.
        dy: Y offset from start_position in metres; negative = down (stage y inverted).
        canvas_x: Pixel left edge in the stitched canvas array.
        canvas_y: Pixel top edge in the stitched canvas array.
    """
    row: int
    col: int
    dx: float
    dy: float
    canvas_x: int
    canvas_y: int

def compute_tile_grid(settings: OverviewAcquisitionSettings) -> list[TilePosition]:
    """Compute physical and canvas positions for every tile in the grid.

    Pure function — no microscope, no side effects.

    Args:
        settings: Overview acquisition settings (hfw, resolution, nrows, ncols, overlap).
    Returns:
        Flat list of TilePosition objects in row-major order (top-left first).
    """
    image_width, image_height = settings.image_settings.resolution
    tile_fov_x = settings.image_settings.hfw
    tile_fov_y = tile_fov_x * (image_height / image_width)

    return compute_tile_grid_from_fov(
        nrows=settings.nrows,
        ncols=settings.ncols,
        fov_x=tile_fov_x,
        fov_y=tile_fov_y,
        image_width=image_width,
        image_height=image_height,
        overlap=settings.overlap,
    )


def compute_tile_grid_from_fov(
    nrows: int,
    ncols: int,
    fov_x: float,
    fov_y: float,
    image_width: int,
    image_height: int,
    overlap: float,
) -> list[TilePosition]:
    """Compute the tile grid from a field of view given directly.

    The same layout as :func:`compute_tile_grid`, for callers that do not have an
    `OverviewAcquisitionSettings`. A fluorescence camera has a real field of view in
    both axes -- pixel size times resolution -- rather than a horizontal field width
    with the vertical inferred from the aspect ratio, so it cannot go through the
    beam-shaped settings object without inventing an `hfw`.

    Keeping one implementation matters more than the convenience: the beam tiler and
    the fluorescence tiler stepping rows in opposite directions is what produced a
    reversed mosaic (#226), and separate layout code is how that happened.

    Args:
        nrows: Number of tile rows.
        ncols: Number of tile columns.
        fov_x: Width of one tile, in metres.
        fov_y: Height of one tile, in metres.
        image_width: Tile width in pixels.
        image_height: Tile height in pixels.
        overlap: Fractional overlap between adjacent tiles.

    Returns:
        Flat list of TilePosition objects in row-major order (top-left first).
    """
    dx_step = fov_x * (1 - overlap)
    dy_step = fov_y * (1 - overlap)

    eff_w = max(1, int(round(image_width  * (1 - overlap))))
    eff_h = max(1, int(round(image_height * (1 - overlap))))

    tiles = []
    for i in range(nrows):
        for j in range(ncols):
            tiles.append(TilePosition(
                row=i, col=j,
                dx=j * dx_step,
                dy=-(i * dy_step),   # negate: stage y axis is inverted
                canvas_x=j * eff_w,
                canvas_y=i * eff_h,
            ))
    return tiles

def _spiral_order(nrows: int, ncols: int) -> list[tuple[int, int]]:
    """Return (row, col) pairs in a clockwise outward spiral from the centre tile.

    Works for any grid shape, including non-square and single-row/column grids.
    The traversal position may temporarily leave the grid bounds while stepping;
    only cells inside [0, nrows) × [0, ncols) are included in the result.
    """
    cr, cc = nrows // 2, ncols // 2
    result: list[tuple[int, int]] = [(cr, cc)]
    r, c = cr, cc
    # right, down, left, up
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0
    steps = 1
    total = nrows * ncols
    # Upper bound on iterations: spiral arms can't exceed grid perimeter
    max_steps = nrows + ncols + 2

    while len(result) < total and steps <= max_steps:
        for _ in range(2):
            dr, dc = dirs[dir_idx % 4]
            for _ in range(steps):
                r += dr
                c += dc
                if 0 <= r < nrows and 0 <= c < ncols:
                    result.append((r, c))
            dir_idx += 1
            if len(result) >= total:
                return result
        steps += 1

    return result

def order_tiles(tiles: list[TilePosition], strategy: TileOrderStrategy) -> list[TilePosition]:
    """Reorder tiles according to the movement strategy.

    Pure function — no microscope, no side effects.

    Args:
        tiles: Flat list of TilePosition objects (any order).
        strategy: TYPEWRITER, SERPENTINE, or SPIRAL.
    Returns:
        New list with tiles in traversal order.
    """
    if strategy is TileOrderStrategy.SPIRAL:
        nrows = max(t.row for t in tiles) + 1
        ncols = max(t.col for t in tiles) + 1
        tile_map = {(t.row, t.col): t for t in tiles}
        return [tile_map[rc] for rc in _spiral_order(nrows, ncols) if rc in tile_map]

    rows = sorted(set(t.row for t in tiles))
    result = []
    for row_idx, row in enumerate(rows):
        row_tiles = sorted([t for t in tiles if t.row == row], key=lambda t: t.col)
        if strategy is TileOrderStrategy.SERPENTINE and row_idx % 2 == 1:
            row_tiles = list(reversed(row_tiles))
        result.extend(row_tiles)
    return result

def validate_tile_stage_positions(
    ordered: list[TilePosition],
    tile_stage_positions: list[FibsemStagePosition],
    limits: dict,
) -> list[tuple[int, int]]:
    """Return (row, col) pairs for any tile positions that exceed stage limits.

    Args:
        ordered: Tile grid positions in acquisition order.
        tile_stage_positions: Projected stage position for each tile (same order).
        limits: Dict[str, RangeLimit] from microscope._stage.limits.
    Returns:
        List of (row, col) tuples for out-of-bounds tiles (empty if all OK).
    """
    return [
        (tile.row, tile.col)
        for tile, sp in zip(ordered, tile_stage_positions)
        if not sp.is_within_limits(limits, axes=["x", "y"])
    ]
