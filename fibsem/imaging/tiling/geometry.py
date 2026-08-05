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
from typing import Optional, Sequence

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
        enabled: Whether this tile is to be acquired. Disabled tiles keep their place
            in the grid -- they still define the canvas extent and the traversal
            pattern -- but are not visited. See :func:`order_tiles`.
    """
    row: int
    col: int
    dx: float
    dy: float
    canvas_x: int
    canvas_y: int
    enabled: bool = True


def _validate_mask(mask, nrows: int, ncols: int) -> None:
    """Reject a mask whose shape does not match the grid.

    A mask is positional: `mask[row][col]`. Silently tolerating the wrong shape
    would skip or acquire the wrong tiles, which is invisible in the result --
    a mosaic of zeros looks the same whether the tile was skipped by mistake or
    the sample was dark there.
    """
    if len(mask) != nrows:
        raise ValueError(
            f"Tile mask has {len(mask)} rows, but the grid has {nrows}."
        )
    for i, row in enumerate(mask):
        if len(row) != ncols:
            raise ValueError(
                f"Tile mask row {i} has {len(row)} columns, but the grid has {ncols}."
            )

def compute_tile_grid(
    settings: OverviewAcquisitionSettings,
    mask: Optional[Sequence[Sequence[bool]]] = None,
) -> list[TilePosition]:
    """Compute physical and canvas positions for every tile in the grid.

    Pure function — no microscope, no side effects.

    Args:
        settings: Overview acquisition settings (hfw, resolution, nrows, ncols, overlap).
        mask: Optional per-tile enable mask, `mask[row][col]`. None enables everything.
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
        mask=mask,
    )


def compute_tile_grid_from_fov(
    nrows: int,
    ncols: int,
    fov_x: float,
    fov_y: float,
    image_width: int,
    image_height: int,
    overlap: float,
    mask: Optional[Sequence[Sequence[bool]]] = None,
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
        mask: Optional per-tile enable mask, `mask[row][col]`. None enables everything.
            Disabled tiles are still returned -- they hold the grid's shape, so the
            canvas extent and the traversal pattern do not change when tiles are
            skipped. Filtering happens in :func:`order_tiles`.

    Returns:
        Flat list of TilePosition objects in row-major order (top-left first),
        including disabled ones.
    """
    if mask is not None:
        _validate_mask(mask, nrows, ncols)

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
                # bool(): a numpy mask yields np.bool_, which does not survive
                # yaml.safe_dump when the grid is recorded alongside the mosaic.
                enabled=True if mask is None else bool(mask[i][j]),
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
    """Put tiles in traversal order, dropping the disabled ones.

    Pure function — no microscope, no side effects.

    Order first, filter second, and in that order deliberately. The traversal is
    derived from the **full** grid extent and disabled tiles are removed from the
    resulting sequence, so a sparse acquisition follows the same path the dense one
    would have and simply misses stops along it. Filtering first would re-derive the
    pattern over the holes: a spiral would wind around the enabled tiles' bounding
    box instead of the grid's centre, and serpentine row parity would flip if a whole
    row were disabled -- a different, and usually longer, stage path.

    This is why `compute_tile_grid*` returns disabled tiles rather than omitting them.

    Args:
        tiles: Flat list of TilePosition objects (any order), including disabled ones.
        strategy: TYPEWRITER, SERPENTINE, or SPIRAL.
    Returns:
        New list of the enabled tiles, in traversal order.
    """
    if strategy is TileOrderStrategy.SPIRAL:
        nrows = max(t.row for t in tiles) + 1
        ncols = max(t.col for t in tiles) + 1
        tile_map = {(t.row, t.col): t for t in tiles}
        ordered = [tile_map[rc] for rc in _spiral_order(nrows, ncols) if rc in tile_map]
        return [t for t in ordered if t.enabled]

    rows = sorted(set(t.row for t in tiles))
    result = []
    for row_idx, row in enumerate(rows):
        row_tiles = sorted([t for t in tiles if t.row == row], key=lambda t: t.col)
        if strategy is TileOrderStrategy.SERPENTINE and row_idx % 2 == 1:
            row_tiles = list(reversed(row_tiles))
        result.extend(t for t in row_tiles if t.enabled)
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


def raise_if_outside_stage_limits(
    ordered: list[TilePosition],
    tile_stage_positions: list[FibsemStagePosition],
    limits: dict,
) -> None:
    """Refuse a grid the stage cannot reach, naming the tiles that put it there.

    Rejected rather than trimmed. A tileset silently missing the tiles the stage ran out
    of travel for is a mosaic that misrepresents the sample: the gaps stitch as zeros,
    indistinguishable from dark sample, and anything downstream consuming it -- targeting,
    correlation -- takes them for data. Better to say so before the run starts, while the
    grid can still be moved.

    Only the tiles that will actually be visited are checked: `ordered` has already had
    the disabled ones dropped, so masking off a corner that falls outside the limits is a
    legitimate way to make a grid acquirable.

    Shared by both tilers so they refuse the same grids for the same reason -- the FM
    runner did not check at all, which is the shape of bug this core exists to prevent.

    Raises:
        ValueError: if any tile is outside *limits*.
    """
    out_of_bounds = validate_tile_stage_positions(ordered, tile_stage_positions, limits)
    if not out_of_bounds:
        return
    details = ", ".join(f"({r},{c})" for r, c in out_of_bounds)
    raise ValueError(
        f"Acquisition grid extends beyond stage limits. "
        f"{len(out_of_bounds)} tile(s) out of bounds: {details}"
    )
