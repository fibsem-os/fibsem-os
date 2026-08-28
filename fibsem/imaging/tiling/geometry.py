"""Tile grid geometry: layout, ordering, and canvas placement.

Deliberately free of any microscope import. Both the beam tiler and the fluorescence
tiler consume this, and they have already drifted once -- the FM tiler stepped rows in
the opposite direction to the beam tiler, reversing mosaic row order (#226). One
definition is the fix.

Importing this module must not pull in `fibsem.microscope`; that is enforced by test,
not convention, because it is a single convenient import away from being broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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


def grid_centre_offset(tiles: Sequence[TilePosition]) -> Tuple[float, float]:
    """Where the grid's centre sits in the layout's own offsets, in metres.

    The layout measures from the top-left tile, and both runners shift by
    `(n - 1) * step / 2` to make the grid straddle the position it is planned around.
    Taken from the tiles' own extent rather than recomputed from rows, columns and
    overlap, so a caller cannot centre a grid on a step size the layout did not use.

    *tiles* must be the **whole** grid, disabled ones included -- they hold its shape,
    which is what the centring is measured from. Ordering and dropping happen after.
    """
    xs = [t.dx for t in tiles]
    ys = [t.dy for t in tiles]
    if not xs:
        return (0.0, 0.0)
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def unreachable_tiles(
    tiles: Sequence[TilePosition],
    strategy: TileOrderStrategy,
    project: Callable[[float, float], FibsemStagePosition],
    limits: Optional[dict],
) -> List[Tuple[int, int]]:
    """Which of a planned grid's tiles the stage cannot travel to.

    The question :func:`raise_if_outside_stage_limits` answers at acquire time, asked
    early enough to be acted on -- from a pre-flight dialog, or while a grid is being
    dragged. Through the same two helpers the runners use, `order_tiles` to drop the
    disabled tiles and :func:`validate_tile_stage_positions` to test what is left, so
    both sides ask about the same tiles: masking off an unreachable corner is a
    legitimate way to make a grid acquirable, and that only works if they agree.

    The runners keep the authoritative check either way, so a caller here that drifts
    fails **open** -- the run is still refused, exactly as it is today. That is the mild
    direction to be wrong in, and worth more than a validate-only seam in the runners.

    Args:
        tiles: the whole grid from `compute_tile_grid`, disabled tiles included.
        strategy: the traversal. Only bears on which tiles are dropped, not on whether
            any of them is reachable -- gone through anyway so the set asked about here
            is the set the runner will visit, by construction.
        project: displayed-plane offsets from the grid's centre to a stage position,
            i.e. `StageProjection.from_plane` bound to that centre. Note the plane
            measures y **downward** while the layout measures it up, which is why the
            offsets handed over are negated in y. A cached projection rather than
            `microscope.project_*_stable_move`: the two agree exactly, but the
            microscope's re-reads the instrument on every call.
        limits: `microscope._stage.limits`. Falsy means unknown, which reads as "no
            answer" rather than "all fine" -- an empty result either way, but the
            caller should say nothing rather than say it is reachable.

    Returns:
        (row, col) for each tile that will be visited and cannot be reached. Empty if
        the grid is acquirable, if every tile is masked off, or if *limits* is unknown.
    """
    if not limits:
        return []
    ordered = order_tiles(list(tiles), strategy)
    if not ordered:
        return []
    centre_dx, centre_dy = grid_centre_offset(tiles)
    positions = [
        project(tile.dx - centre_dx, -(tile.dy - centre_dy)) for tile in ordered
    ]
    return validate_tile_stage_positions(ordered, positions, limits)


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


# ── planning a grid from selected ground ──────────────────────────────────────
#
# The layout above answers "where do these rows and columns land?". Selecting an area
# to image asks the inverse: "what grid covers this ground, and which of its tiles are
# worth visiting?". Both live here so they cannot drift -- a grid sized by one formula
# and laid out by another is a grid that stops covering what it was asked to.

# `(extent - fov) / step` lands a hair above an integer often enough that `ceil` alone
# adds a whole unwanted row and column to grids that fit exactly -- for 126 of the grid
# sizes swept in `test_an_exact_fit_at_camera_scale_does_not_add_a_row`. Subtracted from
# a tile *count*, so it is dimensionless and holds whatever units the caller works in.
_SPAN_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PlaneRegion:
    """An axis-aligned rectangle in a displayed plane, in metres.

    Measured from the same origin the tile offsets are, with y running **down** -- the
    convention `StageProjection.to_plane` answers in.

    Normalised on construction, so build one with :meth:`from_points` rather than the
    constructor: a drag can end up and to the left of where it started, and a rectangle
    carried through a projection can come out with its corners in any order.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_points(cls, points: Iterable[Tuple[float, float]]) -> "PlaneRegion":
        """The bounding box of any number of points.

        Two for a drag; four for a rectangle carried through a projection, where the
        corners have to be projected and re-bounded rather than the width and height
        scaled -- a foreshortened view squashes one axis, and which axis depends on the
        pose the image was taken at.
        """
        points = list(points)
        if not points:
            raise ValueError("A region needs at least one point.")
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def overlaps(self, other: "PlaneRegion") -> bool:
        """Whether the two rectangles share any area.

        Strict, so rectangles that merely touch along an edge do not count: a tile whose
        edge grazes a region contributes none of it.
        """
        return (
            self.x0 < other.x1
            and self.x1 > other.x0
            and self.y0 < other.y1
            and self.y1 > other.y0
        )


@dataclass(frozen=True)
class GridPlan:
    """A tile grid sized and masked to cover a set of regions.

    Attributes:
        rows: Grid rows.
        cols: Grid columns.
        mask: Per-tile enable mask, `mask[row][col]`, ready for `compute_tile_grid`.
        centre_dx: Where the grid's centre sits, in the same plane the regions are in.
        centre_dy: Likewise. The grid straddles this point; a caller projects it back
            to a stage position to get the runner's `centre_position`.
    """

    rows: int
    cols: int
    mask: List[List[bool]]
    centre_dx: float
    centre_dy: float


def tiles_to_span(extent: float, fov: float, step: float) -> int:
    """The fewest tiles whose combined coverage spans *extent*.

    `n` tiles at `step` apart cover `(n - 1) * step + fov`, so this inverts the layout
    in :func:`compute_tile_grid_from_fov` rather than approximating it.

    Args:
        extent: Length to cover, in metres.
        fov: Length of one tile, in metres.
        step: Distance between adjacent tile centres, in metres.

    Returns:
        The tile count, at least 1.

    Raises:
        ValueError: if *step* or *fov* is not positive.
    """
    if fov <= 0:
        raise ValueError(f"Tile field of view must be positive, got {fov}.")
    if step <= 0:
        raise ValueError(f"Tile step must be positive, got {step}.")
    if extent <= fov:
        return 1
    return int(math.ceil((extent - fov) / step - _SPAN_TOLERANCE)) + 1


def plan_grid_over_regions(
    regions: Sequence[PlaneRegion],
    fov_x: float,
    fov_y: float,
    overlap: float,
) -> GridPlan:
    """Size a tile grid to a set of regions, and mask it to the tiles they touch.

    Pure function -- no microscope, no projections, no side effects.

    The grid is sized to the regions' **combined** bounding box, so two regions far
    apart produce a grid spanning both. That is deliberate: one selection is one
    acquisition, and the mask -- not the grid -- is what stops the ground between them
    being visited. It does mean a scattered selection still costs a mosaic spanning it,
    since a disabled tile keeps its place in the canvas.

    The bounding box is not a whole number of steps, so the grid overhangs the regions
    by up to a step in each axis. What the caller acquires is therefore the tiles, not
    the rectangles that produced them.

    Args:
        regions: Rectangles to cover, in one plane, in metres. Regions with no area are
            ignored -- a click that never became a drag selects nothing.
        fov_x: Tile width, in metres.
        fov_y: Tile height, in metres.
        overlap: Fractional overlap between adjacent tiles, in [0, 1).

    Returns:
        A :class:`GridPlan`.

    Raises:
        ValueError: if no region has any area, or *overlap* is out of range.
    """
    covering = [r for r in regions if r.width > 0 and r.height > 0]
    if not covering:
        raise ValueError("No region with any area to cover.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"Overlap must be in [0, 1), got {overlap}.")

    bounds = PlaneRegion.from_points(
        [(r.x0, r.y0) for r in covering] + [(r.x1, r.y1) for r in covering]
    )
    step_x = fov_x * (1.0 - overlap)
    step_y = fov_y * (1.0 - overlap)
    cols = tiles_to_span(bounds.width, fov_x, step_x)
    rows = tiles_to_span(bounds.height, fov_y, step_y)

    centre_dx = (bounds.x0 + bounds.x1) / 2.0
    centre_dy = (bounds.y0 + bounds.y1) / 2.0

    # Tile (row, col) straddles the same offset `compute_tile_grid_from_fov` gives it
    # once the grid is centred: `dx - (cols - 1) * step_x / 2` and its row equivalent,
    # with the row term negated back because that layout measures y upward and a
    # displayed plane measures it down. Row 0 is therefore the smallest y, which is what
    # decides whether a mask lands on the ground it was drawn over or mirrored across it.
    mask: List[List[bool]] = []
    for row in range(rows):
        tile_y = centre_dy + (row - (rows - 1) / 2.0) * step_y
        enabled = []
        for col in range(cols):
            tile_x = centre_dx + (col - (cols - 1) / 2.0) * step_x
            tile = PlaneRegion(
                tile_x - fov_x / 2.0,
                tile_y - fov_y / 2.0,
                tile_x + fov_x / 2.0,
                tile_y + fov_y / 2.0,
            )
            enabled.append(any(tile.overlaps(region) for region in covering))
        mask.append(enabled)

    return GridPlan(
        rows=rows, cols=cols, mask=mask, centre_dx=centre_dx, centre_dy=centre_dy
    )
