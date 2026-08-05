"""Sparse tile grids: the enable mask, and the order-then-filter rule.

A sparse acquisition visits a subset of the grid. Two properties have to hold or the
result is silently wrong rather than loudly broken:

* **Canvas placement is unaffected by the mask.** An acquired tile lands at exactly the
  canvas coordinates it would have had in the dense mosaic, so a sparse overview can be
  compared with, or later filled in from, a dense one.
* **The traversal is the dense traversal with stops removed**, not a pattern re-derived
  over the holes -- see `order_tiles`.
"""

import math

import numpy as np
import pytest

from fibsem.imaging.tiling.geometry import (
    compute_tile_grid_from_fov,
    order_tiles,
)
from fibsem.structures import TileOrderStrategy

GRID_KWARGS = dict(fov_x=1.0, fov_y=1.0, image_width=100, image_height=100, overlap=0.0)


def make_grid(n, predicate=None):
    """An n x n grid; `predicate(row, col)` decides which tiles are enabled."""
    mask = None if predicate is None else [[predicate(i, j) for j in range(n)] for i in range(n)]
    return compute_tile_grid_from_fov(nrows=n, ncols=n, mask=mask, **GRID_KWARGS)


def rc(tiles):
    return [(t.row, t.col) for t in tiles]


def travel(tiles):
    """Total stage path length through the tiles, in grid units."""
    return sum(math.dist((a.dx, a.dy), (b.dx, b.dy)) for a, b in zip(tiles, tiles[1:]))


# ── the mask itself ──────────────────────────────────────────────────────


def test_no_mask_enables_everything():
    assert all(t.enabled for t in make_grid(4))


def test_disabled_tiles_are_still_returned():
    """They hold the grid's shape; `order_tiles` is what drops them."""
    tiles = make_grid(4, lambda i, j: i == j)

    assert len(tiles) == 16
    assert sum(t.enabled for t in tiles) == 4


def test_mask_is_indexed_row_then_col():
    """A transposed mask would be a silent, symmetric-looking wrong answer."""
    tiles = {(t.row, t.col): t for t in make_grid(3, lambda i, j: (i, j) == (0, 2))}

    assert tiles[(0, 2)].enabled
    assert not tiles[(2, 0)].enabled


def test_a_numpy_mask_yields_plain_bools():
    """np.bool_ does not survive yaml.safe_dump, and the mask gets recorded."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    tiles = compute_tile_grid_from_fov(nrows=3, ncols=3, mask=mask, **GRID_KWARGS)

    assert all(type(t.enabled) is bool for t in tiles)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        ([[True] * 3] * 2, "2 rows"),        # too few rows
        ([[True] * 2] * 3, "2 columns"),     # too few columns
        ([[True] * 3] * 4, "4 rows"),        # too many rows
    ],
)
def test_a_mask_of_the_wrong_shape_is_rejected(mask, match):
    """Loudly, because the failure mode is invisible: a skipped tile and a dark tile
    look identical in the mosaic."""
    with pytest.raises(ValueError, match=match):
        compute_tile_grid_from_fov(nrows=3, ncols=3, mask=mask, **GRID_KWARGS)


# ── canvas placement ─────────────────────────────────────────────────────


@pytest.mark.parametrize("overlap", [0.0, 0.1, 0.25])
def test_sparse_tiles_land_where_the_dense_ones_would(overlap):
    """The headline property. Masking changes which tiles are acquired, never where."""
    kwargs = dict(GRID_KWARGS, overlap=overlap)
    predicate = lambda i, j: (i + j) % 2 == 0  # noqa: E731

    dense = {(t.row, t.col): t for t in compute_tile_grid_from_fov(nrows=4, ncols=4, **kwargs)}
    mask = [[predicate(i, j) for j in range(4)] for i in range(4)]
    sparse = compute_tile_grid_from_fov(nrows=4, ncols=4, mask=mask, **kwargs)

    for tile in sparse:
        reference = dense[(tile.row, tile.col)]
        assert (tile.canvas_x, tile.canvas_y) == (reference.canvas_x, reference.canvas_y)
        assert (tile.dx, tile.dy) == (reference.dx, reference.dy)


# ── ordering ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", list(TileOrderStrategy))
def test_ordering_returns_only_enabled_tiles(strategy):
    tiles = make_grid(5, lambda i, j: i == 2)

    ordered = order_tiles(tiles, strategy)

    assert len(ordered) == 5
    assert all(t.enabled for t in ordered)


@pytest.mark.parametrize("strategy", list(TileOrderStrategy))
@pytest.mark.parametrize(
    "predicate",
    [
        lambda i, j: (i + j) % 2 == 0,      # checkerboard
        lambda i, j: i < 3 and j < 3,       # off-centre block
        lambda i, j: i != 1,                # a whole row missing
        lambda i, j: (i, j) != (2, 2),      # a single hole
    ],
    ids=["checkerboard", "off-centre-block", "whole-row-missing", "single-hole"],
)
def test_the_sparse_order_is_the_dense_order_with_stops_removed(strategy, predicate):
    """The rule, stated as one property over several mask shapes."""
    dense = order_tiles(make_grid(5), strategy)
    sparse = order_tiles(make_grid(5, predicate), strategy)

    assert rc(sparse) == [(r, c) for r, c in rc(dense) if predicate(r, c)]


def test_spiral_keeps_the_grid_centre_when_the_enabled_block_is_off_centre():
    """Re-deriving would spiral around the block's centre instead of the grid's.

    Concretely: the enabled top-left 3x3 of a 5x5 starts at the grid centre (2,2) and
    winds outward from there. A spiral computed over the block alone would start at
    (1,1) -- a different pattern, and one that no longer means "outward from where the
    stage already is".
    """
    tiles = make_grid(5, lambda i, j: i < 3 and j < 3)

    ours = order_tiles(tiles, TileOrderStrategy.SPIRAL)
    re_derived = order_tiles([t for t in tiles if t.enabled], TileOrderStrategy.SPIRAL)

    assert (ours[0].row, ours[0].col) == (2, 2)
    assert (re_derived[0].row, re_derived[0].col) == (1, 1)
    assert rc(ours) != rc(re_derived)


def test_serpentine_parity_follows_the_grid_row_not_the_visit_count():
    """Known consequence of the order-then-filter rule, recorded rather than asserted good.

    With row 1 absent, rows 0 and 2 are both even-parity grid rows, so both run
    left-to-right and the stage jumps back across the grid between them. Re-deriving
    parity over the visited rows would reverse row 2 and save that jump.

    This is the one case where the rule costs travel. It is kept because the rule is
    uniform and because the same rule is what makes SPIRAL correct; if the travel
    matters more than the uniformity, this is the test to change.
    """
    tiles = make_grid(4, lambda i, j: i != 1)

    ours = order_tiles(tiles, TileOrderStrategy.SERPENTINE)
    re_derived = order_tiles([t for t in tiles if t.enabled], TileOrderStrategy.SERPENTINE)

    assert rc(ours)[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (2, 0)]
    assert rc(re_derived)[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (2, 3)]
    assert travel(ours) > travel(re_derived)


def test_an_empty_mask_yields_no_tiles_to_visit():
    """The grid still exists -- the canvas keeps its size -- but nothing is acquired."""
    tiles = make_grid(3, lambda i, j: False)

    assert len(tiles) == 9
    assert order_tiles(tiles, TileOrderStrategy.TYPEWRITER) == []
