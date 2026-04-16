import pytest
from matplotlib.figure import Figure

from fibsem.imaging.tiled import TilePosition, compute_tile_grid, order_tiles, plot_tile_positions
from fibsem.structures import ImageSettings, OverviewAcquisitionSettings, TileOrderStrategy


# ---------------------------------------------------------------------------
# compute_tile_grid
# ---------------------------------------------------------------------------

def _make_settings(nrows, ncols, hfw=100e-6, resolution=(1024, 1024), overlap=0.0):
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=resolution, hfw=hfw),
        nrows=nrows, ncols=ncols, overlap=overlap,
    )


def test_compute_tile_grid_count():
    s = _make_settings(3, 4)
    tiles = compute_tile_grid(s)
    assert len(tiles) == 12


def test_compute_tile_grid_row_major_order():
    """Tiles returned in row-major (i, j) order."""
    s = _make_settings(2, 3)
    tiles = compute_tile_grid(s)
    indices = [(t.row, t.col) for t in tiles]
    assert indices == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_compute_tile_grid_top_left_at_origin():
    """Tile (0, 0) is at dx=0, dy=0 (top-left corner = start_position)."""
    s = _make_settings(3, 3)
    tiles = compute_tile_grid(s)
    t00 = next(t for t in tiles if t.row == 0 and t.col == 0)
    assert t00.dx == pytest.approx(0.0)
    assert t00.dy == pytest.approx(0.0)


def test_compute_tile_grid_dx_no_overlap():
    """With no overlap, dx step == hfw."""
    hfw = 150e-6
    s = _make_settings(1, 3, hfw=hfw, overlap=0.0)
    tiles = compute_tile_grid(s)
    assert tiles[1].dx == pytest.approx(hfw)
    assert tiles[2].dx == pytest.approx(2 * hfw)


def test_compute_tile_grid_dy_no_overlap():
    """With no overlap, dy step == tile_fov_y (negative, downward)."""
    hfw = 100e-6
    s = _make_settings(3, 1, hfw=hfw, resolution=(1024, 1024), overlap=0.0)
    tiles = compute_tile_grid(s)
    assert tiles[1].dy == pytest.approx(-hfw)   # row 1, one step down
    assert tiles[2].dy == pytest.approx(-2 * hfw)


def test_compute_tile_grid_with_overlap():
    """Overlap reduces the step size."""
    hfw = 100e-6
    overlap = 0.2
    s = _make_settings(2, 2, hfw=hfw, overlap=overlap)
    tiles = compute_tile_grid(s)
    step = hfw * (1 - overlap)
    assert tiles[1].dx == pytest.approx(step)    # col 1
    assert tiles[2].dy == pytest.approx(-step)   # row 1 (index 2 in row-major for 2x2)


def test_compute_tile_grid_non_square_dy():
    """Non-square tiles: dy_step scaled by aspect ratio."""
    hfw = 150e-6
    w, h = 1536, 1024
    s = _make_settings(2, 1, hfw=hfw, resolution=(w, h), overlap=0.0)
    tile_fov_y = hfw * (h / w)
    tiles = compute_tile_grid(s)
    assert tiles[1].dy == pytest.approx(-tile_fov_y)


def test_compute_tile_grid_canvas_positions_no_overlap():
    """Canvas positions pack tiles contiguously when overlap=0."""
    s = _make_settings(2, 3, resolution=(512, 512), overlap=0.0)
    tiles = compute_tile_grid(s)
    t = {(t.row, t.col): t for t in tiles}
    assert t[(0, 0)].canvas_x == 0 and t[(0, 0)].canvas_y == 0
    assert t[(0, 1)].canvas_x == 512
    assert t[(1, 0)].canvas_y == 512
    assert t[(1, 2)].canvas_x == 1024 and t[(1, 2)].canvas_y == 512


def test_compute_tile_grid_canvas_positions_with_overlap():
    """Canvas step = round(w * (1-overlap)); tiles overlap in canvas too."""
    w, h = 1024, 1024
    overlap = 0.1
    s = _make_settings(2, 2, resolution=(w, h), overlap=overlap)
    tiles = compute_tile_grid(s)
    eff = round(w * (1 - overlap))  # 922
    t = {(t.row, t.col): t for t in tiles}
    assert t[(0, 1)].canvas_x == eff
    assert t[(1, 0)].canvas_y == eff


def test_compute_tile_grid_single_tile():
    """1×1 grid: single tile at origin, canvas at (0, 0)."""
    s = _make_settings(1, 1)
    tiles = compute_tile_grid(s)
    assert len(tiles) == 1
    assert tiles[0].dx == pytest.approx(0.0)
    assert tiles[0].dy == pytest.approx(0.0)
    assert tiles[0].canvas_x == 0
    assert tiles[0].canvas_y == 0


# ---------------------------------------------------------------------------
# order_tiles
# ---------------------------------------------------------------------------

def _grid_3x4():
    return compute_tile_grid(_make_settings(3, 4))


def test_order_tiles_typewriter():
    """Typewriter: all rows left-to-right."""
    tiles = _grid_3x4()
    ordered = order_tiles(tiles, TileOrderStrategy.TYPEWRITER)
    cols_by_row = {}
    for t in ordered:
        cols_by_row.setdefault(t.row, []).append(t.col)
    for row, cols in cols_by_row.items():
        assert cols == sorted(cols), f"Row {row} not L→R in typewriter"


def test_order_tiles_serpentine_even_rows_left_to_right():
    """Serpentine: even rows (0, 2, ...) go left-to-right."""
    tiles = _grid_3x4()
    ordered = order_tiles(tiles, TileOrderStrategy.SERPENTINE)
    cols_row0 = [t.col for t in ordered if t.row == 0]
    cols_row2 = [t.col for t in ordered if t.row == 2]
    assert cols_row0 == sorted(cols_row0)
    assert cols_row2 == sorted(cols_row2)


def test_order_tiles_serpentine_odd_rows_right_to_left():
    """Serpentine: odd rows (1, 3, ...) go right-to-left."""
    tiles = _grid_3x4()
    ordered = order_tiles(tiles, TileOrderStrategy.SERPENTINE)
    cols_row1 = [t.col for t in ordered if t.row == 1]
    assert cols_row1 == sorted(cols_row1, reverse=True)


def test_order_tiles_same_set():
    """Ordering never adds or removes tiles."""
    tiles = _grid_3x4()
    for strategy in TileOrderStrategy:
        ordered = order_tiles(tiles, strategy)
        assert len(ordered) == len(tiles)
        assert {(t.row, t.col) for t in ordered} == {(t.row, t.col) for t in tiles}


def test_order_tiles_single_tile():
    s = _make_settings(1, 1)
    tiles = compute_tile_grid(s)
    for strategy in TileOrderStrategy:
        ordered = order_tiles(tiles, strategy)
        assert len(ordered) == 1


# ---------------------------------------------------------------------------
# plot_tile_positions
# ---------------------------------------------------------------------------

def test_plot_tile_positions_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
        nrows=3, ncols=4, overlap=0.1, tile_order=TileOrderStrategy.SERPENTINE,
    )
    tiles = order_tiles(compute_tile_grid(s), s.tile_order)
    fig = plot_tile_positions(tiles, s)
    assert isinstance(fig, Figure)
