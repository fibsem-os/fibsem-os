import pytest
from matplotlib.figure import Figure

from fibsem.imaging.tiled import (
    TilePosition,
    _spiral_order,
    compute_tile_grid,
    order_tiles,
    plot_tile_positions,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling import grid_centre_offset, unreachable_tiles
from fibsem.imaging.tiling.progress import TiledStatus
from fibsem.structures import (
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
    RangeLimit,
    TileOrderStrategy,
)

# ---------------------------------------------------------------------------
# compute_tile_grid
# ---------------------------------------------------------------------------


def _make_settings(nrows, ncols, hfw=100e-6, resolution=(1024, 1024), overlap=0.0):
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=resolution, hfw=hfw),
        nrows=nrows,
        ncols=ncols,
        overlap=overlap,
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
    assert tiles[1].dy == pytest.approx(-hfw)  # row 1, one step down
    assert tiles[2].dy == pytest.approx(-2 * hfw)


def test_compute_tile_grid_with_overlap():
    """Overlap reduces the step size."""
    hfw = 100e-6
    overlap = 0.2
    s = _make_settings(2, 2, hfw=hfw, overlap=overlap)
    tiles = compute_tile_grid(s)
    step = hfw * (1 - overlap)
    assert tiles[1].dx == pytest.approx(step)  # col 1
    assert tiles[2].dy == pytest.approx(-step)  # row 1 (index 2 in row-major for 2x2)


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


def test_order_tiles_spiral_starts_at_centre():
    """Spiral: first tile is the centre of the grid."""
    tiles = _grid_3x4()
    ordered = order_tiles(tiles, TileOrderStrategy.SPIRAL)
    assert ordered[0].row == 3 // 2
    assert ordered[0].col == 4 // 2


def test_order_tiles_spiral_3x3_full_sequence():
    """Spiral 3×3: verify the exact clockwise-outward traversal."""
    tiles = compute_tile_grid(_make_settings(3, 3))
    ordered = order_tiles(tiles, TileOrderStrategy.SPIRAL)
    expected = [(1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]
    assert [(t.row, t.col) for t in ordered] == expected


def test_order_tiles_spiral_1xN():
    """Spiral on a single-row grid: expands left/right from centre."""
    tiles = compute_tile_grid(_make_settings(1, 5))
    ordered = order_tiles(tiles, TileOrderStrategy.SPIRAL)
    # Must cover all 5 tiles
    assert {(t.row, t.col) for t in ordered} == {(0, j) for j in range(5)}
    # First tile is centre col (5//2 = 2)
    assert ordered[0].col == 2


def test_order_tiles_spiral_same_set():
    """Spiral never drops or duplicates tiles."""
    tiles = _grid_3x4()
    ordered = order_tiles(tiles, TileOrderStrategy.SPIRAL)
    assert len(ordered) == len(tiles)
    assert {(t.row, t.col) for t in ordered} == {(t.row, t.col) for t in tiles}


def test_spiral_order_helper_3x3():
    assert _spiral_order(3, 3) == [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 1),
        (2, 0),
        (1, 0),
        (0, 0),
        (0, 1),
        (0, 2),
    ]


def test_spiral_order_helper_1x1():
    assert _spiral_order(1, 1) == [(0, 0)]


# ---------------------------------------------------------------------------
# plot_tile_positions
# ---------------------------------------------------------------------------


def test_plot_tile_positions_returns_figure():
    import matplotlib

    matplotlib.use("Agg")
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
        nrows=3,
        ncols=4,
        overlap=0.1,
        tile_order=TileOrderStrategy.SERPENTINE,
    )
    tiles = order_tiles(compute_tile_grid(s), s.tile_order)
    fig = plot_tile_positions(tiles, s)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# validate_tile_stage_positions
# ---------------------------------------------------------------------------


def _make_limits(x_max=100e-3, y_max=100e-3):
    return {
        "x": RangeLimit(min=-x_max, max=x_max),
        "y": RangeLimit(min=-y_max, max=y_max),
    }


def _make_pairs(positions_xy):
    """Build (TilePosition list, FibsemStagePosition list) from (x, y) tuples."""
    tiles = [
        TilePosition(row=i, col=0, dx=0, dy=0, canvas_x=0, canvas_y=0)
        for i in range(len(positions_xy))
    ]
    stage_positions = [FibsemStagePosition(x=x, y=y) for x, y in positions_xy]
    return tiles, stage_positions


def test_validate_positions_all_within_limits():
    tiles, sps = _make_pairs([(0.0, 0.0), (10e-3, 5e-3)])
    assert validate_tile_stage_positions(tiles, sps, _make_limits()) == []


def test_validate_positions_one_out_of_bounds():
    tiles, sps = _make_pairs([(0.0, 0.0), (200e-3, 0.0)])
    result = validate_tile_stage_positions(tiles, sps, _make_limits())
    assert result == [(1, 0)]


def test_validate_positions_all_out_of_bounds():
    tiles, sps = _make_pairs([(200e-3, 200e-3), (-200e-3, -200e-3)])
    result = validate_tile_stage_positions(tiles, sps, _make_limits())
    assert len(result) == 2


def test_validate_positions_empty():
    assert validate_tile_stage_positions([], [], _make_limits()) == []


# ---------------------------------------------------------------------------
# Terminal progress state
#
# The only terminal emission used to come from _stitch, so run() on its own -- and any
# cancelled or failed run -- left consumers with a progress bar that just stopped moving.
# Every path emits one now, and `TiledStatus.is_terminal` is what a consumer branches on.
# ---------------------------------------------------------------------------

import threading
from unittest.mock import MagicMock

from fibsem.cancellation import OperationCancelledError
from fibsem.imaging.tiled import TiledAcquisitionRunner


def _runner_with_recorded_signal(monkeypatch, settings=None):
    """A runner whose phases are stubbed, recording what it emits.

    Real `OverviewAcquisitionSettings`, not a mock of one: the terminal payload's
    `total` is now `n_enabled_tiles`, which a `MagicMock(nrows=2, ncols=3)` answers
    with another mock. The assertion `total == 6` then compares a mock to an int and
    the test says the payload is wrong when the payload is fine.
    """
    emitted = []
    microscope = MagicMock()
    microscope.tiled_acquisition_signal.emit = emitted.append

    runner = TiledAcquisitionRunner.__new__(TiledAcquisitionRunner)
    runner.microscope = microscope
    runner.settings = settings if settings is not None else _make_settings(2, 3)
    runner.stop_event = None
    runner._setup = lambda: None
    runner._compute_grid = lambda: None
    runner._autofocus_if_mode = lambda mode: None
    runner._start_state = MagicMock()
    runner._image_settings = MagicMock()
    runner._prev_path = "/tmp"
    return runner, emitted


def test_run_emits_a_terminal_state_on_success(monkeypatch):
    runner, emitted = _runner_with_recorded_signal(monkeypatch)
    runner._run_tile_loop = lambda: setattr(runner, "_n_tiles_acquired", 6)

    runner.run()

    assert emitted[-1].status is TiledStatus.FINISHED
    assert emitted[-1].status.is_terminal
    assert emitted[-1].completed == 6
    assert emitted[-1].total == 6


def test_run_emits_a_terminal_state_when_cancelled(monkeypatch):
    runner, emitted = _runner_with_recorded_signal(monkeypatch)

    def cancel():
        runner._n_tiles_acquired = 2
        raise OperationCancelledError("stopped")

    runner._run_tile_loop = cancel

    with pytest.raises(OperationCancelledError):
        runner.run()

    assert emitted[-1].status is TiledStatus.CANCELLED
    assert emitted[-1].completed == 2, "partial progress is reported, not reset"


def test_run_emits_a_terminal_state_when_it_fails(monkeypatch):
    """A failure must be distinguishable from a cancel, not just 'stopped'."""
    runner, emitted = _runner_with_recorded_signal(monkeypatch)

    def explode():
        raise RuntimeError("stage fell over")

    runner._run_tile_loop = explode

    with pytest.raises(RuntimeError):
        runner.run()

    assert emitted[-1].status is TiledStatus.FAILED
    assert emitted[-1].error and "stage fell over" in emitted[-1].error, (
        "the reason was logged and thrown away, leaving the UI to say only that it failed"
    )


def test_the_terminal_report_carries_what_consumers_draw(monkeypatch):
    """A terminal that could not say how far the run got would leave the bar showing
    whatever it last had, which for a cancel is a lie about how much was acquired."""
    runner, emitted = _runner_with_recorded_signal(monkeypatch)
    runner._run_tile_loop = lambda: setattr(runner, "_n_tiles_acquired", 6)

    runner.run()

    terminal = emitted[-1]
    assert terminal.status.is_terminal
    assert terminal.completed is not None and terminal.total


# ---------------------------------------------------------------------------
# sparse tilesets (FIB-618)
#
# The layout half of the mask is covered in `tests/test_sparse_tiles.py`, against
# the geometry core both tilers share. What is left, and what these cover, is the
# beam side actually *using* it: the settings carrying a mask, the runner passing
# it on, and every count that a progress bar reads coming from the enabled tiles
# rather than from the grid's shape.
# ---------------------------------------------------------------------------


def _mask(nrows, ncols, disabled=()):
    disabled = set(disabled)
    return [[(i, j) not in disabled for j in range(ncols)] for i in range(nrows)]


def test_no_mask_counts_the_whole_grid():
    assert _make_settings(3, 4).n_enabled_tiles == 12


def test_a_mask_counts_only_what_would_be_acquired():
    s = _make_settings(3, 3)
    s.tile_mask = _mask(3, 3, disabled=[(0, 0), (2, 2), (1, 1)])
    assert s.n_enabled_tiles == 6


def test_the_mask_survives_a_round_trip():
    s = _make_settings(2, 2)
    s.tile_mask = _mask(2, 2, disabled=[(0, 1)])
    restored = OverviewAcquisitionSettings.from_dict(s.to_dict())
    assert restored.tile_mask == s.tile_mask


def test_a_numpy_mask_is_stored_as_plain_bools():
    """`np.bool_` does not survive `yaml.safe_dump`, and a mask drawn on a canvas or
    built from an array is exactly how one arrives."""
    np = pytest.importorskip("numpy")
    s = _make_settings(2, 2)
    s.tile_mask = np.array([[True, False], [False, True]])
    stored = s.to_dict()["tile_mask"]
    assert all(type(v) is bool for row in stored for v in row)


def test_settings_with_no_mask_round_trip_to_none():
    """The default has to stay None rather than becoming an all-True grid: None is
    what tells `compute_tile_grid` there is nothing to validate against the shape."""
    restored = OverviewAcquisitionSettings.from_dict(_make_settings(2, 2).to_dict())
    assert restored.tile_mask is None


def _demo_runner(settings, tmp_path):
    """A runner planned against the simulator, stopping before any acquisition."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    emitted = []
    microscope.tiled_acquisition_signal.connect(emitted.append)
    settings.image_settings.path = str(tmp_path)
    settings.image_settings.filename = "overview-image"
    runner = TiledAcquisitionRunner(microscope, settings)
    runner._setup()
    runner._compute_grid()
    return runner, emitted


def test_the_runner_plans_only_the_enabled_tiles(tmp_path):
    """The mask has to reach `compute_tile_grid`. Without it the runner drives the
    stage to every tile in the rectangle and the mask is decoration."""
    settings = _make_settings(3, 3)
    settings.tile_mask = _mask(3, 3, disabled=[(0, 0), (0, 1)])
    runner, _ = _demo_runner(settings, tmp_path)

    assert len(runner._ordered) == 7
    assert (0, 0) not in [(t.row, t.col) for t in runner._ordered]
    # The grid keeps its shape, so the mosaic and the canvas coordinates do not move.
    assert len(runner._tiles) == 9


def test_a_masked_run_does_not_report_a_short_count(tmp_path):
    """The failure this is really about. Every progress payload's `total` used to be
    `nrows * ncols`, so a 3x3 with two tiles masked off finished reading "7 / 9" --
    which is what a cancelled or failed run looks like."""
    settings = _make_settings(3, 3)
    settings.tile_mask = _mask(3, 3, disabled=[(0, 0), (0, 1)])
    runner, emitted = _demo_runner(settings, tmp_path)

    assert emitted, "the runner said nothing while planning"
    assert emitted[0].total == 7

    runner._n_tiles_acquired = 7
    runner._emit_terminal("finished", "Acquisition Complete")
    assert emitted[-1].completed == emitted[-1].total == 7


def test_an_unmasked_run_still_reports_the_whole_grid(tmp_path):
    """The other half: no mask must not quietly change what a dense run reports."""
    runner, emitted = _demo_runner(_make_settings(2, 3), tmp_path)
    assert emitted[0].total == 6
    assert len(runner._ordered) == 6


def test_the_grid_is_measured_from_the_centre_it_is_given(tmp_path):
    """A grid dragged off the stage has to acquire where it was dragged to.

    Asserted on the projected tile positions rather than on `_centre_position`: what
    matters is where the stage is sent, and a runner that stored the centre and then
    measured from somewhere else would pass the weaker check.
    """
    from fibsem import utils
    from fibsem.structures import FibsemStagePosition

    microscope, _ = utils.setup_session(manufacturer="Demo")
    here = microscope.get_stage_position()
    elsewhere = FibsemStagePosition(
        x=here.x + 250e-6, y=here.y - 100e-6, z=here.z, r=here.r, t=here.t
    )

    settings = _make_settings(1, 1)
    settings.image_settings.path = str(tmp_path)
    settings.image_settings.filename = "overview-image"
    runner = TiledAcquisitionRunner(microscope, settings, centre_position=elsewhere)
    runner._setup()
    runner._compute_grid()

    # A 1x1 grid is centred on the centre, so its one tile lands there.
    only_tile = runner._tile_stage_positions[0]
    assert only_tile.x == pytest.approx(elsewhere.x, abs=1e-9)
    assert only_tile.x != pytest.approx(here.x, abs=1e-9)


def test_no_centre_means_wherever_the_stage_is(tmp_path):
    """None is not "the stage position at the time the runner was built" -- it is
    resolved when the run starts, so a stage that moved in between is honoured."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    settings = _make_settings(1, 1)
    settings.image_settings.path = str(tmp_path)
    settings.image_settings.filename = "overview-image"
    runner = TiledAcquisitionRunner(microscope, settings)
    runner._setup()
    runner._compute_grid()

    here = microscope.get_stage_position()
    assert runner._tile_stage_positions[0].x == pytest.approx(here.x, abs=1e-9)


def test_a_run_with_no_tiles_is_refused_before_it_starts(tmp_path):
    """Left to run it walked zero tiles, emitted a *successful* terminal payload,
    restored the stage, and only then died in `_stitch` with "No tiles were acquired"
    -- so a consumer saw the run finish and then saw it fail.

    Refused before `_setup`, so nothing is emitted and no directory is made: a run with
    nothing selected is a configuration error, not an acquisition that failed.
    """
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    emitted = []
    microscope.tiled_acquisition_signal.connect(emitted.append)

    settings = _make_settings(2, 2)
    settings.tile_mask = [[False, False], [False, False]]
    settings.image_settings.path = str(tmp_path)
    settings.image_settings.filename = "overview-image"

    with pytest.raises(ValueError, match="No tiles are selected"):
        TiledAcquisitionRunner(microscope, settings).run_and_stitch()

    assert not emitted, "a refused run told consumers it had started"
    assert not list(tmp_path.iterdir()), "a refused run left a directory behind"


# ---------------------------------------------------------------------------
# unreachable_tiles -- the same question the runner asks, early enough to act on
# ---------------------------------------------------------------------------


def _identity_projection(x: float, y: float) -> FibsemStagePosition:
    """A stage whose coordinates *are* the displayed-plane offsets.

    Not a real projection -- the point of these tests is which tiles get asked about
    and where, not the geometry that answers, which `test_beam_stage_projection.py`
    covers. An identity keeps the limits box readable in the offsets themselves.
    """
    return FibsemStagePosition(x=x, y=y)


def test_the_helper_asks_about_the_offsets_the_runner_projects(tmp_path, monkeypatch):
    """The whole design rests on this: the dialog refuses the grids the runner would.

    Compared as *offsets* rather than as positions, because that is where the two could
    disagree -- the projection is shared already. The negation is the convention
    crossing over: the layout measures y upward, as `project_stable_move` takes it, and
    a displayed plane measures it down, which is what `from_plane` takes.

    A tile is masked off so the comparison covers the dropping as well as the arithmetic.
    """
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")

    settings = _make_settings(3, 4, overlap=0.1)
    settings.tile_mask = [[True] * 4 for _ in range(3)]
    settings.tile_mask[0][0] = False
    settings.image_settings.path = str(tmp_path)
    settings.image_settings.filename = "overview-image"

    runner_offsets = []
    real = microscope.project_stable_move

    def recording(dx, dy, beam_type, base_position):
        runner_offsets.append((dx, dy))
        return real(dx=dx, dy=dy, beam_type=beam_type, base_position=base_position)

    monkeypatch.setattr(microscope, "project_stable_move", recording)
    runner = TiledAcquisitionRunner(microscope, settings)
    runner._setup()
    runner._compute_grid()
    assert runner_offsets, "the runner projected nothing, so this compares nothing"

    helper_offsets = []

    def project(x, y):
        helper_offsets.append((x, y))
        return _identity_projection(x, y)

    unreachable_tiles(
        compute_tile_grid(settings, mask=settings.tile_mask),
        settings.tile_order,
        project,
        _make_limits(),
    )

    assert helper_offsets == [(dx, -dy) for dx, dy in runner_offsets]


def test_a_grid_within_the_travel_is_not_flagged():
    settings = _make_settings(3, 3, hfw=100e-6, resolution=(1024, 1024))
    tiles = compute_tile_grid(settings)
    assert (
        unreachable_tiles(
            tiles,
            settings.tile_order,
            _identity_projection,
            _make_limits(150e-6, 150e-6),
        )
        == []
    )


def test_masking_off_what_cannot_be_reached_makes_a_grid_acquirable():
    """The docstring's promise, and the reason the check goes through `order_tiles`.

    A 3x3 of 100 um tiles centres on offsets of -100, 0, +100 um, so travel that stops
    at +50 um in x puts the whole right-hand column out of range. Turning that column
    off is a legitimate fix, and the runner treats it as one -- this is the dialog
    agreeing.
    """
    settings = _make_settings(3, 3, hfw=100e-6, resolution=(1024, 1024))
    limits = {
        "x": RangeLimit(min=-150e-6, max=50e-6),
        "y": RangeLimit(min=-150e-6, max=150e-6),
    }

    flagged = unreachable_tiles(
        compute_tile_grid(settings), settings.tile_order, _identity_projection, limits
    )
    assert sorted(flagged) == [(0, 2), (1, 2), (2, 2)]

    mask = [[True, True, False] for _ in range(3)]
    assert (
        unreachable_tiles(
            compute_tile_grid(settings, mask=mask),
            settings.tile_order,
            _identity_projection,
            limits,
        )
        == []
    )


@pytest.mark.parametrize(
    "strategy",
    [
        TileOrderStrategy.TYPEWRITER,
        TileOrderStrategy.SERPENTINE,
        TileOrderStrategy.SPIRAL,
    ],
)
def test_the_traversal_does_not_change_which_tiles_are_out_of_range(strategy):
    """Order decides the sequence, not the reach. Pinned because the check goes through
    `order_tiles` for the dropping, and it would be easy to read that as the strategy
    mattering to the answer."""
    settings = _make_settings(3, 3, hfw=100e-6, resolution=(1024, 1024))
    limits = {
        "x": RangeLimit(min=-150e-6, max=50e-6),
        "y": RangeLimit(min=-150e-6, max=150e-6),
    }
    flagged = unreachable_tiles(
        compute_tile_grid(settings), strategy, _identity_projection, limits
    )
    assert sorted(flagged) == [(0, 2), (1, 2), (2, 2)]


def test_unknown_limits_are_not_projected_against():
    """A microscope that does not report its travel is not one that can reach anywhere,
    so nothing is flagged -- the runner still refuses the grid if it is unreachable.

    Asserted on the projection never being asked, not on the empty result: an empty
    `limits` dict makes `is_within_limits` answer True for every axis it is not given,
    so the result is empty whether the guard is there or not. The observable difference
    is that a caller with no limits does no work.
    """
    settings = _make_settings(3, 3)
    tiles = compute_tile_grid(settings)

    for limits in ({}, None):
        projected = []

        def project(x, y):
            projected.append((x, y))
            return _identity_projection(x, y)

        assert unreachable_tiles(tiles, settings.tile_order, project, limits) == []
        assert not projected, f"projected against {limits!r} limits"


def test_a_grid_with_nothing_enabled_is_not_asked_about():
    """`n_enabled_tiles == 0` is already refused, with its own message. Answering
    "nothing is out of range" for it would be true and useless."""
    settings = _make_settings(2, 2)
    mask = [[False, False], [False, False]]
    assert (
        unreachable_tiles(
            compute_tile_grid(settings, mask=mask),
            settings.tile_order,
            _identity_projection,
            _make_limits(1e-9, 1e-9),
        )
        == []
    )


def test_the_centring_comes_from_the_grid_it_is_given():
    """`(n - 1) * step / 2`, taken from the tiles rather than recomputed -- so a caller
    cannot centre a grid on a step size the layout did not use."""
    settings = _make_settings(3, 4, hfw=100e-6, resolution=(1024, 1024), overlap=0.1)
    step = 100e-6 * 0.9
    dx, dy = grid_centre_offset(compute_tile_grid(settings))
    assert dx == pytest.approx(3 * step / 2)
    assert dy == pytest.approx(-2 * step / 2)


def test_the_centring_counts_disabled_tiles_too():
    """They hold the grid's shape. Centring on only the enabled ones would move the
    whole grid when a corner was switched off, and the run would not follow."""
    settings = _make_settings(3, 3, hfw=100e-6, resolution=(1024, 1024))
    dense = grid_centre_offset(compute_tile_grid(settings))
    mask = [[True, True, False], [True, True, False], [True, True, False]]
    assert grid_centre_offset(compute_tile_grid(settings, mask=mask)) == dense


# ---------------------------------------------------------------------------
# the focus sweep the runner now drives (FIB-646)
# ---------------------------------------------------------------------------


def _autofocus_runner(mode, settings=None, af_result="sentinel"):
    """A runner stubbed down to the autofocus path, recording what the sweep was asked.

    Real `OverviewAcquisitionSettings` again, for the reason above: the sweep is read off
    it, and a mock would answer `settings.autofocus_settings.enabled` with another mock,
    which is truthy -- so the guard under test would pass for the wrong reason.
    """
    from unittest.mock import MagicMock

    from fibsem.imaging.tiled import TiledAcquisitionRunner
    from fibsem.structures import AutoFocusMode, BeamType

    s = settings if settings is not None else _make_settings(2, 2, hfw=500e-6)
    s.autofocus_mode = mode

    runner = TiledAcquisitionRunner.__new__(TiledAcquisitionRunner)
    runner.microscope = MagicMock()
    runner.settings = s
    runner.stop_event = None
    runner._af_mode = mode
    runner._af_settings = s.autofocus_settings
    runner._af_unavailable_logged = False
    runner._image_settings = ImageSettings(
        resolution=(1024, 1024), hfw=500e-6, beam_type=BeamType.ION
    )

    calls = []

    def _fake_run_auto_focus(microscope, **kwargs):
        calls.append(kwargs)
        return af_result

    return runner, calls, _fake_run_auto_focus


def test_the_sweep_is_given_the_tiles_own_field_of_view(monkeypatch):
    """`run_auto_focus`'s `hfw` defaults to 150 um. An overview tile is routinely 500 um
    or wider, so taking the default would score probe images framing a different picture
    from the one being focused -- and it would look like nothing more than soft tiles."""
    from fibsem.structures import AutoFocusMode, BeamType

    runner, calls, fake = _autofocus_runner(AutoFocusMode.EACH_TILE)
    monkeypatch.setattr("fibsem.imaging.tiled.run_auto_focus", fake)

    runner._autofocus_if_mode(AutoFocusMode.EACH_TILE)

    assert len(calls) == 1
    assert calls[0]["hfw"] == 500e-6
    assert calls[0]["beam_type"] is BeamType.ION
    assert calls[0]["settings"] is runner._af_settings


def test_a_mode_that_does_not_match_does_not_focus(monkeypatch):
    from fibsem.structures import AutoFocusMode

    runner, calls, fake = _autofocus_runner(AutoFocusMode.ONCE)
    monkeypatch.setattr("fibsem.imaging.tiled.run_auto_focus", fake)

    runner._autofocus_if_mode(AutoFocusMode.EACH_TILE)

    assert calls == []


def test_the_stop_event_is_handed_to_the_sweep(monkeypatch):
    """The vendor call could not take one, so a cancel could only land *between* tiles
    and the column was left wherever the last focus put it. `run_auto_focus` polls
    within the sweep and restores the starting working distance on the way out."""
    import threading

    from fibsem.structures import AutoFocusMode

    runner, calls, fake = _autofocus_runner(AutoFocusMode.EACH_TILE)
    runner.stop_event = threading.Event()
    monkeypatch.setattr("fibsem.imaging.tiled.run_auto_focus", fake)

    runner._autofocus_if_mode(AutoFocusMode.EACH_TILE)

    assert calls[0]["stop_event"] is runner.stop_event


def test_an_unavailable_focus_warns_once_and_lets_the_run_continue(monkeypatch, caplog):
    """None means the backend cannot set the working distance, so the sweep declined
    rather than faking a completed focus (FIB-508, TESCAN ION).

    Unfocused is not the same as wrong -- the tiles are still worth having -- so this
    warns instead of stopping. Once, not per tile: on a 5 x 5 at EACH_TILE the per-tile
    version is 25 identical lines through the middle of the acquisition log.
    """
    import logging as _logging

    from fibsem.structures import AutoFocusMode

    runner, calls, fake = _autofocus_runner(AutoFocusMode.EACH_TILE, af_result=None)
    monkeypatch.setattr("fibsem.imaging.tiled.run_auto_focus", fake)

    with caplog.at_level(_logging.WARNING):
        for _ in range(5):
            runner._autofocus_if_mode(AutoFocusMode.EACH_TILE)

    assert len(calls) == 5, "it should keep trying, not disable itself"
    warnings = [r for r in caplog.records if "Autofocus is unavailable" in r.message]
    assert len(warnings) == 1, f"warned {len(warnings)} times, expected once"


def test_a_sweep_with_no_enabled_passes_is_refused_before_the_first_tile():
    """`run_auto_focus` raises on an all-disabled sweep. Letting that happen at tile 1
    of 25 means the stage has already moved, a folder exists and progress has been
    emitted -- so it is caught in `_setup`, where nothing has happened yet."""
    from unittest.mock import MagicMock

    from fibsem.imaging.tiled import TiledAcquisitionRunner
    from fibsem.structures import AutoFocusMode

    s = _make_settings(2, 2)
    s.autofocus_mode = AutoFocusMode.EACH_TILE
    for p in s.autofocus_settings.passes:
        p.enabled = False

    runner = TiledAcquisitionRunner.__new__(TiledAcquisitionRunner)
    runner.microscope = MagicMock()
    runner.settings = s
    with pytest.raises(ValueError, match="every sweep pass is disabled"):
        runner._setup()


def test_an_all_disabled_sweep_is_fine_when_autofocus_is_off(tmp_path):
    """The guard is about a contradiction, not about the sweep in isolation. NONE plus a
    disabled sweep is coherent -- nothing was going to focus anyway -- and refusing it
    would make an untouched default sweep able to block a run that never wanted one.

    Runs the *whole* of `_setup` rather than stopping at the guard, which is also what
    proves the guard sits before the side effects: the refusing test above never reaches
    the line that makes the tile folder, and this one does.
    """
    from unittest.mock import MagicMock

    from fibsem.imaging.tiled import TiledAcquisitionRunner
    from fibsem.structures import AutoFocusMode

    s = _make_settings(2, 2)
    s.image_settings.path = str(tmp_path)
    s.image_settings.filename = "overview"
    s.autofocus_mode = AutoFocusMode.NONE
    for p in s.autofocus_settings.passes:
        p.enabled = False

    runner = TiledAcquisitionRunner.__new__(TiledAcquisitionRunner)
    runner.microscope = MagicMock()
    runner.settings = s
    runner._setup()  # must not raise

    assert runner._af_mode is AutoFocusMode.NONE
    assert (tmp_path / "overview").is_dir()
