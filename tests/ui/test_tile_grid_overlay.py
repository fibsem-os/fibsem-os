"""The planned tile grid drawn on the canvas.

The thing worth pinning is agreement with the acquisition: the grid is a *preview*, so
if it disagrees with where tiles actually land it is worse than not drawing one. It is
placed from the same `canvas_x/canvas_y` that `stitch_tileset` pastes with, and these
tests hold it to that rather than to a second derivation of the same arithmetic.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.imaging.tiling.geometry import compute_tile_grid_from_fov
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay

WIDTH = HEIGHT = 1024
PIXEL_SIZE = 1e-7
FOV = WIDTH * PIXEL_SIZE
OVERLAP = 0.1


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeCanvas:
    """Just enough canvas for the overlay to draw against, without a real widget."""

    _pixel_size = None
    _view_margin = 0.0

    def __init__(self):
        self.redraws = 0
        self.cursor = None

    def draw_idle(self):
        self.redraws += 1

    def setCursor(self, cursor):   # noqa: N802 - mirrors the Qt API
        self.cursor = cursor

    def unsetCursor(self):         # noqa: N802 - mirrors the Qt API
        self.cursor = None

    def set_view_margin(self, margin):
        self._view_margin = margin


def build(rows=3, cols=3, overlap=OVERLAP, mask=None):
    from matplotlib.figure import Figure

    tiles = compute_tile_grid_from_fov(
        rows, cols, FOV, FOV, WIDTH, HEIGHT, overlap, mask=mask
    )
    overlay = TileGridOverlay()
    # A real axes, so the drawing tests exercise the patches rather than an early
    # return -- `_redraw` is a no-op without one, which would make them pass empty.
    overlay._ax = Figure().add_subplot(111)
    overlay._canvas = _FakeCanvas()
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    overlay._img_w, overlay._img_h = WIDTH, HEIGHT
    return overlay, tiles


def test_the_centre_tile_coincides_with_the_displayed_image(qapp):
    """The middle tile of an odd grid *is* the current view -- the acquisition starts
    where the stage already is. If it does not line up with the image, the whole grid
    is offset and every tile is drawn somewhere the stage will not go."""
    overlay, tiles = build(3, 3)
    centre = next(t for t in tiles if (t.row, t.col) == (1, 1))

    x, y, width, height = overlay._rect_for(centre)

    assert (x, y) == (0.0, 0.0)
    assert (width, height) == (WIDTH, HEIGHT)


def test_tile_spacing_matches_the_overlap(qapp):
    overlay, tiles = build(3, 3, overlap=0.1)
    first = next(t for t in tiles if (t.row, t.col) == (1, 0))
    second = next(t for t in tiles if (t.row, t.col) == (1, 1))

    step = overlay._rect_for(second)[0] - overlay._rect_for(first)[0]

    assert step == pytest.approx(WIDTH * (1 - 0.1), abs=1.0)


def test_the_grid_extent_spans_the_whole_mosaic(qapp):
    overlay, _ = build(3, 3, overlap=0.1)

    span_w, span_h = overlay._extent()

    expected = WIDTH * ((3 - 1) * (1 - 0.1) + 1)
    assert span_w == pytest.approx(expected, abs=2.0)
    assert span_h == pytest.approx(expected, abs=2.0)


def test_a_click_at_a_tile_centre_finds_that_tile(qapp):
    overlay, tiles = build(4, 4)

    for tile in tiles:
        x, y, width, height = overlay._rect_for(tile)
        hit = overlay._tile_at(x + width / 2, y + height / 2)

        assert hit is not None and (hit.row, hit.col) == (tile.row, tile.col)


def test_a_click_in_an_overlap_picks_the_nearer_tile(qapp):
    """Tiles overlap by design, so a click in a seam is inside two rectangles. Which
    one wins has to be decided by distance, not by whichever comes first in the list --
    otherwise the answer depends on the traversal order."""
    overlay, tiles = build(3, 3, overlap=0.3)
    left = next(t for t in tiles if (t.row, t.col) == (1, 0))
    right = next(t for t in tiles if (t.row, t.col) == (1, 1))

    lx, ly, lw, lh = overlay._rect_for(left)
    rx, _, _, _ = overlay._rect_for(right)
    seam_y = ly + lh / 2

    just_left = overlay._tile_at(rx + 1, seam_y)          # inside both, nearer `left`
    just_right = overlay._tile_at(lx + lw - 1, seam_y)    # inside both, nearer `right`

    assert (just_left.row, just_left.col) == (1, 0)
    assert (just_right.row, just_right.col) == (1, 1)


def test_a_click_outside_every_tile_hits_nothing(qapp):
    overlay, _ = build(3, 3)

    assert overlay._tile_at(-5000.0, -5000.0) is None


def test_toggling_reports_the_new_state_not_the_old(qapp):
    overlay, tiles = build(3, 3, mask=[[False, True, True]] + [[True] * 3] * 2)
    received = []
    overlay.tile_toggled.connect(lambda r, c, e: received.append((r, c, e)))

    disabled = next(t for t in tiles if (t.row, t.col) == (0, 0))
    x, y, width, height = overlay._rect_for(disabled)
    overlay._on_canvas_clicked(x + width / 2, y + height / 2, None)

    assert received == [(0, 0, True)]


def test_nothing_is_drawn_without_an_image(qapp):
    """There is nothing to anchor or scale against, so the grid is cleared rather than
    drawn at a guessed position."""
    overlay, _ = build(3, 3)
    overlay._img_w = overlay._img_h = 0

    overlay._redraw()

    assert overlay._artists == []


def test_a_differing_display_pixel_size_rescales_the_grid(qapp):
    """The canvas may be showing a mosaic binned differently from a tile. Conflating
    the two pixel sizes would draw the grid at the wrong scale."""
    tiles = compute_tile_grid_from_fov(3, 3, FOV, FOV, WIDTH, HEIGHT, OVERLAP)
    overlay = TileGridOverlay()
    overlay._img_w, overlay._img_h = WIDTH, HEIGHT

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    full = overlay._rect_for(tiles[0])[2]

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, display_pixel_size=PIXEL_SIZE * 2)
    half = overlay._rect_for(tiles[0])[2]

    assert half == pytest.approx(full / 2)


def test_disabled_tiles_keep_their_place_in_the_grid(qapp):
    """A skipped tile still defines the extent and the layout -- it is simply not
    acquired -- so removing it must not shift its neighbours."""
    dense, _ = build(3, 3)
    sparse, tiles = build(3, 3, mask=[[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])

    centre = next(t for t in tiles if (t.row, t.col) == (1, 1))
    assert centre.enabled is False
    assert sparse._extent() == dense._extent()
    assert sparse._rect_for(centre) == dense._rect_for(
        next(t for t in dense._tiles if (t.row, t.col) == (1, 1))
    )


# ── display options ──────────────────────────────────────────────────────


def test_hiding_the_grid_repaints_the_canvas(qapp):
    """Regression: dropping the artists takes them out of the axes, but the canvas goes
    on showing its last render until asked to redraw -- so "show grid" appeared to do
    nothing. Asserting on `_artists` alone missed this, because the list was correct
    and the screen was not."""
    overlay, _ = build(3, 3)
    overlay._redraw()
    before = overlay._canvas.redraws

    overlay.set_grid_visible(False)

    assert overlay._canvas.redraws > before


def test_hiding_the_grid_removes_the_artists_but_keeps_the_tiles(qapp):
    """Hidden, not discarded: toggling back must not need the grid rebuilt, and the
    tile selection is unaffected either way."""
    overlay, tiles = build(3, 3)
    overlay._redraw()
    assert overlay._artists

    overlay.set_grid_visible(False)
    assert overlay._artists == []
    assert len(overlay._tiles) == len(tiles)
    assert overlay.is_grid_visible is False

    overlay.set_grid_visible(True)
    assert len(overlay._artists) == len(tiles)


def test_the_fill_alpha_is_clamped(qapp):
    overlay, _ = build(2, 2)

    overlay.set_fill_alpha(5.0)
    assert overlay._fill_alpha == 1.0

    overlay.set_fill_alpha(-1.0)
    assert overlay._fill_alpha == 0.0


def test_the_scale_follows_the_canvas_pixel_size(qapp):
    """Regression: the live preview swaps a decimated mosaic in mid-run, so a scale
    captured when the grid was built drew it `preview_stride` times too large over the
    very image it describes."""

    overlay, _ = build(3, 3)
    overlay._display_pixel_size = None      # not pinned -> read from the canvas
    overlay._canvas._pixel_size = PIXEL_SIZE * 2

    assert overlay._scale() == pytest.approx(0.5)

    overlay._canvas._pixel_size = PIXEL_SIZE * 4
    assert overlay._scale() == pytest.approx(0.25)


def test_skipped_tiles_are_grey_regardless_of_the_grid_colour(qapp):
    """Colour is what marks a tile inactive at a glance; the chosen grid colour applies
    to the tiles that will actually be acquired."""
    overlay, _ = build(3, 3, mask=[[False, True, True]] + [[True] * 3] * 2)
    overlay.set_color("#00e5ff")
    overlay._redraw()

    hues = {patch.get_edgecolor()[:3] for patch in overlay._artists}
    assert len(hues) == 2, "skipped and acquired tiles should differ in colour"


# ── resize by dragging an edge ───────────────────────────────────────────


def _event(overlay, x, y, button=1):
    from types import SimpleNamespace

    return SimpleNamespace(
        button=button, inaxes=overlay._ax, xdata=x, ydata=y, x=0, y=0
    )


def _edges(overlay):
    span_w, span_h = overlay._extent()
    return (
        overlay._img_w / 2 + span_w / 2,   # right
        overlay._img_h / 2 + span_h / 2,   # bottom
    )


def test_only_the_outer_edges_grab(qapp):
    """The middle of the grid must stay clickable and pannable -- an edge grab that
    also fired in the interior would make tile toggling impossible."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    right, bottom = _edges(overlay)

    assert overlay._edge_at(right, overlay._img_h / 2) == (True, False)
    assert overlay._edge_at(overlay._img_w / 2, bottom) == (False, True)
    assert overlay._edge_at(right, bottom) == (True, True)      # corner: both axes
    assert overlay._edge_at(overlay._img_w / 2, overlay._img_h / 2) is None


def test_the_bounding_lines_do_not_grab_far_outside_the_grid(qapp):
    """The edges are line segments, not infinite lines: a point level with the right
    edge but far below the grid is not on it."""
    overlay, _ = build(3, 3)
    right, bottom = _edges(overlay)

    assert overlay._edge_at(right, bottom + overlay._img_h) is None


def test_dragging_an_edge_outward_adds_tiles_symmetrically(qapp):
    """The grid is centred on the stage position, so it can only grow both ways at
    once -- the count ticks up every half step, one tile appearing on each side."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    requested = []
    overlay.grid_resize_requested.connect(lambda r, c: requested.append((r, c)))
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._img_h / 2))
    overlay._on_drag(_event(overlay, right + WIDTH, overlay._img_h / 2))

    rows, cols = requested[-1]
    assert rows == 3, "dragging a vertical edge must not change the row count"
    assert cols > 3


def test_dragging_an_edge_inward_removes_tiles(qapp):
    overlay, _ = build(5, 5, overlap=OVERLAP)
    requested = []
    overlay.grid_resize_requested.connect(lambda r, c: requested.append((r, c)))
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._img_h / 2))
    overlay._on_drag(_event(overlay, right - 2 * WIDTH, overlay._img_h / 2))

    assert requested[-1][1] < 5


def test_the_dragged_edge_follows_the_cursor(qapp):
    """Within half a step -- the finest the count can express, since a tile appears on
    each side at once."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    resized = []
    overlay.grid_resize_requested.connect(lambda r, c: resized.append((r, c)))
    right, _ = _edges(overlay)
    step = WIDTH * (1 - OVERLAP)
    overlay._on_press(_event(overlay, right, overlay._img_h / 2))

    for target in (right + step, right + 2 * step, right - step):
        overlay._on_drag(_event(overlay, target, overlay._img_h / 2))
        rows, cols = resized[-1]
        # rebuild at the requested size and measure where its edge landed
        grown, _ = build(rows, cols, overlap=OVERLAP)
        landed = grown._img_w / 2 + grown._extent()[0] / 2

        assert abs(landed - target) <= step / 2 + 1


def test_a_drag_never_shrinks_past_a_single_tile(qapp):
    overlay, _ = build(3, 3, overlap=OVERLAP)
    resized = []
    overlay.grid_resize_requested.connect(lambda r, c: resized.append((r, c)))
    right, _ = _edges(overlay)
    overlay._on_press(_event(overlay, right, overlay._img_h / 2))

    overlay._on_drag(_event(overlay, overlay._img_w / 2, overlay._img_h / 2))

    assert resized[-1][1] >= 1


def test_a_press_away_from_an_edge_starts_no_resize(qapp):
    overlay, _ = build(3, 3)

    overlay._on_press(_event(overlay, overlay._img_w / 2, overlay._img_h / 2))

    assert overlay.is_resizing is False


def test_a_press_on_an_edge_claims_the_gesture_from_the_canvas(qapp):
    """Otherwise the drag pans the view underneath the resize, and the release fires a
    tile toggle on top of it."""
    overlay, _ = build(3, 3)
    overlay._canvas._overlay_consuming_event = False
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._img_h / 2))

    assert overlay.is_resizing is True
    assert overlay._canvas._overlay_consuming_event is True

    overlay._on_release(_event(overlay, right, overlay._img_h / 2))
    assert overlay.is_resizing is False


def test_a_hidden_grid_cannot_be_resized(qapp):
    overlay, _ = build(3, 3)
    right, _ = _edges(overlay)
    overlay.set_grid_visible(False)

    overlay._on_press(_event(overlay, right, overlay._img_h / 2))

    assert overlay.is_resizing is False


# ── hover affordance ─────────────────────────────────────────────────────


def _corners(overlay):
    span_w, span_h = overlay._extent()
    cx, cy = overlay._img_w / 2, overlay._img_h / 2
    return cx - span_w / 2, cx + span_w / 2, cy - span_h / 2, cy + span_h / 2


@pytest.mark.parametrize(
    "corner, expected",
    [
        ("top-left", "SizeFDiagCursor"),
        ("bottom-right", "SizeFDiagCursor"),
        ("top-right", "SizeBDiagCursor"),
        ("bottom-left", "SizeBDiagCursor"),
    ],
)
def test_opposite_corners_share_a_diagonal_cursor(qapp, corner, expected):
    """Top-left and bottom-right lie along one diagonal, the other two along the other.
    Getting this backwards points the cursor across the edge it would drag."""
    from PyQt5.QtCore import Qt

    overlay, _ = build(3, 3)
    left, right, top, bottom = _corners(overlay)
    x = left if "left" in corner else right
    y = top if "top" in corner else bottom

    cursor = overlay._cursor_for(*overlay._edge_sides(x, y))

    assert cursor == getattr(Qt, expected)


def test_the_edges_get_axis_cursors_and_the_interior_none(qapp):
    from PyQt5.QtCore import Qt

    overlay, _ = build(3, 3)
    left, right, top, bottom = _corners(overlay)
    cx, cy = overlay._img_w / 2, overlay._img_h / 2

    assert overlay._cursor_for(*overlay._edge_sides(right, cy)) == Qt.SizeHorCursor
    assert overlay._cursor_for(*overlay._edge_sides(left, cy)) == Qt.SizeHorCursor
    assert overlay._cursor_for(*overlay._edge_sides(cx, top)) == Qt.SizeVerCursor
    assert overlay._cursor_for(*overlay._edge_sides(cx, bottom)) == Qt.SizeVerCursor
    assert overlay._cursor_for(*overlay._edge_sides(cx, cy)) is None


def test_a_hidden_grid_shows_no_resize_cursor(qapp):
    """Nothing is drawn to grab, so promising a drag would be a lie."""
    overlay, _ = build(3, 3)
    _, right, _, _ = _corners(overlay)
    overlay.set_grid_visible(False)

    overlay._update_cursor(_event(overlay, right, overlay._img_h / 2))

    assert overlay._cursor_set is False


def test_a_drag_survives_the_grid_being_cleared_underneath_it(qapp):
    """`_refresh_tile_grid` clears the grid when the camera geometry cannot be read,
    which can happen between two motion events of a drag. The row and column counts
    are derived from the tiles, so an empty grid raised on `max()` -- and an exception
    escaping a handler can take the app down (FIB-329)."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    right, _ = _edges(overlay)
    overlay._on_press(_event(overlay, right, overlay._img_h / 2))

    overlay.clear()

    overlay._on_drag(_event(overlay, right + WIDTH, overlay._img_h / 2))
