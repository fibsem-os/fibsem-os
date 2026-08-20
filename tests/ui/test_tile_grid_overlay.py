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
from fibsem.ui.widgets.canvas.canvas_base import ContentRect
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import (
    MOVE_DRAG_THRESHOLD_PX,
    TileGridOverlay,
)

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


def _seed_content(overlay, width=WIDTH, height=HEIGHT):
    """Tell the overlay how much canvas content there is, via the real protocol."""
    overlay.on_content_changed(ContentRect(0.0, 0.0, width, height))


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
    _seed_content(overlay)
    return overlay, tiles


def _event(overlay, x, y, button=1, px=0.0, py=0.0):
    """A matplotlib mouse event over the overlay's axes.

    `px`/`py` are screen pixels. They matter for the move gesture, which decides
    between a click and a drag by how far the pointer travelled on screen -- in data
    coordinates that distance would change with the zoom.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        button=button, inaxes=overlay._ax, xdata=x, ydata=y, x=px, y=py
    )


def _click(overlay, x, y):
    """Press and release at one point, without moving between the two.

    Ends by letting the deferred toggle out. A release no longer emits `tile_toggled`
    directly -- it waits out the double-click interval first, because a double-click
    here is a stage move rather than a tile edit (FIB-520). Firing the timer by hand
    rather than waiting on it keeps these tests about *which* tile and *what value*,
    and instant; the deferral itself is pinned separately below.
    """
    overlay._on_press(_event(overlay, x, y))
    overlay._on_release(_event(overlay, x, y))
    _flush_toggle(overlay)


def _flush_toggle(overlay):
    """Let a pending toggle out without waiting for the real interval."""
    if overlay._toggle_timer.isActive():
        overlay._toggle_timer.stop()
        overlay._emit_pending_toggle()


def _double_click(overlay, x, y):
    """The four events matplotlib delivers for a double-click, in order.

    press, release, press(dblclick=True), release -- the second press is the first
    thing that says a double-click is happening, and it arrives *after* the release
    that would otherwise have toggled.
    """
    for dblclick in (False, True):
        press = _event(overlay, x, y)
        press.dblclick = dblclick
        overlay._on_press(press)
        release = _event(overlay, x, y)
        release.dblclick = False
        overlay._on_release(release)


def _drag(overlay, start, end, px_travel=50.0):
    """Press at *start*, move to *end*, release. Screen travel defaults to a real drag."""
    overlay._on_press(_event(overlay, *start))
    overlay._on_drag(_event(overlay, *end, px=px_travel, py=px_travel))
    overlay._on_release(_event(overlay, *end, px=px_travel, py=px_travel))
    _flush_toggle(overlay)


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
    _click(overlay, x + width / 2, y + height / 2)

    assert received == [(0, 0, True)]


def test_nothing_is_drawn_without_an_image(qapp):
    """There is nothing to anchor or scale against, so the grid is cleared rather than
    drawn at a guessed position."""
    overlay, _ = build(3, 3)
    _seed_content(overlay, 0, 0)

    overlay._redraw()

    assert overlay._artists == []


def test_a_differing_display_pixel_size_rescales_the_grid(qapp):
    """The canvas may be showing a mosaic binned differently from a tile. Conflating
    the two pixel sizes would draw the grid at the wrong scale."""
    tiles = compute_tile_grid_from_fov(3, 3, FOV, FOV, WIDTH, HEIGHT, OVERLAP)
    overlay = TileGridOverlay()
    _seed_content(overlay)

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


# ── tiles the stage cannot reach ─────────────────────────────────────────


def _edge_hues(overlay):
    return {tuple(round(c, 3) for c in patch.get_edgecolor()[:3]) for patch in overlay._artists}


def _is_mark(artist):
    """The centre cross is a Line2D; the tiles are patches."""
    return artist.__class__.__name__ == "Line2D"


def _marks(overlay):
    return [a for a in overlay._artists if _is_mark(a)]


def _outlines(overlay):
    """The tile patches by (row, col).

    Not `zip(_artists, _tiles)`: a flagged tile contributes two cross lines as well, so
    the artist list is longer than the tile list and the pairing slips from the first
    flagged tile onward.
    """
    outlines = [a for a in overlay._artists if not _is_mark(a)]
    return {(t.row, t.col): p for p, t in zip(outlines, overlay._tiles)}


def test_an_unreachable_tile_recedes_and_is_crossed(qapp):
    """Emphasis inverted on purpose. Earlier attempts added ink to the tiles you cannot
    have -- and the common case is a whole row going out of range at once, so half the
    grid ended up shouting. What you steer by is the region you can still acquire, so
    the rest dims instead."""
    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(0, 0), (0, 1)])

    outlines = _outlines(overlay)
    assert len(_marks(overlay)) == 2, "one cross artist per unreachable tile"

    faded = outlines[(0, 0)].get_edgecolor()[3]
    solid = outlines[(2, 2)].get_edgecolor()[3]
    assert faded < solid, "the unreachable tile did not recede"
    assert tuple(outlines[(0, 0)].get_edgecolor()[:3]) == tuple(
        outlines[(2, 2)].get_edgecolor()[:3]
    ), "it should be the grid's own colour, just quieter"


def test_the_cross_sits_at_the_tile_centre(qapp):
    """Where it does, because the centre is exactly what the reachability check tests --
    which is also why a tile may legitimately overhang the travel box. A mark at the
    centre is what makes that overhang read as intended rather than as a bug."""
    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(1, 1)])

    x, y, width, height = overlay._rect_for(
        next(t for t in overlay._tiles if (t.row, t.col) == (1, 1))
    )
    import math

    line = _marks(overlay)[0]
    xs = [v for v in line.get_xdata() if not math.isnan(v)]
    ys = [v for v in line.get_ydata() if not math.isnan(v)]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(x + width / 2)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(y + height / 2)


def test_the_cross_scales_with_the_tiles(qapp):
    """A fixed size would swamp a large grid, where the tiles are small on screen."""
    from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import UNREACHABLE_MARK_SIZE

    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(1, 1)])
    _, _, width, height = overlay._rect_for(overlay._tiles[0])

    import math

    xs = [v for v in _marks(overlay)[0].get_xdata() if not math.isnan(v)]
    assert max(xs) - min(xs) == pytest.approx(min(width, height) * UNREACHABLE_MARK_SIZE)


def test_a_reachable_tile_is_not_marked(qapp):
    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(0, 0)])
    assert len(_marks(overlay)) == 1


def test_a_tile_that_is_switched_off_is_not_also_flagged(qapp):
    """Turning the tile off is what the flag is *asking* for, so a tile that is already
    off must stop being flagged -- otherwise the grid never stops complaining about a
    problem you have already fixed.

    Compared against another switched-off tile rather than against the disabled colour:
    the colour was never the thing at risk. Flagging a disabled tile leaves its edge
    grey either way and shows up only in the line width -- which a colour assertion
    reads straight past.
    """
    overlay, tiles = build(3, 3, mask=[[False, False, True]] + [[True] * 3] * 2)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(0, 0)])

    def described(coord):
        patch = _outlines(overlay)[coord]
        return (
            tuple(round(c, 3) for c in patch.get_edgecolor()),
            tuple(round(c, 3) for c in patch.get_facecolor()),
            patch.get_linewidth(),
            patch.get_linestyle(),
            patch.get_fill(),
        )

    assert described((0, 0)) == described((0, 1)), (
        "a switched-off tile was drawn differently because it was also flagged"
    )
    assert not _marks(overlay), "a switched-off tile was crossed"


def test_passing_no_flags_leaves_the_previous_ones_alone(qapp):
    """A host that never works reachability out draws exactly what it drew before, and
    one that does is not made to re-send them on every unrelated redraw."""
    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(0, 0)])
    assert overlay._unreachable == {(0, 0)}

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    assert overlay._unreachable == {(0, 0)}, "the flags were dropped by an unrelated redraw"

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[])
    assert overlay._unreachable == set(), "an empty set must clear them"


def test_clearing_the_grid_clears_the_flags(qapp):
    """Left behind they would be re-applied to the next grid drawn, by row and column --
    which is a different grid."""
    overlay, tiles = build(3, 3)
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, unreachable=[(0, 0)])
    overlay.clear()
    assert overlay._unreachable == set()


# ── one repaint per refresh (FIB-751) ────────────────────────────────────


def test_setting_the_anchor_with_the_grid_repaints_once(qapp):
    """A host that sets both used to pay two full repaints, on a path that runs on every
    motion event of a drag. Each repaint re-renders the whole canvas -- every placed
    image, not just the tiles -- so the second one is not a rounding error."""
    overlay, tiles = build(3, 3)
    overlay._canvas.redraws = 0
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, anchor=(10.0, 20.0))
    assert overlay._canvas.redraws == 1

    overlay._canvas.redraws = 0
    overlay.set_anchor((10.0, 20.0))
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    assert overlay._canvas.redraws == 2, "the two-call form is what this replaces"


def test_the_anchor_still_has_two_meanings(qapp):
    """`None` means "follow the content"; omitting it means "leave it as it is". Both
    are answers, which is why the default is a sentinel rather than None."""
    overlay, tiles = build(3, 3)
    overlay.set_anchor((10.0, 20.0))

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    assert overlay._anchor_point == (10.0, 20.0), "omitting the anchor moved the grid"

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, anchor=None)
    assert overlay._anchor_point is None, "None must go back to following the content"


# ── blitting a drag (FIB-752) ────────────────────────────────────────────


class _BlittingCanvas(_FakeCanvas):
    """A canvas that offers the blit calls, and records what it was asked to do.

    The plain `_FakeCanvas` deliberately does not: a host without these has to fall
    back, and that is its own test below.
    """

    def __init__(self):
        super().__init__()
        self.full_draws = 0
        self.captures = 0
        self.restores = 0
        self.blits = 0

    def draw(self):
        self.full_draws += 1

    def copy_from_bbox(self, bbox):
        self.captures += 1
        return f"background-{self.captures}"

    def restore_region(self, background):
        self.restores += 1

    def blit(self, bbox):
        self.blits += 1


def _blitting(rows=3, cols=3):
    """An overlay whose canvas records the blit calls, over a figure that can serve them.

    The recording canvas cannot render, and `draw_artist` needs a renderer that only a
    real draw caches -- so an Agg canvas is attached to the *figure* to provide one.
    Without it the overlay falls back to a repaint, correctly, and the test would be
    asserting the fallback while claiming to assert the blit.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    overlay, tiles = build(rows, cols)
    FigureCanvasAgg(overlay._ax.figure)
    overlay._ax.figure.canvas.draw()
    overlay._canvas = _BlittingCanvas()
    return overlay, tiles


def test_a_drag_frame_blits_instead_of_repainting(qapp):
    """A repaint re-renders every placed overview to move some rectangles -- 165 ms per
    motion event with six of them on a 3x3 grid. A blitted frame restores the canvas as
    it was without the grid and draws only the tiles: 4.2 ms, measured."""
    overlay, tiles = _blitting()
    overlay._move_active = True

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, anchor=(1.0, 2.0))

    canvas = overlay._canvas
    assert canvas.full_draws == 1, "the background is captured once, on the first frame"
    assert canvas.captures == 1
    assert (canvas.restores, canvas.blits) == (1, 1)
    assert canvas.redraws == 0, "a blitted frame must not ask for a full repaint"


def test_the_background_is_captured_once_for_the_whole_drag(qapp):
    overlay, tiles = _blitting()
    overlay._move_active = True

    for step in range(5):
        overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE, anchor=(float(step), 0.0))

    canvas = overlay._canvas
    assert (canvas.full_draws, canvas.captures) == (1, 1)
    assert (canvas.restores, canvas.blits) == (5, 5)


def test_the_tiles_are_animated_so_they_stay_out_of_the_background(qapp):
    """`animated` is the whole mechanism: a full draw skips animated artists, which is
    what makes the captured background the canvas *without* the grid. Drawn into it, the
    grid would smear across every frame of the drag."""
    overlay, tiles = _blitting()
    overlay._move_active = True

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)

    assert all(artist.get_animated() for artist in overlay._artists)


def test_releasing_puts_the_grid_back_into_the_canvas(qapp):
    """The trap this mechanism sets. Artists left `animated` are skipped by every full
    draw, so the grid would vanish the next time anything else repainted -- visible only
    later, and nowhere near the drag that caused it."""
    overlay, tiles = _blitting()
    overlay._move_active = True
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    assert overlay._blit_bg is not None

    overlay._on_release(_event(overlay, 0.0, 0.0))

    assert overlay._blit_bg is None, "the background outlived the drag"
    assert not any(artist.get_animated() for artist in overlay._artists), (
        "the grid is still animated, so a full draw would leave it out"
    )
    assert overlay._canvas.redraws >= 1, "the grid was not put back with a normal draw"


def test_nothing_is_blitted_outside_a_drag(qapp):
    """There is nothing to gain -- a settings change redraws once -- and a background
    captured outside a gesture would go stale the moment anything else on the canvas
    changed."""
    overlay, tiles = _blitting()

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)

    canvas = overlay._canvas
    assert (canvas.captures, canvas.blits) == (0, 0)
    assert canvas.redraws == 1
    assert not any(artist.get_animated() for artist in overlay._artists)


def test_the_background_is_dropped_when_the_canvas_bounds_change(qapp):
    """It was captured against the old bounds, so it no longer describes what is behind
    the grid -- blitting it would paint the previous contents back over the canvas."""
    overlay, tiles = _blitting()
    overlay._move_active = True
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    assert overlay._blit_bg is not None

    _seed_content(overlay, width=WIDTH * 2, height=HEIGHT * 2)

    assert overlay._canvas.captures == 2, "the background was reused after a resize"


def test_a_canvas_that_cannot_blit_still_gets_its_grid(qapp):
    """Not every host is a matplotlib canvas. Falling back costs a frame's efficiency;
    failing would cost the frame."""
    overlay, tiles = build(3, 3)          # the plain fake, with no blit calls
    overlay._move_active = True

    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)

    assert overlay._canvas.redraws >= 1
    assert overlay._artists, "the grid was not drawn at all"
    assert not any(artist.get_animated() for artist in overlay._artists)


def test_a_backend_that_fails_mid_blit_falls_back_to_a_repaint(qapp):
    """Costing a frame's efficiency, not the frame."""
    overlay, tiles = _blitting()
    overlay._move_active = True

    def explode(bbox):
        raise RuntimeError("no blitting here")

    overlay._canvas.blit = explode
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)

    assert overlay._canvas.redraws >= 1, "the frame was dropped instead of repainted"
    assert overlay._blit_bg is None, "a failed blit kept its background"


def test_a_repaint_mid_drag_puts_the_grid_back(qapp):
    """A host that repaints for its own reasons takes the grid off the canvas, because a
    full draw skips animated artists. On the fluorescence tab that is every motion event
    -- it refreshes the stage context from the same handler that moves the grid -- and it
    showed as flicker with the grid missing for stretches of the drag."""
    overlay, tiles = _blitting()
    overlay._move_active = True
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    canvas = overlay._canvas
    before = canvas.captures

    overlay._on_canvas_drawn(None)

    assert canvas.captures == before + 1, "the background was not recaptured"


def test_restoring_after_a_repaint_does_not_ask_the_canvas_to_present(qapp):
    """It runs *inside* the canvas's own paint, which is about to present what the
    renderer holds. Asking it to present again from in there re-enters the paint, which
    Qt reports as `QWidget::repaint: Recursive repaint detected` followed by a stream of
    `QPainter::begin: Paint device returned engine == 0` -- visible only in the log of a
    running app, which is where it was found."""
    overlay, tiles = _blitting()
    overlay._move_active = True
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    canvas = overlay._canvas
    before = canvas.blits

    overlay._on_canvas_drawn(None)

    assert canvas.blits == before, "blitting from inside a draw re-enters the paint"


def test_the_overlay_listens_for_repaints(qapp):
    """The handler above is only worth anything if it is wired to something.

    Asserted on the connection, not by calling the handler: every other test here calls
    it directly, and a version with the `mpl_connect` line deleted passed all of them.
    """
    from matplotlib.figure import Figure

    connected = []

    class _Recording(_BlittingCanvas):
        def mpl_connect(self, event, handler):
            connected.append(event)
            return len(connected)

    overlay = TileGridOverlay()
    overlay.attach(Figure().add_subplot(111), _Recording())

    assert "draw_event" in connected, (
        "nothing restores the grid when a host repaints mid-drag"
    )


def test_nothing_is_restored_outside_a_drag(qapp):
    overlay, tiles = _blitting()
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)
    before = overlay._canvas.captures
    overlay._on_canvas_drawn(None)
    assert overlay._canvas.captures == before


def test_the_end_of_a_drag_is_announced(qapp):
    """Hosts have work that repaints the whole canvas and does not belong on a motion
    event. Without somewhere to put it they did it per event, which defeated the blit."""
    overlay, tiles = _blitting()
    finished = []
    overlay.drag_finished.connect(lambda: finished.append(1))
    overlay._move_active = True
    overlay.set_grid(tiles, (HEIGHT, WIDTH), PIXEL_SIZE)

    overlay._on_release(_event(overlay, 0.0, 0.0))

    assert finished == [1]


def test_a_click_is_not_the_end_of_a_drag(qapp):
    """A press and release that never moved is a tile toggle. Announcing a finished drag
    there would have the host redo its expensive work on every click."""
    overlay, tiles = _blitting()
    finished = []
    overlay.drag_finished.connect(lambda: finished.append(1))

    _click(overlay, 0.0, 0.0)

    assert finished == []


# ── resize by dragging an edge ───────────────────────────────────────────


def _edges(overlay):
    span_w, span_h = overlay._extent()
    return (
        overlay._rect.cx + span_w / 2,   # right
        overlay._rect.cy + span_h / 2,   # bottom
    )


def test_only_the_outer_edges_grab(qapp):
    """The middle of the grid must stay clickable and pannable -- an edge grab that
    also fired in the interior would make tile toggling impossible."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    right, bottom = _edges(overlay)

    assert overlay._edge_at(right, overlay._rect.cy) == (True, False)
    assert overlay._edge_at(overlay._rect.cx, bottom) == (False, True)
    assert overlay._edge_at(right, bottom) == (True, True)      # corner: both axes
    assert overlay._edge_at(overlay._rect.cx, overlay._rect.cy) is None


def test_the_bounding_lines_do_not_grab_far_outside_the_grid(qapp):
    """The edges are line segments, not infinite lines: a point level with the right
    edge but far below the grid is not on it."""
    overlay, _ = build(3, 3)
    right, bottom = _edges(overlay)

    assert overlay._edge_at(right, bottom + overlay._rect.height) is None


def test_dragging_an_edge_outward_adds_tiles_symmetrically(qapp):
    """The grid is centred on the stage position, so it can only grow both ways at
    once -- the count ticks up every half step, one tile appearing on each side."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    requested = []
    overlay.grid_resize_requested.connect(lambda r, c: requested.append((r, c)))
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))
    overlay._on_drag(_event(overlay, right + WIDTH, overlay._rect.cy))

    rows, cols = requested[-1]
    assert rows == 3, "dragging a vertical edge must not change the row count"
    assert cols > 3


def test_dragging_an_edge_inward_removes_tiles(qapp):
    overlay, _ = build(5, 5, overlap=OVERLAP)
    requested = []
    overlay.grid_resize_requested.connect(lambda r, c: requested.append((r, c)))
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))
    overlay._on_drag(_event(overlay, right - 2 * WIDTH, overlay._rect.cy))

    assert requested[-1][1] < 5


def test_the_dragged_edge_follows_the_cursor(qapp):
    """Within half a step -- the finest the count can express, since a tile appears on
    each side at once."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    resized = []
    overlay.grid_resize_requested.connect(lambda r, c: resized.append((r, c)))
    right, _ = _edges(overlay)
    step = WIDTH * (1 - OVERLAP)
    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    for target in (right + step, right + 2 * step, right - step):
        overlay._on_drag(_event(overlay, target, overlay._rect.cy))
        rows, cols = resized[-1]
        # rebuild at the requested size and measure where its edge landed
        grown, _ = build(rows, cols, overlap=OVERLAP)
        landed = grown._rect.cx + grown._extent()[0] / 2

        assert abs(landed - target) <= step / 2 + 1


def test_a_drag_never_shrinks_past_a_single_tile(qapp):
    overlay, _ = build(3, 3, overlap=OVERLAP)
    resized = []
    overlay.grid_resize_requested.connect(lambda r, c: resized.append((r, c)))
    right, _ = _edges(overlay)
    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    overlay._on_drag(_event(overlay, overlay._rect.cx, overlay._rect.cy))

    assert resized[-1][1] >= 1


def test_a_press_away_from_an_edge_starts_no_resize(qapp):
    overlay, _ = build(3, 3)

    overlay._on_press(_event(overlay, overlay._rect.cx, overlay._rect.cy))

    assert overlay.is_resizing is False


def test_a_press_on_an_edge_claims_the_gesture_from_the_canvas(qapp):
    """Otherwise the drag pans the view underneath the resize, and the release fires a
    tile toggle on top of it."""
    overlay, _ = build(3, 3)
    overlay._canvas._overlay_consuming_event = False
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    assert overlay.is_resizing is True
    assert overlay._canvas._overlay_consuming_event is True

    overlay._on_release(_event(overlay, right, overlay._rect.cy))
    assert overlay.is_resizing is False


def test_a_hidden_grid_cannot_be_resized(qapp):
    overlay, _ = build(3, 3)
    right, _ = _edges(overlay)
    overlay.set_grid_visible(False)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    assert overlay.is_resizing is False


# ── hover affordance ─────────────────────────────────────────────────────


def _corners(overlay):
    span_w, span_h = overlay._extent()
    cx, cy = overlay._rect.cx, overlay._rect.cy
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
    cx, cy = overlay._rect.cx, overlay._rect.cy

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

    overlay._update_cursor(_event(overlay, right, overlay._rect.cy))

    assert overlay._cursor_set is False


def test_a_drag_survives_the_grid_being_cleared_underneath_it(qapp):
    """`_refresh_tile_grid` clears the grid when the camera geometry cannot be read,
    which can happen between two motion events of a drag. The row and column counts
    are derived from the tiles, so an empty grid raised on `max()` -- and an exception
    escaping a handler can take the app down (FIB-329)."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    right, _ = _edges(overlay)
    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    overlay.clear()

    overlay._on_drag(_event(overlay, right + WIDTH, overlay._rect.cy))


# ── move by dragging the interior ────────────────────────────────────────


def _centre(overlay):
    return overlay._rect.cx, overlay._rect.cy


def test_dragging_the_interior_moves_the_grid_by_the_drag(qapp):
    """The grid has to land where it was let go, not merely somewhere in that
    direction -- it is a plan of where tiles will be acquired, and a drag that
    approximated the position would plan an acquisition of somewhere else."""
    overlay, _ = build(3, 3)
    moved = []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    cx, cy = _centre(overlay)

    _drag(overlay, (cx, cy), (cx + 300.0, cy - 125.0))

    assert moved[-1] == pytest.approx((cx + 300.0, cy - 125.0))


def test_a_press_that_barely_moves_is_a_click_not_a_drag(qapp):
    """The same press starts both gestures, so the split has to fall somewhere. Below
    the threshold it must toggle the tile: a hand is never perfectly still, and a click
    that nudged the whole grid by a pixel would be worse than one that did nothing."""
    overlay, tiles = build(3, 3)
    moved, toggled = [], []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
    centre = next(t for t in tiles if (t.row, t.col) == (1, 1))
    x, y, width, height = overlay._rect_for(centre)
    point = (x + width / 2, y + height / 2)

    _drag(overlay, point, point, px_travel=MOVE_DRAG_THRESHOLD_PX - 1)

    assert moved == [], "a press that did not travel must not move the grid"
    assert toggled == [(1, 1, False)]


def test_a_real_drag_does_not_also_toggle_the_tile_it_started_on(qapp):
    overlay, _ = build(3, 3)
    toggled = []
    overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
    cx, cy = _centre(overlay)

    _drag(overlay, (cx, cy), (cx + 400.0, cy))

    assert toggled == []


def test_a_press_inside_the_grid_claims_the_gesture(qapp):
    """Otherwise the canvas pans underneath the move, and the grid and the view slide
    apart by the same drag."""
    overlay, _ = build(3, 3)
    overlay._canvas._overlay_consuming_event = False

    overlay._on_press(_event(overlay, *_centre(overlay)))

    assert overlay._canvas._overlay_consuming_event is True


def test_a_press_outside_the_grid_is_left_to_the_canvas(qapp):
    """The canvas still has to be pannable. Claiming everything would take that away
    for the sake of a gesture the pointer was nowhere near."""
    overlay, _ = build(3, 3)
    overlay._canvas._overlay_consuming_event = False
    _, right, _, _ = _corners(overlay)

    overlay._on_press(_event(overlay, right + 5 * WIDTH, overlay._rect.cy))

    assert overlay._canvas._overlay_consuming_event is False


def test_an_edge_press_still_resizes_rather_than_moving(qapp):
    """Edges are inside the bounding box too, so the two gestures overlap. Resize wins,
    matching the cursor shown there."""
    overlay, _ = build(3, 3, overlap=OVERLAP)
    moved = []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    right, _ = _edges(overlay)

    _drag(overlay, (right, overlay._rect.cy), (right + WIDTH, overlay._rect.cy))

    assert moved == []
    assert overlay.is_resizing is False  # released


def test_a_hidden_grid_cannot_be_moved(qapp):
    overlay, _ = build(3, 3)
    moved = []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    overlay.set_grid_visible(False)
    cx, cy = _centre(overlay)

    _drag(overlay, (cx, cy), (cx + 400.0, cy))

    assert moved == []


def test_the_interior_offers_a_move_cursor(qapp):
    """The drag is undiscoverable without it -- nothing about a grid of rectangles
    suggests the whole thing can be pushed around."""
    from PyQt5.QtCore import Qt

    overlay, _ = build(3, 3)

    overlay._update_cursor(_event(overlay, *_centre(overlay)))

    assert overlay._canvas.cursor == Qt.SizeAllCursor


def test_a_move_drag_survives_the_grid_being_cleared_underneath_it(qapp):
    """Same hazard as the resize drag: `_refresh_tile_grid` can clear the grid between
    two motion events, and an exception escaping a handler can take the app down."""
    overlay, _ = build(3, 3)
    cx, cy = _centre(overlay)
    overlay._on_press(_event(overlay, cx, cy))

    overlay.clear()

    overlay._on_drag(_event(overlay, cx + 400.0, cy, px=50.0, py=50.0))
    overlay._on_release(_event(overlay, cx + 400.0, cy, px=50.0, py=50.0))


class TestADoubleClickIsAStageMoveNotATileEdit:
    """FIB-520, found on hardware.

    Double-clicking the canvas moves the stage. Inside the grid that same gesture also
    toggled the tile under the cursor, so a deliberate move silently edited what the
    next run would acquire.

    Both paths see the gesture: the canvas emits `canvas_double_clicked` from the second
    press, and the overlay took raw press/release and treated each release as a click.
    Nothing arbitrated, because the first release lands before anything knows a second
    click is coming.
    """

    def test_a_double_click_does_not_toggle_the_tile(self, qapp):
        """The defect. Measured before the fix: two emissions, not zero."""
        overlay, _ = build(3, 3)
        toggled = []
        overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
        cx, cy = _centre(overlay)

        _double_click(overlay, cx, cy)
        _flush_toggle(overlay)  # nothing should be left waiting, but prove it

        assert toggled == [], (
            "a double-click toggled the tile under the cursor -- it is a stage move, "
            "and editing the acquisition plan on the way to one is silent"
        )

    def test_the_two_emissions_did_not_cancel_each_other(self, qapp):
        """Why the bug was visible rather than harmlessly self-undoing.

        The overlay reads `tile.enabled` from its own tiles, and those are not updated
        between the two releases -- the mask lives in the settings widget and comes back
        through a redraw. So both emissions carried the *same* value. This pins the
        state that made that true, so a later change that starts mutating tiles in place
        cannot quietly turn the double-emission into a no-op and look fixed.
        """
        overlay, tiles = build(3, 3)
        centre = next(t for t in tiles if (t.row, t.col) == (1, 1))
        cx, cy = _centre(overlay)

        _double_click(overlay, cx, cy)

        assert centre.enabled is True, (
            "the overlay mutated its own tile -- it emits and lets the settings widget "
            "own the mask, and two writers is how the two views drift apart"
        )

    def test_a_single_click_still_toggles(self, qapp):
        """The other direction. Suppressing the double-click must not cost the click."""
        overlay, _ = build(3, 3)
        toggled = []
        overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
        cx, cy = _centre(overlay)

        _click(overlay, cx, cy)

        assert toggled == [(1, 1, False)]

    def test_the_toggle_waits_rather_than_firing_and_being_undone(self, qapp):
        """The mechanism, not just the outcome.

        A release schedules the toggle and emits nothing yet; only the timer does. An
        implementation that emitted immediately and undid it on the second press would
        pass the two tests above and still write a mask it had to rewrite.
        """
        overlay, _ = build(3, 3)
        toggled = []
        overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
        cx, cy = _centre(overlay)

        overlay._on_press(_event(overlay, cx, cy))
        overlay._on_release(_event(overlay, cx, cy))

        assert toggled == [], "the toggle was emitted before the double-click interval"
        assert overlay._pending_toggle == (1, 1, False)
        assert overlay._toggle_timer.isActive()

    def test_the_second_press_cancels_the_pending_toggle(self, qapp):
        overlay, _ = build(3, 3)
        cx, cy = _centre(overlay)
        overlay._on_press(_event(overlay, cx, cy))
        overlay._on_release(_event(overlay, cx, cy))

        press = _event(overlay, cx, cy)
        press.dblclick = True
        overlay._on_press(press)

        assert overlay._pending_toggle is None
        assert not overlay._toggle_timer.isActive()

    def test_the_second_press_takes_no_part_in_the_gesture(self, qapp):
        """It returns before claiming, so the release after it does nothing either.

        Without that, the second press would arm `_move_start` again and its release
        would schedule a fresh toggle -- putting the bug straight back, one click later.
        """
        overlay, _ = build(3, 3)
        cx, cy = _centre(overlay)

        press = _event(overlay, cx, cy)
        press.dblclick = True
        overlay._on_press(press)

        assert overlay._move_start is None

    def test_detaching_drops_a_pending_toggle(self, qapp):
        """It would otherwise fire into a host that has stopped listening, or -- on a
        rebuilt tab -- into the wrong one."""
        overlay, _ = build(3, 3)
        toggled = []
        overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
        cx, cy = _centre(overlay)
        overlay._on_press(_event(overlay, cx, cy))
        overlay._on_release(_event(overlay, cx, cy))

        overlay.detach()

        assert overlay._pending_toggle is None
        assert not overlay._toggle_timer.isActive()
        assert toggled == []


# ── paint by dragging with the modifier held ─────────────────────────────


def _shift(event):
    """Attach a Qt event reporting Shift, the way an embedded canvas delivers one.

    `_modifiers_from_event` reads `guiEvent` rather than matplotlib's `key`, because in
    an embedded canvas the Qt state is the reliable one -- so a test that set `key`
    would be exercising a path the app never takes.
    """
    from types import SimpleNamespace

    from PyQt5.QtCore import Qt

    event.guiEvent = SimpleNamespace(modifiers=lambda: Qt.ShiftModifier)
    return event


def _tile_centre(overlay, tiles, row, col):
    tile = next(t for t in tiles if (t.row, t.col) == (row, col))
    x, y, width, height = overlay._rect_for(tile)
    return x + width / 2, y + height / 2


def _sweep(overlay, points, px_step=50.0):
    """Shift-press at the first point, drag through the rest, release at the last."""
    overlay._on_press(_shift(_event(overlay, *points[0])))
    for step, point in enumerate(points[1:], start=1):
        overlay._on_drag(_event(overlay, *point, px=px_step * step, py=0.0))
    last = px_step * max(len(points) - 1, 1)
    overlay._on_release(_event(overlay, *points[-1], px=last, py=0.0))
    _flush_toggle(overlay)


def test_a_shift_drag_paints_every_tile_it_crosses(qapp):
    """One gesture instead of one click per tile -- and one wait on the double-click
    timer instead of one per tile, which is what made clearing a row unusable."""
    overlay, tiles = build(3, 3)
    painted = []
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))

    _sweep(overlay, [_tile_centre(overlay, tiles, 1, col) for col in range(3)])

    assert painted == [(1, 0, False), (1, 1, False), (1, 2, False)]


def test_the_swept_value_is_fixed_at_press_not_re_read_per_tile(qapp):
    """The host writes every emission into the mask and hands back a rebuilt grid, so a
    sweep that asked each tile what it currently was would alternate instead of paint.
    Reproduced by doing exactly what the host does, between motion events.

    The row starts mixed, so a re-reading implementation is forced to disagree: it would
    turn the middle tile *on* while the sweep is turning the row off.
    """
    start = [[True, False, True]]
    overlay, tiles = build(1, 3, mask=start)
    painted = []

    def rewrite(row, col, enabled):
        painted.append((row, col, enabled))
        mask = [list(row_values) for row_values in start]
        for r, c, e in painted:
            mask[r][c] = e
        overlay.set_grid(
            compute_tile_grid_from_fov(
                1, 3, FOV, FOV, WIDTH, HEIGHT, OVERLAP, mask=mask
            ),
            (HEIGHT, WIDTH),
            PIXEL_SIZE,
        )

    overlay.tile_toggled.connect(rewrite)

    _sweep(overlay, [_tile_centre(overlay, tiles, 0, col) for col in range(3)])

    assert painted == [(0, 0, False), (0, 1, False), (0, 2, False)]


def test_a_shift_click_that_never_travels_is_an_ordinary_toggle(qapp):
    """Below the drag threshold the modifier changes nothing. The press has to reach the
    release still looking like a click, so it keeps the deferral that stops a
    double-click editing the tile it moves the stage to (FIB-520)."""
    overlay, tiles = build(3, 3)
    painted = []
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))
    point = _tile_centre(overlay, tiles, 1, 1)

    overlay._on_press(_shift(_event(overlay, *point)))
    overlay._on_drag(_event(overlay, *point, px=MOVE_DRAG_THRESHOLD_PX - 1, py=0.0))
    overlay._on_release(_event(overlay, *point))

    assert painted == [], "a modified click must still wait out the double-click interval"
    _flush_toggle(overlay)
    assert painted == [(1, 1, False)]


def test_a_paint_does_not_also_toggle_the_tile_it_ends_on(qapp):
    """The release would otherwise undo the last tile the sweep just painted."""
    overlay, tiles = build(3, 3)
    painted = []
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))

    _sweep(overlay, [
        _tile_centre(overlay, tiles, 1, 0),
        _tile_centre(overlay, tiles, 1, 1),
    ])

    assert painted == [(1, 0, False), (1, 1, False)]


def test_a_tile_is_painted_once_however_long_the_pointer_lingers(qapp):
    """Motion events arrive far faster than tiles are crossed, and the host rebuilds the
    whole grid on each emission."""
    overlay, tiles = build(3, 3)
    painted = []
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))
    x, y = _tile_centre(overlay, tiles, 1, 1)

    overlay._on_press(_shift(_event(overlay, x, y)))
    for step in range(1, 7):
        overlay._on_drag(_event(overlay, x + step, y, px=50.0, py=0.0))
    overlay._on_release(_event(overlay, x, y, px=50.0, py=0.0))
    _flush_toggle(overlay)

    assert painted == [(1, 1, False)]


def test_an_unmodified_drag_still_moves_the_grid(qapp):
    """The modifier is what distinguishes the two: without it, an interior drag has to
    keep meaning "move", which is the gesture that was there first."""
    overlay, tiles = build(3, 3)
    moved, painted = [], []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))

    _drag(
        overlay,
        _tile_centre(overlay, tiles, 1, 1),
        _tile_centre(overlay, tiles, 1, 2),
    )

    assert moved, "an unmodified interior drag must still move the grid"
    assert painted == []


def test_a_modified_edge_press_still_resizes(qapp):
    """Edges win over the interior for a press, and holding the modifier does not
    change that -- the cursor says resize, so a press there must resize."""
    overlay, _ = build(3, 3)
    left, top, _, height = overlay._bounds()

    overlay._on_press(_shift(_event(overlay, left, top + height / 2)))

    assert overlay.is_resizing
    assert overlay._paint_value is None


def test_the_modifier_turns_the_interior_cursor_into_a_cross(qapp):
    """The only sign the gesture exists, for anyone who has not been told."""
    from PyQt5.QtCore import Qt

    overlay, tiles = build(3, 3)
    point = _tile_centre(overlay, tiles, 1, 1)

    overlay._update_cursor(_event(overlay, *point))
    assert overlay._canvas.cursor == Qt.SizeAllCursor

    overlay._update_cursor(_shift(_event(overlay, *point)))
    assert overlay._canvas.cursor == Qt.CrossCursor


def test_a_paint_survives_the_grid_being_cleared_underneath_it(qapp):
    """`_refresh_tile_grid` clears the grid when the camera geometry cannot be read, and
    that can land between two motion events of a sweep."""
    overlay, tiles = build(3, 3)
    painted = []
    overlay.tile_toggled.connect(lambda r, c, e: painted.append((r, c, e)))
    first = _tile_centre(overlay, tiles, 1, 0)
    second = _tile_centre(overlay, tiles, 1, 1)

    overlay._on_press(_shift(_event(overlay, *first)))
    overlay._on_drag(_event(overlay, *second, px=50.0, py=0.0))
    overlay.clear()
    overlay._on_drag(_event(overlay, *second, px=80.0, py=0.0))
    overlay._on_release(_event(overlay, *second, px=80.0, py=0.0))
    _flush_toggle(overlay)

    assert painted == [(1, 0, False), (1, 1, False)]


def test_detaching_mid_paint_does_not_leave_the_gesture_armed(qapp):
    """Detaching skips the release that clears this, so a re-attached overlay would
    resume painting on the next motion event, with a value captured against a mask that
    no longer exists."""
    overlay, tiles = build(3, 3)
    overlay._on_press(_shift(_event(overlay, *_tile_centre(overlay, tiles, 1, 1))))

    overlay.detach()

    assert overlay._paint_value is None
    assert overlay._painted == set()


# ── a grid whose geometry belongs to someone else ────────────────────────
#
# The sparse selection dialog derives rows, columns and centre from regions drawn on a
# beam overview, so a grid dragged in its preview pane would spring back the moment the
# plan was recomputed. `set_grid_editable(False)` withdraws the two geometry gestures
# and leaves everything else, which is the only combination that pane can use.


def test_the_grid_is_editable_by_default(qapp):
    """The flag is opt-out; every existing caller wants the drag."""
    overlay, _ = build(3, 3)
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))

    assert overlay.is_resizing is True


def test_a_fixed_grid_cannot_be_resized(qapp):
    overlay, _ = build(3, 3, overlap=OVERLAP)
    overlay.set_grid_editable(False)
    requested = []
    overlay.grid_resize_requested.connect(lambda r, c: requested.append((r, c)))
    right, _ = _edges(overlay)

    overlay._on_press(_event(overlay, right, overlay._rect.cy))
    overlay._on_drag(_event(overlay, right + WIDTH, overlay._rect.cy))

    assert overlay.is_resizing is False
    assert requested == []


def test_a_fixed_grid_cannot_be_moved(qapp):
    overlay, _ = build(3, 3)
    overlay.set_grid_editable(False)
    moved = []
    overlay.grid_move_requested.connect(lambda x, y: moved.append((x, y)))
    cx, cy = _centre(overlay)

    _drag(overlay, (cx, cy), (cx + 300.0, cy - 125.0))

    assert moved == []


def test_a_fixed_grid_can_still_have_its_tiles_toggled(qapp):
    """The point of the flag: hand adjustment survives, geometry does not.

    A press inside the grid is held rather than acted on, because the same press starts
    a move and a toggle -- so withdrawing the move must not also withdraw what the
    release reads.
    """
    overlay, tiles = build(3, 3)
    overlay.set_grid_editable(False)
    toggled = []
    overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))
    centre = next(t for t in tiles if (t.row, t.col) == (1, 1))
    x, y, width, height = overlay._rect_for(centre)

    _click(overlay, x + width / 2, y + height / 2)

    assert toggled == [(1, 1, False)]


def test_a_fixed_grid_offers_no_resize_cursor(qapp):
    """An affordance for a gesture the host discards is worse than none: the only way
    to learn it does nothing is to try it."""
    overlay, _ = build(3, 3)
    overlay.set_grid_editable(False)
    right, _ = _edges(overlay)

    overlay._update_cursor(_event(overlay, right, overlay._rect.cy))

    assert overlay._canvas.cursor is None
