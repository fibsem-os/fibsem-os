"""The real-space canvas: images drawn where they were acquired.

The thing worth pinning is that placement is *truthful* — two images acquired a known
distance apart must land that distance apart on screen, in the right direction, at the
right relative size. A canvas that draws them plausibly but wrongly is worse than one
that draws nothing, because it looks like evidence.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python tests/ui/test_real_space_canvas.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.canvas_base import FibsemCanvasBase
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

_app = QApplication.instance() or QApplication(sys.argv)

PIXEL_SIZE = 1e-7  # 100 nm/px
W = H = 100  # so one image spans 10 um


def _img(h=H, w=W):
    return np.zeros((h, w), dtype=np.uint8)


def _canvas(**kw):
    return FibsemRealSpaceCanvas(**kw)


# ── structure ─────────────────────────────────────────────────────────────


def test_it_is_a_base_canvas():
    assert issubclass(FibsemRealSpaceCanvas, FibsemCanvasBase)


def test_an_empty_canvas_has_no_content_and_survives_a_fit():
    c = _canvas()
    assert c._content_extent() is None
    assert c._content_rect().is_empty
    c.reset_view()  # must not raise


def test_the_first_image_sets_the_reference_scale():
    """So a canvas showing one image has axes in that image's own pixels."""
    c = _canvas()
    assert c.reference_pixel_size is None
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert c.reference_pixel_size == PIXEL_SIZE
    xmin, xmax, ymax, ymin = c._content_extent()
    assert (xmax - xmin, ymax - ymin) == (W, H)  # one canvas pixel per image pixel


# ── placement ─────────────────────────────────────────────────────────────


def test_an_image_is_centred_on_its_position():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    xmin, xmax, ymax, ymin = c._content_extent()
    assert (xmin, xmax) == (-W / 2, W / 2)
    assert (ymin, ymax) == (-H / 2, H / 2)


def test_two_images_land_the_right_distance_apart():
    """The core claim: a 10 um separation is 10 um on the canvas."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="a")
    c.add_image(_img(), centre=(10e-6, 0.0), pixel_size=PIXEL_SIZE, key="b")
    a, b = c._placed["a"].extent, c._placed["b"].extent
    centre_a = (a[0] + a[1]) / 2
    centre_b = (b[0] + b[1]) / 2
    assert centre_b - centre_a == pytest.approx(10e-6 / PIXEL_SIZE)  # 100 canvas px


def test_positive_x_is_right_and_positive_y_is_down():
    """Direction, not just distance — a sign error here mirrors the whole mosaic."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="origin")
    c.add_image(_img(), centre=(5e-6, 3e-6), pixel_size=PIXEL_SIZE, key="offset")
    o, f = c._placed["origin"].extent, c._placed["offset"].extent
    assert (f[0] + f[1]) / 2 > (o[0] + o[1]) / 2  # +x drew further right
    assert (f[2] + f[3]) / 2 > (o[2] + o[3]) / 2  # +y drew further down


def test_a_coarser_image_covers_proportionally_more_ground():
    """Different binning must compose at true relative size, not equal size."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="fine")
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE * 4, key="coarse")
    fine = c._placed["fine"].extent
    coarse = c._placed["coarse"].extent
    assert (coarse[1] - coarse[0]) == pytest.approx(4 * (fine[1] - fine[0]))


def test_a_non_square_image_keeps_its_aspect():
    c = _canvas()
    c.add_image(_img(h=50, w=200), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    xmin, xmax, ymax, ymin = c._content_extent()
    assert (xmax - xmin, ymax - ymin) == (200, 50)


def test_placement_is_absolute_not_relative_to_the_previous_image():
    """Positions are absolute, so nothing accumulates drift as tiles arrive (FIB-399)."""
    c = _canvas()
    step = 9.999e-6  # deliberately not a whole number of canvas pixels
    for i in range(10):
        c.add_image(_img(), centre=(i * step, 0.0), pixel_size=PIXEL_SIZE, key=str(i))
    first, last = c._placed["0"].extent, c._placed["9"].extent
    spread = (last[0] + last[1]) / 2 - (first[0] + first[1]) / 2
    assert spread == pytest.approx(9 * step / PIXEL_SIZE)


# ── the union ─────────────────────────────────────────────────────────────


def test_the_content_extent_is_the_union():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.add_image(_img(), centre=(10e-6, 10e-6), pixel_size=PIXEL_SIZE)
    xmin, xmax, ymax, ymin = c._content_extent()
    assert (xmin, ymin) == (-W / 2, -H / 2)  # top-left of the first
    assert (xmax, ymax) == (100 + W / 2, 100 + H / 2)  # bottom-right of the second


def test_the_content_rect_matches_the_union_and_is_offset():
    """Overlays clamp and anchor to this — it must carry the origin, not assume (0, 0)."""
    c = _canvas()
    c.add_image(_img(), centre=(50e-6, 20e-6), pixel_size=PIXEL_SIZE)
    rect = c._content_rect()
    assert (rect.x0, rect.y0) == pytest.approx((500 - W / 2, 200 - H / 2))
    assert (rect.width, rect.height) == pytest.approx((W, H))
    assert (rect.cx, rect.cy) == pytest.approx((500, 200))  # what an overlay anchors on


def test_fit_view_frames_everything_placed():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.add_image(_img(), centre=(50e-6, 0.0), pixel_size=PIXEL_SIZE)
    c.reset_view()
    x0, x1 = c._ax.get_xlim()
    assert x0 <= -W / 2 and x1 >= 500 + W / 2


# ── lifecycle ─────────────────────────────────────────────────────────────


def test_reusing_a_key_replaces_rather_than_stacks():
    """A live preview pushes a frame per tick; stacking artists would leak them."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="preview")
    c.add_image(_img(), centre=(20e-6, 0.0), pixel_size=PIXEL_SIZE, key="preview")
    assert c.placed_keys == ["preview"]
    assert len(c._ax.get_images()) == 1
    assert c._content_rect().cx == pytest.approx(200)  # moved to the new position


def test_distinct_images_accumulate():
    c = _canvas()
    for i in range(3):
        c.add_image(_img(), centre=(i * 20e-6, 0.0), pixel_size=PIXEL_SIZE)
    assert len(c.placed_keys) == 3
    assert len(c._ax.get_images()) == 3


def test_removing_an_image_shrinks_the_union():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="a")
    c.add_image(_img(), centre=(50e-6, 0.0), pixel_size=PIXEL_SIZE, key="b")
    assert c.remove_image("b") is True
    assert c._content_rect().cx == 0
    assert c.remove_image("b") is False  # already gone


def test_clear_images_empties_the_canvas_but_keeps_overlays():
    from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

    class _Rec(CanvasOverlay):
        def __init__(self):
            self.rects = []

        def on_content_changed(self, rect):
            self.rects.append(rect)

    c = _canvas()
    rec = _Rec()
    c.add_overlay(rec)
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.clear_images()
    assert c.placed_keys == []
    assert c._content_extent() is None
    assert rec in c._overlays
    assert rec.rects[-1].is_empty  # told the content went away


def test_there_is_no_placeholder_text():
    """The black backdrop already says "nothing acquired here", and keeps saying it once
    images arrive — the gaps between them mean the same thing. Text would also sit in the
    middle of the tile grid, which is drawn before any acquisition."""
    c = _canvas()
    assert len(c._ax.texts) == 0
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.clear_images()
    assert len(c._ax.texts) == 0


# ── validation and conversion ─────────────────────────────────────────────


def test_a_nonsense_pixel_size_is_rejected():
    c = _canvas()
    for bad in (0, -1e-7, None):
        with pytest.raises(ValueError):
            c.add_image(_img(), centre=(0.0, 0.0), pixel_size=bad)


def test_one_dimensional_data_is_rejected():
    c = _canvas()
    with pytest.raises(ValueError):
        c.add_image(np.zeros(10), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)


def test_rgb_and_rgba_are_placed_like_any_other_picture():
    """The composited path: an FM overview arrives here already blended to colour."""
    c = _canvas()
    c.add_image(np.zeros((64, 128, 3), np.uint8), centre=(0.0, 0.0),
                pixel_size=PIXEL_SIZE, key="rgb")
    c.add_image(np.zeros((64, 128, 4), np.uint8), centre=(0.0, 0.0),
                pixel_size=PIXEL_SIZE, key="rgba")
    rect = c._content_rect()
    assert (rect.width, rect.height) == (128, 64)  # the channel axis is not a dimension
    assert c._placed["rgb"].artist.get_array().shape[-1] == 3


def test_the_display_cap_keeps_the_channel_axis():
    c = FibsemRealSpaceCanvas(display_max_px=128)
    c.add_image(np.zeros((1024, 1024, 3), np.uint8), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert c._ax.get_images()[0].get_array().shape == (128, 128, 3)


def test_stacks_are_rejected_rather_than_misplaced():
    """z and channels have no single answer once many images share a view, so they are
    resolved upstream. Left to matplotlib this raises from inside imshow instead."""
    c = _canvas()
    for bad in (np.zeros((5, 64, 64)), np.zeros((2, 5, 64, 64)), np.zeros((64, 64, 7))):
        with pytest.raises(ValueError, match="Project z and composite channels"):
            c.add_image(bad, centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)


def test_a_rejected_image_does_not_destroy_the_one_it_would_have_replaced():
    """add_image drops the old artist before drawing the new one, so validating late
    would mean a bad call silently deleting a good picture."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="keep")
    with pytest.raises(ValueError):
        c.add_image(np.zeros((5, 64, 64)), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="keep")
    assert c.placed_keys == ["keep"]
    assert len(c._ax.get_images()) == 1


def test_metres_and_canvas_coordinates_round_trip():
    c = _canvas(reference_pixel_size=PIXEL_SIZE)
    assert c.metres_to_canvas(5e-6, -2e-6) == pytest.approx((50.0, -20.0))
    assert c.canvas_to_metres(50.0, -20.0) == pytest.approx((5e-6, -2e-6))


def test_conversion_is_inert_before_a_reference_exists():
    assert _canvas().metres_to_canvas(1.0, 1.0) == (0.0, 0.0)


# ── auto-fit ──────────────────────────────────────────────────────────────


def test_auto_fit_can_be_turned_off_to_keep_the_users_zoom():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.auto_fit = False
    c._ax.set_xlim(0, 10)
    c.add_image(_img(), centre=(500e-6, 0.0), pixel_size=PIXEL_SIZE)
    assert c._ax.get_xlim() == (0, 10)  # the new tile did not re-frame the view


# ── the declared working area ─────────────────────────────────────────────


def test_a_world_extent_frames_at_least_that_much_space():
    """Without one the view hugs the content, so framing jumps as each tile lands."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.set_world_extent(200e-6)
    rect = c._content_rect()
    assert (rect.width, rect.height) == pytest.approx((2000, 2000))  # 200 um at 100 nm/px
    assert (rect.cx, rect.cy) == pytest.approx((0, 0))


def test_a_world_extent_is_a_minimum_not_a_clip():
    """Content beyond the declared area must still be framed, not cropped away."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.set_world_extent(50e-6)
    c.add_image(_img(), centre=(500e-6, 0.0), pixel_size=PIXEL_SIZE, key="far")
    rect = c._content_rect()
    assert rect.x1 >= 5000 + W / 2  # the distant image is still inside the frame


def test_a_world_extent_bounds_the_overlays_not_the_view():
    """The two extents answer different questions and must not be conflated: overlays
    clamp and anchor to the space the canvas represents, while the view frames the data.
    """
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.set_world_extent(500e-6)
    assert c._content_rect().width == pytest.approx(5000)  # overlays see the whole area
    assert c._image_extent() == pytest.approx((-W / 2, W / 2, H / 2, -H / 2))


def test_fit_frames_the_images_not_the_declared_working_area():
    """Fitting to the working area would zoom past the data whenever the declared area
    is bigger than what has been acquired — which is the normal case."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.set_world_extent(500e-6)
    xmin, xmax, _, _ = c._fit_extent()
    assert xmax - xmin < 5000  # nowhere near the 500 um working area
    c.clear_images()
    assert c._fit_extent() is not None  # with nothing acquired, the area is all there is


def test_a_rectangular_and_offset_working_area():
    c = _canvas(reference_pixel_size=PIXEL_SIZE)
    c.set_world_extent(100e-6, 40e-6, centre=(10e-6, -20e-6))
    rect = c._content_rect()
    assert (rect.width, rect.height) == pytest.approx((1000, 400))
    assert (rect.cx, rect.cy) == pytest.approx((100, -200))


def test_clearing_the_world_extent_returns_to_fitting_the_content():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    c.set_world_extent(500e-6)
    assert c._content_rect().width == pytest.approx(5000)
    c.set_world_extent(None)
    assert c._content_rect().width == pytest.approx(W)
    assert c.world_extent is None


def test_a_world_extent_waits_for_a_reference_pixel_size():
    """Set on an empty canvas there is nothing to scale against; it must apply as soon
    as the first image supplies the scale, rather than being silently dropped."""
    c = _canvas()
    c.set_world_extent(200e-6)
    assert c._content_extent() is None  # nothing to scale it against yet
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert c._content_rect().width == pytest.approx(2000)


def test_a_nonsense_world_extent_is_rejected():
    c = _canvas()
    for bad in (0, -1e-6):
        with pytest.raises(ValueError):
            c.set_world_extent(bad)


# ── framing and the crosshair ─────────────────────────────────────────────


def test_a_long_thin_mosaic_still_fills_the_widget():
    """aspect="equal" keeps pixels square, so a framed region shaped unlike the widget
    collapses the axes box instead — 39:1 content left an axes 2% of the window wide.
    The fit pads the short axis so the box stays full and the slack shows as backdrop.
    """
    c = _canvas()
    c.resize(1000, 900)
    for i in range(20):
        c.add_image(_img(), centre=(0.0, i * 20e-6), pixel_size=PIXEL_SIZE, key=str(i))
    images = c._image_extent()
    assert (images[2] - images[3]) / (images[1] - images[0]) > 30  # genuinely degenerate

    fit = c._fit_extent()
    assert (fit[2] - fit[3]) / (fit[1] - fit[0]) == pytest.approx(900 / 1000, rel=0.01)


def test_padding_the_fit_never_hides_an_image():
    """Padding may only grow the framed region — cropping to fit would lose data."""
    c = _canvas()
    c.resize(1000, 900)
    for i in range(6):
        c.add_image(_img(), centre=(0.0, i * 20e-6), pixel_size=PIXEL_SIZE, key=str(i))
    images, fit = c._image_extent(), c._fit_extent()
    assert fit[0] <= images[0] and fit[1] >= images[1]
    assert fit[3] <= images[3] and fit[2] >= images[2]


def test_the_geometry_survives_a_resize_with_nothing_acquired():
    """imshow sets aspect="equal" per artist, so an empty canvas would be "auto" and the
    axes would stretch to the widget — drawing a square grid as a rectangle. Overlays are
    drawn here before anything is acquired, so the frame has to be square from the start.
    """
    def _ratio(c):
        x0, x1 = c._ax.get_xlim()
        y0, y1 = c._ax.get_ylim()
        box = c._ax.get_window_extent()
        return (box.width / abs(x1 - x0)) / (box.height / abs(y1 - y0))

    c = _canvas()
    c.resize(1200, 600)
    c.show()
    c.set_reference_pixel_size(PIXEL_SIZE)
    c.set_world_extent(200e-6)
    _app.processEvents()
    assert _ratio(c) == pytest.approx(1.0, abs=0.01)

    c.resize(600, 900)  # a resize alone used to distort this 3x
    _app.processEvents()  # Qt has to lay the resize out before the axes re-fits to it
    assert _ratio(c) == pytest.approx(1.0, abs=0.01)


def test_a_reference_pixel_size_can_be_fixed_before_any_image():
    """So a planned grid or stage markers have a real frame to be drawn in."""
    c = _canvas()
    assert c.set_reference_pixel_size(PIXEL_SIZE) is True
    assert c.metres_to_canvas(10e-6, 0.0) == pytest.approx((100.0, 0.0))


def test_the_reference_pixel_size_is_not_changed_under_a_placed_image():
    """Their extents were computed against it, so changing it would move them."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert c.set_reference_pixel_size(PIXEL_SIZE * 4) is False
    assert c.reference_pixel_size == PIXEL_SIZE


def test_the_crosshair_marks_the_origin_not_the_middle_of_the_content():
    """The centre of a union of tiles is wherever they happen to average out; the origin
    is the position the caller built the canvas around."""
    c = _canvas()
    c.add_image(_img(), centre=(500e-6, 500e-6), pixel_size=PIXEL_SIZE)  # far off-origin
    assert c._content_rect().cx == pytest.approx(5000)  # content is nowhere near (0, 0)
    (marker,) = c._crosshair_artists
    xs, ys = marker.get_data()
    assert (xs[0], ys[0]) == (0.0, 0.0)


def test_the_crosshair_does_not_scale_with_the_working_area():
    """Arms scaled to the content made the crosshair 5% of a 2 mm area — 100 um across."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    small = c._crosshair_artists[0].get_markersize()
    c.set_world_extent(2000e-6)
    assert c._crosshair_artists[0].get_markersize() == small  # fixed on screen


def test_the_crosshair_can_still_be_hidden():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert c._crosshair_artists
    c.set_crosshair_visible(False)
    assert c._crosshair_artists == []


# ── draw order and in-place updates ───────────────────────────────────────


def test_later_images_draw_over_earlier_ones_by_default():
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="under")
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="over")
    order = [im.get_zorder() for im in c._ax.get_images()]
    assert c._ax.get_images()[-1] is c._placed["over"].artist
    assert order == sorted(order)


def test_zorder_puts_a_coarse_overview_beneath_later_tiles():
    """Arrival order is the wrong answer when a low-res overview is acquired first and
    detailed tiles land on top of it — or acquired last and would bury them."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="tile")
    c.add_image(
        _img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE * 4, key="overview", zorder=-1
    )
    assert c._placed["overview"].artist.get_zorder() < c._placed["tile"].artist.get_zorder()


def test_update_image_keeps_position_scale_and_draw_order():
    """A live preview refreshed by re-adding would climb over tiles it should sit under,
    because destroying the artist and making a new one puts it on top."""
    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="preview", zorder=-1)
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="tile")
    before = c._placed["preview"].extent
    artist = c._placed["preview"].artist

    assert c.update_image("preview", np.full((H, W), 200, dtype=np.uint8)) is True
    assert c._placed["preview"].artist is artist  # same artist, so same z-order
    assert c._placed["preview"].extent == before
    assert c._placed["preview"].artist.get_array().max() == 200  # pixels did change


def test_update_image_reports_an_unknown_key():
    assert _canvas().update_image("nope", _img()) is False


def test_the_default_display_cap_favours_redraw_speed():
    """A redraw costs the total stored pixels, so the default is tuned for the case this
    canvas exists for — many images at once, where 512 px already exceeds what each gets
    on screen. At 1024 px a 100-tile redraw took 2.7 s; at 512 px it takes ~0.7 s."""
    c = _canvas()
    c.add_image(_img(1024, 1024), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    assert max(c._ax.get_images()[0].get_array().shape) <= 512


def test_the_display_cap_bounds_what_each_image_stores():
    """Redraw cost is the total stored pixels across every placed image, so a canvas
    expecting many tiles needs a lower cap than one expecting a few."""
    c = FibsemRealSpaceCanvas(display_max_px=128)
    c.add_image(_img(1024, 1024), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE)
    stored = c._ax.get_images()[0].get_array()
    assert max(stored.shape) <= 128
    # the extent still describes the full physical footprint, not the stored pixels
    xmin, xmax, _, _ = c._content_extent()
    assert xmax - xmin == pytest.approx(1024)


# ── the seam with the real projection ─────────────────────────────────────


def test_a_projected_stage_offset_places_an_image_where_the_projection_says():
    """This canvas deliberately knows nothing about stages — the caller projects, and
    hands over metres in the display plane. This pins that composition, so FIB-412
    wires the two together rather than reimplementing the geometry against it.
    """
    import numpy as np

    from fibsem.fm.reprojection import project_stage_position
    from fibsem.fm.structures import CameraImageTransform, FMImageGeometry
    from fibsem.structures import FibsemStagePosition

    geometry = FMImageGeometry(
        transform=CameraImageTransform.NONE,
        camera_tilt=180.0,
        is_compustage=True,
        shuttle_pre_tilt=0.0,
    )
    base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))
    moved = FibsemStagePosition(x=20e-6, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))
    shape = (H, W)

    # What the caller does: project, then express the offset from the image centre
    # in metres, which is what add_image() takes.
    point = project_stage_position(moved, base, PIXEL_SIZE, shape, geometry)
    centre = ((point.x - W / 2) * PIXEL_SIZE, (point.y - H / 2) * PIXEL_SIZE)

    c = _canvas()
    c.add_image(_img(), centre=(0.0, 0.0), pixel_size=PIXEL_SIZE, key="base")
    c.add_image(_img(), centre=centre, pixel_size=PIXEL_SIZE, key="moved")

    a, b = c._placed["base"].extent, c._placed["moved"].extent
    separation = (b[0] + b[1]) / 2 - (a[0] + a[1]) / 2
    # The 20 um stage step is 200 canvas pixels at 100 nm/px, and lands on the axis
    # and in the direction the projection chose -- not one this canvas invented.
    assert separation == pytest.approx(point.x - W / 2)
    assert abs(separation) == pytest.approx(200)
    assert (b[2] + b[3]) / 2 == pytest.approx((a[2] + a[3]) / 2)  # no drift in y


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print("all passed" if not failed else f"{failed} failed")
