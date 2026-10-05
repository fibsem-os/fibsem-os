"""An image laid over the canvas through its own transform (FIB-1030).

The pixels are never resampled: an ``AxesImage`` on the unit square is mapped onto the
footprint and then through the body's map, so what these tests hold is that map --
where the corners land, that a mirror is a mirror and not a turn, that a squashed view
squashes the footprint and nothing else -- and that the gesture the base class gives
every body reaches this one.
"""

import math
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.canvas_base import ContentRect
from fibsem.ui.widgets.canvas.overlays.image_overlay import ImageOverlay
from fibsem.ui.widgets.canvas.overlays.transform_overlay import MIN_CORNER_REACH_PX

W, H = 200.0, 100.0


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeCanvas:
    def __init__(self):
        self._active_overlay = None
        self._overlay_consuming_event = False

    def draw_idle(self):
        pass

    def setCursor(self, cursor):  # noqa: N802
        pass

    def unsetCursor(self):  # noqa: N802
        pass


def _rgb():
    data = np.zeros((10, 20, 3), dtype=np.uint8)
    data[0, :, 0] = 255  # a red top row, so "which way is up" is answerable
    return data


def build(rotation=0.0, squash=1.0, mirror=False, active=True):
    from matplotlib.figure import Figure

    overlay = ImageOverlay()
    fig = Figure(figsize=(8, 8), dpi=100)
    overlay._ax = fig.add_subplot(111)
    overlay._ax.set_xlim(0, 800)
    overlay._ax.set_ylim(800, 0)
    overlay._canvas = _FakeCanvas()
    if active:
        overlay._canvas._active_overlay = overlay
    overlay.on_content_changed(ContentRect(0.0, 0.0, 800.0, 800.0))
    overlay.set_visible(True)
    overlay.set_image(
        _rgb(),
        W,
        H,
        centre=(400.0, 400.0),
        rotation=rotation,
        squash=squash,
        mirror=mirror,
    )
    return overlay


def _image_artist(overlay):
    from matplotlib.image import AxesImage

    return next(a for a in overlay._artists if isinstance(a, AxesImage))


def _drawn_corner(overlay, u01, v01):
    """Where a point of the unit square the image lives on lands, in canvas units."""
    artist = _image_artist(overlay)
    display = artist.get_transform().transform((u01, v01))
    return tuple(overlay._ax.transData.inverted().transform(display))


class TestTheImageIsDrawnThroughTheBodyMap:
    def test_the_footprint_is_centred_and_the_top_row_is_at_the_top(self, qapp):
        overlay = build()
        # v=0 on the unit square is row 0 with origin="lower"; on a y-down canvas
        # that must be the footprint's *top* edge, i.e. the smaller y.
        assert _drawn_corner(overlay, 0.0, 0.0) == pytest.approx((300.0, 350.0))
        assert _drawn_corner(overlay, 1.0, 1.0) == pytest.approx((500.0, 450.0))
        assert _drawn_corner(overlay, 0.5, 0.5) == pytest.approx((400.0, 400.0))

    def test_row_zero_is_rendered_at_the_top(self, qapp):
        """The transform tests above cannot tell which *row* lands where: that is
        `origin`, which only a render answers. Draw it and look for the red row."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        overlay = build(active=False)
        overlay.set_opacity(1.0)  # blended with the background the red is not red
        fig = overlay._ax.figure
        agg = FigureCanvasAgg(fig)
        agg.draw()
        buffer = np.asarray(agg.buffer_rgba())
        red = (buffer[..., 0] > 200) & (buffer[..., 1] < 80) & (buffer[..., 2] < 80)
        rows = np.nonzero(red)[0]
        assert rows.size, "the red row was not drawn"
        # Buffer row 0 is the top of the figure; the footprint's centre is at canvas
        # y=400 and its top edge at y=350, which is *higher* on the figure.
        centre_row = (
            fig.bbox.height - overlay._ax.transData.transform((400.0, 400.0))[1]
        )
        top_row = fig.bbox.height - overlay._ax.transData.transform((400.0, 350.0))[1]
        assert abs(rows.mean() - top_row) < abs(rows.mean() - centre_row)

    def test_a_turned_image_has_clear_corners(self, qapp):
        """Rotated, the image is resampled into an axis-aligned buffer; without an
        alpha channel matplotlib painted the corners outside the footprint black."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        overlay = build(rotation=30.0, active=False)
        overlay.set_opacity(1.0)
        fig = overlay._ax.figure
        fig.patch.set_facecolor("white")
        overlay._ax.set_facecolor("white")
        agg = FigureCanvasAgg(fig)
        agg.draw()
        buffer = np.asarray(agg.buffer_rgba())
        # The top-left corner of the footprint's *bounding box* lies outside the
        # turned footprint: it must show the white axes, not black.
        xs = [c[0] for c in overlay.corners()]
        ys = [c[1] for c in overlay.corners()]
        corner = overlay._ax.transData.transform((min(xs) + 2.0, min(ys) + 2.0))
        row, col = int(fig.bbox.height - corner[1]), int(corner[0])
        assert buffer[row, col, :3].min() > 200, buffer[row, col]

    @pytest.mark.parametrize("opacity", [1.0, 0.6])
    def test_a_clear_pixel_stays_clear_at_any_opacity(self, qapp, opacity):
        """The opacity multiplies an RGBA image's own alpha rather than replacing
        it -- what lets a fluorescence image be drawn signal only, dark as clear."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        overlay = build(active=False)
        data = np.zeros((10, 20, 4), dtype=np.uint8)
        data[:5, :, 1] = 255  # the top half green and solid
        data[:5, :, 3] = 255  # the bottom half clear
        overlay.set_image(data, W, H)
        overlay.set_opacity(opacity)
        fig = overlay._ax.figure
        fig.patch.set_facecolor("white")
        overlay._ax.set_facecolor("white")
        agg = FigureCanvasAgg(fig)
        agg.draw()
        buffer = np.asarray(agg.buffer_rgba())

        def pixel(x, y):
            px, py = overlay._ax.transData.transform((x, y))
            return buffer[int(fig.bbox.height - py), int(px)]

        assert pixel(400.0, 430.0)[:3].min() == 255  # clear: the white axes
        green = pixel(400.0, 370.0)
        assert green[1] == 255
        assert green[0] == pytest.approx(255 * (1 - opacity), abs=2)

    def test_the_drawn_corners_are_the_corners_the_overlay_reports(self, qapp):
        overlay = build(rotation=33.0, squash=0.7)
        drawn = [
            _drawn_corner(overlay, u, v)
            for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        ]
        for got, expected in zip(drawn, overlay.corners()):
            assert got == pytest.approx(expected)

    def test_a_turn_turns_the_footprint(self, qapp):
        overlay = build(rotation=90.0)
        (x0, y0), (x1, y1) = overlay.corners()[0], overlay.corners()[1]
        # The top edge, W long, now runs down the screen.
        assert (x1 - x0, y1 - y0) == pytest.approx((0.0, W))

    def test_a_mirror_flips_x_and_is_not_a_turn(self, qapp):
        plain, mirrored = build(), build(mirror=True)
        # The top-left corner of the image lands where the top-right did ...
        assert mirrored.corners()[0] == pytest.approx(plain.corners()[1])
        # ... and the top row stays at the top: a mirror, not a half turn.
        assert _drawn_corner(mirrored, 0.5, 0.0)[1] == pytest.approx(350.0)

    def test_the_squash_shortens_only_the_view_axis(self, qapp):
        overlay = build(squash=0.5)
        top_left, top_right, bottom_right, _ = overlay.corners()
        assert math.hypot(*np.subtract(top_right, top_left)) == pytest.approx(W)
        assert math.hypot(*np.subtract(bottom_right, top_right)) == pytest.approx(
            H * 0.5
        )

    def test_opacity_reaches_the_artist(self, qapp):
        overlay = build()
        overlay.set_opacity(0.25)
        assert _image_artist(overlay).get_alpha() == pytest.approx(0.25)

    def test_swapping_the_pixels_keeps_the_placement(self, qapp):
        overlay = build(rotation=12.0)
        overlay.set_image(np.zeros((5, 5, 4), dtype=np.uint8), W, H)
        assert overlay.rotation == 12.0
        assert overlay.centre == (400.0, 400.0)

    def test_the_outline_is_drawn_only_while_active(self, qapp):
        # The image, its outline, the rotate handle's three and the corners' one.
        assert len(build(active=True)._artists) == 1 + 1 + 3 + 1
        assert len(build(active=False)._artists) == 1


class TestTheGestureReachesTheImage:
    @staticmethod
    def _event(overlay, x, y, px=0.0, py=0.0):
        return SimpleNamespace(
            button=1, inaxes=overlay._ax, xdata=x, ydata=y, x=px, y=py
        )

    def test_a_press_inside_the_footprint_drags_it(self, qapp):
        overlay = build()
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        overlay._on_press(self._event(overlay, 350.0, 380.0))
        overlay._on_motion(self._event(overlay, 360.0, 370.0, px=40.0, py=40.0))
        overlay._on_release(self._event(overlay, 360.0, 370.0, px=40.0, py=40.0))
        assert seen[-1] == pytest.approx((410.0, 390.0))

    def test_a_press_outside_the_footprint_is_left_to_the_canvas(self, qapp):
        overlay = build()
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        overlay._on_press(self._event(overlay, 100.0, 100.0))
        overlay._on_motion(self._event(overlay, 130.0, 80.0, px=40.0, py=40.0))
        assert seen == []
        assert not overlay._canvas._overlay_consuming_event

    def test_inside_is_judged_through_the_squash_and_mirror(self, qapp):
        overlay = build(squash=0.5, mirror=True)
        # Half a height below the centre on the *sample* is a quarter on the canvas.
        assert overlay._body_contains(400.0, 400.0 + H * 0.25 - 1)
        assert not overlay._body_contains(400.0, 400.0 + H * 0.25 + 1)


class TestTheCornersScaleTheImage:
    """A corner is taken to change the image's size -- its pixel size, in effect --
    uniformly and about its centre. What the overlay emits is a factor on the size it
    is drawn at now; the host multiplies and re-places."""

    @staticmethod
    def _event(overlay, x, y, px=0.0, py=0.0):
        return SimpleNamespace(
            button=1, inaxes=overlay._ax, xdata=x, ydata=y, x=px, y=py
        )

    def _drag_corner(self, overlay, corner, to):
        seen = []
        overlay.scaled.connect(seen.append)
        overlay._on_press(self._event(overlay, *overlay.corners()[corner]))
        overlay._on_motion(self._event(overlay, *to, px=40.0, py=40.0))
        return seen

    def test_a_corner_dragged_out_scales_by_how_far_out_it_went(self, qapp):
        overlay = build()
        # The bottom-right corner is (500, 450); half as far out again is (550, 475).
        assert self._drag_corner(overlay, 2, (550.0, 475.0)) == [pytest.approx(1.5)]

    def test_a_sideways_wander_does_not_change_the_size(self, qapp):
        overlay = build()
        # Across the corner's diagonal (100, 50), not along it.
        assert self._drag_corner(overlay, 2, (490.0, 470.0)) == [pytest.approx(1.0)]

    @pytest.mark.parametrize("corner", [0, 1, 2, 3])
    def test_every_corner_is_read_through_the_turn_squash_and_mirror(
        self, qapp, corner
    ):
        overlay = build(rotation=30.0, squash=0.5, mirror=True)
        u, v = overlay._footprint_corners()[corner]
        to = overlay.to_canvas(1.2 * u, 1.2 * v)
        assert self._drag_corner(overlay, corner, to) == [pytest.approx(1.2)]

    def test_a_corner_dragged_through_the_centre_stops_short_of_it(self, qapp):
        overlay = build()
        seen = self._drag_corner(overlay, 2, (300.0, 350.0))
        reach = math.hypot(W / 2.0, H / 2.0) * overlay._pixels_per_unit()
        assert seen == [pytest.approx(MIN_CORNER_REACH_PX / reach)]
        assert 0.0 < seen[0] < 1.0

    def test_an_image_already_below_the_floor_is_not_grown_by_taking_a_corner(
        self, qapp
    ):
        overlay = build()
        overlay.set_image(_rgb(), 20.0, 10.0)
        reach = math.hypot(10.0, 5.0) * overlay._pixels_per_unit()
        assert reach < MIN_CORNER_REACH_PX, "not small enough to test the floor"
        # Halfway in from the bottom-right corner at (410, 405).
        assert self._drag_corner(overlay, 2, (405.0, 402.5)) == [pytest.approx(1.0)]

    def test_a_press_on_a_corner_is_a_scale_and_not_a_move(self, qapp):
        overlay = build()
        moved, finished = [], []
        overlay.moved.connect(lambda x, y: moved.append((x, y)))
        overlay.drag_finished.connect(lambda: finished.append(True))
        seen = self._drag_corner(overlay, 0, (280.0, 330.0))
        assert overlay._canvas._overlay_consuming_event
        assert overlay.is_dragging
        overlay._on_release(self._event(overlay, 280.0, 330.0))
        assert moved == []
        assert finished == [True]
        assert not overlay.is_dragging

        # Released, the pointer moving on scales nothing.
        overlay._on_motion(self._event(overlay, 200.0, 300.0))
        assert len(seen) == 1

    def test_the_corners_are_offered_only_while_active(self, qapp):
        overlay = build(active=False)
        seen = self._drag_corner(overlay, 2, (550.0, 475.0))
        assert seen == []

    @pytest.mark.parametrize(
        "rotation, corner, cursor",
        [
            (0.0, 0, "SizeFDiagCursor"),  # top left: the "\" diagonal
            (0.0, 1, "SizeBDiagCursor"),  # top right: the "/" diagonal
            (0.0, 2, "SizeFDiagCursor"),
            (90.0, 0, "SizeBDiagCursor"),  # a quarter turn swaps them
        ],
    )
    def test_the_cursor_points_the_way_the_corner_lies(
        self, qapp, rotation, corner, cursor
    ):
        from PyQt5.QtCore import Qt

        overlay = build(rotation=rotation)
        assert overlay._corner_cursor(corner) == getattr(Qt, cursor)
