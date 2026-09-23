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
        assert len(build(active=True)._artists) == 1 + 1 + 3  # image, outline, handles
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
