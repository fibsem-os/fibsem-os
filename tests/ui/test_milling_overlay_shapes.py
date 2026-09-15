"""Milling pattern shapes on the canvas overlay.

The napari path drew bitmap patterns as their own image layer; when milling moved to
``FibsemImageCanvas``, ``MillingPatternOverlay`` had no branch for them — every
``isinstance`` check missed, because ``FibsemBitmapSettings`` is not a
``FibsemRectangleSettings`` — so a bitmap stage rendered as its crosshair and nothing
else. These pin the artists back in place, along with the two shape properties the
overlay used to ignore: a circle's ``thickness`` / angles, and ``is_exclusion``.

``test_drawing_a_bitmap_leaves_the_view_alone`` is the guard on *how* the image is
drawn: ``ax.imshow`` routes through ``add_image`` → ``update_datalim`` and rescales
the axes to the pattern, which on a live canvas throws away the operator's pan/zoom.
The overlay builds the ``AxesImage`` itself to avoid that.

The overlay itself talks to a matplotlib axes and needs no Qt, but importing it walks
``fibsem.ui.widgets.canvas``, whose package import does — hence the importorskip, which
the thinner CI builds (no UI extra) rely on.

Run directly:
    QT_QPA_PLATFORM=offscreen python tests/ui/test_milling_overlay_shapes.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Annulus,
    Circle,
    Polygon,
    Rectangle,
    Wedge,
)
from matplotlib.text import Annotation  # noqa: E402

from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.patterning.patterns2 import (  # noqa: E402
    BitmapPattern,
    CirclePattern,
    PolygonPattern,
    RectanglePattern,
    TrenchPattern,
)
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemCircleSettings,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.ui.widgets.canvas.overlays.milling_overlay import (  # noqa: E402
    MillingPatternOverlay,
)

RESOLUTION = 512
HFW = 80e-6
PIXEL_SIZE = HFW / RESOLUTION


class _FakeCanvas:
    """The overlay only ever asks its canvas to redraw."""

    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


def _image() -> FibsemImage:
    image = FibsemImage(data=np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8))
    image.metadata = FibsemImageMetadata(
        image_settings=ImageSettings(
            hfw=HFW, resolution=[RESOLUTION, RESOLUTION], beam_type=BeamType.ION
        ),
        pixel_size=Point(PIXEL_SIZE, PIXEL_SIZE),
        microscope_state=MicroscopeState(),
    )
    return image


def _bitmap_stage(name: str = "bitmap") -> FibsemMillingStage:
    array = np.zeros((32, 32, 2), dtype=float)
    array[:, :, 0] = np.linspace(0, 1, 32)[None, :]  # dwell-time gradient
    pattern = BitmapPattern(width=10e-6, height=10e-6, depth=1e-6, array=array)
    pattern.point = Point(0, 0)
    return FibsemMillingStage(name=name, pattern=pattern)


@pytest.fixture
def axes():
    """Axes showing an image, with autoscale still on — as a fresh canvas is.

    Autoscale matters: an ``imshow``-drawn overlay only collapses the view while it
    is on, so an axes with fixed limits would hide the regression.
    """
    fig, ax = plt.subplots()
    ax.imshow(
        np.zeros((RESOLUTION, RESOLUTION)),
        extent=(-0.5, RESOLUTION - 0.5, RESOLUTION - 0.5, -0.5),
    )
    yield ax
    plt.close(fig)


def _overlay(ax) -> MillingPatternOverlay:
    overlay = MillingPatternOverlay()
    overlay.attach(ax, _FakeCanvas())
    return overlay


def test_a_bitmap_stage_draws_its_outline_and_its_bitmap(axes):
    overlay = _overlay(axes)
    overlay.set_stages([_bitmap_stage()], _image())

    kinds = [type(a).__name__ for a in overlay._artists]
    assert kinds.count("AxesImage") == 1, f"bitmap not drawn: {kinds}"
    assert kinds.count("Rectangle") == 1, f"outline not drawn: {kinds}"


def test_the_bitmap_image_sits_inside_the_outline(axes):
    overlay = _overlay(axes)
    overlay.set_stages([_bitmap_stage()], _image())

    outline = next(a for a in overlay._artists if isinstance(a, Rectangle))
    image = next(a for a in overlay._artists if isinstance(a, AxesImage))

    # The pattern is 10 um wide, centred on the image centre.
    width_px = 10e-6 / PIXEL_SIZE
    assert outline.get_width() == pytest.approx(width_px)
    assert outline.get_x() == pytest.approx(RESOLUTION / 2 - width_px / 2, abs=1)

    # The image is drawn on the unit square, through the outline's own transform,
    # so it lands on the outline whatever the pattern's size and rotation. Its
    # transform ends in transData, so undo that to compare in image pixels.
    corners = axes.transData.inverted().transform(
        image.get_transform().transform([(0, 0), (1, 1)])
    )
    assert abs(corners[1][0] - corners[0][0]) == pytest.approx(width_px)
    assert abs(corners[1][1] - corners[0][1]) == pytest.approx(width_px)
    assert min(corners[0][0], corners[1][0]) == pytest.approx(outline.get_x())


def test_bitmap_row_zero_is_drawn_at_the_top_of_the_pattern(axes):
    """AutoScript's row 0 is the top-left cell, so that is where it has to appear.

    The image rides the outline rectangle's transform, whose v=0 is the patch's xy
    corner -- the TOP edge, because the image axes run y-down. imshow's default
    origin="upper" puts row 0 at the extent's `top` (v=1) and drew every bitmap
    upside down.
    """
    stage = _bitmap_stage()
    # light the array's top half, leave the bottom half blank
    stage.pattern.array[:, :, 0] = 0
    stage.pattern.array[:16, :, 0] = 1

    overlay = _overlay(axes)
    overlay.set_stages([stage], _image())

    # Render, and look at where the lit half actually came out. The pattern is
    # axis-aligned, so its halves are two crops of the rendered canvas.
    outline = next(a for a in overlay._artists if isinstance(a, Rectangle))
    x0, y0 = outline.get_x(), outline.get_y()
    w, h = outline.get_width(), outline.get_height()
    fig = axes.figure
    fig.canvas.draw()
    px = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float)

    def _brightness(y_from: float, y_to: float) -> float:
        (xa, ya), (xb, yb) = axes.transData.transform(
            [(x0 + w * 0.2, y_from), (x0 + w * 0.8, y_to)]
        )
        # display y is bottom-up, the buffer's rows are top-down
        rows = slice(int(px.shape[0] - max(ya, yb)), int(px.shape[0] - min(ya, yb)))
        cols = slice(int(min(xa, xb)), int(max(xa, xb)))
        return px[rows, cols].mean()

    # inset from the edges, so the outline's own lines are not what is measured
    top = _brightness(y0 + h * 0.1, y0 + h * 0.4)
    bottom = _brightness(y0 + h * 0.6, y0 + h * 0.9)

    assert top > bottom, (
        f"the bitmap is upside down (top={top:.1f} bottom={bottom:.1f})"
    )


def test_drawing_a_bitmap_leaves_the_view_alone(axes):
    before = axes.get_xlim(), axes.get_ylim()

    overlay = _overlay(axes)
    overlay.set_stages([_bitmap_stage()], _image())

    assert (axes.get_xlim(), axes.get_ylim()) == before


def test_clearing_removes_the_bitmap_artists(axes):
    overlay = _overlay(axes)
    overlay.set_stages([_bitmap_stage()], _image())
    overlay.clear()

    assert overlay._artists == []
    assert not axes.images[1:]  # only the background image is left


def test_the_other_shapes_still_draw(axes):
    """The shape → artist branch now returns a list; rectangles must be unaffected."""
    pattern = RectanglePattern(width=5e-6, height=5e-6, depth=1e-6)
    pattern.point = Point(0, 0)
    overlay = _overlay(axes)
    overlay.set_stages([FibsemMillingStage(name="rect", pattern=pattern)], _image())

    kinds = [type(a).__name__ for a in overlay._artists]
    assert kinds.count("Polygon") == 1, kinds
    assert kinds.count("Line2D") == 2  # the crosshair


# ── circles: annulus / wedge ─────────────────────────────────────────────────

SHAPE = (RESOLUTION, RESOLUTION)


def _circle(**kwargs) -> FibsemCircleSettings:
    params = dict(radius=5e-6, depth=1e-6, centre_x=0, centre_y=0)
    params.update(kwargs)
    return FibsemCircleSettings(**params)


def _artist(ps, axes):
    """The single artist the overlay builds for one shape."""
    (artist,) = _overlay(axes)._shape_to_artists(
        ps, SHAPE, PIXEL_SIZE, "yellow", 1.0, 6
    )
    return artist


def test_a_circle_with_thickness_draws_a_ring_not_a_disc(axes):
    """An annulus used to render as a filled disc: thickness was ignored."""
    patch = _artist(_circle(thickness=1e-6), axes)

    assert isinstance(patch, Annulus)
    # thickness measures inward from the radius: the ring is 4 um .. 5 um.
    assert patch.get_radii() == pytest.approx((5e-6 / PIXEL_SIZE, 5e-6 / PIXEL_SIZE))
    assert patch.get_width() == pytest.approx(1e-6 / PIXEL_SIZE)


def test_a_thickness_wider_than_the_radius_is_clamped(axes):
    """``Annulus`` rejects a width past the centre; the pattern must still draw."""
    patch = _artist(_circle(radius=5e-6, thickness=8e-6), axes)

    assert patch.get_width() == pytest.approx(5e-6 / PIXEL_SIZE)


def test_partial_angles_draw_a_wedge(axes):
    patch = _artist(_circle(start_angle=45, end_angle=135), axes)

    assert isinstance(patch, Wedge)
    assert (patch.theta1, patch.theta2) == (45, 135)


def test_a_plain_circle_is_still_a_circle(axes):
    assert isinstance(_artist(_circle(), axes), Circle)


def test_an_annulus_stage_draws_through_the_stage_path(axes):
    """End to end, as the canvas drives it: CirclePattern is what carries thickness."""
    pattern = CirclePattern(radius=5e-6, depth=1e-6, thickness=1e-6)
    pattern.point = Point(0, 0)
    overlay = _overlay(axes)
    overlay.set_stages([FibsemMillingStage(name="annulus", pattern=pattern)], _image())

    kinds = [type(a).__name__ for a in overlay._artists]
    assert kinds.count("Annulus") == 1, kinds


# ── exclusions ───────────────────────────────────────────────────────────────


def test_an_exclusion_shape_is_black_whatever_the_stage_colour(axes):
    """Exclusions are the region the mill must avoid — black, as in napari + the plot."""
    vertices = np.array([[-5e-6, -5e-6], [5e-6, -5e-6], [5e-6, 5e-6], [-5e-6, 5e-6]])
    pattern = PolygonPattern(vertices=vertices, depth=1e-6, is_exclusion=True)
    pattern.point = Point(0, 0)
    overlay = _overlay(axes)
    overlay.set_stages([FibsemMillingStage(name="keep out", pattern=pattern)], _image())

    patch = next(a for a in overlay._artists if isinstance(a, Polygon))
    assert patch.get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert patch.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))


def test_a_normal_shape_keeps_its_stage_colour(axes):
    """The exclusion branch must not blacken everything else."""
    vertices = np.array([[-5e-6, -5e-6], [5e-6, -5e-6], [5e-6, 5e-6], [-5e-6, 5e-6]])
    pattern = PolygonPattern(vertices=vertices, depth=1e-6, is_exclusion=False)
    pattern.point = Point(0, 0)
    overlay = _overlay(axes)
    overlay.set_stages([FibsemMillingStage(name="poly", pattern=pattern)], _image())

    patch = next(a for a in overlay._artists if isinstance(a, Polygon))
    assert patch.get_edgecolor()[:3] != pytest.approx((0.0, 0.0, 0.0))


# ── scan direction arrows ───────────────────────────────────────────────


def _rect_stage(name: str, scan_direction: str, x_um: float = 0.0):
    pattern = RectanglePattern(
        width=6e-6, height=10e-6, depth=1e-6, scan_direction=scan_direction
    )
    pattern.point = Point(x_um * 1e-6, 0)
    return FibsemMillingStage(name=name, pattern=pattern)


def _arrows(overlay):
    return [a for a in overlay._artists if isinstance(a, Annotation)]


def _arrow_vector(arrow: Annotation):
    (x_s, y_s), (x_e, y_e) = arrow.xyann, arrow.xy
    return x_e - x_s, y_e - y_s


def test_scan_direction_arrows_are_off_by_default(axes):
    overlay = _overlay(axes)
    overlay.set_stages([_rect_stage("a", "TopToBottom")], _image())
    assert _arrows(overlay) == []


def test_each_rectangle_gets_an_arrow_along_its_scan_direction(axes):
    """Top-to-bottom points down the image (y grows downwards); bottom-to-top up."""
    overlay = MillingPatternOverlay(show_scan_direction=True)
    overlay.attach(axes, _FakeCanvas())
    overlay.set_stages(
        [_rect_stage("down", "TopToBottom"), _rect_stage("up", "BottomToTop", 20)],
        _image(),
    )

    arrows = _arrows(overlay)
    assert len(arrows) == 2
    dx, dy = _arrow_vector(arrows[0])
    assert dx == 0 and dy > 0
    dx, dy = _arrow_vector(arrows[1])
    assert dx == 0 and dy < 0
    # the arrow runs most of the rectangle's height, inset from the edges
    assert abs(dy) == pytest.approx(10e-6 / PIXEL_SIZE * 0.7)
    # in the stage's colour: the second stage is not the first's
    assert (
        arrows[0].arrow_patch.get_edgecolor() != arrows[1].arrow_patch.get_edgecolor()
    )


def test_coincident_stages_get_their_arrows_side_by_side(axes):
    """A top-to-bottom pass then a bottom-to-top one at the same position: the
    two opposite arrows sit left and right of centre, not on top of each other."""
    overlay = MillingPatternOverlay(show_scan_direction=True)
    overlay.attach(axes, _FakeCanvas())
    overlay.set_stages(
        [_rect_stage("down", "TopToBottom"), _rect_stage("up", "BottomToTop")],
        _image(),
    )

    first, second = _arrows(overlay)
    assert first.xy[0] < RESOLUTION / 2 < second.xy[0]
    # both stay inside the rectangle
    half_width = 6e-6 / PIXEL_SIZE / 2
    assert abs(first.xy[0] - RESOLUTION / 2) < half_width
    assert abs(second.xy[0] - RESOLUTION / 2) < half_width


def test_a_trench_draws_an_arrow_per_half_toward_the_centre(axes):
    """A trench has no editable scan direction, but each half mills toward the
    lamella: the upper half top-to-bottom, the lower half bottom-to-top."""
    overlay = MillingPatternOverlay(show_scan_direction=True)
    overlay.attach(axes, _FakeCanvas())
    pattern = TrenchPattern(
        width=10e-6,
        upper_trench_height=3e-6,
        lower_trench_height=3e-6,
        spacing=2e-6,
        depth=1e-6,
    )
    pattern.point = Point(0, 0)
    overlay.set_stages([FibsemMillingStage(name="trench", pattern=pattern)], _image())

    arrows = sorted(_arrows(overlay), key=lambda a: a.xyann[1])
    assert len(arrows) == 2
    upper, lower = arrows
    assert _arrow_vector(upper)[1] > 0  # down, toward the centre
    assert _arrow_vector(lower)[1] < 0  # up, toward the centre


def test_clearing_removes_the_arrows(axes):
    overlay = MillingPatternOverlay(show_scan_direction=True)
    overlay.attach(axes, _FakeCanvas())
    overlay.set_stages([_rect_stage("a", "TopToBottom")], _image())
    assert _arrows(overlay)
    overlay.clear()
    assert _arrows(overlay) == []
    assert not any(isinstance(a, Annotation) for a in axes.texts)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
