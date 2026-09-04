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

No Qt needed — the overlay talks to a matplotlib axes.

Run directly:
    python tests/ui/test_milling_overlay_bitmap.py
"""

import matplotlib
import numpy as np
import pytest

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

from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.patterning.patterns2 import (  # noqa: E402
    BitmapPattern,
    CirclePattern,
    PolygonPattern,
    RectanglePattern,
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
