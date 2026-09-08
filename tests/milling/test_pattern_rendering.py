"""Pattern shapes → matplotlib, in the report-plot path.

``bitmap_to_rgba`` is the one place a bitmap pattern becomes pixels, shared by the
report plot (``_add_bitmap_mpl``) and the canvas overlay. It had no tests at all,
which is part of how bitmaps could go missing from the canvas unnoticed.

What is pinned here is the contract the two renderers rely on: dwell time drives a
transparent → colour ramp, blanked cells are opaque black, an array larger than the
pattern is downsized (subpixel cells do not render), and an annulus is the ring
between ``radius - thickness`` and ``radius``.
"""

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_rgba

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Annulus, Circle, Wedge  # noqa: E402

from fibsem.milling.patterning.plotting import (  # noqa: E402
    _add_bitmap_mpl,
    _add_circle_mpl,
    bitmap_to_rgba,
)
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)

COLOUR = "yellow"
OPACITY = 0.5


def _bitmap(dwell: np.ndarray, blanked: np.ndarray = None, **kwargs):
    """A bitmap pattern carrying *dwell* (and optionally *blanked*) directly.

    A float array bypasses the uint8 image → points conversion in
    ``FibsemBitmapSettings.__post_init__``, so the values arrive unchanged.
    """
    array = np.zeros((*dwell.shape, 2), dtype=float)
    array[:, :, 0] = dwell
    if blanked is not None:
        array[:, :, 1] = blanked
    return FibsemBitmapSettings(
        width=10e-6,
        height=10e-6,
        depth=1e-6,
        centre_x=0,
        centre_y=0,
        array=array,
        **kwargs,
    )


def test_dwell_time_drives_a_transparent_to_colour_ramp():
    dwell = np.array([[0.0, 1.0]])
    rgba = bitmap_to_rgba(_bitmap(dwell), 100, 100, COLOUR, OPACITY)

    assert rgba[0, 0, 3] == pytest.approx(0.0)  # no dwell -> nothing drawn
    assert rgba[0, 1, 3] == pytest.approx(OPACITY)  # full dwell -> full opacity
    assert rgba[0, 1, :3] == pytest.approx(to_rgba(COLOUR)[:3], abs=0.01)


def test_blanked_cells_are_black():
    dwell = np.ones((1, 2))
    blanked = np.array([[0, 1]])
    rgba = bitmap_to_rgba(_bitmap(dwell, blanked), 100, 100, COLOUR, OPACITY)

    assert rgba[0, 1, :3] == pytest.approx((0.0, 0.0, 0.0))
    assert rgba[0, 1, 3] == pytest.approx(OPACITY)  # opaque black, not transparent
    assert rgba[0, 0, :3] != pytest.approx((0.0, 0.0, 0.0))  # the unblanked neighbour


def test_a_bitmap_larger_than_the_pattern_is_downsized():
    # 64x64 cells into a 10x20 px pattern: every cell would be subpixel, and
    # subpixel cells are not displayed at all.
    rgba = bitmap_to_rgba(_bitmap(np.ones((64, 64))), 20, 10, COLOUR, OPACITY)

    assert rgba.shape == (10, 20, 4)


def test_a_bitmap_smaller_than_the_pattern_is_left_alone():
    rgba = bitmap_to_rgba(_bitmap(np.ones((4, 4))), 200, 200, COLOUR, OPACITY)

    assert rgba.shape == (4, 4, 4)


def test_flip_y_flips_the_rows():
    dwell = np.array([[0.0], [1.0]])
    plain = bitmap_to_rgba(_bitmap(dwell), 100, 100, COLOUR, OPACITY)
    flipped = bitmap_to_rgba(_bitmap(dwell, flip_y=True), 100, 100, COLOUR, OPACITY)

    assert plain[0, 0, 3] == pytest.approx(flipped[1, 0, 3])
    assert plain[1, 0, 3] == pytest.approx(flipped[0, 0, 3])


def test_a_pattern_with_no_bitmap_draws_nothing():
    """A path-less, array-less bitmap pattern is legal — it must not raise."""
    shape = FibsemBitmapSettings(
        width=10e-6, height=10e-6, depth=1e-6, centre_x=0, centre_y=0
    )
    rgba = bitmap_to_rgba(shape, 100, 100, COLOUR, OPACITY)

    assert shape.bitmap is None
    assert rgba[:, :, 3] == pytest.approx(0.0)


# ── annulus / wedge geometry ─────────────────────────────────────────────────

RESOLUTION = 512
PIXEL_SIZE = 1e-8


def _image() -> FibsemImage:
    image = FibsemImage(data=np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8))
    image.metadata = FibsemImageMetadata(
        image_settings=ImageSettings(
            hfw=RESOLUTION * PIXEL_SIZE,
            resolution=[RESOLUTION, RESOLUTION],
            beam_type=BeamType.ION,
        ),
        pixel_size=Point(PIXEL_SIZE, PIXEL_SIZE),
        microscope_state=MicroscopeState(),
    )
    return image


def _circle(**kwargs) -> FibsemCircleSettings:
    params = dict(radius=1e-6, depth=1e-6, centre_x=0, centre_y=0)
    params.update(kwargs)
    return FibsemCircleSettings(**params)


def _patch(shape: FibsemCircleSettings):
    fig, ax = plt.subplots()
    try:
        _add_circle_mpl(shape, _image(), COLOUR, ax=ax)
        (patch,) = ax.patches
        return patch
    finally:
        plt.close(fig)


def test_a_thickness_draws_the_ring_between_radius_and_radius_minus_thickness():
    """Annulus takes the OUTER radius; passing the inner one drew the wrong ring.

    radius 100 px, thickness 20 px is the ring 80..100 — the same geometry the mask
    path (``draw_annulus_shape``) builds. Passing r=80 put it at 60..80 instead.
    """
    patch = _patch(_circle(radius=100 * PIXEL_SIZE, thickness=20 * PIXEL_SIZE))

    assert isinstance(patch, Annulus)
    assert patch.get_radii() == pytest.approx((100, 100))  # outer
    assert patch.get_width() == pytest.approx(20)  # inward from the outer edge


def test_no_thickness_is_a_plain_circle():
    patch = _patch(_circle(thickness=0))

    assert isinstance(patch, Circle)
    assert patch.get_radius() == pytest.approx(1e-6 / PIXEL_SIZE)


def test_partial_angles_draw_a_wedge():
    patch = _patch(_circle(start_angle=30, end_angle=200))

    assert isinstance(patch, Wedge)
    assert (patch.theta1, patch.theta2) == (30, 200)


def test_an_exclusion_circle_is_black():
    patch = _patch(_circle(is_exclusion=True))

    assert patch.get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))


def test_the_report_plot_draws_bitmaps_the_right_way_up():
    """Row 0 is the top-left cell (AutoScript), so it must land at the pattern's top.

    The image is drawn through the outline rectangle's transform, whose v=0 is the
    patch's xy corner -- the TOP edge on the y-inverted image axes. imshow's default
    origin="upper" puts row 0 at the extent's `top` (v=1), i.e. the bottom of the
    pattern, which is how every bitmap came out upside down.
    """
    array = np.zeros((8, 8, 2), dtype=float)
    array[:4, :, 0] = 1
    shape = FibsemBitmapSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0, array=array
    )

    fig, ax = plt.subplots()
    try:
        _add_bitmap_mpl(shape, _image(), COLOUR, ax=ax)
        drawn = ax.images[-1]
    finally:
        plt.close(fig)

    assert drawn.origin == "lower"
