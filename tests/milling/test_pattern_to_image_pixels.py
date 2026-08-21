"""Pattern metres → image pixels, through one conversion.

FIB-644: the four `_*_pattern_to_image_pixels` helpers each wrote out
`cx + x/ps`, `cy - y/ps` by hand — the inverse of the conversion
`fibsem.conversions` already owns. They now call it.

The reason this mattered is in `test_the_line_converter_disagrees_with_its_siblings`:
they had already drifted apart before anyone noticed.
"""

import numpy as np
import pytest

from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.milling.patterning.plotting import (
    _circle_pattern_to_image_pixels,
    _line_pattern_to_image_pixels,
    _polygon_pattern_to_image_pixels,
    _rect_pattern_to_image_pixels,
)
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    Point,
)

EVEN = (512, 1024)  # (height, width)
ODD = (511, 1023)  # the only shapes where the centre rounding is visible
PS = 1e-8
OFFSET = 5e-6  # 500 px at this pixel size


def _expected(x, y, shape, subpixel):
    pt = microscope_image_to_image_coordinates(
        Point(x=x, y=y), shape, PS, subpixel_precision=subpixel
    )
    return pt.x, pt.y


@pytest.mark.parametrize("shape", [EVEN, ODD])
def test_rectangle_uses_the_shared_conversion(shape):
    p = FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=OFFSET, centre_y=-OFFSET
    )
    px, py, w, h = _rect_pattern_to_image_pixels(p, PS, shape)

    assert (px, py) == _expected(OFFSET, -OFFSET, shape, subpixel=False)
    assert (w, h) == (1e-6 / PS, 2e-6 / PS)


@pytest.mark.parametrize("shape", [EVEN, ODD])
def test_circle_uses_the_shared_conversion(shape):
    p = FibsemCircleSettings(
        radius=3e-6,
        depth=1e-6,
        thickness=0,
        centre_x=OFFSET,
        centre_y=-OFFSET,
        start_angle=0,
        end_angle=360,
    )
    px, py, *_ = _circle_pattern_to_image_pixels(p, PS, shape)

    assert (px, py) == _expected(OFFSET, -OFFSET, shape, subpixel=False)


@pytest.mark.parametrize("shape", [EVEN, ODD])
def test_polygon_converts_every_vertex_the_same_way(shape):
    p = FibsemPolygonSettings(
        vertices=[(OFFSET, -OFFSET), (-OFFSET, OFFSET), (0.0, 0.0)], depth=1e-6
    )
    got = _polygon_pattern_to_image_pixels(p, PS, shape)

    want = np.array([_expected(x, y, shape, subpixel=False) for x, y in p.vertices])
    assert np.array_equal(got, want)


@pytest.mark.parametrize("shape", [EVEN, ODD])
def test_line_uses_the_shared_conversion(shape):
    p = FibsemLineSettings(
        start_x=OFFSET, start_y=-OFFSET, end_x=-OFFSET, end_y=OFFSET, depth=1e-6
    )
    sx, sy, ex, ey = _line_pattern_to_image_pixels(p, PS, shape)

    assert (sx, sy) == _expected(OFFSET, -OFFSET, shape, subpixel=True)
    assert (ex, ey) == _expected(-OFFSET, OFFSET, shape, subpixel=True)


def test_the_line_converter_disagrees_with_its_siblings():
    """The drift, pinned — deliberately, so it cannot be "tidied up" silently.

    Three of the four floor the image centre; the line converter does not. On an
    even image that is the same number, so it went unnoticed. On an odd one they
    are half a pixel apart, and this is milling pattern placement.

    Unifying them is a real decision about where patterns land, not a refactor.
    When someone makes it, this test should fail and be deleted on purpose
    (FIB-644).
    """
    rect = FibsemRectangleSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=OFFSET, centre_y=OFFSET
    )
    line = FibsemLineSettings(
        start_x=OFFSET, start_y=OFFSET, end_x=OFFSET, end_y=OFFSET, depth=1e-6
    )

    rx, ry, _, _ = _rect_pattern_to_image_pixels(rect, PS, ODD)
    lx, ly, _, _ = _line_pattern_to_image_pixels(line, PS, ODD)

    assert abs(lx - rx) == pytest.approx(0.5)
    assert abs(ly - ry) == pytest.approx(0.5)

    # ...and on an even image they agree, which is why it was never spotted.
    rx, ry, _, _ = _rect_pattern_to_image_pixels(rect, PS, EVEN)
    lx, ly, _, _ = _line_pattern_to_image_pixels(line, PS, EVEN)
    assert (lx, ly) == (rx, ry)
