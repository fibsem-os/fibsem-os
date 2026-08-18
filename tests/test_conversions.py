"""Tests for fibsem.conversions.

`is_inside_image_bounds` moved here from `fibsem.ui.napari.utilities`. It is pure
integer geometry with no napari content, and living in a module that imports napari,
`FibsemMicroscope` and `TescanMicroscope` at the top forced `imaging/tiled.py` to
import it lazily inside four separate function bodies to avoid the cycle.

These pin the behaviour as it was, including the off-by-one below, so the move is
provably inert.

The image → microscope conversions had no coverage at all until FIB-362 factored
their shared centring into `image_to_microscope_image_coordinates_px`. They are
read by movement, milling, detection, the minimap and tiled acquisition, so the
tests below pin the behaviour the refactor had to preserve rather than the
refactor itself.
"""

import numpy as np
import pytest

from fibsem.conversions import (
    get_image_pixel_centre,
    image_to_microscope_image_coordinates,
    image_to_microscope_image_coordinates2,
    image_to_microscope_image_coordinates_px,
    is_inside_image_bounds,
    microscope_image_to_image_coordinates,
)
from fibsem.structures import Point

SHAPE = (100, 200)  # (y, x)


@pytest.mark.parametrize(
    "coords",
    [
        (50, 100),   # middle
        (1, 1),      # just inside the low corner
        (99, 199),   # just inside the high corner
        (50.5, 100.5),  # sub-pixel
    ],
)
def test_inside(coords):
    assert is_inside_image_bounds(coords, SHAPE) is True


@pytest.mark.parametrize(
    "coords",
    [
        (-1, 50),     # above
        (100, 50),    # below: shape[0] is exclusive
        (50, -1),     # left
        (50, 200),    # right: shape[1] is exclusive
        (1000, 1000), # far outside
    ],
)
def test_outside(coords):
    assert is_inside_image_bounds(coords, SHAPE) is False


@pytest.mark.parametrize("coords", [(0, 50), (50, 0), (0, 0)])
def test_zero_reports_as_outside(coords):
    """The lower bound is exclusive, so row/column 0 is 'outside' despite being real.

    Intentional, and confirmed as wanted -- not an off-by-one to be tidied up. It
    reads like one, since the upper bound is exclusive as it should be and the lower
    one is too, which is exactly why it is pinned here. Several call sites in
    `imaging/tiled.py` and the minimap widgets use this to decide whether to draw a
    marker; widening the accepted region would change what they render.
    """
    assert is_inside_image_bounds(coords, SHAPE) is False


def test_shape_is_y_x_not_x_y():
    """coords and shape are both (y, x); a transposed shape changes the answer."""
    assert is_inside_image_bounds((50, 150), (100, 200)) is True
    assert is_inside_image_bounds((50, 150), (200, 100)) is False


# ---------------------------------------------------------------------------
# image px → microscope image coordinates
# ---------------------------------------------------------------------------

EVEN = (512, 1024)  # (height, width)
ODD = (511, 1023)   # the only case where the centre rounding is visible
PIXEL_SIZE = 1e-8


def test_the_centre_is_the_origin():
    assert image_to_microscope_image_coordinates_px(
        Point(512.0, 256.0), EVEN, subpixel_precision=True
    ) == Point(0.0, 0.0)


def test_y_is_mirrored_and_x_is_not():
    """The asymmetry this function exists to state once: microscope +y is *up*,
    image +y is down, and x agrees between the two."""
    down_right = image_to_microscope_image_coordinates_px(
        Point(612.0, 356.0), EVEN, subpixel_precision=True
    )
    assert (down_right.x, down_right.y) == (100.0, -100.0)

    up_left = image_to_microscope_image_coordinates_px(
        Point(412.0, 156.0), EVEN, subpixel_precision=True
    )
    assert (up_left.x, up_left.y) == (-100.0, 100.0)


def test_shape_is_height_width():
    """A transposed shape looks fine on a square image, so check a rectangle."""
    corner = image_to_microscope_image_coordinates_px(
        Point(0.0, 0.0), EVEN, subpixel_precision=True
    )
    assert (corner.x, corner.y) == (-512.0, 256.0)


def test_subpixel_precision_only_matters_on_an_odd_dimension():
    """Flooring the centre is the long-standing default. It is exact on an even
    dimension and half a pixel off on an odd one — which is why the correlation
    module opts out and everything else does not."""
    for shape, expected_floored in ((EVEN, (188.0, 156.0)), (ODD, (189.0, 155.0))):
        floored = image_to_microscope_image_coordinates_px(
            Point(700.0, 100.0), shape, subpixel_precision=False
        )
        assert (floored.x, floored.y) == expected_floored

    exact = image_to_microscope_image_coordinates_px(
        Point(700.0, 100.0), ODD, subpixel_precision=True
    )
    assert (exact.x, exact.y) == (188.5, 155.5)


@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("shape", [EVEN, ODD])
def test_the_array_and_shape_forms_agree(shape, subpixel):
    """Two functions, one taking the image and one its shape. Nothing forced
    them to stay in step before they shared a centring."""
    coord = Point(700.0, 100.0)
    from_image = image_to_microscope_image_coordinates(
        coord, np.zeros(shape, dtype=np.uint8), PIXEL_SIZE, subpixel_precision=subpixel
    )
    from_shape = image_to_microscope_image_coordinates2(
        coord, shape, PIXEL_SIZE, subpixel_precision=subpixel
    )
    assert from_image.x == pytest.approx(from_shape.x)
    assert from_image.y == pytest.approx(from_shape.y)


@pytest.mark.parametrize("subpixel", [False, True])
def test_both_forms_pass_the_flag_through(subpixel):
    """The wrappers must not decide the rounding for their callers — this is
    what a hardcoded `True` or `False` inside either of them would break."""
    coord = Point(700.0, 100.0)
    expected = image_to_microscope_image_coordinates_px(
        coord, ODD, subpixel_precision=subpixel
    )
    for actual in (
        image_to_microscope_image_coordinates(
            coord, np.zeros(ODD, dtype=np.uint8), PIXEL_SIZE, subpixel_precision=subpixel
        ),
        image_to_microscope_image_coordinates2(
            coord, ODD, PIXEL_SIZE, subpixel_precision=subpixel
        ),
    ):
        assert actual.x == pytest.approx(expected.x * PIXEL_SIZE)
        assert actual.y == pytest.approx(expected.y * PIXEL_SIZE)


def test_the_default_is_the_floored_centre():
    """Called without the flag, both keep doing what movement and milling have
    always done. Changing this default would move the stage."""
    coord = Point(700.0, 100.0)
    floored = image_to_microscope_image_coordinates_px(
        coord, ODD, subpixel_precision=False
    )
    assert image_to_microscope_image_coordinates2(
        coord, ODD, PIXEL_SIZE
    ).x == pytest.approx(floored.x * PIXEL_SIZE)
    assert image_to_microscope_image_coordinates(
        coord, np.zeros(ODD, dtype=np.uint8), PIXEL_SIZE
    ).y == pytest.approx(floored.y * PIXEL_SIZE)


def test_a_non_2d_shape_is_rejected():
    with pytest.raises(ValueError, match="two integers"):
        image_to_microscope_image_coordinates2(Point(0.0, 0.0), (512, 512, 3), PIXEL_SIZE)


# ---------------------------------------------------------------------------
# the shared centre (FIB-644)
# ---------------------------------------------------------------------------


def test_the_centre_is_y_then_x():
    """Order matters and is easy to get backwards: (cy, cx), matching image
    shape rather than the (x, y) of Point. Checked on a rectangle, since a
    square cannot tell the two apart."""
    assert get_image_pixel_centre(EVEN) == (256, 512)


def test_the_centre_floors_by_default_and_can_be_exact():
    assert get_image_pixel_centre(ODD) == (255, 511)
    assert get_image_pixel_centre(ODD, subpixel_precision=True) == (255.5, 511.5)


def test_the_floored_centre_stays_a_plain_int():
    """It is handed to callers that index, format and serialise it, and the
    numpy scalar an ndarray would give back is not JSON-serialisable."""
    import json

    cy, cx = get_image_pixel_centre(ODD)
    assert (type(cy), type(cx)) == (int, int)
    assert json.dumps([cy, cx]) == "[255, 511]"

    cy, cx = get_image_pixel_centre(ODD, subpixel_precision=True)
    assert (type(cy), type(cx)) == (float, float)


def test_a_non_2d_shape_has_no_centre():
    """An RGB image reaching a 2-D conversion is a bug, not something to
    silently take the first two axes of."""
    with pytest.raises(ValueError):
        get_image_pixel_centre((8, 8, 3))


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [EVEN, ODD])
@pytest.mark.parametrize("subpixel", [False, True])
def test_the_inverse_round_trips(shape, subpixel):
    """The defect FIB-644 tier 1 fixes.

    The inverse used to floor the centre unconditionally, so pairing it with a
    forward conversion at subpixel precision lost half a pixel on an odd image
    dimension — and correlation calls the forward one that way. Now both ends
    can express the same centre, so the trip is exact either way.
    """
    original = Point(700.0, 100.0)
    metres = image_to_microscope_image_coordinates2(
        original, shape, PIXEL_SIZE, subpixel_precision=subpixel
    )
    back = microscope_image_to_image_coordinates(
        metres, shape, PIXEL_SIZE, subpixel_precision=subpixel
    )

    assert back.x == pytest.approx(original.x)
    assert back.y == pytest.approx(original.y)


def test_mismatched_centres_are_what_used_to_lose_half_a_pixel():
    """Pinning the failure itself, so the fix cannot be quietly undone: forward
    exact, inverse floored, on an odd shape."""
    original = Point(700.0, 100.0)
    metres = image_to_microscope_image_coordinates2(
        original, ODD, PIXEL_SIZE, subpixel_precision=True
    )
    back = microscope_image_to_image_coordinates(
        metres, ODD, PIXEL_SIZE, subpixel_precision=False
    )

    assert abs(back.x - original.x) == pytest.approx(0.5)
    assert abs(back.y - original.y) == pytest.approx(0.5)


def test_the_inverse_default_is_unchanged():
    """Called as before — no keyword — it must behave exactly as it always has."""
    p = Point(1.885e-06, 1.555e-06)
    explicit = microscope_image_to_image_coordinates(
        p, ODD, PIXEL_SIZE, subpixel_precision=False
    )
    implicit = microscope_image_to_image_coordinates(p, ODD, PIXEL_SIZE)

    assert (implicit.x, implicit.y) == (explicit.x, explicit.y)


def test_is_importable_without_napari():
    """The point of the move: this must not drag a UI stack in behind it."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from fibsem.conversions import is_inside_image_bounds; "
         "assert 'napari' not in sys.modules, 'napari was imported'; "
         "assert 'fibsem.microscope' not in sys.modules, 'fibsem.microscope was imported'"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
