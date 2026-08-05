"""Tests for fibsem.conversions.

`is_inside_image_bounds` moved here from `fibsem.ui.napari.utilities`. It is pure
integer geometry with no napari content, and living in a module that imports napari,
`FibsemMicroscope` and `TescanMicroscope` at the top forced `imaging/tiled.py` to
import it lazily inside four separate function bodies to avoid the cycle.

These pin the behaviour as it was, including the off-by-one below, so the move is
provably inert.
"""

import pytest

from fibsem.conversions import is_inside_image_bounds

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
