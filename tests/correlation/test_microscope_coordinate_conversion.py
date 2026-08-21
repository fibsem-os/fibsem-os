"""The correlation module converts image px → microscope px exactly once.

FIB-362: the conversion was written out four times in the correlation
module alone — twice in `correlation_v2` (once in a helper nothing called), once
inside `_reproject_poi_via_transform`, and once in
`apply_refractive_index_correction`, where it produces `px_m`, the number
Continue commits to the experiment. `fibsem.conversions` already owned the
concept, so all four now go through
`image_to_microscope_image_coordinates_px` there.

The tests that matter are the cross-checks: each caller is compared against the
shared function *and against each other*, so a formula edited in one place and
not the others fails rather than silently disagreeing.

Correlation asks for `subpixel_precision=True` throughout, which is what its own
arithmetic did. The flag is not cosmetic — see
`test_the_correlation_module_asks_for_the_true_centre`.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fibsem.conversions import image_to_microscope_image_coordinates_px
from fibsem.correlation.correlation_v2 import (
    _reproject_poi_via_transform,
    convert_poi_to_microscope_coordinates,
)
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import Point

_SHAPE = (512, 1024)  # (height, width) — deliberately not square
_ODD_SHAPE = (511, 1023)  # the only case where the centre rounding shows
_PIXEL_SIZE_UM = 0.02
_PIXEL_SIZE_M = _PIXEL_SIZE_UM * 1e-6


def _centred(x: float, y: float, shape=None):
    """The shared conversion, as the correlation module asks for it."""
    pt = image_to_microscope_image_coordinates_px(
        Point(x, y), shape or _SHAPE, subpixel_precision=True
    )
    return pt.x, pt.y


# Identity transform: FM-space coordinates land unchanged as FIB image pixels,
# so the same point can be pushed through the reprojection and the correlation
# output and the two compared directly.
_IDENTITY = {
    "scale": 1.0,
    "rotation_quaternion": np.eye(3).tolist(),
    "translation_around_rotation_center_zero": [0.0, 0.0, 0.0],
}


# ---------------------------------------------------------------------------
# The convention itself
# ---------------------------------------------------------------------------


def test_the_image_centre_is_the_origin():
    assert _centred(512.0, 256.0) == (0.0, 0.0)


def test_y_is_mirrored_and_x_is_not():
    """The asymmetry is the whole reason this is worth writing once: +y is *up*
    in microscope coordinates and down in image pixels, while x agrees."""
    right_and_down = _centred(612.0, 356.0)
    assert right_and_down == (100.0, -100.0)

    left_and_up = _centred(412.0, 156.0)
    assert left_and_up == (-100.0, 100.0)


def test_the_width_sets_x_and_the_height_sets_y():
    """A transposed shape would still look right on a square image. This is the
    check that catches shape[0]/shape[1] being swapped."""
    x, y = _centred(0.0, 0.0)
    assert (x, y) == (-512.0, 256.0)


def test_the_correlation_module_asks_for_the_true_centre():
    """Why every correlation call site passes `subpixel_precision=True`.

    `fibsem.conversions` defaults to flooring the centre, which the movement and
    milling paths have always done. On an even dimension the two agree exactly;
    on an odd one they differ by half a pixel — small enough to miss in review,
    and a real shift in a correlation result. This pins that correlation gets
    the un-floored centre, whatever the default becomes.
    """
    floored = image_to_microscope_image_coordinates_px(
        Point(700.0, 100.0), _ODD_SHAPE, subpixel_precision=False
    )
    assert (floored.x, floored.y) == (189.0, 155.0)
    assert _centred(700.0, 100.0, _ODD_SHAPE) == (188.5, 155.5)

    # ...and on an even dimension there is nothing to choose between them.
    even = image_to_microscope_image_coordinates_px(
        Point(700.0, 100.0), _SHAPE, subpixel_precision=False
    )
    assert (even.x, even.y) == _centred(700.0, 100.0)


# ---------------------------------------------------------------------------
# Every caller agrees — the drift check
#
# Parametrised over an odd shape as well as an even one, and that is not
# padding: on an even dimension `subpixel_precision` makes no difference at all,
# so an even-only test cannot tell whether a call site asked for the right one.
# ---------------------------------------------------------------------------

_SHAPES = [_SHAPE, _ODD_SHAPE]


def _correlation_output_px(x: float, y: float, shape):
    """`px` / `px_m` as the correlation output records them for one POI."""
    poi = convert_poi_to_microscope_coordinates(
        np.array([[x], [y], [0.0]]), shape, _PIXEL_SIZE_UM
    )[0]
    return poi["px"], poi["px_m"]


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_correlation_output_uses_the_shared_conversion(shape):
    px, _ = _correlation_output_px(700.0, 100.0, shape)
    assert tuple(px) == _centred(700.0, 100.0, shape)


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_ghost_reprojection_agrees_with_the_correlation_output(shape):
    """The ghost marker and the real POI are drawn in the same image from the
    same fitted transform. If their conversions disagree the shift between them
    — the only thing the ghost exists to show — is wrong by the difference.

    Note the two paths take the pixel size in different units, metres here and
    micrometres above, which is exactly the kind of detail that rots when the
    arithmetic is duplicated.
    """
    ghost = _reproject_poi_via_transform(
        _IDENTITY, np.array([[700.0, 100.0, 0.0]]), shape, _PIXEL_SIZE_M
    )[0]
    px, px_m = _correlation_output_px(700.0, 100.0, shape)

    assert (ghost.image_px.x, ghost.image_px.y) == (700.0, 100.0)
    assert (ghost.px.x, ghost.px.y) == tuple(px)
    assert (ghost.px.x, ghost.px.y) == _centred(700.0, 100.0, shape)
    assert ghost.px_m.x == pytest.approx(px_m[0])
    assert ghost.px_m.y == pytest.approx(px_m[1])


def _post_result(shape, px=None) -> CorrelationResult:
    """A result ready for a post correction: FIB surface at y=200, POI at y=300."""
    return CorrelationResult(
        poi=[
            CorrelationPointOfInterest(
                image_px=Point(x=700.0, y=300.0),
                px=px if px is not None else Point(),
                px_m=Point(),
            )
        ],
        input_data=CorrelationInputData(
            surface_coordinate=Coordinate(
                point=PointXYZ(y=200.0), point_type=PointType.SURFACE
            ),
            fib_image=SimpleNamespace(
                data=np.zeros(shape, dtype=np.uint8),
                metadata=SimpleNamespace(
                    pixel_size=SimpleNamespace(x=_PIXEL_SIZE_M),
                    image_settings=SimpleNamespace(filename="fake.tif"),
                ),
            ),
        ),
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_post_correction_uses_the_shared_conversion(shape):
    """`px_m` here is the number Continue commits, so this caller drifting is
    the one that reaches the microscope (FIB-321)."""
    result = _post_result(shape)

    result.apply_refractive_index_correction(1.5)

    corrected_y = 350.0  # 200 + (300 - 200) * 1.5
    _, expected_px_y = _centred(700.0, corrected_y, shape)
    assert math.isclose(result.poi[0].image_px.y, corrected_y)
    assert math.isclose(result.poi[0].px.y, expected_px_y)
    assert math.isclose(result.poi[0].px_m.y, expected_px_y * _PIXEL_SIZE_M)


def test_the_post_correction_leaves_x_alone():
    """Only y moves, so px.x must be left as it was rather than recomputed —
    rewriting it would quietly repair an inconsistency instead of showing it."""
    result = _post_result(_SHAPE, px=Point(x=-999.0, y=0.0))  # deliberately wrong

    result.apply_refractive_index_correction(1.5)

    assert result.poi[0].px.x == -999.0


# ---------------------------------------------------------------------------
# The rotation guard
# ---------------------------------------------------------------------------


def test_a_malformed_rotation_raises_rather_than_dropping_the_ghost():
    """A (3,) rotation broadcasts cleanly through the matmul and produces
    plausible garbage, so the shape check earns its keep even though the one
    call site cannot reach it. It used to log and return [], which lost the
    ghost marker and left the run looking fine."""
    bad = dict(_IDENTITY, rotation_quaternion=[1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="3x3"):
        _reproject_poi_via_transform(
            bad, np.array([[700.0, 100.0, 0.0]]), _SHAPE, _PIXEL_SIZE_M
        )


def test_a_quaternion_shaped_rotation_raises_too():
    """The field is named `rotation_quaternion` but stores a 3x3 matrix. An
    actual four-element quaternion arriving here is the misnomer coming true."""
    bad = dict(_IDENTITY, rotation_quaternion=[0.9, 0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="3x3"):
        _reproject_poi_via_transform(
            bad, np.array([[700.0, 100.0, 0.0]]), _SHAPE, _PIXEL_SIZE_M
        )


def test_no_image_means_no_microscope_coordinates():
    """Without a shape and pixel size only image_px can be filled in — and it
    still must be, or the ghost has nowhere to draw."""
    ghost = _reproject_poi_via_transform(
        _IDENTITY, np.array([[700.0, 100.0, 0.0]]), None, None
    )[0]

    assert (ghost.image_px.x, ghost.image_px.y) == (700.0, 100.0)
    assert (ghost.px.x, ghost.px.y) == (0.0, 0.0)
