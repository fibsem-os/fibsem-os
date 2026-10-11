"""Tescan spot burn coordinates: image points to DrawBeam metres.

TESCAN cannot do the blank -> park -> unblank sequence the other backends use: FIB.Scan is a
strict subset of SEM.Scan, missing exactly SetBlanker, GetBlanker and SetBeamPosition. The
spot burn service draws one DrawBeam dot per point instead, through the milling service
(tests/test_tescan_spot_burn_service.py). This file checks where those dots go.
"""

import pytest

from fibsem.drivers.tescan.microscope import TescanMicroscope
from fibsem.structures import Point

HFW = 100e-6
RESOLUTION = (1536, 1024)  # (width, height)


@pytest.mark.parametrize(
    "normalised, expected",
    [
        (Point(0.5, 0.5), Point(0.0, 0.0)),  # centre
        (Point(0.0, 0.5), Point(-HFW / 2, 0.0)),  # left edge
        (Point(1.0, 0.5), Point(HFW / 2, 0.0)),  # right edge
    ],
)
def test_point_to_metres_maps_normalised_to_centre_origin(normalised, expected):
    """(0-1, top-left origin) -> metres from the image centre."""
    got = TescanMicroscope._spot_burn_point_to_metres(
        normalised, hfw=HFW, resolution=RESOLUTION
    )
    assert got.x == pytest.approx(expected.x)
    assert got.y == pytest.approx(expected.y)


def test_point_to_metres_y_axis_points_up():
    """Image y grows downwards, DrawBeam y grows upwards, so the sign must flip."""
    top = TescanMicroscope._spot_burn_point_to_metres(
        Point(0.5, 0.0), hfw=HFW, resolution=RESOLUTION
    )
    bottom = TescanMicroscope._spot_burn_point_to_metres(
        Point(0.5, 1.0), hfw=HFW, resolution=RESOLUTION
    )
    assert top.y > 0
    assert bottom.y < 0
    assert top.y == pytest.approx(-bottom.y)


def test_point_to_metres_uses_pixel_aspect_not_hfw_for_y():
    """y extent is hfw scaled by the pixel aspect ratio, not hfw itself."""
    width, height = RESOLUTION
    bottom = TescanMicroscope._spot_burn_point_to_metres(
        Point(0.5, 1.0), hfw=HFW, resolution=RESOLUTION
    )
    assert bottom.y == pytest.approx(-(HFW / width) * (height / 2))
