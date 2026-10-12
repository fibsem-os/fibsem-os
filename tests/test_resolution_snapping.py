"""A Thermo acquire at an unlisted resolution scans the nearest listed one."""

import pytest

from fibsem.devices.beam import nearest_resolution

# what a JEOL-like scan lists: 4:3 frames only
FOUR_THREE = [(640, 480), (1280, 960), (2560, 1920)]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((1280, 960), (1280, 960)),
        ((1536, 1024), (1280, 960)),
        ((3072, 2048), (2560, 1920)),
        ((384, 256), (640, 480)),
    ],
)
def test_the_nearest_listed_resolution(requested, expected):
    assert nearest_resolution(requested, FOUR_THREE) == expected


def test_nothing_listed_keeps_the_resolution():
    assert nearest_resolution((1000, 700), None) == (1000, 700)
