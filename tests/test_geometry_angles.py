import numpy as np
import pytest

from fibsem import movement
from fibsem.geometry import angles


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (np.pi, np.pi),
        (-np.pi, np.pi),  # the boundary goes to +pi
        (3 * np.pi, np.pi),
        (np.deg2rad(359.9), np.deg2rad(-0.1)),
        (np.deg2rad(-190), np.deg2rad(170)),
        (4 * np.pi + 0.25, 0.25),
    ],
)
def test_wrap_angle(angle, expected):
    assert angles.wrap_angle(angle) == pytest.approx(expected)
    assert type(angles.wrap_angle(angle)) is float


def test_angle_difference_is_unsigned_and_ignores_whole_turns():
    assert angles.angle_difference(np.deg2rad(2), np.deg2rad(358)) == pytest.approx(
        np.deg2rad(4)
    )
    assert angles.angle_difference(np.deg2rad(358), np.deg2rad(2)) == pytest.approx(
        np.deg2rad(4)
    )
    assert angles.angle_difference(-4 * np.pi, 4 * np.pi) == pytest.approx(0)


def test_rotation_checks_take_tolerance_in_degrees():
    assert angles.rotation_angle_is_smaller(0, np.deg2rad(4), atol=5)
    assert not angles.rotation_angle_is_smaller(0, np.deg2rad(6), atol=5)
    assert angles.rotation_angle_is_larger(0, np.deg2rad(180), atol=90)
    assert not angles.rotation_angle_is_larger(0, np.deg2rad(80), atol=90)


def test_movement_re_exports_the_same_functions():
    assert movement.angle_difference is angles.angle_difference
    assert movement.rotation_angle_is_smaller is angles.rotation_angle_is_smaller
    assert movement.rotation_angle_is_larger is angles.rotation_angle_is_larger
    assert movement.wrap_angle is angles.wrap_angle
