"""Comparing and wrapping angles, in radians.

The one home for angle wrapping: stage and manipulator positions compare
their r and t through ``angle_difference``, and ``fibsem.movement`` re-exports these
so its callers keep working.
"""

import numpy as np


def wrap_angle(angle: float) -> float:
    """*angle* in (-pi, pi], so 359.9 degrees is -0.1 degrees."""
    return float(np.pi - (np.pi - angle) % (2 * np.pi))


def wrap_degrees(angle: float) -> float:
    """*angle* in degrees, in (-180, 180]; the degree twin of ``wrap_angle``."""
    return float(180.0 - (180.0 - angle) % 360.0)


def angle_difference(angle1: float, angle2: float) -> float:
    """The unsigned difference between two angles, in [0, pi].

    Whole turns do not count, so 2 and 358 degrees are 4 degrees apart.
    """
    angle1 %= 2 * np.pi
    angle2 %= 2 * np.pi

    large_angle = np.max([angle1, angle2])
    small_angle = np.min([angle1, angle2])

    return min((large_angle - small_angle), (2 * np.pi + small_angle - large_angle))


def rotation_angle_is_larger(angle1: float, angle2: float, atol: float = 90) -> bool:
    """Whether two angles (radians) are more than *atol* DEGREES apart."""
    return angle_difference(angle1, angle2) > (np.deg2rad(atol))


def rotation_angle_is_smaller(angle1: float, angle2: float, atol: float = 5) -> bool:
    """Whether two angles (radians) are less than *atol* DEGREES apart."""
    return angle_difference(angle1, angle2) < (np.deg2rad(atol))
