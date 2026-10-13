"""Stage orientations, as pure functions: which pose a position is in, the milling
angle a tilt gives, and a position carried from one pose to another.

An orientation is a named stage pose, an ``r`` and a ``t``: SEM and FIB, FM on a
compustage, and MILLING, which is the SEM rotation at the tilt the milling angle needs.
The table of them comes from the stage device and the configured milling angle; these
functions take it as an argument, so the same answer serves a live position and a saved
one. Reading the table and the stage off an instrument is the caller's job, today
``FibsemMicroscope``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import numpy as np

from fibsem.movement import angle_difference, rotation_angle_is_smaller
from fibsem.structures import FibsemStagePosition
from fibsem.transformations import convert_stage_tilt_to_milling_angle

#: How far from a pose a position may sit and still be in it.
ROTATION_TOLERANCE = np.deg2rad(5)
TILT_TOLERANCE = 0.1  # radians
#: MILLING is any tilt above this at the SEM rotation that is not a declared pose.
MILLING_MIN_TILT = np.radians(-45)


def classify_orientation(
    r: float, t: float, orientations: Mapping[str, FibsemStagePosition]
) -> str:
    """The orientation a stage at rotation ``r`` and tilt ``t`` (radians) is in.

    The nearest declared pose, of those within tolerance: 5 degrees in r and 0.1 rad in
    t. Each pose the stage declares (SEM, FIB, and FM on a compustage) is a candidate;
    MILLING is not, because it is a range. Nearest rather than first-listed, so a stage
    whose poses sit close together gets the one it is actually at. Failing those,
    MILLING is any tilt above -45 degrees at the SEM rotation, and anything else is
    "NONE".

    Raises:
        ValueError: if SEM, FIB or MILLING is missing from ``orientations``, or has no
            r or t.
    """
    for name in ("SEM", "FIB", "MILLING"):
        pose = orientations.get(name)
        if pose is None:
            raise ValueError(
                "SEM, FIB or MILLING orientation not defined in the system."
            )
        if pose.r is None or pose.t is None:
            raise ValueError(
                "SEM, FIB or MILLING orientation must have both rotation (r) and tilt "
                "(t) defined."
            )
    r = r % (2 * np.pi)

    nearest, nearest_distance = None, np.inf
    for name, pose in orientations.items():
        if name == "MILLING":
            continue
        rotation_error = angle_difference(r, pose.r)
        if rotation_error >= ROTATION_TOLERANCE:
            continue
        if not np.isclose(t, pose.t, atol=TILT_TOLERANCE):
            continue
        distance = (rotation_error / ROTATION_TOLERANCE) ** 2 + (
            (t - pose.t) / TILT_TOLERANCE
        ) ** 2
        if distance < nearest_distance:
            nearest, nearest_distance = name, distance
    if nearest is not None:
        return nearest

    if (
        rotation_angle_is_smaller(r, orientations["SEM"].r, atol=5)
        and MILLING_MIN_TILT < t
    ):
        return "MILLING"
    return "NONE"


def milling_angle_from_tilt(
    stage_tilt: float, pretilt: float, column_tilt: float, turned_over: bool
) -> float:
    """The milling angle, in degrees, a stage tilt (radians) gives at the SEM rotation.

    ``pretilt`` and ``column_tilt`` are the shuttle pre-tilt and the ion column tilt, in
    degrees. Past vertical the sample faces the other way (``turned_over``, which the
    stage device says), so the half turn is unwound first.
    """
    if turned_over:
        stage_tilt += np.radians(180)
    milling_angle = convert_stage_tilt_to_milling_angle(
        stage_tilt=stage_tilt,
        pretilt=np.radians(pretilt),
        column_tilt=np.radians(column_tilt),
    )
    return float(np.degrees(milling_angle))


def compucentric_rotation(
    position: FibsemStagePosition, rotation_centre: Optional[Tuple[float, float]]
) -> FibsemStagePosition:
    """``position`` turned half way round about the rotation centre. A new position.

    Assumes a 180 degree rotation: x and y are reflected through the centre and r gains
    a half turn. A driver's ``rotation_centre`` c is the same centre its images are
    reprojected with (FIB-1081); without one, the centre is the stage origin.
    """
    # The specimen-minus-raw offset is -c, so reflecting about c is -p + 2c.
    cx, cy = (0.0, 0.0) if rotation_centre is None else rotation_centre
    target = deepcopy(position)
    target.r += np.radians(180)
    target.x = 2 * cx - position.x
    target.y = 2 * cy - position.y
    return target


def repose(
    position: FibsemStagePosition,
    current: FibsemStagePosition,
    target: FibsemStagePosition,
    rotation_centre: Optional[Tuple[float, float]],
    has_rotation: bool = True,
) -> FibsemStagePosition:
    """``position``, held in pose ``target`` rather than ``current``. A new position.

    Re-posing is always a rewrite of r and t to the target's. What decides whether x and
    y move with it is whether the *rotation* changes by a half turn: turning the sample
    round swings it about the compucentric centre, so its coordinates are carried round
    with it (:func:`compucentric_rotation`). A change of tilt alone pivots about an axis
    through the sample and leaves x and y where they were.

    The test is "is the rotation a half turn" (within 5 degrees, wrap-aware), measured
    between the two *poses* rather than from ``position.r``: a real stage sits a few
    degrees off nominal, and the correction is a half turn or nothing, so firing it for
    a few degrees of slop would throw the sample across the grid. A stage with no
    rotation axis (``has_rotation`` False, a compustage) has nothing to be compucentric
    about, so x and y stay.
    """
    position = deepcopy(position)
    rotation = angle_difference(current.r, target.r)
    if has_rotation and rotation_angle_is_smaller(rotation, np.pi, atol=5):
        position = compucentric_rotation(position, rotation_centre)
    position.r = target.r
    position.t = target.t
    return position
