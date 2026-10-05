"""Stage orientation and milling angle: what view a stage pose is in, and at what angle.

Pure functions, with no microscope and no I/O. Each takes a stage pose and the
instrument geometry, so the same answer serves the live stage and a saved image.

They used to exist only as methods on ``FibsemMicroscope``, which reached through
``self.system`` for its inputs and so looked as if it needed a connection. It never
did: every term it reads is in ``FibsemHardwareGeometry``, which every image records
(FIB-481; images older than that have it recovered from their embedded ``system``). So
anything without a microscope -- a report, the web monitor, analysis of a saved
experiment -- can name the view an image was taken in, from the image alone (FIB-811).

Derived, not stored. An orientation written into the metadata would only exist on
images saved after it was added, and would keep whatever answer was current when it
was written; derived from the image's own geometry, it covers every image already on
disk and follows any correction to the definitions here.

The microscope keeps its orientation table (``FibsemMicroscope.orientations``): it is
what a move to a named orientation reads, and it is built here, by
:func:`orientation_poses`. Classifying against that table rather than re-deriving it
means a live classification and a live move always agree on where an orientation is.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fibsem.movement import rotation_angle_is_smaller
from fibsem.structures import (
    FibsemHardwareGeometry,
    FibsemImageMetadata,
    FibsemStagePosition,
)
from fibsem.transformations import (
    convert_milling_angle_to_stage_tilt,
    convert_stage_tilt_to_milling_angle,
)


def orientation_poses(
    geometry: FibsemHardwareGeometry, milling_angle: Optional[float] = None
) -> Dict[str, FibsemStagePosition]:
    """The stage rotation and tilt of each named orientation.

    Args:
        geometry: the instrument geometry.
        milling_angle: the site's target milling angle, in degrees. MILLING is
            included only when it is given: it is a setting, not part of the geometry,
            and an image does not record it. Classifying a pose does not need it.

    Returns:
        {"SEM", "FIB", ["MILLING"], ["FM"]} -> pose with r and t set, in radians.
    """
    shuttle_pre_tilt = geometry.shuttle_pre_tilt  # deg

    poses = {
        "SEM": FibsemStagePosition(
            r=np.radians(geometry.rotation_reference),
            t=np.radians(shuttle_pre_tilt),
        ),
        "FIB": FibsemStagePosition(
            r=np.radians(geometry.rotation_180),
            t=np.radians(geometry.fib_column_tilt - shuttle_pre_tilt),
        ),
    }
    if milling_angle is not None:
        poses["MILLING"] = FibsemStagePosition(
            r=np.radians(geometry.rotation_reference),
            t=convert_milling_angle_to_stage_tilt(
                np.radians(milling_angle),
                pretilt=np.radians(shuttle_pre_tilt),
                column_tilt=np.radians(geometry.fib_column_tilt),
            ),
        )

    # FM is an orientation only where reaching the FM *is* a re-pose: on a
    # compustage the objective is under the grid and the stage turns over to face
    # it. On an offset mount the FM is a place, not a pose -- the stage travels
    # there holding whatever orientation it was in -- so there is no FM entry to
    # derive.
    if geometry.is_compustage:
        poses["FIB"].r = np.radians(0)  # Compustage is always at 0 rotation
        poses["FIB"].t -= np.radians(180)

        poses["FM"] = FibsemStagePosition(
            r=np.radians(0),
            t=np.radians(-180),
        )

    return poses


def classify_orientation(
    stage_position: FibsemStagePosition, poses: Dict[str, FibsemStagePosition]
) -> str:
    """Which named orientation a stage pose is at.

    Args:
        stage_position: the pose to classify; r and t must be set.
        poses: the orientation table, as :func:`orientation_poses` builds it. SEM and
            FIB are required; FM is matched only if present.

    Returns:
        "SEM", "FIB", "MILLING", "FM", or "NONE" for a pose at none of them.
    """
    if stage_position.r is None or stage_position.t is None:
        raise ValueError(
            "Stage position must have both rotation (r) and tilt (t) defined."
        )
    stage_rotation = stage_position.r % (2 * np.pi)
    stage_tilt = stage_position.t
    # TODO: also check xyz ranges?

    sem = poses.get("SEM")
    fib = poses.get("FIB")
    # FM is an orientation only on a compustage -- see `orientation_poses`. On an
    # offset mount there is no FM pose to classify against.
    fm = poses.get("FM")
    if sem is None or fib is None:
        raise ValueError("SEM or FIB orientation not defined.")
    if sem.r is None or sem.t is None or fib.r is None or fib.t is None:
        raise ValueError(
            "SEM and FIB orientations must have both rotation (r) and tilt (t) defined."
        )

    is_sem_rotation = rotation_angle_is_smaller(stage_rotation, sem.r, atol=5)
    is_fib_rotation = rotation_angle_is_smaller(stage_rotation, fib.r, atol=5)
    is_fm_rotation = fm is not None and rotation_angle_is_smaller(
        stage_rotation, fm.r, atol=5
    )

    is_sem_tilt = np.isclose(stage_tilt, sem.t, atol=0.1)
    is_fib_tilt = np.isclose(stage_tilt, fib.t, atol=0.1)

    # MILLING is any tilt at the SEM rotation down to a fixed floor, not the
    # configured milling angle's tilt: a lamella milled at 12 degrees and one milled
    # at 20 are both at the milling orientation.
    is_milling_tilt = np.radians(-45) < stage_tilt and not is_sem_tilt
    is_fm_tilt = fm is not None and np.isclose(stage_tilt, fm.t, atol=0.1)

    if is_sem_rotation and is_sem_tilt:
        return "SEM"
    if is_sem_rotation and is_milling_tilt:
        return "MILLING"
    if is_fib_rotation and is_fib_tilt:
        return "FIB"
    if is_fm_rotation and is_fm_tilt:
        return "FM"

    return "NONE"


def stage_orientation(
    stage_position: FibsemStagePosition, geometry: FibsemHardwareGeometry
) -> str:
    """Which named orientation a stage pose is at, under the given geometry.

    Returns:
        "SEM", "FIB", "MILLING", "FM", or "NONE" for a pose at none of them.
    """
    return classify_orientation(stage_position, orientation_poses(geometry))


def stage_milling_angle(
    stage_position: FibsemStagePosition,
    geometry: FibsemHardwareGeometry,
    orientation: Optional[str] = None,
) -> float:
    """The angle the ion beam meets the sample at, for a stage pose, in degrees.

    ``milling_angle = 90 - column_tilt + stage_tilt - pretilt``. This is the angle of
    the pose, not the site's target milling angle.

    Args:
        stage_position: the pose; r and t must be set.
        geometry: the instrument geometry.
        orientation: the pose's orientation, when the caller has already classified
            it (the live microscope classifies against its own table); None
            classifies it from ``geometry``.
    """
    if orientation is None:
        orientation = stage_orientation(stage_position, geometry)

    # NOTE: this is only valid for sem orientation
    if orientation == "FIB":
        return 90  # stage-tilt + pre-tilt + 90 - column-tilt

    stage_tilt = stage_position.t

    if stage_tilt is None:
        raise ValueError("Stage tilt is not available. Cannot calculate milling angle.")

    if geometry.is_compustage and stage_tilt < np.radians(-90):
        # Compustage stage tilt is inverted, so we need to adjust the angle
        stage_tilt += np.radians(180)

    angle = convert_stage_tilt_to_milling_angle(
        stage_tilt=stage_tilt,
        pretilt=np.radians(geometry.shuttle_pre_tilt),
        column_tilt=np.radians(geometry.fib_column_tilt),
    )
    return float(np.degrees(angle))


def _recorded_pose(metadata: FibsemImageMetadata) -> Optional[FibsemStagePosition]:
    state = metadata.microscope_state
    position = state.stage_position if state is not None else None
    if position is None or position.r is None or position.t is None:
        return None
    return position


def image_orientation(metadata: FibsemImageMetadata) -> Optional[str]:
    """The named orientation an image was acquired at, from its own metadata.

    Returns:
        "SEM", "FIB", "MILLING", "FM" or "NONE"; None when the image does not record
        its geometry or its stage pose. That is an unknown, not "NONE": nothing says
        where the stage was.
    """
    position = _recorded_pose(metadata)
    if position is None or metadata.hardware_geometry is None:
        return None
    return stage_orientation(position, metadata.hardware_geometry)


def image_milling_angle(metadata: FibsemImageMetadata) -> Optional[float]:
    """The milling angle of the pose an image was acquired at, in degrees.

    None when the image does not record its geometry or its stage pose.
    """
    position = _recorded_pose(metadata)
    if position is None or metadata.hardware_geometry is None:
        return None
    return stage_milling_angle(position, metadata.hardware_geometry)
