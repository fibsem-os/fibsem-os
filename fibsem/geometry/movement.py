"""Movement geometry: what stage movement a displacement seen in an image needs.

Pure functions, with no microscope and no I/O. Each takes a displacement in the image
the user is looking at, the geometry of the view it was seen in, and the stage pose, and
returns a stage movement. Reading those inputs off an instrument, commanding the move and
restoring the working distance afterwards is the caller's job -- today
``FibsemMicroscope.stable_move`` and ``vertical_move``, which every backend except
Tescan shares.

A beam and the fluorescence camera are the same kind of view here. They differ only in
how far their viewing axis is tilted from the electron column (``view_tilt``) and in the
display transform undone before projecting: a beam's scan rotation, the camera's image
transform. The projection itself is :func:`fibsem.transformations.
view_corrected_stage_movement`, which the saved-image reprojection shares.

Tescan keeps its own model: its y-axis travels on the tilted plate, so its stage
movements are not these (see ``TescanMicroscope._y_corrected_stage_movement``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from fibsem.structures import FibsemHardwareGeometry, FibsemStagePosition
from fibsem.transformations import view_corrected_stage_movement


def undo_scan_rotation(
    dx: float, dy: float, scan_rotation: float
) -> Tuple[float, float]:
    """A displacement in the displayed image, in the beam's unrotated scan frame.

    A half-turn of scan rotation inverts both image axes. Other scan rotations are not
    corrected, as they never have been.
    """
    if np.isclose(scan_rotation, np.pi):
        return dx * -1.0, dy * -1.0
    return dx, dy


def image_to_stage_delta(
    dx: float,
    dy: float,
    view_tilt: float,
    geometry: FibsemHardwareGeometry,
    stage_rotation: float,
    stage_tilt: float,
    is_fib_orientation: Optional[bool] = None,
) -> FibsemStagePosition:
    """Relative stage movement that slides the sample by (dx, dy) in a view.

    The image x-axis is the stage x-axis. The y-displacement moves the sample along its
    own tilted surface, so it is split across the stage y- and z-axes.

    Args:
        dx: displacement along the image x-axis, in metres, display transform undone.
        dy: displacement along the image y-axis, in metres, display transform undone.
        view_tilt: tilt of the viewing axis from the electron column, in radians.
        geometry: the instrument geometry.
        stage_rotation: stage rotation, in radians.
        stage_tilt: stage tilt, in radians.
        is_fib_orientation: whether a compustage is at its FIB orientation, when the
            caller has classified the pose; None derives it from the pose.

    Returns:
        FibsemStagePosition: relative movement in the RAW coordinate system.
    """
    y_move, z_move = view_corrected_stage_movement(
        expected_y=dy,
        view_tilt=view_tilt,
        geometry=geometry,
        stage_rotation=stage_rotation,
        stage_tilt=stage_tilt,
        is_fib_orientation=is_fib_orientation,
    )
    return FibsemStagePosition(
        x=dx, y=y_move, z=z_move, r=0, t=0, coordinate_system="RAW"
    )


def vertical_move_delta(
    dx: float,
    dy: float,
    scan_rotation: float,
    fib_column_tilt: float,
    stage_tilt: float,
    is_compustage: bool,
    relaxation: float = 1.0,
) -> FibsemStagePosition:
    """Relative stage movement that cancels an offset seen in the FIB view by height.

    A chamber-vertical displacement of dz appears in the FIB view as
    dz * sin(column_tilt), whatever the stage tilt, so the height change that cancels an
    observed dy is dy / sin(column_tilt). It is invisible to the electron beam, which is
    what keeps a feature already centred in the SEM where it is.

    The chamber-vertical is then decomposed into the tilted stage axes: y = m*sin(t),
    z = m*cos(t) (FIB-773).

    Args:
        dx: offset along the displayed FIB image x-axis, in metres; moved as is.
        dy: offset along the displayed FIB image y-axis, in metres.
        scan_rotation: the ion beam's scan rotation, in radians.
        fib_column_tilt: the ion column tilt, in degrees.
        stage_tilt: stage tilt, in radians.
        is_compustage: whether the stage is a compustage.
        relaxation: under-relaxation of the correction; 1.0 is geometrically exact.

    Returns:
        FibsemStagePosition: relative movement in the RAW coordinate system.
    """
    dx, dy = undo_scan_rotation(dx, dy, scan_rotation)

    # TODO: ARCTIS Do we need to reverse the direction of the movement because of the inverted stage tilt?
    if is_compustage:
        dy *= -1.0
        if stage_tilt >= np.deg2rad(-90):
            dy *= -1.0

    z_move = dy / np.sin(np.deg2rad(fib_column_tilt)) * relaxation

    return FibsemStagePosition(
        x=dx,
        y=z_move * np.sin(stage_tilt),
        z=z_move * np.cos(stage_tilt),
        coordinate_system="RAW",
    )


def fib_offset_after_sem_move(
    stage_dy: float,
    ion_scan_rotation: float,
    milling_angle: Optional[float] = None,
) -> float:
    """The FIB-view offset to correct after a stable move in the SEM view.

    Restoring coincidence from the SEM is a stable move that centres the feature in the
    SEM, then a vertical move that puts the FIB back. This is the offset handed to the
    second: the stage y the first one travelled, scaled by the milling angle when the
    stage is at the SEM or milling orientation, with the sign the vertical move expects.
    Accurate for small moves; less so over long distances and at high tilt.

    Args:
        stage_dy: stage y travelled by the SEM stable move, in metres.
        ion_scan_rotation: the ion beam's scan rotation, in radians.
        milling_angle: the current milling angle in degrees, when the stage is at the
            SEM or milling orientation; None elsewhere, which leaves dy unscaled.

    Returns:
        The offset along the FIB image y-axis, in metres.
    """
    dy = stage_dy
    if milling_angle is not None:
        dy = dy * np.sin(np.radians(milling_angle))

    # the vertical move also undoes the scan rotation, so it is pre-inverted at 0
    if np.isclose(ion_scan_rotation, 0):
        dy *= -1.0
    return dy


def apply_delta(
    base_position: FibsemStagePosition, delta: FibsemStagePosition
) -> FibsemStagePosition:
    """Where a relative x/y/z movement from ``base_position`` lands.

    A copy: ``base_position`` is not modified. Rotation and tilt are carried over.
    """
    new_position = deepcopy(base_position)
    new_position.x += delta.x
    new_position.y += delta.y
    new_position.z += delta.z
    return new_position
