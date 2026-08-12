import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

from fibsem.movement import rotation_angle_is_smaller

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemHardwareGeometry

def convert_milling_angle_to_stage_tilt(
    milling_angle: float, pretilt: float, column_tilt: float = np.deg2rad(52)
) -> float:
    """Convert the milling angle to the stage tilt angle, based on pretilt and column tilt.
        milling_angle = 90 - column_tilt + stage_tilt - pretilt
        stage_tilt = milling_angle - 90 + pretilt + column_tilt
    Args:
        milling_angle: milling angle (radians)
        pretilt: pretilt angle (radians)
        column_tilt: column tilt angle (radians)
    Returns:
        stage_tilt: stage tilt (radians)"""

    stage_tilt = milling_angle + column_tilt + pretilt - np.deg2rad(90)

    return stage_tilt


def convert_stage_tilt_to_milling_angle(
    stage_tilt: float, pretilt: float, column_tilt: float = np.deg2rad(52)
) -> float:
    """Convert the stage tilt angle to the milling angle, based on pretilt and column tilt.
        milling_angle = 90 - column_tilt + stage_tilt - pretilt
    Args:
        stage_tilt: stage tilt (radians)
        pretilt: pretilt angle (radians)
        column_tilt: column tilt angle (radians)
    Returns:
        milling_angle: milling angle (radians)"""

    milling_angle = np.deg2rad(90) - column_tilt + stage_tilt - pretilt

    return milling_angle


def get_stage_tilt_from_milling_angle(
    microscope: 'FibsemMicroscope', milling_angle: float
) -> float:
    """Get the stage tilt angle from the milling angle, based on pretilt and column tilt.
    Args:
        microscope (FibsemMicroscope): microscope connection
        milling_angle (float): milling angle (radians)
    Returns:
        float: stage tilt angle (radians)
    """
    pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)
    column_tilt = np.deg2rad(microscope.system.ion.column_tilt)
    stage_tilt = convert_milling_angle_to_stage_tilt(
        milling_angle, pretilt, column_tilt
    )
    return stage_tilt

def is_close_to_milling_angle(
    microscope: 'FibsemMicroscope', milling_angle: float, atol: float = np.deg2rad(2)
) -> bool:
    """Check if the stage tilt is close to the milling angle, within a tolerance.
    Args:
        microscope (FibsemMicroscope): microscope connection
        milling_angle (float): milling angle (radians)
        atol (float): tolerance in radians
    Returns:
        bool: True if the stage tilt is within the tolerance of the milling angle
    """
    current_stage_tilt = microscope.get_stage_position().t
    pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)
    column_tilt = np.deg2rad(microscope.system.ion.column_tilt)
    stage_tilt = convert_milling_angle_to_stage_tilt(
        milling_angle, pretilt=pretilt, column_tilt=column_tilt
    )
    logging.info(
        f"The current stage tilt is {np.degrees(current_stage_tilt):.2f} deg, "
        f"the stage tilt for the milling angle is {np.degrees(stage_tilt):.2f} deg"
    )
    return np.isclose(stage_tilt, current_stage_tilt, atol=atol)


# ── the view projection, without a microscope ────────────────────────────────
#
# Every view of the sample runs the same projection and differs only in how far its
# viewing axis is tilted from the electron column: the electron column is 0, the ion
# column is its `column_tilt`, the fluorescence camera is its `camera_tilt`. That is
# already how the live path is written -- `FibsemMicroscope._view_corrected_stage_
# movement(expected_y, view_tilt)`, with `_beam_view_tilt` and `camera_tilt` as its two
# callers -- and these are the microscope-free form of the same thing, parameterised by
# a recorded `FibsemHardwareGeometry` and pose instead of reading a live instrument.
#
# Microscope-free matters because these answer questions *about a saved image*: where a
# stage position falls on it, and where a click on it points. Reading the pose or the
# transform from the live instrument would make the answer depend on state the image
# does not describe -- and looking at a saved image after the stage has moved is exactly
# when someone asks.
#
# They live here rather than beside either modality's reprojection module because both
# use them. They arrived in `fibsem/fm/reprojection.py` with the camera tilt baked in,
# which read as fluorescence-specific and was not.


def _projection_terms(
    geometry: "FibsemHardwareGeometry",
    stage_rotation: float,
    stage_tilt: float,
) -> Tuple[float, float, float]:
    """The sign and angle terms both directions of the projection share.

    Factored out rather than written twice: the forward and the inverse differ only in
    the arithmetic that follows, and a sign convention that drifts between them would
    make a click land somewhere other than where the marker was drawn -- while each
    direction on its own still looked self-consistent.

    Returns:
        (compustage_sign, corrected_pretilt_angle, stage_tilt), where `stage_tilt` has
        the compustage half-turn already folded in.
    """
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    stage_pretilt = np.deg2rad(geometry.shuttle_pre_tilt)
    rotation_flat_to_eb = np.deg2rad(geometry.rotation_reference) % (2 * np.pi)
    rotation_flat_to_ion = np.deg2rad(geometry.rotation_180) % (2 * np.pi)

    stage_rotation = stage_rotation % (2 * np.pi)

    # The forward flips expected_y once for a compustage and a second time at the FIB
    # orientation, so the two cancel there. The orientation is derived from the pose
    # rather than from a live microscope's `get_stage_orientation`.
    #
    # The rotation test is not redundant with the tilt test: a compustage cannot
    # rotate, so `get_stage_orientation` reports FIB only at the reference rotation.
    # Keying on tilt alone would call an unreachable pose FIB and flip the sign
    # against the live path -- which is a disagreement no physical run can surface,
    # and so exactly the kind that survives. `fibsem/imaging/tiling/reprojection.py`
    # still keys on tilt alone; that is FIB-500, deliberately left for its own change
    # because it moves a stage sign.
    compustage_sign = 1.0
    is_fib_orientation = False
    if geometry.is_compustage:
        fib_orientation_tilt = np.deg2rad(
            geometry.fib_column_tilt - geometry.shuttle_pre_tilt - 180
        )
        is_fib_orientation = bool(
            np.isclose(stage_tilt, fib_orientation_tilt, atol=0.1)
            and rotation_angle_is_smaller(stage_rotation, rotation_flat_to_eb, atol=5)
        )
        compustage_sign = 1.0 if is_fib_orientation else -1.0
        stage_tilt = stage_tilt + np.pi

    pretilt_sign = 1.0
    if rotation_angle_is_smaller(stage_rotation, rotation_flat_to_eb, atol=5):
        pretilt_sign = 1.0
    if rotation_angle_is_smaller(stage_rotation, rotation_flat_to_ion, atol=5):
        pretilt_sign = -1.0
    if is_fib_orientation:
        pretilt_sign = -1.0

    corrected_pretilt_angle = pretilt_sign * (stage_pretilt + sem_column_tilt)

    return compustage_sign, corrected_pretilt_angle, stage_tilt


def view_corrected_stage_movement(
    expected_y: float,
    view_tilt: float,
    geometry: "FibsemHardwareGeometry",
    stage_rotation: float,
    stage_tilt: float,
) -> Tuple[float, float]:
    """Split an in-image y-displacement across the stage y- and z-axes.

    The microscope-free form of
    :meth:`FibsemMicroscope._view_corrected_stage_movement`.

    Args:
        expected_y: displacement along the image y-axis, in metres.
        view_tilt: tilt of the viewing axis from the electron column, in radians.
            0 for the electron beam, the ion column tilt for the ion beam, the camera
            tilt for fluorescence.
        geometry: the geometry the image was captured under.
        stage_rotation: stage rotation at acquisition, in radians.
        stage_tilt: stage tilt at acquisition, in radians.

    Returns:
        (dy, dz) stage movement, in metres.
    """
    compustage_sign, corrected_pretilt_angle, stage_tilt = _projection_terms(
        geometry, stage_rotation, stage_tilt
    )

    if geometry.is_compustage:
        expected_y = expected_y * compustage_sign

    perspective_tilt_adjustment = -corrected_pretilt_angle - view_tilt
    y_sample_move = expected_y / np.cos(stage_tilt + perspective_tilt_adjustment)

    return (
        float(y_sample_move * np.cos(corrected_pretilt_angle)),
        float(-y_sample_move * np.sin(corrected_pretilt_angle)),
    )


def inverse_view_corrected_dy(
    dy: float,
    dz: float,
    view_tilt: float,
    geometry: "FibsemHardwareGeometry",
    stage_rotation: float,
    stage_tilt: float,
) -> float:
    """Recover an in-image y-displacement from a y/z stage movement.

    The microscope-free form of
    :meth:`FibsemMicroscope._inverse_view_corrected_stage_movement`, parameterised by
    the geometry rather than reading it off a live instrument. The two are held
    together by a parity test across the full pose matrix, because a projection that
    silently disagrees with the one used to move the stage puts every overlay in the
    wrong place.

    Args:
        dy: stage y movement, in metres.
        dz: stage z movement, in metres.
        view_tilt: tilt of the viewing axis from the electron column, in radians.
        geometry: the geometry the image was captured under.
        stage_rotation: stage rotation at acquisition, in radians.
        stage_tilt: stage tilt at acquisition, in radians.

    Returns:
        The in-image y-displacement that would produce that stage movement, in metres.
    """
    compustage_sign, corrected_pretilt_angle, stage_tilt = _projection_terms(
        geometry, stage_rotation, stage_tilt
    )

    perspective_tilt_adjustment = -corrected_pretilt_angle - view_tilt

    # Undo y_move = y_sample_move * cos(a) and z_move = -y_sample_move * sin(a),
    # taking whichever component is larger so the division stays conditioned.
    cos_pretilt = np.cos(corrected_pretilt_angle)
    sin_pretilt = np.sin(corrected_pretilt_angle)
    if abs(cos_pretilt) > abs(sin_pretilt):
        y_sample_move = dy / cos_pretilt
    else:
        y_sample_move = -dz / sin_pretilt

    expected_y = y_sample_move * np.cos(stage_tilt + perspective_tilt_adjustment)

    if geometry.is_compustage:
        expected_y *= compustage_sign

    return float(expected_y)
