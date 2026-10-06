import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

from fibsem.geometry.frames import (
    StageModel,
    image_y_shift,
    in_plane_move,
    views_back,
)

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
    microscope: "FibsemMicroscope", milling_angle: float
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
    microscope: "FibsemMicroscope", milling_angle: float, atol: float = np.deg2rad(2)
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


def _image_y_flip(
    geometry: "FibsemHardwareGeometry",
    view_tilt: float,
    stage_rotation: float,
    stage_tilt: float,
) -> float:
    """-1 where the instrument mirrors the image, so image y runs against the geometry.

    A compustage's instrument flips an image the beam takes of the back of the grid,
    so it reads as if seen from the front (FIB-1101). Which side a view sees is the
    model's (`fibsem.geometry.frames.views_back`); that the instrument mirrors it is
    the compustage's. Other stages present the image as the beam sees it.
    """
    if not geometry.is_compustage:
        return 1.0
    model = StageModel.from_geometry(geometry)
    return -1.0 if views_back(model, view_tilt, stage_rotation, stage_tilt) else 1.0


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
    flip = _image_y_flip(geometry, view_tilt, stage_rotation, stage_tilt)
    return in_plane_move(
        StageModel.from_geometry(geometry),
        flip * expected_y,
        view_tilt,
        stage_rotation,
        stage_tilt,
    )


def inverse_view_corrected_dy(
    dy: float,
    dz: float,
    view_tilt: float,
    geometry: "FibsemHardwareGeometry",
    stage_rotation: float,
    stage_tilt: float,
) -> float:
    """Where a y/z stage movement lands in the image, as an in-image y-displacement.

    The microscope-free form of
    :meth:`FibsemMicroscope._inverse_view_corrected_stage_movement`, parameterised by
    the geometry rather than reading it off a live instrument. The two are held
    together by a parity test across the full pose matrix, because a projection that
    silently disagrees with the one used to move the stage puts every overlay in the
    wrong place.

    Not just the algebraic undo of :func:`view_corrected_stage_movement`, and the
    difference is the point (FIB-766). The forward map only ever *produces* in-plane
    movements -- a click slides the sample along its own surface -- but this is fed
    arbitrary position deltas: two saved positions can differ in height, and a
    coincidence correction is a chamber-vertical move. So the delta is resolved into
    its in-plane and surface-normal components and each is projected through the view:

        u   = dy*cos(a) - dz*sin(a)       in-plane component
        n   = dy*sin(a) + dz*cos(a)       surface-normal component
        e_y = u*cos(phi) - n*sin(phi)     phi = folded_tilt - a - view_tilt

    On in-plane deltas (n = 0) this agrees exactly with inverting the forward, which
    is what the parity tests pin. Off-plane it is what the old form was not: a
    chamber-vertical move projects to zero in the SEM view (its u and n image shifts
    cancel -- the old form kept only the u half, showing a phantom shift), and to
    sin(view_tilt) times the height in a tilted view (the FIB really does see a
    height change move; the old form discarded dz and showed nothing). Verified
    against an independently calibrated 3D model in tests/test_projection_height.py.

    Args:
        dy: stage y movement, in metres.
        dz: stage z movement, in metres.
        view_tilt: tilt of the viewing axis from the electron column, in radians.
        geometry: the geometry the image was captured under.
        stage_rotation: stage rotation at acquisition, in radians.
        stage_tilt: stage tilt at acquisition, in radians.

    Returns:
        The in-image y-displacement produced by that stage movement, in metres.
    """
    flip = _image_y_flip(geometry, view_tilt, stage_rotation, stage_tilt)
    model = StageModel.from_geometry(geometry)
    return flip * image_y_shift(model, dy, dz, view_tilt, stage_tilt)
