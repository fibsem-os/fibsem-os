"""Reprojection: mapping stage positions onto an acquired image, and back.

Pure maths over image metadata -- no microscope. Note the asymmetry with
`tiled.convert_image_coord_to_stage_position`, which goes the other way and *does*
need a live microscope, because it projects through `project_stable_move` rather than
through the baked-in inverse here.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from fibsem import manufacturers, movement
from fibsem.conversions import is_inside_image_bounds
from fibsem.structures import (
    BeamType,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemStagePosition,
    Point,
)
from fibsem.transformations import inverse_view_corrected_dy


def calculate_reprojected_stage_position(
    image: FibsemImage, pos: FibsemStagePosition
) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    # difference between current position and image position
    delta = pos - image.metadata.stage_position

    # projection of the positions onto the image
    dx = delta.x
    dy = np.sqrt(delta.y**2 + delta.z**2)  # TODO: correct for perspective here
    dy = dy if (delta.y < 0) else -dy

    pt_delta = Point(dx, dy)
    px_delta = pt_delta._to_pixels(image.metadata.pixel_size.x)

    beam_type = image.metadata.image_settings.beam_type
    if beam_type is BeamType.ELECTRON:
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation

    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    # account for compustage tilt, when mounted upside down
    if np.isclose(
        image.metadata.stage_position.t, np.radians(-180), atol=np.radians(5)
    ):
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1] / 2, y=image.data.shape[0] / 2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point


def reproject_stage_positions_onto_image(
    image: FibsemImage, positions: List[FibsemStagePosition], bound: bool = False
) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:
        # hotfix (pat): demo returns None positions #240
        if image.metadata.microscope_state.stage_position.x is None:
            image.metadata.microscope_state.stage_position.x = 0
        if image.metadata.microscope_state.stage_position.y is None:
            image.metadata.microscope_state.stage_position.y = 0
        if image.metadata.microscope_state.stage_position.z is None:
            image.metadata.microscope_state.stage_position.z = 0
        if image.metadata.microscope_state.stage_position.r is None:
            image.metadata.microscope_state.stage_position.r = 0
        if image.metadata.microscope_state.stage_position.t is None:
            image.metadata.microscope_state.stage_position.t = 0

        # automate logic for transforming positions
        # assume only two valid positions are when stage is flat to either beam...
        # r needs to be 180 degrees different
        # currently only one way: Flat to Ion -> Flat to Electron
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position(image, pos)
        pt.name = pos.name

        if bound and not is_inside_image_bounds([pt.y, pt.x], image.data.shape):
            continue

        points.append(pt)

    return points


def calculate_reprojected_stage_position2(
    image: FibsemImage, pos: FibsemStagePosition
) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError(
            "Image metadata or microscope state is not set. Cannot reproject stage position."
        )

    if image.metadata.microscope_state.stage_position is None:
        raise ValueError(
            "Image metadata does not contain a valid stage position. Cannot reproject stage position."
        )

    beam_type = image.metadata.image_settings.beam_type
    base_stage_position = image.metadata.microscope_state.stage_position
    pixel_size = image.metadata.pixel_size.x

    scan_rotation = None
    if beam_type is BeamType.ELECTRON:
        if image.metadata.microscope_state.electron_beam is None:
            raise ValueError(
                "Image metadata does not contain a valid electron beam state. Cannot reproject stage position."
            )
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        if image.metadata.microscope_state.ion_beam is None:
            raise ValueError(
                "Image metadata does not contain a valid ion beam state. Cannot reproject stage position."
            )
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation

    if scan_rotation is None:
        raise ValueError(
            "Image metadata does not contain a valid scan rotation. Cannot reproject stage position."
        )

    # difference between current position and image position
    delta = pos - base_stage_position

    # projection of the positions onto the image
    dx = delta.x
    if dx is None:
        raise ValueError(
            "Stage position x coordinate is None. Cannot reproject stage position."
        )

    # Tescan stage x is inverted wrt image coordinates (see TescanMicroscope.stable_move);
    # the y inversion is handled inside _inverse_y_corrected_stage_movement_tescan.
    system_info = image.metadata.system_info
    if system_info is not None and manufacturers.is_tescan(system_info.manufacturer):
        dx = -dx

    # dy = microscope._inverse_y_corrected_stage_movement(dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore
    dy = _inverse_y_corrected_stage_movement(
        image, dy=delta.y, dz=delta.z, beam_type=beam_type
    )  # type: ignore

    pt_delta = Point(dx, -dy)
    px_delta = pt_delta._to_pixels(pixel_size)

    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1] / 2, y=image.data.shape[0] / 2)
    point = image_centre + px_delta

    return point


def reproject_stage_positions_onto_image2(
    image: FibsemImage, positions: List[FibsemStagePosition], bound: bool = False
) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:
        # compucentric rotation correction
        if image.metadata is None or image.metadata.microscope_state is None:
            raise ValueError(
                "Image metadata or microscope state is not set. Cannot reproject stage position."
            )
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError(
                "Image metadata does not contain a valid stage position. Cannot reproject stage position."
            )
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError(
                "Image metadata does not contain a valid stage position. Cannot reproject stage position."
            )
        if image.metadata.microscope_state.stage_position.r is None:
            raise ValueError(
                "Image metadata does not contain a valid stage position r coordinate. Cannot reproject stage position."
            )
        if pos.r is None:
            raise ValueError(
                "Stage position r coordinate is None. Cannot reproject stage position."
            )
        # automate logic for transforming positions
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position2(image, pos)
        pt.name = pos.name

        if bound and not is_inside_image_bounds((pt.y, pt.x), image.data.shape):
            continue

        points.append(pt)

    return points


X_OFFSET = -0.0005127403888932854
Y_OFFSET = 0.0007937916666666666


def _to_specimen_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(
        x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW"
    )
    specimen_position = pos - specimen_offset

    return specimen_position


def _to_raw_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(
        x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW"
    )
    raw_position = pos + specimen_offset

    return raw_position


def _transform_position(pos: FibsemStagePosition) -> FibsemStagePosition:
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam).
    Args:
        pos: The position flat to the beam.
    Returns:
        The position flat to the other beam."""

    specimen_position = _to_specimen_coordinate_system(pos)
    # print("raw      pos: ", pos)
    # print("specimen pos: ", specimen_position)

    # # inverse xy (rotate 180 degrees)
    specimen_position.x = -specimen_position.x
    specimen_position.y = -specimen_position.y

    # movement offset (calibration for compucentric rotation error)
    specimen_position.x += 50e-6
    specimen_position.y += 25e-6

    # print("rotated pos: ", specimen_position)

    # _to_raw_coordinates
    transformed_position = _to_raw_coordinate_system(specimen_position)
    transformed_position.name = pos.name

    # print("trans   pos: ", transformed_position)
    logging.info(f"Initial position {pos} was transformed to {transformed_position}")

    return transformed_position


def _inverse_y_corrected_stage_movement(
    image: FibsemImage,
    dy: float,
    dz: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> float:
    """Recover the in-image y-displacement from a y/z stage movement, off an image.

    The inverse of `_y_corrected_stage_movement`, answered from the image's own
    metadata rather than a live instrument -- so a saved overview projects as it was
    taken.

    Deferred to :func:`fibsem.transformations.inverse_view_corrected_dy` rather than
    derived here. This function used to carry its own copy of the trigonometry, as did
    `FibsemMicroscope._inverse_view_corrected_stage_movement`, so one decision about the
    geometry lived in three places and only stayed consistent by everyone editing all
    three.

    That is also what closes FIB-500. The copy that lived here decided the compustage
    FIB orientation from tilt alone, where the live path and `fm/reprojection.py` both
    also require the rotation to match the reference. Sharing one implementation makes
    all three agree by construction rather than by three edits. The six combinations
    that change are tilt -128 with a non-zero rotation, where the sign inverts -- and a
    compustage has no rotation axis, so no acquisition can produce them.

    Args:
        image: the image whose geometry and pose the projection is taken from.
        dy: actual y stage movement
        dz: actual z stage movement
        beam_type: beam the image was acquired with. Defaults to ELECTRON.

    Returns:
        float: expected_y input that would produce the given dy, dz movements
    """
    if image.metadata is None or image.metadata.hardware_geometry is None:
        raise ValueError(
            "Image metadata or hardware geometry is not set. Cannot calculate inverse y corrected stage movement."
        )

    # Tescan stages have a different geometry (y rides the tilt module, z stays
    # chamber-vertical), so the inverse is a separate derivation, not the
    # compustage-aware path below.
    system_info = image.metadata.system_info
    if system_info is not None and manufacturers.is_tescan(system_info.manufacturer):
        return _inverse_y_corrected_stage_movement_tescan(
            image, dy=dy, dz=dz, beam_type=beam_type
        )

    geometry = image.metadata.hardware_geometry
    position = image.metadata.stage_position
    # The ion column's tilt is the view tilt for a FIB image; the electron column is the
    # reference axis and so contributes none. Same rule as `_beam_view_tilt` on the live
    # microscope, read from the image's geometry instead of the instrument.
    view_tilt = (
        np.deg2rad(geometry.fib_column_tilt) if beam_type is BeamType.ION else 0.0
    )
    return inverse_view_corrected_dy(
        dy=dy,
        dz=dz,
        view_tilt=view_tilt,
        geometry=geometry,
        stage_rotation=position.r if position.r is not None else 0.0,
        stage_tilt=position.t if position.t is not None else 0.0,
    )


def _inverse_y_corrected_stage_movement_tescan(
    image: FibsemImage,
    dy: float,
    dz: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> float:
    """Tescan inverse of _y_corrected_stage_movement, from image metadata.

    Thin adapter: pulls the geometry and stage pose out of the image metadata and defers
    to inverse_y_corrected_stage_movement_tescan_from_geometry. TescanMicroscope's own
    method is the other adapter over the same core (from the live microscope), so the two
    can no longer drift apart.

    Args:
        dy (float): actual y stage movement (raw stage frame)
        dz (float): actual z stage movement (raw stage frame)
        beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

    Returns:
        float: expected_y input that would produce the given dy, dz movements
    """
    if image.metadata is None or image.metadata.hardware_geometry is None:
        raise ValueError(
            "Image metadata or hardware geometry is not set. Cannot calculate inverse y corrected stage movement."
        )

    return inverse_y_corrected_stage_movement_tescan_from_geometry(
        geometry=image.metadata.hardware_geometry,
        stage_position=image.metadata.stage_position,
        dy=dy,
        dz=dz,
        beam_type=beam_type,
    )


def _tescan_pose_angles(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
) -> Tuple[float, float, float]:
    """(stage_tilt, corrected_pretilt_angle, sample_inclination), all in radians.

    The pre-tilt sign flips when the stage is rotated 180 deg to face the ion beam.
    Single home for that rule: the forward, the inverse, and the microscope's debug
    logging all read these three angles from here, so they cannot drift apart.
    """
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    stage_pretilt = np.deg2rad(geometry.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(geometry.rotation_reference) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(geometry.rotation_180) % (2 * np.pi)

    stage_rotation = (
        stage_position.r % (2 * np.pi) if stage_position.r is not None else 0.0
    )
    stage_tilt = stage_position.t if stage_position.t is not None else 0.0

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_eb, atol=5
    ):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_ion, atol=5
    ):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)
    sample_inclination = stage_tilt - corrected_pretilt_angle
    return stage_tilt, corrected_pretilt_angle, sample_inclination


def y_corrected_stage_movement_tescan_from_geometry(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
    expected_y: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> Tuple[float, float]:
    """Tescan forward of the sample-plane move — the single source of the math.

    The counterpart of :func:`inverse_y_corrected_stage_movement_tescan_from_geometry`.
    ``TescanMicroscope._y_corrected_stage_movement`` and the overview canvas
    (``fibsem/projection.py``) are thin adapters over this function.

    Tescan stage axes (corrected 2026-08-25 from the 2026-07-22 session log): the
    y-axis is mounted ON the tilt module -- a y command travels along the tilted stage
    plate -- while z stays chamber-vertical (+z is DOWN, verified on hardware
    2026-07-23). The two translation axes are therefore non-orthogonal at tilt: the
    in-plane move needs 1/cos(stage_tilt) on y, and the z command cancels only the
    vertical the tilted y-axis introduces beyond the sample plane, which depends on
    the (signed) pre-tilt alone. The previous chamber-fixed-y model overshot the ion
    view 1.65x at the milling pose; a chamber-fixed model cannot overshoot at all, so
    the observation refutes it. See
    https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414
    for the derivation, the sign chain, and the hardware evidence.

    Thermo's pair is parameterised by a *view tilt* and lives in
    :mod:`fibsem.transformations`; Tescan cannot join it, because its axes decompose
    against the stage tilt and pre-tilt separately rather than the pre-tilt alone.

    Args:
        geometry: fixed instrument geometry (column tilts, pre-tilt, rotation refs).
        stage_position: stage pose at acquisition (rotation r and tilt t, radians).
        expected_y: distance along the image y-axis, in metres.
        beam_type: beam perspective to correct for.

    Returns:
        (y_move, z_move) in metres -- before the stage-axis inversion ``stable_move``
        applies (``y_stage = -y_move``, z unchanged). The inverse undoes that same
        inversion on its way in, so the two compose.
    """
    stage_tilt, corrected_pretilt_angle, sample_inclination = _tescan_pose_angles(
        geometry, stage_position
    )
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)
    beam_tilt = sem_column_tilt if beam_type is BeamType.ELECTRON else fib_column_tilt

    # perspective: image-projected dy -> true distance along the sample plane
    y_sample_move = expected_y / np.cos(sample_inclination - beam_tilt)

    y_move = y_sample_move * np.cos(sample_inclination) / np.cos(stage_tilt)
    z_move = y_sample_move * np.sin(corrected_pretilt_angle) / np.cos(stage_tilt)
    return float(y_move), float(z_move)


def inverse_y_corrected_stage_movement_tescan_from_geometry(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
    dy: float,
    dz: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> float:
    """Tescan inverse of _y_corrected_stage_movement — the single source of the math.

    Takes the raw geometry (column tilts, pre-tilt, rotation references) and the stage pose
    at acquisition, and returns the image-space dy that would produce the given raw stage
    deltas. Both the image-metadata path (_inverse_y_corrected_stage_movement_tescan) and
    the live-microscope path (TescanMicroscope._inverse_y_corrected_stage_movement) are thin
    adapters over this function; the microscope builds `geometry` from
    `self.hardware_geometry()`, whose fields are the same ones the image carries.

    Tescan stage axes: y rides ON the tilt module, z is chamber-vertical (+z down) --
    see :func:`y_corrected_stage_movement_tescan_from_geometry` and
    https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414 for the derivation.

    Args:
        geometry: fixed instrument geometry (column tilts, shuttle pre-tilt, rotation refs).
        stage_position: stage pose at acquisition (rotation r and tilt t, radians).
        dy (float): actual y stage movement (raw stage frame).
        dz (float): actual z stage movement (raw stage frame).
        beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

    Returns:
        float: expected_y input that would produce the given dy, dz movements.
    """
    # undo the stage-axis inversion applied in stable_move (y_stage = -y_move)
    # TODO(hardware-verify): keep in sync with the x/y inversion in TescanMicroscope.stable_move.
    dy = -dy

    stage_tilt, corrected_pretilt_angle, sample_inclination = _tescan_pose_angles(
        geometry, stage_position
    )
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)
    beam_tilt = sem_column_tilt if beam_type is BeamType.ELECTRON else fib_column_tilt

    # invert: forward is y = d*cos(incl)/cos(tilt), z = d*sin(pretilt)/cos(tilt);
    # recover d from the better-conditioned component. With zero corrected pre-tilt
    # the z move is identically zero and carries no information, so ties (and every
    # pose where |cos(incl)| dominates) go to the y branch.
    cos_incl = np.cos(sample_inclination)
    sin_pretilt = np.sin(corrected_pretilt_angle)
    if abs(cos_incl) >= abs(sin_pretilt):
        y_sample_move = dy * np.cos(stage_tilt) / cos_incl
    else:
        y_sample_move = dz * np.cos(stage_tilt) / sin_pretilt

    # re-project the sample-plane distance into the image plane
    expected_y = y_sample_move * np.cos(sample_inclination - beam_tilt)

    return expected_y
