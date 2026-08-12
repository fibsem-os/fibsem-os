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

from fibsem import movement
from fibsem.conversions import is_inside_image_bounds
from fibsem.structures import (
    BeamType,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemStagePosition,
    Point,
)


def calculate_reprojected_stage_position(image: FibsemImage, pos: FibsemStagePosition) -> Point:
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
    dy = np.sqrt(delta.y**2 + delta.z**2) # TODO: correct for perspective here
    dy = dy if (delta.y<0) else -dy

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
    if np.isclose(image.metadata.stage_position.t, np.radians(-180), atol=np.radians(5)):
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point

def reproject_stage_positions_onto_image(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
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

def calculate_reprojected_stage_position2(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")

    if image.metadata.microscope_state.stage_position is None:
        raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")


    beam_type = image.metadata.image_settings.beam_type
    base_stage_position = image.metadata.microscope_state.stage_position 
    pixel_size = image.metadata.pixel_size.x

    scan_rotation = None
    if beam_type is BeamType.ELECTRON:
        if image.metadata.microscope_state.electron_beam is None:
            raise ValueError("Image metadata does not contain a valid electron beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        if image.metadata.microscope_state.ion_beam is None:
            raise ValueError("Image metadata does not contain a valid ion beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation

    if scan_rotation is None:
        raise ValueError("Image metadata does not contain a valid scan rotation. Cannot reproject stage position.")

    # difference between current position and image position
    delta = pos - base_stage_position

    # projection of the positions onto the image
    dx = delta.x
    if dx is None:
        raise ValueError("Stage position x coordinate is None. Cannot reproject stage position.")

    # Tescan stage x is inverted wrt image coordinates (see TescanMicroscope.stable_move);
    # the y inversion is handled inside _inverse_y_corrected_stage_movement_tescan. Match
    # case-insensitively — a live scope reports "TESCAN", config/saved images "Tescan"; an
    # exact check silently fell back to Thermo math and placed added positions at the
    # opposite corner. (band-aid string check, tracked in FIB-300)
    system_info = image.metadata.system_info
    if system_info is not None and system_info.manufacturer.upper() == "TESCAN":
        dx = -dx

    # dy = microscope._inverse_y_corrected_stage_movement(dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore
    dy = _inverse_y_corrected_stage_movement(image, dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore

    pt_delta = Point(dx, -dy)
    px_delta = pt_delta._to_pixels(pixel_size)

    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    return point

def reproject_stage_positions_onto_image2(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
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
            raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position.r is None:
            raise ValueError("Image metadata does not contain a valid stage position r coordinate. Cannot reproject stage position.")
        if pos.r is None:
            raise ValueError("Stage position r coordinate is None. Cannot reproject stage position.")
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

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    specimen_position = pos - specimen_offset

    return specimen_position

def _to_raw_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
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
        """
        Calculate the expected_y input from dy, dz stage movements and beam_type.
        This is the inverse of _y_corrected_stage_movement.

        Args:
            dy (float): actual y stage movement
            dz (float): actual z stage movement  
            beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

        Returns:
            float: expected_y input that would produce the given dy, dz movements
        """
        if image.metadata is None or image.metadata.hardware_geometry is None:
            raise ValueError("Image metadata or hardware geometry is not set. Cannot calculate inverse y corrected stage movement.")

        # Tescan stages have a different geometry (z below the tilt axis), so the inverse
        # is a separate derivation, not the compustage-aware path below. Match
        # case-insensitively: a live scope reports "TESCAN", config/saved images "Tescan".
        # (band-aid string check, tracked in FIB-300)
        system_info = image.metadata.system_info
        if system_info is not None and system_info.manufacturer.upper() == "TESCAN":
            return _inverse_y_corrected_stage_movement_tescan(image, dy=dy, dz=dz, beam_type=beam_type)

        geometry = image.metadata.hardware_geometry

        # all angles in radians
        sem_column_tilt = np.deg2rad(geometry.column_tilt)
        fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)

        stage_pretilt = np.deg2rad(geometry.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(geometry.rotation_reference) % (2 * np.pi)
        stage_rotation_flat_to_ion = np.deg2rad(geometry.rotation_180) % (2 * np.pi)

        # current stage position
        current_stage_position = image.metadata.stage_position
        stage_rotation = current_stage_position.r % (2 * np.pi) if current_stage_position.r is not None else 0.0
        stage_tilt = current_stage_position.t if current_stage_position.t is not None else 0.0

        # Handle compustage case. This mirrors FibsemMicroscope._view_corrected_stage_movement:
        # the forward flips expected_y once for compustage, then a second time at the FIB
        # orientation, so the two cancel there. Determined from metadata rather than a live
        # microscope; for the compustage the rotation is always 0, so the orientation is
        # fixed by the tilt alone (FIB sits at column_tilt - pretilt - 180, i.e. ~-128 deg,
        # and the FM pose at -180 deg).
        #
        # NOTE: this differs from the live path and from fm/reprojection.py, which both
        # also require the rotation to match the reference before calling a pose FIB.
        # Deliberately left alone here so FIB-481 is a pure restructuring with no
        # numerical difference; the divergence is FIB-500. It is latent -- a compustage
        # image cannot carry a non-zero rotation, so the two versions only disagree
        # about poses no acquisition can produce.
        compustage_sign = 1.0
        is_fib_orientation = False
        if geometry.is_compustage:
            fib_orientation_tilt = np.deg2rad(
                geometry.fib_column_tilt - geometry.shuttle_pre_tilt - 180
            )
            is_fib_orientation = bool(
                np.isclose(stage_tilt, fib_orientation_tilt, atol=0.1)
            )
            compustage_sign = 1.0 if is_fib_orientation else -1.0
            stage_tilt += np.pi

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
            PRETILT_SIGN = -1.0

        if is_fib_orientation:
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

        # perspective tilt adjustment
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

        # Reverse the calculations from the forward function:
        # Forward: y_move = y_sample_move * cos(corrected_pretilt_angle)
        # Forward: z_move = -y_sample_move * sin(corrected_pretilt_angle)
        # Therefore: y_sample_move can be calculated from either dy or dz

        # Calculate y_sample_move from dy and dz (should be consistent)
        cos_pretilt = np.cos(corrected_pretilt_angle)
        sin_pretilt = np.sin(corrected_pretilt_angle)
        
        if abs(cos_pretilt) > abs(sin_pretilt):
            # Use dy calculation when cos component is larger
            y_sample_move = dy / cos_pretilt
        else:
            # Use dz calculation when sin component is larger
            y_sample_move = -dz / sin_pretilt

        # Reverse: expected_y = y_sample_move * cos(stage_tilt + perspective_tilt_adjustment)
        expected_y = y_sample_move * np.cos(stage_tilt + perspective_tilt_adjustment)

        # Apply compustage correction if needed
        if geometry.is_compustage:
            expected_y *= compustage_sign

        return expected_y


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
        raise ValueError("Image metadata or hardware geometry is not set. Cannot calculate inverse y corrected stage movement.")

    return inverse_y_corrected_stage_movement_tescan_from_geometry(
        geometry=image.metadata.hardware_geometry,
        stage_position=image.metadata.stage_position,
        dy=dy,
        dz=dz,
        beam_type=beam_type,
    )


def y_corrected_stage_movement_tescan_from_geometry(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
    expected_y: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> Tuple[float, float]:
    """Tescan forward of the sample-plane move — the microscope-free form.

    The counterpart of :func:`inverse_y_corrected_stage_movement_tescan_from_geometry`,
    and the microscope-free form of :meth:`TescanMicroscope._y_corrected_stage_movement`.
    Added so a caller that has an image but no live instrument -- an overview canvas
    resolving a click on a saved image -- can run the same projection the stage would.

    Thermo's pair is parameterised by a *view tilt* and lives in
    :mod:`fibsem.transformations`; Tescan cannot join it, because its z-axis sits below
    the tilt axis and the decomposition uses the full chamber-frame sample inclination
    rather than the pre-tilt alone. Same reason ``TescanMicroscope`` overrides the
    method instead of passing a different view tilt.

    Args:
        geometry: fixed instrument geometry (column tilts, pre-tilt, rotation refs).
        stage_position: stage pose at acquisition (rotation r and tilt t, radians).
        expected_y: distance along the image y-axis, in metres.
        beam_type: beam perspective to correct for.

    Returns:
        (y_move, z_move) in the **chamber** frame, in metres -- before the stage-axis
        inversion ``stable_move`` applies (``y_stage = -y_chamber``, z unchanged). The
        inverse undoes that same inversion on its way in, so the two compose.
    """
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)
    stage_pretilt = np.deg2rad(geometry.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(geometry.rotation_reference) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(geometry.rotation_180) % (2 * np.pi)

    stage_rotation = stage_position.r % (2 * np.pi) if stage_position.r is not None else 0.0
    stage_tilt = stage_position.t if stage_position.t is not None else 0.0

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)
    sample_inclination = stage_tilt - corrected_pretilt_angle

    beam_tilt = sem_column_tilt if beam_type is BeamType.ELECTRON else fib_column_tilt

    y_sample_move = expected_y / np.cos(sample_inclination - beam_tilt)

    # z is negated because Tescan +z increases downward -- see
    # TescanMicroscope._y_corrected_stage_movement, verified on hardware 2026-07-23.
    return (
        float(y_sample_move * np.cos(sample_inclination)),
        float(-y_sample_move * np.sin(sample_inclination)),
    )


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

    Tescan stages have the z-axis BELOW the tilt axis: the y/z translation axes are fixed in
    the chamber frame, so the inversion uses the full chamber-frame sample inclination (stage
    tilt + pre-tilt) rather than just the pre-tilt. See
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
    # undo the stage-axis inversion applied in stable_move (y_stage = -y_chamber)
    # TODO(hardware-verify): keep in sync with the x/y inversion in TescanMicroscope.stable_move.
    dy = -dy

    # all angles in radians
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)

    stage_pretilt = np.deg2rad(geometry.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(geometry.rotation_reference) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(geometry.rotation_180) % (2 * np.pi)

    # stage pose at acquisition
    stage_rotation = stage_position.r % (2 * np.pi) if stage_position.r is not None else 0.0
    stage_tilt = stage_position.t if stage_position.t is not None else 0.0

    PRETILT_SIGN = 1.0
    # pretilt angle depends on rotation
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

    # inclination of the sample plane relative to the chamber horizontal
    # TODO(hardware-verify): tilt sense and z direction, see
    # TescanMicroscope._y_corrected_stage_movement.
    sample_inclination = stage_tilt - corrected_pretilt_angle

    beam_tilt = sem_column_tilt if beam_type is BeamType.ELECTRON else fib_column_tilt

    # invert: forward is y = d*cos(incl), z = -d*sin(incl);
    # recover d from the larger component for numerical stability. The sin branch must
    # negate dz to match the forward's negated z.
    cos_incl = np.cos(sample_inclination)
    sin_incl = np.sin(sample_inclination)
    if abs(cos_incl) > abs(sin_incl):
        y_sample_move = dy / cos_incl
    else:
        y_sample_move = -dz / sin_incl

    # re-project the sample-plane distance into the image plane
    expected_y = y_sample_move * np.cos(sample_inclination - beam_tilt)

    return expected_y
