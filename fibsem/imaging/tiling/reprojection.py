"""Reprojection: mapping stage positions onto an acquired image, and back.

Pure maths over image metadata -- no microscope. Note the asymmetry with
`tiled.convert_image_coord_to_stage_position`, which goes the other way and *does*
need a live microscope, because it projects through `project_stable_move` rather than
through the baked-in inverse here.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from fibsem.conversions import is_inside_image_bounds
from fibsem.structures import (
    LEGACY_ROTATION_CENTRE,
    BeamType,
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
            pos = _transform_position(pos, _rotation_centre(image))

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
            pos = _transform_position(pos, _rotation_centre(image))

        pt = calculate_reprojected_stage_position2(image, pos)
        pt.name = pos.name

        if bound and not is_inside_image_bounds((pt.y, pt.x), image.data.shape):
            continue

        points.append(pt)

    return points


# The specimen offset of the instrument LEGACY_ROTATION_CENTRE was calibrated on. Only
# the specimen/raw helpers below still read it; the half turn uses the centre.
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


def _rotation_centre(image: FibsemImage) -> Tuple[float, float]:
    """The half-turn centre an image was acquired under; the legacy one if unrecorded."""
    geometry = getattr(image.metadata, "hardware_geometry", None)
    if geometry is None:
        return LEGACY_ROTATION_CENTRE
    return geometry.rotation_centre


def _transform_position(
    pos: FibsemStagePosition,
    rotation_centre: Tuple[float, float] = LEGACY_ROTATION_CENTRE,
) -> FibsemStagePosition:
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam).

    A half turn reflects x and y through the rotation centre: p -> 2c - p. It is its
    own inverse, so it serves both directions. z, r and t are carried unchanged.

    Args:
        pos: The position flat to the beam.
        rotation_centre: where the half turn is centred, raw (x, y) in metres. The
            default is the one this function always used (LEGACY_ROTATION_CENTRE);
            callers holding an image pass the centre it was acquired under.
    Returns:
        The position flat to the other beam."""

    cx, cy = rotation_centre
    transformed_position = deepcopy(pos)
    transformed_position.x = 2 * cx - pos.x
    transformed_position.y = 2 * cy - pos.y

    # Debug, not info: this runs for every position drawn from the other side of the
    # stage -- three times per aligned image per redraw -- and at info it buried the
    # rest of the log (366 lines in five minutes of aligning an image).
    logging.debug(f"Initial position {pos} was transformed to {transformed_position}")

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
