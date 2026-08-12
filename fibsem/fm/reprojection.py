"""Projecting stage positions onto fluorescence images.

The fluorescence counterpart of :mod:`fibsem.imaging.tiling.reprojection`, and the
inverse of :meth:`FibsemMicroscope.project_fm_stable_move`: that one asks where a click
would send the stage, these ask where the stage already is on screen. Both run the same
projection, so a marker drawn here lands where clicking there would take you.

The projection itself is not fluorescence-specific and lives in
:mod:`fibsem.transformations`, taking the camera's view tilt as an argument. What is
fluorescence-specific, and stays here, is the camera image transform and the shape of a
fluorescence image's metadata.

Microscope-free by design. Everything the projection reads comes from the image's own
:class:`FibsemHardwareGeometry` and stage position, so a saved overview reprojects correctly
even when the stage has since moved or the display transform has since been flipped --
which is precisely when someone is looking at a saved overview. Reading any of it from
a live microscope would make the result depend on state the image does not describe.

Images written before the geometry was recorded carry ``geometry=None``. Those raise
rather than falling back to live state or to a default pose: a marker drawn in the wrong
place is indistinguishable on screen from one drawn in the right place, so guessing is
worse than refusing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from fibsem.fm.structures import FibsemHardwareGeometry
from fibsem.structures import FibsemStagePosition, Point
from fibsem.transformations import (
    inverse_view_corrected_dy,
    view_corrected_stage_movement,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.fm.structures import FluorescenceImage


def project_stage_position(
    position: FibsemStagePosition,
    base_position: FibsemStagePosition,
    pixel_size: float,
    image_shape: Tuple[int, int],
    geometry: FibsemHardwareGeometry,
) -> Point:
    """Where a stage position falls in a displayed FM image, in pixels.

    Args:
        position: the stage position to locate.
        base_position: the stage position the image was acquired at.
        pixel_size: image pixel size, in metres.
        image_shape: image shape as (height, width), in pixels.
        geometry: the geometry the image was captured under.

    Returns:
        Point: pixel coordinates in the displayed image. Deliberately not clamped --
        a position outside the field of view returns coordinates outside the image
        bounds, which is what lets a caller draw a tile grid larger than the image.
    """
    delta = position - base_position

    dy = inverse_view_corrected_dy(
        dy=delta.y,
        dz=delta.z,
        view_tilt=np.deg2rad(geometry.camera_tilt),
        geometry=geometry,
        stage_rotation=base_position.r or 0.0,
        stage_tilt=base_position.t or 0.0,
    )
    # The transform is restricted to flips, making it its own inverse, so the one
    # mapping serves both directions -- see `CameraImageTransform`.
    dx, dy = geometry.transform.apply_to_delta(delta.x, dy)

    # Image y grows downward while the projected displacement grows upward.
    return Point(
        x=image_shape[1] / 2 + dx / pixel_size,
        y=image_shape[0] / 2 - dy / pixel_size,
    )


def project_image_point(
    point: Point,
    base_position: FibsemStagePosition,
    pixel_size: float,
    image_shape: Tuple[int, int],
    geometry: FibsemHardwareGeometry,
) -> FibsemStagePosition:
    """The stage position under a pixel in a displayed FM image.

    The exact inverse of :func:`project_stage_position`, sharing its sign terms, so a
    click resolves to the position the marker was drawn from.

    Args:
        point: pixel coordinates in the displayed image.
        base_position: the stage position the image was acquired at.
        pixel_size: image pixel size, in metres.
        image_shape: image shape as (height, width), in pixels.
        geometry: the geometry the image was captured under.

    Returns:
        FibsemStagePosition: the absolute position under that pixel.
    """
    dx = (point.x - image_shape[1] / 2) * pixel_size
    dy = -(point.y - image_shape[0] / 2) * pixel_size

    dx, dy = geometry.transform.apply_to_delta(dx, dy)
    y_move, z_move = view_corrected_stage_movement(
        expected_y=dy,
        view_tilt=np.deg2rad(geometry.camera_tilt),
        geometry=geometry,
        stage_rotation=base_position.r or 0.0,
        stage_tilt=base_position.t or 0.0,
    )

    position = deepcopy(base_position)
    position.x += dx
    position.y += y_move
    position.z += z_move
    return position


def stage_position_from_fm_image(
    image: "FluorescenceImage",
    point: Point,
) -> FibsemStagePosition:
    """The stage position under a pixel of a fluorescence image.

    The inverse of :func:`reproject_stage_positions_onto_fm_image`, and the projection
    behind clicking a loaded overview. Reads the pose and transform the image records
    rather than the live instrument's: click a saved overview after the stage has moved
    and the answer must still describe that image, not the current view.

    Args:
        image: the image that was clicked.
        point: pixel coordinates within it.

    Returns:
        FibsemStagePosition: the absolute position under that pixel.

    Raises:
        ValueError: if the image does not record the geometry it was captured under,
            or has no stage position.
    """
    metadata = _require_projectable(image)
    height, width = image.data.shape[-2:]

    return project_image_point(
        point=point,
        base_position=metadata.stage_position,
        pixel_size=metadata.pixel_size_x,
        image_shape=(height, width),
        geometry=metadata.geometry,
    )


def _require_projectable(image: "FluorescenceImage"):
    """The metadata a projection needs, or a ValueError naming what is missing."""
    metadata = image.metadata
    if metadata is None:
        raise ValueError("Image has no metadata. Cannot project stage positions.")
    if metadata.geometry is None:
        raise ValueError(
            "Image metadata does not record the geometry it was captured under "
            "(acquired before this was recorded). Cannot project stage positions."
        )
    if metadata.stage_position is None:
        raise ValueError(
            "Image metadata does not contain a stage position. "
            "Cannot project stage positions."
        )
    return metadata


def reproject_stage_positions_onto_fm_image(
    image: "FluorescenceImage",
    positions: List[FibsemStagePosition],
    bound: bool = False,
) -> List[Point]:
    """Reproject stage positions onto a fluorescence image.

    The FM counterpart of
    :func:`fibsem.imaging.tiling.reprojection.reproject_stage_positions_onto_image2`.

    Args:
        image: the image to project onto.
        positions: the stage positions.
        bound: drop points that fall outside the image bounds.

    Returns:
        The positions in image pixel coordinates, carrying their names across.

    Raises:
        ValueError: if the image does not record the geometry it was captured under,
            or has no stage position. Both are absent on images written before this
            was recorded, and neither can be guessed without risking a marker that
            looks correct and is not.
    """
    metadata = _require_projectable(image)
    height, width = image.data.shape[-2:]

    points = []
    for position in positions:
        point = project_stage_position(
            position=position,
            base_position=metadata.stage_position,
            pixel_size=metadata.pixel_size_x,
            image_shape=(height, width),
            geometry=metadata.geometry,
        )
        point.name = position.name
        if bound and not (0 <= point.x < width and 0 <= point.y < height):
            continue
        points.append(point)

    return points
