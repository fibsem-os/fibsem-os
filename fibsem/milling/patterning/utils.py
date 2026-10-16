"""Utility functions for pattern analysis and coordinate conversion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from fibsem.structures import FibsemRectangle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fibsem.milling.patterning.base import BasePattern
    from fibsem.structures import FibsemImage


def create_pattern_mask(
    pattern: "BasePattern",
    image_shape: Tuple[int, int],
    pixelsize: float,
    include_exclusions: bool = False,
) -> np.ndarray:
    """Create a binary mask for a single milling pattern.

    Args:
        pattern: BasePattern to create mask for.
        image_shape: Shape of the image as (height, width).
        pixelsize: Pixel size in meters.
        include_exclusions: Whether to include exclusion patterns in the mask.

    Returns:
        Binary mask as numpy array with same shape as image.
    """
    pattern_mask = np.zeros(image_shape, dtype=bool)

    try:
        shapes = pattern.define()

        # Sort shapes so exclusions are processed last (exclusions take precedence)
        shapes.sort(key=lambda s: hasattr(s, "is_exclusion") and s.is_exclusion)

        for shape in shapes:
            # Skip exclusion patterns unless explicitly requested
            if (
                hasattr(shape, "is_exclusion")
                and shape.is_exclusion
                and not include_exclusions
            ):
                continue

            # Import here to avoid circular import
            from .plotting import draw_pattern_shape

            drawn_pattern = draw_pattern_shape(shape, image_shape, pixelsize)
            shape_mask = np.zeros(image_shape, dtype=bool)

            # Place the pattern in the image mask
            w = drawn_pattern.pattern.shape[1] // 2
            h = drawn_pattern.pattern.shape[0] // 2
            pos = drawn_pattern.position

            xmin, xmax = max(0, pos.x - w), min(image_shape[1], pos.x + w)
            ymin, ymax = max(0, pos.y - h), min(image_shape[0], pos.y + h)

            if xmax > xmin and ymax > ymin:
                pattern_h = min(2 * h, ymax - ymin)
                pattern_w = min(2 * w, xmax - xmin)
                shape_mask[ymin : ymin + pattern_h, xmin : xmin + pattern_w] = (
                    drawn_pattern.pattern[:pattern_h, :pattern_w].astype(bool)
                )

            # If this is an exclusion shape, zero out the region instead of adding it
            if hasattr(shape, "is_exclusion") and shape.is_exclusion:
                pattern_mask &= ~shape_mask
            else:
                pattern_mask |= shape_mask

    except Exception as e:
        logging.debug(f"Failed to create mask for pattern {pattern.name}: {e}")

    return pattern_mask


def get_pattern_bounding_box(
    pattern: "BasePattern", image: FibsemImage, expand_percent: float = 0.0
) -> Tuple[int, int, int, int]:
    """Get the bounding box of a pattern in image pixel coordinates.

    Args:
        pattern: BasePattern to get bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (e.g., 10.0 for 10%).

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in image pixel coordinates.
        Returns (0, 0, 0, 0) if pattern has no area or fails to process.
    """
    try:
        # Create mask for the pattern
        mask = create_pattern_mask(
            pattern,
            image.data.shape,
            pixelsize=image.metadata.pixel_size.x,
            include_exclusions=False,
        )

        # Find coordinates where mask is True
        coords = np.nonzero(mask)

        if len(coords[0]) == 0:
            # No pattern area found
            return (0, 0, 0, 0)

        # Get bounding box coordinates
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Expand bounding box if requested
        if expand_percent > 0:
            width = x_max - x_min
            height = y_max - y_min

            # Calculate expansion in pixels
            expand_x = int(width * expand_percent / 100)
            expand_y = int(height * expand_percent / 100)

            # Apply expansion while staying within image bounds
            image_height, image_width = image.data.shape
            x_min = max(0, x_min - expand_x)
            y_min = max(0, y_min - expand_y)
            x_max = min(image_width - 1, x_max + expand_x)
            y_max = min(image_height - 1, y_max + expand_y)

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        logging.debug(
            f"Failed to calculate bounding box for pattern {pattern.name}: {e}"
        )
        return (0, 0, 0, 0)


def get_patterns_bounding_box(
    patterns: List["BasePattern"], image: FibsemImage, expand_percent: float = 0.0
) -> Tuple[int, int, int, int]:
    """Get the combined bounding box of multiple patterns in image pixel coordinates.

    Args:
        patterns: List of BasePattern to get combined bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (e.g., 10.0 for 10%).

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in image pixel coordinates.
        Returns (0, 0, 0, 0) if no patterns have area.
    """
    if not patterns:
        return (0, 0, 0, 0)

    # Get bounding boxes for all patterns
    all_boxes = [
        get_pattern_bounding_box(pattern, image, expand_percent) for pattern in patterns
    ]

    # Filter out empty boxes
    valid_boxes = [box for box in all_boxes if box != (0, 0, 0, 0)]

    if not valid_boxes:
        return (0, 0, 0, 0)

    # Find overall bounding box
    x_mins, y_mins, x_maxs, y_maxs = zip(*valid_boxes)

    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


def bbox_to_normalized_coords(
    bbox: Tuple[int, int, int, int], image: FibsemImage
) -> Tuple[float, float, float, float]:
    """Convert bounding box from pixel coordinates to normalized coordinates (0-1).

    Args:
        bbox: Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates.
        image: FibsemImage to get dimensions from.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates (0.0-1.0).
        Returns (0.0, 0.0, 0.0, 0.0) if bbox is empty.
    """
    x_min, y_min, x_max, y_max = bbox

    # Check for empty bounding box
    if bbox == (0, 0, 0, 0):
        return (0.0, 0.0, 0.0, 0.0)

    # Get image dimensions
    image_height, image_width = image.data.shape

    # Normalize coordinates (0=0.0, shape=1.0)
    x_min_norm = x_min / image_width
    y_min_norm = y_min / image_height
    x_max_norm = x_max / image_width
    y_max_norm = y_max / image_height

    return (x_min_norm, y_min_norm, x_max_norm, y_max_norm)


def normalized_coords_to_bbox(
    norm_bbox: Tuple[float, float, float, float], image: FibsemImage
) -> Tuple[int, int, int, int]:
    """Convert normalized coordinates (0-1) to pixel bounding box coordinates.

    Args:
        norm_bbox: Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates (0.0-1.0).
        image: FibsemImage to get dimensions from.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates.
        Returns (0, 0, 0, 0) if norm_bbox is empty.
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = norm_bbox

    # Check for empty normalized bounding box
    if norm_bbox == (0.0, 0.0, 0.0, 0.0):
        return (0, 0, 0, 0)

    # Get image dimensions
    image_height, image_width = image.data.shape

    # Convert to pixel coordinates
    x_min = int(x_min_norm * image_width)
    y_min = int(y_min_norm * image_height)
    x_max = int(x_max_norm * image_width)
    y_max = int(y_max_norm * image_height)

    # Ensure coordinates are within bounds
    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_max = max(0, min(y_max, image_height - 1))

    return (x_min, y_min, x_max, y_max)


def get_pattern_reduced_area(
    pattern: "BasePattern", image: FibsemImage, expand_percent: int = 20
) -> FibsemRectangle:
    """Get the bounding box of the pattern in the image, expanded by a percentage.

    Args:
        pattern: BasePattern to get bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (default 20%).

    Returns:
        FibsemRectangle with normalized coordinates (0.0-1.0).
    """
    bbox = get_pattern_bounding_box(
        pattern=pattern, image=image, expand_percent=expand_percent
    )
    xmin, ymin, xmax, ymax = bbox_to_normalized_coords(bbox, image)
    return FibsemRectangle(left=xmin, top=ymin, width=xmax - xmin, height=ymax - ymin)


def bitmap_image_to_points(bitmap_image: NDArray[np.uint8]) -> NDArray[Any]:
    """Convert from a bitmap image to bitmap points.

    Bitmap points have the advantage of higher bit depth and more direct scaling.

    - Bitmap image:
        - 3-channel (RGB) images of type `np.uint8` in the shape (y, x, c)
        - Channel 0 (R) is not used
        - Channel 1 (G) is a flag (0 blanks the point, 1 means no flags)
        - Channel 2 (B) is the dwell time multiplier (255 translates to 1x the pattern value)

    - Bitmap points:
        - 2-channel numpy arrays of type `object` in the shape (y, x, c)
        - Channel 0 is the dwell time multiplier (now a float with the range of 0-1)
        - Channel 1 is a flag (0 means no flags, 1 blanks the point)
        - There is no Channel 2 (this was channel 0)
    """
    points_array = np.empty((*bitmap_image.shape[:2], 2), dtype=object)
    points_array[:, :, 0] = np.interp(bitmap_image[:, :, 2], (0, 255), (0, 1))
    points_array[:, :, 1] = 1 - bitmap_image[:, :, 1]
    return points_array


def pattern_extent(pattern) -> Optional[Tuple[float, float, float, float]]:
    """``(x_min, x_max, y_min, y_max)`` a pattern's settings reach, in metres
    from the field's centre, from its numbers alone: no image needed.
    None for a kind this does not know."""
    from fibsem.structures import (
        FibsemBitmapSettings,
        FibsemCircleSettings,
        FibsemLineSettings,
        FibsemPolygonSettings,
        FibsemRectangleSettings,
    )

    if isinstance(pattern, (FibsemRectangleSettings, FibsemBitmapSettings)):
        cos, sin = (
            abs(np.cos(pattern.rotation or 0.0)),
            abs(np.sin(pattern.rotation or 0.0)),
        )
        half_x = (pattern.width * cos + pattern.height * sin) / 2
        half_y = (pattern.width * sin + pattern.height * cos) / 2
        x, y = pattern.centre_x, pattern.centre_y
        return (x - half_x, x + half_x, y - half_y, y + half_y)
    if isinstance(pattern, FibsemCircleSettings):
        r = pattern.radius + (pattern.thickness or 0.0) / 2
        x, y = pattern.centre_x, pattern.centre_y
        return (x - r, x + r, y - r, y + r)
    if isinstance(pattern, FibsemLineSettings):
        xs, ys = (pattern.start_x, pattern.end_x), (pattern.start_y, pattern.end_y)
        return (min(xs), max(xs), min(ys), max(ys))
    if isinstance(pattern, FibsemPolygonSettings):
        vertices = np.asarray(pattern.vertices, dtype=float).reshape(-1, 2)
        if not len(vertices):
            return None
        return (
            float(vertices[:, 0].min()),
            float(vertices[:, 0].max()),
            float(vertices[:, 1].min()),
            float(vertices[:, 1].max()),
        )
    return None


def pattern_outline(pattern) -> List[List[Tuple[float, float]]]:
    """The lines a pattern's settings trace, in metres from the field's centre
    (y up): a closed ring of points for an area, two points for a line. For a
    kind this does not know, the box `pattern_extent` gives; none at all, [].
    For drawing it on an image without the milling view (FIB-1411)."""
    from fibsem.structures import (
        FibsemBitmapSettings,
        FibsemCircleSettings,
        FibsemLineSettings,
        FibsemPolygonSettings,
        FibsemRectangleSettings,
    )

    if isinstance(pattern, (FibsemRectangleSettings, FibsemBitmapSettings)):
        cos, sin = np.cos(pattern.rotation or 0.0), np.sin(pattern.rotation or 0.0)
        half_w, half_h = pattern.width / 2, pattern.height / 2
        return [
            [
                (
                    pattern.centre_x + x * cos - y * sin,
                    pattern.centre_y + x * sin + y * cos,
                )
                for x, y in (
                    (-half_w, -half_h),
                    (half_w, -half_h),
                    (half_w, half_h),
                    (-half_w, half_h),
                )
            ]
        ]
    if isinstance(pattern, FibsemCircleSettings):
        rings = [pattern.radius]
        if pattern.thickness and pattern.thickness < pattern.radius:
            rings.append(pattern.radius - pattern.thickness)
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return [
            [
                (
                    pattern.centre_x + r * float(np.cos(a)),
                    pattern.centre_y + r * float(np.sin(a)),
                )
                for a in angles
            ]
            for r in rings
        ]
    if isinstance(pattern, FibsemLineSettings):
        return [[(pattern.start_x, pattern.start_y), (pattern.end_x, pattern.end_y)]]
    if isinstance(pattern, FibsemPolygonSettings):
        vertices = np.asarray(pattern.vertices, dtype=float).reshape(-1, 2)
        return [[(float(x), float(y)) for x, y in vertices]] if len(vertices) else []
    extent = pattern_extent(pattern)
    if extent is None:
        return []
    x_min, x_max, y_min, y_max = extent
    return [[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]]


def outside_field(pattern, width: float, height: float) -> bool:
    """Whether *pattern* reaches past a *width* × *height* field (metres),
    centred on the beam, which a real instrument refuses to mill (FIB-1411)."""
    extent = pattern_extent(pattern)
    if extent is None:
        return False
    x_min, x_max, y_min, y_max = extent
    return (
        x_min < -width / 2
        or x_max > width / 2
        or y_min < -height / 2
        or y_max > height / 2
    )


def stage_outside_field(stage, aspect: float = 2 / 3) -> list:
    """The patterns of milling *stage* that reach past its field: its milling
    hfw wide and *aspect* of that high (an image's height over width)."""
    hfw = stage.milling.hfw
    return [
        pattern
        for pattern in stage.define_patterns()
        if outside_field(pattern, hfw, hfw * aspect)
    ]
