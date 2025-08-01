from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, overload

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.collections import PatchCollection
from skimage.transform import resize

from fibsem.utils import format_value
from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemRectangleSettings,
    FibsemBitmapSettings,
    FibsemPatternSettings,
    Point,
)
from .utils import (
    create_pattern_mask,
    get_pattern_bounding_box,
    get_patterns_bounding_box,
)

COLOURS = [
    "yellow",
    "cyan",
    "magenta",
    "lime",
    "orange",
    "hotpink",
    "green",
    "blue",
    "red",
    "purple",
]


PROPERTIES = {
    "line_width": 1,
    "opacity": 0.3,
    "crosshair_size": 20,
    "crosshair_colour": "yellow",
    "rotation_point": "center",
}

OVERLAP_PROPERTIES = {
    "line_width": 0.5,
    "edge_color": "red",
    "face_color": "red",
    "alpha": 0.6,
    "line_style": "--",
}


def _rect_pattern_to_image_pixels(
    pattern: Union[FibsemRectangleSettings, FibsemBitmapSettings], pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Convert rectangle pattern to image pixel coordinates.
    Args:
        pattern: FibsemRectangleSettings: Rectangle pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[float, float, float, float]: Parameters (center_x, center_y, width, height) in image pixel coordinates.
    """
    # get pattern parameters
    width = pattern.width
    height = pattern.height
    mx, my = pattern.centre_x, pattern.centre_y

    # position in metres from image centre
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    # convert parameters to pixels
    width = width / pixel_size
    height = height / pixel_size

    return px, py, width, height

def _circle_pattern_to_image_pixels(
    pattern: FibsemCircleSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float, float, float]:
    """Convert circle pattern to image pixel coordinates.
    Args:
        pattern: FibsemCircleSettings: Circle pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[float, float, float, float, float, float]: Parameters (center_x, center_y, radius, inner_radius, start_angle, end_angle) in image pixel coordinates.
    """
    # get pattern parameters
    radius = pattern.radius
    thickness = pattern.thickness
    mx, my = pattern.centre_x, pattern.centre_y
    start_angle = pattern.start_angle
    end_angle = pattern.end_angle

    # position in metres from image centre
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    # convert parameters to pixels
    radius_px = radius / pixel_size
    inner_radius_px = max(0, (radius - thickness) / pixel_size) if thickness > 0 else 0

    return px, py, radius_px, inner_radius_px, start_angle, end_angle

def _line_pattern_to_image_pixels(
    pattern: FibsemLineSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Convert line pattern to image pixel coordinates.
    Args:
        pattern: FibsemLineSettings: Line pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[int, int, int, int]: Parameters (start_x, start_y, end_x, end_y) in image pixel coordinates.
    """
    # get pattern parameters
    start_x, start_y = pattern.start_x, pattern.start_y
    end_x, end_y = pattern.end_x, pattern.end_y

    # position in metres from image centre
    start_px, start_py = start_x / pixel_size, start_y / pixel_size
    end_px, end_py = end_x / pixel_size, end_y / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] / 2, image_shape[1] / 2
    start_pixel_x = cx + start_px
    start_pixel_y = cy - start_py
    end_pixel_x = cx + end_px
    end_pixel_y = cy - end_py

    return start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y


def _add_rectangle_mpl(
    shape: FibsemRectangleSettings,
    image: FibsemImage,
    colour: str,
    ax: plt.Axes,
    label: str | None = None,
    zorder: int | None = None,
) -> mpatches.Patch | None:
    """Create a rectangle patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    px, py, width, height = _rect_pattern_to_image_pixels(
        shape, pixel_size, image_shape
    )

    patch = mpatches.Rectangle(
        (px - width / 2, py - height / 2),
        width=width,
        height=height,
        angle=math.degrees(-shape.rotation),
        rotation_point=PROPERTIES["rotation_point"],
        linewidth=PROPERTIES["line_width"],
        edgecolor=colour,
        facecolor=colour,
        alpha=PROPERTIES["opacity"],
        zorder=zorder,
    )

    ax.add_patch(patch)

    if label is not None:
        return mpatches.Patch(
            color=colour, linewidth=PROPERTIES["line_width"], label=label
        )


def _add_circle_mpl(
    shape: FibsemCircleSettings,
    image: FibsemImage,
    colour: str,
    ax: plt.Axes,
    label: str | None = None,
    zorder: int | None = None,
) -> mpatches.Patch | None:
    """Create a circle patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    px, py, radius_px, inner_radius_px, start_angle, end_angle = (
        _circle_pattern_to_image_pixels(shape, pixel_size, image_shape)
    )

    if inner_radius_px > 0:
        # annulus/ring pattern
        patch = mpatches.Annulus(
            (px, py),
            r=inner_radius_px,
            width=radius_px - inner_radius_px,
            angle=math.degrees(-shape.rotation),
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
            zorder=zorder,
        )
    elif start_angle != 0 or end_angle != 360:
        # arc/wedge pattern
        patch = mpatches.Wedge(
            (px, py),
            r=radius_px,
            theta1=start_angle,
            theta2=end_angle,
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
            zorder=zorder,
        )
    else:
        # full circle pattern
        patch = mpatches.Circle(
            (px, py),
            radius=radius_px,
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
            zorder=zorder,
        )

    ax.add_patch(patch)

    if label is not None:
        return mpatches.Patch(
            color=colour, linewidth=PROPERTIES["line_width"], label=label
        )


def _add_line_mpl(
    shape: FibsemLineSettings,
    image: FibsemImage,
    colour: str,
    ax: plt.Axes,
    label: str | None = None,
    zorder: int | None = None,
) -> mpatches.Patch | None:
    """Create a line patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y = (
        _line_pattern_to_image_pixels(shape, pixel_size, image_shape)
    )

    patch = mpatches.FancyArrowPatch(
        (start_pixel_x, start_pixel_y),
        (end_pixel_x, end_pixel_y),
        linewidth=PROPERTIES["line_width"] * 2,
        edgecolor=colour,
        facecolor=colour,
        alpha=PROPERTIES["opacity"] + 0.2,
        arrowstyle="-",
        zorder=zorder,
    )

    ax.add_patch(patch)

    if label is not None:
        return mpatches.Patch(
            color=colour, linewidth=PROPERTIES["line_width"], label=label
        )


def _add_bitmap_mpl(
    shape: FibsemBitmapSettings,
    image: FibsemImage,
    colour: str,
    ax: plt.Axes,
    label: str | None = None,
    zorder: int | None = None,
) -> mpatches.Patch | None:
    """Draw a rectangle pattern on an image.
    Args:
        image: FibsemImage: Image to draw pattern on.
        pattern: BitmapPattern: Bitmap pattern to draw.
        colour: str: Colour of bitmap patches (blanked regions are black).
        ax: Axes to plot overlaps on.
        label: str | None: Label to apply to the output patch.
        zorder: int | None: Layer on which to plot.
    Returns:
        Patch to be used for the legend if a label is given, otherwise None
    """
    # common image properties
    pixel_size = image.metadata.pixel_size.x  # assume isotropic
    image_shape = image.data.shape

    # convert from microscope image (real-space) to image pixel-space
    px, py, width, height = _rect_pattern_to_image_pixels(
        shape, pixel_size, (image_shape[0], image_shape[1])
    )

    bitmap = shape.bitmap

    if bitmap is None:
        bitmap = np.zeros((1, 1, 2), dtype=float)

    if shape.flip_y:
        bitmap = np.flip(bitmap, axis=0)

    dwell_time_array = bitmap[:, :, 0].astype(np.float_)
    blanking_array = bitmap[:, :, 1] == 1

    # Ensure no rectangles will be subpixel (these are not displayed)
    target_shape = list(dwell_time_array.shape)
    resize_array = False
    if height < dwell_time_array.shape[0]:
        resize_array = True
        target_shape[0] = round(height)
    if width < dwell_time_array.shape[1]:
        resize_array = True
        target_shape[1] = round(width)

    if resize_array:
        dwell_time_array = resize(
            dwell_time_array,
            output_shape=target_shape,
            preserve_range=True,
            order=1,  # bi-linear interpolation
        )
        blanking_array = resize(
            blanking_array, output_shape=target_shape, preserve_range=True, order=0
        )

    cmap = ListedColormap(
        np.linspace((0, 0, 0, 0), to_rgba(colour, alpha=1), endpoint=True, num=256),
        name=f"{colour}_blend",
        N=256,
    )
    rgba = cmap(dwell_time_array)
    rgba[blanking_array] = (0, 0, 0, 1)

    # Apply opacity to the array (imshow alpha overrides these values).
    rgba[:, :, 3] *= PROPERTIES["opacity"]

    # Draw the edges
    edge_rectangle = mpatches.Rectangle(
        (
            px - width / 2,
            py - height / 2,
        ),  # bottom left corner
        width=width,
        height=height,
        angle=math.degrees(-shape.rotation),
        rotation_point=PROPERTIES["rotation_point"],
        linewidth=PROPERTIES["line_width"],
        edgecolor=colour,
        facecolor="none",
        alpha=PROPERTIES["opacity"],
        zorder=zorder,
    )
    ax.add_patch(edge_rectangle)

    ax.imshow(
        rgba,
        extent=(0, 1, 0, 1),
        transform=edge_rectangle.get_patch_transform()
        + edge_rectangle.get_data_transform(),
        zorder=edge_rectangle.get_zorder(),
    )

    if label:
        return mpatches.Patch(
            color=colour, linewidth=PROPERTIES["line_width"], label=label
        )


def _add_overlaps_mpl(
    milling_stages: List[FibsemMillingStage],
    image: FibsemImage,
    ax: plt.Axes,
    label: str | None = None,
    zorder: int | None = None,
) -> mpatches.Patch | None:
    """Detect overlapping regions between patterns and create patches to highlight them.

    Args:
        milling_stages: List of milling stages to check for overlaps.
        image: FibsemImage for coordinate conversion.
        ax: Axes to plot overlaps on.
        label: str | None: Label to apply to the output patch.
        zorder: int | None: Layer that the overlaps will be plotted.
    Returns:
        Patch to be used for the legend if a label is given, otherwise None
    """
    if len(milling_stages) < 2:
        return []
    
    overlap_patches = []
    
    # Create masks for each pattern
    pattern_masks = []
    for stage in milling_stages:
        stage_mask = create_pattern_mask(stage, image.data.shape, pixelsize=image.metadata.pixel_size.x, include_exclusions=False)
        pattern_masks.append(stage_mask)
    
    # Find overlaps between patterns
    for i in range(len(pattern_masks)):
        for j in range(i + 1, len(pattern_masks)):
            overlap = pattern_masks[i] & pattern_masks[j]
            if np.any(overlap):
                # Create contour patches for overlap regions
                try:
                    import skimage.measure
                    contours = skimage.measure.find_contours(overlap.astype(float), 0.5)
                    
                    for contour in contours:
                        if len(contour) > 3:  # Only create patches for significant overlaps
                            # Swap x,y coordinates for matplotlib (contour gives row,col)
                            contour_xy = contour[:, [1, 0]]
                            patch = mpatches.Polygon(
                                contour_xy,
                                closed=True,
                                linewidth=OVERLAP_PROPERTIES["line_width"],
                                edgecolor=OVERLAP_PROPERTIES["edge_color"],
                                facecolor=OVERLAP_PROPERTIES["face_color"],
                                alpha=OVERLAP_PROPERTIES["alpha"],
                                linestyle=OVERLAP_PROPERTIES["line_style"],
                            )
                            overlap_patches.append(patch)
                except ImportError:
                    # Fallback: create simple rectangle patches for overlap bounding boxes
                    overlap_coords = np.where(overlap)
                    if len(overlap_coords[0]) > 0:
                        y_min, y_max = overlap_coords[0].min(), overlap_coords[0].max()
                        x_min, x_max = overlap_coords[1].min(), overlap_coords[1].max()
                        
                        patch = mpatches.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            linewidth=OVERLAP_PROPERTIES["line_width"],
                            edgecolor=OVERLAP_PROPERTIES["edge_color"],
                            facecolor=OVERLAP_PROPERTIES["face_color"],
                            alpha=OVERLAP_PROPERTIES["alpha"],
                            linestyle=OVERLAP_PROPERTIES["line_style"],
                        )
                        overlap_patches.append(patch)
    if overlap_patches:
        ax.add_collection(
            PatchCollection(overlap_patches, match_original=True, zorder=zorder),
        )

        return mpatches.Patch(
            linewidth=OVERLAP_PROPERTIES["line_width"],
            edgecolor=OVERLAP_PROPERTIES["edge_color"],
            facecolor=OVERLAP_PROPERTIES["face_color"],
            linestyle=OVERLAP_PROPERTIES["line_style"],
            label=label,
        )


@overload
def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = ...,
    scalebar: bool = ...,
    title: str = ...,
    show_current: bool = ...,
    show_preset: bool = ...,
    show_depth: bool = ...,
    highlight_overlaps: bool = ...,
    ax: plt.Axes = ...,
) -> Tuple[None, plt.Axes]: ...


@overload
def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = ...,
    scalebar: bool = ...,
    title: str = ...,
    show_current: bool = ...,
    show_preset: bool = ...,
    show_depth: bool = ...,
    highlight_overlaps: bool = ...,
    ax: None = ...,
) -> Tuple[plt.Figure, plt.Axes]: ...


def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = True,
    scalebar: bool = True,
    title: str = "Milling Patterns",
    show_current: bool = False,
    show_preset: bool = False,
    show_depth: bool = False,
    highlight_overlaps: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """
    Draw milling patterns on an image. Supports patterns composed of multiple shape types.
    Args:
        image: FibsemImage: Image to draw patterns on.
        milling_stages: List[FibsemMillingStage]: Milling stages to draw.
        crosshair: bool: Draw crosshair at centre of image.
        scalebar: bool: Draw scalebar on image.
        title: str: Title for the plot.
        show_current: bool: Show milling current in legend.
        show_preset: bool: Show preset name in legend.
        show_depth: bool: Show pattern depth in microns in legend.
        highlight_overlaps: bool: Highlight overlapping pattern regions.
        ax: Optional[plt.Axes]: Axis that patterns will be plotted on. If no axis is given, a figure and axis will be created.
    Returns:
        Tuple[Optional[plt.Figure], plt.Axes]: Figure (if ax is None) and axis with pattern drawn.
    """
    fig: Optional[plt.Figure] = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image.data, cmap="gray")

    handles: list[mpatches.Patch] = []
    zorder = 0
    for i, stage in enumerate(milling_stages):
        colour = COLOURS[i % len(COLOURS)]
        pattern = stage.pattern

        extra_parts = []
        if show_current:
            extra_parts.append(format_value(stage.milling.milling_current, "A"))
        if show_preset:
            extra_parts.append(stage.milling.preset)
        if show_depth:
            # Get depth from pattern
            depth_m = getattr(pattern, 'depth', None)
            if depth_m is not None:
                depth_um = depth_m * 1e6  # Convert from meters to microns
                extra_parts.append(f"{depth_um:.1f}μm")
        
        extra = ", ".join(extra_parts)

        # Get all shapes from the pattern
        try:
            shapes = pattern.define()
        except Exception:
            logging.error("Failed to define pattern %s", pattern.name, exc_info=True)
            continue

        label = f"{stage.name}"
        if extra:
            label += f" ({extra})"

        # Process each shape individually
        for j, shape in enumerate(shapes):
            handle = None
            zorder += 1
            try:
                # Get the appropriate drawing function based on shape type
                if isinstance(shape, FibsemRectangleSettings):
                    handle = _add_rectangle_mpl(
                        shape,
                        image,
                        colour,
                        ax=ax,
                        label=label if j == 0 else None,
                        zorder=zorder,
                    )
                elif isinstance(shape, FibsemCircleSettings):
                    handle = _add_circle_mpl(
                        shape,
                        image,
                        colour,
                        ax=ax,
                        label=label if j == 0 else None,
                        zorder=zorder,
                    )
                elif isinstance(shape, FibsemLineSettings):
                    handle = _add_line_mpl(
                        shape,
                        image,
                        colour,
                        ax=ax,
                        label=label if j == 0 else None,
                        zorder=zorder,
                    )
                elif isinstance(shape, FibsemBitmapSettings):
                    handle = _add_bitmap_mpl(
                        shape,
                        image,
                        colour,
                        ax=ax,
                        label=label if j == 0 else None,
                        zorder=zorder,
                    )
                else:
                    logging.info(
                        "Unsupported shape type %s, skipping", str(type(shape))
                    )
                    continue
                if handle is not None:
                    handles.append(handle)
            except Exception:
                logging.warning(
                    "Failed to create patch for shape %s",
                    str(type(shape)),
                    exc_info=True,
                )
                continue

    # Detect and highlight overlaps if requested
    if highlight_overlaps:
        handle = _add_overlaps_mpl(
            milling_stages,
            image,
            ax=ax,
            label="Overlaps",
            zorder=zorder + 1,
        )
        if handle is not None:
            handles.append(handle)

    ax.legend(handles=handles)

    # set axis limits
    ax.set_xlim(0, image.data.shape[1])
    ax.set_ylim(image.data.shape[0], 0)

    # draw crosshair at centre of image
    if crosshair:
        cy, cx = image.data.shape[0] // 2, image.data.shape[1] // 2
        ax.plot(cx, cy, "y+", markersize=PROPERTIES["crosshair_size"])

    # draw scalebar
    if scalebar:
        try:
            # optional dependency, best effort
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )

            plt.gca().add_artist(scalebar)
        except ImportError:
            logging.debug("Scalebar not available, skipping")

    # set title
    ax.set_title(title)

    return fig, ax

# Plotting utilities for drawing pattern as numpy arrays

@dataclass
class DrawnPattern:
    pattern: np.ndarray
    position: Point
    is_exclusion: bool

def _create_annulus_shape(width, height, inner_radius, outer_radius):
    # Create a grid of coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    # Generate the donut shape
    donut = np.logical_and(distance <= outer_radius, distance >= inner_radius).astype(int)
    return donut

def draw_annulus_shape(pattern_settings: FibsemCircleSettings, image_shape: Tuple[int, int], pixelsize: float) -> DrawnPattern:
    """Convert an annulus pattern to a np array. Note: annulus can only be plotted as image
    Args:
        pattern_settings: FibsemCircleSettings: Annulus pattern settings.
        image_shape: Tuple[int, int]: Shape of the image (height, width).
        pixelsize: float: Pixel size in meters.
    Returns:
        DrawnPattern: Annulus shape in image.
    """
    
    # image parameters (centre, pixel size)
    icy, icx = image_shape[0] // 2, image_shape[1] // 2
    pixelsize_x, pixelsize_y = pixelsize, pixelsize

    # pattern parameters
    radius = pattern_settings.radius
    thickness = pattern_settings.thickness
    center_x = pattern_settings.centre_x
    center_y = pattern_settings.centre_y

    radius_px = radius / pixelsize_x # isotropic
    shape = int(2 * radius_px)
    inner_radius_ratio = 0 # full circle
    if not np.isclose(thickness, 0):
        inner_radius_ratio = (radius - thickness)/radius
   
    annulus_shape = _create_annulus_shape(width=shape, height=shape, 
                                          inner_radius=inner_radius_ratio, 
                                          outer_radius=1)

    # get pattern centre in image coordinates
    pattern_centre_x = int(icx + center_x / pixelsize_x)
    pattern_centre_y = int(icy - center_y / pixelsize_y)

    pos = Point(x=pattern_centre_x, y=pattern_centre_y)

    return DrawnPattern(pattern=annulus_shape, position=pos, is_exclusion=pattern_settings.is_exclusion)

def draw_rectangle_shape(pattern_settings: FibsemRectangleSettings, image_shape: Tuple[int, int], pixelsize: float) -> DrawnPattern:
    """Convert a rectangle pattern to a np array with rotation support.
    Args:
        pattern_settings: FibsemRectangleSettings: Rectangle pattern settings.
        image_shape: Tuple[int, int]: Shape of the image (height, width).
        pixelsize: float: Pixel size in meters.
    Returns:
        DrawnPattern: Rectangle shape in image with rotation applied.
    """
    from scipy.ndimage import rotate

    # image parameters (centre, pixel size)
    icy, icx = image_shape[0] // 2, image_shape[1] // 2
    pixelsize_x, pixelsize_y = pixelsize, pixelsize

    # pattern parameters
    width = pattern_settings.width
    height = pattern_settings.height
    centre_x = pattern_settings.centre_x
    centre_y = pattern_settings.centre_y
    rotation = pattern_settings.rotation

    # pattern to pixel coords
    w = int(width / pixelsize_x)
    h = int(height / pixelsize_y)
    cx = int(icx + (centre_x / pixelsize_x))  # Fix: use pixelsize_x for x coordinate
    cy = int(icy - (centre_y / pixelsize_y))

    # Create base rectangle shape
    shape = np.ones((h, w), dtype=float)

    # Apply rotation if specified
    if not np.isclose(rotation, 0):
        # Convert radians to degrees for scipy
        rotation_degrees = np.degrees(rotation)
        
        # Rotate the shape, reshape=True to accommodate the rotated rectangle
        shape = rotate(shape, rotation_degrees, reshape=True, order=1, prefilter=False)
        
        # Convert back to binary (rotation may introduce intermediate values)
        shape = (shape > 0.5).astype(float)

    # get pattern centre in image coordinates
    pos = Point(x=cx, y=cy)

    return DrawnPattern(pattern=shape, position=pos, is_exclusion=pattern_settings.is_exclusion)


def draw_bitmap_shape(
    pattern_settings: FibsemBitmapSettings,
    image_shape: Tuple[int, int],
    pixelsize: float,
) -> DrawnPattern:
    from PIL import Image

    # image parameters (centre, pixel size)
    icy, icx = image_shape[0] // 2, image_shape[1] // 2
    pixelsize_x, pixelsize_y = pixelsize, pixelsize

    # pattern parameters
    width = pattern_settings.width
    height = pattern_settings.height
    centre_x = pattern_settings.centre_x
    centre_y = pattern_settings.centre_y
    rotation = pattern_settings.rotation

    # pattern to pixel coords
    w = int(width / pixelsize_x)
    h = int(height / pixelsize_y)
    cx = int(icx + (centre_x / pixelsize_x))  # Fix: use pixelsize_x for x coordinate
    cy = int(icy - (centre_y / pixelsize_y))

    bitmap = pattern_settings.bitmap

    if bitmap is None:
        bitmap = np.zeros((1, 1, 2), dtype=float)

    image_bmp = Image.fromarray(bitmap[:, :, 0].squeeze(-1).astype(float))

    if pattern_settings.flip_y:
        image_bmp = image_bmp.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    image_resized = image_bmp.resize((w, h))

    if not np.isclose(rotation, 0):
        image_resized = image_resized.rotate(pattern_settings.rotation, expand=True)

    # Create base rectangle shape
    shape = np.asarray(image_resized, dtype=np.float_)

    # get pattern centre in image coordinates
    pos = Point(x=cx, y=cy)

    return DrawnPattern(
        pattern=shape, position=pos, is_exclusion=pattern_settings.is_exclusion
    )


def draw_line_shape(
    pattern_settings: FibsemLineSettings,
    image_shape: Tuple[int, int],
    pixelsize: float,
) -> DrawnPattern:
    from skimage.draw import line_aa

    start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y = (
        _line_pattern_to_image_pixels(
            pattern_settings, pixelsize, (image_shape[0], image_shape[1])
        )
    )

    array_max = (
        int(round(end_pixel_y - start_pixel_y)),
        int(round(end_pixel_x - start_pixel_x)),
    )

    shape = np.zeros((array_max[0] + 1, array_max[1] + 1), dtype=float)

    rr, cc, val = line_aa(0, 0, array_max[0], array_max[1])
    shape[rr, cc] = val

    cx = int(round((start_pixel_x + end_pixel_x) / 2))
    cy = int(round((start_pixel_y + end_pixel_y) / 2))

    pos = Point(x=cx, y=cy)

    return DrawnPattern(pattern=shape, position=pos, is_exclusion=False)


def draw_pattern_shape(ps: FibsemPatternSettings, image_shape: Tuple[int, int], pixelsize: float) -> DrawnPattern:
    if isinstance(ps, FibsemCircleSettings):
        return draw_annulus_shape(ps, image_shape, pixelsize)
    elif isinstance(ps, FibsemRectangleSettings):
        return draw_rectangle_shape(ps, image_shape, pixelsize)
    elif isinstance(ps, FibsemLineSettings):
        return draw_line_shape(ps, image_shape, pixelsize)
    elif isinstance(ps, FibsemBitmapSettings):
        return draw_bitmap_shape(ps, image_shape, pixelsize)
    else:
        raise ValueError(f"Unsupported shape type {type(ps)}")

def draw_pattern_in_image(image: np.ndarray, 
                          drawn_pattern: DrawnPattern) -> np.ndarray:

    pattern = drawn_pattern.pattern
    pos = drawn_pattern.position
    
    # place the annulus shape in the image
    w = pattern.shape[1] // 2
    h = pattern.shape[0] // 2

    # fill the annulus shape in the image
    xmin, xmax = pos.x - w, pos.x + w
    ymin, ymax = pos.y - h, pos.y + h
    zero_image = np.zeros_like(image)
    zero_image[ymin:ymax, xmin:xmax] = pattern[:2*h, :2*w].astype(bool)

    # if the pattern is an exclusion, set the image to zero
    if drawn_pattern.is_exclusion:
        image[zero_image == 1] = 0
    else:
        # add the annulus shape to the image, clip to 1
        image = np.clip(image+zero_image, 0, 1)

    return image

def compose_pattern_image(image: np.ndarray, drawn_patterns: List[DrawnPattern]) -> np.ndarray:
    """Create an image with annulus shapes."""
    # create an empty image
    pattern_image = np.zeros_like(image)

    # sort drawn_patterns so that exclusions are drawn last
    drawn_patterns = sorted(drawn_patterns, key=lambda x: x.is_exclusion)

    # add each pattern shape to the image
    for dp in drawn_patterns:
        pattern_image = draw_pattern_in_image(pattern_image, dp)

    return pattern_image


def simple_example(stages: List[FibsemMillingStage], image: FibsemImage) -> plt.Figure:
    """Simple demonstration of masks and bounding boxes."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Show masks
    ax1.imshow(image.data, cmap='gray', alpha=0.7)
    ax1.set_title("Pattern Masks")
    
    for i, stage in enumerate(stages):
        # Create mask
        mask = create_pattern_mask(stage, image.data.shape, pixelsize=image.metadata.pixel_size.x)
        
        # Show mask as colored overlay
        masked = np.ma.masked_where(~mask, mask)
        ax1.imshow(masked, alpha=0.6, cmap='gray', vmin=0, vmax=10)
        
        print(f"{stage.name}: {np.sum(mask)} pixels")
    
    # Plot 2: Show bounding boxes
    ax2.imshow(image.data, cmap='gray')
    ax2.set_title("Bounding Boxes")
    
    for i, stage in enumerate(stages):
        # Get bounding box
        x_min, y_min, x_max, y_max = get_pattern_bounding_box(stage, image)
        
        if (x_min, y_min, x_max, y_max) != (0, 0, 0, 0):
            # Draw bounding box using COLOURS from plotting.py
            bbox = mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=COLOURS[i % len(COLOURS)], facecolor='none'
            )
            ax2.add_patch(bbox)
            ax2.text(x_min, y_min-5, stage.name, color=COLOURS[i % len(COLOURS)], fontsize=9)
    
    # Add combined bounding box
    x_min, y_min, x_max, y_max = get_patterns_bounding_box(stages, image)
    if (x_min, y_min, x_max, y_max) != (0, 0, 0, 0):
        combined_bbox = mpatches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax2.add_patch(combined_bbox)
        ax2.text(x_min, y_max+10, 'Combined', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig