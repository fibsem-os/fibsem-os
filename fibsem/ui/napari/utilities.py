import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import napari
import numpy as np

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # packaging is available in napari envs but fall back gracefully
    InvalidVersion = Version = None

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import (
    FibsemStagePosition,
    Point,
)
from fibsem.ui.napari.properties import (
    IMAGING_CROSSHAIR_LAYER_PROPERTIES,
    IMAGING_SCALEBAR_LAYER_PROPERTIES,
)

CROSSHAIR_LAYER_NAME = IMAGING_CROSSHAIR_LAYER_PROPERTIES["name"]
SCALEBAR_LAYER_NAME = IMAGING_SCALEBAR_LAYER_PROPERTIES["name"]

@dataclass
class NapariShapeOverlay:
    """Represents a shape overlay for napari with its properties."""
    shape: np.ndarray
    color: str
    label: str
    shape_type: str  # "rectangle", "ellipse", or "line"



def create_crosshair_shape(point: Point, size: int, scale: Optional[Tuple[float, float]] = None) -> List[np.ndarray]:
    """Create a crosshair shape at the specified point with given size and scale.
    
    Generates horizontal and vertical line data for creating a crosshair overlay
    in napari. The function converts stage coordinates to pixel coordinates using
    the provided scale factors.
    
    Args:
        point: Stage coordinates where to create the crosshair (x, y in meters)
        size: Half-length of crosshair arms in pixels (total crosshair span = 2 * size)
        scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion from meters to pixels (optional).
               If not provided, defaults to (1, 1)
        
    Returns:
        List of numpy arrays representing crosshair line segments:
        - [0]: Horizontal line array [[y, x_start], [y, x_end]]
        - [1]: Vertical line array [[y_start, x], [y_end, x]]
        
    Note:
        - Modifies the input point in-place by converting coordinates to pixel space
        - Uses napari coordinate convention (y, x) for line endpoints
        - Scale factors should match camera pixel size for accurate positioning
        
    Example:
        >>> point = Point(x=10e-6, y=5e-6)  # 10μm x, 5μm y
        >>> scale = (1e-6, 1e-6)  # 1μm per pixel
        >>> crosshair = create_crosshair_shape(point, 25, scale)
        >>> # Returns 2 line arrays for horizontal and vertical crosshair arms
    """
    if scale is None:
        scale = (1.0, 1.0)

    # Convert stage coordinates to pixel coordinates
    point.x /= scale[0]  
    point.y /= scale[1]  
    
    # Create horizontal line at point
    horizontal_line = np.array([
        [point.y, point.x - size],
        [point.y, point.x + size]
    ])
    
    # Create vertical line at point
    vertical_line = np.array([
        [point.y - size, point.x],
        [point.y + size, point.x]
    ])

    return [horizontal_line, vertical_line]


def create_rectangle_shape(
    center_point: Point, 
    width: float, 
    height: float, 
    scale: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Create a rectangle shape centered at the specified point.
    
    Generates rectangle corner coordinates for creating a bounding box overlay
    in napari. The function converts stage coordinates to pixel coordinates using
    the provided scale factors.
    
    Args:
        center_point: Stage coordinates for rectangle center (x, y in meters)
        width: Rectangle width in meters
        height: Rectangle height in meters  
        scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion from meters to pixels (optional).
               If not provided, defaults to (1, 1)
        
    Returns:
        Numpy array of rectangle corner coordinates in napari format:
        [[y0, x0], [y1, x1], [y2, x2], [y3, x3]] representing the four corners
        in clockwise order starting from top-left
        
    Note:
        - Modifies the input center_point in-place by converting coordinates to pixel space
        - Uses napari coordinate convention (y, x) for corner coordinates
        - Scale factors should match camera pixel size for accurate positioning
        - Rectangle is axis-aligned (no rotation support)
        
    Example:
        >>> center = Point(x=10e-6, y=5e-6)  # 10μm x, 5μm y center
        >>> scale = (1e-6, 1e-6)  # 1μm per pixel
        >>> rect = create_rectangle_shape(center, 20e-6, 15e-6, scale)
        >>> # Returns 4x2 array of corner coordinates for 20x15 μm rectangle
    """
    if scale is None:
        scale = (1.0, 1.0)

    # Convert stage coordinates to pixel coordinates
    center_x = center_point.x / scale[0]
    center_y = center_point.y / scale[1]
    
    # Convert dimensions to pixels
    w_px = width / scale[0]
    h_px = height / scale[1]
    
    # Create rectangle corners in napari format (y, x)
    # Clockwise order starting from top-left
    corners = np.array([
        [center_y - h_px/2, center_x - w_px/2],  # Top-left
        [center_y - h_px/2, center_x + w_px/2],  # Top-right
        [center_y + h_px/2, center_x + w_px/2],  # Bottom-right
        [center_y + h_px/2, center_x - w_px/2]   # Bottom-left
    ])
    
    return corners


def create_circle_shape(
    center_point: Point, 
    radius: float, 
    scale: Optional[Tuple[float, float]] = None,
    num_points: int = 64
) -> np.ndarray:
    """Create a circle shape centered at the specified point.
    
    Generates circle points for creating a circular overlay in napari. The function 
    converts stage coordinates to pixel coordinates using the provided scale factors.
    
    Args:
        center_point: Stage coordinates for circle center (x, y in meters)
        radius: Circle radius in meters
        scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion from meters to pixels (optional).
               If not provided, defaults to (1, 1)
        num_points: Number of points to use for circle approximation (default: 64)
        
    Returns:
        Numpy array of circle points in napari format:
        [[y0, x0], [y1, x1], ...] representing points around the circle circumference
        
    Note:
        - Modifies the input center_point in-place by converting coordinates to pixel space
        - Uses napari coordinate convention (y, x) for point coordinates
        - Scale factors should match camera pixel size for accurate positioning
        - Circle is approximated using regular polygon with num_points vertices
        
    Example:
        >>> center = Point(x=10e-6, y=5e-6)  # 10μm x, 5μm y center
        >>> scale = (1e-6, 1e-6)  # 1μm per pixel
        >>> circle = create_circle_shape(center, 5e-6, scale)
        >>> # Returns array of points for 5μm radius circle
    """
    if scale is None:
        scale = (1.0, 1.0)

    # Convert stage coordinates to pixel coordinates
    center_x = center_point.x / scale[0]
    center_y = center_point.y / scale[1]
    
    # Convert radius to pixels
    radius_px = radius / scale[0]  # Assume square pixels for simplicity
    
    # Calculate bounding box for ellipse
    xmin = center_x - radius_px
    xmax = center_x + radius_px
    ymin = center_y - radius_px
    ymax = center_y + radius_px
    
    # create circle
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]
    return np.array(shape)

def update_text_overlay(viewer: napari.Viewer, microscope: FibsemMicroscope, 
                        stage_position: Optional[FibsemStagePosition] = None, 
                        objective_position: Optional[float] = None) -> None:
    """Update the text overlay in the napari viewer with current stage position and orientation.
    Args:
        viewer: napari viewer object
        microscope: FibsemMicroscope instance
    """
    if microscope is None:
        # Expected before a microscope is connected, not an error. FibsemUI builds the
        # minimap widget during setup, and that widget's `microscope` property reads
        # through to its parent, which is None until connect. The widget already
        # treats that as a normal state (`if self.microscope is None: return`); this
        # was the one path that passed it through to be dereferenced, so every launch
        # logged "Error updating text overlay: 'NoneType' object has no attribute
        # '_stage_position'".
        viewer.text_overlay.text = "<STAGE POSITION UNAVAILABLE>"
        return

    try:

        if isinstance(microscope, TescanMicroscope):
            return  # Tescan systems do not support stage position display yet

        if stage_position is None:
            stage_position = microscope._stage_position
        orientation = microscope.get_stage_orientation(stage_position=stage_position)
        current_grid = microscope.current_grid
        milling_angle = microscope.get_current_milling_angle(stage_position=stage_position)
        obj_txt = ""
        if microscope.fm is not None:
            # if objective_position is None:
                # objective_position = microscope.fm.objective.position
            if objective_position is not None:
                obj_txt = f"OBJECTIVE: {objective_position * constants.METRE_TO_MICRON:.1f} µm"

        # Create combined text for overlay
        overlay_text = (
            f"STAGE: {stage_position.pretty_string} [{orientation}] [{current_grid}]\n"
            f"MILLING ANGLE: {milling_angle:.1f}°\n"
            f"{obj_txt}\n"
        )
        viewer.text_overlay.visible = True
        viewer.text_overlay.position = "bottom_left"
        viewer.text_overlay.text = overlay_text

    except Exception as e:
        logging.warning(f"Error updating text overlay: {e}")
        # Fallback text if stage position unavailable
        viewer.text_overlay.text = "<STAGE POSITION UNAVAILABLE>"
        viewer.text_overlay.visible = False
        viewer.text_overlay.position = "bottom_left"


