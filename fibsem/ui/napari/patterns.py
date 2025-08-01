import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set, Union, Type

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import napari
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Layer as NapariLayer
from napari.layers import Shapes as NapariShapesLayers
from napari.utils import Colormap as NapariColormap
from skimage.transform import resize

from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import (
    BasePattern,
    FiducialPattern,
)
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    Point,
    calculate_fiducial_area_v2,
)
from fibsem.milling.patterning.utils import create_pattern_mask

# colour wheel
COLOURS = ["yellow", "cyan", "magenta", "lime", "orange", "hotpink", "green", "blue", "red", "purple"]
COLOURMAPS = {c: NapariColormap([to_rgba(c, alpha=0), to_rgba(c, alpha=1)]) for c in COLOURS}

SHAPES_LAYER_PROPERTIES = {
    "edge_width": 0.5,
    "opacity": 0.5,
    "blending": "translucent",
    "image_edge_width": 1,
}
IMAGE_LAYER_PROPERTIES = {
    "blending": "additive",
    "opacity": 0.6,
    "cmap": {0: "black", 1: COLOURS[0]} # override with colour wheel
}

IMAGE_PATTERN_TYPES = ("bitmap",)
IGNORE_SHAPES_LAYERS = ["ruler_line", "crosshair", "scalebar", "label", "alignment_area"] # ignore these layers when removing all shapes
STAGE_POSTIION_SHAPE_LAYERS = ["saved-stage-positions", "current-stage-position"] # for minimap
IGNORE_SHAPES_LAYERS.extend(STAGE_POSTIION_SHAPE_LAYERS)
CURRENT_PATTERN_LAYERS: Set[str] = set()

def get_image_pixel_centre(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Get the centre of the image in pixel coordinates."""
    icy, icx = shape[0] // 2, shape[1] // 2
    return icy, icx


def create_affine_matrix(
    scale: Tuple[float, float] = (1, 1),
    rotation: float = 0,
    centre: Tuple[float, float] = (0, 0),
    translation: Tuple[float, float] = (0, 0),
) -> np.ndarray:
    cos_theta = np.cos(-rotation)
    sin_theta = np.sin(-rotation)

    centre_image = np.asarray(
        [[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], dtype=float
    )

    scale_image = np.asarray(
        [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=float
    )

    rotate_image = np.asarray(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]], dtype=float
    )

    translate_image = np.asarray(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=float
    )

    transform = translate_image @ scale_image @ rotate_image @ centre_image

    return transform


def convert_pattern_to_napari_circle(
    pattern_settings: FibsemCircleSettings,
    shape: Tuple[int, int],
    pixelsize: float,
    translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not isinstance(pattern_settings, FibsemCircleSettings):
        raise ValueError(f"Pattern is not a Circle: {pattern_settings}")

    # image centre
    icy, icx = get_image_pixel_centre(shape)

    # pattern to pixel coords
    r = int(pattern_settings.radius / pixelsize)
    cx = int(icx + (pattern_settings.centre_x / pixelsize))
    cy = int(icy - (pattern_settings.centre_y / pixelsize))

    # create corner coords
    xmin, ymin = cx - r, cy - r
    xmax, ymax = cx + r, cy + r

    # create circle
    circle = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]  # ??
    return np.array(circle), {"translate": translation}


def convert_pattern_to_napari_line(
    pattern_settings: FibsemLineSettings,
    shape: Tuple[int, int],
    pixelsize: float,
    translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not isinstance(pattern_settings, FibsemLineSettings):
        raise ValueError(f"Pattern is not a Line: {pattern_settings}")

    # image centre
    icy, icx = get_image_pixel_centre(shape)

    # extract pattern information from settings
    start_x = pattern_settings.start_x
    start_y = pattern_settings.start_y
    end_x = pattern_settings.end_x
    end_y = pattern_settings.end_y

    # pattern to pixel coords
    px0 = int(icx + (start_x / pixelsize))
    py0 = int(icy - (start_y / pixelsize))
    px1 = int(icx + (end_x / pixelsize))
    py1 = int(icy - (end_y / pixelsize))

    # napari shape format [[y_start, x_start], [y_end, x_end]])
    line = [[py0, px0], [py1, px1]]
    return np.array(line), {"translate": translation}


def convert_pattern_to_napari_rect(
    pattern_settings: FibsemRectangleSettings,
    shape: Tuple[int, int],
    pixelsize: float,
    translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not isinstance(pattern_settings, FibsemRectangleSettings):
        raise ValueError(f"Pattern is not a Rectangle: {pattern_settings}")

    # image centre
    icy, icx = get_image_pixel_centre(shape)

    # extract pattern information from settings
    pattern_width = pattern_settings.width
    pattern_height = pattern_settings.height
    pattern_centre_x = pattern_settings.centre_x
    pattern_centre_y = pattern_settings.centre_y
    pattern_rotation = pattern_settings.rotation

    # pattern to pixel coords
    w = int(pattern_width / pixelsize)
    h = int(pattern_height / pixelsize)
    cx = int(icx + (pattern_centre_x / pixelsize))
    cy = int(icy - (pattern_centre_y / pixelsize))
    r = -pattern_rotation  #
    xmin, xmax = -w / 2, w / 2
    ymin, ymax = -h / 2, h / 2
    px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
    py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
    px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
    py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
    px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
    py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
    px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
    py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
    # napari shape format
    rect = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
    return np.array(rect), {"translate": translation}


def create_crosshair_shape(
    centre_point: Point,
    shape: Tuple[int, int],
    pixelsize: float,
    translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    icy, icx = shape[0] // 2, shape[1] // 2

    pattern_centre_x = centre_point.x
    pattern_centre_y = centre_point.y

    cx = int(icx + (pattern_centre_x / pixelsize))
    cy = int(icy - (pattern_centre_y / pixelsize))

    r_angles = [0, np.deg2rad(90)]  #
    w = 40
    h = 1
    crosshair_shapes = []

    for r in r_angles:
        xmin, xmax = -w / 2, w / 2
        ymin, ymax = -h / 2, h / 2
        px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
        py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
        px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
        py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
        px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
        py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
        px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
        py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
        # napari shape format
        rect = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
        crosshair_shapes.append(rect)

    return np.array(crosshair_shapes), {"translate": translation}


def convert_bitmap_pattern_to_napari_image(
    pattern_settings: FibsemBitmapSettings,
    shape: Tuple[int, int],
    pixelsize: float,
    translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    icy, icx = get_image_pixel_centre(shape)

    resize_x = int(pattern_settings.width / pixelsize)
    resize_y = int(pattern_settings.height / pixelsize)

    if pattern_settings.bitmap is None:
        img_array = np.zeros((resize_y, resize_x), dtype=float)
    else:
        # Resize here so that the border displays correctly
        img_array = resize(
            pattern_settings.bitmap[:, :, 0].astype(float), (resize_y, resize_x)
        )

    # scale = (img_array.shape[1] / resize_y, img_array.shape[0] / resize_x)
    # Add border
    ew = SHAPES_LAYER_PROPERTIES["image_edge_width"]
    img_array[:ew, :] = 1
    img_array[:, :ew] = 1
    img_array[img_array.shape[0] - ew - 1 :, :] = 1
    img_array[:, img_array.shape[1] - ew - 1 :] = 1

    return img_array, {
        "affine": create_affine_matrix(
            rotation=-pattern_settings.rotation,
            centre=(
                img_array.shape[0] / 2,
                img_array.shape[1] / 2,
            ),
            translation=(
                icy - pattern_settings.centre_y / pixelsize,
                icx + pattern_settings.centre_x / pixelsize,
            ),
        ),
        "translate": translation,
    }


def remove_all_napari_shapes_layers(
    viewer: napari.Viewer,
    layer_type: Type[NapariLayer] = NapariShapesLayers,
    ignore: List[str] = [],
):
    """Remove all shapes layers from the napari viewer, excluding a specified list."""
    # remove all shapes layers
    layers_to_remove = []
    layers_to_ignore = IGNORE_SHAPES_LAYERS + ignore
    for layer in viewer.layers:
        if layer.name in layers_to_ignore:
            continue
        if isinstance(layer, layer_type) or any([layer_name == layer.name for layer_name in CURRENT_PATTERN_LAYERS]):
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        viewer.layers.remove(layer)  # Not removing the second layer?
        CURRENT_PATTERN_LAYERS.discard(layer.name)

NAPARI_DRAWING_DICT = {
    FibsemRectangleSettings: (convert_pattern_to_napari_rect, "rectangle"),
    FibsemCircleSettings: (convert_pattern_to_napari_circle, "ellipse"),
    FibsemLineSettings: (convert_pattern_to_napari_line, "line"),
    FibsemBitmapSettings: (convert_bitmap_pattern_to_napari_image, "bitmap"),
}


@dataclass
class NapariPattern:
    name: str
    index: int
    shape: np.ndarray
    shape_type: str
    colour: str
    translate: Union[np.ndarray, Tuple[float, float]] = (0, 0)
    affine: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
    )

    @classmethod
    def draw(
        cls,
        name: str,
        index: int,
        pattern_settings: FibsemPatternSettings,
        image_shape: Tuple[int, int],
        pixelsize: float,
        colour: str,
        translation: Union[np.ndarray, Tuple[float, float]] = (0, 0),
    ) -> Optional["NapariPattern"]:
        napari_drawing_fn, shape_type = NAPARI_DRAWING_DICT.get(
            type(pattern_settings), (None, None)
        )
        if napari_drawing_fn is None:
            logging.warning(f"Pattern type {type(pattern_settings)} not supported")
            return None

        shape, kwargs = napari_drawing_fn(
            pattern_settings=pattern_settings,
            shape=image_shape,
            pixelsize=pixelsize,
            translation=translation,
        )

        return cls(
            name=name,
            index=index,
            shape=shape,
            shape_type=shape_type,
            colour=colour,
            **kwargs,
        )


def draw_milling_patterns_in_napari(
    viewer: napari.Viewer,
    image_layer: NapariImageLayer,
    milling_stages: List[FibsemMillingStage],
    pixelsize: float,
    draw_crosshair: bool = True,
    background_milling_stages: Optional[List[FibsemMillingStage]] = None,
) -> List[str]:
    """Draw the milling patterns in napari as a combination of Shapes and Label layers.
    Args:
        viewer: napari viewer instance
        image: image to draw patterns on
        translation: translation of the FIB image layer
        milling_stages): list of milling stages
        draw_crosshair: draw crosshair on the image
        background_milling_stages: optional list of background milling stages to draw
    Returns:
        List[str]: list of milling pattern layers
    """

    # base image properties
    image_shape = image_layer.data.shape
    translation = image_layer.translate

    all_napari_patterns: Dict[str, List[NapariPattern]] = {}

    all_milling_stages = deepcopy(milling_stages)
    if background_milling_stages is not None:
        all_milling_stages.extend(deepcopy(background_milling_stages))
    n_milling_stages = len(milling_stages)

    # convert fibsem patterns to napari shapes
    for i, stage in enumerate(all_milling_stages):
        # shapes for this milling stage
        napari_patterns: List[NapariPattern] = []

        is_background = i >= n_milling_stages
        if is_background:
            napari_layer_colour = "black"
        else:
            napari_layer_colour = COLOURS[i % len(COLOURS)]

        # TODO: QUERY  migrate to using label layers for everything??
        # TODO: re-enable annulus drawing, re-enable bitmaps
        for i, pattern_settings in enumerate(stage.pattern.define(), 1):
            pattern = NapariPattern.draw(
                name=stage.name,
                index=i,
                pattern_settings=pattern_settings,
                image_shape=(image_shape[0], image_shape[1]),
                pixelsize=pixelsize,
                colour=napari_layer_colour,
                translation=translation,
            )
            if pattern is None:
                continue

            napari_patterns.append(pattern)

        # draw the patterns as a shape layer
        if napari_patterns:
            if draw_crosshair:
                crosshair_shape, kwargs = create_crosshair_shape(
                    centre_point=stage.pattern.point,
                    shape=(image_shape[0], image_shape[1]),
                    pixelsize=pixelsize,
                    translation=translation,
                )
                for i, rect in enumerate(crosshair_shape, 1):
                    napari_patterns.append(
                        NapariPattern(
                            name="crosshair",
                            index=i,
                            shape=rect,
                            shape_type="rectangle",
                            colour=napari_layer_colour,
                            translate=translation,
                        )
                    )

            # TODO: properties dict for all parameters
            all_napari_patterns[stage.name] = napari_patterns
    layer_names_used: Set[str] = set()
    if all_napari_patterns:
        opacity = SHAPES_LAYER_PROPERTIES["opacity"]
        blending = SHAPES_LAYER_PROPERTIES["blending"]
        edge_width = SHAPES_LAYER_PROPERTIES["edge_width"]
        shapes_list: List[np.ndarray] = []
        shape_types: List[str] = []
        shape_colours: list[str] = []
        for i, (layer_name, patterns) in enumerate(all_napari_patterns.items()):
            image_list: List[NapariPattern] = []
            for pattern in patterns:
                if pattern.shape_type in IMAGE_PATTERN_TYPES:
                    image_list.append(pattern)
                else:
                    shapes_list.append(pattern.shape)
                    shape_types.append(pattern.shape_type)
                    shape_colours.append(pattern.colour)

            for shape in image_list:
                # Napari applies translate before affine, which causes issues
                # with centring for the rotation and scaling. Applying
                # translate via the affine avoids this issue.
                translate_affine = np.asarray(
                    [[1, 0, shape.translate[0]], [0, 1, shape.translate[1]], [0, 0, 1]]
                )
                affine = translate_affine @ shape.affine
                # Requires a separate layer per-image
                layer_name = f"{shape.name} {shape.shape_type} {shape.index}"
                if layer_name in viewer.layers:
                    # Update layer if it already exists
                    viewer.layers[layer_name].data = shape.shape
                    viewer.layers[layer_name].colormap = COLOURMAPS[shape.colour]
                    viewer.layers[layer_name].opacity = opacity
                    viewer.layers[layer_name].blending = blending
                    viewer.layers[layer_name].affine = affine
                else:
                    viewer.add_image(
                        data=shape.shape,
                        name=layer_name,
                        colormap=COLOURMAPS[shape.colour],
                        opacity=opacity,
                        blending=blending,
                        depiction="plane",
                        affine=affine,
                    )
                layer_names_used.add(layer_name)

        if shapes_list:
            layer_name = "Milling Patterns"
            if layer_name in viewer.layers:
                # need to clear data before updating, to account for different shapes.
                viewer.layers[layer_name].data = []
                viewer.layers[layer_name].data = shapes_list
                viewer.layers[layer_name].shape_type = shape_types
                viewer.layers[layer_name].edge_color = shape_colours
                viewer.layers[layer_name].face_color = shape_colours
                viewer.layers[layer_name].translate = translation
                viewer.layers[layer_name].opacity = opacity
                viewer.layers[layer_name].blending = blending
            else:
                viewer.add_shapes(
                    data=shapes_list,
                    name=layer_name,
                    shape_type=shape_types,
                    edge_width=edge_width,
                    edge_color=shape_colours,
                    face_color=shape_colours,
                    opacity=opacity,
                    blending=blending,
                    translate=translation,
                )
            layer_names_used.add(layer_name)

    CURRENT_PATTERN_LAYERS.update(layer_names_used)

    layer_name_list = list(layer_names_used)

    # remove all un-updated layers (assume they have been deleted)
    remove_all_napari_shapes_layers(
        viewer=viewer, layer_type=NapariShapesLayers, ignore=layer_name_list
    )

    return layer_name_list  # list of milling pattern layers


def convert_point_to_napari(resolution: list, pixel_size: float, centre: Point):
    icy, icx = resolution[1] // 2, resolution[0] // 2

    cx = int(icx + (centre.x / pixel_size))
    cy = int(icy - (centre.y / pixel_size))

    return Point(cx, cy)


def validate_pattern_image_placement(
    image_shape: Tuple[int, int], image: np.ndarray, affine: Optional[np.ndarray] = None
) -> bool:
    corners = [
        [0, 0],
        [image.shape[0], 0],
        [image.shape[0], image.shape[1]],
        [0, image.shape[1]],
    ]

    return validate_pattern_shape_placement(
        image_shape=image_shape, shape=corners, affine=affine
    )


def validate_pattern_shape_placement(
    image_shape: Tuple[int, int],
    shape: List[List[Union[float, int]]],
    affine: Optional[np.ndarray] = None,
):
    """Validate that the pattern shapes are within the image resolution"""
    x_lim = image_shape[1]
    y_lim = image_shape[0]

    shape_array = np.asarray(shape, dtype=float)
    if affine is not None:
        # A bit fiddly but this applies the affine array to the 2D coordinates
        # without broadcasting issues.
        coords = np.pad(shape_array, ((0, 0), (0, 1)), constant_values=1)
        shape_array = (affine[:2, :] @ coords[:, :, np.newaxis]).squeeze(-1)
    ymin = np.min(shape_array[:, 0])
    ymax = np.max(shape_array[:, 0])
    xmin = np.min(shape_array[:, 1])
    xmax = np.max(shape_array[:, 1])

    if xmin < 0 or xmax > x_lim:
        return False
    if ymin < 0 or ymax > y_lim:
        return False

    return True

def is_pattern_placement_valid(pattern: BasePattern, image: FibsemImage) -> bool:
    """Check if the pattern is within the image bounds."""

    if isinstance(pattern, FiducialPattern):
        _, is_not_valid_placement = calculate_fiducial_area_v2(
            image=image,
            fiducial_centre=deepcopy(pattern.point),
            fiducial_length=pattern.height,
        )
        return not is_not_valid_placement

    for pattern_settings in pattern.define():
        draw_func, shape_type = NAPARI_DRAWING_DICT.get(
            type(pattern_settings), (None, None)
        )
        if draw_func is None:
            logging.warning(f"Pattern type {type(pattern_settings)} not supported")
            return False

        napari_shape, kwargs = draw_func(
            pattern_settings=pattern_settings,
            shape=image.data.shape,
            pixelsize=image.metadata.pixel_size.x,
        )
        if shape_type in IMAGE_PATTERN_TYPES:
            is_valid_placement = validate_pattern_image_placement(
                image_shape=image.data.shape,
                image=napari_shape,
                affine=kwargs.get("affine"),
            )
        else:
            is_valid_placement = validate_pattern_shape_placement(
                image_shape=image.data.shape,
                shape=napari_shape,
                affine=kwargs.get("affine"),
            )

        if not is_valid_placement:
            return False

    return True

def convert_reduced_area_to_napari_shape(reduced_area: FibsemRectangle, image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert a reduced area to a napari shape."""
    x0 = reduced_area.left * image_shape[1]
    y0 = reduced_area.top * image_shape[0]
    x1 = x0 + reduced_area.width * image_shape[1]
    y1 = y0 + reduced_area.height * image_shape[0]
    data = [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]
    return np.array(data)

def convert_shape_to_image_area(shape: List[List[int]], image_shape: Tuple[int, int]) -> FibsemRectangle:
    """Convert a napari shape (rectangle) to  a FibsemRectangle expressed as a percentage of the image (reduced area)
    shape: the coordinates of the shape
    image_shape: the shape of the image (usually the ion beam image)    
    """
    # get limits of rectangle
    y0, x0 = shape[0]
    y1, x1 = shape[1]
    """
        0################1
        |               |
        |               |
        |               |
        3################2
    """
    # get min/max coordinates
    x_coords = [x[1] for x in shape]
    y_coords = [x[0] for x in shape]
    x0, x1 = min(x_coords), max(x_coords)
    y0, y1 = min(y_coords), max(y_coords)

    logging.debug(f"convert shape data: {x0}, {x1}, {y0}, {y1}, fib shape: {image_shape}")
        
    # convert to percentage of image
    x0 = x0 / image_shape[1]
    x1 = x1 / image_shape[1]
    y0 = y0 / image_shape[0]
    y1 = y1 / image_shape[0]
    w = x1 - x0
    h = y1 - y0

    reduced_area = FibsemRectangle(left=x0, top=y0, width=w, height=h)
    logging.debug(f"reduced area: {reduced_area}")

    return reduced_area