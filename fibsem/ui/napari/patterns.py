from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import napari
import numpy as np
from matplotlib.colors import to_rgba
from napari.layers import Image as NapariImageLayer
from napari.layers import Layer as NapariLayer
from napari.layers import Shapes as NapariShapesLayers
from napari.utils import Colormap as NapariColormap
from skimage.transform import resize

from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import (
    BasePattern,
    FiducialPattern,
)
from fibsem.milling.patterning.shapes import (  # noqa: F401
    COLOURS,
    IMAGE_PATTERN_TYPES,
    NAPARI_DRAWING_DICT,
    convert_bitmap_pattern_to_napari_image,
    convert_pattern_to_napari_circle,
    convert_pattern_to_napari_line,
    convert_pattern_to_napari_polygon,
    convert_pattern_to_napari_rect,
    convert_point_to_napari,
    convert_reduced_area_to_napari_shape,
    convert_shape_to_image_area,
    create_affine_matrix,
    create_crosshair_shape,
    is_pattern_placement_valid,
    validate_pattern_image_placement,
    validate_pattern_shape_placement,
)
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    Point,
    calculate_fiducial_area_v2,
)
from fibsem.ui.napari.properties import ALIGNMENT_LAYER_PROPERTIES

# The colour wheel and every shape converter live in `fibsem.milling.patterning.shapes`,
# which needs neither napari nor Qt; re-exported here so existing imports keep working.
COLOURMAPS = {
    c: NapariColormap([to_rgba(c, alpha=0), to_rgba(c, alpha=1)]) for c in COLOURS
}

SHAPES_LAYER_PROPERTIES = {
    "edge_width": 0.5,
    "opacity": 0.5,
    "blending": "translucent",
    "image_edge_width": 1,
}
IMAGE_LAYER_PROPERTIES = {
    "blending": "additive",
    "opacity": 0.6,
    "cmap": {0: "black", 1: COLOURS[0]},  # override with colour wheel
}

MILLING_ALIGNMENT_AREA_LAYER_NAME = "Alignment Area"
MILLING_PATTERN_LAYER_NAME = "Milling Patterns"
MILLING_FOV_LAYER_NAME = "Milling FOV"
IGNORE_SHAPES_LAYERS = [
    "ruler_line",
    "crosshair",
    "scalebar",
    "label",
    "overlay-shapes",
    "bbox",
    MILLING_FOV_LAYER_NAME,
    MILLING_ALIGNMENT_AREA_LAYER_NAME,
]  # ignore these layers when removing all shapes
STAGE_POSTIION_SHAPE_LAYERS = [
    "saved-stage-positions",
    "current-stage-position",
    "stage-position",
]  # for minimap
IGNORE_SHAPES_LAYERS.extend(STAGE_POSTIION_SHAPE_LAYERS)
CURRENT_PATTERN_LAYERS: Set[str] = set()


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
        if isinstance(layer, layer_type) or any(
            [layer_name == layer.name for layer_name in CURRENT_PATTERN_LAYERS]
        ):
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        viewer.layers.remove(layer)  # Not removing the second layer?
        CURRENT_PATTERN_LAYERS.discard(layer.name)


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

        if hasattr(pattern_settings, "is_exclusion") and pattern_settings.is_exclusion:
            colour = "black"

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
    colors: Optional[List[str]] = None,
    alignment_area: Optional[FibsemRectangle] = None,
    selected_index: Optional[int] = None,
) -> List[str]:
    """Draw the milling patterns in napari as a combination of Shapes and Label layers.
    Args:
        viewer: napari viewer instance
        image: image to draw patterns on
        translation: translation of the FIB image layer
        milling_stages: list of milling stages
        draw_crosshair: draw crosshair on the image
        background_milling_stages: optional list of background milling stages to draw
        alignment_area: optional alignment area to draw
        selected_index: index of the selected milling stage (0-based), draws with thicker border
    Returns:
        List[str]: list of milling pattern layers
    """
    if colors is None:
        colors = COLOURS

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
            napari_layer_colour = colors[i % len(colors)]

        # TODO: QUERY  migrate to using label layers for everything??
        # TODO: re-enable annulus drawing, re-enable bitmaps
        for i, pattern_settings in enumerate(stage.define_patterns(), 1):
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
    opacity = SHAPES_LAYER_PROPERTIES["opacity"]
    blending = SHAPES_LAYER_PROPERTIES["blending"]
    edge_width = SHAPES_LAYER_PROPERTIES["edge_width"]
    selected_edge_width = edge_width * 3  # thicker border for selected stage
    shapes_list: List[np.ndarray] = []
    shape_types: List[str] = []
    edge_colours: List[str] = []
    face_colours: List[str] = []
    edge_widths: List[float] = []

    if all_napari_patterns:
        for i, (layer_name, patterns) in enumerate(all_napari_patterns.items()):
            is_selected = selected_index is not None and i == selected_index
            image_list: List[NapariPattern] = []
            for pattern in patterns:
                if pattern.shape_type in IMAGE_PATTERN_TYPES:
                    image_list.append(pattern)
                else:
                    shapes_list.append(pattern.shape)
                    shape_types.append(pattern.shape_type)
                    edge_colours.append(pattern.colour)
                    face_colours.append(pattern.colour)
                    edge_widths.append(
                        selected_edge_width if is_selected else edge_width
                    )

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

    # fold alignment area into the milling patterns layer
    if alignment_area is not None:
        alignment_shape = convert_reduced_area_to_napari_shape(
            reduced_area=alignment_area,
            image_shape=image_shape,
        )
        shapes_list.append(alignment_shape)
        shape_types.append(ALIGNMENT_LAYER_PROPERTIES["shape_type"])
        edge_colours.append(ALIGNMENT_LAYER_PROPERTIES["edge_color"])
        face_colours.append(ALIGNMENT_LAYER_PROPERTIES["face_color"])
        edge_widths.append(ALIGNMENT_LAYER_PROPERTIES["edge_width"])

    if shapes_list:
        layer_name = MILLING_PATTERN_LAYER_NAME
        if layer_name in viewer.layers:
            # need to clear data before updating, to account for different shapes.
            viewer.layers[layer_name].data = []
            viewer.layers[layer_name].data = shapes_list
            viewer.layers[layer_name].shape_type = shape_types
            viewer.layers[layer_name].edge_width = edge_widths
            viewer.layers[layer_name].edge_color = edge_colours
            viewer.layers[layer_name].face_color = face_colours
            viewer.layers[layer_name].translate = translation
            viewer.layers[layer_name].opacity = opacity
            viewer.layers[layer_name].blending = blending
        else:
            viewer.add_shapes(
                data=shapes_list,
                name=layer_name,
                shape_type=shape_types,
                edge_width=edge_widths,
                edge_color=edge_colours,
                face_color=face_colours,
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
