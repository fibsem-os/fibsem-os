import numpy as np

# IMAGING


ALIGNMENT_LAYER_PROPERTIES = {
    "name": "alignment_area",
    "shape_type": "rectangle",
    "edge_color": "lime",
    "edge_width": 20,
    "face_color": "transparent",
    "opacity": 0.5,
    "metadata": {"type": "alignment"},
}

IMAGING_CROSSHAIR_LAYER_PROPERTIES = {
    "name": "crosshair",
    "shape_type": "line",
    "edge_width": 5,
    "edge_color": "yellow",
    "face_color": "yellow",
    "opacity": 0.8,
    "blending": "translucent",
}

IMAGING_SCALEBAR_LAYER_PROPERTIES = {
    "name": "scalebar",
    "shape_type": "line",
    "edge_width": 5,
    "edge_color": "yellow",
    "face_color": "yellow",
    "opacity": 0.8,
    "blending": "translucent",
    "text": {   
        "color":"white",
        "translation": np.array([-20, 0]),
        "opacity": 1,
        "sze": 20,
    },
}

# MILLING


## MINIMAP

OVERVIEW_IMAGE_LAYER_PROPERTIES = {
    "name": "overview-image",
    "colormap": "gray",
    "blending": "additive",
    "median_filter_size": 3,
}

GRIDBAR_IMAGE_LAYER_PROPERTIES = {
    "name": "gridbar-image",
    "spacing": 100,
    "width": 20,
}

CORRELATION_IMAGE_LAYER_PROPERTIES = {
    "name": "correlation-image",
    "colormap": "green",
    "blending": "translucent",
    "opacity": 0.2,
    "colours": ["green", "cyan", "magenta", "red", "yellow"],
}

