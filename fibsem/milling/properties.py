from fibsem import constants, config as cfg
from typing import List
from fibsem.structures import CrossSectionPattern
from fibsem.utils import format_resolution_as_str


DEFAULT_DISTANCE_METADATA = {
    "type": float,
    "unit": "m",
    "scale": 1e6,
    "minimum": 0.01,
    "maximum": 1000.0,
    "step": 0.01,
    "decimals": 2,
}

DEFAULT_ANGLE_METADATA = {
    "label": "Rotation",
    "type": float,
    "unit": constants.DEGREE_SYMBOL,
    "scale": None,
    "minimum": 0.0,
    "maximum": 360.0,
    "step": 1.0,
    "decimals": 2,
}
DEFAULT_DURATION_METADATA = {
    "label": "Time",
    "type": float,
    "unit": "s",
    "minimum": 0.0,
    "maximum": 10000.0,
    "step": 0.1,
    "decimals": 1,
    "advanced": True,
    "tooltip": "Specify the duration of the milling pattern in seconds. Set to 0 for automatic calculation.",
}
DEFAULT_SCAN_DIRECTION_METADATA = {
    "label": "Scan Direction",
    "type": str,
    "items": "dynamic",
    "microscope_parameter": "scan_direction",
    "tooltip": "Direction of the scan for the pattern.",
}

DEFAULT_CROSS_SECTION_METADATA = {
    "label": "Cross Section",
    "type": CrossSectionPattern,
    "items": [cs for cs in CrossSectionPattern],
    "tooltip": "The type of cross section for the milling pattern.",
}

DEFAULT_PASSES_METADATA = {
    "label": "Passes",
    "type": int,
    "minimum": 0,
    "maximum": 100,
    "step": 1,
    "advanced": True,
    "tooltip": "Number of passes for the pattern. Set to 0 for automatic calculation." 
}

DEFAULT_IMAGE_RESOLUTION_METADATA = {
    "label": "Image Resolution",
    "type": List[int],
    "items": cfg.STANDARD_RESOLUTIONS_LIST,
    "tooltip": "The imaging resolution in pixels (Width x Height).",
    "format_fn": format_resolution_as_str,
}
