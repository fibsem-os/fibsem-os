from typing import List

from fibsem import config as cfg
from fibsem import constants
from fibsem.structures import CrossSectionPattern, field_meta
from fibsem.utils import format_resolution_as_str

# Shared metadata for the parameters that recur across patterns and strategies.
# Declared through `field_meta` so the keys are checked once here rather than at
# each of the ~110 sites that spread these in -- a typo in a shared dict is a
# typo in every pattern that uses it.

DEFAULT_DISTANCE_METADATA = field_meta(
    type=float,
    unit="m",
    scale=1e6,
    minimum=0.01,
    maximum=1000.0,
    step=0.1,
    decimals=2,
)

DEFAULT_ANGLE_METADATA = field_meta(
    label="Rotation",
    type=float,
    unit=constants.DEGREE_SYMBOL,
    scale=None,
    minimum=0.0,
    maximum=360.0,
    step=1.0,
    decimals=2,
)
DEFAULT_DURATION_METADATA = field_meta(
    label="Time",
    type=float,
    unit="s",
    minimum=0.0,
    maximum=10000.0,
    step=0.1,
    decimals=1,
    advanced=True,
    tooltip="Specify the duration of the milling pattern in seconds. Set to 0 for automatic calculation.",
)
DEFAULT_SCAN_DIRECTION_METADATA = field_meta(
    label="Scan Direction",
    type=str,
    items="dynamic",
    microscope_parameter="scan_direction",
    tooltip="Direction of the scan for the pattern.",
)

DEFAULT_CROSS_SECTION_METADATA = field_meta(
    label="Cross Section",
    type=CrossSectionPattern,
    items=[cs for cs in CrossSectionPattern],
    tooltip="The type of cross section for the milling pattern.",
)

DEFAULT_PASSES_METADATA = field_meta(
    label="Passes",
    type=int,
    minimum=0,
    maximum=100000000,
    step=1,
    advanced=True,
    tooltip="Number of passes for the pattern. Set to 0 for automatic calculation.",
)

DEFAULT_IMAGE_RESOLUTION_METADATA = field_meta(
    label="Image Resolution",
    type=List[int],
    items=cfg.STANDARD_RESOLUTIONS_LIST,
    tooltip="The imaging resolution in pixels (Width x Height).",
    format_fn=format_resolution_as_str,
)
