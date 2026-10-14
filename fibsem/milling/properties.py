from typing import Any, Dict, List

from fibsem import constants
from fibsem.display import Display
from fibsem.structures import CrossSectionPattern, RangeLimit, field_meta
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
    items="dynamic",
    microscope_parameter="resolution",
    tooltip="The imaging resolution in pixels (Width x Height).",
    format_fn=format_resolution_as_str,
)


# The same, in the device Parameter's vocabulary (fibsem.structures.field_meta):
# SI limits and a Display. Each is a function of what differs between the fields
# that use it (the label and the doc, sometimes the range), so a field says it in
# one call rather than spreading a dict and overriding keys. The dicts above go
# once the last pattern has moved to these.

DISTANCE_LIMITS = RangeLimit(0.01e-6, 1000e-6)


def distance_metadata(
    label: str,
    doc: str,
    *,
    limits: RangeLimit = DISTANCE_LIMITS,
    step: float = 0.1,
    decimals: int = 2,
    advanced: bool = False,
) -> Dict[str, Any]:
    """A length in metres, shown in µm."""
    return field_meta(
        unit="m",
        limits=limits,
        doc=doc,
        display=Display(
            label, scale=1e6, step=step, decimals=decimals, advanced=advanced
        ),
    )


def angle_metadata(
    label: str,
    doc: str,
    *,
    limits: RangeLimit = RangeLimit(0.0, 360.0),
    step: float = 1.0,
    advanced: bool = False,
) -> Dict[str, Any]:
    """An angle in degrees, not radians: a pattern's rotation is kept in degrees."""
    return field_meta(
        unit=constants.DEGREE_SYMBOL,
        limits=limits,
        doc=doc,
        display=Display(label, step=step, decimals=2, advanced=advanced),
    )


def duration_metadata(
    doc: str = "Specify the duration of the milling pattern in seconds. Set to 0 for automatic calculation.",
    label: str = "Time",
) -> Dict[str, Any]:
    return field_meta(
        unit="s",
        limits=RangeLimit(0.0, 10000.0),
        doc=doc,
        display=Display(label, step=0.1, decimals=1, advanced=True),
    )


def passes_metadata(
    doc: str = "Number of passes for the pattern. Set to 0 for automatic calculation.",
) -> Dict[str, Any]:
    return field_meta(
        limits=RangeLimit(0, 100000000),
        doc=doc,
        display=Display("Passes", step=1, advanced=True),
    )


def scan_direction_metadata(
    doc: str = "Direction of the scan for the pattern.",
) -> Dict[str, Any]:
    """The milling service's scan directions, which the instrument decides."""
    return field_meta(
        microscope_parameter="scan_direction",
        doc=doc,
        display=Display("Scan Direction"),
    )


def cross_section_metadata(
    doc: str = "The type of cross section for the milling pattern.",
) -> Dict[str, Any]:
    return field_meta(
        choices=list(CrossSectionPattern),
        doc=doc,
        display=Display("Cross Section"),
    )


def image_resolution_metadata(
    doc: str = "The imaging resolution in pixels (Width x Height).",
) -> Dict[str, Any]:
    """The beam's resolutions, which the instrument decides."""
    return field_meta(
        microscope_parameter="resolution",
        doc=doc,
        display=Display("Image Resolution"),
        format_fn=format_resolution_as_str,
    )
