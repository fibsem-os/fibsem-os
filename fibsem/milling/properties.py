from typing import Any, Dict

from fibsem import constants
from fibsem.structures import (  # noqa: F401  (distance_metadata, DISTANCE_LIMITS: re-exported)
    DISTANCE_LIMITS,
    CrossSectionPattern,
    Display,
    RangeLimit,
    distance_metadata,
    field_meta,
)
from fibsem.utils import format_resolution_as_str

# Shared metadata for the parameters that recur across patterns and strategies,
# in the device Parameter's vocabulary (fibsem.structures.field_meta): SI limits
# and a Display. Each is a function of what differs between the fields that use it
# (the label and the doc, sometimes the range), so a field says it in one call.


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
        device_parameter="milling.scan_direction",
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
        device_parameter="beam.resolution",
        doc=doc,
        display=Display("Image Resolution", format_fn=format_resolution_as_str),
    )
