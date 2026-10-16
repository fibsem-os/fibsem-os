from fibsem import constants
from fibsem.structures import (  # noqa: F401  (distance_param, DISTANCE_LIMITS: re-exported)
    DISTANCE_LIMITS,
    CrossSectionPattern,
    Display,
    Parameter,
    RangeLimit,
    distance_param,
)
from fibsem.utils import format_resolution_as_str

# Shared metadata for the parameters that recur across patterns and strategies,
# as a field's Parameter (for fibsem.structures.param): SI limits
# and a Display. Each is a function of what differs between the fields that use it
# (the label and the doc, sometimes the range), so a field says it in one call.


def angle_param(
    label: str,
    doc: str,
    *,
    limits: RangeLimit = RangeLimit(0.0, 360.0),
    step: float = 1.0,
    advanced: bool = False,
) -> Parameter:
    """An angle in degrees, not radians: a pattern's rotation is kept in degrees."""
    return Parameter(
        unit=constants.DEGREE_SYMBOL,
        limits=limits,
        doc=doc,
        display=Display(label, step=step, decimals=2, advanced=advanced),
    )


def duration_param(
    doc: str = "Specify the duration of the milling pattern in seconds. Set to 0 for automatic calculation.",
    label: str = "Time",
) -> Parameter:
    return Parameter(
        unit="s",
        limits=RangeLimit(0.0, 10000.0),
        doc=doc,
        display=Display(label, step=0.1, decimals=1, advanced=True),
    )


def passes_param(
    doc: str = "Number of passes for the pattern. Set to 0 for automatic calculation.",
) -> Parameter:
    return Parameter(
        limits=RangeLimit(0, 100000000),
        doc=doc,
        display=Display("Passes", step=1, advanced=True),
    )


def scan_direction_param(
    doc: str = "Direction of the scan for the pattern.",
) -> Parameter:
    """The milling service's scan directions, which the instrument decides."""
    return Parameter(
        device_parameter="milling.scan_direction",
        doc=doc,
        display=Display("Scan Direction"),
    )


def cross_section_param(
    doc: str = "The type of cross section for the milling pattern.",
) -> Parameter:
    return Parameter(
        choices=list(CrossSectionPattern),
        doc=doc,
        display=Display("Cross Section"),
    )


def image_resolution_param(
    doc: str = "The imaging resolution in pixels (Width x Height).",
) -> Parameter:
    """The beam's resolutions, which the instrument decides."""
    return Parameter(
        device_parameter="beam.resolution",
        doc=doc,
        display=Display("Image Resolution", format_fn=format_resolution_as_str),
    )
