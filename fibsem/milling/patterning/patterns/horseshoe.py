from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    DEFAULT_CROSS_SECTION_METADATA,
    DEFAULT_DISTANCE_METADATA,
    DEFAULT_SCAN_DIRECTION_METADATA,
)
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangleSettings,
)


@dataclass
class HorseshoePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=40.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horseshoe pattern.",
        },
    )
    upper_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Upper Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the upper trench.",
        },
    )
    lower_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Lower Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the lower trench.",
        },
    )
    spacing: float = field(
        default=10.0e-6,
        metadata={
            "label": "Spacing",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Spacing between trenches.",
        },
    )
    depth: float = field(
        default=10.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trenches.",
        },
    )
    side_width: float = field(
        default=5.0e-6,
        metadata={
            "label": "Side Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the side trench.",
        },
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "type": bool,
            "tooltip": "If true, the side trench will be milled on the opposite side",
        },
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "Horseshoe"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the trench milling patterns"""

        point = self.point
        width = self.width
        depth = self.depth
        spacing = self.spacing
        lower_trench_height = self.lower_trench_height
        upper_trench_height = self.upper_trench_height
        cross_section = self.cross_section
        side_width = self.side_width

        # calculate the centre of the upper and lower trench
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)

        # calculate the centre of the side trench
        side_height = spacing + upper_trench_height + lower_trench_height
        side_offset = (width / 2) + (side_width / 2)
        if self.inverted:
            side_offset = -side_offset
        side_x = point.x + side_offset
        # to account for assymetric trench heights
        side_y = point.y + (upper_trench_height - lower_trench_height) / 2

        lower_pattern = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section=cross_section,
        )

        upper_pattern = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )

        side_pattern = FibsemRectangleSettings(
            width=side_width,
            height=side_height,
            depth=depth,
            centre_x=side_x,
            centre_y=side_y,
            scan_direction=self.scan_direction,
            cross_section=cross_section,
        )

        self.shapes = [upper_pattern, lower_pattern, side_pattern]
        return self.shapes


@dataclass
class HorseshoePatternVertical(BasePattern):
    width: float = field(
        default=2.0e-05,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horseshoe vertical pattern.",
        },
    )
    height: float = field(
        default=5.0e-05,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the horseshoe vertical pattern.",
        },
    )
    side_trench_width: float = field(
        default=5.0e-06,
        metadata={
            "label": "Side Trench Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the side trenches.",
        },
    )
    top_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Top Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the top trench.",
        },
    )
    depth: float = field(
        default=4.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trenches.",
        },
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "type": bool,
            "tooltip": "If true, the top trench will be milled below the point.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "HorseshoeVertical"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the horseshoe vertical milling patterns"""

        point = self.point
        width = self.width
        height = self.height
        trench_width = self.side_trench_width
        upper_trench_height = self.top_trench_height
        depth = self.depth
        scan_direction = self.scan_direction
        inverted = self.inverted
        cross_section = self.cross_section

        left_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x - (width / 2) - (trench_width / 2),
            centre_y=point.y,
            scan_direction="LeftToRight",
            cross_section=cross_section,
        )

        right_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x + (width / 2) + (trench_width / 2),
            centre_y=point.y,
            scan_direction="RightToLeft",
            cross_section=cross_section,
        )
        y_offset = (height / 2) + (upper_trench_height / 2)
        if inverted:
            y_offset = -y_offset
        upper_pattern = FibsemRectangleSettings(
            width=width + (2 * trench_width),
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + y_offset,
            scan_direction=scan_direction,
            cross_section=cross_section,
        )

        self.shapes = [upper_pattern, left_pattern, right_pattern]
        return self.shapes
