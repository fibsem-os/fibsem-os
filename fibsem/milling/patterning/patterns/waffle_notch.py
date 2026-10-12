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
)
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangleSettings,
)


@dataclass
class WaffleNotchPattern(BasePattern[FibsemRectangleSettings]):
    vheight: float = field(
        default=2.0e-6,
        metadata={
            "label": "Vertical Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the vertical notch sections.",
        },
    )
    vwidth: float = field(
        default=0.5e-6,
        metadata={
            "label": "Vertical Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the vertical notch sections.",
        },
    )
    hheight: float = field(
        default=0.5e-6,
        metadata={
            "label": "Horizontal Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the horizontal notch sections.",
        },
    )
    hwidth: float = field(
        default=2.0e-6,
        metadata={
            "label": "Horizontal Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horizontal notch sections.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the notch sections.",
        },
    )
    distance: float = field(
        default=2.0e-6,
        metadata={
            "label": "Distance",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Distance between notch sections.",
        },
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "tooltip": "Invert the notch pattern.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "WaffleNotch"
    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(self) -> List[FibsemRectangleSettings]:

        point = self.point
        vwidth = self.vwidth
        vheight = self.vheight
        hwidth = self.hwidth
        hheight = self.hheight
        depth = self.depth
        distance = self.distance
        cross_section = self.cross_section
        inverted = -1 if self.inverted else 1

        # five patterns
        top_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - distance / 2 - vheight / 2 + hheight / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )

        bottom_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + distance / 2 + vheight / 2 - hheight / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section,
        )

        top_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y - distance / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )

        bottom_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y + distance / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section,
        )

        centre_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=distance + hheight,
            depth=depth,
            centre_x=point.x + (hwidth + vwidth) * inverted,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )

        self.shapes = [
            top_vertical_pattern,
            bottom_vertical_pattern,
            top_horizontal_pattern,
            bottom_horizontal_pattern,
            centre_vertical_pattern,
        ]

        return self.shapes
