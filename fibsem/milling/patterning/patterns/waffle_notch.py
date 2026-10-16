from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    cross_section_param,
    distance_param,
)
from fibsem.structures import (
    CrossSectionPattern,
    Display,
    FibsemRectangleSettings,
    param,
)


@dataclass
class WaffleNotchPattern(BasePattern[FibsemRectangleSettings]):
    vheight: float = param(
        2.0e-6,
        distance_param("Vertical Height", "Height of the vertical notch sections."),
    )
    vwidth: float = param(
        0.5e-6,
        distance_param("Vertical Width", "Width of the vertical notch sections."),
    )
    hheight: float = param(
        0.5e-6,
        distance_param("Horizontal Height", "Height of the horizontal notch sections."),
    )
    hwidth: float = param(
        2.0e-6,
        distance_param("Horizontal Width", "Width of the horizontal notch sections."),
    )
    depth: float = param(
        1.0e-6, distance_param("Depth", "Depth of the notch sections.")
    )
    distance: float = param(
        2.0e-6, distance_param("Distance", "Distance between notch sections.")
    )
    inverted: bool = param(
        False, doc="Invert the notch pattern.", display=Display("Inverted")
    )
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
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
