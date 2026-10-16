from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_param,
)
from fibsem.structures import Display, FibsemLineSettings, param


@dataclass
class SerialSectionPattern(BasePattern[FibsemLineSettings]):
    section_thickness: float = param(
        4.0e-6, distance_param("Section Thickness", "Thickness of the section.")
    )
    section_width: float = param(
        50.0e-6, distance_param("Section Width", "Width of the section.")
    )
    section_depth: float = param(
        20.0e-6, distance_param("Section Depth", "Depth of the section.")
    )
    side_width: float = param(
        10.0e-6, distance_param("Side Width", "Width of the side cleaning area.")
    )
    side_height: float = param(
        10.0e-6, distance_param("Side Height", "Height of the side cleaning area.")
    )
    side_depth: float = param(
        40.0e-6, distance_param("Side Depth", "Depth of the side cleaning area.")
    )
    inverted: bool = param(
        False,
        doc="If true, the section will be milled inverted.",
        display=Display("Inverted"),
    )
    use_side_patterns: bool = param(
        True,
        doc="If true, side cleaning patterns will be used.",
        display=Display("Use Side Patterns"),
    )

    name: ClassVar[str] = "SerialSection"
    # ref: "serial-liftout section" https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[FibsemLineSettings]:
        """Calculate the serial liftout sectioning milling patterns"""

        point = self.point
        section_thickness = self.section_thickness
        section_width = self.section_width
        section_depth = self.section_depth
        side_width = self.side_width
        side_height = self.side_height
        side_depth = self.side_depth
        inverted = self.inverted
        use_side_patterns = self.use_side_patterns

        # draw a line of section width
        section_y = section_thickness
        if inverted:
            section_y *= -1.0
            side_height *= -1.0

        # main section pattern
        section_pattern = FibsemLineSettings(
            start_x=point.x - section_width / 2,
            end_x=point.x + section_width / 2,
            start_y=point.y + section_y,
            end_y=point.y + section_y,
            depth=section_depth,
        )

        self.shapes = [section_pattern]

        if use_side_patterns:
            # side cleaning patterns
            left_side_pattern = FibsemLineSettings(
                start_x=point.x - section_width / 2 - side_width / 2,
                end_x=point.x - section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )
            right_side_pattern = FibsemLineSettings(
                start_x=point.x + section_width / 2 - side_width / 2,
                end_x=point.x + section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )

            # side vertical patterns
            left_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x - section_width / 2,
                end_x=point.x - section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            right_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x + section_width / 2,
                end_x=point.x + section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            self.shapes.extend(
                [
                    left_side_pattern,
                    right_side_pattern,
                    left_side_pattern_vertical,
                    right_side_pattern_vertical,
                ]
            )

        return self.shapes
