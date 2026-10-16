from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
)

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_param,
    cross_section_param,
    distance_param,
    duration_param,
    passes_param,
    scan_direction_param,
)
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangleSettings,
    param,
)


@dataclass
class RectanglePattern(BasePattern[FibsemRectangleSettings]):
    width: float = param(
        10.0e-6, distance_param("Width", "Width of the rectangle pattern.")
    )
    height: float = param(
        5.0e-6, distance_param("Height", "Height of the rectangle pattern.")
    )
    depth: float = param(
        1.0e-6, distance_param("Depth", "Depth of the rectangle pattern.")
    )
    rotation: float = param(
        0.0,
        angle_param(
            "Rotation", "Rotation of the rectangle pattern in degrees.", advanced=True
        ),
    )

    time: float = param(0.0, duration_param())
    passes: int = param(0, passes_param())
    scan_direction: str = param("TopToBottom", scan_direction_param())
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
    )
    name: ClassVar[str] = "Rectangle"

    def summary(self) -> str:
        from fibsem.utils import format_value

        return "\n".join(
            [
                f"    Pattern: {self.name}",
                f"        Depth: {format_value(self.depth, unit='m', precision=1)}",
                f"        Width: {format_value(self.width, unit='m', precision=1)}",
                f"        Height: {format_value(self.height, unit='m', precision=1)}",
                f"        Cross Section: {self.cross_section.name}",
            ]
        )

    def define(self) -> List[FibsemRectangleSettings]:

        shape = FibsemRectangleSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            rotation=self.rotation * constants.DEGREES_TO_RADIANS,
            passes=self.passes,
            time=self.time,
            scan_direction=self.scan_direction,
            cross_section=self.cross_section,
        )

        self.shapes = [shape]
        return self.shapes
