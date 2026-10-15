from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_metadata,
    cross_section_metadata,
    distance_metadata,
    duration_metadata,
    passes_metadata,
    scan_direction_metadata,
)
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangleSettings,
)


@dataclass
class RectanglePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Width", "Width of the rectangle pattern."),
    )
    height: float = field(
        default=5.0e-6,
        metadata=distance_metadata("Height", "Height of the rectangle pattern."),
    )
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Depth", "Depth of the rectangle pattern."),
    )
    rotation: float = field(
        default=0.0,
        metadata=angle_metadata(
            "Rotation", "Rotation of the rectangle pattern in degrees.", advanced=True
        ),
    )

    time: float = field(default=0.0, metadata=duration_metadata())
    passes: int = field(default=0, metadata=passes_metadata())
    scan_direction: str = field(
        default="TopToBottom",
        metadata=scan_direction_metadata(),
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle,
        metadata=cross_section_metadata(),
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
