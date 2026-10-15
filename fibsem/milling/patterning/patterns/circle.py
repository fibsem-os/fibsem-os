from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_metadata,
)
from fibsem.structures import (
    FibsemCircleSettings,
    RangeLimit,
)


@dataclass
class CirclePattern(BasePattern[FibsemCircleSettings]):
    radius: float = field(
        default=5.0e-6,
        metadata=distance_metadata("Radius", "Radius of the circle pattern."),
    )
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Depth", "Depth of the circle pattern."),
    )
    thickness: float = field(
        default=0,
        metadata=distance_metadata(
            "Thickness",
            "Thickness of the circle pattern. 0 means solid.",
            limits=RangeLimit(0.0e-6, 1000.0e-6),
        ),
    )

    name: ClassVar[str] = "Circle"

    def summary(self) -> str:
        from fibsem.utils import format_value

        return "\n".join(
            [
                f"    Pattern: {self.name}",
                f"        Radius: {format_value(self.radius, unit='m', precision=1)}",
                f"        Depth: {format_value(self.depth, unit='m', precision=1)}",
            ]
        )

    def define(self) -> List[FibsemCircleSettings]:

        shape = FibsemCircleSettings(
            radius=self.radius,
            depth=self.depth,
            thickness=self.thickness,
            centre_x=self.point.x,
            centre_y=self.point.y,
        )
        self.shapes = [shape]
        return self.shapes
