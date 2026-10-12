from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    DEFAULT_DISTANCE_METADATA,
)
from fibsem.structures import (
    FibsemCircleSettings,
)


@dataclass
class CirclePattern(BasePattern[FibsemCircleSettings]):
    radius: float = field(
        default=5.0e-6,
        metadata={
            "label": "Radius",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Radius of the circle pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the circle pattern.",
        },
    )
    thickness: float = field(
        default=0,
        metadata={
            "label": "Thickness",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": 0.0,
            "tooltip": "Thickness of the circle pattern. 0 means solid.",
        },
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
