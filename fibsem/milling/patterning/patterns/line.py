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
from fibsem.structures import (
    FibsemLineSettings,
    RangeLimit,
    param,
)


@dataclass
class LinePattern(BasePattern[FibsemLineSettings]):
    start_x: float = param(
        -10.0e-6,
        distance_param(
            "Start X",
            "The x coordinate of the line's start.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    end_x: float = param(
        10.0e-6,
        distance_param(
            "End X",
            "The x coordinate of the line's end.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    start_y: float = param(
        0.0,
        distance_param(
            "Start Y",
            "The y coordinate of the line's start.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    end_y: float = param(
        0.0,
        distance_param(
            "End Y",
            "The y coordinate of the line's end.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    depth: float = param(1.0e-6, distance_param("Depth", "Depth of the line pattern."))
    name: ClassVar[str] = "Line"

    def summary(self) -> str:
        from fibsem.utils import format_value

        return "\n".join(
            [
                f"    Pattern: {self.name}",
                f"        Depth: {format_value(self.depth, unit='m', precision=1)}",
            ]
        )

    def define(self) -> List[FibsemLineSettings]:
        shape = FibsemLineSettings(
            start_x=self.start_x,
            end_x=self.end_x,
            start_y=self.start_y,
            end_y=self.end_y,
            depth=self.depth,
        )
        self.shapes = [shape]
        return self.shapes
