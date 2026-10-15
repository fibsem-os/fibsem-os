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
    FibsemLineSettings,
    RangeLimit,
)


@dataclass
class LinePattern(BasePattern[FibsemLineSettings]):
    start_x: float = field(
        default=-10.0e-6,
        metadata=distance_metadata(
            "Start X",
            "The x coordinate of the line's start.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    end_x: float = field(
        default=10.0e-6,
        metadata=distance_metadata(
            "End X",
            "The x coordinate of the line's end.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    start_y: float = field(
        default=0.0,
        metadata=distance_metadata(
            "Start Y",
            "The y coordinate of the line's start.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    end_y: float = field(
        default=0.0,
        metadata=distance_metadata(
            "End Y",
            "The y coordinate of the line's end.",
            limits=RangeLimit(-1000.0e-6, 1000.0e-6),
        ),
    )
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Depth", "Depth of the line pattern."),
    )
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
