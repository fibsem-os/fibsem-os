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
    FibsemLineSettings,
)


@dataclass
class LinePattern(BasePattern[FibsemLineSettings]):
    start_x: float = field(
        default=-10.0e-6,
        metadata={
            "label": "Start X",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": -1000.0,
            "maximum": 1000.0,
        },
    )
    end_x: float = field(
        default=10.0e-6,
        metadata={
            "label": "End X",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": -1000.0,
            "maximum": 1000.0,
        },
    )
    start_y: float = field(
        default=0.0,
        metadata={
            "label": "Start Y",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": -1000.0,
            "maximum": 1000.0,
        },
    )
    end_y: float = field(
        default=0.0,
        metadata={
            "label": "End Y",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": -1000.0,
            "maximum": 1000.0,
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the line pattern.",
        },
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
