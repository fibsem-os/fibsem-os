from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
    Union,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    DEFAULT_DISTANCE_METADATA,
)
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemRectangleSettings,
)


@dataclass
class CloverPattern(BasePattern[Union[FibsemCircleSettings, FibsemRectangleSettings]]):
    radius: float = field(
        default=10.0e-6,
        metadata={
            "label": "Radius",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Radius of the clover leaves.",
        },
    )
    depth: float = field(
        default=5.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the clover pattern.",
        },
    )

    name: ClassVar[str] = "Clover"

    def define(self) -> List[Union[FibsemCircleSettings, FibsemRectangleSettings]]:

        point = self.point
        radius = self.radius
        depth = self.depth

        # three leaf clover pattern

        top_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + radius,
        )

        right_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x + radius,
            centre_y=point.y,
        )

        left_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x - radius,
            centre_y=point.y,
        )

        stem_pattern = FibsemRectangleSettings(
            width=radius / 4,
            height=radius * 2,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - radius,
            scan_direction="TopToBottom",
        )

        self.shapes = [top_pattern, right_pattern, left_pattern, stem_pattern]
        return self.shapes
