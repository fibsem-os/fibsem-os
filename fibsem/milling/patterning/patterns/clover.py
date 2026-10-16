from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
    Union,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_param,
)
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemRectangleSettings,
    param,
)


@dataclass
class CloverPattern(BasePattern[Union[FibsemCircleSettings, FibsemRectangleSettings]]):
    radius: float = param(
        10.0e-6, distance_param("Radius", "Radius of the clover leaves.")
    )
    depth: float = param(
        5.0e-6, distance_param("Depth", "Depth of the clover pattern.")
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
