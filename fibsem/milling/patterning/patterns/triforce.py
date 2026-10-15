from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

import numpy as np

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_metadata,
)
from fibsem.structures import (
    FibsemRectangleSettings,
    Point,
)


@dataclass
class TriForcePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Width", "Width of the triangles."),
    )
    height: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Height", "Height of the triangles."),
    )
    depth: float = field(
        default=5.0e-6,
        metadata=distance_metadata("Depth", "Depth of the triangles."),
    )

    name: ClassVar[str] = "TriForce"

    def define(self) -> List[FibsemRectangleSettings]:
        point = self.point
        height = self.height
        width = self.width
        depth = self.depth
        angle = 30

        self.shapes = []

        # centre of each triangle
        points = [
            Point(point.x, point.y + height),
            Point(point.x - height / 2, point.y),
            Point(point.x + height / 2, point.y),
        ]

        for point in points:
            triangle_shapes = create_triangle_patterns(
                width=width, height=height, depth=depth, point=point, angle=angle
            )
            self.shapes.extend(triangle_shapes)

        return self.shapes


def create_triangle_patterns(
    width: float, height: float, depth: float, angle: float = 30, point: Point = Point()
) -> List[FibsemRectangleSettings]:
    h_offset = height / 2 * np.sin(np.deg2rad(angle))

    left_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(-angle),
        centre_x=point.x - h_offset,
        centre_y=point.y,
        scan_direction="LeftToRight",
    )

    right_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(angle),
        centre_x=point.x + h_offset,
        centre_y=point.y,
        scan_direction="RightToLeft",
    )

    bottom_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(90),
        centre_x=point.x,
        centre_y=point.y - height / 2,
        scan_direction="BottomToTop",
    )

    return [left_pattern, right_pattern, bottom_pattern]
