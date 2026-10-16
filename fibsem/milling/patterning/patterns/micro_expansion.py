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
    FibsemRectangleSettings,
    param,
)


@dataclass
class MicroExpansionPattern(BasePattern[FibsemRectangleSettings]):
    width: float = param(
        0.5e-6, distance_param("Width", "Width of the microexpansion joint pattern.")
    )
    height: float = param(
        15.0e-6, distance_param("Height", "Height of the microexpansion joint pattern.")
    )
    depth: float = param(
        1.0e-6, distance_param("Depth", "Depth of the microexpansion joint pattern.")
    )
    distance: float = param(
        10.0e-6, distance_param("Distance", "Distance between microexpansion joints.")
    )

    name: ClassVar[str] = "MicroExpansion"
    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(self) -> List[FibsemRectangleSettings]:
        """Draw the microexpansion joints for stress relief of lamella"""

        point = self.point
        width = self.width
        height = self.height
        depth = self.depth
        distance = self.distance

        left_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x - distance,
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        right_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x + distance,
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        self.shapes = [left_pattern_settings, right_pattern_settings]
        return self.shapes
