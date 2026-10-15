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
    FibsemRectangleSettings,
)


@dataclass
class MicroExpansionPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=0.5e-6,
        metadata=distance_metadata(
            "Width", "Width of the microexpansion joint pattern."
        ),
    )
    height: float = field(
        default=15.0e-6,
        metadata=distance_metadata(
            "Height", "Height of the microexpansion joint pattern."
        ),
    )
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata(
            "Depth", "Depth of the microexpansion joint pattern."
        ),
    )
    distance: float = field(
        default=10.0e-6,
        metadata=distance_metadata(
            "Distance", "Distance between microexpansion joints."
        ),
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
