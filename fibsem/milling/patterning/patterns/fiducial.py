from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
)

import numpy as np

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_param,
    cross_section_param,
    distance_param,
)
from fibsem.structures import (
    CrossSectionPattern,
    Display,
    FibsemRectangleSettings,
    param,
)


@dataclass
class FiducialPattern(BasePattern[FibsemRectangleSettings]):
    width: float = param(1.0e-6, distance_param("Width", "Width of the fiducial mark."))
    height: float = param(
        10.0e-6, distance_param("Height", "Height of the fiducial mark.")
    )
    depth: float = param(1.0e-6, distance_param("Depth", "Depth of the fiducial mark."))
    rotation: float = param(
        45.0, angle_param("Rotation", "Rotation angle of the fiducial mark.")
    )
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
    )
    asymmetric: bool = param(
        False,
        doc="If true, make the fiducial mark y-shaped.",
        display=Display("Asymmetric", advanced=True),
    )
    name: ClassVar[str] = "Fiducial"

    def define(self) -> List[FibsemRectangleSettings]:
        """Draw a fiducial milling pattern (cross shape)"""

        width = self.width
        height = self.height
        depth = self.depth
        rotation = self.rotation * constants.DEGREES_TO_RADIANS
        cross_section = self.cross_section

        # def calculate_angles(num_steps, start_angle=0):
        #     angles = []
        #     step_size = 180 / num_steps
        #     for i in range(num_steps):
        #         angle = (start_angle + i * step_size) % 180
        #         angles.append(angle)
        #     return angles

        # angles = calculate_angles(self.num_steps, start_angle=self.rotation)
        # print(angles)
        # shapes = []
        # for i, angle in enumerate(angles):
        #     rotation = angle * constants.DEGREES_TO_RADIANS
        #     shape = FibsemRectangleSettings(
        #         width=width,
        #         height=height,
        #         depth=depth,
        #         centre_x=self.point.x,
        #         centre_y=self.point.y,
        #         scan_direction="TopToBottom",
        #         cross_section=cross_section,
        #         rotation=rotation,
        #     )
        #     if i >= 2:
        #         import random
        #         shape.height *= random.uniform(0.3, 0.7) # make it shorter
        #     shapes.append(shape)
        # self.shapes =  shapes
        # return shapes

        left_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation,
        )
        right_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + np.deg2rad(90),
        )
        if self.asymmetric:
            left_pattern.height *= 0.5
            left_pattern.centre_x -= left_pattern.height * 0.5 * np.cos(rotation)
            left_pattern.centre_y += left_pattern.height * 0.5 * np.sin(rotation)

        # a right-angle turn is the same rectangle with its dimensions swapped.
        # Saying it that way keeps the cross correct on backends whose scripting
        # API cannot rotate a shape at all.
        if np.isclose(rotation % np.pi, 0):
            right_pattern.width, right_pattern.height = height, width
            right_pattern.rotation = rotation

        self.shapes = [left_pattern, right_pattern]
        return self.shapes
