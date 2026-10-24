from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
    Union,
)

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_param,
    cross_section_param,
    distance_param,
    passes_param,
    scan_direction_param,
)
from fibsem.structures.params import Display, RangeLimit, param
from fibsem.structures.settings import (
    CrossSectionPattern,
    FibsemCircleSettings,
    FibsemRectangleSettings,
    Point,
)


@dataclass
class ArrayPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = param(2.0e-6, distance_param("Width", "Width of the array pattern."))
    height: float = param(
        2.0e-6, distance_param("Height", "Height of the array pattern.")
    )
    depth: float = param(5.0e-6, distance_param("Depth", "Depth of the array pattern."))
    n_columns: int = param(
        5,
        limits=RangeLimit(1, 100),
        doc="Number of columns in the array pattern.",
        display=Display("Number of Columns", step=1, decimals=0),
    )
    n_rows: int = param(
        5,
        limits=RangeLimit(1, 100),
        doc="Number of rows in the array pattern.",
        display=Display("Number of Rows", step=1, decimals=0),
    )
    pitch_vertical: float = param(
        5.0e-6,
        distance_param(
            "Vertical Pitch", "Vertical pitch between elements in the array pattern."
        ),
    )
    pitch_horizontal: float = param(
        5.0e-6,
        distance_param(
            "Horizontal Pitch",
            "Horizontal pitch between elements in the array pattern.",
        ),
    )
    passes: int = param(0, passes_param())
    rotation: float = param(
        0, angle_param("Rotation", "Rotation angle of the array pattern in degrees.")
    )
    scan_direction: str = param("TopToBottom", scan_direction_param())
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
    )
    use_circle: bool = param(
        False,
        doc="If true, use circular patterns instead of rectangular.",
        display=Display("Use Circle"),
    )

    name: ClassVar[str] = "ArrayPattern"
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
    # ref: weld cross-section/ passes: https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[Union[FibsemRectangleSettings, FibsemCircleSettings]]:

        point = self.point
        width = self.width
        height = self.height
        depth = self.depth
        n_columns = int(self.n_columns)
        n_rows = int(self.n_rows)
        pitch_vertical = self.pitch_vertical
        pitch_horizontal = self.pitch_horizontal
        rotation = self.rotation * constants.DEGREES_TO_RADIANS
        passes = self.passes
        scan_direction = self.scan_direction
        passes = int(passes)
        cross_section = self.cross_section

        # create a 2D array of points
        points: List[Point] = []
        for i in range(n_columns):
            for j in range(n_rows):
                points.append(
                    Point(
                        point.x + (i - (n_columns - 1) / 2) * pitch_horizontal,
                        point.y + (j - (n_rows - 1) / 2) * pitch_vertical,
                    )
                )
        # create patterns
        self.shapes = []
        for pt in points:
            if self.use_circle:
                pattern_settings = FibsemCircleSettings(
                    radius=width / 2,
                    depth=depth,
                    centre_x=pt.x,
                    centre_y=pt.y,
                )
            else:
                pattern_settings = FibsemRectangleSettings(
                    width=width,
                    height=height,
                    depth=depth,
                    centre_x=pt.x,
                    centre_y=pt.y,
                    scan_direction=scan_direction,
                    rotation=rotation,
                    passes=passes,
                    cross_section=cross_section,
                )
            self.shapes.append(pattern_settings)

        return self.shapes  # type: ignore
