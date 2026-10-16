from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
    Union,
)

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_metadata,
    cross_section_metadata,
    distance_metadata,
    passes_metadata,
    scan_direction_metadata,
)
from fibsem.structures import (
    CrossSectionPattern,
    Display,
    FibsemCircleSettings,
    FibsemRectangleSettings,
    Parameter,
    Point,
    RangeLimit,
)


@dataclass
class ArrayPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = field(
        default=2.0e-6,
        metadata=distance_metadata("Width", "Width of the array pattern."),
    )
    height: float = field(
        default=2.0e-6,
        metadata=distance_metadata("Height", "Height of the array pattern."),
    )
    depth: float = field(
        default=5.0e-6,
        metadata=distance_metadata("Depth", "Depth of the array pattern."),
    )
    n_columns: int = field(
        default=5,
        metadata=Parameter(
            limits=RangeLimit(1, 100),
            doc="Number of columns in the array pattern.",
            display=Display("Number of Columns", step=1, decimals=0),
        ).meta(),
    )
    n_rows: int = field(
        default=5,
        metadata=Parameter(
            limits=RangeLimit(1, 100),
            doc="Number of rows in the array pattern.",
            display=Display("Number of Rows", step=1, decimals=0),
        ).meta(),
    )
    pitch_vertical: float = field(
        default=5.0e-6,
        metadata=distance_metadata(
            "Vertical Pitch", "Vertical pitch between elements in the array pattern."
        ),
    )
    pitch_horizontal: float = field(
        default=5.0e-6,
        metadata=distance_metadata(
            "Horizontal Pitch",
            "Horizontal pitch between elements in the array pattern.",
        ),
    )
    passes: int = field(
        default=0,
        metadata=passes_metadata(),
    )
    rotation: float = field(
        default=0,
        metadata=angle_metadata(
            "Rotation", "Rotation angle of the array pattern in degrees."
        ),
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=scan_direction_metadata()
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=cross_section_metadata()
    )
    use_circle: bool = field(
        default=False,
        metadata=Parameter(
            doc="If true, use circular patterns instead of rectangular.",
            display=Display("Use Circle"),
        ).meta(),
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
