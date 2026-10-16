from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_param,
    cross_section_param,
    distance_param,
    duration_param,
    passes_param,
)
from fibsem.structures import (
    CrossSectionPattern,
    Display,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    RangeLimit,
    param,
)


@dataclass
class TrenchPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = param(
        10.0e-6, distance_param("Width", "Width of the trench pattern.")
    )
    depth: float = param(
        1.0e-6, distance_param("Depth", "Depth of the trench pattern.")
    )
    spacing: float = param(
        5.0e-6,
        distance_param("Spacing", "Spacing between the upper and lower trenches."),
    )
    upper_trench_height: float = param(
        5.0e-6, distance_param("Upper Trench Height", "Height of the upper trench.")
    )
    lower_trench_height: float = param(
        5.0e-6, distance_param("Lower Trench Height", "Height of the lower trench.")
    )
    passes: int = param(0, passes_param())
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
    )
    time: float = param(0.0, duration_param())
    fillet: float = param(
        0.0,
        distance_param(
            "Fillet Radius",
            "Fillet radius for the trench corners.",
            limits=RangeLimit(0.0e-6, 1000.0e-6),
            advanced=True,
        ),
    )

    name: ClassVar[str] = "Trench"

    def summary(self) -> str:
        from fibsem.utils import format_value

        return "\n".join(
            [
                f"    Pattern: {self.name}",
                f"        Depth: {format_value(self.depth, unit='m', precision=1)}",
                f"        Width: {format_value(self.width, unit='m', precision=1)}",
                f"        Spacing: {format_value(self.spacing, unit='m', precision=1)}",
                f"        Upper Trench Height: {format_value(self.upper_trench_height, unit='m', precision=1)}",
                f"        Lower Trench Height: {format_value(self.lower_trench_height, unit='m', precision=1)}",
                f"        Passes: {format_value(self.passes, precision=0)}",
                f"        Cross Section: {self.cross_section.name}",
            ]
        )

    def define(self) -> List[Union[FibsemRectangleSettings, FibsemCircleSettings]]:

        point = self.point
        width = self.width
        spacing = self.spacing
        upper_trench_height = self.upper_trench_height
        lower_trench_height = self.lower_trench_height
        depth = self.depth
        cross_section = self.cross_section
        time = self.time
        fillet = self.fillet

        # calculate the centre of the upper and lower trench
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)

        # fillet radius on the corners
        fillet = np.clip(fillet, 0, upper_trench_height / 2)
        if fillet > 0:
            width = max(0, width - 2 * fillet)  # ensure width is not negative

        # mill settings
        lower_trench_settings = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section=cross_section,
            passes=self.passes,
            time=time,
        )

        upper_trench_settings = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            passes=self.passes,
            time=time,
        )

        self.shapes = [upper_trench_settings, lower_trench_settings]

        # add fillet to the corners
        if fillet > 0:
            left_x_pos = point.x - width / 2
            right_x_pos = point.x + width / 2

            fillet_offset = 1.5
            lower_y_pos = (
                centre_lower_y + lower_trench_height / 2 - fillet * fillet_offset
            )
            top_y_pos = (
                centre_upper_y - upper_trench_height / 2 + fillet * fillet_offset
            )

            lower_left_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth / 2,
                centre_x=point.x - width / 2,
                centre_y=lower_y_pos,
            )
            lower_right_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth / 2,
                centre_x=point.x + width / 2,
                centre_y=lower_y_pos,
            )

            # fill the remaining space with rectangles
            lower_left_fill = FibsemRectangleSettings(
                width=fillet,
                height=lower_trench_height - fillet,
                depth=depth,
                centre_x=left_x_pos - fillet / 2,
                centre_y=centre_lower_y - fillet / 2,
                cross_section=cross_section,
                scan_direction="BottomToTop",
                passes=self.passes,
            )
            lower_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=lower_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_lower_y - fillet / 2,
                cross_section=cross_section,
                scan_direction="BottomToTop",
                passes=self.passes,
            )

            top_left_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth,
                centre_x=point.x - width / 2,
                centre_y=top_y_pos,
            )
            top_right_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth,
                centre_x=point.x + width / 2,
                centre_y=top_y_pos,
            )

            top_left_fill = FibsemRectangleSettings(
                width=fillet,
                height=upper_trench_height - fillet,
                depth=depth,
                centre_x=left_x_pos - fillet / 2,
                centre_y=centre_upper_y + fillet / 2,
                cross_section=cross_section,
                scan_direction="TopToBottom",
                passes=self.passes,
            )
            top_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=upper_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_upper_y + fillet / 2,
                cross_section=cross_section,
                scan_direction="TopToBottom",
                passes=self.passes,
            )

            self.shapes.extend(
                [
                    lower_left_fill,
                    lower_right_fill,
                    top_left_fill,
                    top_right_fill,
                    lower_left_fillet,
                    lower_right_fillet,
                    top_left_fillet,
                    top_right_fillet,
                ]
            )

        return self.shapes


@dataclass
class TrenchBitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = param(
        10.0e-6, distance_param("Width", "Width of the trench bitmap pattern.")
    )
    depth: float = param(
        10.0e-6, distance_param("Depth", "Depth of the trench bitmap pattern.")
    )
    spacing: float = param(
        1.0e-6,
        distance_param("Spacing", "Spacing between the upper and lower trenches."),
    )
    upper_trench_height: float = param(
        5.0e-6, distance_param("Upper Trench Height", "Height of the upper trench.")
    )
    lower_trench_height: float = param(
        5.0e-6, distance_param("Lower Trench Height", "Height of the lower trench.")
    )
    passes: int = param(0, passes_param())
    time: float = param(
        0,
        duration_param(
            "Time to mill the trench bitmap pattern. Set to 0 for automatic calculation.",
            label="Time",
        ),
    )
    path: str = param(
        "",
        doc="Path to the upper trench bitmap image file.",
        filepath=True,
        display=Display("File Path"),
    )
    path_lower: str = param(
        "",
        doc="Path to the lower trench bitmap image file. If not provided, the upper trench bitmap will be flipped and used.",
        filepath=True,
        display=Display("File Path Lower"),
    )
    array: Optional[NDArray[Any]] = param(None, display=Display(hidden=True))
    array_lower: Optional[NDArray[Any]] = param(None, display=Display(hidden=True))
    interpolation: Optional[Literal["nearest", "bicubic", "bilinear"]] = param(
        None,
        choices=[None, "nearest", "bilinear", "bicubic"],
        doc="Interpolation method for resizing the bitmap image.",
        display=Display("Interpolation", advanced=True),
    )
    name: ClassVar[str] = "TrenchBitmap"

    def define(self) -> List[FibsemBitmapSettings]:
        path: Optional[str] = self.path.strip()
        if not path:
            path = None
        array = self.array

        # calculate the centre of the upper and lower trench
        centre_lower_y = self.point.y - (
            self.spacing / 2 + self.lower_trench_height / 2
        )
        centre_upper_y = self.point.y + (
            self.spacing / 2 + self.upper_trench_height / 2
        )

        flip_lower_y = False

        array_lower = self.array_lower
        path_lower = self.path_lower.strip()

        if not path_lower:
            path_lower = None

        if array_lower is None:
            array_lower = None
            if path_lower is None:
                path_lower = None
                # Fallback on upper bitmap/path
                flip_lower_y = True
                path_lower = path
                array_lower = array

        # mill settings
        lower_pattern_settings = FibsemBitmapSettings(
            width=self.width,
            height=self.lower_trench_height,
            depth=self.depth,
            rotation=0,
            centre_x=self.point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            passes=self.passes,
            time=self.time,
            flip_y=flip_lower_y,
            path=path_lower,
            array=array_lower,
            interpolate=self.interpolation,
        )

        upper_pattern_settings = FibsemBitmapSettings(
            width=self.width,
            height=self.upper_trench_height,
            depth=self.depth,
            rotation=0,
            centre_x=self.point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            passes=self.passes,
            time=self.time,
            path=path,
            array=array,
            interpolate=self.interpolation,
        )

        self.shapes = [lower_pattern_settings, upper_pattern_settings]
        return self.shapes


# QUERY: we should consolidate this with TrenchPattern?
@dataclass
class TrenchTrapezoidPattern(BasePattern):
    trench_width: float = param(
        10e-6, distance_param("Trench Width", "Width of the trench at the top.")
    )
    trench_height: float = param(
        5e-6, distance_param("Trench Height", "Height of the trench.")
    )
    spacing: float = param(
        5.0e-6, distance_param("Spacing", "Spacing between trenches.")
    )
    depth: float = param(1.0e-6, distance_param("Depth", "Depth of the trench."))
    angle: float = param(60, angle_param("Angle", "Angle of the trench in degrees."))
    name: ClassVar[str] = "TrapezoidTrench"

    # ref: https://www.researchsquare.com/article/rs-6497420/v1

    def define(self):

        width = self.trench_width
        height = self.trench_height
        spacing = self.spacing
        angle = -self.angle  # angle in degrees
        depth = self.depth

        # define trapezoid polygon points
        p1 = (-width / 2, 0)  # xy
        p2 = (width / 2, 0)  # xy
        p3 = (width / 2 - height * np.tan(np.radians(angle)), height)  # xy
        p4 = (-width / 2 + height * np.tan(np.radians(angle)), height)  # xy

        points = np.array([p1, p2, p3, p4])

        # offset all points in y by spacing / 2
        points[:, 1] += spacing / 2

        # create mirrored polygon (mirror over y-axis)
        mirrored_points = points.copy()
        mirrored_points[:, 1] = -mirrored_points[:, 1]  # - spacing /2

        top_trench = FibsemPolygonSettings(vertices=points, depth=depth)
        bottom_trench = FibsemPolygonSettings(vertices=mirrored_points, depth=depth)

        # apply the verticies offset by the point
        top_trench.vertices[:, 0] += self.point.x
        top_trench.vertices[:, 1] -= self.point.y
        bottom_trench.vertices[:, 0] += self.point.x
        bottom_trench.vertices[:, 1] -= self.point.y

        self.shapes = [top_trench, bottom_trench]
        return self.shapes
