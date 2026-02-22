from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields, asdict
from functools import cached_property
from typing import Dict, List, Tuple, Union, Any, Optional, Type, ClassVar, TypeVar, Generic, Literal

import numpy as np
from numpy.typing import NDArray

from fibsem import constants
from fibsem.structures import (
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemRectangleSettings,
    FibsemPolygonSettings,
    Point,
    TFibsemPatternSettings,
    get_fields_with_metadata,
)
from fibsem.milling.properties import (DEFAULT_DISTANCE_METADATA,
                                       DEFAULT_ANGLE_METADATA,
                                       DEFAULT_DURATION_METADATA,
                                       DEFAULT_SCAN_DIRECTION_METADATA,
                                       DEFAULT_CROSS_SECTION_METADATA,
                                       DEFAULT_PASSES_METADATA)

TPattern = TypeVar("TPattern", bound="BasePattern")

####### Combo Patterns

@dataclass
class BasePattern(ABC, Generic[TFibsemPatternSettings]):
    name: ClassVar[str] = field(init=False)
    point: Point = field(default_factory=Point, 
                         metadata={
                            "label": "Point",
                            "type": Point,
                            **DEFAULT_DISTANCE_METADATA,
                            "minimum": -1000.0,
                            "maximum": 1000.0,
                            "tooltip": "Point coordinates for the milling pattern.",
                         })
    shapes: Optional[List[TFibsemPatternSettings]] = field(default=None, init=False)

    @abstractmethod
    def define(self) -> List[TFibsemPatternSettings]:
        pass

    def to_dict(self) -> Dict[str, Any]:
        ddict = asdict(self)
        # Handle any special cases
        if "cross_section" in ddict:
            ddict["cross_section"] = ddict["cross_section"].name
        if "vertices" in ddict:
            ddict["vertices"] = ddict["vertices"].tolist()
        ddict["name"] = self.name
        del ddict["shapes"]
        return ddict

    @classmethod
    def from_dict(cls: Type[TPattern], ddict: Dict[str, Any]) -> TPattern:
        kwargs = {}
        for f in fields(cls):
            if f.name in ddict:
                kwargs[f.name] = ddict[f.name]

        # Construct objects
        point = kwargs.pop("point", None)
        if point is not None:
            kwargs["point"] = Point.from_dict(point)

        cross_section = kwargs.pop("cross_section", None)
        if cross_section is not None:
            kwargs["cross_section"] = CrossSectionPattern[cross_section]

        vertices = kwargs.pop("vertices", None)
        if vertices is not None:
            kwargs["vertices"] = np.array(vertices)

        return cls(**kwargs)

    @cached_property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(
            f.name
            for f in fields(self)
            if f.name not in self._hidden_attributes and f not in fields(BasePattern)
        )

    @cached_property
    def advanced_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are marked as advanced in the metadata."""
        return tuple(f.name for f in fields(self) if f.metadata.get("advanced", False))

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return tuple(f.name for f in fields(self) if f.metadata.get("hidden", False))

    @property
    def volume(self) -> float:
        # calculate the total volume of the milling pattern (sum of all shapes)
        return sum([shape.volume for shape in self.define()])

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return dataclass fields with metadata, filling any missing keys with defaults."""
        return get_fields_with_metadata(self.__class__)


@dataclass
class BitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = field(
        default=10.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the bitmap pattern.",
        },
    )
    height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the bitmap pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the bitmap pattern.",
        },
    )
    rotation: float = field(
        default=0,
        metadata={
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Rotation of the bitmap pattern in degrees.",
        },
    )
    time: float = field(default=0, metadata=DEFAULT_DURATION_METADATA)
    passes: int = field(default=0, metadata=DEFAULT_PASSES_METADATA)
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    path: str = field(
        default="",
        metadata={
            "label": "File Path",
            "type": str,
            "filepath": True,
            "tooltip": "Path to the bitmap image file.",
        },
    )
    array: Optional[NDArray[Any]] = field(default=None, metadata={"hidden": True})
    interpolation: Optional[Literal["nearest", "bilinear", "bicubic"]] = field(
        default=None,
        metadata={
            "label": "Interpolation",
            "advanced": True,
            "items": [None, "nearest", "bilinear", "bicubic"],
            "tooltip": "Interpolation method for resizing the bitmap image.",
        },
    )

    name: ClassVar[str] = "Bitmap"

    def define(self) -> List[FibsemBitmapSettings]:
        path: Optional[str] = self.path.strip()
        if not path:
            path = None
        array = self.array

        shape = FibsemBitmapSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            rotation=self.rotation * constants.DEGREES_TO_RADIANS,
            scan_direction=self.scan_direction,
            passes=self.passes,
            time=self.time,
            path=path,
            array=array,
            interpolate=self.interpolation,
        )
        self.shapes = [shape]
        return self.shapes


@dataclass
class TrenchBitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = field(
        default=10.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the trench bitmap pattern.",
        },
    )
    depth: float = field(
        default=10.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trench bitmap pattern.",
        },
    )
    spacing: float = field(
        default=1.0e-6,
        metadata={
            "label": "Spacing",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Spacing between the upper and lower trenches.",
        },
    )
    upper_trench_height: float = field(
        default=5.0e-6,
        metadata={
            "label": "Upper Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the upper trench.",
        },
    )
    lower_trench_height: float = field(
        default=5.0e-6,
        metadata={
            "label": "Lower Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the lower trench.",
        },
    )
    time: float = field(
        default=0,
        metadata={
            "label": "Time",
            **DEFAULT_DURATION_METADATA,
            "tooltip": "Time to mill the trench bitmap pattern. Set to 0 for automatic calculation.",
        },
    )
    path: str = field(
        default="",
        metadata={
            "label": "File Path",
            "filepath": True,
            "tooltip": "Path to the upper trench bitmap image file.",
        },
    )
    path_lower: str = field(
        default="",
        metadata={
            "label": "File Path Lower",
            "filepath": True,
            "tooltip": "Path to the lower trench bitmap image file. If not provided, the upper trench bitmap will be flipped and used.",
        },
    )
    array: Optional[NDArray[Any]] = field(default=None, metadata={"hidden": True})
    array_lower: Optional[NDArray[Any]] = field(default=None, metadata={"hidden": True})
    interpolation: Optional[Literal["nearest", "bicubic", "bilinear"]] = field(
        default=None,
        metadata={
            "label": "Interpolation",
            "advanced": True,
            "items": [None, "nearest", "bilinear", "bicubic"],
            "tooltip": "Interpolation method for resizing the bitmap image.",
        },
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
            time=self.time,
            path=path,
            array=array,
            interpolate=self.interpolation,
        )

        self.shapes = [lower_pattern_settings, upper_pattern_settings]
        return self.shapes


@dataclass
class RectanglePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=10.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the rectangle pattern.",
        },
    )
    height: float = field(
        default=5.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the rectangle pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the rectangle pattern.",
        },
    )
    rotation: float = field(
        default=0.0,
        metadata={
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Rotation of the rectangle pattern in degrees.",
        },
    )

    time: float = field(default=0.0, metadata=DEFAULT_DURATION_METADATA)
    passes: int = field(default=0, metadata=DEFAULT_PASSES_METADATA)
    scan_direction: str = field(
        default="TopToBottom",
        metadata=DEFAULT_SCAN_DIRECTION_METADATA,
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle,
        metadata=DEFAULT_CROSS_SECTION_METADATA,
    )
    name: ClassVar[str] = "Rectangle"

    def define(self) -> List[FibsemRectangleSettings]:

        shape = FibsemRectangleSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            rotation=self.rotation * constants.DEGREES_TO_RADIANS,
            passes=self.passes,
            time=self.time,
            scan_direction=self.scan_direction,
            cross_section=self.cross_section,
        )

        self.shapes = [shape]
        return self.shapes


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


@dataclass
class CirclePattern(BasePattern[FibsemCircleSettings]):
    radius: float = field(
        default=5.0e-6,
        metadata={
            "label": "Radius",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Radius of the circle pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the circle pattern.",
        },
    )
    thickness: float = field(
        default=0,
        metadata={
            "label": "Thickness",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": 0.0,
            "tooltip": "Thickness of the circle pattern. 0 means solid.",
        },
    )

    name: ClassVar[str] = "Circle"

    def define(self) -> List[FibsemCircleSettings]:
        
        shape = FibsemCircleSettings(
            radius=self.radius,
            depth=self.depth,
            thickness=self.thickness,
            centre_x=self.point.x,
            centre_y=self.point.y,
        )
        self.shapes = [shape]
        return self.shapes


@dataclass
class TrenchPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = field(
        default=10.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the trench pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trench pattern.",
        },
    )
    spacing: float = field(
        default=5.0e-6,
        metadata={
            "label": "Spacing",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Spacing between the upper and lower trenches.",
        },
    )
    upper_trench_height: float = field(
        default=5.0e-6,
        metadata={
            "label": "Upper Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the upper trench.",
        },
    )
    lower_trench_height: float = field(
        default=5.0e-6,
        metadata={
            "label": "Lower Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the lower trench.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )
    time: float = field(default=0.0, metadata=DEFAULT_DURATION_METADATA)
    fillet: float = field(
        default=0.0,
        metadata={
            "label": "Fillet Radius",
            **DEFAULT_DISTANCE_METADATA,
            "minimum": 0.0,
            "advanced": True,
            "tooltip": "Fillet radius for the trench corners.",
        },
    )

    name: ClassVar[str] = "Trench"

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
            width = max(0, width - 2 * fillet) # ensure width is not negative

        # mill settings
        lower_trench_settings = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section = cross_section,
            time = time
        )

        upper_trench_settings = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section = cross_section,
            time = time
        )

        self.shapes = [upper_trench_settings, lower_trench_settings]

        # add fillet to the corners
        if fillet > 0:            
            left_x_pos = point.x - width / 2
            right_x_pos = point.x + width / 2

            fillet_offset = 1.5
            lower_y_pos = centre_lower_y + lower_trench_height / 2 - fillet * fillet_offset
            top_y_pos = centre_upper_y - upper_trench_height / 2 + fillet * fillet_offset

            lower_left_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth/2,
                centre_x=point.x - width / 2,
                centre_y=lower_y_pos,
            )
            lower_right_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth/2,
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
                cross_section = cross_section,
                scan_direction="BottomToTop",

            )
            lower_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=lower_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_lower_y - fillet / 2,
                cross_section = cross_section,
                scan_direction="BottomToTop",
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
                cross_section = cross_section,
                scan_direction="TopToBottom",
            )
            top_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=upper_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_upper_y + fillet / 2,
                cross_section = cross_section,
                scan_direction="TopToBottom",
            )

            self.shapes.extend([lower_left_fill, lower_right_fill, 
                                top_left_fill, top_right_fill, 
                                lower_left_fillet, lower_right_fillet, 
                                top_left_fillet, top_right_fillet])

        return self.shapes


@dataclass
class HorseshoePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=40.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horseshoe pattern.",
        },
    )
    upper_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Upper Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the upper trench.",
        },
    )
    lower_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Lower Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the lower trench.",
        },
    )
    spacing: float = field(
        default=10.0e-6,
        metadata={
            "label": "Spacing",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Spacing between trenches.",
        },
    )
    depth: float = field(
        default=10.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trenches.",
        },
    )
    side_width: float = field(
        default=5.0e-6,
        metadata={
            "label": "Side Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the side trench.",
        },
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "type": bool,
            "tooltip": "If true, the side trench will be milled on the opposite side",
        },
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "Horseshoe"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the trench milling patterns"""

        point = self.point
        width = self.width
        depth = self.depth
        spacing = self.spacing
        lower_trench_height = self.lower_trench_height
        upper_trench_height = self.upper_trench_height
        cross_section = self.cross_section
        side_width = self.side_width

        # calculate the centre of the upper and lower trench
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)

        # calculate the centre of the side trench
        side_height = spacing + upper_trench_height + lower_trench_height
        side_offset = (width / 2) + (side_width / 2)
        if self.inverted:
            side_offset = -side_offset
        side_x = point.x + side_offset
        # to account for assymetric trench heights
        side_y = point.y + (upper_trench_height - lower_trench_height) / 2

        lower_pattern = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section = cross_section
        )

        upper_pattern = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section = cross_section
        )

        side_pattern = FibsemRectangleSettings(
            width=side_width,
            height=side_height,
            depth=depth,
            centre_x=side_x,
            centre_y=side_y,
            scan_direction=self.scan_direction,
            cross_section=cross_section
        )

        self.shapes = [upper_pattern, lower_pattern, side_pattern]
        return self.shapes


@dataclass
class HorseshoePatternVertical(BasePattern):
    width: float = field(
        default=2.0e-05,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horseshoe vertical pattern.",
        },
    )
    height: float = field(
        default=5.0e-05,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the horseshoe vertical pattern.",
        },
    )
    side_trench_width: float = field(
        default=5.0e-06,
        metadata={
            "label": "Side Trench Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the side trenches.",
        },
    )
    top_trench_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Top Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the top trench.",
        },
    )
    depth: float = field(
        default=4.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trenches.",
        },
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "type": bool,
            "tooltip": "If true, the top trench will be milled below the point.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "HorseshoeVertical"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the horseshoe vertical milling patterns"""

        point = self.point
        width = self.width
        height = self.height
        trench_width = self.side_trench_width
        upper_trench_height = self.top_trench_height
        depth = self.depth
        scan_direction = self.scan_direction
        inverted = self.inverted
        cross_section = self.cross_section

        left_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x - (width / 2) - (trench_width / 2),
            centre_y=point.y,
            scan_direction="LeftToRight",
            cross_section=cross_section
        )

        right_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x + (width / 2) + (trench_width / 2),
            centre_y=point.y,
            scan_direction="RightToLeft",
            cross_section=cross_section
        )
        y_offset = (height / 2) + (upper_trench_height / 2)
        if inverted:
            y_offset = -y_offset
        upper_pattern = FibsemRectangleSettings(
            width=width + (2 * trench_width),
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + y_offset,
            scan_direction=scan_direction,
            cross_section=cross_section
        )

        self.shapes = [upper_pattern, left_pattern,right_pattern]
        return self.shapes


@dataclass
class SerialSectionPattern(BasePattern[FibsemLineSettings]):
    section_thickness: float = field(
        default=4.0e-6,
        metadata={
            "label": "Section Thickness",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Thickness of the section.",
        },
    )
    section_width: float = field(
        default=50.0e-6,
        metadata={
            "label": "Section Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the section.",
        },
    )
    section_depth: float = field(
        default=20.0e-6,
        metadata={
            "label": "Section Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the section.",
        },
    )
    side_width: float = field(
        default=10.0e-6,
        metadata={
            "label": "Side Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the side cleaning area.",
        },
    )
    side_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Side Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the side cleaning area.",
        },
    )
    side_depth: float = field(
        default=40.0e-6,
        metadata={
            "label": "Side Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the side cleaning area.",
        },
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "type": bool,
            "tooltip": "If true, the section will be milled inverted.",
        },
    )
    use_side_patterns: bool = field(
        default=True,
        metadata={
            "label": "Use Side Patterns",
            "type": bool,
            "tooltip": "If true, side cleaning patterns will be used.",
        },
    )

    name: ClassVar[str] = "SerialSection"
    # ref: "serial-liftout section" https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[FibsemLineSettings]:
        """Calculate the serial liftout sectioning milling patterns"""

        point = self.point
        section_thickness = self.section_thickness
        section_width = self.section_width
        section_depth = self.section_depth
        side_width = self.side_width
        side_height = self.side_height
        side_depth = self.side_depth
        inverted = self.inverted
        use_side_patterns = self.use_side_patterns

        # draw a line of section width
        section_y = section_thickness
        if inverted:
            section_y *= -1.0
            side_height *= -1.0

        # main section pattern
        section_pattern = FibsemLineSettings(start_x=point.x - section_width / 2, 
                                             end_x=point.x + section_width / 2, 
                                             start_y=point.y + section_y, 
                                             end_y=point.y + section_y, 
                                             depth=section_depth)

        self.shapes = [section_pattern]

        if use_side_patterns:
            # side cleaning patterns
            left_side_pattern = FibsemLineSettings(
                start_x=point.x - section_width / 2 - side_width / 2,
                end_x=point.x - section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )
            right_side_pattern = FibsemLineSettings(
                start_x=point.x + section_width / 2 - side_width / 2,
                end_x=point.x + section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )

            # side vertical patterns
            left_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x - section_width / 2,
                end_x=point.x - section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            right_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x + section_width / 2,
                end_x=point.x + section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            self.shapes.extend([left_side_pattern, right_side_pattern, 
                            left_side_pattern_vertical, 
                            right_side_pattern_vertical])
            
        return self.shapes


@dataclass
class FiducialPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=1.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the fiducial mark.",
        },
    )
    height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the fiducial mark.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the fiducial mark.",
        },
    )
    rotation: float = field(
        default=45.0,
        metadata={
            "label": "Rotation",
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Rotation angle of the fiducial mark.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )
    asymmetric: bool = field(
        default=False,
        metadata={
            "label": "Asymmetric",
            "type": bool,
            "tooltip": "If true, make the fiducial mark y-shaped.",
        },
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
            left_pattern.centre_x -= left_pattern.height*0.5*np.cos(rotation)
            left_pattern.centre_y += left_pattern.height*0.5*np.sin(rotation)

        self.shapes = [left_pattern, right_pattern]
        return self.shapes


@dataclass
class UndercutPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=5.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the undercut pattern.",
        },
    )
    height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the undercut pattern.",
        },
    )
    depth: float = field(
        default=10.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the undercut pattern.",
        },
    )
    trench_width: float = field(
        default=2.0e-6,
        metadata={
            "label": "Trench Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the trench in the undercut pattern.",
        },
    )
    rhs_height: float = field(
        default=10.0e-6,
        metadata={
            "label": "RHS Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the right-hand side of the undercut pattern.",
        },
    )
    h_offset: float = field(
        default=5.0e-6,
        metadata={
            "label": "Horizontal Offset",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Horizontal offset of the undercut pattern.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "Undercut"

    def define(self) -> List[FibsemRectangleSettings]:
        
        point = self.point
        jcut_rhs_height = self.rhs_height
        jcut_lamella_height = self.height
        jcut_width = self.width
        jcut_trench_thickness = self.trench_width
        jcut_depth = self.depth
        jcut_h_offset = self.h_offset
        cross_section = self.cross_section

        # top_jcut
        jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
        jcut_top_centre_y = point.y + jcut_lamella_height
        jcut_top_width = jcut_width
        jcut_top_height = jcut_trench_thickness
        jcut_top_depth = jcut_depth

        top_pattern = FibsemRectangleSettings(
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
            centre_x=point.x,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )
        # rhs jcut
        jcut_rhs_centre_x = point.x + (jcut_width / 2) - jcut_trench_thickness / 2
        jcut_rhs_centre_y = point.y - (jcut_rhs_height / 2) + jcut_trench_thickness / 2
        jcut_rhs_width = jcut_trench_thickness
        jcut_rhs_height = jcut_rhs_height
        jcut_rhs_depth = jcut_depth

        rhs_pattern = FibsemRectangleSettings(
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
            centre_x=jcut_rhs_centre_x,
            centre_y=jcut_rhs_centre_y,
            scan_direction="TopToBottom",
            cross_section = cross_section
        )

        self.shapes = [top_pattern, rhs_pattern]
        return self.shapes


@dataclass
class MicroExpansionPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=0.5e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the microexpansion joint pattern.",
        },
    )
    height: float = field(
        default=15.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the microexpansion joint pattern.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the microexpansion joint pattern.",
        },
    )
    distance: float = field(
        default=10.0e-6,
        metadata={
            "label": "Distance",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Distance between microexpansion joints.",
        },
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
            centre_x=point.x -  distance,
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


@dataclass
class ArrayPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = field(
        default=2.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the array pattern.",
        },
    )
    height: float = field(
        default=2.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the array pattern.",
        },
    )
    depth: float = field(
        default=5.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the array pattern.",
        },
    )
    n_columns: int = field(
        default=5,
        metadata={
            "label": "Number of Columns",
            "type": int,
            "minimum": 1,
            "maximum": 100,
            "step": 1,
            "decimals": 0,
            "tooltip": "Number of columns in the array pattern.",
        },
    )
    n_rows: int = field(
        default=5,
        metadata={
            "label": "Number of Rows",
            "type": int,
            "minimum": 1,
            "maximum": 100,
            "step": 1,
            "decimals": 0,
            "tooltip": "Number of rows in the array pattern.",
        },
    )
    pitch_vertical: float = field(
        default=5.0e-6,
        metadata={
            "label": "Vertical Pitch",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Vertical pitch between elements in the array pattern.",
        },
    )
    pitch_horizontal: float = field(
        default=5.0e-6,
        metadata={
            "label": "Horizontal Pitch",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Horizontal pitch between elements in the array pattern.",
        },
    )
    passes: int = field(
        default=0,
        metadata=DEFAULT_PASSES_METADATA,
    )
    rotation: float = field(
        default=0,
        metadata={
            "label": "Rotation",
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Rotation angle of the array pattern in degrees.",
        },
    )
    scan_direction: str = field(
        default="TopToBottom", metadata=DEFAULT_SCAN_DIRECTION_METADATA
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )
    use_circle: bool = field(
        default=False,
        metadata={
            "label": "Use Circle",
            "type": bool,
            "tooltip": "If true, use circular patterns instead of rectangular.",
        },
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

        return self.shapes # type: ignore


@dataclass
class WaffleNotchPattern(BasePattern[FibsemRectangleSettings]):
    vheight: float = field(
        default=2.0e-6,
        metadata={
            "label": "Vertical Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the vertical notch sections.",
        },
    )
    vwidth: float = field(
        default=0.5e-6,
        metadata={
            "label": "Vertical Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the vertical notch sections.",
        },
    )
    hheight: float = field(
        default=0.5e-6,
        metadata={
            "label": "Horizontal Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the horizontal notch sections.",
        },
    )
    hwidth: float = field(
        default=2.0e-6,
        metadata={
            "label": "Horizontal Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the horizontal notch sections.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the notch sections.",
        },
    )
    distance: float = field(
        default=2.0e-6,
        metadata={
            "label": "Distance",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Distance between notch sections.",
        },
    )
    inverted: bool = field(
        default=False,
        metadata={
            "label": "Inverted",
            "tooltip": "Invert the notch pattern.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=DEFAULT_CROSS_SECTION_METADATA
    )

    name: ClassVar[str] = "WaffleNotch"
    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(self) -> List[FibsemRectangleSettings]:

        point = self.point
        vwidth = self.vwidth
        vheight = self.vheight
        hwidth = self.hwidth
        hheight = self.hheight
        depth = self.depth
        distance = self.distance
        cross_section = self.cross_section
        inverted = -1 if self.inverted else 1

        # five patterns
        top_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - distance / 2 - vheight / 2 + hheight / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + distance / 2 + vheight / 2 - hheight / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        top_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y - distance / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y + distance / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        centre_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=distance + hheight,
            depth=depth,
            centre_x=point.x + (hwidth + vwidth) * inverted,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        self.shapes = [
            top_vertical_pattern,
            bottom_vertical_pattern,
            top_horizontal_pattern,
            bottom_horizontal_pattern,
            centre_vertical_pattern,
        ]

        return self.shapes


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


@dataclass
class TriForcePattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=1.0e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the triangles.",
        },
    )
    height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the triangles.",
        },
    )
    depth: float = field(
        default=5.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the triangles.",
        },
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

            triangle_shapes =  create_triangle_patterns(width=width, 
                                                        height=height, 
                                                        depth=depth, 
                                                        point=point, 
                                                        angle=angle)
            self.shapes.extend(triangle_shapes)
            
        return self.shapes


@dataclass
class TrenchTrapezoidPattern(BasePattern):
    trench_width: float = field(
        default=10e-6,
        metadata={
            "label": "Trench Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the trench at the top.",
        },
    )
    trench_height: float = field(
        default=5e-6,
        metadata={
            "label": "Trench Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Height of the trench.",
        },
    )
    spacing: float = field(
        default=5.0e-6,
        metadata={
            "label": "Spacing",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Spacing between trenches.",
        },
    )
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the trench.",
        },
    )
    angle: float = field(
        default=60,
        metadata={
            "label": "Angle",
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Angle of the trench in degrees.",
        },
    )
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
        mirrored_points[:, 1] = -mirrored_points[:, 1] #- spacing /2

        top_trench = FibsemPolygonSettings(vertices=points, depth=depth)
        bottom_trench = FibsemPolygonSettings(vertices=mirrored_points, depth=depth)

        # apply the verticies offset by the point
        top_trench.vertices[:, 0] += self.point.x
        top_trench.vertices[:, 1] -= self.point.y
        bottom_trench.vertices[:, 0] += self.point.x
        bottom_trench.vertices[:, 1] -= self.point.y

        self.shapes = [top_trench, bottom_trench]
        return self.shapes
        # QUERY: we should consolidate this with TrenchPattern?


@dataclass
class PolygonPattern(BasePattern):
    vertices: np.ndarray[float] = field(default_factory=lambda: np.array([]), metadata={"hidden": True})    # type: ignore[type-arg]
    depth: float = field(
        default=1.0e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the polygon.",
        },
    )
    is_exclusion: bool = field(
        default=False,
        metadata={
            "label": "Is Exclusion",
            "type": bool,
            "tooltip": "If true, the polygon will be an exclusion area.",
        },
    )
    name: ClassVar[str] = "PolygonPattern"

    def define(self) -> List[FibsemPolygonSettings]:
        """Define a polygon milling pattern based on the provided vertices."""
        
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("Vertices must be a 2D array with shape (n, 2)")

        # Create a polygon milling pattern
        polygon = FibsemPolygonSettings(vertices=np.array(self.vertices, copy=True), 
                                        depth=self.depth,
                                        is_exclusion=self.is_exclusion)

        # Offset the vertices by the point
        polygon.vertices[:, 0] += self.point.x
        polygon.vertices[:, 1] -= self.point.y

        self.shapes = [polygon]
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


# Pattern classes are now registered via the plugin system in __init__.py
# Legacy constants maintained for backwards compatibility
DEFAULT_MILLING_PATTERN = RectanglePattern
