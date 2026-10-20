from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    ClassVar,
    List,
    Literal,
    Optional,
)

from numpy.typing import NDArray

from fibsem import constants
from fibsem.codec import NotSaved
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_param,
    distance_param,
    duration_param,
    passes_param,
    scan_direction_param,
)
from fibsem.structures import Display, FibsemBitmapSettings, param


@dataclass
class BitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = param(
        10.0e-6, distance_param("Width", "Width of the bitmap pattern.")
    )
    height: float = param(
        10.0e-6, distance_param("Height", "Height of the bitmap pattern.")
    )
    depth: float = param(
        1.0e-6, distance_param("Depth", "Depth of the bitmap pattern.")
    )
    rotation: float = param(
        0, angle_param("Rotation", "Rotation of the bitmap pattern in degrees.")
    )
    time: float = param(0, duration_param())
    passes: int = param(0, passes_param())
    scan_direction: str = param("TopToBottom", scan_direction_param())
    path: str = param(
        "",
        doc="Path to the bitmap image file.",
        filepath=True,
        display=Display("File Path"),
    )
    array: Annotated[Optional[NDArray[Any]], NotSaved] = param(
        None, display=Display(hidden=True)
    )
    interpolation: Optional[Literal["nearest", "bilinear", "bicubic"]] = param(
        None,
        choices=[None, "nearest", "bilinear", "bicubic"],
        doc="Interpolation method for resizing the bitmap image.",
        display=Display("Interpolation", advanced=True),
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
