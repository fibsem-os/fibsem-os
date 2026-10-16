from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    List,
    Literal,
    Optional,
)

from numpy.typing import NDArray

from fibsem import constants
from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    angle_metadata,
    distance_metadata,
    duration_metadata,
    passes_metadata,
    scan_direction_metadata,
)
from fibsem.structures import Display, FibsemBitmapSettings, Parameter


@dataclass
class BitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Width", "Width of the bitmap pattern."),
    )
    height: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Height", "Height of the bitmap pattern."),
    )
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Depth", "Depth of the bitmap pattern."),
    )
    rotation: float = field(
        default=0,
        metadata=angle_metadata(
            "Rotation", "Rotation of the bitmap pattern in degrees."
        ),
    )
    time: float = field(default=0, metadata=duration_metadata())
    passes: int = field(default=0, metadata=passes_metadata())
    scan_direction: str = field(
        default="TopToBottom", metadata=scan_direction_metadata()
    )
    path: str = field(
        default="",
        metadata=Parameter(
            doc="Path to the bitmap image file.",
            filepath=True,
            display=Display("File Path"),
        ).meta(),
    )
    array: Optional[NDArray[Any]] = field(
        default=None, metadata=Parameter(display=Display(hidden=True)).meta()
    )
    interpolation: Optional[Literal["nearest", "bilinear", "bicubic"]] = field(
        default=None,
        metadata=Parameter(
            choices=[None, "nearest", "bilinear", "bicubic"],
            doc="Interpolation method for resizing the bitmap image.",
            display=Display("Interpolation", advanced=True),
        ).meta(),
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
