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
    DEFAULT_ANGLE_METADATA,
    DEFAULT_DISTANCE_METADATA,
    DEFAULT_DURATION_METADATA,
    DEFAULT_PASSES_METADATA,
    DEFAULT_SCAN_DIRECTION_METADATA,
)
from fibsem.structures import (
    FibsemBitmapSettings,
)


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
