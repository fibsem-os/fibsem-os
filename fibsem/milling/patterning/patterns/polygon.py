from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

import numpy as np

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_metadata,
)
from fibsem.structures import Display, FibsemPolygonSettings, Parameter


@dataclass
class PolygonPattern(BasePattern):
    vertices: np.ndarray[float] = field(
        default_factory=lambda: np.array([]),
        metadata=Parameter(display=Display(hidden=True)).meta(),
    )  # type: ignore[type-arg]
    depth: float = field(
        default=1.0e-6,
        metadata=distance_metadata("Depth", "Depth of the polygon."),
    )
    is_exclusion: bool = field(
        default=False,
        metadata=Parameter(
            doc="If true, the polygon will be an exclusion area.",
            display=Display("Is Exclusion"),
        ).meta(),
    )
    name: ClassVar[str] = "PolygonPattern"

    def define(self) -> List[FibsemPolygonSettings]:
        """Define a polygon milling pattern based on the provided vertices."""

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("Vertices must be a 2D array with shape (n, 2)")

        # Create a polygon milling pattern
        polygon = FibsemPolygonSettings(
            vertices=np.array(self.vertices, copy=True),
            depth=self.depth,
            is_exclusion=self.is_exclusion,
        )

        # Offset the vertices by the point
        polygon.vertices[:, 0] += self.point.x
        polygon.vertices[:, 1] -= self.point.y

        self.shapes = [polygon]
        return self.shapes
