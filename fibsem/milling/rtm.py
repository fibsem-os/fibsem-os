"""Real Time Monitor (RTM): what the ion beam reads back from the patterns as
they mill (FIB-1178).

A driver with RTM reads, at most once per interval, a value at points of each
monitored pattern; the milling service makes each read an `RtmFrame` and emits
it on ``Milling.rtm_frame``. A frame is one read covering every monitored
pattern, keyed by the pattern's draw index, with a read ``sequence``, never a
pass number: the instrument says nothing of passes, and the interval skips some.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from fibsem.structures import CrossSectionPattern

# How densely a driver reads: "low" fewer points per pattern, "high" more.
RTM_MODES = ("low", "high")


@dataclass
class RtmFrame:
    """One RTM read. ``values[i]`` are the values read at ``positions[i]`` (n x 2,
    x then y, in pixels of an image of ``resolution``, width then height) of the
    pattern drawn ``i``-th."""

    sequence: int
    timestamp: float
    interval: float
    resolution: Tuple[int, int]
    values: Dict[int, np.ndarray] = field(default_factory=dict)
    positions: Dict[int, np.ndarray] = field(default_factory=dict)

    def means(self) -> Dict[int, float]:
        """Each pattern's mean value."""
        return {i: float(np.mean(v)) for i, v in self.values.items() if np.size(v) > 0}

    def to_image(self) -> np.ndarray:
        """The read as an image of ``resolution``: each value at its position,
        NaN where nothing was read."""
        width, height = self.resolution
        image = np.full((height, width), np.nan, dtype=float)
        for i, values in self.values.items():
            points = np.asarray(self.positions[i], dtype=int).reshape(-1, 2)
            x, y = points[:, 0], points[:, 1]
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            image[y[inside], x[inside]] = np.asarray(values, dtype=float)[inside]
        return image


def is_monitored(pattern: Any) -> bool:
    """Whether RTM reads *pattern*: not an exclusion, and not a cleaning
    cross-section, which AutoScript doesn't monitor (AST-555)."""
    if getattr(pattern, "is_exclusion", False):
        return False
    return (
        getattr(pattern, "cross_section", None)
        is not CrossSectionPattern.CleaningCrossSection
    )
