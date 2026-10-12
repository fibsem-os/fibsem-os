"""The built-in milling patterns, one module per pattern."""

from fibsem.milling.patterning.patterns.array import ArrayPattern
from fibsem.milling.patterning.patterns.bitmap import BitmapPattern
from fibsem.milling.patterning.patterns.circle import CirclePattern
from fibsem.milling.patterning.patterns.circular_suspension import (
    CircularSuspensionPattern,
)
from fibsem.milling.patterning.patterns.clover import CloverPattern
from fibsem.milling.patterning.patterns.fiducial import FiducialPattern
from fibsem.milling.patterning.patterns.horseshoe import (
    HorseshoePattern,
    HorseshoePatternVertical,
)
from fibsem.milling.patterning.patterns.line import LinePattern
from fibsem.milling.patterning.patterns.micro_expansion import MicroExpansionPattern
from fibsem.milling.patterning.patterns.polygon import PolygonPattern
from fibsem.milling.patterning.patterns.rectangle import RectanglePattern
from fibsem.milling.patterning.patterns.ruler import RulerPattern
from fibsem.milling.patterning.patterns.serial_section import SerialSectionPattern
from fibsem.milling.patterning.patterns.trench import (
    TrenchBitmapPattern,
    TrenchPattern,
    TrenchTrapezoidPattern,
)
from fibsem.milling.patterning.patterns.triforce import (
    TriForcePattern,
    create_triangle_patterns,
)
from fibsem.milling.patterning.patterns.undercut import UndercutPattern
from fibsem.milling.patterning.patterns.waffle_notch import WaffleNotchPattern

__all__ = [
    "ArrayPattern",
    "BitmapPattern",
    "CirclePattern",
    "CircularSuspensionPattern",
    "CloverPattern",
    "FiducialPattern",
    "HorseshoePattern",
    "HorseshoePatternVertical",
    "LinePattern",
    "MicroExpansionPattern",
    "PolygonPattern",
    "RectanglePattern",
    "RulerPattern",
    "SerialSectionPattern",
    "TrenchBitmapPattern",
    "TrenchPattern",
    "TrenchTrapezoidPattern",
    "TriForcePattern",
    "UndercutPattern",
    "WaffleNotchPattern",
    "create_triangle_patterns",
]
