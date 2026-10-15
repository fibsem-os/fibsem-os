"""Compatibility re-exports: the built-in patterns moved to one module each.

``BasePattern`` is in ``fibsem.milling.patterning.base`` and each built-in pattern in
``fibsem.milling.patterning.patterns.<name>``. This module stays so that existing
imports, including those in plugins and saved notebooks, keep working. It also
re-exports the structures and metadata names the old module imported, since callers
imported those from here too.
"""

from fibsem import constants  # noqa: F401
from fibsem.milling.patterning.base import BasePattern, TPattern
from fibsem.milling.patterning.patterns import *  # noqa: F403
from fibsem.milling.patterning.patterns import __all__ as _patterns_all
from fibsem.milling.patterning.patterns.rectangle import RectanglePattern
from fibsem.milling.patterning.patterns.triforce import create_triangle_patterns
from fibsem.structures import (  # noqa: F401
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    Point,
    TFibsemPatternSettings,
    field_meta,
)

DEFAULT_MILLING_PATTERN = RectanglePattern

__all__ = [
    "BasePattern",
    "DEFAULT_MILLING_PATTERN",
    "TPattern",
    "create_triangle_patterns",
    *_patterns_all,
]
