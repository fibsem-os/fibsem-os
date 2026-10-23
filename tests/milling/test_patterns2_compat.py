"""`fibsem.milling.patterning.patterns2` keeps re-exporting what it used to define.

Plugins, notebooks and scripts outside this repository import from it, so every
name it defined before the split into `base` and `patterns/` must still resolve,
and to the same object as its new home.
"""

import importlib

import pytest

from fibsem.milling.patterning import patterns2

OLD_NAMES = [
    "ArrayPattern",
    "BasePattern",
    "BitmapPattern",
    "CirclePattern",
    "CircularSuspensionPattern",
    "CloverPattern",
    "CrossSectionPattern",
    "DEFAULT_MILLING_PATTERN",
    "FibsemCircleSettings",
    "FibsemLineSettings",
    "FibsemRectangleSettings",
    "FiducialPattern",
    "HorseshoePattern",
    "HorseshoePatternVertical",
    "LinePattern",
    "MicroExpansionPattern",
    "Point",
    "PolygonPattern",
    "RectanglePattern",
    "RulerPattern",
    "SerialSectionPattern",
    "TPattern",
    "TrenchBitmapPattern",
    "TrenchPattern",
    "TrenchTrapezoidPattern",
    "TriForcePattern",
    "UndercutPattern",
    "WaffleNotchPattern",
    "create_triangle_patterns",
]


@pytest.mark.parametrize("name", OLD_NAMES)
def test_old_name_still_imports(name):
    assert hasattr(patterns2, name)


PATTERN_CLASSES = [
    n
    for n in OLD_NAMES
    if n.endswith("Pattern") and n not in ("CrossSectionPattern", "TPattern")
]


@pytest.mark.parametrize("name", PATTERN_CLASSES)
def test_old_name_is_the_new_class(name):
    obj = getattr(patterns2, name)
    home = importlib.import_module(obj.__module__)
    assert getattr(home, name) is obj
    if name != "BasePattern":
        assert obj.__module__.startswith("fibsem.milling.patterning.patterns.")
