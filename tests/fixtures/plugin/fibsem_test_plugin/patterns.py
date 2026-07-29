"""Fixture pattern.

This module's imports are load-bearing, and narrower than they look like they
need to be.

``fibsem/milling/base.py:11`` does ``from fibsem.milling.patterning import
get_pattern, DEFAULT_MILLING_PATTERN``, and ``patterning/__init__.py`` ends with
a module-level ``MILLING_PATTERNS = get_patterns()``. So every pattern plugin is
imported *while ``fibsem.milling.base`` is only partially initialised*.

Anything this module imports that reaches ``fibsem.milling.base`` -- including
``fibsem.milling.tasks`` and everything under ``fibsem.applications`` -- raises
ImportError, the plugin loader catches it, and the pattern never registers.

So: patterns import from ``patterns2`` and ``fibsem.structures``, and nothing
else. That is also why a plugin's pattern, strategy and task have to live in
separate modules rather than one.
"""

from dataclasses import dataclass, field
from typing import ClassVar, List

from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import FibsemRectangleSettings

from fibsem_test_plugin import CLASHING_PATTERN_NAME, PATTERN_NAME


@dataclass
class FixturePattern(BasePattern[FibsemRectangleSettings]):
    name: ClassVar[str] = PATTERN_NAME

    width: float = field(default=10.0e-6, metadata={"label": "Width", "unit": "m"})
    depth: float = field(default=1.0e-6, metadata={"label": "Depth", "unit": "m"})

    def define(self) -> List[FibsemRectangleSettings]:
        self.shapes = [
            FibsemRectangleSettings(
                width=self.width,
                height=self.width,
                depth=self.depth,
                centre_x=self.point.x,
                centre_y=self.point.y,
            )
        ]
        return self.shapes


@dataclass
class ClashingPattern(FixturePattern):
    """Takes the name of a built-in pattern on purpose.

    Registering this must not displace the built-in: get_patterns() merges
    plugins first and built-ins last, so built-ins win. Without a plugin that
    actually collides, a test of that precedence asserts nothing.
    """

    name: ClassVar[str] = CLASHING_PATTERN_NAME
