"""Fixture pattern."""

from dataclasses import dataclass
from typing import ClassVar, List

from fibsem.milling.patterning.base import BasePattern
from fibsem.structures import Display, FibsemRectangleSettings, param
from fibsem_test_plugin import CLASHING_PATTERN_NAME, PATTERN_NAME


@dataclass
class FixturePattern(BasePattern[FibsemRectangleSettings]):
    name: ClassVar[str] = PATTERN_NAME

    width: float = param(10.0e-6, unit="m", display=Display("Width"))
    depth: float = param(1.0e-6, unit="m", display=Display("Depth"))

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
