"""
BasePattern plugin system for fibsem milling patterns.

This module provides a plugin architecture for BasePattern implementations,
similar to the MillingStrategy plugin system.
"""

import logging
import typing
from typing import Any, Dict, Optional, Type

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.patterning.patterns import (
    ArrayPattern,
    BitmapPattern,
    CirclePattern,
    CircularSuspensionPattern,
    CloverPattern,
    FiducialPattern,
    HorseshoePattern,
    HorseshoePatternVertical,
    LinePattern,
    MicroExpansionPattern,
    RectanglePattern,
    RulerPattern,
    SerialSectionPattern,
    TrenchBitmapPattern,
    TrenchPattern,
    TrenchTrapezoidPattern,
    TriForcePattern,
    UndercutPattern,
    WaffleNotchPattern,
)
from fibsem.plugins.loader import PluginRegistry, subclass_of

# Built-in patterns registry
BUILTIN_PATTERNS: Dict[str, Type[BasePattern]] = {
    RectanglePattern.name: RectanglePattern,
    LinePattern.name: LinePattern,
    CirclePattern.name: CirclePattern,
    TrenchPattern.name: TrenchPattern,
    HorseshoePattern.name: HorseshoePattern,
    HorseshoePatternVertical.name: HorseshoePatternVertical,
    SerialSectionPattern.name: SerialSectionPattern,
    UndercutPattern.name: UndercutPattern,
    FiducialPattern.name: FiducialPattern,
    ArrayPattern.name: ArrayPattern,
    RulerPattern.name: RulerPattern,
    MicroExpansionPattern.name: MicroExpansionPattern,
    WaffleNotchPattern.name: WaffleNotchPattern,
    CloverPattern.name: CloverPattern,
    TriForcePattern.name: TriForcePattern,
    BitmapPattern.name: BitmapPattern,
    TrenchBitmapPattern.name: TrenchBitmapPattern,
    TrenchTrapezoidPattern.name: TrenchTrapezoidPattern,
    CircularSuspensionPattern.name: CircularSuspensionPattern,
}


# Default pattern
DEFAULT_PATTERN = RectanglePattern


PATTERN_ENTRY_POINT_GROUP = "fibsem.milling.patterns"

# Built-ins, runtime registrations and the fibsem.milling.patterns entry points. To add a
# plugin pattern, add to your package's pyproject.toml:
#
#     [project.entry-points.'fibsem.milling.patterns']
#     my_pattern = "my_package.patterns:MyCustomPattern"
PATTERN_PLUGINS: PluginRegistry[Type[BasePattern]] = PluginRegistry(
    group=PATTERN_ENTRY_POINT_GROUP,
    kind="pattern",
    resolve=subclass_of(BasePattern, lambda cls: cls.name),
    builtins=BUILTIN_PATTERNS,
)


def register_pattern(pattern_cls: Type[BasePattern]) -> None:
    """Register a pattern class at runtime.

    Args:
        pattern_cls: The pattern class to register

    Example:
        >>> from fibsem.milling.patterning import register_pattern
        >>> register_pattern(CustomPattern)
    """
    PATTERN_PLUGINS.register(pattern_cls.name, pattern_cls)


def get_patterns() -> Dict[str, Type[BasePattern]]:
    """Get all available patterns.

    Returns patterns in priority order (highest to lowest):
    1. Built-in patterns
    2. Runtime registered patterns
    3. Plugin patterns

    Returns:
        Dictionary mapping pattern names to pattern classes
    """
    return PATTERN_PLUGINS.all()


def get_pattern_names() -> typing.List[str]:
    """Get list of all available pattern names."""
    return list(get_patterns().keys())


def get_pattern(name: str, config: Optional[Dict[str, Any]] = None) -> BasePattern:
    """Get a pattern instance by name and configuration.

    Args:
        name: Pattern name (case-insensitive)
        config: Pattern configuration dictionary

    Returns:
        Configured pattern instance

    Raises:
        NameError: If pattern name is not found
    """
    try:
        pattern_cls = BasePattern.subclass_named(name)
    except ValueError as error:
        raise NameError(str(error)) from None
    return pattern_cls.from_dict(config or {})


# Loading these at import loads the pattern plugins (`fibsem.plugins.loader`).
MILLING_PATTERNS = get_patterns()
DEFAULT_MILLING_PATTERN = DEFAULT_PATTERN

# Legacy protocol mapping
PROTOCOL_MILL_MAP = {
    "cut": RectanglePattern,
    "fiducial": FiducialPattern,
    "flatten": RectanglePattern,
    "undercut": UndercutPattern,
    "horseshoe": HorseshoePattern,
    "lamella": TrenchPattern,
    "polish_lamella": TrenchPattern,
    "thin_lamella": TrenchPattern,
    "sever": RectanglePattern,
    "sharpen": RectanglePattern,
    "needle": RectanglePattern,
    "copper_release": HorseshoePattern,
    "serial_trench": HorseshoePattern,
    "serial_undercut": RectanglePattern,
    "serial_sever": RectanglePattern,
    "lamella_sever": RectanglePattern,
    "lamella_polish": TrenchPattern,
    "trench": TrenchPattern,
    "notch": WaffleNotchPattern,
    "microexpansion": MicroExpansionPattern,
    "clover": CloverPattern,
    "autolamella": TrenchPattern,
    "MillUndercut": RectanglePattern,
    "rectangle": RectanglePattern,
    "MillRough": TrenchPattern,
    "MillRegularCut": TrenchPattern,
    "MillPolishing": TrenchPattern,
    "mill_rough": TrenchPattern,
    "mill_polishing": TrenchPattern,
}
