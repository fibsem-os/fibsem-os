"""
BasePattern plugin system for fibsem milling patterns.

This module provides a plugin architecture for BasePattern implementations,
similar to the MillingStrategy plugin system.
"""

import logging
import typing
from typing import Any, Dict, Optional, Tuple, Type

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
from fibsem.plugins.loader import PluginRecord, PluginRegistry, subclass_of

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
DEFAULT_PATTERN_NAME = DEFAULT_PATTERN.name


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

# Runtime registered patterns
REGISTERED_PATTERNS: Dict[str, Type[BasePattern]] = PATTERN_PLUGINS.registered


def register_pattern(pattern_cls: Type[BasePattern]) -> None:
    """Register a pattern class at runtime.

    Args:
        pattern_cls: The pattern class to register

    Example:
        >>> from fibsem.milling.patterning import register_pattern
        >>> register_pattern(CustomPattern)
    """
    PATTERN_PLUGINS.register(pattern_cls.name, pattern_cls)


def get_pattern_plugin_records() -> Tuple[PluginRecord, ...]:
    """Every ``fibsem.milling.patterns`` entry point and what became of it."""
    return PATTERN_PLUGINS.plugin_records()


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


# Legacy support - maintain backward compatibility
MILLING_PATTERNS = get_patterns()
MILLING_PATTERN_NAMES = get_pattern_names()
DEFAULT_MILLING_PATTERN = DEFAULT_PATTERN
