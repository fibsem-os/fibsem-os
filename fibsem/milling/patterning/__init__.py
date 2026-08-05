"""
BasePattern plugin system for fibsem milling patterns.

This module provides a plugin architecture for BasePattern implementations,
similar to the MillingStrategy plugin system.
"""

import logging
import typing
from typing import Any, Dict, Optional, Tuple, Type

from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.plugins.loader import PluginRecord, load_entry_point_group, plugin_classes

# Built-in patterns (imported from patterns2.py)
from fibsem.milling.patterning.patterns2 import (
    RectanglePattern,
    LinePattern, 
    CirclePattern,
    TrenchPattern,
    HorseshoePattern,
    HorseshoePatternVertical,
    SerialSectionPattern,
    UndercutPattern,
    FiducialPattern,
    ArrayPattern,
    MicroExpansionPattern,
    WaffleNotchPattern,
    CloverPattern,
    TriForcePattern,
    BitmapPattern,
    TrenchBitmapPattern,
    TrenchTrapezoidPattern,
)

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
    MicroExpansionPattern.name: MicroExpansionPattern,
    WaffleNotchPattern.name: WaffleNotchPattern,
    CloverPattern.name: CloverPattern,
    TriForcePattern.name: TriForcePattern,
    BitmapPattern.name: BitmapPattern,
    TrenchBitmapPattern.name: TrenchBitmapPattern,
    TrenchTrapezoidPattern.name: TrenchTrapezoidPattern,
}

# Runtime registered patterns
REGISTERED_PATTERNS: Dict[str, Type[BasePattern]] = {}

# Default pattern
DEFAULT_PATTERN = RectanglePattern
DEFAULT_PATTERN_NAME = DEFAULT_PATTERN.name


def register_pattern(pattern_cls: Type[BasePattern]) -> None:
    """Register a pattern class at runtime.
    
    Args:
        pattern_cls: The pattern class to register
        
    Example:
        >>> from fibsem.milling.patterning import register_pattern
        >>> register_pattern(CustomPattern)
    """
    global REGISTERED_PATTERNS
    REGISTERED_PATTERNS[pattern_cls.name] = pattern_cls
    logging.info("Registered pattern '%s'", pattern_cls.name)


PATTERN_ENTRY_POINT_GROUP = "fibsem.patterns"


def get_pattern_plugin_records() -> Tuple[PluginRecord, ...]:
    """Every ``fibsem.patterns`` entry point and what became of it.

    Loading happens once per process, on the first call. Includes the plugins
    that failed and the ones a built-in later shadows, neither of which
    survives into :func:`get_patterns` -- see ``fibsem.plugins.report``.

    To add a plugin pattern, add to your package's pyproject.toml:

    [project.entry-points.'fibsem.patterns']
    my_pattern = "my_package.patterns:MyCustomPattern"
    """
    return load_entry_point_group(
        group=PATTERN_ENTRY_POINT_GROUP,
        base_cls=BasePattern,
        name_of=lambda cls: cls.name,
        kind="pattern",
    )


def _get_plugin_patterns() -> Dict[str, Type[BasePattern]]:
    """Plugin patterns that loaded, as ``{name: class}``."""
    return plugin_classes(get_pattern_plugin_records())


def get_patterns() -> Dict[str, Type[BasePattern]]:
    """Get all available patterns.
    
    Returns patterns in priority order (highest to lowest):
    1. Built-in patterns
    2. Runtime registered patterns  
    3. Plugin patterns
    
    Returns:
        Dictionary mapping pattern names to pattern classes
    """
    # This order means that builtins > registered > plugins if there are any name clashes
    return {**_get_plugin_patterns(), **REGISTERED_PATTERNS, **BUILTIN_PATTERNS}


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
    if config is None:
        config = {}

    patterns = get_patterns()
    
    # Try exact match first
    if name in patterns:
        pattern_cls = patterns[name]
        pattern = pattern_cls.from_dict(config)
        return pattern
    
    # Try case-insensitive match for backwards compatibility
    name_lower = name.lower()
    for pattern_name, pattern_cls in patterns.items():
        if pattern_name.lower() == name_lower:
            pattern = pattern_cls.from_dict(config)
            return pattern
    
    # Pattern not found
    available = ", ".join(patterns.keys())
    raise NameError(f"No milling pattern named '{name}'. Available patterns: {available}")


# Legacy support - maintain backward compatibility
MILLING_PATTERNS = get_patterns()
MILLING_PATTERN_NAMES = get_pattern_names()
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
