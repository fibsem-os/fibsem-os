"""Plain values for pinned-behaviour files, and the closeness they are compared by."""

from typing import Any

import numpy as np

from fibsem.structures import FibsemStagePosition


def plain(value: Any) -> Any:
    """``value`` as JSON-able plain data: positions as axis dicts, numbers as floats."""
    if isinstance(value, FibsemStagePosition):
        return {
            axis: None if getattr(value, axis) is None else float(getattr(value, axis))
            for axis in ("x", "y", "z", "r", "t")
        }
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating, int)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value if value is None or isinstance(value, str) else repr(value)


def close_to_pin(actual: Any, expected: Any) -> bool:
    """Equal, with floats compared to 1e-9 relative."""
    if isinstance(expected, float) and isinstance(actual, float):
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-15))
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            close_to_pin(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            close_to_pin(actual[k], expected[k]) for k in expected
        )
    return actual == expected
