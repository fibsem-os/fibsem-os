"""Numpy values as the plain Python they hold, for anything that leaves fibsem.

A numpy scalar or array is not JSON, YAML's safe dumper refuses it, and a backend on
another numpy (odemis unpickles fibsem's requests on the system's numpy 1) cannot
read it. Device writes, command arguments and the wire convert through `plain`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plain(value: Any) -> Any:
    """*value* with every numpy scalar and array in it as plain Python: a scalar as
    its ``int``, ``float`` or ``bool``, an array as a list, inside dicts, lists and
    tuples too. Anything else passes through unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return type(value)(plain(v) for v in value)
    return value
