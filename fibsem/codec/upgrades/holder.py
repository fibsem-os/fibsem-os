"""Sample holder files as older versions wrote them, in the current shape.

- Holder files from before slots carried a calibration record (September 2026) can
  hold a slot position with only some axes (``{x, y, z}``). It has no calibration
  record, so the holder discards it on load; it only has to read.
"""

from __future__ import annotations

from typing import Any, Dict

_POSITION_KEYS = ("x", "y", "z", "r", "t", "coordinate_system")


def slot(data: Dict[str, Any]) -> Dict[str, Any]:
    """``GridSlot`` data from any version, in the current shape."""
    position = data.get("position")
    if not isinstance(position, dict):
        return data
    return {**data, "position": {**dict.fromkeys(_POSITION_KEYS), **position}}


def holder(data: Dict[str, Any]) -> Dict[str, Any]:
    """``SampleHolder`` data from any version, in the current shape."""
    slots = data.get("slots")
    if not isinstance(slots, dict):
        return data
    return {**data, "slots": {name: slot(s) for name, s in slots.items()}}
