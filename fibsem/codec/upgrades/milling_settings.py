"""A milling recipe as older versions wrote it, in the current shape.

A recipe (`fibsem.structures.FibsemMillingSettings`) sits inside the files that hold
milling stages: protocols, experiments and task configs. What changed:

- Before FIB-1437 the recipe held every driver's fields at its top level. Tescan's own
  (``preset``, ``spot_size``, ``rate``, ``dwell_time``, ``spacing``) now sit in its
  block, ``driver: {tescan: {...}}``; the other drivers had none of their own.

Presence decides the step, as the recipe has no version of its own: a key at the top
level is moved into the block, and a value the block already has wins. A recipe is
saved back in the current shape, so an old one is moved here once.
"""

from __future__ import annotations

from typing import Any, Dict

# Tescan's settings, as recipes saved them at the top level before FIB-1437.
TESCAN_TOP_LEVEL_KEYS = ("preset", "spot_size", "rate", "dwell_time", "spacing")


def upgrade(data: Dict[str, Any]) -> Dict[str, Any]:
    """``data``, a recipe from any version, in the current shape."""
    moved = {key: data[key] for key in TESCAN_TOP_LEVEL_KEYS if key in data}
    if not moved:
        return data
    data = {key: value for key, value in data.items() if key not in moved}
    blocks = {
        prefix: dict(block) for prefix, block in (data.get("driver") or {}).items()
    }
    blocks["tescan"] = {**moved, **blocks.get("tescan", {})}
    data["driver"] = blocks
    return data
