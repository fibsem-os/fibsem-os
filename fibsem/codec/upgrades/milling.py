"""Milling recipes as older versions wrote them, in the current shape.

- Tescan's preset, spot size, rate, dwell time and spacing sat at a recipe's top
  level; they are in its ``driver`` block under ``Tescan`` now.

Presence, not a version, decides each step, and a key already in the block wins.
Each step copies what it is given.
"""

from __future__ import annotations

from typing import Any, Dict

TESCAN = "Tescan"

# A recipe's top-level keys that are Tescan's own settings now.
TESCAN_KEYS = ("preset", "spot_size", "rate", "dwell_time", "spacing")


def settings(data: Any) -> Any:
    """A ``FibsemMillingSettings`` with Tescan's keys in its driver block."""
    if not isinstance(data, dict):
        return data
    data = dict(data)
    flat = {key: data.pop(key) for key in TESCAN_KEYS if key in data}
    if flat:
        driver = dict(data.get("driver") or {})
        driver[TESCAN] = {**flat, **(driver.get(TESCAN) or {})}
        data["driver"] = driver
    return data


def stage(data: Any) -> Any:
    """A ``FibsemMillingStage`` whose recipe is in the current shape."""
    if not isinstance(data, dict) or "milling" not in data:
        return data
    return {**data, "milling": settings(data["milling"])}


def task_milling(data: Any) -> Any:
    """A task config's ``milling`` (each milling task by its key) with every
    stage's recipe in the current shape."""
    if not isinstance(data, dict):
        return data
    upgraded: Dict[str, Any] = {}
    for key, config in data.items():
        if isinstance(config, dict) and isinstance(config.get("stages"), list):
            config = {**config, "stages": [stage(s) for s in config["stages"]]}
        upgraded[key] = config
    return upgraded
