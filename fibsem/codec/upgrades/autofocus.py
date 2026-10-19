"""Autofocus settings as older versions wrote them, in the current shape.

- Before passes, a sweep was a flat coarse and fine pair (``coarse_range``,
  ``coarse_step``, ``coarse_enabled`` and the same for ``fine_``).
- A pass gave its span as ``n_steps`` of ``step_size``.
- An empty ``passes`` has always meant the default coarse and fine passes; read as
  written, it would be a sweep that never runs.
"""

from __future__ import annotations

from typing import Any, Dict

_COARSE_FINE = {
    "coarse": (50e-6, 5e-6),
    "fine": (10e-6, 1e-6),
}


def settings(data: Dict[str, Any]) -> Dict[str, Any]:
    """``AutoFocusSettings`` data from any version, in the current shape."""
    data = dict(data)
    if "passes" not in data and ("coarse_range" in data or "fine_range" in data):
        data["passes"] = [
            {
                "search_range": data.pop(f"{name}_range", span),
                "step_size": data.pop(f"{name}_step", step),
                "enabled": data.pop(f"{name}_enabled", True),
            }
            for name, (span, step) in _COARSE_FINE.items()
        ]
    passes = data.pop("passes", None)
    if passes:
        data["passes"] = [sweep_pass(p) for p in passes]
    return data


def sweep_pass(data: Dict[str, Any]) -> Dict[str, Any]:
    """``FocusSweepPass`` data from any version, in the current shape."""
    data = dict(data)
    steps = data.pop("n_steps", None)
    if "search_range" not in data and steps is not None:
        data["search_range"] = steps * data.get("step_size", 0.5e-3)
    return data
