"""A hardware geometry as older versions wrote it, in the current shape.

Saved inside both image metadata and fluorescence image metadata, so each reader
upgrades it with this. What changed, oldest first:

- ``is_compustage`` meant both ``fm_is_pose`` and ``mirrors_back_view``;
- there was no ``rotation_centre`` before FIB-1081, and every such image was drawn
  about `LEGACY_ROTATION_CENTRE`.
"""

from __future__ import annotations

from typing import Any, Dict

from fibsem.structures.system import LEGACY_ROTATION_CENTRE


def upgrade(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """``geometry`` from any version, in the current shape."""
    geometry = dict(geometry)
    if "is_compustage" in geometry:
        legacy = bool(geometry.pop("is_compustage"))
        geometry.setdefault("fm_is_pose", legacy)
        geometry.setdefault("mirrors_back_view", legacy)
    # Absent only on an image saved before the field existed; a recorded None stays.
    geometry.setdefault("rotation_centre", list(LEGACY_ROTATION_CENTRE))
    return geometry
