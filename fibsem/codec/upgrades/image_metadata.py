"""Image metadata as older versions wrote it, in the current shape.

The metadata carries its ``version`` (`fibsem.config.METADATA_VERSION`); a file with
none was written before versioning (``v0``). What changed, oldest first:

- up to v5 the whole ``SystemSettings`` was written as ``system``; v6 replaced it with
  ``system_info`` and ``hardware_geometry`` (FIB-481), so the geometry is recovered
  from the blob here;
- ``hardware_geometry`` changed shape on its own (`hardware_geometry.upgrade`);
- ``system_info`` and ``hardware_geometry`` were written as ``{}`` when unknown.

Presence, not the version, decides each step: the version says which shape a writer
meant, and the keys say which shape it wrote.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fibsem.codec.upgrades.hardware_geometry import upgrade as upgrade_geometry
from fibsem.config import UNVERSIONED_METADATA
from fibsem.structures import LEGACY_ROTATION_CENTRE, FibsemHardwareGeometry


def upgrade(data: Dict[str, Any]) -> Dict[str, Any]:
    """``data`` from any version, in the current shape."""
    data = dict(data)
    data.setdefault("version", UNVERSIONED_METADATA)
    legacy_system = data.pop("system", None) or {}

    info = data.get("system_info") or legacy_system.get("info") or None
    if info is not None and not info.get("fibsem_revision"):
        # An image that did not record the revision it was written by: unknown, not
        # the revision reading it now.
        info = {**info, "fibsem_revision": None}
    data["system_info"] = info

    geometry = data.get("hardware_geometry") or None
    if geometry is not None:
        data["hardware_geometry"] = upgrade_geometry(geometry)
    else:
        data["hardware_geometry"] = legacy_geometry(legacy_system)
    return data


def legacy_geometry(system: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The geometry recovered from a pre-v6 ``system`` blob; None for none.

    Read with ``.get()`` chains, so a partial blob degrades to the values a new
    record has. Whether the FM is a pose and the instrument mirrors its back views is
    recovered the way the reprojection used to detect it: by model name, falling
    back to the simulator flag. That match is wrong -- a capability inferred from a
    name -- which is why v6 records it instead; a pre-v6 file has nothing better.
    """
    if not system:
        return None
    stage = system.get("stage") or {}
    electron = system.get("electron") or {}
    ion = system.get("ion") or {}
    model = (system.get("info") or {}).get("model") or ""
    sim = system.get("sim") or {}
    is_compustage = "Arctis" in model or bool(sim.get("is_compustage", False))

    default = FibsemHardwareGeometry()
    return {
        "column_tilt": electron.get("column_tilt", default.column_tilt),
        "fib_column_tilt": ion.get("column_tilt", default.fib_column_tilt),
        "shuttle_pre_tilt": stage.get("shuttle_pre_tilt", default.shuttle_pre_tilt),
        "rotation_reference": stage.get(
            "rotation_reference", default.rotation_reference
        ),
        "rotation_180": stage.get("rotation_180", default.rotation_180),
        "fm_is_pose": is_compustage,
        "mirrors_back_view": is_compustage,
        "rotation_centre": list(LEGACY_ROTATION_CENTRE),
    }
