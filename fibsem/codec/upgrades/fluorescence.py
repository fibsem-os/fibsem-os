"""Fluorescence structures as older versions wrote them, in the current shape.

A channel appears in protocols, the FM working configuration, the recent channels
and each FM image's metadata, so its upgrade is shared by all of them.
Before FIB-1439 a channel named its filter by ``emission_wavelength`` (None for
reflection, a label for a multi-band filter, or the band's bottom edge in nm).

The rest belongs to one file each:

- the working configuration may have no ``camera_settings``, or a null one;
- an FM image's ``geometry`` changed shape on its own
  (`fibsem.codec.upgrades.hardware_geometry`), and an empty record means none.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fibsem.codec.upgrades.autofocus import settings as upgrade_autofocus
from fibsem.codec.upgrades.hardware_geometry import upgrade as upgrade_geometry


def channel(data: Dict[str, Any]) -> Dict[str, Any]:
    """``ChannelSettings`` (or an image's channel) data from any version."""
    from fibsem.fm.structures import emission_filter_for

    data = dict(data)
    wavelength = data.pop("emission_wavelength", None)
    if "emission_filter" not in data:
        data["emission_filter"] = emission_filter_for(wavelength, {})
    return data


def configuration(data: Dict[str, Any]) -> Dict[str, Any]:
    """``FluorescenceConfiguration`` data from any version."""
    data = dict(data)
    data["channel_settings"] = [channel(c) for c in data.get("channel_settings", [])]
    if not data.get("camera_settings"):
        data.pop("camera_settings", None)
    if data.get("autofocus_settings") is not None:
        data["autofocus_settings"] = upgrade_autofocus(data["autofocus_settings"])
    return data


def image_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """``FluorescenceImageMetadata`` data from any version."""
    data = dict(data)
    data["channels"] = [channel(c) for c in data.get("channels", [])]
    data["geometry"] = _or_none(data.get("geometry"), upgrade_geometry)
    data["stage_position"] = _or_none(data.get("stage_position"), _stage_position)
    data["experiment"] = data.get("experiment") or None
    return data


def _stage_position(data: Dict[str, Any]) -> Dict[str, Any]:
    # Written whole, but read with every key optional: a missing axis is unknown.
    return {
        "x": None,
        "y": None,
        "z": None,
        "r": None,
        "t": None,
        "coordinate_system": "RAW",
        **data,
    }


def _or_none(value: Optional[Dict[str, Any]], upgrade: Any) -> Optional[Dict]:
    """``value`` upgraded, or None for none: an empty record is no record."""
    return upgrade(value) if value else None
