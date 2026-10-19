"""Overview acquisition settings as older versions wrote them, in the current shape.

- v0.5.0 turned focus stacking on with a bare ``use_focus_stack``.
- Before FIB-646 the autofocus mode sat inside ``autofocus_settings`` as ``mode``,
  in a class that held nothing else, so such a file carries no sweep and gets the
  default one. The sweep config never writes ``mode``, which makes it a safe test.
- Before ``autocontrast_mode`` the per-image ``autocontrast`` flag meant one contrast
  for the mosaic: ONCE when it was on, NONE when it was off.
"""

from __future__ import annotations

from typing import Any, Dict

from fibsem.codec.upgrades.autofocus import settings as upgrade_autofocus


def acquisition_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    """``OverviewAcquisitionSettings`` data from any version, in the current shape."""
    data = dict(data)
    stacking = data.pop("use_focus_stack", None)
    if "focus_stack_settings" not in data and stacking is not None:
        data["focus_stack_settings"] = {"enabled": bool(stacking)}

    autofocus = data.pop("autofocus_settings", None) or {}
    if "mode" in autofocus:
        data["autofocus_mode"] = autofocus["mode"]
    elif autofocus:
        data["autofocus_settings"] = upgrade_autofocus(autofocus)

    if "autocontrast_mode" not in data:
        image = data.get("image_settings") or {}
        data["autocontrast_mode"] = "ONCE" if image.get("autocontrast") else "NONE"
    return data
