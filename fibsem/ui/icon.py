"""Central icon factory for the fibsem UI, backed by qtawesome.

Icons are referenced throughout the UI by Iconify-style keys such as
``"mdi:check-circle"``. This module resolves them against qtawesome's *bundled*
Material Design Icons font, so the application renders every icon **fully
offline** - no network access, no icon assets committed to the repo.

Keeping this one indirection point means:
  * the ``"mdi:<name>"`` key convention (including dynamically-built names like
    ``f"mdi:numeric-{n}-box-outline"``) keeps working unchanged;
  * the handful of names that differ between Iconify's MDI and qtawesome's MDI6
    are remapped in exactly one place (see ``_REMAP``);
  * the icon backend can be swapped again from a single file.

Usage mirrors the previous ``QIconifyIcon`` call sites::

    btn.setIcon(fibsem_icon("mdi:check-circle", color="#4caf50"))

Extra keyword arguments pass straight through to ``qtawesome.icon`` (e.g.
``color_active``, ``color_disabled`` for per-state colouring).
"""
from __future__ import annotations

import logging
from typing import Optional

import qtawesome as qta
from qtpy.QtGui import QIcon

# qtawesome's font prefix for Material Design Icons v6.
_MDI = "mdi6"

# Names that exist in Iconify's MDI but not under the same key in MDI6.
_REMAP = {
    "add": "plus",            # iconify aliases "add" -> "plus"; MDI6 has no "add"
    "file-transfer": "file-send",  # MDI6 has no "file-transfer"
}


def fibsem_icon(key: str, color: Optional[str] = None, **kwargs) -> QIcon:
    """Return a :class:`QIcon` for an ``"mdi:<name>"`` (or bare ``<name>``) key.

    Rendered from qtawesome's offline MDI6 font. ``color`` is applied exactly as
    before; any additional ``qtawesome.icon`` keyword arguments are forwarded.
    """
    name = key.split(":", 1)[1] if ":" in key else key
    name = _REMAP.get(name, name)
    if color is not None:
        kwargs["color"] = color
    try:
        return qta.icon(f"{_MDI}.{name}", **kwargs)
    except Exception:
        # Unknown/misspelled icon name: qtawesome raises, but widget code (often
        # a paint/refresh path) expects a QIcon. Degrade gracefully to a blank
        # icon - matching the old superqt.QIconifyIcon, which drew a placeholder
        # rather than crashing - and log so the missing key can be fixed.
        logging.warning("fibsem_icon: unknown icon %r; rendering blank", key)
        return QIcon()
