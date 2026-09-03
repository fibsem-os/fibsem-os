"""Central icon factory for the fibsem UI, backed by qtawesome.

Icons are referenced throughout the UI by Iconify-style keys such as
``"mdi:check-circle"``. This module resolves them against qtawesome's *bundled*
Material Design Icons font, so the application renders every icon **fully
offline** - no network access. The handful of shapes that font does not cover
ship as SVGs in ``fibsem/ui/icons/``, and are resolved from here too (see
``drag_handle_pixmap``) so no call site re-derives that directory.

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
import os
from typing import Optional

try:
    import qtawesome as qta
except ModuleNotFoundError as e:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "The fibsem UI requires 'qtawesome' for offline icon rendering "
        "(added in this version). Install it with:\n"
        "    pip install 'fibsem[ui]'   (or: pip install qtawesome)"
    ) from e
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap

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


# ── Canonical icon keys for shared actions ───────────────────────────────────
# Reference these instead of hardcoding "mdi:..." so the same action reads the same
# everywhere (moving to / updating a saved stage/objective position).
ICON_MOVE_TO_POSITION = "mdi:crosshairs-gps"   # go to a saved/target position
ICON_UPDATE_POSITION = "mdi:map-marker-check"  # overwrite a saved position with the current one


# ── Bundled SVG assets ───────────────────────────────────────────────────────
# Shapes qtawesome's font does not carry live next to this module. Resolve them
# from here rather than counting os.path.dirname() steps back to fibsem/ui at
# each call site: one widget package moved and its private copy of the sum kept
# pointing at a directory that does not exist, so its drag handle silently
# vanished and every row logged a null-pixmap warning instead.
ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")
DRAG_HANDLE_PATH = os.path.join(ICONS_DIR, "drag_handle.svg")

# Every draggable list row draws the handle at this size.
DRAG_HANDLE_WIDTH = 10
DRAG_HANDLE_HEIGHT = 16


def drag_handle_pixmap(
    width: int = DRAG_HANDLE_WIDTH, height: int = DRAG_HANDLE_HEIGHT
) -> QPixmap:
    """Return the shared drag-handle asset, scaled to ``width`` x ``height``.

    Scaling a null pixmap is what prints "QPixmap::scaled: Pixmap is a null
    pixmap" - once per row, on every rebuild - so the check belongs here rather
    than at each call site. ``isNull()`` rather than ``os.path.exists()``: it
    also covers a file that is present but unreadable or not valid SVG. Callers
    always get a pixmap they can hand straight to ``QLabel.setPixmap``, empty if
    the asset is missing.
    """
    pixmap = QPixmap(DRAG_HANDLE_PATH)
    if pixmap.isNull():
        logging.warning("drag handle icon missing or unreadable: %s", DRAG_HANDLE_PATH)
        return pixmap
    return pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
