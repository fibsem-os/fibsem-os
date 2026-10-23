"""Base class for canvas overlays.

Overlays implement a simple duck-typed protocol::

    class MyOverlay:
        def attach(self, ax, canvas: FibsemCanvasBase) -> None: ...
        def detach(self) -> None: ...
        def on_content_changed(self, rect: ContentRect) -> None: ...

``rect`` is the region the canvas is showing, as a
:class:`~fibsem.ui.widgets.canvas.canvas_base.ContentRect`. Overlays that need bounds
or a placement anchor should take them from it — ``rect.cx`` / ``rect.cy`` to centre,
``rect.x0`` / ``rect.x1`` to clamp — rather than assuming content starts at (0, 0).
That is what lets one overlay serve both a canvas holding a single image at the origin
and one holding many images placed in stage space.

Overlays that need Qt signals extend QObject directly.  An overlay that wants
to suppress canvas pan/zoom during a drag sets ``canvas._overlay_consuming_event = True``
on button-press; the canvas clears the flag automatically on button-release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect, FibsemCanvasBase


_QT_MODIFIER_MAP = (
    (Qt.AltModifier, "Alt"),
    (Qt.ShiftModifier, "Shift"),
    (Qt.ControlModifier, "Control"),
    (Qt.MetaModifier, "Meta"),
)


def _modifiers_from_event(event) -> Tuple[str, ...]:
    """Active keyboard modifiers as napari-style strings, e.g. ``("Alt",)``.

    Reads the underlying Qt event (``event.guiEvent``) — the Qt modifier state is
    the reliable source in an embedded canvas, whereas matplotlib's
    ``MouseEvent.key`` depends on canvas keyboard focus.  Falls back to the
    application-wide modifier state when no Qt event is attached.
    """
    gui = getattr(event, "guiEvent", None)
    mods = gui.modifiers() if gui is not None else QApplication.keyboardModifiers()
    return tuple(name for flag, name in _QT_MODIFIER_MAP if mods & flag)


class CanvasOverlay:
    """No-op base for canvas overlays.  Sub-class or use duck-typing."""

    def attach(self, ax, canvas: "FibsemCanvasBase") -> None:
        """Called once when the overlay is added.  Create artists / connect events."""

    def detach(self) -> None:
        """Remove all artists and disconnect mpl events."""

    def on_content_changed(self, rect: "ContentRect") -> None:
        """Called after ax.cla() + new content drawn.  Re-create artists here.

        Defaults to forwarding to the older :meth:`on_image_changed`, so an overlay
        written against the previous protocol keeps working without modification.
        """
        self.on_image_changed(rect.width, rect.height)

    def on_image_changed(self, width: int, height: int) -> None:
        """Deprecated — implement :meth:`on_content_changed` instead.

        Retained so overlays predating the rect-based protocol still receive updates;
        the canvas calls it only when an overlay defines no ``on_content_changed``.
        """
