"""Base class for FibsemImageCanvas overlays.

Overlays implement a simple duck-typed protocol::

    class MyOverlay:
        def attach(self, ax, canvas: FibsemImageCanvas) -> None: ...
        def detach(self) -> None: ...
        def on_image_changed(self, width: int, height: int) -> None: ...

Overlays that need Qt signals extend QObject directly.  An overlay that wants
to suppress canvas pan/zoom during a drag sets ``canvas._overlay_consuming_event = True``
on button-press; the canvas clears the flag automatically on button-release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.image_canvas import FibsemImageCanvas


class CanvasOverlay:
    """No-op base for canvas overlays.  Sub-class or use duck-typing."""

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        """Called once when the overlay is added.  Create artists / connect events."""

    def detach(self) -> None:
        """Remove all artists and disconnect mpl events."""

    def on_image_changed(self, width: int, height: int) -> None:
        """Called after ax.cla() + new image drawn.  Re-create artists here."""
