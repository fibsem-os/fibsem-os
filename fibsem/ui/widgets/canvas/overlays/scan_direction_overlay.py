"""``ScanDirectionArrowOverlay``: an arrow on FibsemImageCanvas showing the milling
scan direction, in pixel space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


class ScanDirectionArrowOverlay(CanvasOverlay):
    """Draws a yellow arrow indicating the milling scan direction.

    Only ``"TopToBottom"`` and ``"BottomToTop"`` are supported; all other
    values result in no arrow being shown.

    Call :meth:`set_arrow` with the pattern bounding box in pixel coordinates
    to update the arrow, or :meth:`clear` to hide it.
    """

    _SUPPORTED = {"TopToBottom", "BottomToTop"}

    def __init__(self, color: str = "yellow"):
        self._color = color
        self._ax = None
        self._canvas = None
        self._artist = None
        # (x_start, y_start, x_end, y_end) in pixel coords
        self._params: Optional[tuple] = None

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artist()
        self._ax = None
        self._canvas = None

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._remove_artist()
        if not rect.is_empty and self._params is not None:
            self._draw()

    def set_arrow(self, cx: float, cy: float, h_px: float, scan_direction: str) -> None:
        """Position the arrow based on pattern centre and height.

        Args:
            cx: pattern centre x in pixel coords
            cy: pattern centre y in pixel coords
            h_px: pattern height in pixels
            scan_direction: ``"TopToBottom"`` or ``"BottomToTop"``
        """
        if scan_direction not in self._SUPPORTED:
            self.clear()
            return
        margin = h_px * 0.15
        if scan_direction == "TopToBottom":
            # Arrow from near top edge → near bottom edge (↓ in image coords)
            x_s, y_s = cx, cy - h_px / 2 + margin
            x_e, y_e = cx, cy + h_px / 2 - margin
        else:  # BottomToTop
            # Arrow from near bottom edge → near top edge (↑ in image coords)
            x_s, y_s = cx, cy + h_px / 2 - margin
            x_e, y_e = cx, cy - h_px / 2 + margin
        self._params = (x_s, y_s, x_e, y_e)
        self._remove_artist()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Hide the arrow."""
        self._params = None
        self._remove_artist()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artist(self):
        if self._artist is not None:
            try:
                self._artist.remove()
            except Exception:
                pass
            self._artist = None

    def _draw(self):
        if self._ax is None or self._params is None:
            return
        x_s, y_s, x_e, y_e = self._params
        self._artist = self._ax.annotate(
            "",
            xy=(x_e, y_e),
            xytext=(x_s, y_s),
            arrowprops=dict(
                arrowstyle="-|>",
                color=self._color,
                lw=2.0,
                mutation_scale=18,
            ),
            zorder=10,
        )
