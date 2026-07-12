"""Milling-shape pattern overlays (pixel-space) for FibsemImageCanvas.

``PatternOverlay`` — milling pattern shapes as matplotlib patches.
``ScanDirectionArrowOverlay`` — arrow indicating the milling scan direction.
"""

from __future__ import annotations

import logging
from typing import Optional

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_logger = logging.getLogger(__name__)


class PatternOverlay(CanvasOverlay):
    """Renders milling pattern shapes as matplotlib patches.

    Coordinates must be in image pixel space (caller handles unit conversion).
    Supported pattern attributes:
      - Rectangle / Bitmap : ``centre_x, centre_y, width, height``
      - Circle             : ``centre_x, centre_y, radius``
      - Line               : ``start_x, start_y, end_x, end_y``
      - Polygon            : ``vertices`` (N×2 array)
    """

    def __init__(self, patterns=(), color: str = "cyan", alpha: float = 0.4):
        self._patterns = list(patterns)
        self._color = color
        self._alpha = alpha
        self._ax = None
        self._canvas = None
        self._artists: list = []

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._remove_artists()
        if width > 0:
            self._draw()

    def set_patterns(self, patterns) -> None:
        self._patterns = list(patterns)
        self._remove_artists()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artists(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def _draw(self):
        if self._ax is None:
            return
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        kw = dict(
            edgecolor=self._color,
            facecolor=self._color,
            alpha=self._alpha,
            linewidth=1.5,
            zorder=6,
        )
        for pat in self._patterns:
            try:
                artist = _pattern_to_artist(pat, kw, mpatches, mlines)
                if artist is not None:
                    self._ax.add_artist(artist)
                    self._artists.append(artist)
            except Exception:
                _logger.exception("PatternOverlay: failed to render %r", pat)


def _pattern_to_artist(pat, kw, mpatches, mlines):
    name = type(pat).__name__
    if any(k in name for k in ("Rectangle", "Bitmap")):
        return mpatches.Rectangle(
            (pat.centre_x - pat.width / 2, pat.centre_y - pat.height / 2),
            pat.width,
            pat.height,
            **kw,
        )
    if "Circle" in name:
        return mpatches.Circle(
            (pat.centre_x, pat.centre_y),
            pat.radius,
            **{**kw, "facecolor": "none"},
        )
    if "Line" in name:
        return mlines.Line2D(
            [pat.start_x, pat.end_x],
            [pat.start_y, pat.end_y],
            color=kw["edgecolor"],
            linewidth=2,
            alpha=kw["alpha"],
            zorder=kw["zorder"],
        )
    if "Polygon" in name and hasattr(pat, "vertices"):
        return mpatches.Polygon(pat.vertices, closed=True, **kw)
    return None


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

    def on_image_changed(self, width: int, _height: int) -> None:
        self._remove_artist()
        if width > 0 and self._params is not None:
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
