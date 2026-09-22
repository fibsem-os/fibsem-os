"""A cryo grid's bars, drawn where they should be.

The Overview tab has always been able to show the grid bars, as an alignment reference:
does the overview sit where I think it does on the grid? It did that by *generating a
raster* of bars matched to the overview's pixel size, adding it as a napari layer, and
letting you drag it into place with napari's transform tool.

Neither half survives the move to a real-space canvas, and neither needs to. Grid bars
are a regular lattice at a known pitch, and a real-space canvas already knows how many
canvas pixels a metre is -- so the bars can simply be *drawn*, exactly, at any zoom,
with no raster to generate and no pixel size to match. What is lost is the ability to
drag them somewhere other than where they belong, which is tracked separately (FIB-608)
and is a better tool than the one being replaced.

Spacing and width are in metres, and the lattice is centred on a point the caller
supplies -- the grid centre, in practice, which is the stage origin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect, FibsemCanvasBase

logger = logging.getLogger(__name__)

_DEFAULT_COLOUR = "#4dd0e1"
# Light: a full-coverage lattice at a typical pitch covers ~40% of the canvas, and
# at 0.45 it washed the overview out rather than annotating it.
_DEFAULT_ALPHA = 0.28


class GridBarOverlay(CanvasOverlay):
    """A lattice of grid bars, in canvas coordinates.

    Call :meth:`set_lattice` with a centre, a pitch and a bar width -- all already in
    canvas units -- and :meth:`set_visible` to show or hide it. Converting from metres
    is the caller's job, because only the caller knows the frame; see
    :class:`~fibsem.ui.widgets.canvas.stage_frame.StageFrame`.

    Display-only: captures no mouse events, so it coexists with pan/zoom and with
    whatever else the canvas is doing.
    """

    # Bars either side of centre, per axis. The span is whatever the canvas is
    # showing, so zoomed out over a declared working area with a fine pitch this
    # would otherwise be thousands of artists, each redrawn on every pan. Past the
    # cap the bars are closer together than the screen can resolve, so it costs
    # nothing visible.
    MAX_BARS_PER_AXIS = 200

    def __init__(
        self,
        *,
        colour: str = _DEFAULT_COLOUR,
        alpha: float = _DEFAULT_ALPHA,
        zorder: float = 3.0,
    ) -> None:
        self._ax = None
        self._canvas: "FibsemCanvasBase | None" = None
        self._artists: list = []
        self._centre: Optional[Tuple[float, float]] = None
        self._pitch: float = 0.0
        self._bar_width: float = 0.0
        self._visible: bool = False
        self._colour = colour
        self._alpha = alpha
        self._zorder = zorder
        # The rectangle the canvas is showing, so the lattice can be extended to fill
        # it. Bars are drawn across the content rather than for a fixed count: a
        # lattice that stopped at the edge of the first overview would say the grid
        # stops there too.
        self._rect: Optional["ContentRect"] = None

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemCanvasBase") -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._rect = rect
        self._remove_artists()
        if self._visible and not rect.is_empty:
            self._draw()

    # ── public API ────────────────────────────────────────────────────────

    def set_lattice(
        self, centre: Tuple[float, float], pitch: float, bar_width: float
    ) -> None:
        """Place the lattice, in canvas coordinates. A pitch of 0 draws nothing."""
        self._centre = centre
        self._pitch = float(pitch)
        self._bar_width = float(bar_width)
        self._redraw()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the bars without discarding the lattice."""
        visible = bool(visible)
        if visible == self._visible:
            return
        self._visible = visible
        self._redraw()

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def bar_count(self) -> int:
        """How many bars are currently drawn. Two artists per bar (one per axis)."""
        return len(self._artists)

    # ── drawing ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._remove_artists()
        if self._visible:
            self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _remove_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()

    def _offsets(self, half_span: float) -> List[float]:
        """Bar centres either side of the lattice centre, out to *half_span*."""
        if self._pitch <= 0 or half_span <= 0:
            return []
        count = int(half_span / self._pitch)
        if count > self.MAX_BARS_PER_AXIS:
            logger.debug(
                f"Grid bar lattice capped at {self.MAX_BARS_PER_AXIS} per axis "
                f"(pitch would need {count})."
            )
            count = self.MAX_BARS_PER_AXIS
        return [i * self._pitch for i in range(-count, count + 1)]

    def _draw(self) -> None:
        if self._ax is None or self._centre is None or self._rect is None:
            return
        if self._pitch <= 0 or self._bar_width <= 0:
            return

        rect = self._rect
        cx, cy = self._centre
        # Out to the far corner of what is shown, so the lattice reaches the edges
        # whichever way the view is panned.
        half_x = max(abs(rect.x0 - cx), abs(rect.x1 - cx))
        half_y = max(abs(rect.y0 - cy), abs(rect.y1 - cy))

        style = dict(
            color=self._colour,
            alpha=self._alpha,
            zorder=self._zorder,
            solid_capstyle="butt",
        )
        # Drawn as wide lines rather than filled rectangles: matplotlib scales a
        # line's width in *points*, not data units, so a bar would keep its on-screen
        # thickness through a zoom and stop describing a real width. `_bar_width` is
        # in canvas units, so the conversion has to happen against the axes -- see
        # `_linewidth_points`.
        linewidth = self._linewidth_points()
        for offset in self._offsets(half_y):
            (line,) = self._ax.plot(
                [rect.x0, rect.x1],
                [cy + offset, cy + offset],
                linewidth=linewidth,
                **style,
            )
            self._artists.append(line)
        for offset in self._offsets(half_x):
            (line,) = self._ax.plot(
                [cx + offset, cx + offset],
                [rect.y0, rect.y1],
                linewidth=linewidth,
                **style,
            )
            self._artists.append(line)

    def _linewidth_points(self) -> float:
        """The bar width in points, so it tracks the zoom like the image does.

        A bar is a physical thing with a physical width; drawn at a fixed point size it
        would grow relative to the sample as you zoomed out, which is exactly the lie
        an alignment reference must not tell. Falls back to a thin line if the axes
        cannot be measured yet -- visible and honest about being approximate, rather
        than absent.
        """
        try:
            bbox = self._ax.get_window_extent()
            xmin, xmax = self._ax.get_xlim()
            if xmax == xmin or bbox.width <= 0:
                return 1.0
            pixels_per_unit = bbox.width / abs(xmax - xmin)
            # 72 points per inch; matplotlib's dpi converts pixels to inches.
            dpi = self._ax.figure.dpi or 72.0
            return max(0.5, self._bar_width * pixels_per_unit * 72.0 / dpi)
        except Exception as e:
            logger.debug(f"Could not size the grid bars against the axes: {e}")
            return 1.0
