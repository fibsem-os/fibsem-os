"""A cryo grid's bars, drawn where they should be -- and dragged to where they are.

The Overview tab has always been able to show the grid bars, as an alignment reference:
does the overview sit where I think it does on the grid? It did that by *generating a
raster* of bars matched to the overview's pixel size, adding it as a napari layer, and
letting you drag it into place with napari's transform tool.

Neither half survives the move to a real-space canvas, and neither needs to. Grid bars
are a regular lattice at a known pitch, and a real-space canvas already knows how many
canvas pixels a metre is -- so the bars can simply be *drawn*, exactly, at any zoom,
with no raster to generate and no pixel size to match. Dragging them somewhere other
than where the holder says they belong is the gesture in
:class:`~fibsem.ui.widgets.canvas.overlays.transform_overlay.TransformGestureOverlay`:
move by dragging, turn by the handle, only while the canvas has made this its active
overlay (FIB-608).

The lattice is square on the sample and drawn through the view: a beam looking at the
sample from off its normal sees the pitch along the squashed axis shortened by the
view's foreshortening, so the bars land on the bars in the picture in every view rather
than in the two the tab was built against (FIB-615). Spacing and width are in canvas
units along the unsquashed axis; the squash is the caller's, read off its frame.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from fibsem.ui.widgets.canvas.overlays.transform_overlay import (
    TransformGestureOverlay,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from PyQt5.QtCore import QObject

logger = logging.getLogger(__name__)

_DEFAULT_COLOUR = "#4dd0e1"
# A faint fill with a crisp edge: a bar is aligned by its edges, and a solid fill at
# a real bar's width covers a third of the picture it is meant to be checked against.
_DEFAULT_FILL_ALPHA = 0.08
_DEFAULT_EDGE_ALPHA = 0.55


class GridBarOverlay(TransformGestureOverlay):
    """A lattice of grid bars, in canvas coordinates.

    Call :meth:`set_lattice` with a centre, a pitch and a bar width -- all already in
    canvas units -- plus the rotation and the view's squash, and :meth:`set_visible`
    to show or hide it. Converting from metres is the caller's job, because only the
    caller knows the frame; see
    :class:`~fibsem.ui.widgets.canvas.stage_frame.StageFrame`.

    Display-only until the canvas makes it the active overlay, so it coexists with
    pan/zoom and with whatever else the canvas is doing.
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
        fill_alpha: float = _DEFAULT_FILL_ALPHA,
        edge_alpha: float = _DEFAULT_EDGE_ALPHA,
        zorder: float = 3.0,
        parent: Optional["QObject"] = None,
    ) -> None:
        super().__init__(parent)
        self._pitch: float = 0.0
        self._bar_width: float = 0.0
        self._radius: Optional[float] = None
        self._colour = colour
        self._fill_alpha = fill_alpha
        self._edge_alpha = edge_alpha
        self._zorder = zorder

    # ── public API ────────────────────────────────────────────────────────

    def set_lattice(
        self,
        centre: Tuple[float, float],
        pitch: float,
        bar_width: float,
        rotation: float = 0.0,
        squash: float = 1.0,
        radius: Optional[float] = None,
    ) -> None:
        """Place the lattice, in canvas coordinates. A pitch of 0 draws nothing.

        `pitch`, `bar_width` and `radius` are along the unsquashed axis; `squash` is
        the view's surface foreshortening, applied along canvas y; `rotation` is
        degrees, clockwise on screen. `radius` clips the bars to the grid's own disc
        about the lattice centre -- a grid's bars stop at its rim -- and None lets
        them run to the edge of whatever is shown.
        """
        self._pitch = float(pitch)
        self._bar_width = float(bar_width)
        self._radius = float(radius) if radius else None
        self.set_placement(centre, rotation, squash)

    @property
    def pitch(self) -> float:
        return self._pitch

    @property
    def radius(self) -> Optional[float]:
        return self._radius

    @property
    def bar_count(self) -> int:
        """How many bars are currently drawn. One artist per bar."""
        return sum(1 for a in self._artists if getattr(a, "_gridbar", False))

    # ── drawing ───────────────────────────────────────────────────────────

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

    def _half_span(self) -> float:
        """How far, in the lattice's own frame, a bar has to reach to cross the view.

        Out to the farthest corner of what is shown, so the lattice reaches the edges
        whichever way the view is panned and however the lattice is turned. Measured
        in the lattice frame -- the corners brought back through the inverse map --
        so a squashed view asks for the bars it can actually see.
        """
        rect = self._rect
        corners = (
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x0, rect.y1),
            (rect.x1, rect.y1),
        )
        return max(math.hypot(*self.from_canvas(x, y)) for x, y in corners)

    def _draw_body(self, animated: bool) -> list:
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Polygon

        if self._pitch <= 0 or self._bar_width <= 0:
            return []
        half = self._radius if self._radius is not None else self._half_span()
        style = dict(
            closed=True,
            facecolor=to_rgba(self._colour, self._fill_alpha),
            edgecolor=to_rgba(self._colour, self._edge_alpha),
            linewidth=1.0,
            zorder=self._zorder,
            animated=animated,
        )
        # Polygons in canvas units rather than wide lines: a line's width is in
        # points, so it only described the bar's real width at the zoom it was drawn
        # at, and a zoom later the bars were the wrong size until something redrew
        # them. A polygon scales with the picture, and its edge is what gets aligned.
        w = self._bar_width / 2.0
        artists = []
        for offset in self._offsets(half):
            if self._radius is not None:
                # The chord of the disc at this offset: bars stop at the grid's rim.
                reach = math.sqrt(max(self._radius**2 - offset**2, 0.0))
                if reach <= 0:
                    continue
            else:
                reach = half
            for corners in (
                # A bar of constant u runs along v ...
                (
                    (offset - w, -reach),
                    (offset + w, -reach),
                    (offset + w, reach),
                    (offset - w, reach),
                ),
                # ... and a bar of constant v runs along u.
                (
                    (-reach, offset - w),
                    (reach, offset - w),
                    (reach, offset + w),
                    (-reach, offset + w),
                ),
            ):
                bar = Polygon([self.to_canvas(u, v) for u, v in corners], **style)
                bar._gridbar = True
                self._ax.add_patch(bar)
                artists.append(bar)
        return artists
