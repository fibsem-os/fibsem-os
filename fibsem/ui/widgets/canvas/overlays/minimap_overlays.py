"""Generic shape overlay for the minimap (Overview tab) canvas — display-only.

One :class:`MinimapShapesOverlay` renders a flat list of :class:`ShapeSpec` (rectangle
/ circle / crosshair + optional label) on a :class:`FibsemImageCanvas`. The host widget
groups specs **by entity** (LamellaMarkers / CurrentPosition / ReferenceFrame) into
separate overlay instances so each redraws on its own trigger — unlike the napari
minimap, which grouped by shape *type* (one Shapes layer per geometry kind) and had to
rebuild everything on any change.

Geometry is plain image-pixel coordinates: a spec's ``(cx, cy)`` is the shape centre in
image pixels (col, row), matching the axes' ``extent``. Callers compute those from
``tiled.reproject_stage_positions_onto_image2`` exactly as before — no napari vertex
arrays.

Display-only: captures no mouse events, so it coexists with the canvas pan/zoom +
click-to-move + right-click menu.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.patches as mpatches

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_logger = logging.getLogger(__name__)

_DEFAULT_CROSSHAIR_HALF_PX = 40  # crosshair arm half-length, image pixels
_LABEL_FONTSIZE = 7


@dataclass
class ShapeSpec:
    """One overlay primitive in image-pixel coordinates.

    ``kind`` is ``"rect"`` (uses ``width``/``height``), ``"circle"`` (uses ``radius``),
    or ``"crosshair"`` (a ``+`` of ``crosshair_half`` arms). ``label`` is optional text
    drawn near the shape in the shape's colour.
    """

    kind: str
    cx: float
    cy: float
    color: str
    width: float = 0.0
    height: float = 0.0
    radius: float = 0.0
    label: str = ""


class MinimapShapesOverlay(CanvasOverlay):
    """Display-only overlay rendering a list of :class:`ShapeSpec`.

    Call :meth:`set_shapes` to (re)draw, :meth:`set_visible` to toggle, :meth:`clear`
    to hide. Specs are cached so the overlay redraws itself after an image swap
    (``on_image_changed``).
    """

    def __init__(
        self,
        *,
        zorder: float = 5.0,
        linewidth: float = 1.2,
        crosshair_half_px: int = _DEFAULT_CROSSHAIR_HALF_PX,
    ) -> None:
        self._ax = None
        self._canvas: "FibsemImageCanvas | None" = None
        self._artists: list = []
        self._specs: list[ShapeSpec] = []
        self._visible: bool = True
        self._zorder = zorder
        self._linewidth = linewidth
        self._crosshair_half = crosshair_half_px

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        # ax was cleared + a new image drawn → re-create artists from cached specs
        self._remove_artists()
        if width > 0 and self._specs and self._visible:
            self._draw()

    # ── public API ────────────────────────────────────────────────────────

    def set_shapes(self, specs) -> None:
        """Replace the drawn shapes with *specs* and redraw."""
        self._specs = list(specs)
        self._remove_artists()
        if self._visible:
            self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the whole group without discarding its specs."""
        if visible == self._visible:
            return
        self._visible = visible
        self._remove_artists()
        if visible:
            self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove all shapes (group draws nothing until the next set_shapes)."""
        self.set_shapes([])

    # ── drawing ───────────────────────────────────────────────────────────

    def _remove_artists(self) -> None:
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def _draw(self) -> None:
        if self._ax is None:
            return
        for spec in self._specs:
            try:
                self._draw_spec(spec)
            except Exception:
                _logger.exception("MinimapShapesOverlay: failed to draw %r", spec)

    def _draw_spec(self, spec: ShapeSpec) -> None:
        if spec.kind == "rect":
            patch = mpatches.Rectangle(
                (spec.cx - spec.width / 2.0, spec.cy - spec.height / 2.0),
                spec.width, spec.height,
                fill=False, edgecolor=spec.color, linewidth=self._linewidth,
                zorder=self._zorder,
            )
            self._ax.add_patch(patch)
            self._artists.append(patch)
            self._draw_label(spec, spec.cx - spec.width / 2.0,
                              spec.cy - spec.height / 2.0, va="bottom")
        elif spec.kind == "circle":
            patch = mpatches.Circle(
                (spec.cx, spec.cy), spec.radius,
                fill=False, edgecolor=spec.color, linewidth=self._linewidth,
                zorder=self._zorder,
            )
            self._ax.add_patch(patch)
            self._artists.append(patch)
            self._draw_label(spec, spec.cx, spec.cy - spec.radius, va="bottom")
        elif spec.kind == "crosshair":
            h = self._crosshair_half
            kw = dict(color=spec.color, linewidth=self._linewidth, zorder=self._zorder)
            (l1,) = self._ax.plot([spec.cx - h, spec.cx + h], [spec.cy, spec.cy], **kw)
            (l2,) = self._ax.plot([spec.cx, spec.cx], [spec.cy - h, spec.cy + h], **kw)
            self._artists.extend([l1, l2])
            self._draw_label(spec, spec.cx + h * 0.15, spec.cy + h * 0.15,
                             va="top", ha="left")

    def _draw_label(self, spec: ShapeSpec, x: float, y: float,
                    *, va: str = "bottom", ha: str = "left") -> None:
        if not spec.label:
            return
        t = self._ax.text(
            x, y, spec.label, color=spec.color, fontsize=_LABEL_FONTSIZE,
            ha=ha, va=va, zorder=self._zorder + 0.5, clip_on=True,
        )
        self._artists.append(t)
