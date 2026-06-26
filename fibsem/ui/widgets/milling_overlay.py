"""Milling pattern overlay for FibsemImageCanvas — display-only.

Renders ``FibsemMillingStage`` patterns (rectangle / circle / line / polygon) on a
:class:`FibsemImageCanvas`, one colour per stage, with a crosshair at each stage's
point-of-interest. Reuses the metres→pixel converters from
``fibsem.ui.napari.patterns`` (which already emit rotated, y-flipped pixel geometry).

Display-only: the overlay captures no mouse events, so it coexists cleanly with
the canvas pan/zoom, double-click-to-move, and right-click menu. Pattern movement
is handled by the host widget (right-click → menu → ``_move_patterns``).

Lifecycle: add it to a canvas once via ``canvas.add_overlay(overlay)``. It draws
nothing until :meth:`set_stages` is called with non-empty stages; :meth:`clear`
(or ``set_stages([], …)``) removes all artists, so it's invisible when there is no
milling.

Deferred (see design doc): direct drag-to-move, FOV rect, alignment area,
selected-stage highlight, background stages, annulus / bitmap shapes.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from fibsem.structures import (
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
)
from fibsem.ui.napari.patterns import (
    COLOURS,
    convert_pattern_to_napari_line,
    convert_pattern_to_napari_polygon,
    convert_pattern_to_napari_rect,
    get_image_pixel_centre,
)
from fibsem.ui.widgets.image_canvas import CanvasOverlay, FibsemImageCanvas

_logger = logging.getLogger(__name__)

_CROSSHAIR_HALF_PX = 20  # crosshair arm half-length, pixels
_PATTERN_ZORDER = 6
_FILL_ALPHA = 0.4  # semi-transparent fill; edge stays solid
_LINEWIDTH = 1.0  # default pattern edge width
_LINEWIDTH_SELECTED = 2.5  # selected stage edge width
_BACKGROUND_COLOUR = "black"  # background milling stages


class MillingPatternOverlay(CanvasOverlay):
    """Display-only overlay rendering milling stage patterns + per-stage crosshairs.

    Call :meth:`set_stages` to (re)draw, :meth:`clear` to hide.
    """

    def __init__(self) -> None:
        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._artists: list = []
        self._stages: list = []
        self._background_stages: list = []
        self._selected_index: Optional[int] = None
        self._image: Optional[FibsemImage] = None
        self._legend = None

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        # ax was cleared + a new image drawn; re-create artists from cached stages
        self._remove_artists()
        if width > 0 and (self._stages or self._background_stages) and self._image is not None:
            self._draw()

    # ── public API ────────────────────────────────────────────────────────

    def set_stages(
        self,
        stages: Sequence,
        image: FibsemImage,
        *,
        background_stages: Sequence = (),
        selected_index: Optional[int] = None,
    ) -> None:
        """Display *stages* against *image*.

        ``background_stages`` are drawn in black behind the foreground stages;
        ``selected_index`` (into *stages*) is drawn with a thicker edge, on top.
        """
        self._stages = list(stages)
        self._background_stages = list(background_stages)
        self._selected_index = selected_index
        self._image = image
        self._remove_artists()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove all pattern artists (no milling → nothing drawn)."""
        self._stages = []
        self._background_stages = []
        self._selected_index = None
        self._remove_artists()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── drawing ───────────────────────────────────────────────────────────

    def _remove_artists(self) -> None:
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

    def _draw(self) -> None:
        if self._ax is None or self._image is None:
            return
        if not self._stages and not self._background_stages:
            return
        if self._image.metadata is None or self._image.metadata.pixel_size is None:
            return
        shape = self._image.data.shape[:2]
        pixelsize = self._image.metadata.pixel_size.x

        # background stages first (black, behind the foreground)
        for stage in self._background_stages:
            try:
                self._draw_stage(
                    stage, shape, pixelsize, _BACKGROUND_COLOUR,
                    linewidth=_LINEWIDTH, zorder=_PATTERN_ZORDER - 2,
                )
            except Exception:
                _logger.exception("MillingPatternOverlay: failed to draw background stage")

        # foreground stages (per-colour; selected is thicker and on top)
        for i, stage in enumerate(self._stages):
            colour = COLOURS[i % len(COLOURS)]
            selected = i == self._selected_index
            linewidth = _LINEWIDTH_SELECTED if selected else _LINEWIDTH
            zorder = _PATTERN_ZORDER + 2 if selected else _PATTERN_ZORDER
            try:
                self._draw_stage(
                    stage, shape, pixelsize, colour, linewidth=linewidth, zorder=zorder
                )
            except Exception:
                _logger.exception(
                    "MillingPatternOverlay: failed to draw stage %r",
                    getattr(stage, "name", i),
                )
        self._draw_legend()

    def _draw_legend(self) -> None:
        """Colour-keyed legend of stage names, top-right."""
        handles = [
            mpatches.Patch(
                facecolor=to_rgba(COLOURS[i % len(COLOURS)], _FILL_ALPHA),
                edgecolor=COLOURS[i % len(COLOURS)],
                label=getattr(stage, "name", f"Stage {i + 1}"),
            )
            for i, stage in enumerate(self._stages)
        ]
        if not handles:
            return
        self._legend = self._ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            facecolor="#1e2124",
            edgecolor="#555555",
            labelcolor="#d1d2d4",
            framealpha=0.85,
        )
        self._legend.set_zorder(10)

    def _draw_stage(self, stage, shape, pixelsize: float, colour: str,
                    *, linewidth: float, zorder: float) -> None:
        for pattern_settings in stage.define_patterns():
            artist = self._shape_to_artist(
                pattern_settings, shape, pixelsize, colour, linewidth, zorder
            )
            if artist is not None:
                self._ax.add_artist(artist)
                self._artists.append(artist)
        self._draw_crosshair(stage.pattern.point, shape, pixelsize, colour, zorder + 0.5)

    def _shape_to_artist(self, ps, shape, pixelsize: float, colour: str,
                         linewidth: float, zorder: float):
        # Solid edge + same-colour fill at _FILL_ALPHA. Independent face/edge
        # alphas via RGBA (a patch-level ``alpha`` would dim the edge too).
        patch_kw = dict(
            edgecolor=colour,
            facecolor=to_rgba(colour, _FILL_ALPHA),
            linewidth=linewidth,
            zorder=zorder,
        )
        if isinstance(ps, FibsemRectangleSettings):
            verts, _ = convert_pattern_to_napari_rect(ps, shape, pixelsize)
            return mpatches.Polygon(verts[:, ::-1], closed=True, **patch_kw)  # (y,x)→(x,y)
        if isinstance(ps, FibsemCircleSettings):
            icy, icx = get_image_pixel_centre(shape)
            cx = icx + ps.centre_x / pixelsize
            cy = icy - ps.centre_y / pixelsize
            return mpatches.Circle((cx, cy), ps.radius / pixelsize, **patch_kw)
        if isinstance(ps, FibsemLineSettings):
            verts, _ = convert_pattern_to_napari_line(ps, shape, pixelsize)
            (y0, x0), (y1, x1) = verts
            return mlines.Line2D(
                [x0, x1], [y0, y1], color=colour, linewidth=linewidth, zorder=zorder,
            )
        if isinstance(ps, FibsemPolygonSettings):
            verts, _ = convert_pattern_to_napari_polygon(ps, shape, pixelsize)
            return mpatches.Polygon(verts[:, ::-1], closed=True, **patch_kw)
        return None  # bitmap / annulus / unknown — deferred

    def _draw_crosshair(self, point, shape, pixelsize: float, colour: str,
                        zorder: float) -> None:
        icy, icx = get_image_pixel_centre(shape)
        cx = icx + point.x / pixelsize
        cy = icy - point.y / pixelsize
        h = _CROSSHAIR_HALF_PX
        kw = dict(color=colour, linewidth=1, alpha=0.9, zorder=zorder)
        (l1,) = self._ax.plot([cx - h, cx + h], [cy, cy], **kw)
        (l2,) = self._ax.plot([cx, cx], [cy - h, cy + h], **kw)
        self._artists.extend([l1, l2])
