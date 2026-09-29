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
selected-stage highlight, background stages.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Sequence

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage

from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.milling.patterning.shapes import (
    COLOURS,
    convert_pattern_to_napari_line,
    convert_pattern_to_napari_polygon,
    convert_pattern_to_napari_rect,
)
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    Point,
)
from fibsem.ui.tokens import (
    CANVAS_BG,
)
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_logger = logging.getLogger(__name__)

_CROSSHAIR_HALF_PX = 20  # crosshair arm half-length, pixels
_PATTERN_ZORDER = 6
_FILL_ALPHA = 0.4  # semi-transparent fill; edge stays solid
_LINEWIDTH = 1.0  # default pattern edge width
_LINEWIDTH_SELECTED = 2.5  # selected stage edge width
_BACKGROUND_COLOUR = "black"  # background milling stages
_EXCLUSION_COLOUR = "black"  # exclusion shapes, whatever their stage's colour


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

    def on_content_changed(self, rect: "ContentRect") -> None:
        # ax was cleared + new content drawn; re-create artists from cached stages
        self._remove_artists()
        if (
            not rect.is_empty
            and (self._stages or self._background_stages)
            and self._image is not None
        ):
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
                    stage,
                    shape,
                    pixelsize,
                    _BACKGROUND_COLOUR,
                    linewidth=_LINEWIDTH,
                    zorder=_PATTERN_ZORDER - 2,
                )
            except Exception:
                _logger.exception(
                    "MillingPatternOverlay: failed to draw background stage"
                )

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
            facecolor=CANVAS_BG,
            edgecolor="#555555",
            labelcolor="#d1d2d4",
            framealpha=0.85,
        )
        self._legend.set_zorder(10)

    def _draw_stage(
        self,
        stage,
        shape,
        pixelsize: float,
        colour: str,
        *,
        linewidth: float,
        zorder: float,
    ) -> None:
        for pattern_settings in stage.define_patterns():
            for artist in self._shape_to_artists(
                pattern_settings, shape, pixelsize, colour, linewidth, zorder
            ):
                self._ax.add_artist(artist)
                self._artists.append(artist)
        self._draw_crosshair(
            stage.pattern.point, shape, pixelsize, colour, zorder + 0.5
        )

    def _shape_to_artists(
        self, ps, shape, pixelsize: float, colour: str, linewidth: float, zorder: float
    ) -> List:
        """Artists for one pattern shape (empty when the shape is unsupported).

        Most shapes are a single patch; a bitmap is an outline plus the image
        drawn inside it, hence the list.
        """
        if getattr(ps, "is_exclusion", False):
            # Exclusion zones are the region the mill must not touch. Black in the
            # napari path and in the report plot; black here too. Lines have no
            # is_exclusion, hence the getattr.
            colour = _EXCLUSION_COLOUR
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
            return [
                mpatches.Polygon(verts[:, ::-1], closed=True, **patch_kw)
            ]  # (y,x)→(x,y)
        if isinstance(ps, FibsemBitmapSettings):
            return self._bitmap_artists(ps, shape, pixelsize, colour, linewidth, zorder)
        if isinstance(ps, FibsemCircleSettings):
            return self._circle_artists(ps, shape, pixelsize, patch_kw)
        if isinstance(ps, FibsemLineSettings):
            verts, _ = convert_pattern_to_napari_line(ps, shape, pixelsize)
            (y0, x0), (y1, x1) = verts
            return [
                mlines.Line2D(
                    [x0, x1],
                    [y0, y1],
                    color=colour,
                    linewidth=linewidth,
                    zorder=zorder,
                )
            ]
        if isinstance(ps, FibsemPolygonSettings):
            verts, _ = convert_pattern_to_napari_polygon(ps, shape, pixelsize)
            return [mpatches.Polygon(verts[:, ::-1], closed=True, **patch_kw)]
        return []  # annulus / unknown — deferred

    def _circle_artists(
        self, ps: FibsemCircleSettings, shape, pixelsize: float, patch_kw
    ) -> List:
        """Circle, annulus (``thickness``) or wedge (partial angles), as the shape asks.

        ``thickness`` measures inward from ``radius``, so the ring runs from
        ``radius - thickness`` to ``radius`` — matplotlib's ``Annulus`` takes that as
        the *outer* radius plus the width, not the inner radius.
        """
        centre = microscope_image_to_image_coordinates(
            Point(x=ps.centre_x, y=ps.centre_y), shape, pixelsize
        )
        radius = ps.radius / pixelsize
        thickness = min(ps.thickness, ps.radius) / pixelsize if ps.thickness > 0 else 0

        if thickness > 0:
            return [
                mpatches.Annulus(
                    (centre.x, centre.y),
                    r=radius,
                    width=thickness,
                    angle=math.degrees(-ps.rotation),
                    **patch_kw,
                )
            ]
        if ps.start_angle != 0 or ps.end_angle != 360:
            return [
                mpatches.Wedge(
                    (centre.x, centre.y),
                    r=radius,
                    theta1=ps.start_angle,
                    theta2=ps.end_angle,
                    **patch_kw,
                )
            ]
        return [mpatches.Circle((centre.x, centre.y), radius, **patch_kw)]

    def _bitmap_artists(
        self,
        ps: FibsemBitmapSettings,
        shape,
        pixelsize: float,
        colour: str,
        linewidth: float,
        zorder: float,
    ) -> List:
        """Outline + colourised bitmap image for one bitmap pattern.

        The image is an :class:`AxesImage` on the unit square, mapped onto the
        outline through the rectangle's own transform, so rotation and size come
        out of the patch rather than being applied to the pixel data. Row 0 of the
        bitmap is drawn at the top of the pattern, matching AutoScript's top-left
        origin; ``flip_y`` mirrors it. It is built
        directly instead of via ``ax.imshow`` on purpose: ``imshow`` routes through
        ``add_image`` → ``update_datalim``, which would autoscale the axes to the
        image and throw away the user's pan/zoom.
        """
        # Imported here, not at module import: it pulls in pyplot + skimage, which
        # the rest of this overlay does not need.
        from fibsem.milling.patterning.plotting import bitmap_to_rgba

        centre = microscope_image_to_image_coordinates(
            Point(x=ps.centre_x, y=ps.centre_y), shape, pixelsize
        )
        width = ps.width / pixelsize
        height = ps.height / pixelsize

        outline = mpatches.Rectangle(
            (centre.x - width / 2, centre.y - height / 2),  # bottom-left corner
            width=width,
            height=height,
            angle=math.degrees(-ps.rotation),
            rotation_point="center",
            edgecolor=colour,
            facecolor="none",
            linewidth=linewidth,
            zorder=zorder,
        )
        outline.set_transform(self._ax.transData)

        # origin="lower": the patch transform maps v=0 to the rectangle's xy corner,
        # which is its TOP edge on the y-inverted image axes, so row 0 of the bitmap
        # has to go at the extent's `bottom` to land at the top of the pattern.
        image = AxesImage(self._ax, extent=(0, 1, 0, 1), origin="lower", zorder=zorder)
        image.set_data(bitmap_to_rgba(ps, width, height, colour, _FILL_ALPHA))
        image.set_transform(
            outline.get_patch_transform() + outline.get_data_transform()
        )
        image.set_clip_path(self._ax.patch)  # add_artist does not clip, add_image does
        return [outline, image]

    def _draw_crosshair(
        self, point, shape, pixelsize: float, colour: str, zorder: float
    ) -> None:
        centre = microscope_image_to_image_coordinates(point, shape, pixelsize)
        cx, cy = centre.x, centre.y
        h = _CROSSHAIR_HALF_PX
        kw = dict(color=colour, linewidth=1, alpha=0.9, zorder=zorder)
        (l1,) = self._ax.plot([cx - h, cx + h], [cy, cy], **kw)
        (l2,) = self._ax.plot([cx, cx], [cy - h, cy + h], **kw)
        self._artists.extend([l1, l2])
