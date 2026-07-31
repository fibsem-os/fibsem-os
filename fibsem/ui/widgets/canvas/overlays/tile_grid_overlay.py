"""Planned tile grid, drawn on the canvas and clickable to include/exclude tiles.

Shows where an overview acquisition will put its tiles before it runs, and lets a tile
be toggled by clicking it. The same selection is editable in `TileMaskWidget`; this is
a second view of that one mask, not a second copy of it.

Placement comes from `TilePosition.canvas_x/canvas_y` -- the pixel offsets
`stitch_tileset` pastes each tile at -- rather than from the metre offsets `dx/dy`.
Both would work, but only one of them is the number the stitcher actually uses, so
this way the preview cannot disagree with the mosaic it is previewing. It also avoids
restating the sign convention: the runner applies `dx - offset` but `dy + offset`, and
getting that backwards would draw a grid that looks plausible and is upside down.

The grid is a regular lattice in the displayed frame at any stage tilt, because tiles
are *defined* by displayed-frame offsets handed to `project_fm_stable_move`. The tilt
is absorbed into the stage moves, not into where the tiles appear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QObject, Qt, pyqtSignal

from fibsem.imaging.tiling.geometry import TilePosition
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

# Magenta rather than the app's blue: this is drawn over fluorescence data, which is
# routinely cyan, green or blue, and an overlay the same hue as the sample is one you
# have to hunt for. Magenta is the one corner of the wheel FM channels rarely occupy.
ENABLED_EDGE = "#ff3ec8"
# No fill by default: filled squares hide the data they are drawn over, and the tile
# boundaries -- which are the thing worth seeing -- are the outlines, not the areas.
# The fill is still available from the options panel, where it doubles as an overlap
# map (two overlapping tiles read brighter than one), but it is opt-in.
ENABLED_ALPHA = 0.0
# Grey, so a skipped tile reads as inactive at a glance rather than as a differently
# dashed active one. Distinguished by colour *and* line style, since either alone is
# easy to miss in a large grid.
DISABLED_EDGE = "#9aa0a6"
DISABLED_ALPHA = 0.75
LINE_WIDTH = 0.8

# Grab zone for an edge drag, as a fraction of a tile. Proportional rather than a
# fixed pixel count so it stays usable however far the view is zoomed out.
EDGE_GRAB_FRACTION = 0.08
MAX_TILES_PER_AXIS = 100  # matches the row/column spin boxes


class TileGridOverlay(QObject, CanvasOverlay):
    """The planned tile grid, with click-to-toggle.

    Emits :attr:`tile_toggled` rather than mutating anything: the mask belongs to the
    settings widget, and two widgets writing the same state is how they drift apart.
    """

    tile_toggled = pyqtSignal(int, int, bool)      # row, col, enabled
    grid_resize_requested = pyqtSignal(int, int)   # rows, cols

    def __init__(self, parent: Optional[QObject] = None) -> None:
        QObject.__init__(self, parent)
        self._tiles: List[TilePosition] = []
        self._tile_shape: Tuple[int, int] = (0, 0)     # (height, width) px
        self._tile_pixel_size: float = 0.0
        self._display_pixel_size: Optional[float] = None
        self._overlap: float = 0.0
        self._color: str = ENABLED_EDGE
        self._fill_alpha: float = ENABLED_ALPHA
        self._visible: bool = True

        self._ax = None
        self._canvas: Optional["FibsemImageCanvas"] = None
        self._artists: list = []
        self._img_w: int = 0
        self._img_h: int = 0
        self._previous_margin: Optional[float] = None
        self._cids: List[int] = []
        self._resize_axes: Optional[Tuple[bool, bool]] = None  # (horizontal, vertical)
        self._cursor_set: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas
        canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_drag),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            try:
                self._canvas.canvas_clicked.disconnect(self._on_canvas_clicked)
            except TypeError:
                pass
            for cid in self._cids:
                self._canvas.mpl_disconnect(cid)
            self._cids = []
            if self._cursor_set:
                self._canvas.unsetCursor()
                self._cursor_set = False
            # The margin is a property of the canvas, not of this overlay, so leaving
            # it widened would keep every later image zoomed out for no visible reason.
            self._restore_margin()
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        self._redraw()

    # ── content ──────────────────────────────────────────────────────────

    def set_grid(
        self,
        tiles: Sequence[TilePosition],
        tile_shape: Tuple[int, int],
        tile_pixel_size: float,
        display_pixel_size: Optional[float] = None,
        overlap: float = 0.0,
    ) -> None:
        """Set the grid to draw.

        Args:
            tiles: the planned grid, as returned by the tiling geometry core.
            tile_shape: one tile's (height, width), in pixels.
            tile_pixel_size: the pixel size those dimensions are in, in metres.
            display_pixel_size: pixel size of the image being drawn on. Defaults to
                `tile_pixel_size` -- the usual case, since the canvas is showing a
                frame from the same camera.
            overlap: fraction adjacent tiles share. Passed rather than derived from
                the tile spacing, because a single-row or single-column grid has no
                spacing to derive it from -- and that is exactly the case a resize
                drag has to grow out of.
        """
        self._tiles = list(tiles)
        self._tile_shape = tile_shape
        self._tile_pixel_size = tile_pixel_size
        self._display_pixel_size = display_pixel_size
        self._overlap = overlap
        self._redraw()

    def clear(self) -> None:
        self._tiles = []
        self._redraw()

    def set_color(self, color: str) -> None:
        """Recolour the grid.

        Exposed because no single colour works over every sample: the default avoids
        the hues FM channels usually occupy, but a magenta channel makes a magenta
        grid invisible, and only the person looking at the data knows that.
        """
        self._color = color
        self._redraw()

    def set_fill_alpha(self, alpha: float) -> None:
        """Opacity of the enabled-tile fill. 0 leaves outlines only."""
        self._fill_alpha = max(0.0, min(1.0, float(alpha)))
        self._redraw()

    def set_grid_visible(self, visible: bool) -> None:
        """Hide the grid without discarding it, for an unobstructed look at the data."""
        self._visible = bool(visible)
        self._redraw()

    @property
    def is_grid_visible(self) -> bool:
        return self._visible

    @property
    def is_resizing(self) -> bool:
        """True while an edge is being dragged.

        The host needs this: resizing changes the grid's footprint, and refitting the
        view on every step of a drag would zoom the canvas out from under the cursor.
        """
        return self._resize_axes is not None

    def fit_view(self, padding: float = 0.05) -> None:
        """Zoom out so the whole grid is visible, not just the tile at the centre.

        A 3x3 at 10% overlap spans ~2.8x the field of view per axis, so at the default
        view the outer tiles are off-screen -- and an outer tile that cannot be seen
        cannot be clicked.
        """
        if self._canvas is None or not self._tiles or self._img_w <= 0:
            return

        span_w, span_h = self._extent()
        if span_w <= 0 or span_h <= 0:
            return

        # `set_view_margin` is a fraction of the *image* size per side.
        margin = max(
            (span_w / self._img_w - 1.0) / 2.0,
            (span_h / self._img_h - 1.0) / 2.0,
            0.0,
        )
        if self._previous_margin is None:
            self._previous_margin = getattr(self._canvas, "_view_margin", 0.0)
        self._canvas.set_view_margin(margin + padding)

    # ── drawing ──────────────────────────────────────────────────────────

    def _scale(self) -> float:
        """Tile pixels per displayed pixel.

        Read from the canvas at draw time rather than cached at `set_grid` time: the
        displayed image changes underneath the grid during an acquisition -- the live
        preview is a decimated mosaic, so its pixels are `preview_stride` times coarser
        than a tile's -- and a scale captured before that swap draws the grid stride
        times too large over the very image it is meant to describe.
        """
        display = self._display_pixel_size
        if display is None and self._canvas is not None:
            display = getattr(self._canvas, "_pixel_size", None)
        display = display or self._tile_pixel_size
        if not display:
            return 1.0
        return self._tile_pixel_size / display

    def _extent(self) -> Tuple[float, float]:
        """The whole grid's size in display pixels."""
        if not self._tiles:
            return 0.0, 0.0
        tile_h, tile_w = self._tile_shape
        scale = self._scale()
        span_w = (max(t.canvas_x for t in self._tiles) + tile_w) * scale
        span_h = (max(t.canvas_y for t in self._tiles) + tile_h) * scale
        return span_w, span_h

    def _rect_for(self, tile: TilePosition) -> Tuple[float, float, float, float]:
        """(x, y, width, height) of one tile, in display pixels."""
        tile_h, tile_w = self._tile_shape
        scale = self._scale()
        span_w, span_h = self._extent()

        # The grid is centred on the image centre, matching the runner: it measures
        # from the top-left tile and shifts so the grid straddles the start position.
        origin_x = self._img_w / 2 - span_w / 2
        origin_y = self._img_h / 2 - span_h / 2

        return (
            origin_x + tile.canvas_x * scale,
            origin_y + tile.canvas_y * scale,
            tile_w * scale,
            tile_h * scale,
        )

    def _redraw(self) -> None:
        self._remove_artists()
        if self._ax is None or not self._tiles or self._img_w <= 0 or not self._visible:
            # Still repaint: removing the artists takes them out of the axes, but the
            # canvas keeps showing the last render until it is asked to redraw -- so
            # returning early here left a hidden grid on screen.
            if self._canvas is not None:
                self._canvas.draw_idle()
            return

        from matplotlib.colors import to_rgba
        from matplotlib.patches import Rectangle

        for tile in self._tiles:
            x, y, width, height = self._rect_for(tile)
            # Alpha is baked into the face colour rather than set on the patch:
            # `alpha=` applies to the edge as well, which drew the outlines at the
            # fill's opacity and left them all but invisible -- so the grid could only
            # be seen by turning the fill up until it obscured the data.
            patch = Rectangle(
                (x, y), width, height,
                fill=tile.enabled,
                facecolor=(
                    to_rgba(self._color, self._fill_alpha) if tile.enabled else "none"
                ),
                edgecolor=(
                    to_rgba(self._color, 1.0) if tile.enabled
                    else to_rgba(DISABLED_EDGE, DISABLED_ALPHA)
                ),
                linestyle="-" if tile.enabled else "--",
                linewidth=LINE_WIDTH,
                zorder=20,
            )
            self._ax.add_patch(patch)
            self._artists.append(patch)

        if self._canvas is not None:
            self._canvas.draw_idle()

    def _remove_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._artists = []

    # ── interaction ──────────────────────────────────────────────────────

    def _on_canvas_clicked(self, x: float, y: float, modifiers) -> None:
        tile = self._tile_at(x, y)
        if tile is None:
            return
        self.tile_toggled.emit(tile.row, tile.col, not tile.enabled)

    def _tile_at(self, x: float, y: float) -> Optional[TilePosition]:
        """The tile under a point, or None.

        Nearest centre among the tiles containing the point, rather than the first
        match: tiles overlap by design, so a click in a seam is inside two of them and
        "first in the list" would make the answer depend on the traversal order.
        """
        hits = []
        for tile in self._tiles:
            left, top, width, height = self._rect_for(tile)
            if left <= x <= left + width and top <= y <= top + height:
                centre_x, centre_y = left + width / 2, top + height / 2
                hits.append(((centre_x - x) ** 2 + (centre_y - y) ** 2, tile))
        if not hits:
            return None
        return min(hits, key=lambda item: item[0])[1]

    # ── resize by dragging an edge ───────────────────────────────────────

    def _edge_sides(self, x: float, y: float) -> Tuple[int, int]:
        """Which outer edges a point is on, as (x_side, y_side).

        -1 for the low edge, +1 for the high one, 0 for neither. Signed rather than
        boolean because the hover cursor has to name a *corner* -- top-left and
        top-right want opposite diagonals -- and "on both axes" cannot distinguish them.
        """
        if not self._tiles or self._tile_shape[1] <= 0 or self._img_w <= 0:
            return 0, 0

        span_w, span_h = self._extent()
        left = self._img_w / 2 - span_w / 2
        top = self._img_h / 2 - span_h / 2
        # Proportional to a tile rather than a fixed pixel count, so the grab zone
        # stays usable at any zoom without querying the transform.
        tolerance = self._tile_shape[1] * self._scale() * EDGE_GRAB_FRACTION

        # Only count an edge if the point is actually alongside the grid, otherwise
        # the extended lines of the bounding box grab far outside it.
        within_x = left - tolerance <= x <= left + span_w + tolerance
        within_y = top - tolerance <= y <= top + span_h + tolerance

        x_side = 0
        if within_y:
            if abs(x - left) <= tolerance:
                x_side = -1
            elif abs(x - (left + span_w)) <= tolerance:
                x_side = 1

        y_side = 0
        if within_x:
            if abs(y - top) <= tolerance:
                y_side = -1
            elif abs(y - (top + span_h)) <= tolerance:
                y_side = 1

        return x_side, y_side

    def _edge_at(self, x: float, y: float) -> Optional[Tuple[bool, bool]]:
        """Which outer edges a point grabs, as (horizontal, vertical), or None.

        A corner returns both, so dragging it resizes each axis at once.
        """
        x_side, y_side = self._edge_sides(x, y)
        if not (x_side or y_side):
            return None
        return bool(x_side), bool(y_side)

    def _count_for(self, distance: float, tile_extent: float) -> int:
        """Rows or columns whose half-extent best matches a distance from the centre.

        The grid is centred on the stage position, so it can only grow symmetrically:
        `half = ((n - 1) * step + tile) / 2`. Inverting that is what makes the dragged
        edge follow the cursor -- and why the count ticks up every *half* step, one
        tile appearing on each side.
        """
        step = tile_extent * (1.0 - self._overlap)
        if step <= 0:
            return 1
        count = round((2.0 * distance - tile_extent) / step) + 1
        return max(1, min(MAX_TILES_PER_AXIS, int(count)))

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self._ax or event.xdata is None:
            return
        if not self._visible:
            return
        edges = self._edge_at(event.xdata, event.ydata)
        if edges is None:
            return

        self._resize_axes = edges
        if self._canvas is not None:
            # Tells the canvas an overlay owns this gesture: it cancels the pending pan
            # and suppresses the click that would otherwise toggle a tile on release.
            self._canvas._overlay_consuming_event = True

    def _cursor_for(self, x_side: int, y_side: int):
        """The resize cursor naming what dragging here would do, or None."""
        if x_side and y_side:
            # Opposite corners share a diagonal: top-left/bottom-right lie along "\",
            # top-right/bottom-left along "/".
            return Qt.SizeFDiagCursor if x_side == y_side else Qt.SizeBDiagCursor
        if x_side:
            return Qt.SizeHorCursor
        if y_side:
            return Qt.SizeVerCursor
        return None

    def _update_cursor(self, event) -> None:
        """Show a resize cursor over the grid's edges.

        Without it the drag is undiscoverable -- nothing on screen suggests the
        boundary is draggable, and no one hunts for an affordance they have no reason
        to believe exists.
        """
        if self._canvas is None:
            return

        cursor = None
        if (
            self._visible
            and self._tiles
            and event.inaxes is self._ax
            and event.xdata is not None
        ):
            cursor = self._cursor_for(*self._edge_sides(event.xdata, event.ydata))

        if cursor is not None:
            self._canvas.setCursor(cursor)
            self._cursor_set = True
        elif self._cursor_set:
            # Only unset a cursor this overlay set, so it cannot clobber one another
            # overlay is showing.
            self._canvas.unsetCursor()
            self._cursor_set = False

    def _on_drag(self, event) -> None:
        if self._resize_axes is None:
            self._update_cursor(event)
            return
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        horizontal, vertical = self._resize_axes
        tile_h, tile_w = self._tile_shape
        scale = self._scale()
        rows = max(t.row for t in self._tiles) + 1
        cols = max(t.col for t in self._tiles) + 1

        if horizontal:
            cols = self._count_for(
                abs(event.xdata - self._img_w / 2), tile_w * scale
            )
        if vertical:
            rows = self._count_for(
                abs(event.ydata - self._img_h / 2), tile_h * scale
            )

        self.grid_resize_requested.emit(rows, cols)

    def _on_release(self, event) -> None:
        self._resize_axes = None
        self._update_cursor(event)

    def _restore_margin(self) -> None:
        if self._previous_margin is not None and self._canvas is not None:
            self._canvas.set_view_margin(self._previous_margin)
            self._previous_margin = None
