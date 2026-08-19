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

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from fibsem.imaging.tiling.geometry import TilePosition
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect
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

# How far the pointer must travel before a press inside the grid counts as a move
# rather than a click on a tile. In screen pixels, so the split does not shift with
# the zoom, and the same 3 the canvas uses for its own click-versus-drag test -- one
# threshold everywhere means a gesture cannot read as a click to one and a drag to
# the other.
MOVE_DRAG_THRESHOLD_PX = 3

# Held to make an interior drag paint tiles rather than move the grid. Shift because
# it is the only modifier not already spent inside the axes: Ctrl and Alt reach the
# toolbar, and the one Shift binding -- Shift+scroll to focus -- is a wheel gesture,
# which cannot collide with a button-1 drag.
PAINT_MODIFIER = "Shift"


class TileGridOverlay(QObject, CanvasOverlay):
    """The planned tile grid: click-to-toggle, Shift+drag-to-paint, edge-to-resize,
    drag-to-move.

    Emits rather than mutating: rows, columns, the mask and the position the grid is
    planned around all belong to the settings widget, and two widgets writing the same
    state is how they drift apart.
    """

    tile_toggled = pyqtSignal(int, int, bool)      # row, col, enabled
    grid_resize_requested = pyqtSignal(int, int)   # rows, cols
    grid_move_requested = pyqtSignal(float, float)  # new centre, canvas coordinates

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
        self._rect: Optional["ContentRect"] = None  # canvas content bounds
        self._anchor_point: Optional[Tuple[float, float]] = None  # see set_anchor
        self._previous_margin: Optional[float] = None
        self._cids: List[int] = []
        self._resize_axes: Optional[Tuple[bool, bool]] = None  # (horizontal, vertical)
        # Whether dragging may change the grid's *geometry*. Tile toggling is separate
        # and always allowed -- see `set_grid_editable`.
        self._geometry_editable: bool = True
        self._cursor_set: bool = False
        # A press inside the grid, held until it resolves into a move or a tile click:
        # (press_px_x, press_px_y, press_x, press_y, anchor_x, anchor_y). The screen
        # coordinates drive the threshold, the data coordinates the displacement.
        self._move_start: Optional[Tuple[float, float, float, float, float, float]] = None
        self._move_active: bool = False
        # A paint drag: the value being swept in, the tile the press landed on, and the
        # tiles already given it. `_paint_value` doubles as "this press is a paint" --
        # it is set only at press, so letting go of Shift part way through does not
        # abandon the sweep half-done.
        self._paint_value: Optional[bool] = None
        self._paint_origin: Optional[Tuple[int, int]] = None
        self._paint_active: bool = False
        self._painted: set = set()
        # A toggle waiting to find out whether it was half of a double-click. See
        # `_schedule_toggle` for why it waits rather than firing and being undone.
        self._pending_toggle: Optional[Tuple[int, int, bool]] = None
        self._toggle_timer = QTimer(self)
        self._toggle_timer.setSingleShot(True)
        self._toggle_timer.timeout.connect(self._emit_pending_toggle)

    # ── lifecycle ────────────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas
        # Raw press/motion/release rather than the canvas's `canvas_clicked`: a press
        # inside the grid can turn into either a tile toggle or a move of the whole
        # grid, and only the pointer's subsequent travel says which -- a signal that
        # has already decided it was a click arrives too late to tell them apart.
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_drag),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def detach(self) -> None:
        # A toggle still waiting out the double-click interval would otherwise fire into
        # a host that has stopped listening -- or, on a rebuilt tab, into the wrong one.
        self._cancel_pending_toggle()
        # Same for a gesture caught mid-flight: detaching skips the release that would
        # have cleared these, so a re-attached overlay would resume painting on the
        # next motion event, with a value captured against a mask that is long gone.
        self._paint_value = None
        self._paint_origin = None
        self._paint_active = False
        self._painted.clear()
        if self._canvas is not None:
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

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._rect = rect
        self._redraw()

    def _content(self) -> Optional["ContentRect"]:
        """Content bounds, or None if the canvas has not reported usable ones yet."""
        rect = self._rect
        return None if rect is None or rect.is_empty else rect

    def set_anchor(self, centre: Optional[Tuple[float, float]]) -> None:
        """Centre the grid on *centre* in canvas coordinates, or None to follow content.

        A tileset is planned around a *stage position*, not around whatever is currently
        displayed. On a canvas showing one image those coincide, which is why following
        the content worked; on a real-space canvas they do not — the content centre is
        wherever the acquired images happen to average out, and it moves as more arrive.

        Anchoring explicitly also lets the grid be drawn before anything is acquired,
        which is when a planned grid is most useful.
        """
        self._anchor_point = None if centre is None else (float(centre[0]), float(centre[1]))
        self._redraw()

    def _anchor(self) -> Optional[Tuple[float, float]]:
        """Where the grid is centred, or None if there is nothing to centre it on."""
        if self._anchor_point is not None:
            return self._anchor_point
        rect = self._content()
        return None if rect is None else (rect.cx, rect.cy)

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

    def set_grid_editable(self, editable: bool) -> None:
        """Whether dragging may move or resize the grid. Toggling tiles is unaffected.

        For a host that derives the grid from something else and would only have to undo
        the drag -- the sparse selection dialog computes rows, columns and centre from
        the regions drawn on a beam overview, so a grid dragged here would spring back
        the moment the plan was recomputed, which reads as a broken control rather than
        an absent one.

        The resize cursor goes with it. An affordance that advertises a gesture the host
        discards is worse than no affordance, since the only way to learn it does nothing
        is to try it.
        """
        self._geometry_editable = bool(editable)

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

    @property
    def is_dragging(self) -> bool:
        """True while any grid gesture is in flight — a resize *or* a move.

        Both emit on every motion event, so anything the host does in response has to
        be cheap and must not move the camera. Moving was missed when this was only
        :attr:`is_resizing`: dragging the grid re-declared the canvas's working area on
        each step, and re-declaring refits, so the view snapped onto the grid.
        """
        return self._resize_axes is not None or self._move_active

    def fit_view(self, padding: float = 0.05) -> None:
        """Zoom out so the whole grid is visible, not just the tile at the centre.

        A 3x3 at 10% overlap spans ~2.8x the field of view per axis, so at the default
        view the outer tiles are off-screen -- and an outer tile that cannot be seen
        cannot be clicked.
        """
        rect = self._content()
        if self._canvas is None or not self._tiles or rect is None:
            return

        span_w, span_h = self._extent()
        if span_w <= 0 or span_h <= 0:
            return

        # `set_view_margin` is a fraction of the *content* size per side.
        margin = max(
            (span_w / rect.width - 1.0) / 2.0,
            (span_h / rect.height - 1.0) / 2.0,
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

    def _bounds(self) -> Tuple[float, float, float, float]:
        """(left, top, width, height) of the whole grid, in display pixels.

        The grid is centred on the anchor, matching the runner: it measures from the
        top-left tile and shifts so the grid straddles the position it is planned
        around. Callers guard on `_anchor()`; the (0, 0) fallback keeps this total.
        """
        span_w, span_h = self._extent()
        anchor = self._anchor()
        cx, cy = anchor if anchor is not None else (0.0, 0.0)
        return cx - span_w / 2, cy - span_h / 2, span_w, span_h

    def _rect_for(self, tile: TilePosition) -> Tuple[float, float, float, float]:
        """(x, y, width, height) of one tile, in display pixels."""
        tile_h, tile_w = self._tile_shape
        scale = self._scale()
        left, top, _, _ = self._bounds()

        return (
            left + tile.canvas_x * scale,
            top + tile.canvas_y * scale,
            tile_w * scale,
            tile_h * scale,
        )

    def _redraw(self) -> None:
        self._remove_artists()
        if self._ax is None or not self._tiles or self._anchor() is None or not self._visible:
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
        if not self._tiles or self._tile_shape[1] <= 0 or self._anchor() is None:
            return 0, 0

        left, top, span_w, span_h = self._bounds()
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
        if not self._visible or not self._tiles:
            return

        if getattr(event, "dblclick", False):
            # The second press of a double-click, and the only warning that the first
            # click was never a tile edit: the canvas turns this same press into a stage
            # move. Drop the toggle the first release left pending and take no part in
            # the gesture -- returning without claiming leaves `_move_start` unset, so
            # the release that follows does nothing either.
            #
            # `getattr` because matplotlib sets `dblclick` on real events and the tests
            # build the minimum an overlay reads.
            self._cancel_pending_toggle()
            return

        edges = self._edge_at(event.xdata, event.ydata)
        if edges is not None and self._geometry_editable:
            self._resize_axes = edges
            self._claim(event)
            return

        anchor = self._anchor()
        if anchor is None or not self._within(event.xdata, event.ydata):
            # Outside the grid, so this press belongs to the canvas -- leave it to pan.
            return
        # Inside: held rather than acted on, because the same press starts both a move
        # and a tile toggle and only the pointer's next few pixels say which.
        self._move_start = (
            event.x, event.y, event.xdata, event.ydata, anchor[0], anchor[1]
        )
        if self._painting(event):
            tile = self._tile_at(event.xdata, event.ydata)
            if tile is not None:
                # Captured here and never read from the tiles again: the host rewrites
                # the grid after every emission, so by the second tile `not tile.enabled`
                # would be answering a question about an already-edited mask, and the
                # sweep would alternate instead of painting. `TileMaskGrid` keeps a
                # `_drag_value` for the same reason.
                self._paint_value = not tile.enabled
                self._paint_origin = (tile.row, tile.col)
        self._claim(event)

    def _painting(self, event) -> bool:
        """Whether the paint modifier is held for this event.

        Imported lazily rather than at module scope: every overlay keeps `canvas_base`
        to a TYPE_CHECKING import, and going through its helper -- rather than reading
        Qt directly here -- keeps one answer to "which modifiers are down", which
        matters because the reliable source is the Qt event and not `MouseEvent.key`.
        """
        from fibsem.ui.widgets.canvas.canvas_base import _modifiers_from_event

        return PAINT_MODIFIER in _modifiers_from_event(event)

    def _claim(self, event) -> None:
        """Tell the canvas this overlay owns the gesture.

        It cancels the pending pan and suppresses `canvas_clicked` on release, so a
        drag inside the grid moves the grid rather than the view. The click that
        suppression costs is re-emitted from `_on_release` when the press turns out to
        have been a click after all.
        """
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = True

    def _within(self, x: float, y: float) -> bool:
        """Whether a point is inside the grid's bounding box."""
        left, top, width, height = self._bounds()
        return left <= x <= left + width and top <= y <= top + height

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

    def _cursor_at(self, x: float, y: float, painting: bool = False):
        """The cursor for a point over the grid, or None if it is over nothing.

        Edges win over the interior, matching which gesture a press there starts --
        with the paint modifier held too, since an edge press still resizes.
        """
        cursor = self._cursor_for(*self._edge_sides(x, y))
        if cursor is not None:
            return cursor
        if not self._within(x, y):
            return None
        return Qt.CrossCursor if painting else Qt.SizeAllCursor

    def _update_cursor(self, event) -> None:
        """Show a resize cursor over the grid's edges, and a cross while painting.

        Without it the drag is undiscoverable -- nothing on screen suggests the
        boundary is draggable, and no one hunts for an affordance they have no reason
        to believe exists. The cross does the same job for the paint modifier: holding
        Shift over the grid changes the pointer, which is the only sign the gesture
        exists for anyone who has not been told.
        """
        if self._canvas is None:
            return

        cursor = None
        if (
            self._visible
            and self._tiles
            and self._geometry_editable
            and event.inaxes is self._ax
            and event.xdata is not None
        ):
            cursor = self._cursor_at(
                event.xdata, event.ydata, painting=self._painting(event)
            )

        if cursor is not None:
            self._canvas.setCursor(cursor)
            self._cursor_set = True
        elif self._cursor_set:
            # Only unset a cursor this overlay set, so it cannot clobber one another
            # overlay is showing.
            self._canvas.unsetCursor()
            self._cursor_set = False

    def _on_drag(self, event) -> None:
        if self._paint_value is not None:
            self._drag_paint(event)
            return
        if self._move_start is not None:
            # Still held when the geometry is fixed: it is what the release reads to
            # decide the press was a click, and so a tile toggle.
            if self._geometry_editable:
                self._drag_move(event)
            return
        if self._resize_axes is None:
            self._update_cursor(event)
            return
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        anchor = self._anchor()
        if not self._tiles or anchor is None:
            # The grid can be cleared mid-drag -- `_refresh_tile_grid` does exactly
            # that if the camera geometry cannot be read -- and the row/column counts
            # below are derived from the tiles, so this would raise on an empty max().
            return

        horizontal, vertical = self._resize_axes
        tile_h, tile_w = self._tile_shape
        scale = self._scale()
        rows = max(t.row for t in self._tiles) + 1
        cols = max(t.col for t in self._tiles) + 1

        if horizontal:
            cols = self._count_for(
                abs(event.xdata - anchor[0]), tile_w * scale
            )
        if vertical:
            rows = self._count_for(
                abs(event.ydata - anchor[1]), tile_h * scale
            )

        self.grid_resize_requested.emit(rows, cols)

    def _drag_move(self, event) -> None:
        """Follow the pointer with the whole grid, once it has clearly moved.

        Emits an absolute centre rather than a delta, measured from where the anchor
        was at press: the host re-anchors on every step, so accumulating deltas would
        compound whatever it rounds the position to -- and it *does* round, because it
        resolves the centre to a stage position and projects it back.
        """
        press_x, press_y, x0, y0, anchor_x, anchor_y = self._move_start
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if not self._move_active:
            travelled = (event.x - press_x) ** 2 + (event.y - press_y) ** 2
            if travelled < MOVE_DRAG_THRESHOLD_PX ** 2:
                return
            self._move_active = True

        self.grid_move_requested.emit(
            anchor_x + (event.xdata - x0), anchor_y + (event.ydata - y0)
        )

    def _drag_paint(self, event) -> None:
        """Give every tile the drag crosses the value the first one was heading for.

        A sweep sets a run of tiles to one value instead of toggling each in turn, so
        emptying a row is one gesture rather than one click per tile -- and the result
        does not depend on what each tile happened to be on the way past, which
        toggling does. On a 5x5 that is the difference between one drag and twenty-five
        waits on the double-click timer.
        """
        press_x, press_y = self._move_start[0], self._move_start[1]
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if not self._paint_active:
            travelled = (event.x - press_x) ** 2 + (event.y - press_y) ** 2
            if travelled < MOVE_DRAG_THRESHOLD_PX ** 2:
                return
            # The move gesture's threshold, deliberately: a modified press that never
            # travels has to reach `_on_release` still looking like a click, so that
            # Shift+click stays an ordinary toggle and keeps the double-click deferral
            # that stops a stage move editing the tile under it (FIB-520).
            self._paint_active = True
            if self._paint_origin is not None:
                self._paint(*self._paint_origin)
        tile = self._tile_at(event.xdata, event.ydata)
        if tile is not None:
            self._paint(tile.row, tile.col)

    def _paint(self, row: int, col: int) -> None:
        """Emit one tile's new value, at most once per drag.

        Motion events arrive far faster than tiles are crossed and the host rebuilds
        the whole grid on each emission, so without this a slow sweep over three tiles
        would ask for hundreds of rebuilds of the same three values.
        """
        if (row, col) in self._painted:
            return
        self._painted.add((row, col))
        self.tile_toggled.emit(row, col, self._paint_value)

    def _on_release(self, event) -> None:
        # A press inside the grid that never became a drag is a click on a tile. It has
        # to be emitted here because claiming the gesture suppressed the canvas's own
        # click signal -- claiming is what stops the view panning out from under a move.
        pressed_inside = self._move_start is not None
        # A paint counts as a drag here as well: the tiles it crossed are already
        # emitted, so letting the release toggle one more would undo the last of them.
        was_dragged = self._move_active or self._paint_active
        self._move_start = None
        self._move_active = False
        self._resize_axes = None
        self._paint_value = None
        self._paint_origin = None
        self._paint_active = False
        self._painted.clear()

        if (
            pressed_inside
            and not was_dragged
            and event.button == 1
            and event.inaxes is self._ax
            and event.xdata is not None
        ):
            tile = self._tile_at(event.xdata, event.ydata)
            if tile is not None:
                self._schedule_toggle(tile.row, tile.col, not tile.enabled)

        self._update_cursor(event)

    # ── the click / double-click split ───────────────────────────────────

    def _schedule_toggle(self, row: int, col: int, enabled: bool) -> None:
        """Hold a toggle for the double-click interval before emitting it.

        Double-clicking the canvas moves the stage. Inside the grid both gestures start
        the same way, and this release arrives before anything knows whether a second
        click is coming -- so a double-click used to move the stage *and* toggle the
        tile under it. Twice, in fact: the overlay's own tiles are not updated between
        the two releases, so both emissions carried the same value rather than undoing
        each other (FIB-520).

        Deferring, rather than emitting and undoing on the second press. Undoing is
        snappier and writes a mask it then has to rewrite; this state decides what the
        next run acquires, and never being briefly wrong is worth more here than never
        being briefly slow.

        The wait is the user's own `doubleClickInterval`, so it is as short as it can be
        and still catch the double-clicks they actually make.
        """
        self._pending_toggle = (row, col, enabled)
        self._toggle_timer.start(QApplication.doubleClickInterval())

    def _cancel_pending_toggle(self) -> None:
        self._toggle_timer.stop()
        self._pending_toggle = None

    def _emit_pending_toggle(self) -> None:
        pending, self._pending_toggle = self._pending_toggle, None
        if pending is not None:
            self.tile_toggled.emit(*pending)

    def _restore_margin(self) -> None:
        if self._previous_margin is not None and self._canvas is not None:
            self._canvas.set_view_margin(self._previous_margin)
            self._previous_margin = None
