"""Static + interactive scatter-point overlays for FibsemImageCanvas.

``PointsOverlay`` — non-interactive scatter markers with optional labels.
``FieldOfViewOverlay`` — the same, plus the field of view each point covers.
``PointOverlay`` — interactive points (select / drag / delete / add).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.ui.tokens import (
    CANVAS_BG,
)
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


class PointsOverlay(CanvasOverlay):
    """Non-interactive scatter points.  Call set_points() to update."""

    def __init__(
        self,
        points: List[Tuple[float, float]] = (),
        color: str = "white",
        marker: str = "o",
        size: int = 8,
        label_prefix: str = "",
        edge_width: Optional[float] = None,
    ):
        self._points = list(points)
        self._color = color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
        self._edge_width = edge_width
        self._labels: Optional[List[str]] = None
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

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._remove_artists()
        if not rect.is_empty:
            self._draw()

    def set_points(
        self,
        points: List[Tuple[float, float]],
        labels: Optional[List[str]] = None,
    ) -> None:
        """Replace the points, optionally labelling each one.

        `labels` takes precedence over `label_prefix`: positions that carry their own
        names (lamellae, saved positions) should show those rather than an index into
        whatever order they happened to arrive in.
        """
        self._points = list(points)
        self._labels = list(labels) if labels is not None else None
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

    def _marker_edge(self):
        """Edge colour and width for the marker.

        Unfilled markers (+, x, ...) have no face, so they are drawn entirely in their
        edge colour: giving them a white one painted every point white whatever colour
        it was asked for, and left the label as the only thing carrying the colour.
        Filled markers (o, s, ...) keep the thin white outline, which is what makes them
        legible against a bright image.

        Same rule as `PointOverlay._marker_edge` below, which had it right.

        `edge_width` overrides the width for one overlay. For an unfilled marker that
        is the stroke thickness of the glyph itself, so it is the only way to give a
        bare crosshair more ink -- which the current-stage marker needs and its
        neighbours do not: it carries no field-of-view box and no label (it is passed
        an empty one), deliberately, so that driving to a lamella does not double that
        lamella's box. The glyph is all it has. Left as None everywhere else so this
        stays a property of that one marker rather than of every point on the canvas.
        """
        from matplotlib.lines import Line2D

        if self._marker in Line2D.filled_markers:
            return "white", self._edge_width or 0.8
        return self._color, self._edge_width or 2.0

    def _draw(self):
        if self._ax is None:
            return
        for index, (x, y) in enumerate(self._points):
            self._draw_marker(x, y)
            label = self._label_for(index)
            if label:
                self._draw_label(label, x, y)

    def _label_for(self, index: int) -> Optional[str]:
        """The text for the point at *index* (0-based), or None for none.

        `labels` takes precedence over `label_prefix` -- see :meth:`set_points`. An
        empty string is a label that draws nothing, which is how a caller marks a
        point it wants unnamed without giving up labels for the rest.
        """
        if self._labels is not None and index < len(self._labels):
            return self._labels[index]
        if self._label_prefix:
            return f"{self._label_prefix}{index + 1}"
        return None

    def _draw_marker(self, x: float, y: float) -> None:
        """Draw the glyph for one point.

        Split out of :meth:`_draw` so a subclass can add to what a point is drawn as
        without restating the loop or the label rules -- see
        :class:`FieldOfViewOverlay`.
        """
        edge_color, edge_width = self._marker_edge()
        (line,) = self._ax.plot(
            x,
            y,
            marker=self._marker,
            markersize=self._size,
            color=self._color,
            markeredgecolor=edge_color,
            markeredgewidth=edge_width,
            linestyle="none",
            zorder=8,
        )
        self._artists.append(line)

    def _draw_label(self, label: str, x: float, y: float) -> None:
        """Name the point at *x*, *y* -- beside the glyph, offset clear of it."""
        ann = self._ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            color=self._color,
            fontsize=8,
            zorder=9,
        )
        self._artists.append(ann)


# How the field-of-view box is drawn. Light, because the box is context for the
# marker rather than the subject -- but not as light as it was.
#
# These were 0.6 and 6.5, and reported from a session as hard to see on bright
# FIB/SEM overviews. Not a contrast problem: nothing here sets an alpha, and a
# saturated colour separates from greyscale image data by hue at any brightness.
# There was simply too little of it. At 0.6 this was the thinnest line on the canvas
# by some way, against 1.0-2.5 for every sibling overlay (alignment 2.5, pattern 1.5,
# milling 1.0), and the label was the smallest text on it.
#
# The original 0.6 had a reason, which is why this is not a straight increase: at
# full weight the edge was expected to compete with the magenta lattice behind it --
# gridbars on the beam tab, the planned tile grid on either -- and read as part of
# that grid rather than as one position. Re-checked at 1.2 against that lattice and
# it does not happen. What separates the two is hue, not weight: the box is cyan and
# the lattice is magenta, about as far apart as the wheel allows. 1.2 also matches
# the lattice's own 1.0, so the box reads as a peer of the grid rather than as a
# hairline drawn on top of it. Anything that narrows that hue gap -- recolouring
# either the box or the tile grid -- brings the original concern back, so re-check
# then rather than assuming this settled it.
#
# An outline was tried before reaching for weight, and looked worse: a halo wider
# than the line it surrounds reads as a botched double border rather than a haloed
# line. More ink, not more halo.
FOV_BOX_LINEWIDTH = 1.2
FOV_LABEL_FONTSIZE = 7.5


class FieldOfViewOverlay(PointsOverlay):
    """Points drawn with the field of view each one stands for.

    A crosshair says where a position is. It does not say how much of the sample an
    image taken there would cover, which on an overview spanning millimetres is the
    thing actually being judged -- whether two lamellae would land in one frame,
    whether a marker sits far enough inside a grid square to mill.

    The extent is held in **metres** and converted at draw time rather than stored in
    canvas units. The canvas scale is not fixed: it arrives with the first image, and
    the fluorescence tab sets it from the camera before then -- so a box stored in
    canvas units would silently come to mean a different number of microns. The
    conversion is `metres_to_canvas`, a straight division by the reference pixel size:
    the box is a frame in the plane being *displayed*, not a shape lying on the tilted
    sample, so the surface foreshortening correction must not be applied to it -- doing
    that to something in this frame is what made an overlay 3.9x too tall (FIB-615).

    With no extent, or before the canvas has a scale, this draws exactly what
    :class:`PointsOverlay` draws: the crosshair and a label beside it.
    """

    def __init__(self, *args, extent: Optional[Tuple[float, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extent = self._as_extent(extent)

    @staticmethod
    def _as_extent(extent) -> Optional[Tuple[float, float]]:
        """*extent* as a positive (width, height) in metres, or None for no box."""
        if not extent:
            return None
        width, height = extent
        if not width or not height or width <= 0 or height <= 0:
            return None
        return (float(width), float(height))

    def set_extent(self, width: Optional[float], height: Optional[float]) -> None:
        """Set the field of view every point covers, in metres.

        A no-op when it is the size already held. Hosts call this from the same
        refresh that redraws the markers, which on the fluorescence tab runs on
        settings changes and stage moves; repainting a canvas to tell it what it
        already knows is the cost that made the grid drag stutter (FIB-752).
        """
        extent = self._as_extent((width, height))
        if extent == self._extent:
            return
        self._extent = extent
        self._remove_artists()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def half_extent_canvas(self) -> Optional[Tuple[float, float]]:
        """Half the box's width and height in canvas units, or None for no box.

        Public because picking needs it. A click inside the box selects the position
        it belongs to, and the box you can hit has to be the box you can see -- so both
        come from here rather than from two callers doing the same division.
        """
        if self._extent is None or self._canvas is None:
            return None
        to_canvas = getattr(self._canvas, "metres_to_canvas", None)
        if to_canvas is None:  # not a real-space canvas; nothing to scale against
            return None
        width, height = to_canvas(*self._extent)
        # (0, 0) is what `metres_to_canvas` answers before the canvas has a scale.
        if width <= 0 or height <= 0:
            return None
        return width / 2, height / 2

    def covers(self, point: Tuple[float, float], x: float, y: float) -> bool:
        """Whether the box drawn around *point* contains the canvas point (x, y).

        For picking: clicking anywhere inside a position's field of view should select
        it. The box tested here is the box drawn -- both come from
        :meth:`half_extent_canvas` -- so what you can hit cannot drift from what you
        can see.

        False when there is no box, which leaves a caller's own hit radius as the only
        target. That is why picking should treat this as a *union* with the radius
        rather than a replacement for it: the box is not always the bigger target. A
        100 um box is under 24 px tall past ~2.8 um per screen pixel, which on a
        ~1000 px canvas is a field of about 2.8 mm -- roughly a whole grid, i.e. the
        view with the most markers in it. Picking by box alone would make them harder
        to hit exactly there.
        """
        half = self.half_extent_canvas()
        if half is None:
            return False
        half_w, half_h = half
        return abs(x - point[0]) <= half_w and abs(y - point[1]) <= half_h

    def _draw_marker(self, x: float, y: float) -> None:
        """The crosshair, and the frame an image taken there would cover."""
        super()._draw_marker(x, y)
        half = self.half_extent_canvas()
        if half is None:
            return
        from matplotlib.patches import Rectangle

        half_w, half_h = half
        box = Rectangle(
            (x - half_w, y - half_h),
            2 * half_w,
            2 * half_h,
            fill=False,
            edgecolor=self._color,
            linewidth=FOV_BOX_LINEWIDTH,
            # Under the marker so the crosshair stays legible where a neighbouring
            # box crosses it, and over the stage context (zorder 4) beneath.
            zorder=7,
        )
        self._ax.add_patch(box)
        self._artists.append(box)

    def _draw_label(self, label: str, x: float, y: float) -> None:
        """Name the position above the top-left corner of its box.

        Beside the crosshair -- where :class:`PointsOverlay` puts it -- lands the text
        inside the box, over the sample the box exists to show. Above and outside keeps
        both readable, and lines the names up along the top edges where several boxes
        sit at similar heights.

        The y axis is inverted (image convention, origin upper), so the top edge is at
        the *smaller* y.
        """
        half = self.half_extent_canvas()
        if half is None:
            super()._draw_label(label, x, y)
            return
        half_w, half_h = half
        ann = self._ax.annotate(
            label,
            xy=(x - half_w, y - half_h),
            xytext=(0, 2),
            textcoords="offset points",
            color=self._color,
            fontsize=FOV_LABEL_FONTSIZE,
            ha="left",
            va="bottom",
            zorder=9,
        )
        self._artists.append(ann)


_PICK_RADIUS_PX = 12  # screen-space hit radius for point picking


class PointOverlay(QObject):
    """Interactive points overlay.

    * Left-click a point → selects it (highlighted colour + larger marker)
    * Left-click empty area → deselects
    * Drag a selected point → moves it, clamped to image bounds (blitted)
    * Right-click empty area → adds a new point (when ``add_on_right_click=True``)
    * Delete / Backspace → removes the selected point

    With ``multi_select=True`` the selection is an ordered list of indices, the last
    being the current point (the shape of ``CorrelationPointStore``'s selection):

    * Ctrl/Cmd-click a point → toggles it; Shift-click a point → adds it
    * Shift-drag on empty area → box-selects, adding what the box covers. A plain drag
      on empty area still pans, and keeps the selection; only a click there clears it.
    * Drag a selected point → moves the whole selection rigidly, clamped as a group
    * Delete / Backspace → removes every selected point

    Parameters
    ----------
    color : str
        Default point colour.
    selected_color : str
        Colour when a point is selected.
    marker : str
        Matplotlib marker style.
    size : float
        Marker size in points (selected markers are drawn at ``size * 1.4``).
    label_prefix : str
        If non-empty, each point gets an annotation ``label_prefix + (index+1)``.
    add_on_right_click : bool
        If True (default), right-clicking adds a new point.
    removable : bool
        If True (default), Delete/Backspace removes the selected point.
    modal : bool
        If True, the overlay handles input *only* while it is the canvas's active
        overlay (e.g. spot burn — inert in Move mode). If False (default), it also
        responds when no overlay is active (always-on, backward-compatible).
    multi_select : bool
        If True, several points can be selected, moved and removed together (see
        above). If False (default), at most one point is selected, as before.
    """

    point_added = pyqtSignal(int, float, float)  # index, x, y
    point_selected = pyqtSignal(int, float, float)  # index, x, y
    point_dragging = pyqtSignal(int, float, float)  # index, x, y  (each motion step)
    point_moved = pyqtSignal(int, float, float)  # index, x, y  (on release)
    point_removed = pyqtSignal(int)  # index the point held (emitted after removal)
    # Group counterparts, for one action on two or more points (multi_select only). A
    # single point still reports through point_moved / point_removed, so every action
    # is exactly one signal.
    points_moved = pyqtSignal(list)  # indices (on release)
    points_removed = pyqtSignal(list)  # indices the points held, ascending
    # Any selection change the user made on the canvas, with the indices now selected.
    # Programmatic changes (set_selected / set_selection) are silent, like point_selected.
    selection_changed = pyqtSignal(list)

    def __init__(
        self,
        color: str = "cyan",
        selected_color: str = "yellow",
        marker: str = "o",
        size: float = 10.0,
        label_prefix: str = "",
        add_on_right_click: bool = True,
        removable: bool = True,
        modal: bool = False,
        edge_width: Optional[float] = None,
        legend_label: Optional[str] = None,
        numbered: bool = False,
        multi_select: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._color = color
        self._selected_color = selected_color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
        self._add_on_right_click = add_on_right_click
        self._removable = removable
        self._modal = modal
        self._edge_width = edge_width  # override the default marker edge width if set
        self._legend_label = legend_label  # opt-in patch legend for this overlay
        self._legend = None
        self._numbered = numbered  # annotate each point with its 1-based index
        self._visible = True  # toggled by set_visible (points kept, artists hidden)
        self._multi_select = multi_select

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._rect: Optional["ContentRect"] = None  # canvas content bounds

        self._points: List[List[float]] = []  # [[x, y], ...]  mutable for drag
        self._artists: List = []  # Line2D per point (index-aligned)
        self._anns: List = []  # Annotation per point (or None)
        # Optional per-point overrides (index-aligned), else the global style is used
        self._point_colors: Optional[List[str]] = None
        self._point_labels: Optional[List[str]] = None

        self._selection: List[int] = []  # ordered; the last is the current point
        self._drag_idx: Optional[int] = None
        # the points moving with the drag (the grabbed one included) and where each
        # started, index-aligned
        self._drag_group: List[int] = []
        self._drag_starts: List[Tuple[float, float]] = []
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_xy: Tuple[float, float] = (0.0, 0.0)
        # a plain click on a member of a multi-selection keeps the group so it can be
        # dragged; released without moving, it narrows the selection to that point
        self._collapse_on_release = False
        # a plain press on empty area clears a multi-selection only if it is released
        # without moving: a drag there is a pan, and must not cost the selection
        self._clear_press_xy: Optional[Tuple[float, float]] = None
        self._box_start: Optional[Tuple[float, float]] = None
        self._box_end: Tuple[float, float] = (0.0, 0.0)
        self._box_artist = None
        self._blit_bg = None

        self._cids: List[int] = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in self._cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids = []
        self._remove_all_artists()
        self._ax = None
        self._canvas = None

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._rect = rect
        self._remove_all_artists()
        if not rect.is_empty and self._ax is not None:
            self._draw_all()

    def _clamp_to_content(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp a point to the last addressable pixel inside the content bounds."""
        rect = self._rect
        if rect is None or rect.is_empty:
            return 0.0, 0.0
        return (
            max(rect.x0, min(x, rect.x1 - 1)),
            max(rect.y0, min(y, rect.y1 - 1)),
        )

    # ── public API ────────────────────────────────────────────────────────

    def set_points(
        self,
        points: List[Tuple[float, float]],
        colors: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Replace all points.

        ``colors`` / ``labels``, when given, are index-aligned per-point overrides
        (e.g. one colour + name per detection feature); otherwise the global
        ``color`` / ``label_prefix`` style is used.
        """
        self._points = [[float(x), float(y)] for x, y in points]
        self._point_colors = list(colors) if colors is not None else None
        self._point_labels = list(labels) if labels is not None else None
        self._selection = []
        self._remove_all_artists()
        if self._ax is not None and self._rect is not None and not self._rect.is_empty:
            self._draw_all()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def add_point(self, x: float, y: float) -> int:
        """Append a point and return its index."""
        idx = len(self._points)
        self._points.append([float(x), float(y)])
        if self._ax is not None:
            self._append_artist(idx)
        if self._canvas is not None:
            self._canvas.draw_idle()
        return idx

    def remove_point(self, index: int) -> None:
        """Remove the point at *index*."""
        if index < 0 or index >= len(self._points):
            return
        for lst in (self._artists, self._anns):
            a = lst.pop(index)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._points.pop(index)
        self._selection = [
            i - 1 if i > index else i for i in self._selection if i != index
        ]
        if self._label_prefix or self._numbered:
            self._refresh_ann_text()
        if self._canvas is not None:
            self._canvas.draw_idle()
        # Emit LAST. A direct-connected listener may rebuild this overlay from its
        # own model (set_points) while handling the signal; emitting before the pop
        # made the pop below run on the rebuilt lists and remove a second point.
        self.point_removed.emit(index)

    def remove_points(self, indices) -> None:
        """Remove several points in one action.

        One ``points_removed`` with the indices the points held (ascending); a single
        index goes through :meth:`remove_point` and reports as ``point_removed``. A
        subclass that overrides ``remove_point`` to keep its own model in step (as
        correlation does) must override this too, or leave ``multi_select`` off.
        """
        gone = sorted({int(i) for i in indices if 0 <= i < len(self._points)})
        if len(gone) <= 1:
            if gone:
                self.remove_point(gone[0])
            return
        for index in reversed(gone):  # back to front, so earlier indices stay valid
            for lst in (self._artists, self._anns):
                if index < len(lst):
                    a = lst.pop(index)
                    if a is not None:
                        try:
                            a.remove()
                        except Exception:
                            pass
            self._points.pop(index)
        self._selection = [
            i - sum(1 for g in gone if g < i) for i in self._selection if i not in gone
        ]
        if self._label_prefix or self._numbered:
            self._refresh_ann_text()
        if self._canvas is not None:
            self._canvas.draw_idle()
        self.points_removed.emit(gone)  # last, for the reason remove_point gives

    def clear_points(self) -> None:
        self._selection = []
        self._remove_all_artists()
        self._points.clear()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p[0], p[1]) for p in self._points]

    def set_visible(self, visible: bool) -> None:
        """Show or hide all markers/labels without discarding the points.

        State is remembered and re-applied across image rebuilds (a hidden
        overlay stays hidden when a new image arrives).
        """
        self._visible = visible
        for a in self._artists + self._anns:
            if a is not None:
                a.set_visible(visible)
        self._draw_legend()  # add/remove the legend to match visibility
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_selected(self, index: Optional[int]) -> None:
        """Programmatically select a point (e.g. from a synced table).

        Silent — does not emit ``point_selected`` — so it will not loop back onto a
        producer that is driving the selection. Pass ``None`` (or an out-of-range
        index) to clear the selection.
        """
        self.set_selection([] if index is None else [index])

    def set_selection(self, indices) -> None:
        """Programmatically select several points; the last is the current one.

        Silent, like :meth:`set_selected`. Out-of-range indices are dropped; without
        ``multi_select`` only the last one is kept.
        """
        if self._apply_selection(indices) and self._canvas is not None:
            self._canvas.draw_idle()

    def selected_indices(self) -> List[int]:
        """The selected indices, in selection order (the last is the current point)."""
        return list(self._selection)

    @property
    def _selected(self) -> Optional[int]:
        """The current point — the last selected — or None. Read-only: the selection
        is changed through :meth:`_apply_selection`."""
        return self._selection[-1] if self._selection else None

    def _apply_selection(self, indices) -> bool:
        """Make *indices* the selection and restyle the points whose state changed.
        Returns whether anything changed. Draws and emits nothing."""
        n = len(self._points)
        new: List[int] = []
        for i in indices:
            if i is not None and 0 <= i < n and i not in new:
                new.append(int(i))
        if not self._multi_select:
            new = new[-1:]
        if new == self._selection:
            return False
        changed = set(new).symmetric_difference(self._selection)
        self._selection = new
        for idx in sorted(changed):
            self._update_artist_appearance(idx)
        return True

    # ── private: artists ──────────────────────────────────────────────────

    def _input_allowed(self) -> bool:
        """Whether this overlay may handle input now (modal-aware).

        Modal overlays respond only while they are the canvas's active overlay;
        non-modal overlays also respond when nothing is active (default).
        """
        if self._canvas is None:
            return True
        if self._modal:
            return self._canvas._active_overlay is self
        return self._canvas._overlay_input_allowed(self)

    def _remove_all_artists(self):
        for lst in (self._artists, self._anns):
            for a in lst:
                if a is not None:
                    try:
                        a.remove()
                    except Exception:
                        pass
            lst.clear()
        self._remove_legend()
        self._remove_box()

    def _draw_all(self):
        for idx in range(len(self._points)):
            self._append_artist(idx)
        self._draw_legend()

    def _remove_legend(self) -> None:
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

    def _legend_entries(self):
        """``(handles, labels)`` for the legend, or two empty lists for none.

        One entry, drawn with the overlay's own glyph, because the base's points
        all look alike. Split from :meth:`_draw_legend` so a subclass whose points
        do *not* — correlation shows several PointTypes on one surface — can vary
        the content without restating the styling.

        "No points, no legend" is decided here rather than in :meth:`_draw_legend`
        so a subclass carrying entries that its own points do not back — the
        correlation result markers, which live on a separate overlay — can still
        show them.
        """
        from matplotlib.lines import Line2D

        if not self._points or not self._legend_label:
            return [], []
        handle = Line2D(
            [],
            [],
            linestyle="None",
            marker=self._marker,
            markersize=9,
            color=self._color,
            markeredgewidth=(self._edge_width if self._edge_width is not None else 2.0),
            label=self._legend_label,
        )
        return [handle], [self._legend_label]

    def _draw_legend(self) -> None:
        """Opt-in legend (top-left), styled like the milling-stage legend."""
        self._remove_legend()
        if self._ax is None or not self._visible:
            return
        handles, labels = self._legend_entries()
        if not handles:
            return
        from matplotlib.legend import Legend

        # build the Legend directly (not ax.legend()) so it doesn't replace another
        # overlay's primary legend (e.g. the milling stages, top-right)
        leg = Legend(
            self._ax,
            handles,
            labels,
            loc="upper left",
            fontsize=8,
            facecolor=CANVAS_BG,
            edgecolor="#555555",
            labelcolor="#d1d2d4",
            framealpha=0.85,
        )
        leg.set_zorder(10)
        self._ax.add_artist(leg)
        self._legend = leg

    def _on_right_click(self, x: float, y: float) -> None:
        """Handle a right-click at content coordinates *x*, *y*.

        Adds a point immediately and selects it. Split out from ``_on_press`` so a
        subclass can defer instead — correlation has to ask which ``PointType``
        before a point exists, so it emits a request and adds nothing here.
        """
        idx = self.add_point(x, y)
        changed = self._apply_selection([idx])
        self.point_added.emit(idx, x, y)
        if changed:
            self.selection_changed.emit(self.selected_indices())
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _point_marker(self, idx: int) -> str:
        """Marker glyph for a point.

        One glyph for every point here. Overridable so a subclass can vary it per
        point without reimplementing the artist path — correlation draws SURFACE
        types as "+" and everything else as "o".
        """
        return self._marker

    def _marker_edge(self, idx: int, color: str, selected: bool):
        """Edge colour/width for the marker. Unfilled markers (+, x, ...) are drawn
        in their edge colour, so they take the point colour and a thicker line;
        filled markers (o, s, ...) keep a thin white outline for contrast.

        ``edge_width`` (if set) overrides the normal-state width; the selected state
        adds a fixed bump, so backward-compatible defaults are preserved when unset.
        """
        from matplotlib.lines import Line2D

        if self._point_marker(idx) in Line2D.filled_markers:
            base = self._edge_width if self._edge_width is not None else 0.8
            return "white", (base + 1.2 if selected else base)
        base = self._edge_width if self._edge_width is not None else 2.0
        return color, (base + 0.8 if selected else base)

    def _point_color(self, idx: int, selected: bool) -> str:
        """Per-point colour override if set, else the global selected/normal colour.
        (Per-point points keep their own colour even when selected — size + edge
        convey the selection instead.)"""
        if self._point_colors is not None and idx < len(self._point_colors):
            return self._point_colors[idx]
        return self._selected_color if selected else self._color

    def _point_label(self, idx: int) -> Optional[str]:
        """Per-point label override if set, else ``label_prefix + (idx+1)``, else the
        bare 1-based index when ``numbered``, else None."""
        if self._point_labels is not None and idx < len(self._point_labels):
            return self._point_labels[idx]
        if self._label_prefix:
            return f"{self._label_prefix}{idx + 1}"
        if self._numbered:
            return str(idx + 1)
        return None

    def _append_artist(self, idx: int):
        if self._ax is None:
            return
        x, y = self._points[idx]
        selected = idx in self._selection
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(idx, color, selected)
        (line,) = self._ax.plot(
            x,
            y,
            marker=self._point_marker(idx),
            markersize=ms,
            color=color,
            markeredgecolor=edge_color,
            markeredgewidth=mew,
            linestyle="none",
            zorder=8,
            animated=False,
            visible=self._visible,
        )
        self._artists.append(line)
        ann = None
        label = self._point_label(idx)
        if label is not None:
            ann = self._ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
                zorder=9,
                animated=False,
                visible=self._visible,
            )
        self._anns.append(ann)

    def _update_artist_appearance(self, idx: int):
        if idx >= len(self._artists):
            return
        selected = idx in self._selection
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(idx, color, selected)
        line = self._artists[idx]
        line.set_color(color)
        line.set_markersize(ms)
        line.set_markeredgecolor(edge_color)
        line.set_markeredgewidth(mew)
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.set_color(color)

    def _update_artist_position(self, idx: int):
        if idx >= len(self._artists):
            return
        x, y = self._points[idx]
        self._artists[idx].set_xdata([x])
        self._artists[idx].set_ydata([y])
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.xy = (x, y)

    def _refresh_ann_text(self):
        for idx, ann in enumerate(self._anns):
            if ann is not None:
                label = self._point_label(idx)
                if label is not None:
                    ann.set_text(label)

    # ── hit testing ───────────────────────────────────────────────────────

    def _hit_point(self, event) -> Optional[int]:
        if not self._points or self._ax is None:
            return None
        trans = self._ax.transData
        best_idx, best_dist = None, _PICK_RADIUS_PX
        for i, (px, py) in enumerate(self._points):
            sx, sy = trans.transform((px, py))
            d = ((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    # ── blit helpers ──────────────────────────────────────────────────────

    def _start_drag(self, idx: int, event):
        if self._canvas is None or self._ax is None:
            return
        self._drag_idx = idx
        # a selected point carries the rest of a multi-selection with it
        if self._multi_select and idx in self._selection:
            self._drag_group = list(self._selection)
        else:
            self._drag_group = [idx]
        self._drag_starts = [
            (self._points[i][0], self._points[i][1]) for i in self._drag_group
        ]
        px, py = self._points[idx]
        self._drag_offset = (event.xdata - px, event.ydata - py)
        self._drag_start_xy = (px, py)  # so a no-move select-click skips point_moved
        self._canvas._overlay_consuming_event = True
        self._set_drag_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _set_drag_animated(self, animated: bool) -> None:
        for i in self._drag_group:
            if i < len(self._artists):
                self._artists[i].set_animated(animated)
            ann = self._anns[i] if i < len(self._anns) else None
            if ann is not None:
                ann.set_animated(animated)

    def _blit_artists(self) -> List:
        """Artists redrawn on every drag step, in draw order.

        The dragged points and their labels. Split out so a subclass with an artist
        that *tracks* a point — correlation's surface datum line — can keep it in
        step during the drag instead of leaving it behind until release.
        """
        artists = []
        for i in self._drag_group:
            artists.append(self._artists[i])
            ann = self._anns[i] if i < len(self._anns) else None
            if ann is not None:
                artists.append(ann)
        return artists

    def _blit(self):
        if self._canvas is None or self._ax is None:
            return
        if self._blit_bg is None or self._drag_idx is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        for artist in self._blit_artists():
            self._ax.draw_artist(artist)
        self._canvas.blit(self._ax.bbox)

    # ── box selection (multi_select) ──────────────────────────────────────

    def _start_box(self, event) -> None:
        from matplotlib.patches import Rectangle

        self._box_start = (event.xdata, event.ydata)
        self._box_end = (event.xdata, event.ydata)
        self._canvas._overlay_consuming_event = True  # the canvas drops its pan
        self._box_artist = Rectangle(
            self._box_start,
            0.0,
            0.0,
            fill=False,
            edgecolor=self._selected_color,
            linestyle="--",
            linewidth=1.0,
            zorder=9,
            animated=True,
        )
        self._ax.add_patch(self._box_artist)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _update_box(self, x: float, y: float) -> None:
        self._box_end = (x, y)
        if self._box_artist is None or self._canvas is None or self._ax is None:
            return
        x0, y0 = self._box_start
        self._box_artist.set_bounds(min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        if self._blit_bg is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._box_artist)
        self._canvas.blit(self._ax.bbox)

    def _finish_box(self) -> None:
        """Add every point inside the box to the selection."""
        (x0, y0), (x1, y1) = self._box_start, self._box_end
        self._box_start = None
        self._blit_bg = None
        self._remove_box()
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        inside = [
            i
            for i, (px, py) in enumerate(self._points)
            if xlo <= px <= xhi and ylo <= py <= yhi
        ]
        changed = self._apply_selection(self._selection + inside)
        if self._canvas is not None:
            self._canvas.draw_idle()
        if changed:
            self.selection_changed.emit(self.selected_indices())

    def _remove_box(self) -> None:
        if self._box_artist is not None:
            try:
                self._box_artist.remove()
            except Exception:
                pass
            self._box_artist = None

    # ── mouse / key events ────────────────────────────────────────────────

    def _modifiers(self, event) -> Tuple[str, ...]:
        """Held modifiers, e.g. ``("Shift",)`` — only consulted with multi_select, so
        a single-select overlay ignores them exactly as it always has. Cmd on macOS
        arrives as "Control"."""
        if not self._multi_select:
            return ()
        from fibsem.ui.widgets.canvas.canvas_base import _modifiers_from_event

        return _modifiers_from_event(event)

    def _on_press(self, event):
        if self._canvas is None or self._ax is None:
            return
        if not self._input_allowed():  # another overlay owns input (modal-aware)
            return
        if event.inaxes is not self._ax or event.xdata is None or event.dblclick:
            return
        if self._canvas._overlay_consuming_event:
            return

        if event.button == 3:  # right-click → add a new point
            if not self._add_on_right_click:
                return
            self._on_right_click(*self._clamp_to_content(event.xdata, event.ydata))
            return

        if event.button != 1:
            return

        mods = self._modifiers(event)
        hit = self._hit_point(event)
        if hit is not None and ("Control" in mods or "Shift" in mods):
            # Ctrl toggles, Shift adds; neither drags. The press is still claimed so
            # the canvas does not pan if the hand moves a little.
            self._canvas._overlay_consuming_event = True
            if "Control" in mods:
                if hit in self._selection:
                    new = [i for i in self._selection if i != hit]
                else:
                    new = self._selection + [hit]
            else:
                new = self._selection + [hit]
            if self._apply_selection(new):
                self._canvas.draw_idle()
                self.selection_changed.emit(self.selected_indices())
        elif hit is not None:
            # A member of a multi-selection keeps the group, so it can be dragged;
            # a click anywhere else selects just the point.
            keep_group = len(self._selection) > 1 and hit in self._selection
            changed = False if keep_group else self._apply_selection([hit])
            self._collapse_on_release = keep_group
            # Drag state first, emit last: a listener may rebuild this overlay
            # (set_points) while handling the signal, and the drag has to have
            # captured its artist and blit background before that can happen.
            self._start_drag(hit, event)
            self.point_selected.emit(hit, self._points[hit][0], self._points[hit][1])
            if changed:
                self.selection_changed.emit(self.selected_indices())
        elif "Shift" in mods:
            self._start_box(event)  # adds to the selection on release
        elif "Control" in mods:
            return  # a near-miss with a modifier held keeps the selection
        elif self._selection and self._multi_select:
            self._clear_press_xy = (event.x, event.y)  # decided on release
        elif self._selection:
            # left-click empty → deselect
            self._apply_selection([])
            self._canvas.draw_idle()
            self.selection_changed.emit([])

    def _on_motion(self, event):
        if self._box_start is not None:
            if event.xdata is not None and event.ydata is not None:
                self._update_box(event.xdata, event.ydata)
            return
        if self._drag_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        gx, gy = event.xdata - self._drag_offset[0], event.ydata - self._drag_offset[1]
        if len(self._drag_group) == 1:
            x, y = self._clamp_to_content(gx, gy)
            self._points[self._drag_idx] = [x, y]
            self._update_artist_position(self._drag_idx)
        else:
            dx, dy = self._clamp_group_delta(
                gx - self._drag_start_xy[0], gy - self._drag_start_xy[1]
            )
            for i, (sx, sy) in zip(self._drag_group, self._drag_starts):
                self._points[i] = [sx + dx, sy + dy]
                self._update_artist_position(i)
            x, y = self._points[self._drag_idx]
        self._blit()
        self.point_dragging.emit(self._drag_idx, x, y)

    def _clamp_group_delta(self, dx: float, dy: float) -> Tuple[float, float]:
        """Clamp a group move so every point stays inside the content bounds. The
        group moves rigidly: it stops at the edge rather than squashing against it."""
        rect = self._rect
        if rect is None or rect.is_empty or not self._drag_starts:
            return 0.0, 0.0
        xs = [p[0] for p in self._drag_starts]
        ys = [p[1] for p in self._drag_starts]
        dx = max(rect.x0 - min(xs), min(dx, rect.x1 - 1 - max(xs)))
        dy = max(rect.y0 - min(ys), min(dy, rect.y1 - 1 - max(ys)))
        return dx, dy

    def _on_release(self, event):
        if self._canvas is None:
            return
        self._canvas._overlay_consuming_event = False
        if self._box_start is not None:
            self._finish_box()
            return
        if self._clear_press_xy is not None:
            (px, py), self._clear_press_xy = self._clear_press_xy, None
            moved = (
                event.x is None
                or ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5 >= 3
            )
            if not moved and self._apply_selection([]):
                self._canvas.draw_idle()
                self.selection_changed.emit([])
            return
        if self._drag_idx is not None:
            idx = self._drag_idx
            group = self._drag_group
            collapse = self._collapse_on_release
            self._set_drag_animated(False)
            self._drag_idx = None
            self._drag_group = []
            self._drag_starts = []
            self._collapse_on_release = False
            self._blit_bg = None
            # Only a real move emits point_moved (a select-click without a drag
            # leaves the position unchanged; point_selected already covered it).
            if tuple(self._points[idx]) != self._drag_start_xy:
                if len(group) == 1:
                    self.point_moved.emit(
                        idx, self._points[idx][0], self._points[idx][1]
                    )
                else:
                    self.points_moved.emit(sorted(group))
            elif collapse and self._apply_selection([idx]):
                self.selection_changed.emit(self.selected_indices())
            self._canvas.draw_idle()

    def _on_key(self, event):
        if not self._input_allowed():  # another overlay owns input (modal-aware)
            return
        if not self._removable:
            return
        if event.key in ("delete", "backspace") and self._selection:
            if len(self._selection) == 1:
                self.remove_point(self._selection[0])
            else:
                self.remove_points(self._selection)
            self.selection_changed.emit(self.selected_indices())
