"""Static + interactive scatter-point overlays for FibsemImageCanvas.

``PointsOverlay`` — non-interactive scatter markers with optional labels.
``PointOverlay`` — interactive points (select / drag / delete / add).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    ):
        self._points = list(points)
        self._color = color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
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

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        self._points = list(points)
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
        for i, (x, y) in enumerate(self._points, 1):
            (line,) = self._ax.plot(
                x,
                y,
                marker=self._marker,
                markersize=self._size,
                color=self._color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linestyle="none",
                zorder=8,
            )
            self._artists.append(line)
            if self._label_prefix:
                ann = self._ax.annotate(
                    f"{self._label_prefix}{i}",
                    xy=(x, y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    color=self._color,
                    fontsize=8,
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
    """

    point_added = pyqtSignal(int, float, float)  # index, x, y
    point_selected = pyqtSignal(int, float, float)  # index, x, y
    point_dragging = pyqtSignal(int, float, float)  # index, x, y  (each motion step)
    point_moved = pyqtSignal(int, float, float)  # index, x, y  (on release)
    point_removed = pyqtSignal(int)  # index (before removal)

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

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        self._points: List[List[float]] = []  # [[x, y], ...]  mutable for drag
        self._artists: List = []  # Line2D per point (index-aligned)
        self._anns: List = []  # Annotation per point (or None)
        # Optional per-point overrides (index-aligned), else the global style is used
        self._point_colors: Optional[List[str]] = None
        self._point_labels: Optional[List[str]] = None

        self._selected: Optional[int] = None
        self._drag_idx: Optional[int] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_xy: Tuple[float, float] = (0.0, 0.0)
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

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        self._remove_all_artists()
        if width > 0 and self._ax is not None:
            self._draw_all()

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
        self._selected = None
        self._remove_all_artists()
        if self._ax is not None and self._img_w:
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
        self.point_removed.emit(index)
        for lst in (self._artists, self._anns):
            a = lst.pop(index)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._points.pop(index)
        if self._selected == index:
            self._selected = None
        elif self._selected is not None and self._selected > index:
            self._selected -= 1
        if self._label_prefix or self._numbered:
            self._refresh_ann_text()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear_points(self) -> None:
        self._selected = None
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
        n = len(self._points)
        idx = index if (index is not None and 0 <= index < n) else None
        if idx == self._selected:
            return
        prev = self._selected
        self._selected = idx
        if prev is not None:
            self._update_artist_appearance(prev)
        if idx is not None:
            self._update_artist_appearance(idx)
        if self._canvas is not None:
            self._canvas.draw_idle()

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

    def _draw_legend(self) -> None:
        """Opt-in legend (top-left); the swatch is the overlay's own marker glyph
        (e.g. a "+" for the POI), styled like the milling-stage legend."""
        self._remove_legend()
        if (
            not self._legend_label
            or self._ax is None
            or not self._points
            or not self._visible
        ):
            return
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D

        handle = Line2D(
            [], [], linestyle="None", marker=self._marker, markersize=9,
            color=self._color,
            markeredgewidth=(self._edge_width if self._edge_width is not None else 2.0),
            label=self._legend_label,
        )
        # build the Legend directly (not ax.legend()) so it doesn't replace another
        # overlay's primary legend (e.g. the milling stages, top-right)
        leg = Legend(
            self._ax,
            [handle],
            [self._legend_label],
            loc="upper left",
            fontsize=8,
            facecolor="#1e2124",
            edgecolor="#555555",
            labelcolor="#d1d2d4",
            framealpha=0.85,
        )
        leg.set_zorder(10)
        self._ax.add_artist(leg)
        self._legend = leg

    def _marker_edge(self, color: str, selected: bool):
        """Edge colour/width for the marker. Unfilled markers (+, x, ...) are drawn
        in their edge colour, so they take the point colour and a thicker line;
        filled markers (o, s, ...) keep a thin white outline for contrast.

        ``edge_width`` (if set) overrides the normal-state width; the selected state
        adds a fixed bump, so backward-compatible defaults are preserved when unset.
        """
        from matplotlib.lines import Line2D
        if self._marker in Line2D.filled_markers:
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
        selected = idx == self._selected
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(color, selected)
        (line,) = self._ax.plot(
            x,
            y,
            marker=self._marker,
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
        selected = idx == self._selected
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(color, selected)
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
        px, py = self._points[idx]
        self._drag_offset = (event.xdata - px, event.ydata - py)
        self._drag_start_xy = (px, py)  # so a no-move select-click skips point_moved
        self._canvas._overlay_consuming_event = True
        self._artists[idx].set_animated(True)
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _blit(self):
        if self._canvas is None or self._ax is None:
            return
        if self._blit_bg is None or self._drag_idx is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._artists[self._drag_idx])
        ann = self._anns[self._drag_idx] if self._drag_idx < len(self._anns) else None
        if ann is not None:
            self._ax.draw_artist(ann)
        self._canvas.blit(self._ax.bbox)

    # ── mouse / key events ────────────────────────────────────────────────

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
            x = max(0.0, min(event.xdata, (self._img_w or 1) - 1))
            y = max(0.0, min(event.ydata, (self._img_h or 1) - 1))
            idx = self.add_point(x, y)
            old_sel = self._selected
            self._selected = idx
            if old_sel is not None:
                self._update_artist_appearance(old_sel)
            self._update_artist_appearance(idx)
            self.point_added.emit(idx, x, y)
            self._canvas.draw_idle()
            return

        if event.button != 1:
            return

        hit = self._hit_point(event)
        if hit is not None:
            old_sel = self._selected
            self._selected = hit
            if old_sel is not None and old_sel != hit:
                self._update_artist_appearance(old_sel)
            self._update_artist_appearance(hit)
            self.point_selected.emit(hit, self._points[hit][0], self._points[hit][1])
            self._start_drag(hit, event)
        elif self._selected is not None:
            # left-click empty → deselect
            old_sel = self._selected
            self._selected = None
            self._update_artist_appearance(old_sel)
            self._canvas.draw_idle()

    def _on_motion(self, event):
        if self._drag_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        W = self._img_w or 1
        H = self._img_h or 1
        x = max(0.0, min(event.xdata - self._drag_offset[0], W - 1))
        y = max(0.0, min(event.ydata - self._drag_offset[1], H - 1))
        self._points[self._drag_idx] = [x, y]
        self._update_artist_position(self._drag_idx)
        self.point_dragging.emit(self._drag_idx, x, y)
        self._blit()

    def _on_release(self, event):
        if self._canvas is None:
            return
        self._canvas._overlay_consuming_event = False
        if self._drag_idx is not None:
            idx = self._drag_idx
            self._drag_idx = None
            self._blit_bg = None
            self._artists[idx].set_animated(False)
            ann = self._anns[idx] if idx < len(self._anns) else None
            if ann is not None:
                ann.set_animated(False)
            # Only a real move emits point_moved (a select-click without a drag
            # leaves the position unchanged; point_selected already covered it).
            if tuple(self._points[idx]) != self._drag_start_xy:
                self.point_moved.emit(idx, self._points[idx][0], self._points[idx][1])
            self._canvas.draw_idle()

    def _on_key(self, event):
        if not self._input_allowed():  # another overlay owns input (modal-aware)
            return
        if not self._removable:
            return
        if event.key in ("delete", "backspace") and self._selected is not None:
            self.remove_point(self._selected)
