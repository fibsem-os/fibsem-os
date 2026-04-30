"""ImagePointCanvas — matplotlib canvas with draggable Coordinate markers.

Displays a 2-D image (numpy array) and overlays Coordinate points that can
be selected by clicking and repositioned by dragging.

Performance
-----------
- Large images are downsampled to _MAX_DISPLAY_PX before passing to imshow,
  so matplotlib renders far fewer pixels on each pan/zoom redraw.
  Coordinates remain in original pixel space (imshow extent is set to the
  original image dimensions), so all hit-testing and point positions are correct.
- Point artists use animated=True + blitting: dragging only repaints the
  marker layer, not the image.
- Pan/zoom redraws are throttled to ~60 fps via a QTimer so rapid scroll
  events collapse into a single render.

Signals
-------
point_selected     : Coordinate        — emitted when a point is clicked/selected
point_moved        : Coordinate        — emitted after a drag completes (on release)
point_removed      : Coordinate        — emitted on Delete/Backspace key
canvas_clicked     : (float, float)    — left-click on empty canvas (data coords)
point_add_requested: (float, float, PointType) — right-click menu item chosen

Interactions
------------
  Left-click (point)     select / begin drag
  Left-drag  (point)     move point — clamped to image bounds
  Left-drag  (empty)     pan
  Right-click            context menu → Add <Type> at that position
  Scroll wheel           zoom centred on cursor
  Delete / Backspace     remove selected point
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu, QSizePolicy, QWidget

from fibsem.correlation.structures import Coordinate, PointType
_logger = logging.getLogger(__name__)

_POINT_COLORS: Dict[PointType, str] = {
    PointType.FIB:     "#00ff00",
    PointType.FM:      "#00e5ff",
    PointType.POI:     "#ff00ff",
    PointType.SURFACE: "#ff9800",
}
_POINT_MARKERS: Dict[PointType, str] = {
    PointType.FIB:     "o",
    PointType.FM:      "o",
    PointType.POI:     "o",
    PointType.SURFACE: "+",
}
_MARKER_SIZE     = 10
_SELECTED_SIZE   = 14
_PICK_RADIUS_PX  = 15
_ZOOM_FACTOR     = 1.15
_MAX_DISPLAY_PX  = 2048
_REDRAW_INTERVAL = 32     # ms (~60 fps cap for pan/zoom)


def _downsample(image: np.ndarray, max_px: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_px and w <= max_px:
        return image
    stride = max(1, math.ceil(max(h, w) / max_px))
    if image.ndim == 2:
        return image[::stride, ::stride]
    return image[::stride, ::stride, :]


def _generate_names(coordinates: List[Coordinate]) -> List[str]:
    counters: Dict[PointType, int] = {}
    names = []
    for c in coordinates:
        counters[c.point_type] = counters.get(c.point_type, 0) + 1
        names.append(f"{c.point_type.value} {counters[c.point_type]}")
    return names


class ImagePointCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas: image + draggable Coordinate markers."""

    point_selected      = pyqtSignal(object)               # Coordinate
    point_moved         = pyqtSignal(object)               # Coordinate
    point_removed       = pyqtSignal(object)               # Coordinate
    canvas_clicked      = pyqtSignal(float, float)         # x, y data coords
    point_add_requested = pyqtSignal(float, float, object) # x, y, PointType

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        allowed_point_types: Optional[List[PointType]] = None,
    ) -> None:
        self._fig = Figure(facecolor="#1e2124")
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._allowed_types = allowed_point_types

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e2124")
        self._ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._coordinates: List[Coordinate] = []
        self._selected: Optional[Coordinate] = None
        self._dragging: Optional[Coordinate] = None
        self._pan_start: Optional[Tuple] = None

        # Image bounds for drag clamping (set in set_image)
        self._img_x_max: float = 0.0
        self._img_y_max: float = 0.0

        self._point_artists: list = []
        self._label_artists: list = []
        self._overlay_point_artists: list = []
        self._overlay_label_artists: list = []
        self._background = None

        # Scalebar
        self._pixel_size: Optional[float] = None   # metres
        self._show_scalebar: bool = False
        self._scalebar_artist = None

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_REDRAW_INTERVAL)
        self._redraw_timer.timeout.connect(self._flush_redraw)

        self.mpl_connect("draw_event",           self._on_draw_event)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)
        self.mpl_connect("resize_event",         self._on_resize)
        self.mpl_connect("axes_leave_event",     self._on_axes_leave)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray, *, cmap: str = "gray") -> None:
        """Display a 2-D image array (H×W or H×W×C)."""
        self._ax.cla()
        self._ax.axis("off")
        self._point_artists.clear()
        self._label_artists.clear()
        self._background = None

        h, w = image.shape[:2]
        self._img_x_max = float(w - 1)
        self._img_y_max = float(h - 1)

        _logger.debug("Original image size: %d×%d", w, h)

        display = _downsample(image, _MAX_DISPLAY_PX)
        _logger.debug("Displayed image size: %d×%d", display.shape[1], display.shape[0])
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        if image.ndim == 2:
            self._ax.imshow(display, cmap=cmap, origin="upper", aspect="equal",
                            interpolation="nearest", extent=extent)
        else:
            self._ax.imshow(display, origin="upper", aspect="equal",
                            interpolation="nearest", extent=extent)

        self._ax.set_xlim(-0.5, w - 0.5)
        self._ax.set_ylim(h - 0.5, -0.5)
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._scalebar_artist = None  # axes cleared above — reset reference
        self._refresh_scalebar()
        self._overlay_point_artists.clear()
        self._overlay_label_artists.clear()
        self._rebuild_artists()

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        """Replace the full set of displayed coordinates."""
        self._coordinates = list(coords)
        if self._selected not in self._coordinates:
            self._selected = None
        self._rebuild_artists()

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        """Highlight a coordinate from an external source."""
        if coord is self._selected:
            return
        self._selected = coord
        self._apply_styles()
        self._blit_points()

    def refresh_coordinate(self, coord: Coordinate) -> None:
        """Redraw after an external edit to a coordinate's x/y values."""
        try:
            idx = self._coordinates.index(coord)
        except ValueError:
            return
        self._sync_artist_position(idx)
        self._blit_points()

    def update_display(self, image: np.ndarray) -> None:
        """Fast update: replace displayed pixel data without rebuilding axes or point artists.

        Use this instead of set_image() when only the pixel data changes
        (e.g. z-slice change, channel toggle) but image dimensions are unchanged.
        Falls back to set_image() if no image has been displayed yet.
        """
        imgs = self._ax.get_images()
        if not imgs:
            self.set_image(image)
            return
        imgs[0].set_data(image)
        self._background = None   # cached bg is stale
        self.draw_idle()

    def set_pixel_size(self, pixel_size_m: float) -> None:
        """Set physical pixel size (metres) for the scale bar."""
        self._pixel_size = pixel_size_m
        if self._show_scalebar:
            self._refresh_scalebar()
            self._background = None
            self.draw()

    def set_scalebar_visible(self, visible: bool) -> None:
        """Show or hide the scale bar."""
        self._show_scalebar = visible
        self._refresh_scalebar()
        self._background = None
        self.draw()

    def _refresh_scalebar(self) -> None:
        """Add or remove the ScaleBar artist on the current axes."""
        if self._scalebar_artist is not None:
            try:
                self._scalebar_artist.remove()
            except ValueError:
                pass
            self._scalebar_artist = None
        if self._show_scalebar and self._pixel_size is not None:
            try:
                from matplotlib_scalebar.scalebar import ScaleBar
                self._scalebar_artist = ScaleBar(
                    dx=self._pixel_size,
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                )
                self._ax.add_artist(self._scalebar_artist)
            except Exception:
                pass

    def reset_view(self) -> None:
        """Fit the view to the full image extent."""
        images = self._ax.get_images()
        if images:
            ext = images[0].get_extent()
            self._ax.set_xlim(ext[0], ext[1])
            self._ax.set_ylim(ext[2], ext[3])
            self._schedule_redraw()

    def render_to_axes(self, ax) -> None:
        """Render current canvas content (image + all point/overlay artists) onto ax.

        Intended for saving to a matplotlib figure — bypasses the blit cache so
        animated=True artists are captured correctly.
        """
        images = self._ax.get_images()
        if images:
            img = images[0]
            ax.imshow(
                img.get_array(),
                cmap=img.get_cmap(),
                norm=img.norm,
                extent=img.get_extent(),
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xlim(self._ax.get_xlim())
            ax.set_ylim(self._ax.get_ylim())

        for line in self._point_artists + self._overlay_point_artists:
            ax.plot(
                line.get_xdata(), line.get_ydata(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                color=line.get_color(),
                markeredgecolor=line.get_markeredgecolor(),
                markeredgewidth=line.get_markeredgewidth(),
                linestyle="none",
                zorder=line.get_zorder(),
            )

        for ann in self._label_artists + self._overlay_label_artists:
            ax.annotate(
                ann.get_text(),
                xy=ann.xy,
                xytext=(7, 5),
                textcoords="offset points",
                color=ann.get_color(),
                fontsize=ann.get_fontsize(),
                fontweight=ann.get_fontweight(),
                zorder=ann.get_zorder(),
            )

        if self._show_scalebar and self._pixel_size is not None:
            try:
                from matplotlib_scalebar.scalebar import ScaleBar
                ax.add_artist(ScaleBar(
                    dx=self._pixel_size,
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                ))
            except Exception:
                pass

        ax.axis("off")

    def add_overlay_points(
        self,
        points: List[Tuple[float, float]],
        *,
        color: str = "#ff0000",
        label_prefix: str = "",
        size: int = 7,
        marker: str = "o",
    ) -> None:
        """Append a group of non-interactive overlay markers (e.g. correlation result).

        Call clear_overlay() before the first add_overlay_points() when replacing
        a previous result set.  Multiple add_overlay_points() calls accumulate.
        """
        for i, (x, y) in enumerate(points, start=1):
            (line,) = self._ax.plot(
                x, y,
                marker=marker,
                markersize=size,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linestyle="none",
                zorder=8,
                animated=True,
            )
            self._overlay_point_artists.append(line)

            label = f"{label_prefix}{i}" if label_prefix else str(i)
            ann = self._ax.annotate(
                label,
                xy=(x, y),
                xytext=(7, 5),
                textcoords="offset points",
                color=color,
                fontsize=8,
                animated=True,
                zorder=9,
            )
            self._overlay_label_artists.append(ann)

        self._background = None
        self.draw_idle()

    def clear_overlay(self) -> None:
        """Remove all overlay markers added via add_overlay_points()."""
        for a in self._overlay_point_artists + self._overlay_label_artists:
            a.remove()
        self._overlay_point_artists.clear()
        self._overlay_label_artists.clear()
        self._background = None
        self.draw_idle()

    # ------------------------------------------------------------------
    # Blitting
    # ------------------------------------------------------------------

    def _on_draw_event(self, _) -> None:
        self._background = self.copy_from_bbox(self._ax.bbox)
        for a in (self._point_artists + self._label_artists
                  + self._overlay_point_artists + self._overlay_label_artists):
            self._ax.draw_artist(a)
        self.update()

    def _blit_points(self) -> None:
        if self._background is None:
            self.draw_idle()
            return
        self.restore_region(self._background)
        for a in (self._point_artists + self._label_artists
                  + self._overlay_point_artists + self._overlay_label_artists):
            self._ax.draw_artist(a)
        self.blit(self._ax.bbox)

    # ------------------------------------------------------------------
    # Pan/zoom throttle
    # ------------------------------------------------------------------

    def _schedule_redraw(self) -> None:
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _flush_redraw(self) -> None:
        self._background = None
        self.draw_idle()

    # ------------------------------------------------------------------
    # Artist management
    # ------------------------------------------------------------------

    def _rebuild_artists(self) -> None:
        for a in self._point_artists + self._label_artists:
            a.remove()
        self._point_artists.clear()
        self._label_artists.clear()

        names = _generate_names(self._coordinates)
        for coord, name in zip(self._coordinates, names):
            color = _POINT_COLORS.get(coord.point_type, "white")
            is_sel = coord is self._selected

            (line,) = self._ax.plot(
                coord.point.x, coord.point.y,
                marker=_POINT_MARKERS.get(coord.point_type, "o"),
                markersize=_SELECTED_SIZE if is_sel else _MARKER_SIZE,
                color=color,
                markeredgecolor="white" if is_sel else "none",
                markeredgewidth=2.0,
                linestyle="none",
                zorder=10 if is_sel else 5,
                animated=True,
            )
            self._point_artists.append(line)

            ann = self._ax.annotate(
                name,
                xy=(coord.point.x, coord.point.y),
                xytext=(7, 5),
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontweight="bold" if is_sel else "normal",
                animated=True,
                zorder=11 if is_sel else 6,
            )
            self._label_artists.append(ann)

        self.draw_idle()

    def _apply_styles(self) -> None:
        for coord, marker, label in zip(
            self._coordinates, self._point_artists, self._label_artists
        ):
            is_sel = coord is self._selected
            marker.set_markersize(_SELECTED_SIZE if is_sel else _MARKER_SIZE)
            marker.set_markeredgecolor("white" if is_sel else "none")
            marker.set_zorder(10 if is_sel else 5)
            label.set_fontweight("bold" if is_sel else "normal")
            label.set_zorder(11 if is_sel else 6)

    def _sync_artist_position(self, idx: int) -> None:
        coord = self._coordinates[idx]
        self._point_artists[idx].set_xdata([coord.point.x])
        self._point_artists[idx].set_ydata([coord.point.y])
        self._label_artists[idx].xy = (coord.point.x, coord.point.y)

    # ------------------------------------------------------------------
    # Cursor
    # ------------------------------------------------------------------

    def _update_cursor(self, screen_x: float, screen_y: float) -> None:
        if self._dragging is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._pan_start is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._find_nearest(screen_x, screen_y) is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected is not None:
                self.point_removed.emit(self._selected)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax:
            return

        if event.button == 3:
            if event.xdata is not None and event.ydata is not None:
                self._show_add_menu(event.xdata, event.ydata)
            return

        if event.button == 1:
            nearest = self._find_nearest(event.x, event.y)
            if nearest is not None:
                changed = nearest is not self._selected
                self._selected = nearest
                self._dragging = nearest
                self._apply_styles()
                self._blit_points()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                if changed:
                    self.point_selected.emit(nearest)
            else:
                inv = self._ax.transData.inverted()
                self._pan_start = (event.x, event.y,
                                   self._ax.get_xlim(), self._ax.get_ylim(), inv)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event) -> None:
        if self._pan_start is not None:
            if event.x is None or event.y is None:
                return
            screen_x0, screen_y0, xlim0, ylim0, inv0 = self._pan_start
            x0, y0 = inv0.transform((screen_x0, screen_y0))
            x1, y1 = inv0.transform((event.x, event.y))
            dx, dy = x1 - x0, y1 - y0
            self._ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
            self._ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
            self._schedule_redraw()
            return

        if self._dragging is not None:
            if event.inaxes is not self._ax or event.xdata is None:
                return
            # Clamp to image bounds
            x = max(0.0, min(self._img_x_max, event.xdata))
            y = max(0.0, min(self._img_y_max, event.ydata))
            self._dragging.point.x = x
            self._dragging.point.y = y
            try:
                idx = self._coordinates.index(self._dragging)
                self._sync_artist_position(idx)
            except ValueError:
                pass
            self._blit_points()
            return

        # Idle motion — update cursor only
        if event.inaxes is self._ax:
            self._update_cursor(event.x, event.y)

    def _on_release(self, event) -> None:
        if event.button == 1:
            if self._dragging is not None:
                self.point_moved.emit(self._dragging)
                self._dragging = None
            elif self._pan_start is not None:
                sx0, sy0, *_ = self._pan_start
                if ((event.x - sx0) ** 2 + (event.y - sy0) ** 2) ** 0.5 < 3:
                    if event.xdata is not None and event.ydata is not None:
                        self.canvas_clicked.emit(event.xdata, event.ydata)
                self._pan_start = None
            # Restore hover cursor
            if event.inaxes is self._ax:
                self._update_cursor(event.x, event.y)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        factor = 1.0 / _ZOOM_FACTOR if event.button == "up" else _ZOOM_FACTOR
        cx, cy = event.xdata, event.ydata
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        self._ax.set_xlim(cx + (xlim[0] - cx) * factor, cx + (xlim[1] - cx) * factor)
        self._ax.set_ylim(cy + (ylim[0] - cy) * factor, cy + (ylim[1] - cy) * factor)
        self._schedule_redraw()

    def _on_resize(self, _) -> None:
        self._background = None
        self.draw_idle()

    def _on_axes_leave(self, _) -> None:
        if self._dragging is None and self._pan_start is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Right-click menu
    # ------------------------------------------------------------------

    def _show_add_menu(self, x: float, y: float) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #2b2d31; color: #F0F1F2; border: 1px solid #3a3d42; }"
            "QMenu::item:selected { background: #2d3f5c; }"
        )
        for pt in (self._allowed_types or list(PointType)):
            action = QAction(f"Add {pt.value}", self)
            action.setData((x, y, pt))
            menu.addAction(action)

        chosen = menu.exec_(QCursor.pos())
        if chosen is not None:
            x, y, pt = chosen.data()
            self.point_add_requested.emit(x, y, pt)

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def _find_nearest(self, screen_x: float, screen_y: float) -> Optional[Coordinate]:
        best: Optional[Coordinate] = None
        best_dist = float("inf")
        for coord in self._coordinates:
            sx, sy = self._ax.transData.transform((coord.point.x, coord.point.y))
            dist = ((sx - screen_x) ** 2 + (sy - screen_y) ** 2) ** 0.5
            if dist < _PICK_RADIUS_PX and dist < best_dist:
                best = coord
                best_dist = dist
        return best
