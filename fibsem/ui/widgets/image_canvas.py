"""FibsemImageCanvas — reusable matplotlib image canvas with pluggable overlays.

Zoom: scroll wheel centred on cursor.
Pan: left-drag on empty canvas area.

Overlays implement a simple duck-typed protocol::

    class MyOverlay:
        def attach(self, ax, canvas: FibsemImageCanvas) -> None: ...
        def detach(self) -> None: ...
        def on_image_changed(self, width: int, height: int) -> None: ...

Overlays that need Qt signals extend QObject directly.  An overlay that wants
to suppress canvas pan/zoom during a drag sets ``canvas._overlay_consuming_event = True``
on button-press; the canvas clears the flag automatically on button-release.

Classes
-------
CanvasOverlay       — plain base (no-op hooks; sub-class or duck-type)
FibsemImageCanvas   — the canvas
PointsOverlay       — static scatter markers with labels
RectOverlay         — configurable rectangle (drag-only or drag+resize)
PatternOverlay      — milling shape patches (placeholder, coords in pixel space)
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRectangle
from PyQt5.QtCore import QObject, QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QSizePolicy
from superqt import QIconifyIcon

from fibsem.structures import FibsemImage

_logger = logging.getLogger(__name__)

_MAX_DISPLAY_PX = 2048
_ZOOM_FACTOR = 1.15
_REDRAW_INTERVAL = 32  # ms (~60 fps)
_BG = "#1e2124"

_RECT_FRAC = 0.25
_RECT_OFFSET = (1.0 - _RECT_FRAC) / 2.0

_OVERLAY_BTN_STYLE = (
    "QPushButton { background: rgba(40,41,48,180); border: 1px solid #555;"
    " border-radius: 3px; padding: 0px; }"
    "QPushButton:hover { background: rgba(74,74,74,200); }"
    "QPushButton:pressed { background: rgba(30,30,30,220); }"
    "QPushButton:checked { background: rgba(90,92,100,200); border-color: #FFFFFF; }"
)
_OVERLAY_ICON_SIZE = QSize(14, 14)
_OVERLAY_BTN_SIZE = 22
_OVERLAY_MARGIN = 4
_OVERLAY_GAP = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _downsample(arr: np.ndarray, max_px: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h <= max_px and w <= max_px:
        return arr
    stride = max(1, math.ceil(max(h, w) / max_px))
    return arr[::stride, ::stride] if arr.ndim == 2 else arr[::stride, ::stride, :]


def _default_extents(H: int, W: int) -> Tuple[float, float, float, float]:
    """Return (x0, x1, y0, y1) for a centred 25 % box."""
    rw, rh = W * _RECT_FRAC, H * _RECT_FRAC
    x0, y0 = W * _RECT_OFFSET, H * _RECT_OFFSET
    return x0, x0 + rw, y0, y0 + rh


# ---------------------------------------------------------------------------
# Overlay base
# ---------------------------------------------------------------------------


class CanvasOverlay:
    """No-op base for canvas overlays.  Sub-class or use duck-typing."""

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        """Called once when the overlay is added.  Create artists / connect events."""

    def detach(self) -> None:
        """Remove all artists and disconnect mpl events."""

    def on_image_changed(self, width: int, height: int) -> None:
        """Called after ax.cla() + new image drawn.  Re-create artists here."""


# ---------------------------------------------------------------------------
# FibsemImageCanvas
# ---------------------------------------------------------------------------


class FibsemImageCanvas(FigureCanvasQTAgg):
    """Reusable matplotlib canvas for FibsemImage.

    * Scroll-wheel zoom centred on cursor
    * Left-drag pan on empty area
    * Pluggable overlay objects via add_overlay() / remove_overlay()
    * Optional scalebar (auto-populated from FibsemImage.metadata.pixel_size)
    """

    canvas_clicked = pyqtSignal(float, float)  # left single-click (x, y) px
    canvas_double_clicked = pyqtSignal(float, float)  # left double-click (x, y) px
    canvas_right_clicked = pyqtSignal(float, float)  # right single-click (x, y) px
    canvas_scrolled = pyqtSignal(float, float, int)  # (x, y) px, direction +1/-1

    def __init__(self, parent=None):
        self._fig = Figure(facecolor=_BG)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(_BG)
        self._ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None
        self._overlays: List[CanvasOverlay] = []
        self._pan_start: Optional[Tuple] = None

        # Overlays set this True on press to suppress canvas pan
        self._overlay_consuming_event: bool = False

        self._pixel_size: Optional[float] = None
        self._scalebar_artist = None
        self._scalebar_visible: bool = True
        self._crosshair_visible: bool = True
        self._crosshair_artists: list = []

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_REDRAW_INTERVAL)
        self._redraw_timer.timeout.connect(self.draw_idle)

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("resize_event", lambda _: self.draw_idle())

        # Overlay buttons (parented to self; repositioned in resizeEvent)
        self._overlay_buttons: List[QPushButton] = []
        self.btn_reset_view = self._add_overlay_button(
            "mdi:fit-to-screen-outline", "Reset view", self.reset_view
        )
        self.btn_toggle_scalebar = self._add_overlay_button(
            "mdi:arrow-expand-horizontal", "Hide scalebar", self.toggle_scalebar, checkable=True
        )
        self.btn_toggle_scalebar.setChecked(True)
        self.btn_toggle_crosshair = self._add_overlay_button(
            "mdi:crosshairs", "Hide crosshair", self.toggle_crosshair, checkable=True
        )
        self.btn_toggle_crosshair.setChecked(True)

        self._plot_empty()

    # ── properties ────────────────────────────────────────────────────────

    @property
    def img_width(self) -> Optional[int]:
        return self._img_w

    @property
    def img_height(self) -> Optional[int]:
        return self._img_h

    # ── public API ────────────────────────────────────────────────────────

    def set_image(self, image: FibsemImage, cmap: str = "gray") -> None:
        """Display a FibsemImage.  Notifies all registered overlays."""
        h, w = image.data.shape[:2]
        self._img_w, self._img_h = w, h

        self._ax.cla()
        self._ax.set_facecolor(_BG)
        self._ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        display = _downsample(image.filtered_data, _MAX_DISPLAY_PX)
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        kw = dict(
            origin="upper", aspect="equal", interpolation="nearest", extent=extent
        )
        if image.data.ndim == 2:
            self._ax.imshow(display, cmap=cmap, **kw)
        else:
            self._ax.imshow(display, **kw)

        self._ax.set_xlim(-0.5, w - 0.5)
        self._ax.set_ylim(h - 0.5, -0.5)

        # Scalebar
        self._scalebar_artist = None
        try:
            if image.metadata and image.metadata.pixel_size:
                px = image.metadata.pixel_size.x
                if px and px > 0:
                    self._pixel_size = px
        except Exception:
            pass
        self._refresh_scalebar()
        self._refresh_crosshair()

        for overlay in self._overlays:
            try:
                overlay.on_image_changed(w, h)
            except Exception:
                _logger.exception("Overlay on_image_changed failed: %r", overlay)

        self.draw_idle()

    def update_display(self, arr: np.ndarray) -> None:
        """Fast pixel-data swap without resetting overlays.

        Use for z-slice navigation where image dimensions don't change.
        Falls back to a no-op if no image has been set yet.
        """
        imgs = self._ax.get_images()
        if not imgs:
            return
        imgs[0].set_data(_downsample(arr, _MAX_DISPLAY_PX))
        self.draw_idle()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the yellow crosshair centred on the image."""
        self._crosshair_visible = visible
        self._refresh_crosshair()
        self.draw_idle()

    def clear(self) -> None:
        """Clear the image and show placeholder text."""
        self._img_w = self._img_h = None
        self._ax.cla()
        self._scalebar_artist = None
        self._crosshair_artists = []
        self._plot_empty()
        for overlay in self._overlays:
            try:
                overlay.on_image_changed(0, 0)
            except Exception:
                pass
        self.draw_idle()

    # ── overlay buttons ───────────────────────────────────────────────────

    def _add_overlay_button(
        self,
        icon_name: str,
        tooltip: str,
        callback,
        checkable: bool = False,
    ) -> QPushButton:
        """Create an overlay button parented to this canvas and register it.

        Buttons are stacked right-to-left in the top-right corner and
        repositioned automatically on resize.  Returns the button.
        """
        btn = QPushButton(self)
        btn.setIcon(QIconifyIcon(icon_name, color="#aaaaaa"))
        btn.setIconSize(_OVERLAY_ICON_SIZE)
        btn.setFixedSize(_OVERLAY_BTN_SIZE, _OVERLAY_BTN_SIZE)
        btn.setToolTip(tooltip)
        btn.setCheckable(checkable)
        btn.setStyleSheet(_OVERLAY_BTN_STYLE)
        btn.clicked.connect(callback)
        btn.raise_()
        self._overlay_buttons.append(btn)
        self._reposition_overlay_buttons()
        return btn

    def _reposition_overlay_buttons(self) -> None:
        """Place overlay buttons right-to-left in the top-right corner."""
        x = self.width() - _OVERLAY_MARGIN
        for btn in self._overlay_buttons:
            x -= btn.width()
            btn.move(x, _OVERLAY_MARGIN)
            x -= _OVERLAY_GAP

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay_buttons()

    def reset_view(self) -> None:
        """Fit the view to the full image extent."""
        imgs = self._ax.get_images()
        if imgs:
            ext = imgs[0].get_extent()  # (xmin, xmax, ymax, ymin)
            self._ax.set_xlim(ext[0], ext[1])
            self._ax.set_ylim(ext[2], ext[3])
            self._schedule_redraw()

    def add_overlay(self, overlay: CanvasOverlay) -> None:
        """Register an overlay and attach it to the current axes."""
        self._overlays.append(overlay)
        overlay.attach(self._ax, self)
        if self._img_w is not None:
            try:
                overlay.on_image_changed(self._img_w, self._img_h)
            except Exception:
                _logger.exception("Overlay on_image_changed failed: %r", overlay)
        self.draw_idle()

    def remove_overlay(self, overlay: CanvasOverlay) -> None:
        if overlay in self._overlays:
            try:
                overlay.detach()
            except Exception:
                _logger.exception("Overlay detach failed: %r", overlay)
            self._overlays.remove(overlay)
            self.draw_idle()

    def clear_overlays(self) -> None:
        for o in list(self._overlays):
            self.remove_overlay(o)

    # ── internals ─────────────────────────────────────────────────────────

    def _plot_empty(self):
        self._ax.set_facecolor(_BG)
        self._ax.axis("off")
        self._ax.text(
            0.5,
            0.5,
            "No image",
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            fontsize=11,
            color="#bbbbbb",
        )

    def toggle_scalebar(self) -> None:
        """Show or hide the scalebar and update the button tooltip."""
        self._scalebar_visible = not self._scalebar_visible
        self.btn_toggle_scalebar.setChecked(self._scalebar_visible)
        self.btn_toggle_scalebar.setToolTip(
            "Hide scalebar" if self._scalebar_visible else "Show scalebar"
        )
        self._refresh_scalebar()
        self.draw_idle()

    def toggle_crosshair(self) -> None:
        """Show or hide the crosshair and update the button tooltip."""
        self.set_crosshair_visible(not self._crosshair_visible)
        self.btn_toggle_crosshair.setChecked(self._crosshair_visible)
        self.btn_toggle_crosshair.setToolTip(
            "Hide crosshair" if self._crosshair_visible else "Show crosshair"
        )

    def _refresh_scalebar(self):
        if self._scalebar_artist is not None:
            try:
                self._scalebar_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._scalebar_artist = None
        if self._pixel_size is not None and self._scalebar_visible:
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

    def _refresh_crosshair(self):
        for a in self._crosshair_artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._crosshair_artists = []
        if not self._crosshair_visible or self._img_w is None:
            return
        cx, cy = self._img_w / 2.0, self._img_h / 2.0
        half_w = self._img_w * 0.05 / 2
        half_h = self._img_h * 0.05 / 2
        kw = dict(color="yellow", linewidth=1, alpha=0.8, zorder=7)
        (h_line,) = self._ax.plot([cx - half_w, cx + half_w], [cy, cy], **kw)
        (v_line,) = self._ax.plot([cx, cx], [cy - half_h, cy + half_h], **kw)
        self._crosshair_artists = [h_line, v_line]

    def _schedule_redraw(self):
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        if event.dblclick:
            if event.button == 1:
                self.canvas_double_clicked.emit(event.xdata, event.ydata)
            return  # don't start a pan on double-click
        if event.button == 3:
            self.canvas_right_clicked.emit(event.xdata, event.ydata)
            return
        if event.button != 1:
            return
        inv = self._ax.transData.inverted()
        self._pan_start = (
            event.x,
            event.y,
            self._ax.get_xlim(),
            self._ax.get_ylim(),
            inv,
        )

    def _on_motion(self, event):
        # Overlay in drag mode — cancel any pending pan
        if self._overlay_consuming_event:
            self._pan_start = None
            return
        if self._pan_start is None:
            return
        if event.x is None or event.y is None:
            return
        sx0, sy0, xlim0, ylim0, inv0 = self._pan_start
        x0, y0 = inv0.transform((sx0, sy0))
        x1, y1 = inv0.transform((event.x, event.y))
        dx, dy = x1 - x0, y1 - y0
        self._ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self._ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self._schedule_redraw()

    def _on_release(self, event):
        was_consuming = self._overlay_consuming_event
        self._overlay_consuming_event = False
        if event.button == 1 and self._pan_start is not None:
            sx0, sy0, *_ = self._pan_start
            dist = ((event.x - sx0) ** 2 + (event.y - sy0) ** 2) ** 0.5
            if (
                dist < 3
                and not was_consuming
                and event.xdata is not None
                and event.ydata is not None
            ):
                self.canvas_clicked.emit(event.xdata, event.ydata)
        self._pan_start = None

    def _on_scroll(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        direction = 1 if event.button == "up" else -1
        self.canvas_scrolled.emit(event.xdata, event.ydata, direction)
        factor = 1.0 / _ZOOM_FACTOR if direction == 1 else _ZOOM_FACTOR
        cx, cy = event.xdata, event.ydata
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        self._ax.set_xlim(cx + (xlim[0] - cx) * factor, cx + (xlim[1] - cx) * factor)
        self._ax.set_ylim(cy + (ylim[0] - cy) * factor, cy + (ylim[1] - cy) * factor)
        self._schedule_redraw()


# ---------------------------------------------------------------------------
# PointsOverlay — static scatter markers with optional labels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# RectOverlay — configurable drag / drag+resize rectangle
# ---------------------------------------------------------------------------

_HANDLE_RADIUS_PX = 8  # screen-space hit radius for corner handles
# Corner handles: name → (x_fraction, y_fraction) within the rect
_CORNERS = {"tl": (0.0, 0.0), "tr": (1.0, 0.0), "bl": (0.0, 1.0), "br": (1.0, 1.0)}


class RectOverlay(QObject):
    """Rectangle overlay using ``MplRectangle`` + manual mouse handling.

    Both modes share the same drag/release logic.  ``resizable=True`` additionally
    draws four corner handles and supports resize by dragging them.

    Parameters
    ----------
    color : str
        Edge colour (and handle colour).
    facecolor : str | None
        Fill colour.  ``None`` → transparent.
    alpha : float
        Opacity of edge/fill.
    linewidth : int
        Edge linewidth in points.
    linestyle : str
        Matplotlib linestyle string (``"solid"``, ``"--"``, etc.)
    resizable : bool
        ``True`` → drag + four corner resize handles.
        ``False`` → drag only.

    Examples
    --------
    FIB — yellow filled, drag-only::

        RectOverlay(color="yellow", facecolor="yellow", alpha=0.5, resizable=False)

    FM — white dotted, drag + resize::

        RectOverlay(color="white", facecolor=None, linestyle="--", resizable=True)
    """

    rect_changed = pyqtSignal(dict)  # {x0,y0,x1,y1,cx,cy,width,height} pixels

    def __init__(
        self,
        color: str = "yellow",
        facecolor: Optional[str] = None,
        alpha: float = 0.5,
        linewidth: int = 2,
        linestyle: str = "solid",
        resizable: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._color = color
        self._facecolor = facecolor if facecolor is not None else "none"
        self._alpha = alpha
        self._linewidth = linewidth
        self._linestyle = linestyle
        self._resizable = resizable

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        # Rect state in data coords
        self._x0 = self._y0 = self._x1 = self._y1 = 0.0
        self._saved: Optional[Tuple[float, float, float, float]] = None  # x0,y0,w,h

        # Artists
        self._patch: Optional[MplRectangle] = None
        self._handles: dict = {}  # name → Line2D marker

        # Drag state: None | "move" | "tl" | "tr" | "bl" | "br"
        self._drag_mode: Optional[str] = None
        self._drag_start_data: Optional[Tuple[float, float]] = None
        self._drag_start_rect: Optional[Tuple[float, float, float, float]] = None
        self._blit_bg = None  # background region captured at drag start

        self._cids: List[int] = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in self._cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids = []
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        if self._ax is None or width == 0 or height == 0:
            return
        self._rebuild()

    # ── public helpers ────────────────────────────────────────────────────

    def get_rect(self) -> dict:
        return _xywh_to_dict(
            self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0
        )

    def set_rect(self, x0: float, y0: float, width: float, height: float) -> None:
        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x0 + width, y0 + height
        self._saved = (x0, y0, width, height)
        self._update_artists()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── build / teardown ──────────────────────────────────────────────────

    def _remove_artists(self):
        if self._patch is not None:
            try:
                self._patch.remove()
            except Exception:
                pass
            self._patch = None
        for h in self._handles.values():
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def _rebuild(self):
        self._remove_artists()

        if self._saved is not None:
            x0, y0, w, h = self._saved
        else:
            ex = _default_extents(self._img_h, self._img_w)
            x0, y0, w, h = ex[0], ex[2], ex[1] - ex[0], ex[3] - ex[2]
            self._saved = (x0, y0, w, h)

        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x0 + w, y0 + h

        self._patch = MplRectangle(
            (self._x0, self._y0),
            w,
            h,
            linewidth=self._linewidth,
            edgecolor=self._color,
            facecolor=self._facecolor,
            linestyle=self._linestyle,
            alpha=self._alpha,
            zorder=5,
        )
        self._ax.add_patch(self._patch)

        if self._resizable:
            for name in _CORNERS:
                hx, hy = self._handle_pos(name)
                (line,) = self._ax.plot(
                    hx,
                    hy,
                    "s",
                    markersize=7,
                    color=self._color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
                self._handles[name] = line

    def _update_artists(self):
        if self._patch is not None:
            self._patch.set_xy((self._x0, self._y0))
            self._patch.set_width(self._x1 - self._x0)
            self._patch.set_height(self._y1 - self._y0)
        for name, line in self._handles.items():
            hx, hy = self._handle_pos(name)
            line.set_xdata([hx])
            line.set_ydata([hy])

    def _handle_pos(self, name: str) -> Tuple[float, float]:
        fx, fy = _CORNERS[name]
        return self._x0 + fx * (self._x1 - self._x0), self._y0 + fy * (
            self._y1 - self._y0
        )

    # ── hit testing ───────────────────────────────────────────────────────

    def _hit_handle(self, event) -> Optional[str]:
        """Return corner name if click is within _HANDLE_RADIUS_PX of a handle."""
        trans = self._ax.transData
        for name, line in self._handles.items():
            hx, hy = self._handle_pos(name)
            sx, sy = trans.transform((hx, hy))
            if ((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5 < _HANDLE_RADIUS_PX:
                return name
        return None

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.button != 1:
            return
        if self._patch is None or event.xdata is None:
            return
        # Another overlay already claimed this event
        if self._canvas._overlay_consuming_event:
            return

        # Check corner handles first (resizable only)
        if self._resizable:
            hit = self._hit_handle(event)
            if hit is not None:
                self._start_drag(hit, event)
                return

        # Check rect body
        contains, _ = self._patch.contains(event)
        if contains:
            self._start_drag("move", event)

    def _set_animated(self, val: bool):
        if self._patch is not None:
            self._patch.set_animated(val)
        for h in self._handles.values():
            h.set_animated(val)

    def _blit(self):
        """Restore background and redraw only the overlay artists."""
        if self._blit_bg is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        if self._patch is not None:
            self._ax.draw_artist(self._patch)
        for h in self._handles.values():
            self._ax.draw_artist(h)
        self._canvas.blit(self._ax.bbox)

    def _start_drag(self, mode: str, event):
        self._drag_mode = mode
        self._drag_start_data = (event.xdata, event.ydata)
        self._drag_start_rect = (self._x0, self._y0, self._x1, self._y1)
        self._canvas._overlay_consuming_event = True
        # Mark artists animated so they're excluded from the background draw
        self._set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _on_motion(self, event):
        if self._drag_mode is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self._drag_start_data[0]
        dy = event.ydata - self._drag_start_data[1]
        rx0, ry0, rx1, ry1 = self._drag_start_rect
        W, H = self._img_w, self._img_h

        if self._drag_mode == "move":
            w, h = rx1 - rx0, ry1 - ry0
            self._x0 = max(0.0, min(rx0 + dx, W - w))
            self._y0 = max(0.0, min(ry0 + dy, H - h))
            self._x1 = self._x0 + w
            self._y1 = self._y0 + h
        elif self._drag_mode == "tl":
            self._x0 = max(0.0, min(rx0 + dx, self._x1 - 1))
            self._y0 = max(0.0, min(ry0 + dy, self._y1 - 1))
        elif self._drag_mode == "tr":
            self._x1 = max(self._x0 + 1, min(rx1 + dx, W))
            self._y0 = max(0.0, min(ry0 + dy, self._y1 - 1))
        elif self._drag_mode == "bl":
            self._x0 = max(0.0, min(rx0 + dx, self._x1 - 1))
            self._y1 = max(self._y0 + 1, min(ry1 + dy, H))
        elif self._drag_mode == "br":
            self._x1 = max(self._x0 + 1, min(rx1 + dx, W))
            self._y1 = max(self._y0 + 1, min(ry1 + dy, H))

        self._update_artists()
        self._blit()

    def _on_release(self, event):
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = False
        if self._drag_mode is not None:
            self._drag_mode = None
            self._saved = (self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0)
            # Restore normal rendering
            self._set_animated(False)
            self._blit_bg = None
            self._canvas.draw_idle()
            self.rect_changed.emit(self.get_rect())


def _xywh_to_dict(x: float, y: float, w: float, h: float) -> dict:
    return {
        "x0": round(x),
        "y0": round(y),
        "x1": round(x + w),
        "y1": round(y + h),
        "cx": round(x + w / 2),
        "cy": round(y + h / 2),
        "width": round(w),
        "height": round(h),
    }


# ---------------------------------------------------------------------------
# PointOverlay — interactive scatter points (select / drag / delete)
# ---------------------------------------------------------------------------

_PICK_RADIUS_PX = 12  # screen-space hit radius for point picking


class PointOverlay(QObject):
    """Interactive points overlay.

    * Click on empty image area → adds a new point (when ``add_on_click=True``)
    * Click on a point → selects it (highlighted colour + larger marker)
    * Drag a selected point → moves it, clamped to image bounds (blitted)
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
    add_on_click : bool
        If True (default), clicking on empty canvas adds a new point.
    """

    point_added = pyqtSignal(int, float, float)  # index, x, y
    point_selected = pyqtSignal(int, float, float)  # index, x, y
    point_moved = pyqtSignal(int, float, float)  # index, x, y
    point_removed = pyqtSignal(int)  # index (before removal)

    def __init__(
        self,
        color: str = "cyan",
        selected_color: str = "yellow",
        marker: str = "o",
        size: float = 10.0,
        label_prefix: str = "",
        add_on_click: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._color = color
        self._selected_color = selected_color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
        self._add_on_click = add_on_click

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        self._points: List[List[float]] = []  # [[x, y], ...]  mutable for drag
        self._artists: List = []  # Line2D per point (index-aligned)
        self._anns: List = []  # Annotation per point (or None)

        self._selected: Optional[int] = None
        self._drag_idx: Optional[int] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
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

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        """Replace all points."""
        self._points = [[float(x), float(y)] for x, y in points]
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
        if self._label_prefix:
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

    # ── private: artists ──────────────────────────────────────────────────

    def _remove_all_artists(self):
        for lst in (self._artists, self._anns):
            for a in lst:
                if a is not None:
                    try:
                        a.remove()
                    except Exception:
                        pass
            lst.clear()

    def _draw_all(self):
        for idx in range(len(self._points)):
            self._append_artist(idx)

    def _append_artist(self, idx: int):
        if self._ax is None:
            return
        x, y = self._points[idx]
        selected = idx == self._selected
        color = self._selected_color if selected else self._color
        ms = self._size * 1.4 if selected else self._size
        mew = 2.0 if selected else 0.8
        (line,) = self._ax.plot(
            x,
            y,
            marker=self._marker,
            markersize=ms,
            color=color,
            markeredgecolor="white",
            markeredgewidth=mew,
            linestyle="none",
            zorder=8,
            animated=False,
        )
        self._artists.append(line)
        ann = None
        if self._label_prefix:
            ann = self._ax.annotate(
                f"{self._label_prefix}{idx + 1}",
                xy=(x, y),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
                zorder=9,
                animated=False,
            )
        self._anns.append(ann)

    def _update_artist_appearance(self, idx: int):
        if idx >= len(self._artists):
            return
        selected = idx == self._selected
        color = self._selected_color if selected else self._color
        ms = self._size * 1.4 if selected else self._size
        mew = 2.0 if selected else 0.8
        line = self._artists[idx]
        line.set_color(color)
        line.set_markersize(ms)
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
                ann.set_text(f"{self._label_prefix}{idx + 1}")

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
        if event.inaxes is not self._ax or event.button != 1:
            return
        if event.xdata is None or event.dblclick:
            return
        if self._canvas._overlay_consuming_event:
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
        elif self._add_on_click:
            x = max(0.0, min(event.xdata, (self._img_w or 1) - 1))
            y = max(0.0, min(event.ydata, (self._img_h or 1) - 1))
            idx = self.add_point(x, y)
            self._canvas._overlay_consuming_event = True
            old_sel = self._selected
            self._selected = idx
            if old_sel is not None:
                self._update_artist_appearance(old_sel)
            self._update_artist_appearance(idx)
            self.point_added.emit(idx, x, y)
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
            self.point_moved.emit(idx, self._points[idx][0], self._points[idx][1])
            self._canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("delete", "backspace") and self._selected is not None:
            self.remove_point(self._selected)


# ---------------------------------------------------------------------------
# PatternOverlay — milling shape patches (pixel-space coordinates)
# ---------------------------------------------------------------------------


class PatternOverlay(CanvasOverlay):
    """Renders milling pattern shapes as matplotlib patches.

    Coordinates must be in image pixel space (caller handles unit conversion).
    Supported pattern attributes:
      - Rectangle / Bitmap : ``centre_x, centre_y, width, height``
      - Circle             : ``centre_x, centre_y, radius``
      - Line               : ``start_x, start_y, end_x, end_y``
      - Polygon            : ``vertices`` (N×2 array)
    """

    def __init__(self, patterns=(), color: str = "cyan", alpha: float = 0.4):
        self._patterns = list(patterns)
        self._color = color
        self._alpha = alpha
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

    def set_patterns(self, patterns) -> None:
        self._patterns = list(patterns)
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
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        kw = dict(
            edgecolor=self._color,
            facecolor=self._color,
            alpha=self._alpha,
            linewidth=1.5,
            zorder=6,
        )
        for pat in self._patterns:
            try:
                artist = _pattern_to_artist(pat, kw, mpatches, mlines)
                if artist is not None:
                    self._ax.add_artist(artist)
                    self._artists.append(artist)
            except Exception:
                _logger.exception("PatternOverlay: failed to render %r", pat)


def _pattern_to_artist(pat, kw, mpatches, mlines):
    name = type(pat).__name__
    if any(k in name for k in ("Rectangle", "Bitmap")):
        return mpatches.Rectangle(
            (pat.centre_x - pat.width / 2, pat.centre_y - pat.height / 2),
            pat.width,
            pat.height,
            **kw,
        )
    if "Circle" in name:
        return mpatches.Circle(
            (pat.centre_x, pat.centre_y),
            pat.radius,
            **{**kw, "facecolor": "none"},
        )
    if "Line" in name:
        return mlines.Line2D(
            [pat.start_x, pat.end_x],
            [pat.start_y, pat.end_y],
            color=kw["edgecolor"],
            linewidth=2,
            alpha=kw["alpha"],
            zorder=kw["zorder"],
        )
    if "Polygon" in name and hasattr(pat, "vertices"):
        return mpatches.Polygon(pat.vertices, closed=True, **kw)
    return None


class ScanDirectionArrowOverlay(CanvasOverlay):
    """Draws a yellow arrow indicating the milling scan direction.

    Only ``"TopToBottom"`` and ``"BottomToTop"`` are supported; all other
    values result in no arrow being shown.

    Call :meth:`set_arrow` with the pattern bounding box in pixel coordinates
    to update the arrow, or :meth:`clear` to hide it.
    """

    _SUPPORTED = {"TopToBottom", "BottomToTop"}

    def __init__(self, color: str = "yellow"):
        self._color = color
        self._ax = None
        self._canvas = None
        self._artist = None
        # (x_start, y_start, x_end, y_end) in pixel coords
        self._params: Optional[tuple] = None

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artist()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, _height: int) -> None:
        self._remove_artist()
        if width > 0 and self._params is not None:
            self._draw()

    def set_arrow(self, cx: float, cy: float, h_px: float, scan_direction: str) -> None:
        """Position the arrow based on pattern centre and height.

        Args:
            cx: pattern centre x in pixel coords
            cy: pattern centre y in pixel coords
            h_px: pattern height in pixels
            scan_direction: ``"TopToBottom"`` or ``"BottomToTop"``
        """
        if scan_direction not in self._SUPPORTED:
            self.clear()
            return
        margin = h_px * 0.15
        if scan_direction == "TopToBottom":
            # Arrow from near top edge → near bottom edge (↓ in image coords)
            x_s, y_s = cx, cy - h_px / 2 + margin
            x_e, y_e = cx, cy + h_px / 2 - margin
        else:  # BottomToTop
            # Arrow from near bottom edge → near top edge (↑ in image coords)
            x_s, y_s = cx, cy + h_px / 2 - margin
            x_e, y_e = cx, cy - h_px / 2 + margin
        self._params = (x_s, y_s, x_e, y_e)
        self._remove_artist()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Hide the arrow."""
        self._params = None
        self._remove_artist()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artist(self):
        if self._artist is not None:
            try:
                self._artist.remove()
            except Exception:
                pass
            self._artist = None

    def _draw(self):
        if self._ax is None or self._params is None:
            return
        x_s, y_s, x_e, y_e = self._params
        self._artist = self._ax.annotate(
            "",
            xy=(x_e, y_e),
            xytext=(x_s, y_s),
            arrowprops=dict(
                arrowstyle="-|>",
                color=self._color,
                lw=2.0,
                mutation_scale=18,
            ),
            zorder=10,
        )
