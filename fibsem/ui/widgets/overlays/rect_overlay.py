"""Configurable rectangle overlay (drag-only or drag+resize) for FibsemImageCanvas."""

from __future__ import annotations

from typing import List, Optional, Tuple

from matplotlib.patches import Rectangle as MplRectangle
from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.ui.widgets.overlays.base import CanvasOverlay  # noqa: F401  (re-exported by package)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.image_canvas import FibsemImageCanvas


_RECT_FRAC = 0.25
_RECT_OFFSET = (1.0 - _RECT_FRAC) / 2.0


def _default_extents(H: int, W: int) -> Tuple[float, float, float, float]:
    """Return (x0, x1, y0, y1) for a centred 25 % box."""
    rw, rh = W * _RECT_FRAC, H * _RECT_FRAC
    x0, y0 = W * _RECT_OFFSET, H * _RECT_OFFSET
    return x0, x0 + rw, y0, y0 + rh


_HANDLE_RADIUS_PX = 8  # screen-space hit radius for corner handles
# Corner handles: name -> (x_fraction, y_fraction) within the rect
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

        self._interactive: bool = True
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
                    visible=self._interactive,
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
        if not self._interactive:
            return
        # Another overlay owns input on this canvas
        if self._canvas is not None and not self._canvas._overlay_input_allowed(self):
            return
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
