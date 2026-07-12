"""Drag-to-measure ruler overlay (distance in SI units) for FibsemImageCanvas."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from fibsem.ui.stylesheets import CANVAS_BG as _BG
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


_RULER_PICK_PX = 10  # screen-space hit radius for ruler endpoints / line body


def _clampf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _format_distance(metres: float) -> str:
    """Auto-scale a distance in metres to a short SI string."""
    a = abs(metres)
    if a == 0:
        return "0 nm"
    if a < 1e-6:
        return f"{metres * 1e9:.1f} nm"
    if a < 1e-3:
        return f"{metres * 1e6:.2f} µm"
    if a < 1.0:
        return f"{metres * 1e3:.3f} mm"
    return f"{metres:.4f} m"


class RulerOverlay(CanvasOverlay):
    """Two-endpoint measure line on a :class:`FibsemImageCanvas`.

    Drag either endpoint — or the line body — to measure; the label shows the
    distance using the canvas pixel size (SI-formatted), or pixels when the
    pixel size is unknown.  Endpoints live in data (image-pixel) coordinates, so
    the ruler survives zoom / pan and image swaps.

    Inert until :meth:`set_visible(True)` (driven by the canvas ruler button):
    it builds no artists and captures no input while hidden, so canvases that
    never enable it are unaffected.
    """

    def __init__(self, color: str = "#ffd23f"):
        self._color = color
        self._ax = None
        self._canvas: Optional["FibsemImageCanvas"] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None
        self._pixel_size: Optional[float] = None

        # endpoints in data coords (None until seeded)
        self._p1: Optional[List[float]] = None
        self._p2: Optional[List[float]] = None
        self._visible: bool = False

        # artists
        self._line = None   # Line2D segment
        self._dots = None   # Line2D endpoint markers
        self._label = None  # Annotation (distance)

        # drag state: None | "p1" | "p2" | "line"
        self._drag: Optional[str] = None
        self._drag_start_data: Optional[Tuple[float, float]] = None
        self._drag_start_pts = None
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
        if self._canvas is not None:
            self._pixel_size = self._canvas._pixel_size
        if self._ax is None or width == 0 or height == 0 or not self._visible:
            self._remove_artists()  # inert while hidden / between images
            return
        if self._p1 is None or self._p2 is None:
            self._seed_default()
        else:
            self._clamp()
        self._rebuild()

    # ── public ────────────────────────────────────────────────────────────

    def set_visible(self, visible: bool) -> None:
        """Show/hide the ruler.  Endpoints persist while hidden."""
        self._visible = visible
        if not visible:
            self._remove_artists()
        elif self._img_w:
            if self._p1 is None or self._p2 is None:
                self._seed_default()
            self._rebuild()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def measurement(self) -> Optional[float]:
        """Current distance in metres (or pixels when no pixel size), or None."""
        if self._p1 is None or self._p2 is None:
            return None
        d = math.hypot(self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        return d * self._pixel_size if self._pixel_size else d

    # ── build / teardown ──────────────────────────────────────────────────

    def _seed_default(self) -> None:
        if not self._img_w or not self._img_h:
            return
        cx, cy = self._img_w / 2.0, self._img_h / 2.0
        half = self._img_w * 0.125
        self._p1 = [cx - half, cy]
        self._p2 = [cx + half, cy]

    def _clamp(self) -> None:
        if self._img_w is None:
            return
        for p in (self._p1, self._p2):
            p[0] = _clampf(p[0], 0.0, self._img_w)
            p[1] = _clampf(p[1], 0.0, self._img_h)

    def _remove_artists(self) -> None:
        for a in (self._line, self._dots, self._label):
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._line = self._dots = self._label = None

    def _xs_ys(self) -> Tuple[List[float], List[float]]:
        return [self._p1[0], self._p2[0]], [self._p1[1], self._p2[1]]

    def _rebuild(self) -> None:
        self._remove_artists()
        if self._p1 is None or self._p2 is None:
            return
        xs, ys = self._xs_ys()
        (self._line,) = self._ax.plot(
            xs, ys, color=self._color, linewidth=1.6,
            solid_capstyle="round", zorder=8,
        )
        (self._dots,) = self._ax.plot(
            xs, ys, linestyle="none", marker="o", markersize=6,
            markerfacecolor=self._color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
        )
        mx, my = (xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0
        self._label = self._ax.annotate(
            self._text(), xy=(mx, my), xytext=(0, 9),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8, color="#ffffff", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG,
                      edgecolor=self._color, alpha=0.8, linewidth=0.8),
        )

    def _text(self) -> str:
        d = math.hypot(self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        if self._pixel_size:
            return _format_distance(d * self._pixel_size)
        return f"{d:.0f} px"

    def _update_artists(self) -> None:
        xs, ys = self._xs_ys()
        self._line.set_data(xs, ys)
        self._dots.set_data(xs, ys)
        self._label.xy = ((xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0)
        self._label.set_text(self._text())

    # ── hit testing (screen space) ────────────────────────────────────────

    def _screen(self, p: List[float]) -> Tuple[float, float]:
        return self._ax.transData.transform((p[0], p[1]))

    def _hit(self, event) -> Optional[str]:
        for name, p in (("p1", self._p1), ("p2", self._p2)):
            sx, sy = self._screen(p)
            if math.hypot(event.x - sx, event.y - sy) <= _RULER_PICK_PX:
                return name
        x1, y1 = self._screen(self._p1)
        x2, y2 = self._screen(self._p2)
        if self._seg_dist(event.x, event.y, x1, y1, x2, y2) <= _RULER_PICK_PX * 0.6:
            return "line"
        return None

    @staticmethod
    def _seg_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = _clampf(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if not self._visible or self._line is None:
            return
        if self._canvas is not None and not self._canvas._overlay_input_allowed(self):
            return
        if event.inaxes is not self._ax or event.button != 1 or event.xdata is None:
            return
        if self._canvas._overlay_consuming_event:
            return
        hit = self._hit(event)
        if hit is None:
            return
        self._drag = hit
        self._drag_start_data = (event.xdata, event.ydata)
        self._drag_start_pts = (list(self._p1), list(self._p2))
        self._canvas._overlay_consuming_event = True
        self._set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _on_motion(self, event) -> None:
        if self._drag is None or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._drag_start_data[0]
        dy = event.ydata - self._drag_start_data[1]
        p1, p2 = self._drag_start_pts
        W, H = self._img_w, self._img_h
        if self._drag == "p1":
            self._p1 = [_clampf(p1[0] + dx, 0, W), _clampf(p1[1] + dy, 0, H)]
        elif self._drag == "p2":
            self._p2 = [_clampf(p2[0] + dx, 0, W), _clampf(p2[1] + dy, 0, H)]
        else:  # move the whole line, clamped so both endpoints stay in bounds
            minx, maxx = min(p1[0], p2[0]), max(p1[0], p2[0])
            miny, maxy = min(p1[1], p2[1]), max(p1[1], p2[1])
            dx = _clampf(dx, -minx, W - maxx)
            dy = _clampf(dy, -miny, H - maxy)
            self._p1 = [p1[0] + dx, p1[1] + dy]
            self._p2 = [p2[0] + dx, p2[1] + dy]
        self._update_artists()
        self._blit()

    def _on_release(self, event) -> None:
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = False
        if self._drag is not None:
            self._drag = None
            self._set_animated(False)
            self._blit_bg = None
            if self._canvas is not None:
                self._canvas.draw_idle()

    # ── blitting ──────────────────────────────────────────────────────────

    def _set_animated(self, val: bool) -> None:
        for a in (self._line, self._dots, self._label):
            if a is not None:
                a.set_animated(val)

    def _blit(self) -> None:
        if self._blit_bg is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        for a in (self._line, self._dots, self._label):
            if a is not None:
                self._ax.draw_artist(a)
        self._canvas.blit(self._ax.bbox)
