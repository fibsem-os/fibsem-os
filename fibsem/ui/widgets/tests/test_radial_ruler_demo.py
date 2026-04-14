"""Combined demo: Radial Menu + Measure Tools on a FibsemImage.

Run:
    python fibsem/ui/widgets/tests/test_radial_ruler_demo.py

Right-click → radial menu:
  N — Ruler        left-click drag → line distance
  E — Rectangle    left-click drag → width × height + area
  S — Angle        two left-click drags → angle between arms
  W — Profile      left-click drag → intensity plot

Keyboard:
  P   — toggle annotation pin tool (click to place a pin)
  Del — clear all pins
  Z   — hold for loupe magnifier
  Esc — cancel active tool / clear selection
"""

import sys

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.angle_measure import AngleMeasureOverlay
from fibsem.ui.widgets.annotation_pin import AnnotationPinOverlay
from fibsem.ui.widgets.drag_distance import (
    DragDistanceOverlay, RectMeasureOverlay,
    _fmt_distance, _fmt_area,
)
from fibsem.ui.widgets.hud_ticker import HUDTicker
from fibsem.ui.widgets.lamella_task_image_widget import ZoomableImageView
from fibsem.ui.widgets.profile_line import ProfileLineOverlay
from fibsem.ui.widgets.radial_menu import RadialMenuOverlay
from fibsem.ui.widgets.zoom_loupe import ZoomLoupeOverlay

_TOOL_HINTS = {
    "none":    "Right-click = menu  |  P = pin  |  Hold Z = loupe",
    "ruler":   "Ruler — left-click drag    Shift=H  Ctrl=V  |  Right-click=menu",
    "rect":    "Rectangle — left-click drag    Shift=square  |  Right-click=menu",
    "angle":   "Angle — drag arm 1, release, drag arm 2, release  |  Right-click=menu",
    "profile": "Profile — left-click drag a line  |  Right-click=menu",
    "pin":     "Pin — left-click to place a pin    Del=clear all  |  Right-click=menu",
}


def _fibsem_to_pixmap(img: FibsemImage) -> QPixmap:
    arr = img.data
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MeasureImageView(ZoomableImageView):
    """ZoomableImageView with measurement tool mouse event hooks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_tool: str = "none"
        self._on_right_click = None
        self._on_left_press  = None
        self._on_mouse_move  = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton and self._on_right_click:
            self._on_right_click(event.globalPos())
        elif event.button() == Qt.LeftButton and self._active_tool != "none" and self._on_left_press:
            self._on_left_press(event.globalPos())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._on_mouse_move:
            self._on_mouse_move(event.pos())
        super().mouseMoveEvent(event)


class DemoWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Measure Tools Demo")
        self.setMinimumSize(900, 750)
        self.setStyleSheet("background: #1a1b1e;")

        # Fake FibsemImage: 800×600, 100 µm HFW → 125 nm/px
        self._image = FibsemImage.generate_blank_image(
            resolution=(800, 600), hfw=100e-6, random=True,
        )
        self._pixel_size = self._image.metadata.pixel_size.x

        # ── Layout ────────────────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._status_label = QtWidgets.QLabel(_TOOL_HINTS["none"])
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "color: #ccc; font-size: 12px; background: #24252a; padding: 6px;"
        )
        layout.addWidget(self._status_label)

        self._view = MeasureImageView(self)
        self._view.viewport().setMouseTracking(True)
        self._view.set_pixmap(_fibsem_to_pixmap(self._image))
        self._view.setMinimumHeight(420)
        layout.addWidget(self._view)

        # Profile plot docked below the image
        from fibsem.ui.widgets.profile_line import _ProfilePlotWidget
        self._profile_plot = _ProfilePlotWidget()
        layout.addWidget(self._profile_plot)

        # HUD ticker
        self._ticker = HUDTicker(parent=self._view)
        self._ticker.set_image(self._image)
        self._ticker.show()

        # ── Tools ─────────────────────────────────────────────────────
        self._quad_menu = RadialMenuOverlay(items=[
            ("Ruler",     self._select_ruler),
            ("Rectangle", self._select_rect),
            ("Angle",     self._select_angle),
            ("Profile",   self._select_profile),
            ("Pin",       lambda: self._set_tool("pin")),
            ("Clear",     self._clear),
        ])

        self._ruler   = DragDistanceOverlay(pixel_size=self._pixel_size,
                                            on_measure=self._on_ruler_result)
        self._rect    = RectMeasureOverlay(pixel_size=self._pixel_size,
                                           on_measure=self._on_rect_result)
        self._angle   = AngleMeasureOverlay(pixel_size=self._pixel_size,
                                            on_measure=self._on_angle_result)
        self._profile = ProfileLineOverlay(
            view=self._view,
            image_data=self._image.data,
            pixel_size=self._pixel_size,
        )
        self._profile.plot_widget = self._profile_plot

        self._pins  = AnnotationPinOverlay(view=self._view, parent=self._view)
        self._pins.show()

        self._loupe = ZoomLoupeOverlay(
            view=self._view,
            pixmap=_fibsem_to_pixmap(self._image),
            zoom=4.0, diameter=220,
        )

        self._view._on_right_click = self._quad_menu.show_at
        self._view._on_left_press  = self._on_left_press
        self._view._on_mouse_move  = self._loupe.on_mouse_move

    # ── Tool selection ────────────────────────────────────────────────

    def _select_ruler(self)   -> None: self._set_tool("ruler")
    def _select_rect(self)    -> None: self._set_tool("rect")
    def _select_angle(self)   -> None: self._set_tool("angle")
    def _select_profile(self) -> None: self._set_tool("profile")

    def _set_tool(self, name: str) -> None:
        self._view._active_tool = name
        pan = name == "none"
        self._view.setDragMode(
            QtWidgets.QGraphicsView.ScrollHandDrag if pan
            else QtWidgets.QGraphicsView.NoDrag
        )
        self._set_status(_TOOL_HINTS[name])

    def _clear(self) -> None:
        self._set_tool("none")

    # ── Mouse dispatch ────────────────────────────────────────────────

    def _on_left_press(self, global_pos) -> None:
        tool = self._view._active_tool
        if tool == "pin":
            self._pins.add_pin_at(global_pos)
            return

        zoom = self._view.transform().m11()
        eff_ps = self._pixel_size / zoom

        if tool == "ruler":
            self._ruler._pixel_size = eff_ps
            self._ruler.start(global_pos, self._view.viewport())
        elif tool == "rect":
            self._rect._pixel_size = eff_ps
            self._rect.start(global_pos, self._view.viewport())
        elif tool == "angle":
            self._angle._pixel_size = eff_ps
            if self._angle._phase == 0:
                self._angle.start(global_pos, self._view.viewport())
        elif tool == "profile":
            self._profile._pixel_size = eff_ps
            self._profile.start(global_pos, self._view.viewport())

    # ── Result callbacks ──────────────────────────────────────────────

    def _on_ruler_result(self, d: float) -> None:
        self._set_status(f"Distance: {_fmt_distance(d)}    {_TOOL_HINTS['ruler']}")

    def _on_rect_result(self, w: float, h: float, area: float) -> None:
        self._set_status(
            f"Rectangle: {_fmt_distance(w)} × {_fmt_distance(h)}  "
            f"area={_fmt_area(area)}    {_TOOL_HINTS['rect']}"
        )

    def _on_angle_result(self, deg: float) -> None:
        self._set_status(f"Angle: {deg:.2f}°    {_TOOL_HINTS['angle']}")

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    # ── Resize ────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._ticker._refit()
        self._pins._refit()

    # ── Key events ────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._loupe.on_key_press(key):
            return
        if key == Qt.Key_P:
            active = "none" if self._view._active_tool == "pin" else "pin"
            self._set_tool(active)
        elif key == Qt.Key_Delete:
            self._pins.clear()
        elif key == Qt.Key_Escape:
            self._clear()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if not self._loupe.on_key_release(event.key()):
            super().keyReleaseEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DemoWidget()
    w.show()
    sys.exit(app.exec_())
