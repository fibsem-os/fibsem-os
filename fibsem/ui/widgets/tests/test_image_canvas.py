"""Standalone test script for FibsemImageCanvas and overlays.

Run directly:
    python fibsem/ui/widgets/tests/test_image_canvas.py
"""

import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.image_canvas import (
    FibsemImageCanvas,
    PatternOverlay,
    PointOverlay,
    PointsOverlay,
    RectOverlay,
)
from fibsem.ui.stylesheets import NAPARI_STYLE


class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FibsemImageCanvas — test")
        self.resize(900, 650)
        self.setStyleSheet(NAPARI_STYLE)

        # ── canvas ────────────────────────────────────────────────────
        self.canvas = FibsemImageCanvas()
        self.canvas.canvas_clicked.connect(
            lambda x, y: self._log(f"click          ({x:.1f}, {y:.1f})")
        )
        self.canvas.canvas_double_clicked.connect(
            lambda x, y: self._log(f"double-click   ({x:.1f}, {y:.1f})")
        )
        self.canvas.canvas_right_clicked.connect(
            lambda x, y: self._log(f"right-click    ({x:.1f}, {y:.1f})")
        )
        self.canvas.canvas_scrolled.connect(
            lambda x, y, d: self._log(
                f"scroll  dir={'+' if d > 0 else '-'}1  ({x:.1f}, {y:.1f})"
            )
        )

        # ── overlays ──────────────────────────────────────────────────
        self.rect_drag = RectOverlay(
            color="yellow",
            facecolor="yellow",
            alpha=0.5,
            linewidth=2,
            linestyle="solid",
            resizable=False,
        )
        self.rect_resize = RectOverlay(
            color="white",
            facecolor=None,
            alpha=0.9,
            linewidth=2,
            linestyle="--",
            resizable=True,
        )
        self.points = PointsOverlay(color="cyan", marker="o", size=8, label_prefix="P")
        self.patterns = PatternOverlay(color="magenta", alpha=0.4)
        self.point_overlay = PointOverlay(
            color="lime",
            selected_color="yellow",
            marker="o",
            size=10,
            label_prefix="Q",
        )

        self.rect_drag.rect_changed.connect(
            lambda d: self._log(
                f"drag-rect  cx={d['cx']} cy={d['cy']}  {d['width']}×{d['height']}"
            )
        )
        self.rect_resize.rect_changed.connect(
            lambda d: self._log(
                f"resize-rect  cx={d['cx']} cy={d['cy']}  {d['width']}×{d['height']}"
            )
        )
        self.point_overlay.point_added.connect(
            lambda i, x, y: self._log(f"point added    Q{i + 1} ({x:.1f}, {y:.1f})")
        )
        self.point_overlay.point_selected.connect(
            lambda i, x, y: self._log(f"point selected Q{i + 1} ({x:.1f}, {y:.1f})")
        )
        self.point_overlay.point_moved.connect(
            lambda i, x, y: self._log(f"point moved    Q{i + 1} ({x:.1f}, {y:.1f})")
        )
        self.point_overlay.point_removed.connect(
            lambda i: self._log(f"point removed  Q{i + 1}")
        )

        # ── controls ──────────────────────────────────────────────────
        btn_load = QPushButton("Load image")
        btn_clear = QPushButton("Clear")
        btn_reset = QPushButton("Reset view")
        btn_crosshair = QPushButton("Toggle crosshair")
        btn_drag_rect = QPushButton("Toggle drag-rect")
        btn_resize_rect = QPushButton("Toggle resize-rect")
        btn_points = QPushButton("Toggle points")
        btn_patterns = QPushButton("Toggle patterns")
        btn_point_overlay = QPushButton("Toggle point-overlay")

        btn_load.clicked.connect(self._load)
        btn_clear.clicked.connect(self.canvas.clear)
        btn_reset.clicked.connect(self.canvas.reset_view)
        btn_crosshair.clicked.connect(self._toggle_crosshair)
        btn_drag_rect.clicked.connect(lambda: self._toggle(self.rect_drag))
        btn_resize_rect.clicked.connect(lambda: self._toggle(self.rect_resize))
        btn_points.clicked.connect(lambda: self._toggle(self.points))
        btn_patterns.clicked.connect(lambda: self._toggle(self.patterns))
        btn_point_overlay.clicked.connect(lambda: self._toggle(self.point_overlay))

        self._crosshair_on = False

        ctrl = QVBoxLayout()
        for btn in (
            btn_load,
            btn_clear,
            btn_reset,
            btn_crosshair,
            btn_drag_rect,
            btn_resize_rect,
            btn_points,
            btn_patterns,
            btn_point_overlay,
        ):
            ctrl.addWidget(btn)
        ctrl.addStretch()

        self._log_label = QLabel("—")
        self._log_label.setWordWrap(True)
        self._log_label.setStyleSheet(
            "color: #ccc; font-family: arial; font-size: 10px;"
        )
        ctrl.addWidget(self._log_label)

        main = QHBoxLayout(self)
        main.addWidget(self.canvas, 1)
        main.addLayout(ctrl)

        self._overlay_state: dict = {}  # overlay → currently attached?

    def _load(self):
        img = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
        self.canvas.set_image(img)

    def _toggle(self, overlay):
        if self._overlay_state.get(id(overlay), False):
            self.canvas.remove_overlay(overlay)
            self._overlay_state[id(overlay)] = False
            self._log(f"{type(overlay).__name__} removed")
        else:
            # Seed points/patterns with dummy data on first add
            if overlay is self.points and not self._overlay_state.get("points_set"):
                w = self.canvas.img_width or 512
                h = self.canvas.img_height or 512
                self.points.set_points(
                    [(w * 0.3, h * 0.3), (w * 0.7, h * 0.5), (w * 0.5, h * 0.8)]
                )
                self._overlay_state["points_set"] = True
            self.canvas.add_overlay(overlay)
            self._overlay_state[id(overlay)] = True
            self._log(f"{type(overlay).__name__} added")

    def _toggle_crosshair(self):
        self._crosshair_on = not self._crosshair_on
        self.canvas.set_crosshair_visible(self._crosshair_on)
        self._log(f"crosshair {'on' if self._crosshair_on else 'off'}")

    def _log(self, msg: str):
        self._log_label.setText(msg)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = TestWindow()
    win.show()

    # Pre-load an image on startup
    win._load()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
