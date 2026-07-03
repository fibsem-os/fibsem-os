"""Standalone demo: active-overlay input model + toolbar mode toggle.

Run:
    PYTHONPATH=<worktree> python fibsem/ui/widgets/tests/test_active_overlay_mode.py

A FibsemImageCanvas with a PointOverlay (right-click adds). Use the "Enter POI
mode" button to make the overlay active — while active the contextual toolbar
toggle (top-right) appears, the overlay owns input, and the canvas's
double-click / right-click signals are suppressed. Click the toolbar toggle to
drop to Move (the two signals fire again, logged below) and back.
"""
import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.overlays.point_overlay import PointOverlay


def main() -> None:
    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(820, 640)
    canvas = FibsemImageCanvas()
    canvas.set_image(FibsemImage(data=(np.random.rand(512, 512) * 255).astype(np.uint8)))

    overlay = PointOverlay(color="magenta", selected_color="cyan", marker="o", size=9)
    canvas.add_overlay(overlay)

    log = QLabel("Move mode: double-click and right-click fire (stage move / milling menu).")
    log.setWordWrap(True)
    canvas.canvas_double_clicked.connect(lambda x, y, m: log.setText(f"double-click → STAGE MOVE @ ({x:.0f},{y:.0f})"))
    canvas.canvas_right_clicked.connect(lambda x, y, m: log.setText(f"right-click → MILLING MENU @ ({x:.0f},{y:.0f})"))

    btn_enter = QPushButton("Enter POI mode")
    btn_exit = QPushButton("Exit mode")
    btn_enter.clicked.connect(lambda: canvas.enter_overlay_mode(overlay, "POI", icon="mdi:map-marker"))
    btn_exit.clicked.connect(lambda: canvas.exit_overlay_mode(overlay))

    bar = QHBoxLayout()
    bar.addWidget(btn_enter)
    bar.addWidget(btn_exit)
    bar.addStretch()
    lay = QVBoxLayout(win)
    lay.addWidget(canvas, 1)
    lay.addLayout(bar)
    lay.addWidget(log)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
