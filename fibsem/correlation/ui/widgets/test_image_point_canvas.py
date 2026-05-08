"""Test script for ImagePointCanvas.

Shows the canvas on the left with a synthetic image and pre-seeded points.
Controls on the right: add points of each type, clear, load a real FIB image,
and an event log.

Usage
-----
    python fibsem/correlation/ui/widgets/test_image_point_canvas.py
"""
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas


def _synthetic_image(h: int = 512, w: int = 512) -> np.ndarray:
    """Gradient + Gaussian blobs — stands in for a real FIB image."""
    rng = np.random.default_rng(42)
    base = np.linspace(0.0, 0.3, w)[None, :] + np.linspace(0.0, 0.3, h)[:, None]
    noise = rng.normal(0, 0.05, (h, w))
    img = base + noise

    # A few bright blobs
    for cx, cy in [(120, 150), (300, 200), (420, 380), (200, 400)]:
        yy, xx = np.mgrid[:h, :w]
        blob = 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 20 ** 2))
        img += blob

    return np.clip(img, 0, 1)


def _default_coords() -> list[Coordinate]:
    return [
        Coordinate(PointXYZ(120, 150, 0), PointType.FIB),
        Coordinate(PointXYZ(300, 200, 0), PointType.FIB),
        Coordinate(PointXYZ(420, 380, 0), PointType.FM),
        Coordinate(PointXYZ(200, 400, 0), PointType.FM),
        Coordinate(PointXYZ(310, 290, 5), PointType.POI),
        Coordinate(PointXYZ(256, 256, 0), PointType.SURFACE),
    ]


_ADD_COLORS = {
    PointType.FIB:     "#4caf50",
    PointType.FM:      "#00bcd4",
    PointType.POI:     "#e040fb",
    PointType.SURFACE: "#f44336",
}

_COUNTERS: dict[PointType, int] = {t: 0 for t in PointType}


class TestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImagePointCanvas — test")
        self.resize(1100, 700)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- Left: canvas ---
        self.canvas = ImagePointCanvas()
        splitter.addWidget(self.canvas)

        # --- Right: controls ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        right_layout.addWidget(QLabel("<b>Add point</b>"))
        for pt in PointType:
            btn = QPushButton(f"Add {pt.value}")
            btn.setStyleSheet(
                f"color: {_ADD_COLORS[pt]}; border: 1px solid {_ADD_COLORS[pt]};"
                "padding: 4px; border-radius: 3px;"
            )
            btn.clicked.connect(lambda checked, t=pt: self._add_point(t))
            right_layout.addWidget(btn)

        right_layout.addSpacing(4)

        btn_clear = QPushButton("Clear all points")
        btn_clear.clicked.connect(self._clear_points)
        right_layout.addWidget(btn_clear)

        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset)
        right_layout.addWidget(btn_reset)

        btn_view = QPushButton("Reset view")
        btn_view.clicked.connect(self.canvas.reset_view)
        right_layout.addWidget(btn_view)

        right_layout.addSpacing(4)

        right_layout.addWidget(QLabel("<i>Scroll: zoom  |  Drag (empty): pan<br>Right-click: add point</i>"))

        right_layout.addSpacing(4)

        btn_load = QPushButton("Load FIB image…")
        btn_load.clicked.connect(self._load_image)
        right_layout.addWidget(btn_load)

        right_layout.addSpacing(8)
        right_layout.addWidget(QLabel("<b>Event log</b>"))

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: arial; font-size: 11px;")
        right_layout.addWidget(self.log)

        splitter.addWidget(right)
        splitter.setSizes([750, 350])

        # Wire canvas signals
        self.canvas.point_selected.connect(self._on_selected)
        self.canvas.point_moved.connect(self._on_moved)
        self.canvas.point_removed.connect(self._on_removed)
        self.canvas.canvas_clicked.connect(self._on_canvas_click)
        self.canvas.point_add_requested.connect(self._on_add_requested)

        # Seed
        self._image = _synthetic_image()
        self._coords: list[Coordinate] = _default_coords()
        self.canvas.set_image(self._image)
        self.canvas.set_coordinates(self._coords)
        self._log("Ready — drag points to move, click to select")

    # ------------------------------------------------------------------

    def _add_point(self, pt: PointType) -> None:
        _COUNTERS[pt] += 1
        # Place new point near image centre with a small offset
        cx = 256 + _COUNTERS[pt] * 12
        cy = 256 + _COUNTERS[pt] * 8
        coord = Coordinate(PointXYZ(cx, cy, 0), pt)
        self._coords.append(coord)
        self.canvas.set_coordinates(self._coords)
        self._log(f"Added {pt.value} at ({cx}, {cy})")

    def _clear_points(self) -> None:
        self._coords = []
        self.canvas.set_coordinates(self._coords)
        self._log("Cleared all points")

    def _reset(self) -> None:
        self._coords = _default_coords()
        self.canvas.set_image(self._image)
        self.canvas.set_coordinates(self._coords)
        self._log("Reset to default coordinates")

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.tif *.tiff *.png *.jpg)"
        )
        if not path:
            return
        try:
            from fibsem.structures import FibsemImage
            fib = FibsemImage.load(path)
            img = fib.filtered_data
            if img.ndim == 3:
                img = img[0]
            self._image = img.astype(np.float32)
            if self._image.max() > 1.0:
                self._image = self._image / self._image.max()
            self.canvas.set_image(self._image)
            self.canvas.set_coordinates(self._coords)
            self._log(f"Loaded: {path}  shape={img.shape}")
        except Exception as exc:
            self._log(f"Load error: {exc}")

    def _on_selected(self, coord: Coordinate) -> None:
        self._log(
            f"Selected {coord.point_type.value}  "
            f"x={coord.point.x:.1f}  y={coord.point.y:.1f}  z={coord.point.z:.1f}"
        )

    def _on_moved(self, coord: Coordinate) -> None:
        self._log(
            f"Moved   {coord.point_type.value}  "
            f"x={coord.point.x:.1f}  y={coord.point.y:.1f}  z={coord.point.z:.1f}"
        )

    def _on_removed(self, coord: Coordinate) -> None:
        if coord in self._coords:
            self._coords.remove(coord)
            self.canvas.set_coordinates(self._coords)
            self._log(f"Removed {coord.point_type.value} at ({coord.point.x:.1f}, {coord.point.y:.1f})")

    def _on_canvas_click(self, x: float, y: float) -> None:
        self._log(f"Canvas click  x={x:.1f}  y={y:.1f}  (no point hit)")

    def _on_add_requested(self, x: float, y: float, pt: PointType) -> None:
        coord = Coordinate(PointXYZ(x, y, 0), pt)
        self._coords.append(coord)
        self.canvas.set_coordinates(self._coords)
        self._log(f"Added {pt.value} at ({x:.1f}, {y:.1f})")

    def _log(self, msg: str) -> None:
        self.log.append(msg)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#2b2d31"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#F0F1F2"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1e2124"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2b2d31"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#F0F1F2"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#3a3d42"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#F0F1F2"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2d3f5c"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    win = TestWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
