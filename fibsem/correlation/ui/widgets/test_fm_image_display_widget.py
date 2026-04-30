"""Test script for FMImageDisplayWidget.

Shows the widget on the left and controls + event log on the right.

Usage
-----
    python fibsem/correlation/ui/widgets/test_fm_image_display_widget.py
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.correlation.ui.widgets.fm_image_display_widget import FMImageDisplayWidget

_DEV_PATH = "/home/patrick/github/fibsem/fibsem/applications/test-data"
_FM_IMAGE = "zstack-Feature-1-Active-002.ome.tiff"


class TestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FMImageDisplayWidget — test")
        self.resize(1100, 750)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- Left: widget under test ---
        self.fm_widget = FMImageDisplayWidget()
        splitter.addWidget(self.fm_widget)

        # --- Right: controls + log ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        btn_load_dev = QPushButton("Load dev FM image")
        btn_load_dev.clicked.connect(self._load_dev)
        right_layout.addWidget(btn_load_dev)

        btn_load = QPushButton("Load FM image…")
        btn_load.clicked.connect(self._load_dialog)
        right_layout.addWidget(btn_load)

        btn_add_pts = QPushButton("Add test points")
        btn_add_pts.clicked.connect(self._add_test_points)
        right_layout.addWidget(btn_add_pts)

        btn_clear_pts = QPushButton("Clear points")
        btn_clear_pts.clicked.connect(self._clear_points)
        right_layout.addWidget(btn_clear_pts)

        btn_reset_view = QPushButton("Reset view")
        btn_reset_view.clicked.connect(self.fm_widget.reset_view)
        right_layout.addWidget(btn_reset_view)

        right_layout.addSpacing(8)
        right_layout.addWidget(QLabel("<b>Event log</b>"))

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: arial; font-size: 11px;")
        right_layout.addWidget(self.log)

        splitter.addWidget(right)
        splitter.setSizes([750, 350])

        # Wire signals
        self.fm_widget.point_selected.connect(
            lambda c: self._log(f"Selected {c.point_type.value}  ({c.point.x:.1f}, {c.point.y:.1f})")
        )
        self.fm_widget.point_moved.connect(
            lambda c: self._log(f"Moved   {c.point_type.value}  ({c.point.x:.1f}, {c.point.y:.1f})")
        )
        self.fm_widget.point_removed.connect(self._on_removed)
        self.fm_widget.canvas_clicked.connect(
            lambda x, y: self._log(f"Click  ({x:.1f}, {y:.1f})")
        )
        self.fm_widget.point_add_requested.connect(self._on_add_requested)

        self._coords: list[Coordinate] = []

        # Auto-load dev image if present
        self._load_dev()

    def _load_dev(self) -> None:
        path = os.path.join(_DEV_PATH, _FM_IMAGE)
        if not os.path.exists(path):
            self._log(f"Dev image not found: {path}")
            return
        self._load_path(path)

    def _load_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FM image", _DEV_PATH,
            "Images (*.tif *.tiff *.ome.tiff)"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            from fibsem.fm.structures import FluorescenceImage
            fm = FluorescenceImage.load(path)
            self._log(
                f"Loaded: {os.path.basename(path)}\n"
                f"  shape: {fm.data.shape}  (C, Z, Y, X)\n"
                f"  channels: {len(fm.metadata.channels)}\n"
                f"  z-slices: {fm.data.shape[1]}"
            )
            self._coords = []
            self.fm_widget.set_fm_image(fm)
        except Exception as exc:
            self._log(f"Load error: {exc}")

    def _add_test_points(self) -> None:
        # Scatter a few FM-type points near centre
        from fibsem.fm.structures import FluorescenceImage
        # Use image shape if available, else defaults
        h, w = 512, 512
        if self.fm_widget._fm_image is not None:
            h, w = self.fm_widget._fm_image.data.shape[2:]
        cx, cy = w // 2, h // 2
        self._coords = [
            Coordinate(PointXYZ(cx - 80, cy - 60, 0), PointType.FM),
            Coordinate(PointXYZ(cx + 60, cy - 40, 0), PointType.FM),
            Coordinate(PointXYZ(cx - 30, cy + 80, 0), PointType.FM),
            Coordinate(PointXYZ(cx + 90, cy + 50, 0), PointType.POI),
        ]
        self.fm_widget.set_coordinates(self._coords)
        self._log(f"Added {len(self._coords)} test points")

    def _clear_points(self) -> None:
        self._coords = []
        self.fm_widget.set_coordinates(self._coords)
        self._log("Cleared points")

    def _on_removed(self, coord: Coordinate) -> None:
        if coord in self._coords:
            self._coords.remove(coord)
            self.fm_widget.set_coordinates(self._coords)
        self._log(f"Removed {coord.point_type.value} at ({coord.point.x:.1f}, {coord.point.y:.1f})")

    def _on_add_requested(self, x: float, y: float, pt: PointType) -> None:
        coord = Coordinate(PointXYZ(x, y, 0), pt)
        self._coords.append(coord)
        self.fm_widget.set_coordinates(self._coords)
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
