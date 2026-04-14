"""Quick demo for the QuadMenuOverlay widget.

Run:
    python scripts/test_radial_menu.py

Right-click and hold anywhere in the grey area, drag to a sector, release.
"""

import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter

from fibsem.ui.widgets.radial_menu import QuadMenuOverlay


class DemoWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quad Menu Demo — right-click to open")
        self.setFixedSize(500, 400)

        self._label = QtWidgets.QLabel(
            "Right-click anywhere\nDrag to a section and release", self
        )
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: white; font-size: 16px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._label)

        self._quad_menu = QuadMenuOverlay(
            callbacks={
                "N": lambda: self._on_select("North"),
                "E": lambda: self._on_select("East"),
                "S": lambda: self._on_select("South"),
                "W": lambda: self._on_select("West"),
            }
        )

    def _on_select(self, name: str) -> None:
        print(f"Selected: {name}")
        self._label.setText(f"Selected: {name}\n\nRight-click to try again")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._quad_menu.show_at(event.globalPos())
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 45))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DemoWidget()
    w.show()
    sys.exit(app.exec_())
