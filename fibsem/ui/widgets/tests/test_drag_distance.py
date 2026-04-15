"""Demo for DragDistanceOverlay.

Run:
    python fibsem/ui/widgets/tests/test_drag_distance.py

Right-click and drag to measure distance.
Hold Shift  → constrain to horizontal
Hold Ctrl   → constrain to vertical
Release     → prints distance
"""

import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter

from fibsem.ui.widgets.drag_distance import ConstraintMode, DragDistanceOverlay


class DemoWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drag Distance Demo")
        self.setFixedSize(600, 500)

        self._label = QtWidgets.QLabel(
            "Right-click and drag to measure\n"
            "Hold Shift = horizontal  |  Hold Ctrl = vertical",
            self,
        )
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #ccc; font-size: 14px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._label)

        self._measure = DragDistanceOverlay(
            scale=1.0,
            unit="px",
            on_measure=self._on_measure,
        )

    def _on_measure(self, distance: float) -> None:
        print(f"Distance: {distance:.1f} px")
        self._label.setText(
            f"Distance: {distance:.1f} px\n\nRight-click and drag to measure again"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._measure.start(event.globalPos(), self)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 35))
        # Draw a subtle grid for visual reference
        p.setPen(QColor(50, 50, 58))
        step = 50
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DemoWidget()
    w.show()
    sys.exit(app.exec_())
