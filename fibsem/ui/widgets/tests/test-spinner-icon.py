"""Test script for SVG spinner icons using superqt's QIconifyIcon + QTimer rotation."""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTransform
from superqt import QIconifyIcon


class SpinnerLabel(QLabel):
    """A QLabel that spins a QIconifyIcon at a given speed."""

    def __init__(self, icon_name: str, color: str = "#d6d6d6", size: int = 32,
                 step_deg: int = 15, interval_ms: int = 50, parent=None):
        super().__init__(parent)
        self._pixmap = QIconifyIcon(icon_name, color=color).pixmap(size, size)
        self._angle = 0
        self._step = step_deg
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self._render()

    def _tick(self):
        self._angle = (self._angle + self._step) % 360
        self._render()

    def _render(self):
        t = QTransform().rotate(self._angle)
        self.setPixmap(self._pixmap.transformed(t, Qt.SmoothTransformation))

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self._angle = 0
        self._render()


class SpinnerDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spinner Icon Demo")
        self.setStyleSheet("background-color: #262930; color: #d6d6d6;")
        self.resize(500, 300)

        root = QVBoxLayout(self)
        root.setSpacing(16)

        # Grid of spinner variants
        grid = QGridLayout()
        grid.setSpacing(16)

        self._spinners: list[SpinnerLabel] = []

        variants = [
            ("mdi:loading",         "#d6d6d6", 32, 15, 50,  "mdi:loading  step=15°  50ms"),
            ("mdi:loading",         "#4fc3f7", 48, 20, 40,  "mdi:loading  step=20°  40ms  blue  48px"),
            ("mdi:sync",            "#d6d6d6", 32, 10, 50,  "mdi:sync     step=10°  50ms"),
            ("mdi:cog-outline",     "#f0c040", 32, 5,  30,  "mdi:cog-outline  step=5°  30ms  slow"),
            ("mdi:refresh",         "#a5d6a7", 32, 30, 80,  "mdi:refresh  step=30°  80ms  fast"),
            ("mdi:progress-clock",  "#d6d6d6", 32, 15, 50,  "mdi:progress-clock"),
        ]

        for row, (icon, color, size, step, interval, label_text) in enumerate(variants):
            spinner = SpinnerLabel(icon, color=color, size=size,
                                   step_deg=step, interval_ms=interval)
            self._spinners.append(spinner)
            grid.addWidget(spinner, row, 0, Qt.AlignCenter)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            grid.addWidget(lbl, row, 1, Qt.AlignVCenter)

        root.addLayout(grid)

        # Controls
        btn_row = QHBoxLayout()
        self.btn_toggle = QPushButton("Start all")
        self.btn_toggle.setFixedWidth(120)
        self.btn_toggle.clicked.connect(self._toggle)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_toggle)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._running = False

    def _toggle(self):
        self._running = not self._running
        for s in self._spinners:
            s.start() if self._running else s.stop()
        self.btn_toggle.setText("Stop all" if self._running else "Start all")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SpinnerDemo()
    w.show()
    sys.exit(app.exec_())
