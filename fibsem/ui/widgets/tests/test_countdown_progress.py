"""Small example: countdown progress bar that drains left → right each second."""

import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_BAR_STYLE = """
QProgressBar {
    background-color: #3a3d42;
    border: none;
    border-radius: 4px;
    height: 22px;
    text-align: center;
    color: white;
    font-size: 11px;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #e65100;
    border-radius: 4px;
}
"""

_BTN_STYLE = """
QPushButton {
    background-color: #3a3d42;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 11px;
}
QPushButton:hover { background-color: #4a4d52; }
QPushButton:pressed { background-color: #2a2d32; }
"""


class CountdownProgressBar(QWidget):
    """A progress bar that counts down from *duration* seconds.

    The bar starts full and drains left → right each second.
    """

    def __init__(self, duration: int = 30, parent=None):
        super().__init__(parent)
        self._duration = duration
        self._remaining = duration

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(self._duration)
        self.bar.setValue(self._duration)
        # InvertedAppearance: the chunk shrinks from left → right as value decreases
        # self.bar.setInvertedAppearance(True)
        self.bar.setFormat(f"{self._duration}s remaining")
        self.bar.setTextVisible(True)
        self.bar.setStyleSheet(_BAR_STYLE)
        layout.addWidget(self.bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._btn_toggle = QPushButton("Start")
        self._btn_toggle.setStyleSheet(_BTN_STYLE)
        self._btn_toggle.clicked.connect(self.toggle)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setStyleSheet(_BTN_STYLE)
        self._btn_reset.clicked.connect(self.reset)

        btn_row.addWidget(self._btn_toggle)
        btn_row.addWidget(self._btn_reset)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._remaining <= 0:
            self.reset()
        self._timer.start()
        self._btn_toggle.setText("Pause")

    def pause(self):
        self._timer.stop()
        self._btn_toggle.setText("Resume")

    def toggle(self):
        if self._timer.isActive():
            self.pause()
        else:
            self.start()

    def reset(self):
        self._timer.stop()
        self._remaining = self._duration
        self.bar.setValue(self._duration)
        self.bar.setFormat(f"{self._duration}s remaining")
        self._btn_toggle.setText("Start")

    def set_duration(self, seconds: int):
        self._duration = seconds
        self.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick(self):
        self._remaining -= 1
        self.bar.setValue(self._remaining)
        if self._remaining > 0:
            self.bar.setFormat(f"{self._remaining}s remaining")
        else:
            self.bar.setFormat("Done")
            self._timer.stop()
            self._btn_toggle.setText("Reset")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    w = QWidget()
    w.setWindowTitle("Countdown Progress Bar")
    w.setStyleSheet("background-color: #262930;")
    layout = QVBoxLayout(w)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)

    label = QLabel("Countdown Timer Example")
    label.setStyleSheet("color: #d1d2d4; font-size: 13px; font-weight: bold;")
    layout.addWidget(label)

    layout.addWidget(CountdownProgressBar(duration=30))
    layout.addWidget(CountdownProgressBar(duration=10))

    w.resize(320, 180)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
