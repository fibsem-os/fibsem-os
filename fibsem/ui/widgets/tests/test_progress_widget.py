"""Demo / manual test for FibsemProgressWidget.

Exercises all four modes (numeric, countdown, combined, indeterminate)
with simulate buttons.  Run without hardware:

    python fibsem/ui/widgets/tests/test_progress_widget.py
"""

import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.widgets.progress_widget import FibsemProgressWidget


# ---------------------------------------------------------------------------
# Helper — a labelled demo section
# ---------------------------------------------------------------------------

def _group(title: str, widget: QWidget, *buttons: QPushButton) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.addWidget(widget)
    btn_row = QHBoxLayout()
    for btn in buttons:
        btn_row.addWidget(btn)
    layout.addLayout(btn_row)
    return box


# ---------------------------------------------------------------------------
# Main demo window
# ---------------------------------------------------------------------------

class ProgressDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FibsemProgressWidget Demo")
        self.setStyleSheet("background-color: #262930; color: #d1d2d4;")

        outer = QVBoxLayout(self)
        outer.setSpacing(12)
        outer.setContentsMargins(16, 16, 16, 16)

        outer.addWidget(QLabel("FibsemProgressWidget — all modes"))

        # ── Numeric ──────────────────────────────────────────────────────
        self._num_widget = FibsemProgressWidget()
        self._num_step = 0
        self._num_total = 15

        btn_num_start = QPushButton("Start")
        btn_num_start.clicked.connect(self._start_numeric)
        btn_num_reset = QPushButton("Reset")
        btn_num_reset.clicked.connect(self._num_widget.reset)

        outer.addWidget(_group("Numeric (5 / 15)", self._num_widget, btn_num_start, btn_num_reset))

        # ── Countdown ────────────────────────────────────────────────────
        self._cd_widget = FibsemProgressWidget()
        self._cd_remaining = 30.0
        self._cd_total = 30.0
        self._cd_timer = QTimer(self)
        self._cd_timer.setInterval(1000)
        self._cd_timer.timeout.connect(self._tick_countdown)

        btn_cd_start = QPushButton("Start")
        btn_cd_start.clicked.connect(self._start_countdown)
        btn_cd_reset = QPushButton("Reset")
        btn_cd_reset.clicked.connect(self._reset_countdown)

        outer.addWidget(_group("Countdown (30s)", self._cd_widget, btn_cd_start, btn_cd_reset))

        # ── Combined ─────────────────────────────────────────────────────
        self._comb_widget = FibsemProgressWidget()
        self._comb_step = 0
        self._comb_total = 5
        self._comb_exposure = 8.0  # seconds per point
        self._comb_remaining = self._comb_exposure
        self._comb_timer = QTimer(self)
        self._comb_timer.setInterval(1000)
        self._comb_timer.timeout.connect(self._tick_combined)

        btn_comb_start = QPushButton("Start")
        btn_comb_start.clicked.connect(self._start_combined)
        btn_comb_reset = QPushButton("Reset")
        btn_comb_reset.clicked.connect(self._reset_combined)

        outer.addWidget(_group("Combined (5 points × 8s)", self._comb_widget, btn_comb_start, btn_comb_reset))

        # ── Message / indeterminate ───────────────────────────────────────
        self._msg_widget = FibsemProgressWidget()

        btn_msg_show = QPushButton("Show")
        btn_msg_show.clicked.connect(lambda: self._msg_widget.update_progress({"message": "Moving stage..."}))
        btn_msg_done = QPushButton("Done")
        btn_msg_done.clicked.connect(lambda: self._msg_widget.update_progress({"finished": True}))
        btn_msg_reset = QPushButton("Reset")
        btn_msg_reset.clicked.connect(self._msg_widget.reset)

        outer.addWidget(_group("Indeterminate / message", self._msg_widget, btn_msg_show, btn_msg_done, btn_msg_reset))

        self.resize(400, 480)

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    def _start_numeric(self):
        self._num_step = 0
        self._num_timer = QTimer(self)
        self._num_timer.setInterval(600)
        self._num_timer.timeout.connect(self._tick_numeric)
        self._tick_numeric()
        self._num_timer.start()

    def _tick_numeric(self):
        self._num_step += 1
        if self._num_step > self._num_total:
            self._num_widget.update_progress({"finished": True})
            self._num_timer.stop()
            return
        self._num_widget.update_progress({
            "current": self._num_step,
            "total": self._num_total,
            "message": f"Acquiring tile {self._num_step}…",
        })

    # ------------------------------------------------------------------
    # Countdown helpers
    # ------------------------------------------------------------------

    def _start_countdown(self):
        self._cd_remaining = self._cd_total
        self._tick_countdown()
        self._cd_timer.start()

    def _reset_countdown(self):
        self._cd_timer.stop()
        self._cd_widget.reset()

    def _tick_countdown(self):
        if self._cd_remaining <= 0:
            self._cd_widget.update_progress({"finished": True})
            self._cd_timer.stop()
            return
        self._cd_widget.update_progress({
            "remaining_seconds": self._cd_remaining,
            "total_seconds": self._cd_total,
        })
        self._cd_remaining -= 1

    # ------------------------------------------------------------------
    # Combined helpers
    # ------------------------------------------------------------------

    def _start_combined(self):
        self._comb_step = 0
        self._comb_remaining = self._comb_exposure
        self._tick_combined()
        self._comb_timer.start()

    def _reset_combined(self):
        self._comb_timer.stop()
        self._comb_widget.reset()

    def _tick_combined(self):
        self._comb_remaining -= 1
        if self._comb_remaining <= 0:
            self._comb_step += 1
            if self._comb_step >= self._comb_total:
                self._comb_widget.update_progress({"finished": True})
                self._comb_timer.stop()
                return
            self._comb_remaining = self._comb_exposure

        self._comb_widget.update_progress({
            "current": self._comb_step,
            "total": self._comb_total,
            "remaining_seconds": self._comb_remaining,
            "total_seconds": self._comb_exposure,
            "message": f"Burning point {self._comb_step + 1}/{self._comb_total}",
        })


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ProgressDemo()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
