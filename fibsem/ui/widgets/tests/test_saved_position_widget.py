"""Standalone test script for SavedPositionListWidget — no hardware required."""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import FibsemStagePosition
from fibsem.ui.widgets.saved_position_widget import SavedPositionListWidget


def _make_position(name: str, x: float, y: float, z: float, r: float = 0.0, t: float = 0.0) -> FibsemStagePosition:
    pos = FibsemStagePosition(x=x, y=y, z=z, r=r, t=t, coordinate_system="RAW")
    pos.name = name
    return pos


SAMPLE = [
    _make_position("Lamella 01", x=0.001, y=0.002, z=0.003, r=0.0, t=-0.401),
    _make_position("Lamella 02", x=0.005, y=0.010, z=0.020, r=0.785, t=-0.524),
    _make_position("Grid Center", x=0.0, y=0.0, z=0.003, r=0.0, t=0.0),
]

_counter = len(SAMPLE) + 1


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SavedPositionListWidget — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.resize(700, 400)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Saved Positions (no hardware)")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        # Widget under test — no microscope connected
        self.widget = SavedPositionListWidget(microscope=None)
        root.addWidget(self.widget)

        self.log_label = QLabel("(events appear here)")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        root.addWidget(self.log_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_add = QPushButton("Add mock position")
        btn_add.clicked.connect(self._add_mock)
        btn_row.addWidget(btn_add)

        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(lambda: self.widget.set_positions([]))
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()
        root.addLayout(btn_row)

        self.widget.move_to_requested.connect(
            lambda pos: self._log(f"Move to: {pos.name} — {pos.pretty_string}")
        )
        self.widget.positions_updated.connect(
            lambda positions: self._log(f"Positions updated ({len(positions)} total)")
        )
        self.widget.position_selected.connect(
            lambda pos: self._log(f"Selected: {pos.name}")
        )

        for pos in SAMPLE:
            self.widget.add_position(pos)

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)

    def _add_mock(self) -> None:
        global _counter
        import random
        pos = _make_position(
            name=f"Position {_counter:02d}",
            x=random.uniform(-0.01, 0.01),
            y=random.uniform(-0.01, 0.01),
            z=random.uniform(0.001, 0.005),
            r=random.uniform(0, 3.14),
            t=random.uniform(-0.6, 0.0),
        )
        self.widget.add_position(pos)
        _counter += 1
        self._log(f"Added: {pos.name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
