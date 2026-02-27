"""Test script for LamellaCardContainer."""
import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    DefectState,
    Lamella,
)
from fibsem.ui.widgets.lamella_card_widget import LamellaCardContainer


def _make_lamella(number, petname, last_task="", in_progress="", has_defect=False, requires_rework=False):
    lamella = Lamella(path=Path(f"/tmp/test/{petname}"), number=number, petname=petname)
    if last_task:
        state = AutoLamellaTaskState(name=last_task, status=AutoLamellaTaskStatus.Completed)
        state.end_timestamp = time.time()
        lamella.task_history.append(state)
    if in_progress:
        lamella.task_state.name = in_progress
        lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    if has_defect:
        lamella.defect = DefectState(has_defect=True, requires_rework=requires_rework, description="test defect")
    return lamella


SAMPLE = [
    _make_lamella(1, "01-humble-molly"),
    _make_lamella(2, "01-hearty-wombat", last_task="Acquire Reference Image"),
    _make_lamella(3, "02-jolly-koala", last_task="Mill Rough", in_progress="Mill Polishing"),
    _make_lamella(4, "03-brave-falcon", last_task="Mill Rough", has_defect=True, requires_rework=True),
    _make_lamella(5, "04-eager-otter", last_task="Mill Polishing", has_defect=True),
    _make_lamella(6, "05-swift-eagle", last_task="Mill Rough"),
    _make_lamella(7, "06-bold-panda", last_task="Acquire Reference Image", has_defect=True, requires_rework=True),
]


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lamella Card Container — Test")
        self.setStyleSheet("background: #1e2124; color: #d6d6d6;")
        self.resize(1200, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── button bar ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_add = QPushButton("Add Lamella")
        btn_add.clicked.connect(self._add_random)
        btn_row.addWidget(btn_add)

        btn_refresh = QPushButton("Refresh All")
        btn_refresh.clicked.connect(self._container.refresh_all if hasattr(self, "_container") else lambda: None)
        btn_row.addWidget(btn_refresh)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self._container.clear())
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()

        btn_row.addWidget(QLabel("Columns:"))
        self._col_spin = QSpinBox()
        self._col_spin.setRange(1, 8)
        self._col_spin.setValue(4)
        self._col_spin.setFixedWidth(50)
        btn_row.addWidget(self._col_spin)

        self._selected_label = QLabel("Selected: —")
        self._selected_label.setStyleSheet("color: #909090; font-size: 12px;")
        btn_row.addWidget(self._selected_label)

        # ── scrollable card container ────────────────────────────────────
        self._container = LamellaCardContainer()
        self._container.defect_changed.connect(
            lambda lam: print(f"Defect changed: {lam.name}")
        )
        self._container.lamella_selected.connect(self._on_lamella_selected)

        # rewire refresh button now that container exists
        btn_refresh.clicked.disconnect()
        btn_refresh.clicked.connect(self._container.refresh_all)
        self._col_spin.valueChanged.connect(self._container.set_columns)

        scroll = QScrollArea()
        scroll.setWidget(self._container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        root.addLayout(btn_row)
        root.addWidget(scroll)

        for lamella in SAMPLE:
            self._container.add_lamella(lamella)

        self._counter = len(SAMPLE) + 1

    def _on_lamella_selected(self, lamella) -> None:
        if lamella is None:
            self._selected_label.setText("Selected: —")
            print("Selection cleared")
        else:
            self._selected_label.setText(f"Selected: {lamella.name}")
            print(f"Selected: {lamella.name}")

    def _add_random(self) -> None:
        import random
        adjectives = ["calm", "deft", "fierce", "gentle", "nimble"]
        animals = ["crane", "viper", "lynx", "bison", "heron"]
        name = f"{self._counter:02d}-{random.choice(adjectives)}-{random.choice(animals)}"
        lamella = _make_lamella(
            self._counter, name,
            last_task="Acquire Reference Image",
            has_defect=random.random() < 0.3,
            requires_rework=random.random() < 0.5,
        )
        self._container.add_lamella(lamella)
        self._counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
