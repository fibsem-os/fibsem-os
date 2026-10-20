"""Test script for LamellaListWidget."""

import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    DefectState,
    DefectType,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella.rows import LamellaListWidget
from fibsem.workflows.status import TaskState, TaskStatus


def _make_lamella(
    number: int,
    petname: str,
    last_task: str = "",
    in_progress: str = "",
    defect_state: DefectType = DefectType.NONE,
) -> Lamella:
    lamella = Lamella(path=Path(f"/tmp/test/{petname}"), number=number, petname=petname)

    if last_task:
        state = TaskState(name=last_task, status=TaskStatus.Completed)
        state.end_timestamp = time.time()
        lamella.task_history.append(state)

    if in_progress:
        lamella.task_state.name = in_progress
        lamella.task_state.status = TaskStatus.InProgress

    if defect_state != DefectType.NONE:
        lamella.defect = DefectState(state=defect_state, description="test defect")

    return lamella


SAMPLE = [
    _make_lamella(1, "01-humble-molly"),
    _make_lamella(2, "01-hearty-wombat", last_task="Acquire Reference Image"),
    _make_lamella(
        3, "02-jolly-koala", last_task="Mill Rough", in_progress="Mill Polishing"
    ),
    _make_lamella(
        4, "03-brave-falcon", last_task="Mill Rough", defect_state=DefectType.REWORK
    ),
    _make_lamella(
        5, "04-eager-otter", last_task="Mill Polishing", defect_state=DefectType.FAILURE
    ),
]


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lamella List Widget — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.resize(700, 420)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Run Experiment — Lamella")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        self.lamella_list = LamellaListWidget()
        root.addWidget(self.lamella_list)

        self.log_label = QLabel("(events will appear here)")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        root.addWidget(self.log_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_add = QPushButton("Add Lamella")
        btn_add.clicked.connect(self._add_random)
        btn_row.addWidget(btn_add)

        btn_refresh = QPushButton("Refresh All")
        btn_refresh.clicked.connect(self.lamella_list.refresh_all)
        btn_row.addWidget(btn_refresh)

        btn_selected = QPushButton("Show Selected")
        btn_selected.clicked.connect(self._show_selected)
        btn_row.addWidget(btn_selected)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.lamella_list.clear)
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()
        root.addLayout(btn_row)

        self.lamella_list.edit_requested.connect(
            lambda lam: self._log(f"Edit: {lam.name}")
        )
        self.lamella_list.remove_requested.connect(
            lambda lam: self._log(f"Removed: {lam.name}")
        )
        self.lamella_list.selection_changed.connect(
            lambda sel: self._log(f"Selected ({len(sel)}): {[lam.name for lam in sel]}")
        )

        for lamella in SAMPLE:
            self.lamella_list.add_lamella(lamella)

        self._counter = len(SAMPLE) + 1

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)

    def _add_random(self) -> None:
        import random

        adjectives = ["calm", "deft", "fierce", "gentle", "nimble"]
        animals = ["crane", "viper", "lynx", "bison", "heron"]
        name = (
            f"{self._counter:02d}-{random.choice(adjectives)}-{random.choice(animals)}"
        )
        lamella = _make_lamella(
            self._counter,
            name,
            last_task="Acquire Reference Image",
            defect_state=random.choice(
                [DefectType.NONE, DefectType.REWORK, DefectType.FAILURE]
            ),
        )
        self.lamella_list.add_lamella(lamella)
        self._counter += 1
        self._log(f"Added: {name}")

    def _show_selected(self) -> None:
        sel = self.lamella_list.get_selected()
        names = "\n".join(l.name for l in sel) if sel else "(none)"
        QMessageBox.information(self, "Selected", names)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
