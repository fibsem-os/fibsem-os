"""Test script for SelectedLamellaWidget (objective + pose list)."""
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Lamella
from fibsem.structures import FibsemStagePosition, MicroscopeState
from fibsem.ui.widgets.selected_lamella_widget import SelectedLamellaWidget


def _pose(x: float, y: float, z: float, r: float = 0.0, t: float = 0.0,
          objective: float = None) -> MicroscopeState:
    state = MicroscopeState(
        stage_position=FibsemStagePosition(x=x, y=y, z=z, r=r, t=t)
    )
    if objective is not None:
        state.objective_position = objective
    return state


def _make_lamella(number: int, petname: str, with_objective: bool,
                  with_fluorescence: bool) -> Lamella:
    lamella = Lamella(path=Path(f"/tmp/test/{petname}"), number=number, petname=petname)
    lamella.milling_pose = _pose(1e-3 * number, 2e-3, 3e-3, r=0.1, t=0.2)
    if with_fluorescence:
        # objective in metres (500 µm) when requested
        lamella.fluorescence_pose = _pose(
            1e-3 * number, 2e-3, 3.5e-3, objective=5e-4 if with_objective else None
        )
    return lamella


SAMPLE = {
    "A — milling + fluorescence + objective": _make_lamella(
        1, "01-humble-molly", with_objective=True, with_fluorescence=True
    ),
    "B — milling + fluorescence (no objective)": _make_lamella(
        2, "02-jolly-koala", with_objective=False, with_fluorescence=True
    ),
    "C — milling only": _make_lamella(
        3, "03-brave-falcon", with_objective=False, with_fluorescence=False
    ),
}


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Selected Lamella Widget — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.resize(480, 360)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Selected Lamella — Test")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        self.widget = SelectedLamellaWidget()
        root.addWidget(self.widget)

        self.log_label = QLabel("(events will appear here)")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        self.log_label.setWordWrap(True)
        root.addWidget(self.log_label)

        # one button per sample lamella, plus a clear-to-None button
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        for label, lamella in SAMPLE.items():
            btn = QPushButton(label.split(" ", 1)[0])  # "A" / "B" / "C"
            btn.setToolTip(label)
            btn.clicked.connect(lambda _=False, lam=lamella: self._select(lam))
            btn_row.addWidget(btn)
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._select(None))
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

        # log all five signals; mutate the model where the real app would
        self.widget.objective_position_changed.connect(
            lambda v: self._log(f"objective_position_changed: {v:.1f} µm")
        )
        self.widget.use_current_objective_requested.connect(
            lambda: self._log("use_current_objective_requested")
        )
        self.widget.apply_objective_to_all_requested.connect(
            lambda: self._log("apply_objective_to_all_requested")
        )
        self.widget.pose_update_requested.connect(self._on_pose_update)
        self.widget.pose_move_to_requested.connect(
            lambda name: self._log(f"pose_move_to_requested: {name}")
        )

        self._current: Lamella = None
        self._select(SAMPLE["A — milling + fluorescence + objective"])

    def _select(self, lamella) -> None:
        self._current = lamella
        self.widget.set_lamella(lamella)
        self._log(f"set_lamella: {lamella.name if lamella else None}")

    def _on_pose_update(self, name: str) -> None:
        """Mimic AutoLamellaUI: store a new pose then refresh the row in place."""
        if self._current is None:
            return
        new_state = _pose(9e-3, 8e-3, 7e-3)
        self._current.poses[name] = new_state
        self.widget.refresh_pose(name, new_state.stage_position.pretty)
        self._log(f"pose_update_requested: {name} (refreshed in place)")

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
