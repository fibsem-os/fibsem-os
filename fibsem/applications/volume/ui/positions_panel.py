"""The volume window's Positions panel, the first tab beside the quad view:
what the microscope does with the positions now. Add one here, move to one,
update its poses from where the microscope is, remove one. What is stored on a
position from its images (the point of interest, the alignment area) and its
name and slices are edited on the Positions tab.

Every edit is the experiment's own method, the one the App Server's position
edits call, with its refusal said on a toast. The panel redraws when the
experiment's positions change, whoever changed them.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.volume.poses import (
    POSES,
    pose_stage_position,
    set_pose_now,
)
from fibsem.applications.volume.position import SEM_POSE, VolumePosition
from fibsem.constants import RADIANS_TO_DEGREES
from fibsem.ui import stylesheets
from fibsem.ui.run.run_refusal import refused, refused_during_run
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.ui.widgets.microscope_panel import MicroscopePanel
    from fibsem.workflows.session import AppSession

__all__ = ["PositionsPanel"]

STORED_NOTE = (
    "The point of interest, alignment area, name and slices are edited on the "
    "Positions tab."
)


def progress_text(experiment: "VolumeExperiment", position: VolumePosition) -> str:
    """What a position's row says on the right: "Failed", or slices cut."""
    if position.task_state.status is TaskStatus.Failed:
        return "Failed"
    return f"{experiment.slices_cut(position)} of {position.iterations}"


def pose_text(position: VolumePosition, pose_name: str) -> str:
    """A pose's tilt and when it was set; "Not set" without one (the imaging
    pose says it follows the milling pose)."""
    if pose_name not in position.poses:
        if pose_name == SEM_POSE and MILLING_POSE in position.poses:
            return "Same as milling"
        return "Not set"
    state = position.poses[pose_name]
    tilt = state.stage_position.t if state.stage_position is not None else None
    when = datetime.fromtimestamp(state.timestamp).strftime("%H:%M")
    if tilt is None:
        return when
    return f"{tilt * RADIANS_TO_DEGREES:.1f}° · {when}"


class PositionsPanel(QWidget):
    """The panel's Positions tab (a ``MicroscopePanel`` item tab)."""

    # the positions were drawn again
    items_changed = pyqtSignal()

    title = "Positions"
    # No item selection for the Image and Fluorescence tabs to follow.
    lamella_list = None

    def __init__(
        self,
        session: "AppSession",
        panel: "MicroscopePanel",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self._panel = panel

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Positions")
        title.setStyleSheet("font-weight: bold;")
        self.add_button = QPushButton("Add position here")
        self.add_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.add_button.setToolTip("Add a position milled where the microscope is now.")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.add_button)
        layout.addLayout(header)

        self.list = QListWidget()
        self.list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        layout.addWidget(self.list, 1)

        self.details = QWidget()
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        self.poses_title = QLabel()
        self.poses_title.setStyleSheet("font-weight: bold;")
        details.addWidget(self.poses_title)
        poses = QGridLayout()
        self.pose_labels: Dict[str, QLabel] = {}
        self.set_pose_buttons: Dict[str, QPushButton] = {}
        self.move_to_buttons: Dict[str, QPushButton] = {}
        for row, pose_name in enumerate(POSES):
            label = QLabel()
            set_button = QPushButton("Set from current")
            move_button = QPushButton("Move to")
            for button in (set_button, move_button):
                button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            set_button.clicked.connect(lambda _, n=pose_name: self._on_set_pose(n))
            move_button.clicked.connect(lambda _, n=pose_name: self._on_move_to(n))
            poses.addWidget(QLabel(pose_name), row, 0)
            poses.addWidget(label, row, 1)
            poses.addWidget(set_button, row, 2)
            poses.addWidget(move_button, row, 3)
            poses.setColumnStretch(1, 1)
            self.pose_labels[pose_name] = label
            self.set_pose_buttons[pose_name] = set_button
            self.move_to_buttons[pose_name] = move_button
        details.addLayout(poses)
        note = QLabel(STORED_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        details.addWidget(note)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet(stylesheets.DANGER_BUTTON_STYLESHEET)
        remove_row = QHBoxLayout()
        remove_row.addWidget(self.remove_button)
        remove_row.addStretch(1)
        details.addLayout(remove_row)
        layout.addWidget(self.details)

        self.add_button.clicked.connect(self._on_add)
        self.list.currentItemChanged.connect(lambda *_: self.update_selected())
        self.remove_button.clicked.connect(self._on_remove)
        self.refresh()

    # --- what the panel shows -------------------------------------------------

    @property
    def experiment(self) -> Optional["VolumeExperiment"]:
        return self.session.experiment

    @property
    def selected(self) -> Optional[VolumePosition]:
        item = self.list.currentItem()
        experiment = self.experiment
        if item is None or experiment is None:
            return None
        return experiment.get_position_by_name(item.data(Qt.UserRole))

    def refresh(self) -> None:
        self.update_items()

    def _row(self, experiment: "VolumeExperiment", position: VolumePosition) -> QWidget:
        """A position's row: its name and slices, Move to and Update."""
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(6, 2, 6, 2)
        name = QLabel(position.name)
        progress = QLabel(progress_text(experiment, position))
        move_button = QPushButton("Move to")
        update_button = QPushButton("Update")
        move_button.setToolTip(f"Move to {position.name}'s milling pose.")
        update_button.setToolTip(
            f"Set {position.name}'s milling pose from where the microscope is now."
        )
        for button in (move_button, update_button):
            button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        for label in (name, progress):
            label.setStyleSheet("background: transparent;")
        name_ = position.name
        move_button.clicked.connect(lambda: self._on_move_to(MILLING_POSE, name_))
        update_button.clicked.connect(lambda: self._on_set_pose(MILLING_POSE, name_))
        line.addWidget(name, 1)
        line.addWidget(progress)
        line.addWidget(move_button)
        line.addWidget(update_button)
        return row

    def update_items(self, latest: bool = False) -> None:
        """Redraw the list, keeping the selection; *latest* selects the last."""
        experiment = self.experiment
        current = self.selected
        keep = current.name if current is not None else None
        self.list.blockSignals(True)
        self.list.clear()
        positions = experiment.positions if experiment is not None else []
        for position in positions:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, position.name)
            row = self._row(experiment, position)
            item.setSizeHint(row.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
        row = next(
            (i for i, p in enumerate(positions) if p.name == keep),
            0 if positions else -1,
        )
        if latest and positions:
            row = len(positions) - 1
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.add_button.setEnabled(experiment is not None)
        self.update_selected()
        self.items_changed.emit()

    def update_selected(self) -> None:
        position = self.selected
        self.details.setEnabled(position is not None)
        self.poses_title.setText(f"{position.name} · poses" if position else "Poses")
        for pose_name, label in self.pose_labels.items():
            label.setText(pose_text(position, pose_name) if position else "")

    # --- edits ----------------------------------------------------------------

    def _not_busy(self, name: str, what: str) -> bool:
        return not refused(self.session.runner.busy_refusal([name], what))

    def _position(self, name: Optional[str]) -> Optional[VolumePosition]:
        if name is None:
            return self.selected
        return self.experiment.get_position_by_name(name)

    def _on_add(self) -> None:
        experiment, microscope = self.experiment, self.session.microscope
        if experiment is None:
            return
        if microscope is None:
            refused("Connect to the microscope to add a position where it is.")
            return
        if refused_during_run(self.session.runner, "add a position where the stage is"):
            return
        try:
            experiment.add_position(
                experiment.next_position_name(), microscope.get_microscope_state()
            )
        except ValueError as exc:
            refused(str(exc))
            return
        self.update_items(latest=True)

    def _on_set_pose(self, pose_name: str, name: Optional[str] = None) -> None:
        position, microscope = self._position(name), self.session.microscope
        if position is None:
            return
        if microscope is None:
            refused("Connect to the microscope to set a pose.")
            return
        if self._not_busy(position.name, "set its pose"):
            refused(set_pose_now(self.experiment, microscope, position.name, pose_name))
        self.update_selected()

    def _on_move_to(self, pose_name: str, name: Optional[str] = None) -> None:
        position = self._position(name)
        movement = self._panel.movement_widget
        if position is None or movement is None:
            refused("Connect to the microscope to move to a pose.")
            return
        stage_position, why = pose_stage_position(position, pose_name)
        if refused(why):
            return
        reply = QMessageBox.question(
            self,
            "Move to Pose",
            f"Move to {position.name}'s {pose_name} pose?\n{stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            movement.move_to_position(stage_position)

    def _on_remove(self) -> None:
        position = self.selected
        if position is None:
            return
        reply = QMessageBox.question(
            self,
            "Remove Position",
            f"Remove {position.name}? Its folder stays on disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self._not_busy(position.name, "remove it"):
            refused(self.experiment.remove_position(position.name))
        self.update_items()
