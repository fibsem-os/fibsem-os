"""The volume window's Positions panel, the first tab beside the quad view: the
positions, "Add position here", and the selected one's name, slices, poses,
point of interest and alignment area.

Every edit is the experiment's own method, the one the App Server's position
edits call, with its refusal said on a toast. The panel redraws when the
experiment's positions change, whoever changed them.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.volume.position import SEM_POSE, VolumePosition
from fibsem.constants import RADIANS_TO_DEGREES
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.widgets.canvas.overlays.poi_marker import PoiMarker
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.ui.widgets.microscope_panel import MicroscopePanel
    from fibsem.workflows.session import AppSession

__all__ = ["PositionsPanel"]

POSES = (MILLING_POSE, SEM_POSE)
PICK_POI = "Pick on FIB Image"
DRAW_AREA = "Draw on FIB Image"


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
        view_controller: Callable[[], Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self._panel = panel
        self.poi_marker = PoiMarker(
            image_widget=lambda: panel.image_widget,
            view_controller=view_controller,
        )

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
        form = QFormLayout(self.details)
        form.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel()
        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(1, 100000)
        form.addRow("Name", self.name_label)
        form.addRow("Slices", self.slices_spin)

        self.pose_combo = QComboBox()
        self.pose_combo.addItems(POSES)
        self.pose_label = QLabel()
        pose_row = QHBoxLayout()
        pose_row.addWidget(self.pose_combo)
        pose_row.addWidget(self.pose_label, 1)
        form.addRow("Pose", pose_row)
        self.set_pose_button = QPushButton("Set from current")
        self.move_to_button = QPushButton("Move to")
        for button in (self.set_pose_button, self.move_to_button):
            button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        buttons = QHBoxLayout()
        buttons.addWidget(self.set_pose_button)
        buttons.addWidget(self.move_to_button)
        form.addRow("", buttons)

        self.poi_label = QLabel()
        self.poi_button = QPushButton(PICK_POI)
        form.addRow("Point of interest", self._row(self.poi_label, self.poi_button))
        self.area_label = QLabel()
        self.area_button = QPushButton(DRAW_AREA)
        form.addRow("Alignment area", self._row(self.area_label, self.area_button))
        for button in (self.poi_button, self.area_button):
            button.setCheckable(True)
            button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet(stylesheets.DANGER_BUTTON_STYLESHEET)
        remove_row = QHBoxLayout()
        remove_row.addWidget(self.remove_button)
        remove_row.addStretch(1)
        form.addRow("", remove_row)
        layout.addWidget(self.details)

        self.add_button.clicked.connect(self._on_add)
        self.list.currentItemChanged.connect(lambda *_: self.update_selected())
        self.slices_spin.editingFinished.connect(self._on_slices)
        self.pose_combo.currentIndexChanged.connect(lambda _: self._show_pose())
        self.set_pose_button.clicked.connect(self._on_set_pose)
        self.move_to_button.clicked.connect(self._on_move_to)
        self.poi_button.toggled.connect(self._on_poi_toggled)
        self.area_button.toggled.connect(self._on_area_toggled)
        self.remove_button.clicked.connect(self._on_remove)
        self.refresh()

    @staticmethod
    def _row(label: QLabel, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(label, 1)
        row.addWidget(button)
        return row

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

    def update_items(self, latest: bool = False) -> None:
        """Redraw the list, keeping the selection; *latest* selects the last."""
        experiment = self.experiment
        current = self.selected
        keep = current.name if current is not None else None
        self.list.blockSignals(True)
        self.list.clear()
        positions = experiment.positions if experiment is not None else []
        for position in positions:
            item = QListWidgetItem(
                f"{position.name}\t{progress_text(experiment, position)}"
            )
            item.setData(Qt.UserRole, position.name)
            self.list.addItem(item)
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
        if position is None:
            self.name_label.clear()
            self.poi_label.clear()
            self.area_label.clear()
            self.pose_label.clear()
            return
        self.name_label.setText(position.name)
        self.slices_spin.blockSignals(True)
        self.slices_spin.setValue(position.iterations)
        self.slices_spin.blockSignals(False)
        self._show_pose()
        poi = position.poi
        self.poi_label.setText(f"x {poi.x * 1e6:.2f} µm · y {poi.y * 1e6:.2f} µm")
        area = position.alignment_area
        self.area_label.setText(
            f"{area.left:.2f}, {area.top:.2f} · {area.width:.2f} × {area.height:.2f}"
        )

    def _show_pose(self) -> None:
        position = self.selected
        if position is not None:
            self.pose_label.setText(pose_text(position, self.pose_combo.currentText()))

    # --- edits ----------------------------------------------------------------

    def _said(self, why: str) -> bool:
        """Say a refusal; True when there was none."""
        if why:
            notification_service.show_toast(why, "warning")
        return not why

    def _not_busy(self, name: str, what: str) -> bool:
        return self._said(self.session.runner.busy_refusal([name], what))

    def _on_add(self) -> None:
        experiment, microscope = self.experiment, self.session.microscope
        if experiment is None:
            return
        if microscope is None:
            self._said("Connect to the microscope to add a position where it is.")
            return
        if self.session.runner.is_running:
            self._said("A workflow is running; add a position when it has stopped.")
            return
        try:
            experiment.add_position(
                experiment.next_position_name(), microscope.get_microscope_state()
            )
        except ValueError as exc:
            self._said(str(exc))
            return
        self.update_items(latest=True)

    def _on_slices(self) -> None:
        position = self.selected
        if position is None or self.slices_spin.value() == position.iterations:
            return
        if self._not_busy(position.name, "change its slices"):
            self._said(
                self.experiment.patch_position(
                    position.name, iterations=self.slices_spin.value()
                )
            )
        self.update_items()

    def _on_set_pose(self) -> None:
        position, microscope = self.selected, self.session.microscope
        if position is None:
            return
        if microscope is None:
            self._said("Connect to the microscope to set a pose.")
            return
        pose_name = self.pose_combo.currentText()
        if self._not_busy(position.name, "set its pose"):
            self._said(
                self.experiment.set_pose(
                    position.name, pose_name, microscope.get_microscope_state()
                )
            )
        self._show_pose()

    def _on_move_to(self) -> None:
        position = self.selected
        movement = self._panel.movement_widget
        if position is None or movement is None:
            self._said("Connect to the microscope to move to a pose.")
            return
        pose_name = self.pose_combo.currentText()
        if MILLING_POSE not in position.poses:
            self._said(f"{position.name} has no milling pose.")
            return
        stage_position = position.pose(pose_name).stage_position
        reply = QMessageBox.question(
            self,
            "Move to Pose",
            f"Move to {position.name}'s {pose_name} pose?\n{stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            movement.move_to_position(stage_position)

    def _on_poi_toggled(self, picking: bool) -> None:
        """Pressed: the marker goes up on the FIB Image. Released: where it was
        dragged to is the point of interest."""
        position = self.selected
        if picking:
            if position is None or not self.poi_marker.show(position.poi):
                self._said("Take a FIB Image to pick the point of interest on.")
                self._release(self.poi_button)
                return
            self.poi_button.setText("Done")
            return
        self.poi_button.setText(PICK_POI)
        poi = self.poi_marker.read()
        self.poi_marker.remove()
        if position is not None and poi is not None:
            if self._not_busy(position.name, "move its point of interest"):
                self._said(self.experiment.patch_position(position.name, poi=poi))
        self.update_selected()

    def _on_area_toggled(self, drawing: bool) -> None:
        """Pressed: the area is drawn on the FIB Image to drag. Released: the
        dragged area is the position's."""
        position = self.selected
        image_widget = self._panel.image_widget
        if drawing:
            if position is None or image_widget is None:
                self._said("Connect to the microscope to draw the alignment area.")
                self._release(self.area_button)
                return
            image_widget.toggle_alignment_area(position.alignment_area)
            self.area_button.setText("Done")
            return
        self.area_button.setText(DRAW_AREA)
        if image_widget is None:
            return
        area = image_widget.get_alignment_area()
        image_widget.clear_alignment_area()
        if position is not None and area is not None:
            if self._not_busy(position.name, "move its alignment area"):
                self._said(
                    self.experiment.patch_position(position.name, alignment_area=area)
                )
        self.update_selected()

    @staticmethod
    def _release(button: QPushButton) -> None:
        button.blockSignals(True)
        button.setChecked(False)
        button.blockSignals(False)

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
            self._said(self.experiment.remove_position(position.name))
        self.update_items()
