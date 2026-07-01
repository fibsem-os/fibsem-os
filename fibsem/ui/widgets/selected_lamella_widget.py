from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Lamella
from fibsem.constants import METRE_TO_MICRON
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueSpinBox,
)
from fibsem.ui.widgets.lamella_pose_list_widget import LamellaPoseListWidget


class SelectedLamellaWidget(QWidget):
    """Presentational panel for the currently selected lamella.

    Shows the objective-position control and the lamella's poses. This is a
    view: it reads display state from the lamella passed to :meth:`set_lamella`
    and emits signals for user actions. All microscope/experiment side effects
    are handled by the parent (AutoLamellaUI) via the connected signals.
    """

    objective_position_changed = pyqtSignal(float)     # value in µm
    use_current_objective_requested = pyqtSignal()
    apply_objective_to_all_requested = pyqtSignal()
    pose_update_requested = pyqtSignal(str)            # pose name
    pose_move_to_requested = pyqtSignal(str)           # pose name

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- objective position row ---
        self.label_objective_position = QLabel("Objective Position")
        self.spinbox_objective_position = ValueSpinBox(
            suffix="µm", decimals=1, step=1.0, minimum=-20000.0, maximum=20000.0
        )
        self.btn_objective_actions = IconToolButton(
            "mdi:dots-horizontal", tooltip="Actions"
        )
        self.btn_objective_actions.setPopupMode(IconToolButton.InstantPopup)
        self.btn_objective_actions.setStyleSheet(
            "QToolButton::menu-indicator { image: none; }"
        )
        obj_menu = QMenu(self)
        self._action_use_current_obj_pos = obj_menu.addAction(
            "Use Current Objective Position"
        )
        self._action_apply_obj_to_all = obj_menu.addAction("Apply to All Lamella")
        self.btn_objective_actions.setMenu(obj_menu)

        # --- pose list ---
        self.pose_list = LamellaPoseListWidget()

        # --- layout ---
        content = QWidget()
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        obj_row = QWidget()
        obj_row_layout = QHBoxLayout(obj_row)
        obj_row_layout.setContentsMargins(0, 0, 0, 0)
        obj_row_layout.setSpacing(2)
        obj_row_layout.addWidget(self.spinbox_objective_position)
        obj_row_layout.addWidget(self.btn_objective_actions)
        content_layout.addWidget(self.label_objective_position, 0, 0)
        content_layout.addWidget(obj_row, 0, 1)
        content_layout.addWidget(self.pose_list, 1, 0, 1, 2)

        self._panel = TitledPanel(
            "Selected Lamella", content=content, collapsible=False
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._panel)

        # --- wire internal widgets to public signals ---
        self.spinbox_objective_position.valueChanged.connect(
            self.objective_position_changed
        )
        self._action_use_current_obj_pos.triggered.connect(
            self.use_current_objective_requested
        )
        self._action_apply_obj_to_all.triggered.connect(
            self.apply_objective_to_all_requested
        )
        self.pose_list.update_requested.connect(self.pose_update_requested)
        self.pose_list.move_to_requested.connect(self.pose_move_to_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Refresh the panel display from *lamella* (or clear it if None)."""
        if lamella is None:
            self.label_objective_position.setVisible(False)
            self.spinbox_objective_position.setVisible(False)
            self.btn_objective_actions.setVisible(False)
            self.pose_list.set_lamella(None)
            self.pose_list.setVisible(False)
            return

        # objective position (shown in µm). The controls are shown whenever the
        # lamella has a fluorescence pose, so the objective can still be set/restored
        # (e.g. via "Use Current Objective Position") even when the position is unset.
        # If it were gated on obj_pos being set, a wiped (None) objective would hide
        # the only UI to recover it.
        has_fluorescence_pose = lamella.fluorescence_pose is not None
        obj_pos = (
            lamella.fluorescence_pose.objective_position
            if has_fluorescence_pose
            else None
        )
        self.set_objective_value_um(obj_pos * METRE_TO_MICRON if obj_pos is not None else 0.0)
        self.label_objective_position.setVisible(has_fluorescence_pose)
        self.spinbox_objective_position.setVisible(has_fluorescence_pose)
        self.btn_objective_actions.setVisible(has_fluorescence_pose)
        # "Apply to All" copies the current value to every lamella; disable it when the
        # objective is unset so the 0.0 placeholder cannot clobber other lamellae's focus.
        # "Use Current Objective Position" stays enabled as the recovery path.
        self._action_apply_obj_to_all.setEnabled(obj_pos is not None)

        # poses
        self.pose_list.set_lamella(lamella)
        self.pose_list.setVisible(bool(lamella.poses))

    def refresh_pose(self, pose_name: str, pretty: str) -> None:
        """Update one pose row's position in place, without rebuilding the list."""
        self.pose_list.refresh_pose(pose_name, pretty)

    def objective_value_um(self) -> float:
        """Current objective spinbox value, in µm."""
        return self.spinbox_objective_position.value()

    def set_objective_value_um(self, um: float) -> None:
        """Set the objective spinbox value (µm) without emitting a change."""
        self.spinbox_objective_position.blockSignals(True)
        self.spinbox_objective_position.setValue(um)
        self.spinbox_objective_position.blockSignals(False)
