import logging
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import numpy as np
from napari.qt.threading import thread_worker
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui import notification_service
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from superqt.utils import qdebounced
from fibsem.ui.utils import message_box_ui
from fibsem.ui.napari.utilities import update_text_overlay
from fibsem.ui.widgets.custom_widgets import (
    ValueSpinBox,
)

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

OBJECTIVE_CONFIG = {
    "position": {
        "step_size": 1.0,  # µm
        "decimals": 1,  # number of decimal places
        "suffix": " µm",  # unit suffix
        "tooltip": "Objective position in microns relative to current position",
    },
    "step_size_control": {
        "range": (0.1, 250.0),  # µm
        "step": 0.1,  # µm
        "default": 1.0,  # µm
        "decimals": 1,  # number of decimal places
        "suffix": " µm",  # unit suffix
        "tooltip": "Step size for objective movement in microns",
    },
}
MAX_OBJECTIVE_STEP_SIZE_UM = 100.0  # µm
_SAFE_TILT_TOLERANCE_DEG = 0.5


class _InsertObjectiveDialog(QDialog):
    """Combined stage-tilt selection + insert confirmation dialog."""

    def __init__(self, tilt_deg: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert Objective")
        self.setMinimumWidth(480)
        self.selected_tilt: Optional[float] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel(
            f"Current stage tilt: <b>{tilt_deg:.1f}°</b><br><br>"
            "Select the target stage tilt. The stage will move to the selected "
            "tilt if needed before inserting the objective."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._btn_0 = QPushButton("Insert at 0°")
        self._btn_180 = QPushButton("Insert at -180°")
        btn_cancel = QPushButton("Cancel")
        self._btn_0.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self._btn_180.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        btn_row.addWidget(self._btn_0)
        btn_row.addWidget(self._btn_180)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._btn_0.clicked.connect(lambda: self._accept(0.0))
        self._btn_180.clicked.connect(lambda: self._accept(-180.0))
        btn_cancel.clicked.connect(self.reject)

    def _accept(self, tilt: float) -> None:
        self.selected_tilt = tilt
        self.accept()


class ObjectiveControlWidget(QWidget):
    def __init__(self, fm: FluorescenceMicroscope, parent: Optional['FMAcquisitionWidget'] = None, microscope=None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent
        # microscope is required for insert (stage tilt read/move); fall back to
        # the parent widget's microscope when embedded in a full acquisition UI.
        self.microscope = microscope
        if self.microscope is None and parent is not None:
            self.microscope = getattr(parent, "microscope", None)
        self.initUI()

    def initUI(self):
        self.setContentsMargins(0, 0, 0, 0)
        self.pushButton_insert_objective = QPushButton("Insert Objective", self)
        self.pushButton_retract_objective = QPushButton("Retract Objective", self)
        self.pushButton_refresh_position = QPushButton("Refresh Position Data", self)
        self.pushButton_refresh_position.setToolTip("Refresh the objective position data from the microscope")
        self.pushButton_refresh_position.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_refresh_position.setVisible(False)

        # add focus position controls
        self.label_focus_position = QLabel("Focus Position", self)
        self.doubleSpinBox_focus_position = ValueSpinBox(parent=self)
        self.doubleSpinBox_focus_position.setRange(self.fm.objective.limits[0] * METRE_TO_MICRON,
                                                   self.fm.objective.limits[1] * METRE_TO_MICRON)
        # Set initial value from objective's focus position if available
        if self.fm.objective.focus_position is not None:
            self.doubleSpinBox_focus_position.setValue(self.fm.objective.focus_position * METRE_TO_MICRON)
        else:
            self.doubleSpinBox_focus_position.setValue(0.0)
        self.doubleSpinBox_focus_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_focus_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_focus_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_focus_position.setToolTip("Focus position in microns - set and move to autofocus position")
        self.doubleSpinBox_focus_position.setKeyboardTracking(False)
        self.pushButton_move_to_focus = QPushButton("Move to Focus Position", self)
        self.pushButton_move_to_focus.setToolTip("Move objective to stored focus position")
        self.pushButton_set_focus_position = QPushButton("Set Focus Position", self)
        self.pushButton_set_focus_position.setToolTip("Set current objective position as focus position")

        # add double spin box for objective position
        self.label_objective_control = QLabel("Current Position", self)
        self.label_objective_step_size = QLabel("Step Size", self)
        self.doubleSpinBox_objective_position = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MICRON,
                                                        self.fm.objective.limits[1] * METRE_TO_MICRON)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MICRON)  # Convert m to µm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_position.setToolTip(OBJECTIVE_CONFIG["position"]["tooltip"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        self.doubleSpinBox_objective_step_size = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_step_size.setRange(*OBJECTIVE_CONFIG["step_size_control"]["range"])
        self.doubleSpinBox_objective_step_size.setSingleStep(OBJECTIVE_CONFIG["step_size_control"]["step"])
        self.doubleSpinBox_objective_step_size.setValue(OBJECTIVE_CONFIG["step_size_control"]["default"])
        self.doubleSpinBox_objective_step_size.setDecimals(OBJECTIVE_CONFIG["step_size_control"]["decimals"])
        self.doubleSpinBox_objective_step_size.setSuffix(OBJECTIVE_CONFIG["step_size_control"]["suffix"])
        self.doubleSpinBox_objective_step_size.setToolTip(OBJECTIVE_CONFIG["step_size_control"]["tooltip"])
        self.doubleSpinBox_objective_step_size.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # add user-defined limit controls
        self.label_objective_limit = QLabel("Limit Position", self)
        self.doubleSpinBox_objective_limit = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_limit.setRange(0.0, self.fm.objective.limits[1] * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.setValue(self.fm.objective.limit_position * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_limit.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_limit.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_limit.setToolTip("User-defined upper limit for objective position in microns")
        self.doubleSpinBox_objective_limit.setKeyboardTracking(False)

        # instructions label
        self.label_instructions = QLabel("Instructions: Use Shift + Mouse Wheel to move objective by the step size", self)
        self.label_instructions.setStyleSheet("font-style: italic; color: gray;")

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.pushButton_insert_objective, 0, 0)
        layout.addWidget(self.pushButton_retract_objective, 0, 1)
        layout.addWidget(self.label_objective_limit, 1, 0)
        layout.addWidget(self.doubleSpinBox_objective_limit, 1, 1)
        layout.addWidget(self.label_focus_position, 2, 0)
        layout.addWidget(self.doubleSpinBox_focus_position, 2, 1)
        layout.addWidget(self.pushButton_set_focus_position, 3, 0, 1, 1)
        layout.addWidget(self.pushButton_move_to_focus, 3, 1, 1, 1)
        layout.addWidget(self.label_objective_control, 4, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 4, 1)
        layout.addWidget(self.label_objective_step_size, 5, 0)
        layout.addWidget(self.doubleSpinBox_objective_step_size, 5, 1)
        layout.addWidget(self.pushButton_refresh_position, 6, 0, 1, 2)
        layout.addWidget(self.label_instructions, 7, 0, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_focus_position.valueChanged.connect(self.on_focus_position_changed)
        self.pushButton_move_to_focus.clicked.connect(self.move_to_focus_position)
        self.pushButton_set_focus_position.clicked.connect(self.set_focus_position)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value))
        self.doubleSpinBox_objective_limit.valueChanged.connect(self.on_objective_limit_changed)
        self.pushButton_refresh_position.clicked.connect(lambda: self.update_objective_position_labels(None))

        # set stylesheets
        self.pushButton_insert_objective.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_retract_objective.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_focus.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_set_focus_position.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        # set initial ranges based on user-defined limit
        self.on_objective_limit_changed(self.doubleSpinBox_objective_limit.value())

        # Debounce state for mouse wheel objective movement
        self._wheel_target_um: float = 0.0
        self._execute_wheel_move = qdebounced(self._execute_wheel_move_impl, timeout=150)

        if self.parent_widget is not None and hasattr(self.parent_widget, "viewer"):
            self.parent_widget.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

    def _set_objective_actions_enabled(self, enabled: bool) -> None:
        """Enable/disable insert & retract buttons while a threaded move runs."""
        self.pushButton_insert_objective.setEnabled(enabled)
        self.pushButton_retract_objective.setEnabled(enabled)

    def _on_objective_action_finished(self, *_) -> None:
        """Re-enable controls and refresh labels after a threaded objective move."""
        self._set_objective_actions_enabled(True)
        self.update_objective_position_labels()

    def _on_objective_action_error(self, exc: Exception) -> None:
        """Report a failed objective operation without crashing the app."""
        logging.error(f"Objective operation failed: {exc}", exc_info=exc)
        self._set_objective_actions_enabled(True)
        self.update_objective_position_labels()
        message_box_ui(
            title="Objective Error",
            text=f"The objective operation failed:\n\n{exc}",
            buttons=QMessageBox.Ok,
            parent=self,
        )

    def insert_objective(self):
        """Insert the objective. The hardware move runs on a worker thread."""
        if self.fm.objective.state == "Inserted":
            message_box_ui(
                title="Objective Already Inserted",
                text="The objective is already inserted.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        if self.microscope is None:
            message_box_ui(
                title="Objective Error",
                text="Cannot insert objective: no microscope connection available.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        stage_pos = self.microscope.get_stage_position()
        tilt_deg = np.degrees(stage_pos.t)

        dlg = _InsertObjectiveDialog(tilt_deg=tilt_deg, parent=self)
        if dlg.exec_() != QDialog.Accepted or dlg.selected_tilt is None:
            logging.info("Objective insertion cancelled by user.")
            return

        target_tilt_deg = dlg.selected_tilt
        target_stage_pos = None
        if not np.isclose(tilt_deg, target_tilt_deg, atol=_SAFE_TILT_TOLERANCE_DEG):
            stage_pos.t = np.radians(target_tilt_deg)
            target_stage_pos = stage_pos

        self._set_objective_actions_enabled(False)
        worker = self._insert_objective_worker(target_stage_pos)
        worker.returned.connect(self._on_objective_action_finished)
        worker.errored.connect(self._on_objective_action_error)
        worker.start()

    @thread_worker
    def _insert_objective_worker(self, target_stage_pos) -> None:
        """Worker: move stage to the insert tilt (if needed), then insert."""
        if target_stage_pos is not None:
            logging.info(
                f"Moving stage to {np.degrees(target_stage_pos.t):.0f}° tilt "
                "before inserting objective."
            )
            self.microscope.move_stage_absolute(target_stage_pos)
        logging.info("Inserting objective...")
        self.fm.objective.insert()
        logging.info("Objective inserted.")

    def retract_objective(self):
        """Retract the objective. The hardware move runs on a worker thread."""
        if self.fm.objective.state == "Retracted":
            message_box_ui(
                title="Objective Already Retracted",
                text="The objective is already retracted.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        # confirmation dialog
        ret = message_box_ui(
            title="Retract Objective",
            text="Are you sure you want to retract the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective retraction cancelled by user.")
            return

        self._set_objective_actions_enabled(False)
        worker = self._retract_objective_worker()
        worker.returned.connect(self._on_objective_action_finished)
        worker.errored.connect(self._on_objective_action_error)
        worker.start()

    @thread_worker
    def _retract_objective_worker(self) -> None:
        """Worker: retract the objective."""
        logging.info("Retracting objective...")
        self.fm.objective.retract()
        logging.info("Objective retracted.")

    def update_objective_position_labels(self, objective_position: Optional[float] = None):
        """Update the objective position input and label."""
        if objective_position is None:
            objective_position = self.fm.objective.position
        self.doubleSpinBox_objective_position.blockSignals(True)  # Block signals to prevent recursion
        self.doubleSpinBox_objective_position.setValue(objective_position * METRE_TO_MICRON)  # Convert m to µm
        self.doubleSpinBox_objective_position.blockSignals(False)  # Unblock signals

        if self.parent_widget is not None and hasattr(self.parent_widget, "viewer"):
            update_text_overlay(self.parent_widget.viewer,
                                self.parent_widget.microscope,
                                objective_position=objective_position)  # TODO: migrate to objective_position_changed signal

    @pyqtSlot(float)
    def on_objective_position_changed(self, position: float):
        """Handle changes to the objective position."""

        is_large_change = abs(self.fm.objective.position - (position * MICRON_TO_METRE)) > 1e-3  # 1 mm threshold

        if is_large_change:
            logging.info(f"Large change in objective position requested: {position:.1f} µm")

            ret = message_box_ui(
                title="Large Objective Movement",
                text=f"Are you sure you want to move the objective position to {position:.1f} µm?",
                parent=self
            )

            if ret is False:
                logging.info("Objective position move cancelled by user.")
                # Reset the spin box to the current position
                self.update_objective_position_labels()
                return

        logging.info(f"Changing objective position to: {position:.1f} µm")
        try:
            self.fm.objective.move_absolute(position * MICRON_TO_METRE)
            logging.info(f"Objective moved to position: {position:.1f} µm")
        except Exception as e:
            logging.error(f"Failed to move objective: {e}", exc_info=e)
            message_box_ui(
                title="Objective Error",
                text=f"Failed to move the objective:\n\n{e}",
                buttons=QMessageBox.Ok,
                parent=self,
            )

        # Update the objective position label (re-syncs to the actual position)
        self.update_objective_position_labels()

    @pyqtSlot(float)
    def on_objective_limit_changed(self, limit_um: float):
        """Handle changes to the user-defined objective limit."""
        limit_m = limit_um * MICRON_TO_METRE
        logging.info(f"Updating objective user limit to: {limit_um:.1f} µm")
        self.fm.objective.limit_position = limit_m
        max_limit_um = min(limit_um, self.fm.objective.limits[1] * METRE_TO_MICRON)
        min_limit_um = self.fm.objective.limits[0] * METRE_TO_MICRON
        self.doubleSpinBox_objective_position.setRange(min_limit_um, max_limit_um)
        self.doubleSpinBox_focus_position.setRange(min_limit_um, max_limit_um)

    @pyqtSlot(float)
    def on_focus_position_changed(self, position: float):
        """Handle changes to the focus position setting."""
        logging.info(f"Focus position updated to: {position:.1f} µm")
        self.fm.objective.focus_position = position * MICRON_TO_METRE  # Convert µm to m

    def move_to_focus_position(self):
        """Move the objective to the stored focus position."""
        if self.fm.objective.focus_position is None:
            logging.warning("No focus position has been set")
            return

        focus_position_um = self.fm.objective.focus_position * METRE_TO_MICRON

        # confirmation dialog for large movements
        current_position_um = self.fm.objective.position * METRE_TO_MICRON
        is_large_change = abs(current_position_um - focus_position_um) > 1000.0  # 1 mm threshold

        if is_large_change:
            ret = message_box_ui(
                title="Move to Focus Position",
                text=f"Move objective to focus position {focus_position_um:.1f} µm?",
                parent=self
            )
            if ret is False:
                logging.info("Move to focus position cancelled by user.")
                return

        logging.info(f"Moving objective to focus position: {focus_position_um:.1f} µm")
        try:
            self.fm.objective.move_absolute(self.fm.objective.focus_position)
            logging.info(f"Objective moved to focus position: {focus_position_um:.1f} µm")
        except Exception as e:
            logging.error(f"Failed to move objective to focus position: {e}", exc_info=e)
            message_box_ui(
                title="Objective Error",
                text=f"Failed to move the objective:\n\n{e}",
                buttons=QMessageBox.Ok,
                parent=self,
            )

        # Update the objective position label
        self.update_objective_position_labels()

    def set_focus_position(self):
        """Set the focus position to the current objective position."""
        current_position_um = self.fm.objective.position * METRE_TO_MICRON

        ret = message_box_ui(
            title="Set Focus Position",
            text=f"Set focus position to {current_position_um:.1f} µm?",
            parent=self,
        )

        if ret is False:
            logging.info("Set focus position cancelled by user.")
            return

        # set microscope objective focus position
        logging.info(f"Setting focus position to {current_position_um:.1f} µm")
        self.doubleSpinBox_focus_position.setValue(current_position_um)
        self.fm.objective.focus_position = self.fm.objective.position

    def _set_focus_position(self, position: float):
        """Set the focus position programmatically.
        Args:
            position (float): Focus position in meters.
        """
        logging.info(f"Programmatically setting focus position to: {position * METRE_TO_MICRON:.1f} µm")
        self.doubleSpinBox_focus_position.blockSignals(True)
        self.doubleSpinBox_focus_position.setValue(position * METRE_TO_MICRON)
        self.doubleSpinBox_focus_position.blockSignals(False)
        self.fm.objective.focus_position = position

    def _set_limit_position(self, position: float):
        """Set the user-defined limit position programmatically.
        Args:
            position (float): Limit position in meters.
        """
        logging.info(f"Programmatically setting limit position to: {position * METRE_TO_MICRON:.1f} µm")
        self.doubleSpinBox_objective_limit.blockSignals(True)
        self.doubleSpinBox_objective_limit.setValue(position * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.blockSignals(False)
        self.fm.objective.limit_position = position

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events in the napari viewer.

        This method avoids all blocking microscope calls. Position is read from
        the spinbox (already synced), and the actual hardware move is debounced
        via _execute_wheel_move so rapid scroll events coalesce into a single move.
        """

        if self.parent_widget is None:
            event.handled = False
            return

        if 'Shift' not in event.modifiers:
            event.handled = False  # Let napari handle zooming if Shift is not pressed
            return

        # Prevent objective movement during acquisitions
        if self.parent_widget.is_acquisition_active:
            logging.info("Objective movement disabled during acquisition")
            event.handled = True
            return

        # Calculate step size based on wheel delta
        objective_step_size = self.doubleSpinBox_objective_step_size.value()  # µm
        step_um = np.clip(objective_step_size * event.delta[1], -MAX_OBJECTIVE_STEP_SIZE_UM, MAX_OBJECTIVE_STEP_SIZE_UM)

        # Read current position from the spinbox (no hardware call)
        current_pos_um = self.doubleSpinBox_objective_position.value()
        new_pos_um = current_pos_um + step_um

        logging.info(f"Mouse wheel: step={step_um:.1f} µm, target={new_pos_um:.1f} µm")

        # Update spinbox immediately for visual feedback (no hardware call)
        self.doubleSpinBox_objective_position.blockSignals(True)
        self.doubleSpinBox_objective_position.setValue(new_pos_um)
        self.doubleSpinBox_objective_position.blockSignals(False)

        # Store target and trigger debounced hardware move
        self._wheel_target_um = new_pos_um
        self._execute_wheel_move()

        event.handled = True

    def _execute_wheel_move_impl(self):
        """Execute the actual hardware move after scrolling settles."""
        position_um = self._wheel_target_um
        position_m = position_um * MICRON_TO_METRE
        logging.info(f"Executing debounced wheel move to: {position_um:.1f} µm")
        try:
            self.fm.objective.move_absolute(position_m)
            self.update_objective_position_labels(objective_position=position_m)
        except Exception as e:
            logging.error(f"Objective wheel move failed: {e}", exc_info=e)
            notification_service.show_toast(f"Objective move failed: {e}", "warning")
            self.update_objective_position_labels()  # re-sync to actual position