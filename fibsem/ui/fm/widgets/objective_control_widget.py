import logging
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)
import numpy as np
from fibsem.constants import METRE_TO_MILLIMETRE, MILLIMETRE_TO_METRE
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from superqt.utils import qdebounced
from fibsem.ui.utils import message_box_ui
from fibsem.ui.napari.utilities import update_text_overlay

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

OBJECTIVE_CONFIG = {
    "position": {
        "step_size": 0.001,  # mm
        "decimals": 4,  # number of decimal places
        "suffix": " mm",  # unit suffix
        "tooltip": "Objective position in millimeters relative to current position",
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
MAX_OBJECTIVE_STEP_SIZE_MM = 0.1  # mm

class ObjectiveControlWidget(QWidget):    
    def __init__(self, fm: FluorescenceMicroscope, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent
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
        self.doubleSpinBox_focus_position = QDoubleSpinBox(self)
        self.doubleSpinBox_focus_position.setRange(self.fm.objective.limits[0] * METRE_TO_MILLIMETRE,
                                                   self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        # Set initial value from objective's focus position if available
        if self.fm.objective.focus_position is not None:
            self.doubleSpinBox_focus_position.setValue(self.fm.objective.focus_position * METRE_TO_MILLIMETRE)
        else:
            self.doubleSpinBox_focus_position.setValue(0.0)
        self.doubleSpinBox_focus_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_focus_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_focus_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_focus_position.setToolTip("Focus position in millimeters - set and move to autofocus position")
        self.doubleSpinBox_focus_position.setKeyboardTracking(False)
        self.pushButton_move_to_focus = QPushButton("Move to Focus Position", self)
        self.pushButton_move_to_focus.setToolTip("Move objective to stored focus position")
        self.pushButton_set_focus_position = QPushButton("Set Focus Position", self)
        self.pushButton_set_focus_position.setToolTip("Set current objective position as focus position")

        # add double spin box for objective position
        self.label_objective_control = QLabel("Current Position", self)
        self.label_objective_step_size = QLabel("Step Size", self)
        self.doubleSpinBox_objective_position = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MILLIMETRE,
                                                        self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MILLIMETRE)  # Convert m to mm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_position.setToolTip(OBJECTIVE_CONFIG["position"]["tooltip"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        self.doubleSpinBox_objective_step_size = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_step_size.setRange(*OBJECTIVE_CONFIG["step_size_control"]["range"])
        self.doubleSpinBox_objective_step_size.setSingleStep(OBJECTIVE_CONFIG["step_size_control"]["step"])
        self.doubleSpinBox_objective_step_size.setValue(OBJECTIVE_CONFIG["step_size_control"]["default"])
        self.doubleSpinBox_objective_step_size.setDecimals(OBJECTIVE_CONFIG["step_size_control"]["decimals"])
        self.doubleSpinBox_objective_step_size.setSuffix(OBJECTIVE_CONFIG["step_size_control"]["suffix"])
        self.doubleSpinBox_objective_step_size.setToolTip(OBJECTIVE_CONFIG["step_size_control"]["tooltip"])
        self.doubleSpinBox_objective_step_size.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # add user-defined limit controls
        self.label_objective_limit = QLabel("Limit Position", self)
        self.doubleSpinBox_objective_limit = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_limit.setRange(0.0, self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_limit.setValue(self.fm.objective.limit_position * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_limit.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_limit.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_limit.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_limit.setToolTip("User-defined upper limit for objective position in millimeters")
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
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value * 1e-3))  # Convert from um to mm
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
        self._wheel_target_mm: float = 0.0
        self._execute_wheel_move = qdebounced(self._execute_wheel_move_impl, timeout=150)

        if self.parent_widget is not None and hasattr(self.parent_widget, "viewer"):
            self.parent_widget.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

    def insert_objective(self):
        """Insert the objective."""

        # confirmation dialog
        ret = message_box_ui(
            title="Insert Objective",
            text="Are you sure you want to insert the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective insertion cancelled by user.")
            return

        logging.info("Inserting objective...")
        self.fm.objective.insert()
        logging.info("Objective inserted.")
        self.update_objective_position_labels()

    def retract_objective(self):
        """Retract the objective."""
        # confirmation dialog
        ret = message_box_ui(
            title="Retract Objective",
            text="Are you sure you want to retract the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective retraction cancelled by user.")
            return

        self.fm.objective.retract()
        logging.info("Objective retracted.")
        self.update_objective_position_labels()

    def update_objective_position_labels(self, objective_position: Optional[float] = None):
        """Update the objective position input and label."""
        if objective_position is None:
            objective_position = self.fm.objective.position
        self.doubleSpinBox_objective_position.blockSignals(True)  # Block signals to prevent recursion
        self.doubleSpinBox_objective_position.setValue(objective_position * METRE_TO_MILLIMETRE)  # Convert m to mm
        self.doubleSpinBox_objective_position.blockSignals(False)  # Unblock signals

        if self.parent_widget is not None and hasattr(self.parent_widget, "viewer"):
            update_text_overlay(self.parent_widget.viewer,
                                self.parent_widget.microscope,
                                objective_position=objective_position)  # TODO: migrate to objective_position_changed signal

    @pyqtSlot(float)
    def on_objective_position_changed(self, position: float):
        """Handle changes to the objective position."""

        is_large_change = abs(self.fm.objective.position - (position * MILLIMETRE_TO_METRE)) > 1e-3  # 1 mm threshold

        if is_large_change:
            logging.info(f"Large change in objective position requested: {position:.2f} mm")

            ret = message_box_ui(
                title="Large Objective Movement",
                text=f"Are you sure you want to move the objective position to {position:.2f} mm?",
                parent=self
            )

            if ret is False:
                logging.info("Objective position move cancelled by user.")
                # Reset the spin box to the current position
                self.update_objective_position_labels()
                return

        logging.info(f"Changing objective position to: {position:.2f} mm")
        self.fm.objective.move_absolute(position * MILLIMETRE_TO_METRE)
        logging.info(f"Objective moved to position: {position:.2f} mm")

        # Update the objective position label
        self.update_objective_position_labels()

    @pyqtSlot(float)
    def on_objective_limit_changed(self, limit_mm: float):
        """Handle changes to the user-defined objective limit."""
        limit_m = limit_mm * MILLIMETRE_TO_METRE
        logging.info(f"Updating objective user limit to: {limit_mm:.3f} mm")
        self.fm.objective.limit_position = limit_m
        max_limit_mm = min(limit_mm, self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        min_limit_mm = self.fm.objective.limits[0] * METRE_TO_MILLIMETRE
        self.doubleSpinBox_objective_position.setRange(min_limit_mm, max_limit_mm)
        self.doubleSpinBox_focus_position.setRange(min_limit_mm, max_limit_mm)

    @pyqtSlot(float)
    def on_focus_position_changed(self, position: float):
        """Handle changes to the focus position setting."""
        logging.info(f"Focus position updated to: {position:.3f} mm")
        self.fm.objective.focus_position = position * MILLIMETRE_TO_METRE  # Convert mm to m

    def move_to_focus_position(self):
        """Move the objective to the stored focus position."""
        if self.fm.objective.focus_position is None:
            logging.warning("No focus position has been set")
            return

        focus_position_mm = self.fm.objective.focus_position * METRE_TO_MILLIMETRE

        # confirmation dialog for large movements
        current_position_mm = self.fm.objective.position * METRE_TO_MILLIMETRE
        is_large_change = abs(current_position_mm - focus_position_mm) > 1e-3  # 1 mm threshold

        if is_large_change:
            ret = message_box_ui(
                title="Move to Focus Position",
                text=f"Move objective to focus position {focus_position_mm:.3f} mm?",
                parent=self
            )
            if ret is False:
                logging.info("Move to focus position cancelled by user.")
                return

        logging.info(f"Moving objective to focus position: {focus_position_mm:.3f} mm")
        self.fm.objective.move_absolute(self.fm.objective.focus_position)
        logging.info(f"Objective moved to focus position: {focus_position_mm:.3f} mm")

        # Update the objective position label
        self.update_objective_position_labels()

    def set_focus_position(self):
        """Set the focus position to the current objective position."""
        current_position_mm = self.fm.objective.position * METRE_TO_MILLIMETRE

        ret = message_box_ui(
            title="Set Focus Position",
            text=f"Set focus position to {current_position_mm:.3f} mm?",
            parent=self,
        )

        if ret is False:
            logging.info("Set focus position cancelled by user.")
            return

        # set microscope objective focus position
        logging.info(f"Setting focus position to {current_position_mm:.3f} mm")
        self.doubleSpinBox_focus_position.setValue(current_position_mm)
        self.fm.objective.focus_position = self.fm.objective.position

    def _set_focus_position(self, position: float):
        """Set the focus position programmatically.
        Args:
            position (float): Focus position in meters.
        """
        logging.info(f"Programmatically setting focus position to: {position * METRE_TO_MILLIMETRE:.3f} mm")
        self.doubleSpinBox_focus_position.blockSignals(True)
        self.doubleSpinBox_focus_position.setValue(position * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_focus_position.blockSignals(False)
        self.fm.objective.focus_position = position

    def _set_limit_position(self, position: float):
        """Set the user-defined limit position programmatically.
        Args:
            position (float): Limit position in meters.
        """
        logging.info(f"Programmatically setting limit position to: {position * METRE_TO_MILLIMETRE:.3f} mm")
        self.doubleSpinBox_objective_limit.blockSignals(True)
        self.doubleSpinBox_objective_limit.setValue(position * METRE_TO_MILLIMETRE)
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
        objective_step_size = self.doubleSpinBox_objective_step_size.value() * 1e-3 # convert from um to mm
        step_mm = np.clip(objective_step_size * event.delta[1], -MAX_OBJECTIVE_STEP_SIZE_MM, MAX_OBJECTIVE_STEP_SIZE_MM)

        # Read current position from the spinbox (no hardware call)
        current_pos_mm = self.doubleSpinBox_objective_position.value()
        new_pos_mm = current_pos_mm + step_mm

        logging.info(f"Mouse wheel: step={step_mm:.4f} mm, target={new_pos_mm:.4f} mm")

        # Update spinbox immediately for visual feedback (no hardware call)
        self.doubleSpinBox_objective_position.blockSignals(True)
        self.doubleSpinBox_objective_position.setValue(new_pos_mm)
        self.doubleSpinBox_objective_position.blockSignals(False)

        # Store target and trigger debounced hardware move
        self._wheel_target_mm = new_pos_mm
        self._execute_wheel_move()

        event.handled = True

    def _execute_wheel_move_impl(self):
        """Execute the actual hardware move after scrolling settles."""
        position_mm = self._wheel_target_mm
        position_m = position_mm * MILLIMETRE_TO_METRE
        logging.info(f"Executing debounced wheel move to: {position_mm:.4f} mm")
        self.fm.objective.move_absolute(position_m)
        self.update_objective_position_labels(objective_position=position_m)
