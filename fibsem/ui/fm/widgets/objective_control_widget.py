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

from fibsem.constants import METRE_TO_MILLIMETRE, MILLIMETRE_TO_METRE
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)
from fibsem.ui.utils import message_box_ui

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

OBJECTIVE_CONFIG = {
    "position": {
        "step_size": 0.001,  # mm
        "decimals": 3,  # number of decimal places
        "suffix": " mm",  # unit suffix
        "tooltip": "Objective position in millimeters relative to current position",
    },
    "step_size_control": {
        "range": (1.0, 50.0),  # µm
        "step": 0.1,  # µm
        "default": 1.0,  # µm
        "decimals": 1,  # number of decimal places
        "suffix": " µm",  # unit suffix
        "tooltip": "Step size for objective movement in microns",
    },
}
MAX_OBJECTIVE_STEP_SIZE = 0.05  # mm


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
        self.pushButton_move_to_focus = QPushButton("Move to Focus", self)

        # add double spin box for objective position
        self.label_objective_control = QLabel("Position", self)
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

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.pushButton_insert_objective, 0, 0)
        layout.addWidget(self.pushButton_retract_objective, 0, 1)
        layout.addWidget(self.label_focus_position, 1, 0)
        layout.addWidget(self.doubleSpinBox_focus_position, 1, 1)
        layout.addWidget(self.pushButton_move_to_focus, 2, 0, 1, 2)
        layout.addWidget(self.label_objective_control, 3, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 3, 1) 
        layout.addWidget(self.label_objective_step_size, 4, 0)
        layout.addWidget(self.doubleSpinBox_objective_step_size, 4, 1)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_focus_position.valueChanged.connect(self.on_focus_position_changed)
        self.pushButton_move_to_focus.clicked.connect(self.move_to_focus_position)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value * 1e-3))  # Convert from um to mm

        # set stylesheets
        self.pushButton_insert_objective.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_retract_objective.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_move_to_focus.setStyleSheet(BLUE_PUSHBUTTON_STYLE)

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

    def update_objective_position_labels(self):
        """Update the objective position input and label."""
        objective_position = self.fm.objective.position * METRE_TO_MILLIMETRE  # Convert m to mm
        self.doubleSpinBox_objective_position.blockSignals(True)  # Block signals to prevent recursion
        self.doubleSpinBox_objective_position.setValue(objective_position)  # Convert m to mm
        self.doubleSpinBox_objective_position.blockSignals(False)  # Unblock signals
        
        if self.parent_widget is not None:
            self.parent_widget.display_stage_position_overlay()  # Update the stage position overlay in the parent widget

    @pyqtSlot(float)
    def on_objective_position_changed(self, position: float):
        """Handle changes to the objective position."""

        is_large_change = abs(self.fm.objective.position - (position * MILLIMETRE_TO_METRE)) > 1e-3  # 1 mm threshold

        if is_large_change:
            logging.info(f"Large change in objective position requested: {position:.2f} mm")
        
            ret = message_box_ui(
                title="Large Objective Movement",
                text=f"Are you sure you want to change the objective position to {position:.2f} mm?",
                parent=self)
            
            if ret is False:
                logging.info("Objective position change cancelled by user.")
                # Reset the spin box to the current position
                self.update_objective_position_labels()
                return

        logging.info(f"Changing objective position to: {position:.2f} mm")
        self.fm.objective.move_absolute(position * MILLIMETRE_TO_METRE)
        logging.info(f"Objective moved to position: {position:.2f} mm")

        # Update the objective position label
        self.update_objective_position_labels()

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
