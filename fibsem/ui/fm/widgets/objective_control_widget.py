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
    "step_size": 0.001,  # mm
    "decimals": 3,  # number of decimal places
    "suffix": " mm",  # unit suffix
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
        self.label_header = QLabel("Objective", self)
        self.pushButton_insert_objective = QPushButton("Insert Objective", self)
        self.pushButton_retract_objective = QPushButton("Retract Objective", self)

        # add double spin box for objective position
        self.label_objective_control = QLabel("Position", self)
        self.label_objective_step_size = QLabel("Step Size", self)
        self.doubleSpinBox_objective_position = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MILLIMETRE,
                                                        self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MILLIMETRE)  # Convert m to mm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["suffix"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        self.doubleSpinBox_objective_step_size = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_step_size.setRange(1.0, 50.0)      # Set a reasonable range for step size
        self.doubleSpinBox_objective_step_size.setSingleStep(0.1)       # Set step size for the spin box
        self.doubleSpinBox_objective_step_size.setValue(1.0)            # Default step size (1.0 µm)
        self.doubleSpinBox_objective_step_size.setSuffix(" µm")
        self.doubleSpinBox_objective_step_size.setToolTip("Step size for objective movement in microns")
        self.doubleSpinBox_objective_step_size.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # Add a label to display the current objective position
        self.label_objective_position = QLabel(f"Current Objective Position: {self.fm.objective.position*METRE_TO_MILLIMETRE:.2f} mm", self)

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 2)
        layout.addWidget(self.pushButton_insert_objective, 1, 0)
        layout.addWidget(self.pushButton_retract_objective, 1, 1)
        layout.addWidget(self.label_objective_control, 2, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 2, 1) 
        layout.addWidget(self.label_objective_step_size, 3, 0)
        layout.addWidget(self.doubleSpinBox_objective_step_size, 3, 1)
        layout.addWidget(self.label_objective_position, 4, 0, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value * 1e-3))  # Convert from um to mm

        # set stylesheets
        self.pushButton_insert_objective.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_retract_objective.setStyleSheet(RED_PUSHBUTTON_STYLE)

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
        self.label_objective_position.setText(f"Current Objective Position: {objective_position:.2f} mm")
        
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
