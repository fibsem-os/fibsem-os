import logging
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscopes.simulator import FibsemMicroscope
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
)

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

class StagePositionControlWidget(QWidget):
    """Widget for controlling stage positions and orientations."""

    def __init__(self, microscope: FibsemMicroscope, parent: 'FMAcquisitionWidget'):
        super().__init__(parent)
        self.microscope = microscope
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        """Initialize the stage position control UI."""

        self.button_sem_orientation = QPushButton("Move to SEM Orientation", self)
        self.button_sem_orientation.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.button_sem_orientation.clicked.connect(self.move_to_sem_orientation)

        self.button_fm_orientation = QPushButton("Move to FM Orientation", self)
        self.button_fm_orientation.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.button_fm_orientation.clicked.connect(self.move_to_fm_orientation)

        # Milling angle controls    
        self.milling_angle_spinbox = QDoubleSpinBox(self)
        self.milling_angle_spinbox.setRange(0, 45)
        self.milling_angle_spinbox.setValue(self.microscope.system.stage.milling_angle)
        self.milling_angle_spinbox.setSuffix("°")
        self.milling_angle_spinbox.setDecimals(1)
        self.milling_angle_spinbox.setSingleStep(1.0)
        self.milling_angle_spinbox.valueChanged.connect(self.update_milling_angle)

        self.button_move_to_milling = QPushButton("Move to Milling Angle", self)
        self.button_move_to_milling.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.button_move_to_milling.clicked.connect(self.move_to_milling_angle)

        orientation_layout = QGridLayout()
        orientation_layout.addWidget(self.button_sem_orientation, 0, 0)
        orientation_layout.addWidget(self.button_fm_orientation, 0, 1)
        orientation_layout.addWidget(self.milling_angle_spinbox, 1, 0)
        orientation_layout.addWidget(self.button_move_to_milling, 1, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(orientation_layout)
        self.setLayout(layout)

        orientation_layout.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.set_orientation_tooltips()

    def update_milling_angle(self, value):
        """Update the stored milling angle value."""
        self.microscope.system.stage.milling_angle = value
        # Update tooltip with new milling angle
        self.update_milling_tooltip()

    def set_orientation_tooltips(self):
        """Set tooltips for orientation buttons showing rotation and tilt angles."""
        try:
            sem = self.microscope.get_orientation("SEM")
            fm = self.microscope.get_orientation("FM")

            # Set tooltips for orientation buttons
            self.button_sem_orientation.setToolTip(sem.pretty_orientation)
            self.button_fm_orientation.setToolTip(fm.pretty_orientation)

            self.milling_angle_spinbox.setToolTip("The milling angle is the difference between the stage and the fib viewing angle.")

            self.update_milling_tooltip()

        except Exception as e:
            logging.warning(f"Could not set orientation tooltips: {e}")

    def update_milling_tooltip(self):
        """Update the milling angle button tooltip with current orientation."""
        try:
            milling = self.microscope.get_orientation("MILLING")
            self.button_move_to_milling.setToolTip(milling.pretty_orientation)
        except Exception as e:
            logging.warning(f"Could not update milling tooltip: {e}")

    def move_to_sem_orientation(self):
        """Move stage to SEM (electron beam) orientation."""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Movement",
            "Move stage to SEM orientation?\n\nThis will change the stage position to the electron beam viewing angle.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            logging.info("SEM orientation movement cancelled by user")
            return

        try:
            self.microscope.move_to_microscope("FIBSEM")
            logging.info("Moved to SEM orientation")
        except Exception as e:
            logging.error(f"Failed to move to SEM orientation: {e}")
            QMessageBox.warning(self, "Movement Error", f"Failed to move to SEM orientation:\n{str(e)}")

        self.parent_widget.display_stage_position_overlay()

    def move_to_fm_orientation(self):
        """Move stage to FM (fluorescence microscopy) orientation."""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Movement",
            "Move stage to FM orientation?\n\nThis will change the stage position to the fluorescence microscopy viewing angle.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            logging.info("FM orientation movement cancelled by user")
            return

        try:
            self.microscope.move_to_microscope("FM")
            logging.info("Moved to FM orientation")
        except Exception as e:
            logging.error(f"Failed to move to FM orientation: {e}")
            QMessageBox.warning(self, "Movement Error", f"Failed to move to FM orientation:\n{str(e)}")

        self.parent_widget.display_stage_position_overlay()

    def move_to_milling_angle(self):
        """Move stage to the specified milling angle."""
        # Get current milling angle for display
        milling_angle = self.milling_angle_spinbox.value()

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Movement", 
            f"Move stage to milling angle ({milling_angle}°)?\n\nThis will change the stage position to the specified milling orientation.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            logging.info(f"Milling angle movement to {milling_angle}° cancelled by user")
            return

        try:
            mill_orientation = self.microscope.get_orientation("MILLING")
            self.microscope.move_stage_absolute(mill_orientation)
            logging.info(f"Moved to milling angle: {milling_angle}°")
        except Exception as e:
            logging.error(f"Failed to move to milling angle: {e}")
            QMessageBox.warning(self, "Movement Error", f"Failed to move to milling angle:\n{str(e)}")

        self.parent_widget.display_stage_position_overlay()
