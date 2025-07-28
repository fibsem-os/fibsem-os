import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QDoubleSpinBox, QPushButton, QMessageBox, QComboBox
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import ImageSettings, BeamType
from fibsem.config import SQUARE_RESOLUTIONS_LIST
from fibsem.ui.stylesheets import GREEN_PUSHBUTTON_STYLE

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget


class SEMAcquisitionWidget(QWidget):

    """Widget for acquiring images and managing image layers in napari."""
    
    def __init__(self, microscope: FibsemMicroscope, parent: 'FMAcquisitionWidget'):
        super().__init__(parent)
        self.microscope = microscope
        self.image_settings = ImageSettings(resolution=(2048, 2048), 
                                            dwell_time=1e-6, 
                                            hfw=500e-6, 
                                            beam_type=BeamType.ELECTRON)
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        """Initialize the image acquisition UI."""
        layout = QGridLayout()

        # field of view, dwell time
        self.fov_label = QLabel("Field of View (FOV)")
        self.fov_spinbox = QDoubleSpinBox()
        self.fov_spinbox.setRange(100.0, 2000.0)
        self.fov_spinbox.setValue(500.0)
        self.fov_spinbox.setSuffix(" μm")
        self.fov_spinbox.setSingleStep(10.0)
        self.fov_spinbox.setDecimals(1)

        self.dwell_time_label = QLabel("Dwell Time")
        self.dwell_time_spinbox = QDoubleSpinBox()
        self.dwell_time_spinbox.setRange(0.2, 10.0)
        self.dwell_time_spinbox.setValue(1.0)
        self.dwell_time_spinbox.setSuffix(" μs")
        self.dwell_time_spinbox.setSingleStep(0.1)
        self.dwell_time_spinbox.setDecimals(1)

        self.label_resolution = QLabel("Resolution")
        self.combobox_resolution = QComboBox()
        for res in SQUARE_RESOLUTIONS_LIST:
            self.combobox_resolution.addItem(f"{res[0]}x{res[1]}", userData=res)
        self.combobox_resolution.setCurrentIndex(SQUARE_RESOLUTIONS_LIST.index([2048, 2048]))  # Default to 2048x2048

        # Image acquisition controls
        self.button_acquire_image = QPushButton("Acquire Image")
        self.button_acquire_image.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.button_acquire_image.clicked.connect(self.acquire_image)

        layout.addWidget(self.fov_label, 0, 0)
        layout.addWidget(self.fov_spinbox, 0, 1)
        layout.addWidget(self.dwell_time_label, 1, 0)
        layout.addWidget(self.dwell_time_spinbox, 1, 1)
        layout.addWidget(self.label_resolution, 2, 0)
        layout.addWidget(self.combobox_resolution, 2, 1)

        layout.addWidget(self.button_acquire_image, 3, 0, 1, 2)

        self.fov_spinbox.valueChanged.connect(self.update_fov)
        self.dwell_time_spinbox.valueChanged.connect(self.update_dwell_time)
        self.combobox_resolution.currentIndexChanged.connect(self.update_resolution)

        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

    def update_fov(self, value: float) -> None:
        """Update the field of view based on the spinbox value."""
        self.image_settings.hfw = value * 1e-6
        self.parent_widget.display_stage_position_overlay()

    def update_dwell_time(self, value: float) -> None:
        """Update the dwell time based on the spinbox value."""
        self.image_settings.dwell_time = value * 1e-6

    def update_resolution(self, index: int) -> None:
        """Update the image resolution based on the combobox selection."""
        resolution = self.combobox_resolution.currentData()
        if resolution:
            self.image_settings.resolution = resolution
        else:
            logging.warning("No valid resolution selected in combobox.")

    def acquire_image(self) -> None:
        """Acquire an SEM Overview Image"""
        # TODO: thread this acquisition to avoid blocking the UI
        try:

            if self.microscope.get_stage_orientation() != "FM":  # NOTE: refactor once able to use at T=0
                QMessageBox.warning(self, "Orientation Error", "Please switch to FM orientation before acquiring an image.")
                return

            image = self.microscope.acquire_image(image_settings=self.image_settings)
            if image is None:
                return

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            image.save(os.path.join(self.parent_widget.experiment.path, f"overview-{timestamp}.tif"))

            self.parent_widget.update_persistent_image_signal.emit(image)
        except Exception as e:
            logging.error(f"Failed to acquire image: {e}")
