
import logging
from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)

from fibsem.fm.structures import CameraImageTransform, CameraSettings

if TYPE_CHECKING:
    from fibsem.fm.microscope import FluorescenceMicroscope

CAMERA_CONFIG = {
    "gain": {
        "range": (0, 100),
        "step": 1.0,
        "suffix": " %",
        "tooltip": "Camera gain in percentage (0 to 100)",
    },
    "binning": {
        "available_values": [1, 2, 4, 8],
        "tooltip": "Pixel binning (1x1, 2x2, 4x4, 8x8)",
    },
    "transform": {
        "tooltip": "Image transformation (flip/rotate)",
    },
}

# Mapping for transform display names
TRANSFORM_DISPLAY_NAMES = {
    CameraImageTransform.NONE: "None",
    CameraImageTransform.FLIP_X: "Flip X",
    CameraImageTransform.FLIP_Y: "Flip Y",
    CameraImageTransform.FLIP_XY: "Flip X+Y",
    CameraImageTransform.ROTATE_90_CW: "Rotate 90° CW",
    CameraImageTransform.ROTATE_90_CCW: "Rotate 90° CCW",
    CameraImageTransform.ROTATE_180: "Rotate 180°"
}


class CameraWidget(QWidget):
    """Widget for camera control settings (gain, binning, transform)."""

    def __init__(self, fm: 'FluorescenceMicroscope', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        """Initialize the UI components for the camera widget."""

        # Gain
        self.label_gain = QLabel("Gain", self)
        self.spinBox_gain = QDoubleSpinBox(self)
        self.spinBox_gain.setRange(*CAMERA_CONFIG["gain"]["range"])
        self.spinBox_gain.setSingleStep(CAMERA_CONFIG["gain"]["step"])
        self.spinBox_gain.setSuffix(CAMERA_CONFIG["gain"]["suffix"])
        self.spinBox_gain.setToolTip(CAMERA_CONFIG["gain"]["tooltip"])
        self.spinBox_gain.setKeyboardTracking(False)
        self.spinBox_gain.setValue(self.fm.camera.gain * 100)  # Convert to percentage

        # Binning
        self.label_binning = QLabel("Binning", self)
        self.combobox_binning = QComboBox(self)
        for b in CAMERA_CONFIG["binning"]["available_values"]:
            self.combobox_binning.addItem(f"{b}x{b}", b)
        self.combobox_binning.setToolTip(CAMERA_CONFIG["binning"]["tooltip"])

        # Set current binning
        current_binning = self.fm.camera.binning
        for i in range(self.combobox_binning.count()):
            if self.combobox_binning.itemData(i) == current_binning:
                self.combobox_binning.setCurrentIndex(i)
                break

        # Image Transform
        self.label_transform = QLabel("Image Transform", self)
        self.comboBox_transform = QComboBox(self)
        self.comboBox_transform.setToolTip(CAMERA_CONFIG["transform"]["tooltip"])

        # Populate transform combobox with enum values
        for transform in CameraImageTransform:
            display_name = TRANSFORM_DISPLAY_NAMES.get(transform, transform.name)
            self.comboBox_transform.addItem(display_name, transform)

        # Set current transform
        current_transform = self.fm._transform
        for i in range(self.comboBox_transform.count()):
            if self.comboBox_transform.itemData(i) == current_transform:
                self.comboBox_transform.setCurrentIndex(i)
                break

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_gain, 0, 0)
        layout.addWidget(self.spinBox_gain, 0, 1)
        layout.addWidget(self.label_binning, 1, 0)
        layout.addWidget(self.combobox_binning, 1, 1)
        layout.addWidget(self.label_transform, 2, 0)
        layout.addWidget(self.comboBox_transform, 2, 1)
        self.setLayout(layout)

        # Connect signals
        self.spinBox_gain.valueChanged.connect(self._on_gain_changed)
        self.combobox_binning.currentIndexChanged.connect(self._on_binning_changed)
        self.comboBox_transform.currentIndexChanged.connect(self._on_transform_changed)

    def _on_gain_changed(self, value: float):
        """Handle gain value change."""
        self.fm.set_gain(value / 100)  # Convert percentage to fraction

    def _on_binning_changed(self, idx: int):
        """Handle binning change."""
        binning = self.combobox_binning.itemData(idx)
        self.fm.set_binning(binning)

    def _on_transform_changed(self, idx: int):
        """Handle transform change."""
        transform = self.comboBox_transform.itemData(idx)
        self.fm.set_image_transform(transform)

    @property
    def camera_settings(self) -> CameraSettings:
        """Get the CameraSettings instance with current widget values."""
        return CameraSettings(
            gain=self.spinBox_gain.value() / 100,  # Convert percentage to fraction
            binning=self.combobox_binning.currentData(),
            transform=self.comboBox_transform.currentData()
        )

    @camera_settings.setter
    def camera_settings(self, value: CameraSettings):
        """Set the camera settings and update the display."""
        # Block signals to prevent recursive updates
        self.spinBox_gain.blockSignals(True)
        self.combobox_binning.blockSignals(True)
        self.comboBox_transform.blockSignals(True)

        # Update UI values
        self.spinBox_gain.setValue(value.gain * 100)  # Convert to percentage

        # Set binning
        for i in range(self.combobox_binning.count()):
            if self.combobox_binning.itemData(i) == value.binning:
                self.combobox_binning.setCurrentIndex(i)
                break

        # Set transform
        for i in range(self.comboBox_transform.count()):
            if self.comboBox_transform.itemData(i) == value.transform:
                self.comboBox_transform.setCurrentIndex(i)
                break

        # Unblock signals
        self.spinBox_gain.blockSignals(False)
        self.combobox_binning.blockSignals(False)
        self.comboBox_transform.blockSignals(False)

        # set values in the microscope
        self.fm.set_gain(value.gain)
        self.fm.set_binning(value.binning)
        self.fm.set_image_transform(value.transform)

    @property
    def gain(self) -> float:
        """Get current gain value (as percentage)."""
        return self.spinBox_gain.value()

    @property
    def binning(self) -> int:
        """Get current binning value."""
        return self.combobox_binning.currentData()

    @property
    def transform(self) -> Optional[CameraImageTransform]:
        """Get current transform value."""
        return self.comboBox_transform.currentData()


if __name__ == "__main__":
    # Example usage
    from PyQt5.QtWidgets import QApplication
    from fibsem import utils

    microscope, settings = utils.setup_session()

    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    app = QApplication([])
    widget = CameraWidget(fm=microscope.fm)
    widget.show()
    app.exec_()
