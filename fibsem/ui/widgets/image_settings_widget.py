from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from fibsem.config import STANDARD_RESOLUTIONS_LIST
from fibsem.constants import MICRO_TO_SI, SI_TO_MICRO
from fibsem.structures import ImageSettings

# GUI Configuration Constants
WIDGET_CONFIG = {
    "dwell_time": {
        "range": (0.001, 1000),
        "decimals": 2,
        "step": 0.01,
        "default": 1.0,
        "suffix": " μs",
    },
    "hfw": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
    },
    "line_integration": {"range": (1, 255), "default": 1},
    "scan_interlacing": {"range": (1, 8), "default": 1},
    "frame_integration": {"range": (1, 512), "default": 1},
    "resolution": {"default": [1536, 1024]},
}


class ImageSettingsWidget(QWidget):
    settings_changed = pyqtSignal(ImageSettings)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the ImageSettings widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings (line integration,
                          scan interlacing, frame integration, drift correction)
        """
        super().__init__(parent)
        self._settings = ImageSettings()
        self._show_advanced = show_advanced
        self._setup_ui()
        self._connect_signals()
        self.update_from_settings(self._settings)
        # Initial visibility update
        self._update_drift_correction_visibility()
        self._update_advanced_visibility()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Resolution
        layout.addWidget(QLabel("Resolution"), 0, 0)
        self.resolution_combo = QComboBox()
        for res in STANDARD_RESOLUTIONS_LIST:
            self.resolution_combo.addItem(f"{res[0]}x{res[1]}", res)
        # Set default resolution
        default_resolution = WIDGET_CONFIG["resolution"]["default"]
        default_index = self.resolution_combo.findData(default_resolution)
        if default_index >= 0:
            self.resolution_combo.setCurrentIndex(default_index)
        layout.addWidget(self.resolution_combo, 0, 1)

        # Dwell time
        layout.addWidget(QLabel("Dwell Time"), 1, 0)
        self.dwell_time_spinbox = QDoubleSpinBox()
        dwell_config = WIDGET_CONFIG["dwell_time"]
        self.dwell_time_spinbox.setRange(*dwell_config["range"])
        self.dwell_time_spinbox.setDecimals(dwell_config["decimals"])
        self.dwell_time_spinbox.setSingleStep(dwell_config["step"])
        self.dwell_time_spinbox.setValue(dwell_config["default"])
        self.dwell_time_spinbox.setSuffix(dwell_config["suffix"])
        layout.addWidget(self.dwell_time_spinbox, 1, 1)

        # Field of View
        layout.addWidget(QLabel("Field of View"), 2, 0)
        self.hfw_spinbox = QDoubleSpinBox()
        hfw_config = WIDGET_CONFIG["hfw"]
        self.hfw_spinbox.setRange(*hfw_config["range"])
        self.hfw_spinbox.setDecimals(hfw_config["decimals"])
        self.hfw_spinbox.setSingleStep(hfw_config["step"])
        self.hfw_spinbox.setValue(hfw_config["default"])
        self.hfw_spinbox.setSuffix(hfw_config["suffix"])
        layout.addWidget(self.hfw_spinbox, 2, 1)

        # Line Integration
        self.line_integration_label = QLabel("Line Integration")
        layout.addWidget(self.line_integration_label, 3, 0)
        self.line_integration_spinbox = QSpinBox()
        line_config = WIDGET_CONFIG["line_integration"]
        self.line_integration_spinbox.setRange(*line_config["range"])
        self.line_integration_spinbox.setValue(line_config["default"])
        layout.addWidget(self.line_integration_spinbox, 3, 1)

        # Scan Interlacing
        self.scan_interlacing_label = QLabel("Scan Interlacing")
        layout.addWidget(self.scan_interlacing_label, 4, 0)
        self.scan_interlacing_spinbox = QSpinBox()
        scan_config = WIDGET_CONFIG["scan_interlacing"]
        self.scan_interlacing_spinbox.setRange(*scan_config["range"])
        self.scan_interlacing_spinbox.setValue(scan_config["default"])
        layout.addWidget(self.scan_interlacing_spinbox, 4, 1)

        # Frame Integration
        self.frame_integration_label = QLabel("Frame Integration")
        layout.addWidget(self.frame_integration_label, 5, 0)
        self.frame_integration_spinbox = QSpinBox()
        frame_config = WIDGET_CONFIG["frame_integration"]
        self.frame_integration_spinbox.setRange(*frame_config["range"])
        self.frame_integration_spinbox.setValue(frame_config["default"])
        layout.addWidget(self.frame_integration_spinbox, 5, 1)

        # Boolean options
        self.autocontrast_check = QCheckBox("Auto Contrast")
        layout.addWidget(self.autocontrast_check, 6, 0)

        self.drift_correction_check = QCheckBox("Drift Correction")
        layout.addWidget(self.drift_correction_check, 6, 1)

    def _connect_signals(self):
        """Connect widget signals to their respective handlers."""
        self.resolution_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.dwell_time_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.hfw_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.line_integration_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.scan_interlacing_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.frame_integration_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.frame_integration_spinbox.valueChanged.connect(
            self._update_drift_correction_visibility
        )
        self.autocontrast_check.toggled.connect(self._emit_settings_changed)
        self.drift_correction_check.toggled.connect(self._emit_settings_changed)

    def _update_advanced_visibility(self):
        """Show/hide advanced settings based on the show_advanced flag.

        Advanced settings include: line integration, scan interlacing,
        frame integration, and drift correction controls.
        """
        self.line_integration_label.setVisible(self._show_advanced)
        self.line_integration_spinbox.setVisible(self._show_advanced)
        self.scan_interlacing_label.setVisible(self._show_advanced)
        self.scan_interlacing_spinbox.setVisible(self._show_advanced)
        self.frame_integration_label.setVisible(self._show_advanced)
        self.frame_integration_spinbox.setVisible(self._show_advanced)

        # Drift correction visibility depends on both advanced flag and frame integration
        self._update_drift_correction_visibility()

    def _update_drift_correction_visibility(self):
        """Update drift correction checkbox visibility.

        Drift correction is only shown when advanced settings are enabled
        AND frame integration value is greater than 1.
        """
        show_drift_correction = (
            self._show_advanced and self.frame_integration_spinbox.value() > 1
        )
        self.drift_correction_check.setVisible(show_drift_correction)
        if not show_drift_correction:
            self.drift_correction_check.setChecked(False)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        self._update_advanced_visibility()

    def toggle_advanced(self):
        """Toggle the visibility of advanced settings.

        Switches between showing and hiding the advanced controls.
        """
        self.set_show_advanced(not self._show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self._show_advanced

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> ImageSettings:
        """Get the current ImageSettings from the widget values.

        Returns:
            ImageSettings object with values from the UI controls.
            Units are converted from display units (μs, μm) to SI units (s, m).
            Integration values of 1 are converted to None.
            Updates only the fields controlled by this widget, preserving
            all other fields from the stored settings.
        """
        resolution = self.resolution_combo.currentData()

        # Map 1 to None for integration values
        line_integration = (
            None
            if self.line_integration_spinbox.value() == 1
            else self.line_integration_spinbox.value()
        )
        scan_interlacing = (
            None
            if self.scan_interlacing_spinbox.value() == 1
            else self.scan_interlacing_spinbox.value()
        )
        frame_integration = (
            None
            if self.frame_integration_spinbox.value() == 1
            else self.frame_integration_spinbox.value()
        )

        # Update only the fields controlled by this widget
        self._settings.resolution = tuple(resolution) if resolution else (1536, 1024)
        self._settings.dwell_time = self.dwell_time_spinbox.value() * MICRO_TO_SI  # Convert μs to s
        self._settings.hfw = self.hfw_spinbox.value() * MICRO_TO_SI  # Convert μm to m
        self._settings.autocontrast = self.autocontrast_check.isChecked()
        self._settings.line_integration = line_integration
        self._settings.scan_interlacing = scan_interlacing
        self._settings.frame_integration = frame_integration
        self._settings.drift_correction = self.drift_correction_check.isChecked()
        
        return self._settings

    def update_from_settings(self, settings: ImageSettings):
        """Update all widget values from an ImageSettings object.

        Args:
            settings: ImageSettings object to load values from.
                     Units are converted from SI units (s, m) to display units (μs, μm).
                     None values for integration are converted to 1.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.blockSignals(True)

        # Set resolution
        resolution_list = list(settings.resolution)
        index = self.resolution_combo.findData(resolution_list)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

        self.dwell_time_spinbox.setValue(
            settings.dwell_time * SI_TO_MICRO
        )  # Convert s to μs
        self.hfw_spinbox.setValue(settings.hfw * SI_TO_MICRO)  # Convert m to μm

        # Set integration values (map None to 1)
        self.line_integration_spinbox.setValue(
            settings.line_integration if settings.line_integration is not None else 1
        )
        self.scan_interlacing_spinbox.setValue(
            settings.scan_interlacing if settings.scan_interlacing is not None else 1
        )
        self.frame_integration_spinbox.setValue(
            settings.frame_integration if settings.frame_integration is not None else 1
        )

        self.autocontrast_check.setChecked(settings.autocontrast)
        self.drift_correction_check.setChecked(settings.drift_correction)

        # Update visibility based on settings
        self._update_advanced_visibility()

        self.blockSignals(False)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout

    app = QApplication(sys.argv)

    # Create main window
    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    # Create the ImageSettings widget
    settings_widget = ImageSettingsWidget(show_advanced=False)
    layout.addWidget(settings_widget)

    # Add advanced settings toggle checkbox
    advanced_checkbox = QCheckBox("Show Advanced Settings")
    advanced_checkbox.setChecked(settings_widget.get_show_advanced())
    advanced_checkbox.toggled.connect(settings_widget.set_show_advanced)
    layout.addWidget(advanced_checkbox)

    # Add a button to print current settings
    def print_settings():
        settings = settings_widget.get_settings()
        print("Current ImageSettings:")
        for field, value in settings.__dict__.items():
            print(f"  {field}: {value}")

    print_button = QPushButton("Print Current Settings")
    print_button.clicked.connect(print_settings)
    layout.addWidget(print_button)

    # Connect to settings change signal
    def on_settings_changed(settings: ImageSettings):
        print(f"Settings changed - {settings}")

    settings_widget.settings_changed.connect(on_settings_changed)

    main_widget.setWindowTitle("ImageSettings Widget Test")
    # main_widget.show()
    import napari

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main_widget, area="right")

    napari.run()
    # sys.exit(app.exec_())
