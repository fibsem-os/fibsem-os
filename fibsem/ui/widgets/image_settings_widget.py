from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon
from typing import Optional

from fibsem.config import STANDARD_RESOLUTIONS_ZIP
from fibsem.constants import MICRO_TO_SI, SI_TO_MICRO
from fibsem.structures import ImageSettings
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, WheelBlocker

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
        "step": 50.0,
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

    def __init__(self, parent: Optional[QWidget] = None,
                 show_advanced: bool = False,
                 show_save: bool = False):
        """Initialize the ImageSettings widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings (line integration,
                          scan interlacing, frame integration, drift correction)
            show_save: Whether to show save controls (save image, path, filename)
        """
        super().__init__(parent)
        self._settings = ImageSettings()
        self._show_advanced = show_advanced
        self._show_save = show_save
        self._setup_ui()
        self._connect_signals()
        self.update_from_settings(self._settings)
        # Initial visibility update
        self._update_drift_correction_visibility()
        self._update_advanced_visibility()
        self._update_save_controls_visibility()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)
        self.setLayout(outer_layout)

        # --- Header row ---
        self.btn_advanced = QToolButton()
        self.btn_advanced.setIcon(QIconifyIcon("mdi:tune", color="#c0c0c0"))
        self.btn_advanced.setToolTip("Show advanced settings")
        self.btn_advanced.setCheckable(True)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_advanced)
        outer_layout.addWidget(header_row)

        # --- Settings grid ---
        grid_widget = QWidget()
        layout = QGridLayout(grid_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(grid_widget)

        # Resolution
        self.resolution_label = QLabel("Resolution")
        self.resolution_combo = QComboBox()
        for res_str, res in STANDARD_RESOLUTIONS_ZIP:
            self.resolution_combo.addItem(res_str, res)
        # Set default resolution
        default_resolution = WIDGET_CONFIG["resolution"]["default"]
        default_index = self.resolution_combo.findData(default_resolution)
        if default_index >= 0:
            self.resolution_combo.setCurrentIndex(default_index)
        self.resolution_combo.installEventFilter(WheelBlocker(parent=self.resolution_combo))
        layout.addWidget(self.resolution_label, 0, 0)
        layout.addWidget(self.resolution_combo, 0, 1)

        # Dwell time
        self.dwell_label = QLabel("Dwell Time")
        self.dwell_time_spinbox = QDoubleSpinBox()
        self.dwell_time_spinbox.installEventFilter(WheelBlocker(parent=self.dwell_time_spinbox))
        dwell_config = WIDGET_CONFIG["dwell_time"]
        self.dwell_time_spinbox.setRange(*dwell_config["range"])
        self.dwell_time_spinbox.setDecimals(dwell_config["decimals"])
        self.dwell_time_spinbox.setSingleStep(dwell_config["step"])
        self.dwell_time_spinbox.setValue(dwell_config["default"])
        self.dwell_time_spinbox.setSuffix(dwell_config["suffix"])
        layout.addWidget(self.dwell_label, 1, 0)
        layout.addWidget(self.dwell_time_spinbox, 1, 1)

        # Field of View
        self.hfw_label = QLabel("Field of View")
        self.hfw_spinbox = QDoubleSpinBox()
        self.hfw_spinbox.installEventFilter(WheelBlocker(parent=self.hfw_spinbox))
        hfw_config = WIDGET_CONFIG["hfw"]
        self.hfw_spinbox.setRange(*hfw_config["range"])
        self.hfw_spinbox.setDecimals(hfw_config["decimals"])
        self.hfw_spinbox.setSingleStep(hfw_config["step"])
        self.hfw_spinbox.setValue(hfw_config["default"])
        self.hfw_spinbox.setSuffix(hfw_config["suffix"])
        layout.addWidget(self.hfw_label, 2, 0)
        layout.addWidget(self.hfw_spinbox, 2, 1)

        # Line Integration
        self.line_integration_label = QLabel("Line Integration")
        self.line_integration_spinbox = QSpinBox()
        self.line_integration_spinbox.installEventFilter(WheelBlocker(parent=self.line_integration_spinbox))
        line_config = WIDGET_CONFIG["line_integration"]
        self.line_integration_spinbox.setRange(*line_config["range"])
        self.line_integration_spinbox.setValue(line_config["default"])
        layout.addWidget(self.line_integration_label, 3, 0)
        layout.addWidget(self.line_integration_spinbox, 3, 1)

        # Scan Interlacing
        self.scan_interlacing_label = QLabel("Scan Interlacing")
        self.scan_interlacing_spinbox = QSpinBox()
        self.scan_interlacing_spinbox.installEventFilter(WheelBlocker(parent=self.scan_interlacing_spinbox))
        scan_config = WIDGET_CONFIG["scan_interlacing"]
        self.scan_interlacing_spinbox.setRange(*scan_config["range"])
        self.scan_interlacing_spinbox.setValue(scan_config["default"])
        layout.addWidget(self.scan_interlacing_label, 4, 0)
        layout.addWidget(self.scan_interlacing_spinbox, 4, 1)

        # Frame Integration
        self.frame_integration_label = QLabel("Frame Integration")
        self.frame_integration_spinbox = QSpinBox()
        self.frame_integration_spinbox.installEventFilter(WheelBlocker(parent=self.frame_integration_spinbox))
        frame_config = WIDGET_CONFIG["frame_integration"]
        self.frame_integration_spinbox.setRange(*frame_config["range"])
        self.frame_integration_spinbox.setValue(frame_config["default"])
        layout.addWidget(self.frame_integration_label, 5, 0)
        layout.addWidget(self.frame_integration_spinbox, 5, 1)

        # Drift Correction
        self.drift_correction_label = QLabel("Drift Correction")
        self.drift_correction_check = QCheckBox()
        layout.addWidget(self.drift_correction_label, 6, 0)
        layout.addWidget(self.drift_correction_check, 6, 1)

        # Auto Contrast
        self.autocontrast_label = QLabel("Auto Contrast")
        self.autocontrast_check = QCheckBox()
        layout.addWidget(self.autocontrast_label, 7, 0)
        layout.addWidget(self.autocontrast_check, 7, 1)

        # Save Image
        self.save_image_label = QLabel("Save Image")
        self.save_image_check = QCheckBox()
        layout.addWidget(self.save_image_label, 8, 0)
        layout.addWidget(self.save_image_check, 8, 1)

        # Path
        self.path_label = QLabel("Path")
        self.path_edit = QDirectoryLineEdit()
        layout.addWidget(self.path_label, 9, 0)
        layout.addWidget(self.path_edit, 9, 1)

        # Filename
        self.filename_label = QLabel("Filename")
        self.filename_edit = QLineEdit()
        layout.addWidget(self.filename_label, 10, 0)
        layout.addWidget(self.filename_edit, 10, 1)

        self._save_widgets: list[QWidget] = [
            self.save_image_label, self.save_image_check,
            self.path_label, self.path_edit,
            self.filename_label, self.filename_edit,
        ]

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
        self.save_image_check.toggled.connect(self._emit_settings_changed)
        self.save_image_check.toggled.connect(self._update_save_visibility)
        self.path_edit.textChanged.connect(self._emit_settings_changed)
        self.filename_edit.textChanged.connect(self._emit_settings_changed)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)

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
        self.drift_correction_label.setVisible(self._show_advanced)
        self.drift_correction_check.setVisible(self._show_advanced)

        # Drift correction enabled state depends on frame integration value
        self._update_drift_correction_visibility()

    def _update_drift_correction_visibility(self):
        """Update drift correction enabled state.

        Drift correction is only enabled when frame integration > 1.
        When disabled, a tooltip explains the requirement.
        """
        enabled = self.frame_integration_spinbox.value() > 1
        tooltip = "" if enabled else "Requires Frame Integration > 1"
        self.drift_correction_label.setEnabled(enabled)
        self.drift_correction_label.setToolTip(tooltip)
        self.drift_correction_check.setEnabled(enabled)
        self.drift_correction_check.setToolTip(tooltip)
        if not enabled:
            self.drift_correction_check.setChecked(False)

    def _update_save_visibility(self):
        """Enable/disable path and filename controls based on save_image checkbox."""
        enabled = self.save_image_check.isChecked()
        tooltip = "" if enabled else "Enable 'Save Image' to set path/filename"
        for w in [self.path_label, self.path_edit, self.filename_label, self.filename_edit]:
            w.setEnabled(enabled)
            w.setToolTip(tooltip)

    def _update_save_controls_visibility(self):
        """Show/hide all save controls (save image, path, filename)."""
        for w in self._save_widgets:
            w.setVisible(self._show_save)

    def set_show_advanced_button(self, show: bool):
        """Show or hide the advanced settings toggle button."""
        self.btn_advanced.setVisible(show)

    def set_show_save(self, show: bool):
        """Show or hide the save controls (save image, path, filename)."""
        self._show_save = show
        self._update_save_controls_visibility()

    def _on_advanced_toggled(self, checked: bool):
        tooltip = "Hide advanced settings" if checked else "Show advanced settings"
        self.btn_advanced.setToolTip(tooltip)
        self.set_show_advanced(checked)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        self.btn_advanced.blockSignals(True)
        self.btn_advanced.setChecked(show_advanced)
        self.btn_advanced.blockSignals(False)
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

    def set_show_autocontrast(self, show: bool):
        """Show or hide the auto contrast controls."""
        self.autocontrast_label.setVisible(show)
        self.autocontrast_check.setVisible(show)

    def show_field_of_view(self, show: bool):
        """Show or hide the Field of View (HFW) control.

        Args:
            show: True to show the HFW control, False to hide it
        """
        self.hfw_spinbox.setVisible(show)
        self.hfw_label.setVisible(show)

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
        self._settings.save = self.save_image_check.isChecked()
        self._settings.path = self.path_edit.text() or None
        self._settings.filename = self.filename_edit.text()

        return self._settings

    def update_from_settings(self, settings: ImageSettings):
        """Update all widget values from an ImageSettings object.

        Args:
            settings: ImageSettings object to load values from.
                     Units are converted from SI units (s, m) to display units (μs, μm).
                     None values for integration are converted to 1.
        """
        self._settings = settings

        # Block signals on individual widgets to prevent recursive updates
        self.resolution_combo.blockSignals(True)
        self.dwell_time_spinbox.blockSignals(True)
        self.hfw_spinbox.blockSignals(True)
        self.line_integration_spinbox.blockSignals(True)
        self.scan_interlacing_spinbox.blockSignals(True)
        self.frame_integration_spinbox.blockSignals(True)
        self.autocontrast_check.blockSignals(True)
        self.drift_correction_check.blockSignals(True)
        self.save_image_check.blockSignals(True)

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
        self.save_image_check.setChecked(settings.save)
        self.path_edit.lineEdit.blockSignals(True)
        self.filename_edit.blockSignals(True)
        self.path_edit.setText(str(settings.path) if settings.path else "")
        self.filename_edit.setText(settings.filename if settings.filename else "")
        self.path_edit.lineEdit.blockSignals(False)
        self.filename_edit.blockSignals(False)

        # Unblock signals
        self.resolution_combo.blockSignals(False)
        self.dwell_time_spinbox.blockSignals(False)
        self.hfw_spinbox.blockSignals(False)
        self.line_integration_spinbox.blockSignals(False)
        self.scan_interlacing_spinbox.blockSignals(False)
        self.frame_integration_spinbox.blockSignals(False)
        self.autocontrast_check.blockSignals(False)
        self.drift_correction_check.blockSignals(False)
        self.save_image_check.blockSignals(False)

        # Update visibility based on settings
        self._update_advanced_visibility()
        self._update_save_visibility()
        self._update_save_controls_visibility()


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
    main_widget.show()
    # import napari

    # viewer = napari.Viewer()
    # viewer.window.add_dock_widget(main_widget, area="right")

    # napari.run()
    sys.exit(app.exec_())