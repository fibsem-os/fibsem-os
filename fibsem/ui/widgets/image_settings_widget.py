from __future__ import annotations

from typing import Optional

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
    QVBoxLayout,
    QWidget,
)

from fibsem.devices.beam import STANDARD_RESOLUTIONS, Beam
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, ImageSettings, get_fields_with_metadata
from fibsem.ui import stylesheets
from fibsem.ui.tokens import (
    NEUTRAL_400,
)
from fibsem.ui.utils import find_data, install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    QDirectoryLineEdit,
    align_form,
)
from fibsem.ui.widgets.form_builder import (
    configure_spinbox,
    effective_scale,
    reported_metadata,
    runtime_overrides,
)

# How each control is shown: ImageSettings' field metadata, which takes the beam's
# display hint for the fields bound to a beam parameter.
_META = get_fields_with_metadata(ImageSettings)


def _field(
    name: str,
    microscope: Optional[FibsemMicroscope] = None,
    beam_type: Optional[BeamType] = None,
) -> dict:
    """The field's metadata, with what the beam allows laid over it for a field bound
    to a beam parameter: what ``beam_type`` reports, else what every beam declares."""
    metadata = _META[name]
    beam_parameter = metadata.get("microscope_parameter")
    if not beam_parameter:
        return metadata
    beam = None if microscope is None else microscope.beams.get(beam_type)
    parameter = None if beam is None else beam.parameters.get(beam_parameter)
    return runtime_overrides(
        metadata, reported_metadata(parameter or getattr(Beam, beam_parameter))
    )


def _scale(name: str) -> float:
    return effective_scale(_META[name]) or 1.0


class ImageSettingsWidget(QWidget):
    settings_changed = pyqtSignal(ImageSettings)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        show_advanced: bool = False,
        show_save: bool = False,
        always_save: bool = False,
    ):
        """Initialize the ImageSettings widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings (line integration,
                          scan interlacing, frame integration, drift correction)
            show_save: Whether to show save controls (save image, path, filename)
            always_save: Hide the save checkbox and always return save=True.
                         Path/filename controls remain visible and enabled.
                         Implies show_save=True.
        """
        super().__init__(parent)
        self._settings = ImageSettings()
        self._show_advanced = show_advanced
        self._always_save = always_save
        self._show_save = show_save or always_save
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
        self.btn_advanced = IconToolButton(
            icon="mdi:tune",
            color=NEUTRAL_400,
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

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
        align_form(layout)
        outer_layout.addWidget(grid_widget)

        # Resolution
        self.resolution_label = QLabel(_META["resolution"]["label"])
        self.resolution_combo = QComboBox()
        for width, height in STANDARD_RESOLUTIONS:
            self.resolution_combo.addItem(f"{width}x{height}", (width, height))
        install_wheel_blocker(self.resolution_combo)
        layout.addWidget(self.resolution_label, 0, 0)
        layout.addWidget(self.resolution_combo, 0, 1)

        # Dwell time
        self.dwell_label = QLabel(_META["dwell_time"]["label"])
        self.dwell_time_spinbox = QDoubleSpinBox()
        install_wheel_blocker(self.dwell_time_spinbox)
        configure_spinbox(self.dwell_time_spinbox, _field("dwell_time"))
        layout.addWidget(self.dwell_label, 1, 0)
        layout.addWidget(self.dwell_time_spinbox, 1, 1)

        # Field of View
        self.hfw_label = QLabel(_META["hfw"]["label"])
        self.hfw_spinbox = QDoubleSpinBox()
        install_wheel_blocker(self.hfw_spinbox)
        configure_spinbox(self.hfw_spinbox, _field("hfw"))
        layout.addWidget(self.hfw_label, 2, 0)
        layout.addWidget(self.hfw_spinbox, 2, 1)

        # Line Integration
        self.line_integration_label = QLabel(_META["line_integration"]["label"])
        self.line_integration_spinbox = QSpinBox()
        install_wheel_blocker(self.line_integration_spinbox)
        self.line_integration_spinbox.setRange(
            _META["line_integration"]["minimum"], _META["line_integration"]["maximum"]
        )
        layout.addWidget(self.line_integration_label, 3, 0)
        layout.addWidget(self.line_integration_spinbox, 3, 1)

        # Scan Interlacing
        self.scan_interlacing_label = QLabel(_META["scan_interlacing"]["label"])
        self.scan_interlacing_spinbox = QSpinBox()
        install_wheel_blocker(self.scan_interlacing_spinbox)
        self.scan_interlacing_spinbox.setRange(
            _META["scan_interlacing"]["minimum"], _META["scan_interlacing"]["maximum"]
        )
        layout.addWidget(self.scan_interlacing_label, 4, 0)
        layout.addWidget(self.scan_interlacing_spinbox, 4, 1)

        # Frame Integration
        self.frame_integration_label = QLabel(_META["frame_integration"]["label"])
        self.frame_integration_spinbox = QSpinBox()
        install_wheel_blocker(self.frame_integration_spinbox)
        self.frame_integration_spinbox.setRange(
            _META["frame_integration"]["minimum"], _META["frame_integration"]["maximum"]
        )
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
        self.path_edit.button_browse.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
        )
        layout.addWidget(self.path_label, 9, 0)
        layout.addWidget(self.path_edit, 9, 1)

        # Filename
        self.filename_label = QLabel("Filename")
        self.filename_edit = QLineEdit()
        layout.addWidget(self.filename_label, 10, 0)
        layout.addWidget(self.filename_edit, 10, 1)

        self._save_widgets: list[QWidget] = [
            self.save_image_label,
            self.save_image_check,
            self.path_label,
            self.path_edit,
            self.filename_label,
            self.filename_edit,
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
        for w in [
            self.path_label,
            self.path_edit,
            self.filename_label,
            self.filename_edit,
        ]:
            w.setEnabled(enabled)
            w.setToolTip(tooltip)

    def _update_save_controls_visibility(self):
        """Show/hide all save controls (save image, path, filename)."""
        if self._always_save:
            self.save_image_label.setVisible(False)
            self.save_image_check.setVisible(False)
            for w in [
                self.path_label,
                self.path_edit,
                self.filename_label,
                self.filename_edit,
            ]:
                w.setVisible(True)
                w.setEnabled(True)
        else:
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
        self.btn_advanced.set_icon_state(show_advanced)
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

    def set_available_resolutions(
        self, resolutions: list, default: str | None = None
    ) -> None:
        """Replace the resolution combo items with a custom list.

        Args:
            resolutions: List of (display_str, value) tuples, e.g. ("1536x1024", (1536, 1024)).
            default: Optional display string to select as the default item.
        """
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for res_str, res in resolutions:
            self.resolution_combo.addItem(res_str, tuple(res))
        if default is not None:
            idx = self.resolution_combo.findText(default)
            if idx >= 0:
                self.resolution_combo.setCurrentIndex(idx)
        self.resolution_combo.blockSignals(False)

    def use_beam(self, microscope: FibsemMicroscope, beam_type: BeamType) -> None:
        """Offer what ``beam_type`` can acquire: its resolutions, and its field of view
        and dwell time limits. What the beam doesn't report keeps the standard values."""
        resolutions = _field("resolution", microscope, beam_type).get("items")
        if resolutions:
            selected = self.resolution_combo.currentText()
            self.set_available_resolutions(
                [(f"{w}x{h}", (w, h)) for w, h in resolutions], default=selected
            )
        for name, spinbox in (
            ("hfw", self.hfw_spinbox),
            ("dwell_time", self.dwell_time_spinbox),
        ):
            spinbox.blockSignals(True)
            configure_spinbox(spinbox, _field(name, microscope, beam_type))
            spinbox.blockSignals(False)

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
        self._settings.dwell_time = self.dwell_time_spinbox.value() / _scale(
            "dwell_time"
        )
        self._settings.hfw = self.hfw_spinbox.value() / _scale("hfw")
        self._settings.autocontrast = self.autocontrast_check.isChecked()
        self._settings.line_integration = line_integration
        self._settings.scan_interlacing = scan_interlacing
        self._settings.frame_integration = frame_integration
        self._settings.drift_correction = self.drift_correction_check.isChecked()
        self._settings.save = (
            True if self._always_save else self.save_image_check.isChecked()
        )
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
        # a value the beam doesn't list is added rather than snapped to a neighbour
        resolution = tuple(settings.resolution)
        index = find_data(self.resolution_combo, resolution)
        if index < 0:
            self.resolution_combo.addItem(
                f"{resolution[0]}x{resolution[1]}", resolution
            )
            index = self.resolution_combo.count() - 1
        self.resolution_combo.setCurrentIndex(index)

        self.dwell_time_spinbox.setValue(settings.dwell_time * _scale("dwell_time"))
        self.hfw_spinbox.setValue(settings.hfw * _scale("hfw"))

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
