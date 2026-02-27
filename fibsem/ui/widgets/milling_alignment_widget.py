from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QLabel, QSpinBox, QWidget

from fibsem.structures import MillingAlignment
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

# GUI Configuration Constants
WIDGET_CONFIG = {
    "enabled": {"default": True, "label": "Enable Initial Alignment",
                "tooltip": "Align between imaging and milling current before starting milling"},
    "use_contrast": {"default": True, "label": "Use Auto Contrast",
                     "tooltip": "Autocontrast before acquiring alignment image"},
    "use_autofocus": {"default": False, "label": "Use Auto Focus",
                      "tooltip": "Autofocus before acquiring alignment image"},
    "steps": {"range": (1, 9), "default": 3},
}


class FibsemMillingAlignmentWidget(QWidget):
    """Widget for editing MillingAlignment settings.

    Contains enabled checkbox, autocontrast/autofocus options,
    and alignment imaging settings (resolution and dwell time).
    """

    settings_changed = pyqtSignal(MillingAlignment)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the MillingAlignment widget.

        Args:
            parent: Parent widget
            show_advanced: Currently unused, kept for compatibility
        """
        super().__init__(parent)
        self._settings = MillingAlignment()
        self._show_advanced = show_advanced
        self._setup_ui()
        self._connect_signals()
        self.update_from_settings(self._settings)
        # Initial enabled state update
        self._update_controls_enabled()
        self._update_advanced_visibility()

    def _setup_ui(self):
        """Create and configure all UI elements.

        Sets up the grid layout with enabled checkbox, autocontrast/autofocus options,
        and alignment imaging settings widget.
        """
        layout = QGridLayout()
        self.setLayout(layout)

        # Enabled checkbox
        enabled_config = WIDGET_CONFIG["enabled"]
        self.enabled_checkbox = QCheckBox(enabled_config["label"])
        self.enabled_checkbox.setChecked(enabled_config["default"])
        self.enabled_checkbox.setToolTip(enabled_config["tooltip"])
        layout.addWidget(self.enabled_checkbox, 0, 0, 1, 2)

        # use autocontrast and use autofocus on the same row
        use_contrast_config = WIDGET_CONFIG["use_contrast"]
        self.autocontrast_checkbox = QCheckBox(use_contrast_config["label"])
        self.autocontrast_checkbox.setChecked(use_contrast_config["default"])
        self.autocontrast_checkbox.setToolTip(use_contrast_config["tooltip"])
        layout.addWidget(self.autocontrast_checkbox, 1, 0, 1, 1)

        use_autofocus_config = WIDGET_CONFIG["use_autofocus"]
        self.autofocus_checkbox = QCheckBox(use_autofocus_config["label"])
        self.autofocus_checkbox.setChecked(use_autofocus_config["default"])
        self.autofocus_checkbox.setToolTip(use_autofocus_config["tooltip"])
        layout.addWidget(self.autofocus_checkbox, 1, 1, 1, 1)

        # Alignment steps
        self.steps_label = QLabel("Alignment Steps")
        layout.addWidget(self.steps_label, 2, 0)
        self.steps_spinbox = QSpinBox()
        steps_config = WIDGET_CONFIG["steps"]
        self.steps_spinbox.setRange(*steps_config["range"])
        self.steps_spinbox.setValue(steps_config["default"])
        layout.addWidget(self.steps_spinbox, 2, 1)

        # Image settings widget
        self.image_settings_widget = ImageSettingsWidget(show_advanced=False)
        # Hide HFW, autocontrast, and drift correction - only show resolution and dwell time
        self.image_settings_widget.show_field_of_view(False)
        self.image_settings_widget.set_show_autocontrast(False)
        self.image_settings_widget.drift_correction_check.setVisible(False)
        layout.addWidget(self.image_settings_widget, 3, 0, 1, 2)

    def _connect_signals(self):
        """Connect widget signals to their respective handlers.

        Connects checkbox and spinbox signals to update methods and
        settings change emission. Each control change triggers both
        UI updates and settings change notifications.
        """
        self.enabled_checkbox.toggled.connect(self._emit_settings_changed)
        self.enabled_checkbox.toggled.connect(self._update_controls_enabled)
        self.autocontrast_checkbox.toggled.connect(self._emit_settings_changed)
        self.autofocus_checkbox.toggled.connect(self._emit_settings_changed)
        self.steps_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.image_settings_widget.settings_changed.connect(self._emit_settings_changed)

    def _update_advanced_visibility(self):
        """Show/hide advanced settings based on the show_advanced flag.

        Image settings are always shown but with limited fields.
        """
        pass

    def _update_controls_enabled(self):
        """Enable/disable controls based on the enabled checkbox.

        When the main enabled checkbox is unchecked, all other controls
        are disabled to provide clear visual feedback that alignment is turned off.
        """
        enabled = self.enabled_checkbox.isChecked()
        self.autocontrast_checkbox.setEnabled(enabled)
        self.autofocus_checkbox.setEnabled(enabled)
        self.steps_label.setEnabled(enabled)
        self.steps_spinbox.setEnabled(enabled)
        self.image_settings_widget.setEnabled(enabled)

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings.
        
        Called whenever any control value changes to notify listeners
        of the updated MillingAlignment settings.
        """
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> MillingAlignment:
        """Get the current MillingAlignment from the widget values.

        Returns:
            MillingAlignment object with values from the UI controls.
            Updates only the fields controlled by this widget, preserving
            all other fields from the stored settings.
        """
        # Update only the fields controlled by this widget
        self._settings.enabled = self.enabled_checkbox.isChecked()
        self._settings.use_autocontrast = self.autocontrast_checkbox.isChecked()
        self._settings.use_autofocus = self.autofocus_checkbox.isChecked()
        self._settings.steps = self.steps_spinbox.value()
        self._settings.imaging = self.image_settings_widget.get_settings()
        # interval_enabled, interval, and rect are preserved from stored settings
        return self._settings

    def update_from_settings(self, settings: MillingAlignment):
        """Update all widget values from a MillingAlignment object.

        Args:
            settings: MillingAlignment object to load values from.
                     All checkbox and spinbox values are updated from the settings.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.enabled_checkbox.blockSignals(True)
        self.autocontrast_checkbox.blockSignals(True)
        self.autofocus_checkbox.blockSignals(True)
        self.steps_spinbox.blockSignals(True)

        self.enabled_checkbox.setChecked(settings.enabled)
        self.autocontrast_checkbox.setChecked(settings.use_autocontrast)
        self.autofocus_checkbox.setChecked(settings.use_autofocus)
        self.steps_spinbox.setValue(settings.steps)

        # Update image settings widget
        self.image_settings_widget.update_from_settings(settings.imaging)

        # Update enabled states
        self._update_controls_enabled()

        # Update advanced visibility
        self._update_advanced_visibility()

        self.enabled_checkbox.blockSignals(False)
        self.autocontrast_checkbox.blockSignals(False)
        self.autofocus_checkbox.blockSignals(False)
        self.steps_spinbox.blockSignals(False)

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
