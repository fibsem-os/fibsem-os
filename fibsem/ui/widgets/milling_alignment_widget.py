from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QLabel, QSpinBox, QGridLayout, QWidget

from fibsem.structures import MillingAlignment

# GUI Configuration Constants
WIDGET_CONFIG = {
    "interval": {
        "range": (10, 3600),  # 10 seconds to 1 hour
        "default": 30,
        "suffix": " s",
    },
    "enabled": {"default": True, "label": "Enable Initial Alignment"},
    "interval_enabled": {"default": False, "label": "Enable Interval Alignment"},
    "rect_precision": 3,  # Decimal places for rectangle display
}


class MillingAlignmentWidget(QWidget):
    """Widget for editing MillingAlignment settings.

    Contains enabled checkbox, interval settings, and a label
    displaying the rectangle values for drift correction alignment.
    """

    settings_changed = pyqtSignal(MillingAlignment)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the MillingAlignment widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings (interval settings and rectangle)
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
        
        Sets up the grid layout with enabled checkbox, interval controls,
        and rectangle display. Advanced controls are initially visible
        but will be hidden based on show_advanced flag.
        """
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Enabled checkbox
        enabled_config = WIDGET_CONFIG["enabled"]
        self.enabled_checkbox = QCheckBox(enabled_config["label"])
        self.enabled_checkbox.setChecked(enabled_config["default"])
        layout.addWidget(self.enabled_checkbox, 0, 0, 1, 2)

        # Interval enabled checkbox
        interval_enabled_config = WIDGET_CONFIG["interval_enabled"]
        self.interval_enabled_checkbox = QCheckBox(interval_enabled_config["label"])
        self.interval_enabled_checkbox.setChecked(interval_enabled_config["default"])
        layout.addWidget(self.interval_enabled_checkbox, 1, 0, 1, 2)

        # Interval spinbox
        self.interval_label = QLabel("Interval")
        layout.addWidget(self.interval_label, 2, 0)
        self.interval_spinbox = QSpinBox()
        interval_config = WIDGET_CONFIG["interval"]
        self.interval_spinbox.setRange(*interval_config["range"])
        self.interval_spinbox.setValue(interval_config["default"])
        self.interval_spinbox.setSuffix(interval_config["suffix"])
        layout.addWidget(self.interval_spinbox, 2, 1)

        # Rectangle display
        self.rect_display_label = QLabel("Rectangle:")
        layout.addWidget(self.rect_display_label, 3, 0)
        self.rect_label = QLabel("left=0.0, top=0.0, width=1.0, height=1.0")
        # self.rect_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        layout.addWidget(self.rect_label, 3, 1)

    def _connect_signals(self):
        """Connect widget signals to their respective handlers.
        
        Connects checkbox and spinbox signals to update methods and
        settings change emission. Each control change triggers both
        UI updates and settings change notifications.
        """
        self.enabled_checkbox.toggled.connect(self._emit_settings_changed)
        self.enabled_checkbox.toggled.connect(self._update_controls_enabled)
        self.interval_enabled_checkbox.toggled.connect(self._emit_settings_changed)
        self.interval_enabled_checkbox.toggled.connect(self._update_interval_enabled)
        self.interval_spinbox.valueChanged.connect(self._emit_settings_changed)

    def _update_advanced_visibility(self):
        """Show/hide advanced settings based on the show_advanced flag.
        
        Advanced settings include: interval settings and rectangle display.
        """
        self.interval_enabled_checkbox.setVisible(self._show_advanced)
        self.interval_label.setVisible(self._show_advanced)
        self.interval_spinbox.setVisible(self._show_advanced)
        self.rect_display_label.setVisible(self._show_advanced)
        self.rect_label.setVisible(self._show_advanced)

    def _update_controls_enabled(self):
        """Enable/disable controls based on the enabled checkbox.
        
        When the main enabled checkbox is unchecked, all other controls
        (interval settings and rectangle display) are disabled to provide
        clear visual feedback that alignment is turned off.
        """
        enabled = self.enabled_checkbox.isChecked()
        self.interval_enabled_checkbox.setEnabled(enabled)
        self.interval_label.setEnabled(enabled)
        self.interval_spinbox.setEnabled(enabled)
        self.rect_display_label.setEnabled(enabled)
        self.rect_label.setEnabled(enabled)
        self._update_interval_enabled()

    def _update_interval_enabled(self):
        """Enable/disable interval spinbox based on interval enabled checkbox.
        
        The interval spinbox is only enabled when both the main alignment
        is enabled AND the interval alignment checkbox is checked.
        """
        interval_enabled = (
            self.interval_enabled_checkbox.isChecked()
            and self.enabled_checkbox.isChecked()
        )
        self.interval_spinbox.setEnabled(interval_enabled)

    def _update_rect_label(self):
        """Update the rectangle label text with current rectangle values.
        
        Formats the rectangle coordinates and dimensions using the configured
        precision for display. Shows left, top, width, and height values.
        """
        rect = self._settings.rect
        precision = WIDGET_CONFIG["rect_precision"]
        self.rect_label.setText(
            f"left={rect.left:.{precision}f}, top={rect.top:.{precision}f}, "
            f"width={rect.width:.{precision}f}, height={rect.height:.{precision}f}"
        )

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
        self._settings.interval_enabled = self.interval_enabled_checkbox.isChecked()
        self._settings.interval = self.interval_spinbox.value()
        # rect is preserved as-is since it's display-only
        return self._settings

    def update_from_settings(self, settings: MillingAlignment):
        """Update all widget values from a MillingAlignment object.

        Args:
            settings: MillingAlignment object to load values from.
                     All checkbox and spinbox values are updated from the settings.
                     Rectangle display is refreshed with the new rectangle data.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.blockSignals(True)

        self.enabled_checkbox.setChecked(settings.enabled)
        self.interval_enabled_checkbox.setChecked(settings.interval_enabled)
        self.interval_spinbox.setValue(settings.interval)

        # Update rectangle display
        self._update_rect_label()

        # Update enabled states
        self._update_controls_enabled()
        
        # Update advanced visibility
        self._update_advanced_visibility()

        self.blockSignals(False)

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


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout

    app = QApplication(sys.argv)

    # Create main window
    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    # Create the MillingAlignment widget
    alignment_widget = MillingAlignmentWidget(show_advanced=False)
    layout.addWidget(alignment_widget)

    # Add advanced settings toggle checkbox
    from PyQt5.QtWidgets import QCheckBox as TestCheckBox
    advanced_checkbox = TestCheckBox("Show Advanced Settings")
    advanced_checkbox.setChecked(alignment_widget.get_show_advanced())
    advanced_checkbox.toggled.connect(alignment_widget.set_show_advanced)
    layout.addWidget(advanced_checkbox)

    # Add a button to print current settings
    def print_settings():
        settings = alignment_widget.get_settings()
        print("Current MillingAlignment:")
        print(f"  enabled: {settings.enabled}")
        print(f"  interval_enabled: {settings.interval_enabled}")
        print(f"  interval: {settings.interval}")
        print(f"  rect: {settings.rect}")

    print_button = QPushButton("Print Current Settings")
    print_button.clicked.connect(print_settings)
    layout.addWidget(print_button)

    # Connect to settings change signal
    def on_settings_changed(settings: MillingAlignment):
        print(
            f"Settings changed - enabled: {settings.enabled}, interval: {settings.interval}"
        )

    alignment_widget.settings_changed.connect(on_settings_changed)

    main_widget.setWindowTitle("MillingAlignment Widget Test")
    main_widget.show()

    sys.exit(app.exec_())
