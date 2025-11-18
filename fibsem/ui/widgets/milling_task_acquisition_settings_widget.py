from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QVBoxLayout, QWidget, QHBoxLayout

from fibsem.milling.tasks import MillingTaskAcquisitionSettings
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget


class FibsemMillingTaskAcquisitionSettingsWidget(QWidget):
    """Widget for editing MillingTaskAcquisitionSettings.

    Contains an enabled checkbox and an ImageSettingsWidget for
    configuring acquisition settings during milling tasks.
    """

    settings_changed = pyqtSignal(MillingTaskAcquisitionSettings)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the MillingTaskAcquisitionSettings widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings in the image settings widget
        """
        super().__init__(parent)
        self._settings = MillingTaskAcquisitionSettings()
        self._setup_ui(show_advanced)
        self._connect_signals()
        self.update_from_settings(self._settings)
        # Initial enabled state update
        self._update_image_settings_enabled()

    def _setup_ui(self, show_advanced: bool):
        """Create and configure all UI elements.

        Args:
            show_advanced: Whether to show advanced settings in the image settings widget
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Enabled checkbox
        self.acquire_sem_checkbox = QCheckBox("Acquire SEM Image")
        self.acquire_sem_checkbox.setChecked(True)
        self.acquire_sem_checkbox.setToolTip(
            self._settings.__dataclass_fields__["acquire_sem"].metadata.get(
                "tooltip", ""
            )
        )
        self.acquire_fib_checkbox = QCheckBox("Acquire FIB Image")
        self.acquire_fib_checkbox.setChecked(True)
        self.acquire_fib_checkbox.setToolTip(
            self._settings.__dataclass_fields__["acquire_fib"].metadata.get(
                "tooltip", ""
            )
        )

        # Image settings widget
        self.image_settings_widget = ImageSettingsWidget(show_advanced=show_advanced)

        hbox = QHBoxLayout()
        hbox.addWidget(self.acquire_sem_checkbox)
        hbox.addWidget(self.acquire_fib_checkbox)
        layout.addLayout(hbox)
        layout.addWidget(self.image_settings_widget)

    def _connect_signals(self):
        """Connect widget signals to their respective handlers."""
        self.acquire_sem_checkbox.toggled.connect(self._emit_settings_changed)
        self.acquire_fib_checkbox.toggled.connect(self._emit_settings_changed)
        self.acquire_sem_checkbox.toggled.connect(self._update_image_settings_enabled)
        self.acquire_fib_checkbox.toggled.connect(self._update_image_settings_enabled)
        self.image_settings_widget.settings_changed.connect(self._emit_settings_changed)

    def _update_image_settings_enabled(self):
        """Enable/disable the image settings widget based on the enabled checkbox."""
        self.image_settings_widget.setEnabled(
            self.acquire_sem_checkbox.isChecked()
            or self.acquire_fib_checkbox.isChecked()
        )

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> MillingTaskAcquisitionSettings:
        """Get the current MillingTaskAcquisitionSettings from the widget values.

        Returns:
            MillingTaskAcquisitionSettings object with values from the UI controls.
            Updates only the fields controlled by this widget, preserving
            all other fields from the stored settings.
        """
        # Update only the fields controlled by this widget
        self._settings.acquire_sem = self.acquire_sem_checkbox.isChecked()
        self._settings.acquire_fib = self.acquire_fib_checkbox.isChecked()
        self._settings.imaging = self.image_settings_widget.get_settings()
        return self._settings

    def update_from_settings(self, settings: MillingTaskAcquisitionSettings):
        """Update all widget values from a MillingTaskAcquisitionSettings object.

        Args:
            settings: MillingTaskAcquisitionSettings object to load values from.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.blockSignals(True)
        self.acquire_sem_checkbox.blockSignals(True)
        self.acquire_fib_checkbox.blockSignals(True)
        self.acquire_sem_checkbox.setChecked(settings.acquire_sem)
        self.acquire_fib_checkbox.setChecked(settings.acquire_fib)
        self.image_settings_widget.update_from_settings(settings.imaging)

        # Update enabled state of image settings widget
        self._update_image_settings_enabled()
        self.acquire_sem_checkbox.blockSignals(False)
        self.acquire_fib_checkbox.blockSignals(False)
        self.blockSignals(False)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in the image settings widget.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self.image_settings_widget.set_show_advanced(show_advanced)

    def toggle_advanced(self):
        """Toggle the visibility of advanced settings in the image settings widget."""
        self.image_settings_widget.toggle_advanced()

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self.image_settings_widget.get_show_advanced()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout

    app = QApplication(sys.argv)

    # Create main window
    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    # Create the MillingTaskAcquisitionSettings widget
    settings_widget = FibsemMillingTaskAcquisitionSettingsWidget(show_advanced=False)
    layout.addWidget(settings_widget)

    # Add advanced settings toggle checkbox
    advanced_checkbox = QCheckBox("Show Advanced Settings")
    advanced_checkbox.setChecked(settings_widget.get_show_advanced())
    advanced_checkbox.toggled.connect(settings_widget.set_show_advanced)
    layout.addWidget(advanced_checkbox)

    # Add a button to print current settings
    def print_settings():
        settings = settings_widget.get_settings()
        print("Current MillingTaskAcquisitionSettings:")
        print(f"  enabled: {settings.enabled}")
        print(f"  imaging: {settings.imaging}")

    print_button = QPushButton("Print Current Settings")
    print_button.clicked.connect(print_settings)
    layout.addWidget(print_button)

    # Connect to settings change signal
    def on_settings_changed(settings: MillingTaskAcquisitionSettings):
        print(f"Settings changed - enabled: {settings.enabled}")

    settings_widget.settings_changed.connect(on_settings_changed)

    main_widget.setWindowTitle("MillingTaskAcquisitionSettings Widget Test")
    main_widget.show()

    sys.exit(app.exec_())
