from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QLabel, QWidget

from fibsem.structures import MillingAlignment
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    ValueSpinBox,
    align_form,
)
from fibsem.ui.widgets.form_builder import configure_spinbox
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

# Each control's label, tooltip and range are MillingAlignment's Parameters.
_SPECS = MillingAlignment.params


def _check_row(layout: QGridLayout, row: int, name: str) -> QCheckBox:
    """A switch as a form row: its label, then a bare checkbox in the field column --
    the shape every other boolean in the editor has."""
    spec = _SPECS[name]
    checkbox = QCheckBox()
    checkbox.setChecked(getattr(MillingAlignment(), name))
    checkbox.setToolTip(spec.doc)
    label = QLabel(spec.label)
    label.setToolTip(spec.doc)
    layout.addWidget(label, row, 0)
    layout.addWidget(checkbox, row, 1, alignment=Qt.AlignLeft)
    return checkbox


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
        layout.setContentsMargins(4, 4, 4, 4)  # as the other forms in the column
        self.setLayout(layout)
        align_form(layout)

        self.enabled_checkbox = _check_row(layout, 0, "enabled")
        self.autocontrast_checkbox = _check_row(layout, 1, "use_autocontrast")
        self.autofocus_checkbox = _check_row(layout, 2, "use_autofocus")

        steps = _SPECS["steps"]
        self.steps_label = QLabel(steps.label)
        layout.addWidget(self.steps_label, 3, 0)
        self.steps_spinbox = IntegerValueSpinBox(tooltip=steps.doc)
        self.steps_spinbox.setRange(int(steps.limits.min), int(steps.limits.max))
        self.steps_spinbox.setValue(MillingAlignment().steps)
        layout.addWidget(self.steps_spinbox, 3, 1)

        full_frame = _SPECS["full_frame_below"]
        self.full_frame_label = QLabel(full_frame.label)
        self.full_frame_label.setToolTip(full_frame.doc)
        layout.addWidget(self.full_frame_label, 4, 0)
        self.full_frame_spinbox = ValueSpinBox()
        configure_spinbox(self.full_frame_spinbox, full_frame)
        self.full_frame_spinbox.setSpecialValueText("Off")
        layout.addWidget(self.full_frame_spinbox, 4, 1)

        # Image settings widget
        self.image_settings_widget = ImageSettingsWidget(show_advanced=False)
        # Hide HFW, autocontrast, and drift correction - only show resolution and dwell time
        self.image_settings_widget.show_field_of_view(False)
        self.image_settings_widget.set_show_autocontrast(False)
        self.image_settings_widget.drift_correction_check.setVisible(False)
        layout.addWidget(self.image_settings_widget, 5, 0, 1, 2)

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
        self.full_frame_spinbox.valueChanged.connect(self._emit_settings_changed)
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
        self.full_frame_label.setEnabled(enabled)
        self.full_frame_spinbox.setEnabled(enabled)
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
        self._settings.full_frame_below = (
            self.full_frame_spinbox.value() / _SPECS["full_frame_below"].scale
        )
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
        self.full_frame_spinbox.blockSignals(True)

        self.enabled_checkbox.setChecked(settings.enabled)
        self.autocontrast_checkbox.setChecked(settings.use_autocontrast)
        self.autofocus_checkbox.setChecked(settings.use_autofocus)
        self.steps_spinbox.setValue(settings.steps)
        self.full_frame_spinbox.setValue(
            settings.full_frame_below * _SPECS["full_frame_below"].scale
        )

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
        self.full_frame_spinbox.blockSignals(False)

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
