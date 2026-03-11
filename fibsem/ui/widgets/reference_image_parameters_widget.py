import copy 
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.constants import SI_TO_MICRO, MICRO_TO_SI
from fibsem.structures import ReferenceImageParameters
from fibsem.ui import stylesheets
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox

# GUI Configuration Constants
WIDGET_CONFIG = {
    "field_of_view1": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 80.0,
        "suffix": " μm",
        "tooltip": "Field of view for first reference image",
    },
    "field_of_view2": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
        "tooltip": "Field of view for second reference image",
    },
    "acquire_sem": {
        "tooltip": "Whether to acquire SEM reference images",
    },
    "acquire_fib": {
        "tooltip": "Whether to acquire FIB reference images",
    },
    "acquire_image1": {
        "tooltip": "Whether to acquire first reference image",
    },
    "acquire_image2": {
        "tooltip": "Whether to acquire second reference image",
    },
}


class ReferenceImageParametersWidget(QWidget):
    settings_changed = pyqtSignal(ReferenceImageParameters)

    def __init__(self, parent=None):
        """Initialize the ReferenceImageParameters widget.

        Args:
            parent: Parent widget
            show_imaging_settings: Whether to show the nested ImageSettings controls
        """
        super().__init__(parent)
        self._settings = ReferenceImageParameters()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        self.setLayout(main_layout)

        # ── Acquisition panel ────────────────────────────────────────
        acq_content = QWidget()
        acq_layout = QGridLayout(acq_content)
        acq_layout.setContentsMargins(4, 4, 4, 4)

        # Beam Type Options
        self.acquire_sem_check = QCheckBox("Acquire SEM")
        self.acquire_sem_check.setChecked(True)
        self.acquire_sem_check.setToolTip(WIDGET_CONFIG["acquire_sem"]["tooltip"])

        self.acquire_fib_check = QCheckBox("Acquire FIB")
        self.acquire_fib_check.setChecked(True)
        self.acquire_fib_check.setToolTip(WIDGET_CONFIG["acquire_fib"]["tooltip"])

        # Image 1 Options
        self.acquire_image1_label = QLabel("Acquire Image 1")
        self.acquire_image1_label.setToolTip(WIDGET_CONFIG["acquire_image1"]["tooltip"])
        self.acquire_image1_check = QCheckBox()
        self.acquire_image1_check.setChecked(True)
        self.acquire_image1_check.setToolTip(WIDGET_CONFIG["acquire_image1"]["tooltip"])

        self.fov1_label = QLabel("Field of View 1")
        self.fov1_label.setToolTip(WIDGET_CONFIG["field_of_view1"]["tooltip"])
        fov1_config = WIDGET_CONFIG["field_of_view1"]
        self.fov1_spinbox = ValueSpinBox(
            suffix=fov1_config["suffix"],
            minimum=fov1_config["range"][0],
            maximum=fov1_config["range"][1],
            step=fov1_config["step"],
            decimals=fov1_config["decimals"],
            tooltip=fov1_config["tooltip"],
        )
        self.fov1_spinbox.setValue(fov1_config["default"])

        # Image 2 Options
        self.acquire_image2_label = QLabel("Acquire Image 2")
        self.acquire_image2_label.setToolTip(WIDGET_CONFIG["acquire_image2"]["tooltip"])
        self.acquire_image2_check = QCheckBox()
        self.acquire_image2_check.setChecked(True)
        self.acquire_image2_check.setToolTip(WIDGET_CONFIG["acquire_image2"]["tooltip"])

        self.fov2_label = QLabel("Field of View 2")
        self.fov2_label.setToolTip(WIDGET_CONFIG["field_of_view2"]["tooltip"])
        fov2_config = WIDGET_CONFIG["field_of_view2"]
        self.fov2_spinbox = ValueSpinBox(
            suffix=fov2_config["suffix"],
            minimum=fov2_config["range"][0],
            maximum=fov2_config["range"][1],
            step=fov2_config["step"],
            decimals=fov2_config["decimals"],
            tooltip=fov2_config["tooltip"],
        )
        self.fov2_spinbox.setValue(fov2_config["default"])

        # Info/Warning label for acquisition information
        self.info_label = QLabel()
        self.info_label.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.info_label.setWordWrap(True)
        self.info_label.setVisible(False)

        # Layout arrangement
        acq_layout.addWidget(self.acquire_sem_check, 0, 0)
        acq_layout.addWidget(self.acquire_fib_check, 0, 1)
        acq_layout.addWidget(self.acquire_image1_label, 1, 0)
        acq_layout.addWidget(self.acquire_image1_check, 1, 1)
        acq_layout.addWidget(self.fov1_label, 2, 0)
        acq_layout.addWidget(self.fov1_spinbox, 2, 1)
        acq_layout.addWidget(self.acquire_image2_label, 3, 0)
        acq_layout.addWidget(self.acquire_image2_check, 3, 1)
        acq_layout.addWidget(self.fov2_label, 4, 0)
        acq_layout.addWidget(self.fov2_spinbox, 4, 1)
        acq_layout.addWidget(self.info_label, 5, 0, 1, 2)

        acq_panel = TitledPanel("Acquisition", content=acq_content)
        acq_panel._btn_collapse.setChecked(True)

        # ── Imaging Settings panel ───────────────────────────────────
        self.imaging_widget = ImageSettingsWidget(show_advanced=False, parent=self)
        self.imaging_widget.show_field_of_view(False)
        self.imaging_widget.set_show_advanced_button(False)

        self._btn_advanced_imaging = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        self.imaging_settings_panel = TitledPanel("Imaging Settings", content=self.imaging_widget)
        self.imaging_settings_panel.add_header_widget(self._btn_advanced_imaging)
        self.imaging_settings_panel._btn_collapse.setChecked(True)

        main_layout.addWidget(acq_panel)
        main_layout.addWidget(self.imaging_settings_panel)
        main_layout.addStretch()

    def _connect_signals(self):
        """Connect widget signals to their respective handlers."""
        self.fov1_spinbox.valueChanged.connect(self._on_fov_changed)
        self.fov2_spinbox.valueChanged.connect(self._on_fov_changed)
        self.acquire_sem_check.toggled.connect(self._on_acquisition_toggled)
        self.acquire_fib_check.toggled.connect(self._on_acquisition_toggled)
        self.acquire_image1_check.toggled.connect(self._on_image1_toggled)
        self.acquire_image2_check.toggled.connect(self._on_image2_toggled)
        self._btn_advanced_imaging.toggled.connect(self.imaging_widget.set_show_advanced)

        self.imaging_widget.settings_changed.connect(self._emit_settings_changed)

    def _on_fov_changed(self):
        """Handle FOV spinbox value changes."""
        self._update_information_text()
        self._emit_settings_changed()

    def _on_acquisition_toggled(self):
        """Handle acquisition checkbox toggles."""
        self._update_information_text()
        self._emit_settings_changed()

    def _on_image1_toggled(self, checked: bool):
        """Handle Image 1 checkbox toggle to enable/disable FOV1 control."""
        self.acquire_image1_label.setEnabled(checked)
        self.fov1_label.setEnabled(checked)
        self.fov1_spinbox.setEnabled(checked)
        self._update_information_text()
        self._emit_settings_changed()

    def _on_image2_toggled(self, checked: bool):
        """Handle Image 2 checkbox toggle to enable/disable FOV2 control."""
        self.acquire_image2_label.setEnabled(checked)
        self.fov2_label.setEnabled(checked)
        self.fov2_spinbox.setEnabled(checked)
        self._update_information_text()
        self._emit_settings_changed()

    def _update_information_text(self):
        """Update info/warning label based on acquisition settings."""
        # Check acquisition settings
        acquire_sem = self.acquire_sem_check.isChecked()
        acquire_fib = self.acquire_fib_check.isChecked()
        acquire_image1 = self.acquire_image1_check.isChecked()
        acquire_image2 = self.acquire_image2_check.isChecked()

        # Check if no beam types are selected
        no_beams = not acquire_sem and not acquire_fib

        # Check if no images are selected
        no_images = not acquire_image1 and not acquire_image2

        # Determine if any images will be acquired
        will_acquire = not (no_beams or no_images)

        # Enable/disable imaging settings based on whether images will be acquired
        self.imaging_settings_panel.setEnabled(will_acquire)

        # Build message
        if not will_acquire:
            # Warning case - no images will be acquired
            message = "No reference images will be acquired"
            self.info_label.setVisible(True)
        else:
            # Valid acquisition - show what will be acquired
            # Build beam types string
            beam_types = []
            if acquire_sem:
                beam_types.append("SEM")
            if acquire_fib:
                beam_types.append("FIB")
            beam_str = " and ".join(beam_types)

            # Build FOV string
            fovs = []
            if acquire_image1:
                fovs.append(f"{self.fov1_spinbox.value():.0f}μm")
            if acquire_image2:
                fovs.append(f"{self.fov2_spinbox.value():.0f}μm")
            fov_str = " and ".join(fovs)

            # Construct message
            message = f"Acquire {beam_str} reference images at {fov_str} after task completion"
        self.info_label.setText(message)
        self.info_label.setVisible(True)

    def show_field_of_view(self, field: int, show: bool):
        """Show or hide a specific field of view control.

        Args:
            field: Field number (1 or 2)
            show: True to show the control, False to hide it
        """
        if field == 1:
            self.acquire_image1_label.setVisible(show)
            self.acquire_image1_check.setVisible(show)
            self.fov1_label.setVisible(show)
            self.fov1_spinbox.setVisible(show)
        elif field == 2:
            self.acquire_image2_label.setVisible(show)
            self.acquire_image2_check.setVisible(show)
            self.fov2_label.setVisible(show)
            self.fov2_spinbox.setVisible(show)

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> ReferenceImageParameters:
        """Get the current ReferenceImageParameters from the widget values.

        Returns:
            ReferenceImageParameters object with values from the UI controls.
            Units are converted from display units (μm) to SI units (m).
        """

        # Update settings
        self._settings.imaging = self.imaging_widget.get_settings()
        self._settings.field_of_view1 = self.fov1_spinbox.value() * MICRO_TO_SI  # Convert μm to m
        self._settings.field_of_view2 = self.fov2_spinbox.value() * MICRO_TO_SI  # Convert μm to m
        self._settings.acquire_sem = self.acquire_sem_check.isChecked()
        self._settings.acquire_fib = self.acquire_fib_check.isChecked()
        self._settings.acquire_image1 = self.acquire_image1_check.isChecked()
        self._settings.acquire_image2 = self.acquire_image2_check.isChecked()

        return self._settings

    def update_from_settings(self, settings: ReferenceImageParameters):
        """Update all widget values from a ReferenceImageParameters object.

        Args:
            settings: ReferenceImageParameters object to load values from.
                     Units are converted from SI units (m) to display units (μm).
        """
        self._settings = copy.deepcopy(settings)

        # Block signals on individual widgets to prevent recursive updates
        self.fov1_spinbox.blockSignals(True)
        self.fov2_spinbox.blockSignals(True)
        self.acquire_sem_check.blockSignals(True)
        self.acquire_fib_check.blockSignals(True)
        self.acquire_image1_check.blockSignals(True)
        self.acquire_image2_check.blockSignals(True)

        # Set field of view values
        self.fov1_spinbox.setValue(self._settings.field_of_view1 * SI_TO_MICRO)  # Convert m to μm
        self.fov2_spinbox.setValue(self._settings.field_of_view2 * SI_TO_MICRO)  # Convert m to μm

        # Set acquisition options
        self.acquire_sem_check.setChecked(self._settings.acquire_sem)
        self.acquire_fib_check.setChecked(self._settings.acquire_fib)
        self.acquire_image1_check.setChecked(self._settings.acquire_image1)
        self.acquire_image2_check.setChecked(self._settings.acquire_image2)

        # Update FOV control enabled state based on checkbox state
        self.acquire_image1_label.setEnabled(settings.acquire_image1)
        self.fov1_label.setEnabled(settings.acquire_image1)
        self.fov1_spinbox.setEnabled(settings.acquire_image1)
        self.acquire_image2_label.setEnabled(settings.acquire_image2)
        self.fov2_label.setEnabled(settings.acquire_image2)
        self.fov2_spinbox.setEnabled(settings.acquire_image2)

        # Update imaging settings widget if available
        self.imaging_widget.update_from_settings(self._settings.imaging)

        # Unblock signals
        self.fov1_spinbox.blockSignals(False)
        self.fov2_spinbox.blockSignals(False)
        self.acquire_sem_check.blockSignals(False)
        self.acquire_fib_check.blockSignals(False)
        self.acquire_image1_check.blockSignals(False)
        self.acquire_image2_check.blockSignals(False)

        # Update info/warning label (called after unblocking signals to avoid recursion)
        self._update_information_text()
