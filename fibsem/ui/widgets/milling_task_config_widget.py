from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QScrollArea,
)
from superqt import QCollapsible

from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.constants import SI_TO_MICRO, MICRO_TO_SI
from fibsem.ui.widgets.milling_alignment_widget import MillingAlignmentWidget
from fibsem.ui.widgets.milling_task_acquisition_settings_widget import (
    MillingTaskAcquisitionSettingsWidget,
)

# GUI Configuration Constants
WIDGET_CONFIG = {
    "name": {"default": "Milling Task", "placeholder": "Enter task name..."},
    "field_of_view": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
    },
}


class MillingTaskConfigWidget(QWidget):
    """Widget for editing FibsemMillingTaskConfig settings.

    Contains basic task settings (name, field of view),
    alignment settings, and acquisition settings. Stages and
    channel are ignored as requested.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the MillingTaskConfig widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings in sub-widgets
        """
        super().__init__(parent)
        self._settings = FibsemMillingTaskConfig()
        self._show_advanced = show_advanced
        self._setup_ui()
        self.update_from_settings(self._settings)
        self._connect_signals()

    def _setup_ui(self):
        """Create and configure all UI elements.

        Creates a scroll area containing a vertical layout with basic settings at the top,
        followed by alignment and acquisition settings in group boxes.
        """
        # Main layout for the widget
        main_layout = QVBoxLayout()
        # main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Create content widget that will be scrolled
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)

        # Basic settings group
        self.basic_content = QWidget()
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(0, 0, 0, 0)
        self.basic_content.setLayout(basic_layout)

        # Task name
        basic_layout.addWidget(QLabel("Name"), 0, 0)
        self.name_edit = QLineEdit()
        name_config = WIDGET_CONFIG["name"]
        self.name_edit.setText(name_config["default"])
        self.name_edit.setPlaceholderText(name_config["placeholder"])
        basic_layout.addWidget(self.name_edit, 0, 1)

        # Field of view
        basic_layout.addWidget(QLabel("Field of View"), 1, 0)
        self.field_of_view_spinbox = QDoubleSpinBox()
        fov_config = WIDGET_CONFIG["field_of_view"]
        self.field_of_view_spinbox.setRange(*fov_config["range"])
        self.field_of_view_spinbox.setDecimals(fov_config["decimals"])
        self.field_of_view_spinbox.setSingleStep(fov_config["step"])
        self.field_of_view_spinbox.setValue(fov_config["default"])
        self.field_of_view_spinbox.setSuffix(fov_config["suffix"])
        basic_layout.addWidget(self.field_of_view_spinbox, 1, 1)

        self.basic_group = QCollapsible("Basic Settings", self)
        self.basic_group.addWidget(self.basic_content)
        self.basic_group.expand(animate=False)
        self.basic_group.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.basic_group)

        # Alignment settings group
        self.alignment_widget = MillingAlignmentWidget(
            show_advanced=self._show_advanced
        )
        self.alignment_group = QCollapsible("Alignment", self)
        self.alignment_group.addWidget(self.alignment_widget)
        self.alignment_group.expand(animate=False)
        self.alignment_group.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.alignment_group)

        # Acquisition settings group
        self.acquisition_widget = MillingTaskAcquisitionSettingsWidget(
            show_advanced=self._show_advanced
        )
        self.acquisition_group = QCollapsible("Acquisition", self)
        self.acquisition_group.addWidget(self.acquisition_widget)
        self.acquisition_group.expand(animate=False)
        self.acquisition_group.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.acquisition_group)

        # Add stretch to push everything to the top
        layout.addStretch()

    def _connect_signals(self):
        """Connect widget signals to their respective handlers.

        Connects all basic controls and sub-widget signals to emit
        settings change notifications when any value changes.
        """
        self.name_edit.textChanged.connect(self._emit_settings_changed)
        self.field_of_view_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.alignment_widget.settings_changed.connect(self._emit_settings_changed)
        self.acquisition_widget.settings_changed.connect(self._emit_settings_changed)

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings.

        Called whenever any control value changes to notify listeners
        of the updated FibsemMillingTaskConfig settings.
        """
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> FibsemMillingTaskConfig:
        """Get the current FibsemMillingTaskConfig from the widget values.

        Returns:
            FibsemMillingTaskConfig object with values from the UI controls.
            Updates only the fields controlled by this widget, preserving
            all other fields from the stored settings (like stages).
        """
        # Update only the fields controlled by this widget
        self._settings.name = self.name_edit.text()
        self._settings.field_of_view = (
            self.field_of_view_spinbox.value() * MICRO_TO_SI
        )  # Convert μm to m
        self._settings.alignment = self.alignment_widget.get_settings()
        self._settings.acquisition = self.acquisition_widget.get_settings()
        # channel and stages are preserved as-is (ignored as requested)

        return self._settings

    def update_from_settings(self, settings: FibsemMillingTaskConfig):
        """Update all widget values from a FibsemMillingTaskConfig object.

        Args:
            settings: FibsemMillingTaskConfig object to load values from.
                     All basic controls and sub-widgets are updated from the settings.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.blockSignals(True)

        self.name_edit.setText(settings.name)
        self.field_of_view_spinbox.setValue(
            settings.field_of_view * SI_TO_MICRO
        )  # Convert m to μm

        # Update sub-widgets
        self.alignment_widget.update_from_settings(settings.alignment)
        self.acquisition_widget.update_from_settings(settings.acquisition)

        self.blockSignals(False)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in sub-widgets.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        self.alignment_widget.set_show_advanced(show_advanced)
        self.acquisition_widget.set_show_advanced(show_advanced)

    def toggle_advanced(self):
        """Toggle the visibility of advanced settings in sub-widgets.

        Switches between showing and hiding the advanced controls in
        alignment and acquisition widgets.
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
    from PyQt5.QtWidgets import QApplication, QPushButton, QCheckBox

    app = QApplication(sys.argv)

    # Create main window
    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    # Create the MillingTaskConfig widget
    config_widget = MillingTaskConfigWidget(show_advanced=False)
    layout.addWidget(config_widget)

    # Add advanced settings toggle checkbox
    advanced_checkbox = QCheckBox("Show Advanced Settings")
    advanced_checkbox.setChecked(config_widget.get_show_advanced())
    advanced_checkbox.toggled.connect(config_widget.set_show_advanced)
    layout.addWidget(advanced_checkbox)

    # Add a button to print current settings
    def print_settings():
        settings = config_widget.get_settings()
        print("Current FibsemMillingTaskConfig:")
        print(f"  name: {settings.name}")
        print(f"  field_of_view: {settings.field_of_view}")
        print(f"  channel: {settings.channel} (preserved)")
        print(f"  alignment: {settings.alignment}")
        print(f"  acquisition: {settings.acquisition}")
        print(f"  stages: {len(settings.stages)} stages (preserved)")

    print_button = QPushButton("Print Current Settings")
    print_button.clicked.connect(print_settings)
    layout.addWidget(print_button)

    # Connect to settings change signal
    def on_settings_changed(settings: FibsemMillingTaskConfig):
        print(
            f"Settings changed - name: {settings.name}, fov: {settings.field_of_view}"
        )

    config_widget.settings_changed.connect(on_settings_changed)

    main_widget.setWindowTitle("MillingTaskConfig Widget Test")
    # main_widget.show()

    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main_widget, area='right')

    napari.run()
    # sys.exit(app.exec_())
