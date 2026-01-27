import copy
import logging
import os
from typing import Optional, TYPE_CHECKING, Union

import napari
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)

from fibsem import constants, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.patterns2 import FiducialPattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, Point
from fibsem.ui import stylesheets
from fibsem.ui.widgets.milling_alignment_widget import FibsemMillingAlignmentWidget
from fibsem.ui.widgets.milling_stage_editor_widget import FibsemMillingStageEditorWidget
from fibsem.ui.widgets.milling_task_acquisition_settings_widget import (
    FibsemMillingTaskAcquisitionSettingsWidget,
)
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2
from fibsem.ui.widgets.custom_widgets import WheelBlocker


if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget

# GUI Configuration Constants
WIDGET_CONFIG = {
    "name": {"default": "Milling Task", "placeholder": "Enter task name..."},
    "field_of_view": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
        "tooltip": "Field of view in micrometers (μm)",
        "keyboard_tracking": False,
    },
}

INSTRUCTIONS_TEXT = """Shift + Left Click to Move Selected Pattern
Ctrl + Shift + Left Click to Move All Patterns"""

# TODO: add options checkboxes, show advanced, show milling patterns, etc

class MillingTaskConfigWidget(QWidget):
    """Widget for editing FibsemMillingTaskConfig settings.

    Contains basic task settings (name, field of view),
    alignment settings, and acquisition settings. Stages and
    channel are ignored as requested.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)
    milling_progress_signal = pyqtSignal(dict)
    correlation_result_updated_signal = pyqtSignal(Point)

    def __init__(self, microscope: FibsemMicroscope, 
                 milling_task_config: Optional[FibsemMillingTaskConfig] = None,
                 milling_enabled: bool = True,
                correlation_enabled: bool = True,
                 parent: Optional[Union[QWidget, 'FibsemMillingTaskWidget']] = None):
        """Initialize the MillingTaskConfig widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings in sub-widgets
        """
        super().__init__(parent)
        self.microscope = microscope
        self._show_advanced = False
        self._milling_enabled = milling_enabled
        self._correlation_enabled = correlation_enabled
        self._settings = milling_task_config or FibsemMillingTaskConfig()
        self.parent_widget = parent
        self._setup_ui()
        self.update_from_settings(self._settings)
        self._connect_signals()

    def _setup_ui(self):
        """Create and configure all UI elements.

        Creates a scroll area containing a vertical layout with basic settings at the top,
        followed by alignment and acquisition settings in group boxes.
        """
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        # Create content widget that will be scrolled
        content_widget = QWidget()
        layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        content_widget.setLayout(layout)

        use_scroll_area = True
        try:
            from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget
            use_scroll_area = not isinstance(self.parent_widget, FibsemMillingTaskWidget)
        except Exception:
            use_scroll_area = True

        if use_scroll_area:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # scroll_area.setContentsMargins(0, 0, 0, 0)
            scroll_area.setWidget(content_widget)
            main_layout.addWidget(scroll_area)
        else:
            layout.setContentsMargins(0, 0, 0, 0)
            content_widget.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(content_widget)

        # Basic settings group
        self.basic_group = QGroupBox("Task", self)
        basic_layout = QGridLayout()
        self.basic_group.setLayout(basic_layout)

        # Task name
        self.name_edit = QLineEdit()
        name_config = WIDGET_CONFIG["name"]
        self.name_edit.setText(name_config["default"])
        self.name_edit.setPlaceholderText(name_config["placeholder"])

        # Field of view
        self.field_of_view_spinbox = QDoubleSpinBox()
        fov_config = WIDGET_CONFIG["field_of_view"]
        self.field_of_view_spinbox.setRange(*fov_config["range"])
        self.field_of_view_spinbox.setDecimals(fov_config["decimals"])
        self.field_of_view_spinbox.setSingleStep(fov_config["step"])
        self.field_of_view_spinbox.setValue(fov_config["default"])
        self.field_of_view_spinbox.setSuffix(fov_config["suffix"])
        self.field_of_view_spinbox.setToolTip(fov_config["tooltip"])
        self.field_of_view_spinbox.setKeyboardTracking(fov_config["keyboard_tracking"])
        self.field_of_view_spinbox.installEventFilter(WheelBlocker(self.field_of_view_spinbox))

        basic_layout.addWidget(QLabel("Name"), 0, 0)
        basic_layout.addWidget(self.name_edit, 0, 1)
        basic_layout.addWidget(QLabel("Field of View"), 1, 0)
        basic_layout.addWidget(self.field_of_view_spinbox, 1, 1)

        self.advanced_checkbox = QCheckBox("Show Advanced Settings", self)
        self.advanced_checkbox.setToolTip("Show advanced configuration options (alignment, acquisition, strategy).")
        self.advanced_checkbox.setChecked(self._show_advanced)
        self.advanced_checkbox.toggled.connect(self.set_show_advanced)
        self.show_patterns_checkbox = QCheckBox("Show Milling Patterns", self)
        self.show_patterns_checkbox.setToolTip("Show milling patterns in the viewer.")
        self.show_patterns_checkbox.setChecked(True)
        self.show_patterns_checkbox.stateChanged.connect(self.toggle_pattern_visibility)
        basic_layout.addWidget(self.show_patterns_checkbox, 2, 0, 1, 1)
        basic_layout.addWidget(self.advanced_checkbox, 2, 1, 1, 1)
        basic_layout.setColumnStretch(0, 1)  # Labels column - expandable
        basic_layout.setColumnStretch(1, 1)  # Input widgets column - expandable

        # Alignment settings group
        self.alignment_widget = FibsemMillingAlignmentWidget(
            parent=self,
            show_advanced=self._show_advanced
        )
        # self.alignment_widget.setContentsMargins(0, 0, 0, 0)
        self.alignment_group = QGroupBox("Alignment", self)
        self.alignment_group.setLayout(QVBoxLayout())
        self.alignment_group.layout().addWidget(self.alignment_widget)
        # self.alignment_group.setContentsMargins(0, 0, 0, 0)

        # Acquisition settings group
        self.acquisition_widget = FibsemMillingTaskAcquisitionSettingsWidget(
            parent=self,
            show_advanced=self._show_advanced
        )
        # self.acquisition_widget.setContentsMargins(0, 0, 0, 0)
        self.acquisition_group = QGroupBox("Acquisition", self)
        self.acquisition_group.setLayout(QVBoxLayout())
        self.acquisition_group.layout().addWidget(self.acquisition_widget)
        # self.acquisition_group.setContentsMargins(0, 0, 0, 0)

        # Milling stages editor
        try:
            viewer = self.parent_widget.parent_widget.viewer
        except Exception:        
            viewer = napari.current_viewer()

        self.milling_editor_widget = FibsemMillingStageEditorWidget(
            viewer=viewer,
            microscope=self.microscope,
            milling_stages=[],
            parent=self
        )
        self.milling_editor_widget.setContentsMargins(0, 0, 0, 0)
        self.milling_group = QGroupBox("Milling Stages", self)
        self.milling_group.setLayout(QVBoxLayout())
        self.milling_group.layout().addWidget(self.milling_editor_widget)
        self.milling_group.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.basic_group)           # type: ignore
        layout.addWidget(self.alignment_group)      # type: ignore
        layout.addWidget(self.acquisition_group)    # type: ignore
        layout.addWidget(self.milling_group)         # type: ignore

        self._update_advanced_visibility()

        self.pushButton_open_correlation = QPushButton("Open Correlation")
        self.pushButton_open_correlation.setToolTip("Open the correlation tool to align the FIB and FM images.")
        self.pushButton_open_correlation.clicked.connect(self.open_3d_correlation_widget)
        self.pushButton_open_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.enable_correlation_button(self._correlation_enabled)
        main_layout.addWidget(self.pushButton_open_correlation)

        # NOTES: shouldn't know anything about viewer, or microscope
        # only need microscope to get 'dynamic' values such as milling current
        # should make this 'optional' and pass in from parent
        # viewer is used to display the milling stages
        # could we do this at a higher level and just subscribe to the settings_changed signal?
        layout.addStretch()

        # add instructions label
        self.label_instructions = QLabel(INSTRUCTIONS_TEXT, self) 
        self.label_instructions.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        main_layout.addWidget(self.label_instructions)

        self.milling_widget = FibsemMillingWidget2(
            microscope=self.microscope,
            parent=self
        )
        main_layout.addWidget(self.milling_widget)
        self.milling_widget.setVisible(self._milling_enabled)

    def _connect_signals(self):
        """Connect widget signals to their respective handlers.

        Connects all basic controls and sub-widget signals to emit
        settings change notifications when any value changes.
        """
        self.name_edit.textChanged.connect(self._emit_settings_changed)
        self.field_of_view_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.alignment_widget.settings_changed.connect(self._emit_settings_changed)
        self.acquisition_widget.settings_changed.connect(self._emit_settings_changed)
        self.milling_editor_widget._milling_stages_updated.connect(self._emit_settings_changed)

        if (self.parent_widget is not None
            and hasattr(self.parent_widget, "image_widget")
            and self.parent_widget.image_widget is not None):
            self.parent_widget.image_widget.viewer_update_signal.connect(self._on_viewer_image_updated) 

    def _on_viewer_image_updated(self):
        try:
            fib_image = self.parent_widget.image_widget.ib_image
            self.milling_editor_widget.image = fib_image
            self.milling_editor_widget.update_milling_stage_display()
        except Exception as e:
            logging.error(f"An error occured when updating the image from the viewer: {e}")

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings.

        Called whenever any control value changes to notify listeners
        of the updated FibsemMillingTaskConfig settings.
        """
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_config(self) -> FibsemMillingTaskConfig:
        """Alias for get_settings to match naming conventions."""
        return self.get_settings()
    
    def set_config(self, config: FibsemMillingTaskConfig):
        """Alias for update_from_settings to match naming conventions."""
        self.update_from_settings(config)

    def get_settings(self) -> FibsemMillingTaskConfig:
        """Get the current FibsemMillingTaskConfig from the widget values.

        Returns:
            FibsemMillingTaskConfig object with values from the UI controls.
        """
        self._settings.name = self.name_edit.text()
        self._settings.field_of_view = self.field_of_view_spinbox.value() * constants.MICRO_TO_SI
        self._settings.alignment = self.alignment_widget.get_settings()
        self._settings.acquisition = self.acquisition_widget.get_settings()
        self._settings.stages = self.milling_editor_widget.get_milling_stages()

        return copy.deepcopy(self._settings)

    def update_from_settings(self, settings: FibsemMillingTaskConfig):
        """Update all widget values from a FibsemMillingTaskConfig object.

        Args:
            settings: FibsemMillingTaskConfig object to load values from.
                     All basic controls and sub-widgets are updated from the settings.
        """
        # Block signals to prevent recursive updates
        self.blockSignals(True)

        self._settings = copy.deepcopy(settings)

        self.name_edit.setText(settings.name)
        self.field_of_view_spinbox.setValue(settings.field_of_view * constants.SI_TO_MICRO)
        # Update sub-widgets
        self.alignment_widget.update_from_settings(settings.alignment)
        self.acquisition_widget.update_from_settings(settings.acquisition)
        self.milling_editor_widget.update_from_settings(settings.stages)

        self.milling_editor_widget.update_milling_stage_display()
        self.blockSignals(False)

    def clear(self):
        """Reset all widget values to their defaults."""
        default_settings = FibsemMillingTaskConfig()
        self.update_from_settings(default_settings)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in sub-widgets.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        if self.advanced_checkbox.isChecked() != show_advanced:
            self.advanced_checkbox.blockSignals(True)
            self.advanced_checkbox.setChecked(show_advanced)
            self.advanced_checkbox.blockSignals(False)
        self.alignment_widget.set_show_advanced(show_advanced)
        self.acquisition_widget.set_show_advanced(show_advanced)
        self.milling_editor_widget.set_show_advanced(show_advanced)
        self._update_advanced_visibility()

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

    def toggle_pattern_visibility(self, visible: bool):
        """Toggle the visibility of milling patterns in the viewer.

        Args:
            visible: True to show patterns, False to hide them
        """
        self.milling_editor_widget._toggle_pattern_visibility(visible)

    def _update_advanced_visibility(self):
        """Show or hide advanced sections based on the toggle state."""
        self.alignment_group.setVisible(self._show_advanced)
        self.acquisition_group.setVisible(self._show_advanced)

    def set_background_milling_stages(self, background_stages):
        """Set background milling stages to be displayed in the milling stage editor.
        
        Args:
            background_stages: List of FibsemMillingStage objects to show as background
        """
        self.milling_editor_widget.set_background_milling_stages(background_stages)

    def enable_correlation_button(self, enable: bool):
        """Enable or disable the correlation button.

        Args:
            enable: True to enable the button, False to disable it
        """
        self.pushButton_open_correlation.setEnabled(enable)
        self.pushButton_open_correlation.setVisible(enable)

    def open_3d_correlation_widget(self):
        """Open the correlation widget, connect relevant signals."""

        # attempt to close the windows
        try:
            if self.correlation_widget is not None:
                self.correlation_widget.close()
                self.correlation_widget = None
        except Exception as e:
            logging.warning(f"Error closing correlation widget: {e}")
    
        self.viewer = self.milling_editor_widget.viewer

        # snapshot existing layers
        self.existing_layers = [layer.name for layer in self.viewer.layers]
        for layer_name in self.existing_layers:
            self.viewer.layers[layer_name].visible = False

        from fibsem.correlation.app import CorrelationUI
        self.correlation_widget = CorrelationUI(viewer=self.viewer, parent_ui=self)
        self.correlation_widget.continue_pressed_signal.connect(self.handle_correlation_continue_signal)

        # load fib image, if available
        if (self.parent_widget is not None
            and hasattr(self.parent_widget, "image_widget")
            and self.parent_widget.image_widget is not None):
            fib_image: FibsemImage = self.parent_widget.image_widget.ib_image
        else:
            fib_image: FibsemImage = self.milling_editor_widget.image

        if fib_image is not None and fib_image.metadata is not None:
            md = fib_image.metadata.image_settings
            try:
                filename = os.path.join(md.path, md.filename)
                project_path = str(md.path)
            except Exception as e:
                logging.warning(f"Error getting fib image filename: {e}")
                filename = "FIB Image"
                project_path = ""
            self.correlation_widget.set_project_path(project_path)
            self.correlation_widget.load_fib_image(image=fib_image.filtered_data,
                                                    pixel_size=fib_image.metadata.pixel_size.x,
                                                    filename=filename)
        # create a button to run correlation
        self.viewer.window.add_dock_widget(
            self.correlation_widget,
            area="right",
            name="3DCT Correlation",
            tabify=True
        )

    def handle_correlation_continue_signal(self, data: dict):
        """Handle the correlation signal from the correlation widget."""

        logging.info(f"correlation-data: {data}")

        poi = data.get("poi", None)
        if poi is None:
            self._close_correlation_widget()
            return
        point = Point(x=poi[0], y=poi[1])

        milling_config = self.get_config()
        for milling_stage in milling_config.stages:
            # don't move fiducial patterns
            if isinstance(milling_stage.pattern, FiducialPattern):
                continue
            milling_stage.pattern.point = point
        logging.debug(f"Moved patterns to {point}")

        self.set_config(milling_config)
        self._emit_settings_changed()

        self._close_correlation_widget()
        self.correlation_result_updated_signal.emit(point)

    def _close_correlation_widget(self):
        """Close the correlation widget and clean up."""
        # restore layers visibility
        for layer_name in self.existing_layers:
            self.viewer.layers[layer_name].visible = True

        # remove tab widget, delete object
        self.viewer.window.remove_dock_widget(self.correlation_widget)
        self.correlation_widget = None
        self.existing_layers = []

if __name__ == "__main__":

    import os
    from pathlib import Path

    import napari
    from PyQt5.QtWidgets import QTabWidget, QWidget

    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
        cfg
    )
    from superqt import QCollapsible, QIconifyIcon

    viewer = napari.Viewer()
    main_widget = QTabWidget()

    # set tab to side
    qwidget = QWidget()
    icon1 = QIconifyIcon("material-symbols:experiment", color="white")
    main_widget.addTab(qwidget, icon1, "Experiment") # type: ignore
    layout = QVBoxLayout()
    qwidget.setLayout(layout)
    qwidget.setContentsMargins(0, 0, 0, 0)
    layout.setContentsMargins(0, 0, 0, 0)

    microscope, settings = utils.setup_session(debug=False)

    protocol = AutoLamellaTaskProtocol.load(cfg.TASK_PROTOCOL_PATH)
    milling_task_config = protocol.task_config["Rough Milling"].milling["mill_rough"]  # type: ignore

    # Create the MillingTaskConfig widget
    config_widget = MillingTaskConfigWidget(microscope=microscope)
    layout.addWidget(config_widget)
    config_widget.update_from_settings(milling_task_config)

    # Connect to settings change signal
    def on_task_config_changed(task_config: FibsemMillingTaskConfig):
        print(f"Task Config changed: {utils.current_timestamp_v3(timeonly=False)}")
        print(f"  name: {task_config.name}")
        print(f"  field_of_view: {task_config.field_of_view}")
        print(f"  alignment: {task_config.alignment}")
        print(f"  acquisition: {task_config.acquisition}")
        print(f"  stages: {len(task_config.stages)} stages")
        for stage in task_config.stages:
            print(f"    Stage Name: {stage.name}")
            print(f"    Milling: {stage.milling}")
            print(f"    Pattern: {stage.pattern}")
            print(f"    Strategy: {stage.strategy.config}")
            print("---------------------"*3)

    config_widget.settings_changed.connect(on_task_config_changed)
    main_widget.setWindowTitle("MillingTaskConfig Widget Test")

    viewer.window.add_dock_widget(main_widget, add_vertical_stretch=False, area='right')

    napari.run()
