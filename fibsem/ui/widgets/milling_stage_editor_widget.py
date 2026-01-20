import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import napari
import napari.utils.notifications
from napari.layers import Image as NapariImageLayer
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)
from superqt import QCollapsible

from fibsem import conversions, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    FibsemMillingStage,
    get_strategy,
)
from fibsem.milling.patterning import (
    MILLING_PATTERN_NAMES,
    get_pattern,
)
from fibsem.milling.patterning.patterns2 import (
    BasePattern,
    LinePattern,
)
from fibsem.milling.strategy import (
    MillingStrategy,
    get_strategy_names,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import QFilePathLineEdit, _create_combobox_control
from fibsem.ui.utils import WheelBlocker
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    is_pattern_placement_valid,
)
from fibsem.ui.napari.utilities import is_position_inside_layer

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
    from fibsem.applications.autolamella.ui import AutoLamellaUI

# QUERY: 
# migrate this into three sub widgets: MillingSettingsWidget, PatternSettingsWidget, MillingStrategyConfigWidge?

class FibsemMillingStageWidget(QWidget):
    _milling_stage_changed = pyqtSignal(FibsemMillingStage)

    def __init__(self, 
                 microscope: FibsemMicroscope, 
                 milling_stage: FibsemMillingStage, 
                 parent=None):
        super().__init__(parent)

        self.parameters: Dict[str, Dict[str, Tuple[QLabel, QWidget, Optional[float]]]] = {} # param: label, control, scale
        self.microscope = microscope
        self._milling_stage = milling_stage
        self._manufacturer = self.microscope.manufacturer

        self._create_widgets()
        self._initialise_widgets()
        self.setContentsMargins(0, 0, 0, 0)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def _create_widgets(self):
        """Create the main widgets for the milling stage editor."""
        self.milling_widget = QWidget(self)
        self.milling_widget.setObjectName("widget-milling-settings")
        self.milling_widget.setLayout(QGridLayout())

        self.pattern_widget = QWidget(self)
        self.pattern_widget.setObjectName("widget-milling-pattern")
        self.pattern_widget.setLayout(QGridLayout())

        self.strategy_widget = QWidget(self)
        self.strategy_widget.setObjectName("widget-milling-strategy")
        self.strategy_widget.setLayout(QGridLayout())

        # create label and combobox
        self.label_pattern_name = QLabel(self)
        self.label_pattern_name.setText("Name")
        self.comboBox_selected_pattern = QComboBox(self)
        self.comboBox_selected_pattern.addItems(MILLING_PATTERN_NAMES)
        self.wheel_blocker1 = WheelBlocker()
        self.comboBox_selected_pattern.installEventFilter(self.wheel_blocker1)
        self.comboBox_selected_pattern.currentTextChanged.connect(self._on_pattern_changed)
        self.pattern_widget.layout().addWidget(self.label_pattern_name, 0, 0, 1, 1)
        self.pattern_widget.layout().addWidget(self.comboBox_selected_pattern, 0, 1, 1, 1)
        self.pattern_widget.layout().setColumnStretch(0, 1)  # Labels column - expandable
        self.pattern_widget.layout().setColumnStretch(1, 1)  # Input widgets column - expandable

        # create strategy widget
        self.label_strategy_name = QLabel(self)
        self.label_strategy_name.setText("Name")
        self.comboBox_selected_strategy = QComboBox(self)
        self.strategy_widget.layout().addWidget(self.label_strategy_name, 0, 0, 1, 1)
        self.strategy_widget.layout().addWidget(self.comboBox_selected_strategy, 0, 1, 1, 1)

        self.comboBox_selected_strategy.addItems(get_strategy_names())
        self.wheel_blocker2 = WheelBlocker()
        self.comboBox_selected_strategy.installEventFilter(self.wheel_blocker2)
        self.comboBox_selected_strategy.currentTextChanged.connect(self._on_strategy_changed)

        # Create the widgets list to hold all the widgets
        self._widgets = [
            self.milling_widget,
            self.pattern_widget,
            self.strategy_widget,
        ]

        # Add the widgets to the main layout
        self.gridlayout = QGridLayout(self)
        label = QLabel(self)
        label.setText("Stage Name")
        label.setObjectName("label-milling-stage-name")
        self.lineEdit_milling_stage_name = QLineEdit(self)
        self.lineEdit_milling_stage_name.setText(self._milling_stage.name)
        self.lineEdit_milling_stage_name.setObjectName("lineEdit-name-stage")
        self.lineEdit_milling_stage_name.setToolTip("The name of the milling stage.")
        self.lineEdit_milling_stage_name.editingFinished.connect(self._update_setting)
        self.milling_widget.layout().addWidget(label, 0, 0, 1, 1)
        self.milling_widget.layout().addWidget(self.lineEdit_milling_stage_name, 0, 1, 1, 1)
        self.milling_widget.layout().setColumnStretch(0, 1)  # Labels column - expandable
        self.milling_widget.layout().setColumnStretch(1, 1)  # Input widgets column - expandable

        # add group boxes for each widget
        self.milling_groupbox = QGroupBox("Milling", self)
        self.milling_groupbox.setLayout(QVBoxLayout())
        self.milling_groupbox.layout().addWidget(self.milling_widget)
        
        self.pattern_groupbox = QGroupBox("Pattern", self)
        self.pattern_groupbox.setLayout(QVBoxLayout())
        self.pattern_groupbox.layout().addWidget(self.pattern_widget)

        self.strategy_groupbox = QGroupBox("Strategy", self)
        self.strategy_groupbox.setLayout(QVBoxLayout())
        self.strategy_groupbox.layout().addWidget(self.strategy_widget)

        self.gridlayout.addWidget(self.milling_groupbox, self.gridlayout.rowCount(), 0, 1, 2) # type: ignore
        self.gridlayout.addWidget(self.pattern_groupbox, self.gridlayout.rowCount(), 0, 1, 2) # type: ignore
        self.gridlayout.addWidget(self.strategy_groupbox, self.gridlayout.rowCount(), 0, 1, 2) # type: ignore

    def _initialise_widgets(self):
        """Initialise the widgets with the current milling stage settings."""
        # MILLING SETTINGS
        milling_params = self._milling_stage.milling.get_parameters(self._manufacturer)
        milling_field_metadata = self._milling_stage.milling.field_metadata
        self._create_controls(self.milling_widget, milling_params, "milling", milling_field_metadata)

        # PATTERN
        self.comboBox_selected_pattern.blockSignals(True)
        self.comboBox_selected_pattern.setCurrentText(self._milling_stage.pattern.name)
        self.comboBox_selected_pattern.blockSignals(False)
        self._update_pattern_widget(self._milling_stage.pattern)  # Set default pattern

        # STRATEGY
        self.comboBox_selected_strategy.blockSignals(True)
        self.comboBox_selected_strategy.setCurrentText(self._milling_stage.strategy.name)
        self.comboBox_selected_strategy.blockSignals(False)
        self._update_strategy_widget(self._milling_stage.strategy)  # Set default strategy

    def toggle_advanced_settings(self, show: bool):
        """Toggle the visibility of advanced settings."""
        ms = self._milling_stage
        wp = self.parameters
        for param in ms.pattern.advanced_attributes:

            label, control, _ = wp["pattern"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        for param in ms.strategy.config.advanced_attributes:
            label, control, _ = wp["strategy.config"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        for param in ms.milling.advanced_attributes:
            label, control, _ = wp["milling"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        # consider strategy as advanced, so hide it as well
        self.strategy_groupbox.setVisible(show)

    def clear_widget(self, widget: QWidget, row_threshold: int = -1):
        """Clear the widget's layout, removing all items below a certain row threshold."""

        items_to_remove = []
        grid_layout = widget.layout()
        if grid_layout is None or not isinstance(grid_layout, QGridLayout):
            raise ValueError(f"Widget {widget} does not have a layout. Expected QGridLayout, got {type(grid_layout)}.")

        # iterate through the items in the grid layout
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item is not None:
                row, col, rowspan, colspan = grid_layout.getItemPosition(i)
                if row > row_threshold:
                    items_to_remove.append(item)

        # Remove the items
        for item in items_to_remove:
            grid_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()

    def _on_pattern_changed(self, pattern_name: str):
        """Update the pattern widget with the selected pattern's parameters."""
        # TODO: convert the comboBox_selected_pattern to use currentData,
        # that way we can pass the pattern object directly (and restore it from the previous state)
        pattern = get_pattern(pattern_name)

        self._milling_stage.pattern = pattern
        self._update_pattern_widget(pattern)
        self._milling_stage_changed.emit(self._milling_stage)

    def _update_pattern_widget(self, pattern: BasePattern):
        """Update the pattern widget with the selected pattern's parameters."""

        params = {k: getattr(pattern, k) for k in pattern.required_attributes if hasattr(pattern, k)}
        params["point"] = pattern.point  # add point as a special case

        self._create_controls(self.pattern_widget, params, "pattern", pattern.field_metadata)

    def _on_strategy_changed(self, strategy_name: str):
        """Update the strategy widget with the selected strategy's parameters."""
        strategy = get_strategy(strategy_name, {"config": {}})

        # update strategy and widget
        self._milling_stage.strategy = strategy
        self._update_strategy_widget(strategy)
        self._milling_stage_changed.emit(self._milling_stage)

    def _update_strategy_widget(self, strategy: MillingStrategy[Any]):
        """Update the strategy widget with the selected strategy's parameters."""
        params = {k: getattr(strategy.config, k) for k in strategy.config.required_attributes}

        self._create_controls(self.strategy_widget, params, "strategy.config", strategy.config.field_metadata)

    def _create_controls(self, widget: QWidget,
                         params: Dict[str, Any],
                         control_type: str,
                         field_metadata: Dict[str, Dict[str, Any]]):
        """Create controls for the given parameters and add them to the widget."""

        # clear previous controls
        if control_type == "pattern":
            self.clear_widget(self.pattern_widget, row_threshold=0)
        if control_type == "strategy.config":
            self.clear_widget(self.strategy_widget, row_threshold=0)

        self.parameters[control_type] = {}
        grid_layout = widget.layout()
        if grid_layout is None:
            raise ValueError(f"Widget {widget} does not have a layout. Expected QGridLayout, got {type(grid_layout)}.")
        if not isinstance(grid_layout, QGridLayout):
            raise TypeError(f"Expected QGridLayout, got {type(grid_layout)} for widget {widget}.")

        # point controls (special case). but why do they have to be?
        if control_type == "pattern":
            self._create_point_controls(control_type, params, field_metadata, grid_layout)

        for name, value in params.items():

            conf = field_metadata.get(name, {})

            # skip parameters that should not be shown in the GUI
            if conf.get("hidden", False) is True:
                continue
            # QUERY: use a data structure for field metadata instead of dict?
            label_text = conf.get("label") or name.replace("_", " ").title()
            scale = conf.get("scale")
            units = conf.get("unit")
            minimum = conf.get("minimum")
            maximum = conf.get("maximum")
            step_size = conf.get("step")
            decimals = conf.get("decimals")
            items = conf.get("items", [])
            tooltip = conf.get("tooltip", None)
            parameter_mapping = conf.get("microscope_parameter", name)
            is_filepath = conf.get("filepath", False)
            dimensions = conf.get("dimensions", None)
            format_fn = conf.get("format_fn", None)

            # display unit -> prefix + unit
            display_unit = None
            if units is not None and scale is not None:
                display_unit = utils._get_display_unit(scale, units)
            elif units is not None:
                display_unit = units

            # adjust scale for dimensions
            if dimensions is not None and scale is not None:
                scale = scale ** dimensions

            # set label text
            label = QLabel(label_text)

            # add combobox controls
            if items:
                if items == "dynamic":
                    items = self.microscope.get_available_values_cached(parameter_mapping, BeamType.ION)
                control = _create_combobox_control(value, items, units, format_fn)

            # add line edit controls
            elif isinstance(value, str):
                if is_filepath:
                    control = QFilePathLineEdit()
                else:
                    control = QLineEdit()
                control.setText(value)
            # add checkbox controls
            elif isinstance(value, bool):
                control = QCheckBox()
                control.setChecked(value)
            elif isinstance(value, (float, int)):

                control = QDoubleSpinBox()
                if display_unit is not None:
                    control.setSuffix(f' {display_unit}')
                if scale is not None:
                    value = value * scale
                if minimum is not None:
                    control.setMinimum(minimum)
                if maximum is not None:
                    control.setMaximum(maximum)
                if step_size is not None:
                    control.setSingleStep(step_size)
                if decimals is not None:
                    control.setDecimals(decimals)
                control.setValue(value)
                control.setKeyboardTracking(False)
            else: # unsupported type
                continue

            # Set tooltip for both label and control
            if tooltip:
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)

            grid_layout.addWidget(label, grid_layout.rowCount(), 0)
            grid_layout.addWidget(control, grid_layout.rowCount() - 1, 1)

            label.setObjectName(f"label-{control_type}-{name}")
            control.setObjectName(f"control-{control_type}-{name}")
            self.parameters[control_type][name] = (label, control, scale)

            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._update_setting)
            elif isinstance(control, (QLineEdit, QFilePathLineEdit)):
                control.editingFinished.connect(self._update_setting)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._update_setting)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self._update_setting)

    # add callback to update settings when control value changes
    def _update_setting(self):
        obj = self.sender()
        if not obj:
            logging.warning("No sender object for _update_setting")
            return
        obj_name = obj.objectName()
        _, control_type, name = obj_name.split("-", 2)

        if isinstance(obj, QComboBox):
            value = obj.currentData()
        elif isinstance(obj, (QLineEdit, QFilePathLineEdit)):
            value = obj.text()
        elif isinstance(obj, QCheckBox):
            value = obj.isChecked()
        elif isinstance(obj, (QSpinBox, QDoubleSpinBox)):
            value = obj.value()
            # apply scale if defined
            scale = self.parameters[control_type][name][2]
            if scale is not None:
                value /= scale
        else:
            logging.warning(f"Unsupported control type: {obj_name} {type(obj)} for {control_type}-{name}")
            return

        # update the milling_stage object
        if hasattr(self._milling_stage, control_type):

            # special case for pattern point
            if "point" in name:
                if "x" in name:
                    setattr(self._milling_stage.pattern.point, "x", value)
                elif "y" in name:
                    setattr(self._milling_stage.pattern.point, "y", value)
            elif control_type == "name":
                setattr(self._milling_stage, "name", value)
            else:
                setattr(getattr(self._milling_stage, control_type), name, value)
        elif hasattr(self._milling_stage, "strategy") and control_type == "strategy.config":
            # Special case for strategy config
            setattr(self._milling_stage.strategy.config, name, value)
        else:
            logging.debug(f"Warning: {control_type} not found in milling_stage object. Cannot update {name}.")

        self._milling_stage_changed.emit(self._milling_stage)  # notify changes

    def _create_point_controls(self, control_type: str, 
                               params: Dict[str, Any], 
                               field_metadata: Dict[str, Dict[str, Any]], 
                               grid_layout: QGridLayout):
        """Create controls for the point parameter (x, y)."""

        point_field_metadata = field_metadata["point"]
        label_text = point_field_metadata["label"]
        units = point_field_metadata["unit"]
        scale = point_field_metadata["scale"]
        minimum = point_field_metadata["minimum"]
        maximum = point_field_metadata["maximum"]
        step_size = point_field_metadata["step"]
        decimals = point_field_metadata["decimals"]
        tooltip = point_field_metadata["tooltip"]
        display_unit = None
        if units is not None and scale is not None:
            display_unit = utils._get_display_unit(scale, units)
        elif units is not None:
            display_unit = units

        # create label for point
        pt_label = QLabel(self)
        pt_label.setText(label_text)
        pt_label.setObjectName(f"label-{control_type}-point")
        pt_label.setToolTip(tooltip)

        hbox_layout = QHBoxLayout()
        for attr in ["x", "y"]:
            control = QDoubleSpinBox(self)
            control.setSuffix(f" {display_unit}")
            control.setRange(minimum, maximum)
            control.setSingleStep(step_size)
            value = getattr(params["point"], attr)
            if scale is not None:
                value *= scale
            control.setValue(value)
            control.setObjectName(f"control-pattern-point.{attr}")
            control.setKeyboardTracking(False)
            control.setDecimals(decimals)
            control.valueChanged.connect(self._update_setting)

            self.parameters[control_type][f"point.{attr}"] = (pt_label, control, scale)
            hbox_layout.addWidget(control)

        # add both point controls to widget, set the padding to 0 to match other visual
        point_widget = QWidget(self)
        point_widget.setObjectName(f"point-widget-{control_type}")
        point_widget.setToolTip(tooltip)
        hbox_layout.setContentsMargins(0, 0, 0, 0)
        point_widget.setContentsMargins(0, 0, 0, 0)
        point_widget.setLayout(hbox_layout)

        # add to the grid layout
        row = grid_layout.rowCount()
        grid_layout.addWidget(pt_label, row, 0, 1, 1)
        grid_layout.addWidget(point_widget, row, 1, 1, 1)

    def get_milling_stage(self) -> FibsemMillingStage:
        return self._milling_stage

    def set_point(self, point: Point) -> None:
        """Set the point for the milling pattern."""

        # Update the point controls
        control: QDoubleSpinBox
        for attr in ["x", "y"]:
            label, control, scale = self.parameters["pattern"][f"point.{attr}"]
            value = getattr(point, attr) * scale
            control.setValue(value)


class FibsemMillingStageEditorWidget(QWidget):
    _milling_stages_updated = pyqtSignal(list)
    """A widget to edit the milling stage settings."""

    def __init__(self,
                 viewer: napari.Viewer,
                 microscope: FibsemMicroscope,
                 milling_stages: List[FibsemMillingStage],
                 parent: Optional['MillingTaskConfigWidget']=None):
        super().__init__(parent)

        self.microscope = microscope
        self._milling_stages = milling_stages
        self._background_milling_stages: List[FibsemMillingStage] = []
        self.is_updating_pattern = False
        self._show_advanced: bool = False
        self.is_movement_locked: bool = False

        self.viewer = viewer

        self.parent_widget = parent
        self.image: Optional[FibsemImage] = None
        self.image_layer: Optional[NapariImageLayer] = None
        self.image_widget: Optional['FibsemImageSettingsWidget'] = None
        if self.parent_widget is not None:
            if self.parent_widget.parent() is not None:
                super_parent: 'AutoLamellaUI' = self.parent_widget.parent()
                if hasattr(super_parent, "image_widget") and super_parent.image_widget is not None:
                    self.image_widget = super_parent.image_widget
                    self.image = super_parent.image_widget.ib_image
                    self.image_layer = super_parent.image_widget.ib_layer
        if self.image is None:
            self.image = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
            if self.viewer is not None:
                self.image_layer = self.viewer.add_image(data=self.image.data, name="FIB Image") # type: ignore
            else:
                self.image_layer = None
        self._widgets: List[FibsemMillingStageWidget] = []

        # add widget for scroll content
        self.milling_stage_content = QWidget()
        self.milling_stage_layout = QVBoxLayout(self.milling_stage_content)

        # add a list widget to hold the milling stages, with re-ordering support
        self.list_widget_milling_stages = QListWidget(self)
        self.list_widget_milling_stages.setDragDropMode(QListWidget.InternalMove)
        self.list_widget_milling_stages.setDefaultDropAction(Qt.MoveAction)
        self.list_widget_milling_stages.setMaximumHeight(60)
        model = self.list_widget_milling_stages.model()
        if model is None:
            raise ValueError("List widget model is None. Ensure the list widget is properly initialized.")
        model.rowsMoved.connect(self._reorder_milling_stages)

        # add milling widgets for each milling stage
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)

        # add/remove buttons for milling stages
        self.pushButton_add = QPushButton("Add Milling Stage", self)
        self.pushButton_add.clicked.connect(lambda: self._add_milling_stage(None))
        self.pushButton_add.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove = QPushButton("Remove Selected Stage", self)
        self.pushButton_remove.clicked.connect(self._remove_selected_milling_stage)
        self.pushButton_remove.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pushButton_add)
        button_layout.addWidget(self.pushButton_remove)

        # add checkboxes for show advanced settings, show milling crosshair, show milling patterns
        self.checkBox_show_milling_crosshair = QCheckBox("Show Milling Crosshair", self)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.setToolTip("Show the milling crosshair in the viewer.")
        self.checkBox_show_milling_patterns = QCheckBox("Show Milling Patterns", self)
        self.checkBox_show_milling_patterns.setChecked(True)
        self.checkBox_show_milling_patterns.setToolTip("Show the milling patterns in the viewer.")
        self.checkBox_show_milling_patterns.setVisible(False)
        self.checkBox_show_milling_crosshair.setVisible(False)

        self.checkbox_show_advanced = QCheckBox("Show Advanced Settings", self)
        self.checkbox_show_advanced.setChecked(self._show_advanced)
        self.checkbox_show_advanced.setToolTip("Show advanced settings for milling stages.")
        self.checkbox_show_advanced.stateChanged.connect(self.set_show_advanced)
        self.checkbox_show_advanced.setVisible(False)
        self.label_warning = QLabel(self)
        self.label_warning.setText("")
        self.label_warning.setStyleSheet("color: orange; font-style: italic;")
        # callbacks for checkboxes
        # self.checkBox_show_milling_crosshair.stateChanged.connect(self.update_milling_stage_display)
        # self.checkBox_show_milling_patterns.stateChanged.connect(self._toggle_pattern_visibility)

        # grid layout for checkboxes
        self._grid_layout_checkboxes = QGridLayout()
        self._grid_layout_checkboxes.addWidget(self.checkBox_show_milling_patterns, 0, 0, 1, 1)
        self._grid_layout_checkboxes.addWidget(self.checkBox_show_milling_crosshair, 0, 1, 1, 1)
        self._grid_layout_checkboxes.addWidget(self.checkbox_show_advanced, 1, 1, 1, 1)
        self._grid_layout_checkboxes.addWidget(self.label_warning, 2, 0, 1, 2)

        # add widgets to main widget/layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addLayout(self._grid_layout_checkboxes)
        self.main_layout.addWidget(self.list_widget_milling_stages)
        self.main_layout.addWidget(self.milling_stage_content)

        # self.setContentsMargins(0, 0, 0, 0)
        # self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.milling_stage_content.setContentsMargins(0, 0, 0, 0)
        self.milling_stage_layout.setContentsMargins(0, 0, 0, 0)

        # connect signals
        self.list_widget_milling_stages.itemSelectionChanged.connect(self._on_selected_stage_changed)
        self.list_widget_milling_stages.itemChanged.connect(self.update_milling_stage_display)
        self.list_widget_milling_stages.itemChanged.connect(self._on_milling_stage_updated)
        if self.viewer is not None:
            self.viewer.mouse_drag_callbacks.append(self._on_single_click)

        # set initial selection to the first item
        if self.list_widget_milling_stages.count() > 0:
            self.list_widget_milling_stages.setCurrentRow(0)

        self.set_show_advanced(self._show_advanced)

        self._update_empty_state()

    def set_movement_lock(self, locked: bool):
        self.is_movement_locked = locked

    def set_show_advanced(self, show_advanced: bool):
        self._show_advanced = show_advanced
        for widget in self._widgets:
            widget.toggle_advanced_settings(show_advanced)

    def _update_empty_state(self) -> None:
        has_stages = self.list_widget_milling_stages.count() > 0
        self.list_widget_milling_stages.setVisible(has_stages)
        # self.checkbox_show_advanced.setVisible(has_stages)
        self.pushButton_remove.setEnabled(has_stages)

        if has_stages:
            self.label_warning.setStyleSheet("color: orange; font-style: italic;")
            self.label_warning.setText("")
            self.label_warning.setVisible(False)
            return

        self.label_warning.setText("No milling stages defined. Please add a milling stage.")
        self.label_warning.setStyleSheet("color: gray; font-style: italic;")
        self.label_warning.setVisible(True)

    def _toggle_pattern_visibility(self, state: int):
        """Toggle the visibility of milling patterns in the viewer."""
        visible = bool(state == Qt.Checked)
        if self.milling_pattern_layers:
            for layer in self.milling_pattern_layers:
                if layer in self.viewer.layers:
                    self.viewer.layers[layer].visible = visible

    def _reorder_milling_stages(self, parent, start, end, destination, row):
        """Sync the object list when UI is reordered"""
        logging.info(f"Reordering milling stages: start={start}, end={end}, destination={destination}, row={row}")        

        # get
        dest_index = row if row < start else row - (end - start + 1)

        # Move objects in the list
        objects_to_move = self._milling_stages[start:end+1]
        del self._milling_stages[start:end+1]

        for i, obj in enumerate(objects_to_move):
            self._milling_stages.insert(dest_index + i, obj)

        logging.info(f"Objects reordered: {[obj.name for obj in self._milling_stages]}")

        # when we re-order, we need to re-order the widgets as well
        dest_widgets = self._widgets[start:end+1]
        del self._widgets[start:end+1]
        for i, widget in enumerate(dest_widgets):
            self._widgets.insert(dest_index + i, widget)

        self.update_milling_stage_display()
        self._on_milling_stage_updated()

    def _remove_selected_milling_stage(self):
        """Remove the selected milling stage from the list widget."""
        selected_items = self.list_widget_milling_stages.selectedItems()
        if not selected_items:
            logging.info("No milling stage selected for removal.")
            return

        for item in selected_items:
            index = self.list_widget_milling_stages.row(item)
            self.list_widget_milling_stages.takeItem(index)
            # also remove the corresponding widget
            if index < len(self._widgets):
                widget = self._widgets.pop(index)
                widget.deleteLater()

            self._milling_stages.pop(index)  # Remove from the milling stages list
            logging.info(f"Removed item: {item.text()} at index {index}")

        self._on_milling_stage_updated()
        self.update_milling_stage_display()
        self._update_empty_state()

    def clear_milling_stages(self):
        """Clear all milling stages from the editor."""
        self._milling_stages.clear()
        self.list_widget_milling_stages.clear()

        # clear previous widgets
        for widget in self._widgets:
            widget.deleteLater()
        self._widgets.clear()

        self.update_milling_stage_display()
        self._update_empty_state()

    def update_from_settings(self, milling_stages: List[FibsemMillingStage]):
        """Update the editor with the given milling stages.
        Wrapper to match external API.
        """
        self.set_milling_stages(milling_stages)

    def set_milling_stages(self, milling_stages: List[FibsemMillingStage]):
        """Set the milling stages to be displayed in the editor."""

        self.clear_milling_stages()  # Clear existing milling stages
        self._milling_stages = copy.deepcopy(milling_stages)
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)

        # select the first milling stage if available
        if self._milling_stages:
            self.list_widget_milling_stages.setCurrentRow(0)
        self._update_empty_state()

    def set_background_milling_stages(self, milling_stages: List[FibsemMillingStage]):
        """Set the background milling stages to be displayed in the editor."""
        self._background_milling_stages = copy.deepcopy(milling_stages)

    def _update_list_widget_text(self):
        """Update the text of the list widget items to reflect the current milling stages."""
        for i, milling_stage in enumerate(self._milling_stages):
            if i < self.list_widget_milling_stages.count():
                item = self.list_widget_milling_stages.item(i)
                # update the text of the item
                if item:
                    item.setText(milling_stage.pretty_name)

    def _add_milling_stage(self, milling_stage: Optional[FibsemMillingStage] = None):
        """Add a new milling stage to the editor."""
        if milling_stage is None:
            # create a default milling stage if not provided
            num = len(self._milling_stages) + 1
            name = f"Milling Stage {num}"
            # TODO: display alignment area
            # use a copy of the currently selected milling stage, if possible
            current_index = self.list_widget_milling_stages.currentRow()
            if current_index >= 0 and current_index < len(self._milling_stages):
                milling_stage = copy.deepcopy(self._milling_stages[current_index])
                milling_stage.name = name
                milling_stage.num = num
            else:
                milling_stage = FibsemMillingStage(name=name, num=num)

        # Create a new widget for the milling stage
        logging.info(f"Added new milling stage: {milling_stage.name}")
        self._milling_stages.append(milling_stage)  # Add to the milling stages list

        self._add_milling_stage_widget(milling_stage)

        self.list_widget_milling_stages.setCurrentRow(self.list_widget_milling_stages.count() - 1)
        self._on_milling_stage_updated()
        self._update_empty_state()

    def _add_milling_stage_widget(self, milling_stage: FibsemMillingStage):
        """Add a milling stage widget to the editor."""

        # create milling stage widget, connect signals
        ms_widget = FibsemMillingStageWidget(microscope=self.microscope, 
                                            milling_stage=milling_stage)
        ms_widget._milling_stage_changed.connect(self.update_milling_stage_display)
        ms_widget._milling_stage_changed.connect(self._on_milling_stage_updated)
        ms_widget._milling_stage_changed.connect(self._update_list_widget_text)

        # create related list widget item
        # TODO: migrate to setData, so we can store the milling stage object directly
        item = QListWidgetItem(milling_stage.pretty_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.list_widget_milling_stages.addItem(item)

        # add the widgets
        self.milling_stage_layout.addWidget(ms_widget)
        self._widgets.append(ms_widget)

    def _on_selected_stage_changed(self):
        """Handle the selection change in the list widget."""
        selected_items = self.list_widget_milling_stages.selectedItems()
        if not selected_items:
            # hide all widgets
            for widget in self._widgets:
                widget.hide()

        # hide all widgets, except selected (only single-selection supported)
        index = self.list_widget_milling_stages.currentRow()
        for i, widget in enumerate(self._widgets):
            widget.setVisible(i==index)

        self._widgets[index].toggle_advanced_settings(self._show_advanced)

        # refresh display
        self.update_milling_stage_display()

    def _get_selected_milling_stages(self) -> List[FibsemMillingStage]:
        """Return the milling stages that are selected (checked) in the list widget."""
        checked_indexes = []
        for i in range(self.list_widget_milling_stages.count()):
            item = self.list_widget_milling_stages.item(i)
            if item.checkState() == Qt.Checked:
                checked_indexes.append(i)

        milling_stages = [
            widget.get_milling_stage()
            for i, widget in enumerate(self._widgets)
            if i in checked_indexes
        ]
        return milling_stages

    def get_milling_stages(self) -> List[FibsemMillingStage]:
        """Public method to get the currently selected milling stages."""
        return self._get_selected_milling_stages()

    @property
    def selected_stage_name(self) -> Optional[str]:
        """Return the name of the currently selected milling stage."""
        selected_idx = self.list_widget_milling_stages.currentRow()
        if selected_idx >= 0 and selected_idx < len(self._milling_stages):
            return self._milling_stages[selected_idx].name
        return None

    def update_milling_stage_display(self):
        """Update the display of milling stages in the viewer."""
        if self.is_updating_pattern:
            return # block updates while updating patterns

        if self.viewer is None or self.image_layer is None:
            return

        milling_stages = self.get_milling_stages()

        if not milling_stages:
            try:
                for layer in self.milling_pattern_layers:
                    if layer in self.viewer.layers:
                        self.viewer.layers.remove(layer)
                if "Milling Alignment Area" in self.viewer.layers:
                    self.viewer.layers.remove("Milling Alignment Area") # type: ignore
            except Exception as e:
                logging.debug(f"Error removing milling pattern layers: {e}")
            self.milling_pattern_layers = []
            return

        if self.image is None:
            image = FibsemImage.generate_blank_image(hfw=milling_stages[0].milling.hfw)
            self.set_image(image)

        if self.image is None or self.image.metadata is None:
            raise ValueError("Image metadata is not set. Cannot update milling stage display.")

        self._validate_image_field_of_view()

        try:
            alignment_area = self.parent_widget.alignment_widget._settings.rect
        except Exception:
            alignment_area = None

        # logging.info(f"Selected milling stages: {[stage.name for stage in milling_stages]}")
        # logging.info(f"Background milling stages: {[stage.name for stage in self._background_milling_stages]}")
        # logging.info(f"Updating milling stage display with image HFW: {self.image.metadata.image_settings.hfw*1e6} um and pixel size: {self.image.metadata.pixel_size.x} m")

        selected_index = self.list_widget_milling_stages.currentRow()
        if selected_index < 0:
            selected_index = None

        self.milling_pattern_layers = draw_milling_patterns_in_napari(
            viewer=self.viewer,
            image_layer=self.image_layer,
            milling_stages=milling_stages,
            pixelsize=self.image.metadata.pixel_size.x,
            draw_crosshair=self.checkBox_show_milling_crosshair.isChecked(),
            background_milling_stages=self._background_milling_stages,
            alignment_area=alignment_area, # NOTE: we need to update this for each milling task, rather than read from lamella.alignment_area
            selected_index=selected_index,
        )

        if self.image_widget is not None:
            self.image_widget.restore_active_layer_for_movement()

    def _validate_image_field_of_view(self):
        """Validate that the milling task and displayed image have the same field of view."""
        try:
            milling_fov = self.parent_widget._settings.field_of_view
            image_fov = self.image.metadata.image_settings.hfw
            msg = ""
            if not np.isclose(milling_fov, image_fov):
                milling_fov_um = utils.format_value(milling_fov, unit='m', precision=0)
                image_fov_um = utils.format_value(image_fov, unit='m', precision=0)
                msg = f"Milling Task FoV ({milling_fov_um}), is not the same as Image FoV ({image_fov_um})"
                self.label_warning.setVisible(True)
            else: 
                self.label_warning.setVisible(False)
            self.label_warning.setText(msg)
        except Exception as e:
            logging.warning(f"An error occured while checking the field of view of milling and image: {e}")

    def set_image(self, image: FibsemImage) -> None:
        """Set the image for the milling stage editor."""
        if self.viewer is None:
            return

        self.image = image
        try:
            self.image_layer.data = image.filtered_data # type: ignore
        except Exception as e:
            self.image_layer = self.viewer.add_image(name="FIB Image", data=image.filtered_data, opacity=0.7) # type: ignore
        self.update_milling_stage_display()

    @property
    def is_correlation_open(self) -> bool:
        """Check if correlation tool is opened"""
        if self.parent_widget is not None and hasattr(self.parent_widget, "correlation_widget"):
            correlation_widget = self.parent_widget.correlation_widget
            if correlation_widget is not None and correlation_widget.isVisible():
                return True
        return False

    def _on_single_click(self, viewer: napari.Viewer, event):
        """Handle single click events to move milling patterns.

        Shift+Click: Move selected pattern only
        Shift+Control+Click: Move all patterns (maintaining relative positions)
        """
        if event.button != 1 or 'Shift' not in event.modifiers or self._milling_stages == []:
            return

        if not self.image_layer:
            logging.warning("No target layer found for the click event.")
            return

        if self.is_movement_locked:
            logging.warning("Movement is locked. Cannot move milling patterns.")
            return

        if self.is_correlation_open:
            logging.info("Correlation tool is open, ignoring click event.")
            return

        if not is_position_inside_layer(event.position, self.image_layer):
            logging.warning("Click position is outside the image layer.")
            return

        current_idx = self.list_widget_milling_stages.currentRow()

        if current_idx < 0 or current_idx >= len(self._milling_stages):
            logging.warning("No milling stage selected or index out of range.")
            return

        if self.image is None or self.image.metadata is None:
            logging.warning("Image metadata is not set. Cannot convert coordinates.")
            return

        # convert from image coordinates to microscope coordinates
        coords = self.image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        # Control modifier: move all patterns, otherwise move only selected
        move_all = bool('Control' in event.modifiers)
        self.move_patterns_to_point(point_clicked, move_all=move_all)

    def _on_milling_stage_updated(self, milling_stage: Optional[FibsemMillingStage] = None):
        """Callback when a milling stage is updated."""

        # If we are currently updating the pattern, we don't want to emit the signal
        if self.is_updating_pattern:
            return

        milling_stages = self.get_milling_stages()
        self._milling_stages_updated.emit(milling_stages)

    def move_patterns_to_point(self, point: Point, move_all: bool = True) -> bool:
        """Move milling patterns to a specified point.

        Args:
            point: The target point in microscope coordinates.
            move_all: If True, move all patterns relative to the selected pattern.
                      If False, only move the currently selected pattern.

        Returns:
            True if patterns were successfully moved, False otherwise.
        """
        if not self._milling_stages:
            logging.warning("No milling stages to move.")
            return False

        current_idx = self.list_widget_milling_stages.currentRow()
        if current_idx < 0 or current_idx >= len(self._milling_stages):
            current_idx = 0

        if self.image is None or self.image.metadata is None:
            logging.warning("Image metadata is not set. Cannot validate pattern placement.")
            return False

        # calculate the difference between the target point and the current pattern point
        diff = point - self._milling_stages[current_idx].pattern.point

        # validate that patterns will be within bounds after moving
        new_points: List[Tuple[int, Point]] = []  # (index, new_point)
        for idx, milling_stage in enumerate(self._milling_stages):
            if not move_all and idx != current_idx:
                continue

            pattern_copy = copy.deepcopy(milling_stage.pattern)

            # handle LinePattern special case
            if isinstance(pattern_copy, LinePattern):
                pattern_copy.start_x += diff.x
                pattern_copy.start_y += diff.y
                pattern_copy.end_x += diff.x
                pattern_copy.end_y += diff.y

            new_point = pattern_copy.point + diff
            pattern_copy.point = new_point

            if not is_pattern_placement_valid(pattern=pattern_copy, image=self.image):
                msg = f"{milling_stage.name} pattern would be outside the FIB image."
                logging.warning(msg)
                napari.utils.notifications.show_warning(msg)
                return False

            new_points.append((idx, new_point))

        # apply the movement to patterns
        self.is_updating_pattern = True
        for idx, new_point in new_points:
            self._widgets[idx].set_point(new_point)

            # handle LinePattern special case
            pattern = self._milling_stages[idx].pattern
            if isinstance(pattern, LinePattern):
                pattern.start_x += diff.x
                pattern.start_y += diff.y
                pattern.end_x += diff.x
                pattern.end_y += diff.y

        self.is_updating_pattern = False
        self._on_milling_stage_updated()
        self.update_milling_stage_display()

        num_moved = len(new_points)
        logging.info(f"Moved {num_moved} pattern(s) to {point}")
        return True
