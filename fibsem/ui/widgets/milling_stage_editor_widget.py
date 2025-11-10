import copy
import logging
import os
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
)
from superqt import QCollapsible

from fibsem import config as cfg
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
    CrossSectionPattern,
    Enum,
    FibsemImage,
    Point,
)
from fibsem.ui import stylesheets
from fibsem.ui.FibsemMillingWidget import WheelBlocker
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    is_pattern_placement_valid,
)
from fibsem.ui.napari.utilities import is_position_inside_layer
from fibsem.utils import format_value

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget

MILLING_SETTINGS_GUI_CONFIG = {
    "patterning_mode": {
        "label": "Patterning Mode",
        "type": str,
        "items": ["Serial", "Parallel"],
        "tooltip": "The mode of patterning used for milling.",
    },
    "milling_current": {
        "label": "Milling Current",
        "units": "A",
        "tooltip": "The current used for milling.",
        "items": "dynamic",
    },
    "milling_voltage": {
        "label": "Milling Voltage",
        "units": "V",
        "items": "dynamic",
        "tooltip": "The voltage used for milling.",
    },
    "hfw": {
        "label": "Field of View",
        "type": float,
        "units": "µm",
        "scale": 1e6,
        "default": 150.0,
        "minimum": 20.0,
        "maximum": 950.0,
        "step": 10.0,
        "decimals": 2,
        "tooltip": "The horizontal field width (fov) for milling.",
    },
    "application_file": {
        "label": "Application File",
        "type": str,
        "items":  "dynamic",
        "tooltip": "The ThermoFisher application file for milling.",
    },
    "acquire_images": {
        "label": "Acquire After Milling",
        "type": bool,
        "default": True,
        "tooltip": "Acquire images after milling.",
    },
    "rate": {
        "label": "Milling Rate",
        "units": "mm3/s",
        "scale": 1e9,
        "tooltip": "The milling rate in mm³/s.",
    },
    "preset": {
        "label": "Milling Preset",
        "type": str,
        "items": "dynamic",
        "tooltip": "The preset for milling parameters.",
    },
    "dwell_time": {
        "label": "Dwell Time",
        "units": "us",
        "scale": 1e6,
        # "default": 0.1,
        # "minimum": 0.01,
        # "maximum": 10.0,
        # "step": 0.01,
        "decimals": 2,
        "tooltip": "The dwell time for each point in the milling pattern.",
    },
    "spot_size": {
        "label": "Spot Size",
        "type": float,
        "units": "um",
        "scale": 1e6,
        # "default": 10.0,
        # "minimum": 1.0,
        # "maximum": 100.0,
        # "step": 1.0,
        "decimals": 2,
        "tooltip": "The spot size for the ion beam during milling.",
    },
}

MILLING_PATTERN_GUI_CONFIG = {
    "width": {
        "label": "Width",
        "tooltip": "The width of the milling pattern.",
    },
    "height": {
        "label": "Height",
        "tooltip": "The height of the milling pattern.",
    },
    "depth": {
        "label": "Depth",
        "tooltip": "The depth of the milling pattern.",
    },
    "rotation": {
        "label": "Rotation",
        "type": float,
        "scale": None,
        "units": "°",
        "minimum": 0.0,
        "maximum": 360.0,
        "step": 1.0,
        "tooltip": "The rotation angle of the milling pattern.",
    },
    "time": {
        "label": "Time",
        "units": "s",
        "scale": None,
        "tooltip": "The time for which the milling pattern will be applied.",
    },
    "cross_section": {
        "label": "Cross Section",
        "type": CrossSectionPattern,
        "items": [cs for cs in CrossSectionPattern],
        "tooltip": "The type of cross section for the milling pattern.",
    },
    "scan_direction": {
        "label": "Scan Direction",
        "type": str,
        "items": "dynamic",
        "tooltip": "The scan direction for the milling pattern.",
    },
    "upper_trench_height": {
        "label": "Upper Trench Height",
        "tooltip": "The height of the upper trench in the milling pattern.",
    }, 
    "lower_trench_height": {
        "label": "Lower Trench Height",
        "tooltip": "The height of the lower trench in the milling pattern.",
    },
    "fillet": {
        "label": "Fillet",
        "tooltip": "The fillet radius for the milling pattern.",
    },
    "spacing": {
        "label": "Spacing",
        "tooltip": "The spacing between the trenches in the milling pattern.",
    },
    "side_width": {
        "label": "Side Width",
        "tooltip": "The width of the sides in the milling pattern.",
    },
    "passes": {
        "label": "Passes",
        "scale": None,
        "units": "",
        "tooltip": "The number of passes for the milling pattern.",
    },
    "n_rows": {
        "label": "Rows",
        "type": int,
        "units": "",
        "minimum": 1,
        "maximum": 100,
        "step": 1,
        "scale": None,
        "tooltip": "The number of rows in the array.",
    },
    "n_columns": {
        "label": "Columns",
        "type": int,
        "units": "",
        "minimum": 1,
        "maximum": 100,
        "step": 1,
        "scale": None,
        "tooltip": "The number of columns in the array.",
    },
}

MILLING_STRATEGY_GUI_CONFIG = {
    "overtilt": {
        "label": "Overtilt",
        "type": float,
        "units": "°",
        "scale": None,
        "minimum": 0.0,
        "maximum": 10,
        "step": 1.0,
        "decimals": 2,
        "tooltip": "The overtilt angle for the milling strategy.",
    },  
    "resolution": {
        "label": "Resolution",
        "type": List[int],
        "items": cfg.STANDARD_RESOLUTIONS_LIST,
        "tooltip": "The imaging resolution for the milling strategy.",
    },
}

MILLING_ALIGNMENT_GUI_CONFIG = {
    "enabled": {
        "label": "Initial Alignment",
        "type": bool,
        "default": True,
        "tooltip": "Enable initial milling alignment between imaging and milling current.",
    },
}

MILLING_IMAGING_GUI_CONFIG = {
    "resolution": {
        "label": "Image Resolution",
        "type": List[int],
        "items": cfg.STANDARD_RESOLUTIONS_LIST,
        "tooltip": "The resolution for the acquired images.",
    },
    "hfw": {
        "label": "Horizontal Field Width",
        "units": "µm",
        "scale": 1e6,
        "tooltip": "The horizontal field width for the acquired images.",
    },
    "dwell_time": {
        "label": "Dwell Time",
        "units": "µs",
        "scale": 1e6,
        "tooltip": "The dwell time for each pixel in the acquired images.",
    },
    "autocontrast": {
        "label": "Autocontrast",
        "type": bool,
        "default": True,
        "tooltip": "Enable autocontrast for the acquired images.",
    },
}

DEFAULT_PARAMETERS: Dict[str, Any] = {
    "type": float,
    "units": "µm",
    "scale": 1e6,
    "minimum": 0.0,
    "maximum": 1000.0,
    "step": 0.01,
    "decimals": 2,
    "tooltip": "Default parameter for milling settings.",
}

GUI_CONFIG: Dict[str, Dict] = {
    "milling": MILLING_SETTINGS_GUI_CONFIG,
    "pattern": MILLING_PATTERN_GUI_CONFIG,
    "strategy": MILLING_STRATEGY_GUI_CONFIG,
    "alignment": MILLING_ALIGNMENT_GUI_CONFIG,
    "imaging": MILLING_IMAGING_GUI_CONFIG,}

# mapping from milling settings to microscope parameters
PARAMETER_MAPPING = {
    "milling_current": "current",
    "milling_voltage": "voltage",
}


# MILLING_TASK:
#   MILLING_ALIGNMENT
#   MILLING_ACQUISITION
#   MILLING_STAGE_1:
#       MILLING_SETTINGS
#       MILLING_PATTERN
#       MILLING_STRATEGY
#   MILLING_STAGE_2:
#       ...

# TODO: 
# what to do when no microscope available???
# milling stage name?


class FibsemMillingStageWidget(QWidget):
    _milling_stage_changed = pyqtSignal(FibsemMillingStage)

    def __init__(self, 
                 microscope: FibsemMicroscope, # TODO: don't require this!, but if its available, use it for dynamic items 
                 milling_stage: FibsemMillingStage, 
                 manufacturer: str = "TFS", 
                 parent=None):
        super().__init__(parent)

        self.parameters: Dict[str, Dict[str, Tuple[QLabel, QWidget, float]]] = {} # param: label, control, scale
        self.microscope = microscope
        self._milling_stage = milling_stage
        self._manufacturer = manufacturer  # Manufacturer for dynamic items (TFS, TESCAN)

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
        label = QLabel(self)
        label.setText("Name")
        self.comboBox_selected_pattern = QComboBox(self)
        self.comboBox_selected_pattern.addItems(MILLING_PATTERN_NAMES)
        self.wheel_blocker1 = WheelBlocker()
        self.comboBox_selected_pattern.installEventFilter(self.wheel_blocker1)
        self.comboBox_selected_pattern.currentTextChanged.connect(self._on_pattern_changed)
        self.pattern_widget.layout().addWidget(label, 0, 0, 1, 1)
        self.pattern_widget.layout().addWidget(self.comboBox_selected_pattern, 0, 1, 1, 1)

        # create strategy widget
        label = QLabel(self)
        label.setText("Name")
        self.comboBox_selected_strategy = QComboBox(self)
        self.strategy_widget.layout().addWidget(label, 0, 0, 1, 1)
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
        label.setText("Milling Stage:")
        label.setObjectName("label-milling-stage-name")
        self.lineEdit_milling_stage_name = QLineEdit(self)
        self.lineEdit_milling_stage_name.setText(self._milling_stage.name)
        self.lineEdit_milling_stage_name.setObjectName("lineEdit-name-stage")
        self.lineEdit_milling_stage_name.setToolTip("The name of the milling stage.")
        self.lineEdit_milling_stage_name.editingFinished.connect(self._update_setting)
        # TODO: move milling stage name into milling settings widget?
        self.gridlayout.addWidget(label, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.lineEdit_milling_stage_name, 0, 1, 1, 1)
        for title, widget in zip(["Milling", "Pattern", "Strategy"], self._widgets):
            collapsible = QCollapsible(title, parent=self)
            collapsible.addWidget(widget)
            self.gridlayout.addWidget(collapsible, self.gridlayout.rowCount(), 0, 1, 2) # type: ignore

    def _initialise_widgets(self):
        """Initialise the widgets with the current milling stage settings."""
        # MILLING SETTINGS
        milling_parames = self._milling_stage.milling.get_parameters(self._manufacturer)
        self._create_controls(self.milling_widget, milling_parames, "milling", GUI_CONFIG["milling"].copy())

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
        # self.strategy_widget.setVisible(show)

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
        # TODO: convert the comboBox_selected_pattern to use currentData, 
        # that way we can pass the pattern object directly (and restore it from the previous state)
        pattern = get_pattern(pattern_name)

        self._milling_stage.pattern = pattern  # Update the milling stage's pattern

        self._update_pattern_widget(pattern)
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes

    def _update_pattern_widget(self, pattern: BasePattern):
        """Update the pattern widget with the selected pattern's parameters."""

        params = {k: getattr(pattern, k) for k in pattern.required_attributes if hasattr(pattern, k)}
        params["point"] = pattern.point  # add point as a special case

        self._create_controls(self.pattern_widget, params, "pattern", GUI_CONFIG["pattern"].copy())

    def _on_strategy_changed(self, strategy_name: str):
        """Update the strategy widget with the selected strategy's parameters."""
        strategy = get_strategy(strategy_name, {"config": {}})

        # update strategy and widget
        self._milling_stage.strategy = strategy
        self._update_strategy_widget(strategy)
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes

    def _update_strategy_widget(self, strategy: MillingStrategy[Any]):
        """Update the strategy widget with the selected strategy's parameters."""
        params = {k: getattr(strategy.config, k) for k in strategy.config.required_attributes}

        self._create_controls(self.strategy_widget, params, "strategy.config", GUI_CONFIG["strategy"].copy())

    def _create_controls(self, widget: QWidget, params: Dict[str, Any], cls: str, config: Dict[str, Any]):
        """Create controls for the given parameters and add them to the widget."""

        # clear previous controls
        if cls == "pattern":
            self.clear_widget(self.pattern_widget, row_threshold=0)
        if cls == "strategy.config":
            self.clear_widget(self.strategy_widget, row_threshold=0)

        self.parameters[cls] = {}
        grid_layout = widget.layout()
        if grid_layout is None:
            raise ValueError(f"Widget {widget} does not have a layout. Expected QGridLayout, got {type(grid_layout)}.")
        if not isinstance(grid_layout, QGridLayout):
            raise TypeError(f"Expected QGridLayout, got {type(grid_layout)} for widget {widget}.")

        # point controls (special case). but why do they have to be?
        if cls == "pattern":
            gui_config = config.get("point", {})
            label_text = gui_config.get("label", "Point")
            minimum = gui_config.get("minimum", DEFAULT_PARAMETERS["minimum"])
            maximum = gui_config.get("maximum", DEFAULT_PARAMETERS["maximum"])
            step_size   = gui_config.get("step", DEFAULT_PARAMETERS["step"])
            units = gui_config.get("units", DEFAULT_PARAMETERS["units"])
            scale = gui_config.get("scale", DEFAULT_PARAMETERS["scale"])
            decimals = gui_config.get("decimals", DEFAULT_PARAMETERS["decimals"])

            # points are a special case? 
            pt_label = QLabel(self)
            pt_label.setText(label_text)
            pt_label.setObjectName(f"label-{cls}-point")
            pt_label.setToolTip(gui_config.get("tooltip", "Point coordinates for the milling pattern."))

            hbox_layout = QHBoxLayout()
            for attr in ["x", "y"]:
                # create double spin boxes for point coordinates
                control = QDoubleSpinBox(self)
                control.setSuffix(f" {units}")
                control.setRange(-1000, 1000)
                control.setSingleStep(step_size)
                value = getattr(params["point"], attr)
                if scale is not None:
                    value *= scale
                control.setValue(value)
                control.setObjectName(f"control-pattern-point.{attr}")
                control.setKeyboardTracking(False)
                control.setDecimals(decimals)
                control.valueChanged.connect(self._update_setting)

                self.parameters[cls][f"point.{attr}"] = (pt_label, control, scale)
                hbox_layout.addWidget(control)

            # add both point controls to widget, set the padding to 0 to match other visual
            point_widget = QWidget(self)
            point_widget.setObjectName(f"point-widget-{cls}")
            point_widget.setToolTip(gui_config.get("tooltip", "Point coordinates for the milling pattern."))
            hbox_layout.setContentsMargins(0, 0, 0, 0)
            point_widget.setContentsMargins(0, 0, 0, 0)
            point_widget.setLayout(hbox_layout)

            # add to the grid layout
            row = grid_layout.rowCount()
            grid_layout.addWidget(pt_label, row, 0, 1, 1)
            grid_layout.addWidget(point_widget, row, 1, 1, 1)

        for name, value in params.items():

            if cls == "milling" and name in ["milling_channel", "hfw", "acquire_images"]:
                continue

            # get the GUI configuration for the parameter
            gui_config: Dict[str, Any] = config.get(name, {})
            label_text = gui_config.get("label", name.replace("_", " ").title())
            scale = gui_config.get("scale", DEFAULT_PARAMETERS["scale"])
            units = gui_config.get("units", DEFAULT_PARAMETERS["units"])
            minimum = gui_config.get("minimum", DEFAULT_PARAMETERS["minimum"])
            maximum = gui_config.get("maximum", DEFAULT_PARAMETERS["maximum"])
            step_size = gui_config.get("step", DEFAULT_PARAMETERS["step"])
            decimals = gui_config.get("decimals", DEFAULT_PARAMETERS["decimals"])
            items = gui_config.get("items", [])

            # set label text
            label = QLabel(label_text)

            # add combobox controls
            if items:
                if items == "dynamic":
                    items = self.microscope.get_available_values(PARAMETER_MAPPING.get(name, name), BeamType.ION)

                control = QComboBox()
                for item in items:
                    if isinstance(item, (float, int)):
                        item_str = format_value(val=item,
                                                unit=units,
                                                precision=gui_config.get("decimals", 1))
                    elif isinstance(item, Enum):
                        item_str = item.name # TODO: migrate to QEnumComboBox
                    elif "resolution" in name:
                        item_str = f"{item[0]}x{item[1]}"
                    else:
                        item_str = str(item)
                    control.addItem(item_str, item)

                if isinstance(value, tuple) and len(value) == 2:
                    value = list(value)  # Convert tuple to list for easier handling

                # find the closest match to the current value (should only be used for numerical values)
                idx = control.findData(value)
                if idx == -1:
                    # get the closest value
                    closest_value = min(items, key=lambda x: abs(x - value))
                    idx = control.findData(closest_value)
                if idx == -1:
                    logging.debug(f"Warning: No matching item or nearest found for {name} with value {value}. Using first item.")
                    idx = 0
                control.setCurrentIndex(idx)

            # add line edit controls
            elif isinstance(value, str):
                control = QLineEdit()
                control.setText(value)
            # add checkbox controls
            elif isinstance(value, bool):
                control = QCheckBox()
                control.setChecked(value)
            elif isinstance(value, (float, int)):

                control = QDoubleSpinBox()
                if units is not None:
                    control.setSuffix(f' {units}')
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
            else:
                continue

            # Set tooltip for both label and control
            if tooltip := gui_config.get("tooltip"):
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)

            grid_layout.addWidget(label, grid_layout.rowCount(), 0)
            grid_layout.addWidget(control, grid_layout.rowCount() - 1, 1)

            label.setObjectName(f"label-{cls}-{name}")
            control.setObjectName(f"control-{cls}-{name}")
            self.parameters[cls][name] = (label, control, scale)

            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._update_setting)
            elif isinstance(control, QLineEdit):
                control.textChanged.connect(self._update_setting)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._update_setting)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self._update_setting)

    # add callback to update settings when control value changes
    def _update_setting(self):
        obj = self.sender()
        if not obj:
            return
        obj_name = obj.objectName()
        _, cls, name = obj_name.split("-", 2)

        if isinstance(obj, QComboBox):
            value = obj.currentData()
        elif isinstance(obj, QLineEdit):
            value = obj.text()
        elif isinstance(obj, QCheckBox):
            value = obj.isChecked()
        elif isinstance(obj, (QSpinBox, QDoubleSpinBox)):
            value = obj.value()
            # apply scale if defined
            scale = self.parameters[cls][name][2]
            if scale is not None:
                value /= scale
        else:
            return

        # update the milling_stage object
        if hasattr(self._milling_stage, cls):

            # special case for pattern point
            if "point" in name:
                if "x" in name:
                    setattr(self._milling_stage.pattern.point, "x", value)
                elif "y" in name:
                    setattr(self._milling_stage.pattern.point, "y", value)
            elif cls == "name":
                setattr(self._milling_stage, "name", value)
            else:
                setattr(getattr(self._milling_stage, cls), name, value)
        elif hasattr(self._milling_stage, "strategy") and cls == "strategy.config":
            # Special case for strategy config
            setattr(self._milling_stage.strategy.config, name, value)
        else:
            logging.debug(f"Warning: {cls} not found in milling_stage object. Cannot update {name}.")

        self._milling_stage_changed.emit(self._milling_stage)  # notify changes

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
        self.image = None
        self.image_layer = None
        if self.parent_widget is not None:
            if self.parent_widget.parent() is not None:
                super_parent = self.parent_widget.parent()
                if hasattr(super_parent, "image_widget"):
                    self.image = super_parent.image_widget.ib_image
                    self.image_layer = super_parent.image_widget.ib_layer
        if self.image is None:
            self.image: FibsemImage = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
            if self.viewer is not None:
                self.image_layer: NapariImageLayer = self.viewer.add_image(data=self.image.data, name="FIB Image") # type: ignore
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
        self.main_layout.addWidget(self.milling_stage_content)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addLayout(self._grid_layout_checkboxes)
        self.main_layout.addWidget(self.list_widget_milling_stages)

        self.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
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

    def set_movement_lock(self, locked: bool):
        self.is_movement_locked = locked

    def set_show_advanced(self, show_advanced: bool):
        self._show_advanced = show_advanced
        for widget in self._widgets:
            widget.toggle_advanced_settings(show_advanced)

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

    def clear_milling_stages(self):
        """Clear all milling stages from the editor."""
        self._milling_stages.clear()
        self.list_widget_milling_stages.clear()

        # clear previous widgets
        for widget in self._widgets:
            widget.deleteLater()
        self._widgets.clear()

        self.update_milling_stage_display()

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
            except Exception as e:
                logging.debug(f"Error removing milling pattern layers: {e}")
            self.milling_pattern_layers = []
            return

        if self.image is None:
            image = FibsemImage.generate_blank_image(hfw=milling_stages[0].milling.hfw)
            self.set_image(image)

        if self.image.metadata is None:
            raise ValueError("Image metadata is not set. Cannot update milling stage display.")

        self._validate_image_field_of_view()

        # logging.info(f"Selected milling stages: {[stage.name for stage in milling_stages]}")
        # logging.info(f"Background milling stages: {[stage.name for stage in self._background_milling_stages]}")
        # logging.info(f"Updating milling stage display with image HFW: {self.image.metadata.image_settings.hfw*1e6} um and pixel size: {self.image.metadata.pixel_size.x} m")

        self.milling_pattern_layers = draw_milling_patterns_in_napari(
            viewer=self.viewer,
            image_layer=self.image_layer,
            milling_stages=milling_stages,
            pixelsize=self.image.metadata.pixel_size.x,
            draw_crosshair=self.checkBox_show_milling_crosshair.isChecked(),
            background_milling_stages=self._background_milling_stages,
        )

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
            self.image_layer.data = image.data # type: ignore
        except Exception as e:
            self.image_layer = self.viewer.add_image(name="FIB Image", data=image.data, opacity=0.7) # type: ignore
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
        """Handle single click events to move milling patterns."""
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

        if self.image.metadata is None:
            logging.warning("Image metadata is not set. Cannot convert coordinates.")
            return

        # convert from image coordinates to microscope coordinates
        coords = self.image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        # conditions to move:
        #   all moved patterns are within the fib image
        new_points: List[Point] = []
        has_valid_patterns: bool = True
        is_moving_all_patterns: bool = bool('Control' in event.modifiers)
        use_relative_movement: bool = True

        # calculate the difference between the clicked point and the current pattern point (used for relative movement)
        diff = point_clicked - self._milling_stages[current_idx].pattern.point

        # loop to check through all patterns to see if they are in bounds
        for idx, milling_stage in enumerate(self._milling_stages):
            if not is_moving_all_patterns:
                if idx != current_idx:
                    continue

            pattern_renew = copy.deepcopy(milling_stage.pattern)

            # special case: if the pattern is a line, we also need to update start_x, start_y, end_x, end_y to move with the click
            if isinstance(pattern_renew, LinePattern):
                pattern_renew.start_x += diff.x
                pattern_renew.start_y += diff.y
                pattern_renew.end_x += diff.x
                pattern_renew.end_y += diff.y

                # TODO: resolve line special cases
                # this doesnt work if the line is rotated at all
                # if the line goes out of bounds it is not reset correctly, need to fix this

            # update the pattern point
            point = pattern_renew.point + diff if use_relative_movement else point_clicked
            pattern_renew.point = point
            
            # test if the pattern is within the image bounds
            if not is_pattern_placement_valid(pattern=pattern_renew, image=self.image):
                has_valid_patterns = False
                msg = f"{milling_stage.name} pattern is not within the FIB image."
                logging.warning(msg)
                napari.utils.notifications.show_warning(msg)
                break
            # otherwise, add the new point to the list
            new_points.append(copy.deepcopy(point))

        if has_valid_patterns:
    
            # block redraw until all patterns are updated
            self.is_updating_pattern = True
            if is_moving_all_patterns:
                for idx, new_point in enumerate(new_points):
                    self._widgets[idx].set_point(new_point)
            else: # only moving selected pattern
                self._widgets[current_idx].set_point(point_clicked)

        self.is_updating_pattern = False
        self._on_milling_stage_updated()
        self.update_milling_stage_display()  # force refresh the milling stages display

    def _on_milling_stage_updated(self, milling_stage: Optional[FibsemMillingStage] = None):
        """Callback when a milling stage is updated."""

        # If we are currently updating the pattern, we don't want to emit the signal
        if self.is_updating_pattern:
            return

        milling_stages = self.get_milling_stages()
        self._milling_stages_updated.emit(milling_stages)
