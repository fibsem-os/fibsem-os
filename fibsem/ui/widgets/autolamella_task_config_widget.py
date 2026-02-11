import inspect
import logging
from dataclasses import fields
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)
from superqt import QCollapsible

from fibsem import utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget


class ParameterWidget:
    """Base class for parameter editing widgets."""
    
    def __init__(self, name: str, value: Any, annotation: type):
        self.name = name
        self.value = value
        self.annotation = annotation
        self.widget = None
        
    def create_widget(self) -> QWidget:
        """Create the appropriate widget for this parameter type."""
        raise NotImplementedError
        
    def get_value(self) -> Any:
        """Get the current value from the widget."""
        raise NotImplementedError
        
    def set_value(self, value: Any) -> None:
        """Set the value in the widget."""
        raise NotImplementedError


class BoolParameterWidget(ParameterWidget):
    """Widget for boolean parameters."""
    
    def create_widget(self) -> QWidget:
        self.widget = QCheckBox()
        self.widget.setChecked(bool(self.value))
        return self.widget
        
    def get_value(self) -> bool:
        return self.widget.isChecked()

    def set_value(self, value: bool) -> None:
        self.widget.setChecked(bool(value))


class IntParameterWidget(ParameterWidget):
    """Widget for integer parameters."""
    
    def create_widget(self) -> QWidget:
        self.widget = QSpinBox()
        self.widget.setRange(-2147483648, 2147483647)  # 32-bit int range
        self.widget.setValue(int(self.value))
        self.widget.setKeyboardTracking(False)
        return self.widget

    def get_value(self) -> int:
        return self.widget.value()

    def set_value(self, value: int) -> None:
        self.widget.setValue(int(value))


def get_si_prefix_suffix(scale: float, units: str) -> str:
    """Get the appropriate SI prefix suffix based on scale factor."""
    si_prefixes = {
        1e12: 'p',    # pico
        1e9: 'n',     # nano
        1e6: 'μ',     # micro
        1e3: 'm',     # milli
        1.0: '',      # no prefix
        1e-3: 'k',    # kilo
        1e-6: 'M',    # mega
        1e-9: 'G',    # giga
        1e-12: 'T',   # tera
    }
    
    prefix = si_prefixes.get(scale, '')
    return f" {prefix}{units}" if units else ""


class FloatParameterWidget(ParameterWidget):
    """Widget for float parameters with units and scaling support."""
    
    def __init__(self, name: str, value: Any, annotation: type, metadata: Optional[dict] = None):
        super().__init__(name, value, annotation)
        self.metadata = metadata or {}
        self.scale = self.metadata.get('scale', 1.0)
        self.units = self.metadata.get('units', '')
        
    def create_widget(self) -> QWidget:
        self.widget = QDoubleSpinBox()
        self.widget.setRange(-1e10, 1e10)
        self.widget.setDecimals(2)
        self.widget.setKeyboardTracking(False)
        
        # Apply scaling for display (multiply by scale to show user-friendly values)
        display_value = float(self.value) * self.scale
        self.widget.setValue(display_value)
        
        # Add units suffix if available
        suffix = get_si_prefix_suffix(self.scale, self.units)
        if suffix:
            self.widget.setSuffix(suffix)
        
        return self.widget
        
    def get_value(self) -> float:
        # Convert from display value back to stored value (divide by scale)
        display_value = self.widget.value()
        return display_value / self.scale
        
    def set_value(self, value: float) -> None:
        # Convert to display value (multiply by scale)
        display_value = float(value) * self.scale
        self.widget.setValue(display_value)


class StringParameterWidget(ParameterWidget):
    """Widget for string parameters."""
    
    def create_widget(self) -> QWidget:
        self.widget = QLineEdit()
        self.widget.setText(str(self.value))
        return self.widget
        
    def get_value(self) -> str:
        return self.widget.text()
        
    def set_value(self, value: str) -> None:
        self.widget.setText(str(value))


class EnumParameterWidget(ParameterWidget):
    """Widget for enum parameters."""
    
    def create_widget(self) -> QWidget:
        self.widget = QComboBox()
        
        # Add enum values to combo box
        for enum_value in self.annotation:
            self.widget.addItem(str(enum_value.value), enum_value)
            
        # Set current value
        current_index = 0
        for i, enum_value in enumerate(self.annotation):
            if enum_value == self.value or enum_value.value == self.value:
                current_index = i
                break
        self.widget.setCurrentIndex(current_index)
        
        return self.widget
        
    def get_value(self) -> Any:
        return self.widget.currentData()
        
    def set_value(self, value: Any) -> None:
        for i in range(self.widget.count()):
            if self.widget.itemData(i) == value:
                self.widget.setCurrentIndex(i)
                break


class ListParameterWidget(ParameterWidget):
    """Widget for list parameters (simplified as comma-separated values)."""
    
    def create_widget(self) -> QWidget:
        self.widget = QLineEdit()
        
        # Convert list to comma-separated string
        if isinstance(self.value, list):
            text = ", ".join(str(item) for item in self.value)
        else:
            text = str(self.value)
        self.widget.setText(text)
        self.widget.setPlaceholderText("Enter comma-separated values")
        
        return self.widget
        
    def get_value(self) -> List[Any]:
        text = self.widget.text().strip()
        if not text:
            return []
            
        # Split by comma and convert based on list type annotation
        items = [item.strip() for item in text.split(",")]
        
        # Try to determine the list item type
        origin = getattr(self.annotation, '__origin__', None)
        if origin is list:
            args = getattr(self.annotation, '__args__', ())
            if args:
                item_type = args[0]
                try:
                    if item_type == int:
                        return [int(item) for item in items]
                    elif item_type == float:
                        return [float(item) for item in items]
                    elif item_type == bool:
                        return [item.lower() in ('true', '1', 'yes') for item in items]
                except ValueError:
                    pass
        
        return items  # Return as strings if conversion fails
        
    def set_value(self, value: List[Any]) -> None:
        if isinstance(value, list):
            text = ", ".join(str(item) for item in value)
        else:
            text = str(value)
        self.widget.setText(text)


class AutoLamellaTaskConfigWidget(QWidget):
    """Widget for configuring AutoLamella task parameters."""
    
    config_changed = pyqtSignal(AutoLamellaTaskConfig)
    
    def __init__(self, task_config: Optional[AutoLamellaTaskConfig] = None, 
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.task_config = task_config
        self.parameter_widgets: Dict[str, ParameterWidget] = {}
        self.milling_task_widget: FibsemMillingTaskWidget

        self._setup_ui()
        if self.task_config:
            self._update_from_config()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create collapsible section for task parameters
        self.task_params_collapsible = QCollapsible("Task Parameters", self)
        
        # Create content widget for parameters
        self.params_widget = QWidget()
        self.grid_layout = QGridLayout(self.params_widget)
        self.current_row = 0
        
        self.task_params_collapsible.addWidget(self.params_widget)

        # Create collapsible section for milling parameters
        self.milling_params_collapsible = QCollapsible("Milling Task Parameters", self)

        # initialise milling_task_widget 
        microscope, settings = utils.setup_session()  # TODO: pass in from the parent if available...
        self.milling_task_widget = FibsemMillingTaskWidget(
            microscope=microscope, task_configs={}, 
            milling_enabled=False, 
            parent=self
        )
        self.milling_task_widget.setMinimumHeight(600)
        self.milling_params_collapsible.addWidget(self.milling_task_widget)

        self.main_layout.addWidget(self.task_params_collapsible)    # type: ignore
        self.main_layout.addWidget(self.milling_params_collapsible) # type: ignore
        
        # Connect milling widget signals
        self.milling_task_widget.task_config_updated.connect(self._on_milling_config_updated)

        self.main_layout.addStretch()

    def set_task_config(self, task_config: Optional[AutoLamellaTaskConfig]):
        """Set the task configuration to edit."""
        self.task_config = task_config
        self._update_from_config()
        
    def _update_from_config(self):
        """Update the UI from the current task configuration."""
        if not self.task_config:
            return
            
        # Clear existing widgets
        self._clear_form()
        
        # Show/hide task parameters section
        if self.task_config.parameters:
            # Get all configurable parameters using the parameters property
            for param_name in self.task_config.parameters:
                # Get field info and current value
                field = next(f for f in fields(self.task_config) if f.name == param_name)
                value = getattr(self.task_config, param_name)
                
                # Create parameter widget
                param_widget = self._create_parameter_widget(param_name, value, field.type, field.metadata)
                if param_widget:
                    self.parameter_widgets[param_name] = param_widget
                    
                    # Add to form
                    widget = param_widget.create_widget()
                    
                    # Connect change signals to update config
                    self._connect_widget_signals(widget, param_name)
                    
                    # Create label with tooltip if available
                    label = QLabel(self._format_field_name(param_name))
                    if hasattr(field, 'metadata') and 'help' in field.metadata:
                        label.setToolTip(field.metadata['help'])
                        
                    self.grid_layout.addWidget(label, self.current_row, 0)
                    self.grid_layout.addWidget(widget, self.current_row, 1)
                    self.current_row += 1
            
            self.task_params_collapsible.show()
        else:
            self.task_params_collapsible.hide()
    
        # Show/hide milling parameters section
        if self.task_config.milling:
            self.milling_task_widget.set_task_configs(self.task_config.milling)
            self.milling_params_collapsible.show()
        else:
            self.milling_task_widget.set_task_configs({})
            self.milling_task_widget.config_widget.milling_editor_widget.clear_milling_stages()
            self.milling_params_collapsible.hide()

    def _create_parameter_widget(self, name: str, value: Any, annotation: type, metadata: Optional[dict] = None) -> Optional[ParameterWidget]:
        """Create the appropriate parameter widget for the given type."""
        metadata = metadata or {}

        # Handle Union types (e.g., Optional[T])
        origin = getattr(annotation, '__origin__', None)
        if origin is Union:
            args = getattr(annotation, '__args__', ())
            # Find the non-None type for Optional[T]
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                annotation = non_none_types[0]

        # Determine widget type based on annotation
        if annotation is bool:
            return BoolParameterWidget(name, value, annotation)
        elif annotation is int:
            return IntParameterWidget(name, value, annotation)
        elif annotation is float:
            return FloatParameterWidget(name, value, annotation, metadata)
        elif annotation is str:
            return StringParameterWidget(name, value, annotation)
        elif inspect.isclass(annotation) and issubclass(annotation, Enum):
            return EnumParameterWidget(name, value, annotation)
        elif origin is list or (inspect.isclass(annotation) and issubclass(annotation, list)):
            return ListParameterWidget(name, value, annotation)
        else:
            logging.warning(f"Unsupported parameter type '{annotation}' for field '{name}'. Using string widget as fallback.")
            # Fallback to string widget for unknown types
            return StringParameterWidget(name, value, annotation)
    
    def _connect_widget_signals(self, widget: QWidget, field_name: str):
        """Connect widget change signals to update the configuration."""
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, QLineEdit):
            widget.editingFinished.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(lambda: self._on_parameter_changed(field_name))
    
    def _on_parameter_changed(self, field_name: str):
        """Handle parameter value changes."""
        if field_name in self.parameter_widgets:
            param_widget = self.parameter_widgets[field_name]
            new_value = param_widget.get_value()

            # Update the task config
            setattr(self.task_config, field_name, new_value)

            # Emit change signal
            self.config_changed.emit(self.task_config)

    def _on_milling_config_updated(self, task_name: str, milling_config):
        """Handle milling task config updates."""
        if self.task_config and hasattr(self.task_config, 'milling'):
            # Update the milling config in the task config
            if not self.task_config.milling:
                self.task_config.milling = {}
            self.task_config.milling[task_name] = milling_config

            # Emit change signal
            self.config_changed.emit(self.task_config)

    def _clear_form(self):
        """Clear all widgets from the grid layout."""
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child and child.widget():
                child.widget().setParent(None)
        self.parameter_widgets.clear()
        self.current_row = 0
    
    def _format_field_name(self, field_name: str) -> str:
        """Format field name for display (convert snake_case to Title Case)."""
        return field_name.replace('_', ' ').title()
    
    def get_task_config(self) -> Optional[AutoLamellaTaskConfig]:
        """Get the current task configuration."""
        return self.task_config


class AutoLamellaTaskParametersConfigWidget(QWidget):
    """Widget for configuring AutoLamella task parameters."""
    
    config_changed = pyqtSignal(AutoLamellaTaskConfig)
    parameter_changed = pyqtSignal(str, object)  # field name, new value

    def __init__(self, task_config: Optional[AutoLamellaTaskConfig] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.task_config = task_config
        self.parameter_widgets: Dict[str, ParameterWidget] = {}

        self._setup_ui()
        if self.task_config:
            self._update_from_config()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create content widget for parameters
        self.params_widget = QWidget()
        self.grid_layout = QGridLayout(self.params_widget)
        self.current_row = 0
        self.params_groupbox = QGroupBox("Task")
        self.params_groupbox.setLayout(self.grid_layout)

        self.main_layout.addWidget(self.params_groupbox)    # type: ignore

    def set_task_config(self, task_config: Optional[AutoLamellaTaskConfig]):
        """Set the task configuration to edit."""
        self.task_config = task_config
        self._update_from_config()

    def _update_from_config(self):
        """Update the UI from the current task configuration."""
        if not self.task_config:
            return

        # Clear existing widgets
        self._clear_form()

        # Show/hide task parameters section
        if not self.task_config.parameters:
            self.hide()
            return

        # Get all configurable parameters using the parameters property
        for param_name in self.task_config.parameters:
            # Get field info and current value
            field = next(f for f in fields(self.task_config) if f.name == param_name)
            value = getattr(self.task_config, param_name)

            # Create parameter widget
            param_widget = self._create_parameter_widget(name = param_name, 
                                                         value = value, 
                                                         annotation = field.type, 
                                                         metadata = field.metadata)
            if param_widget:
                self.parameter_widgets[param_name] = param_widget

                # Add to form
                widget = param_widget.create_widget()

                # Connect change signals to update config
                self._connect_widget_signals(widget, param_name)

                # Create label with tooltip if available
                label = QLabel(self._format_field_name(param_name, field.metadata))
                if hasattr(field, 'metadata') and 'help' in field.metadata:
                    label.setToolTip(field.metadata['help'])

                self.grid_layout.addWidget(label, self.current_row, 0)
                self.grid_layout.addWidget(widget, self.current_row, 1)
                self.current_row += 1

        self.show()

    def _create_parameter_widget(self,
                                 name: str,
                                 value: Any,
                                 annotation: type,
                                 metadata: Optional[dict] = None) -> Optional[ParameterWidget]:
        """Create the appropriate parameter widget for the given type."""
        metadata = metadata or {}

        # Handle Union types (e.g., Optional[T])
        origin = getattr(annotation, '__origin__', None)
        if origin is Union:
            args = getattr(annotation, '__args__', ())
            # Find the non-None type for Optional[T]
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                annotation = non_none_types[0]

        # Determine widget type based on annotation
        if annotation == bool:
            return BoolParameterWidget(name, value, annotation)
        elif annotation == int:
            return IntParameterWidget(name, value, annotation)
        elif annotation == float:
            return FloatParameterWidget(name, value, annotation, metadata)
        elif annotation == str:
            return StringParameterWidget(name, value, annotation)
        elif inspect.isclass(annotation) and issubclass(annotation, Enum):
            return EnumParameterWidget(name, value, annotation)
        elif origin is list or (inspect.isclass(annotation) and issubclass(annotation, list)):
            return ListParameterWidget(name, value, annotation)
        else:
            # Fallback to string widget for unknown types
            return StringParameterWidget(name, value, annotation)
    
    def _connect_widget_signals(self, widget: QWidget, field_name: str):
        """Connect widget change signals to update the configuration."""
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, QLineEdit):
            widget.editingFinished.connect(lambda: self._on_parameter_changed(field_name))
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(lambda: self._on_parameter_changed(field_name))
    
    def _on_parameter_changed(self, field_name: str):
        """Handle parameter value changes."""
        if field_name in self.parameter_widgets:
            param_widget = self.parameter_widgets[field_name]
            new_value = param_widget.get_value()

            # Update the task config
            setattr(self.task_config, field_name, new_value)

            # Emit change signal
            # self.config_changed.emit(self.task_config)
            self.parameter_changed.emit(field_name, new_value)

    def _clear_form(self):
        """Clear all widgets from the grid layout."""
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child and child.widget():
                child.widget().setParent(None)
        self.parameter_widgets.clear()
        self.current_row = 0

    def _format_field_name(self, field_name: str, metadata: Optional[dict] = None) -> str:
        """Format field name for display (convert snake_case to Title Case)."""
        if metadata and 'label' in metadata:
            return metadata['label']
        return field_name.replace('_', ' ').title()
    
    def get_task_config(self) -> Optional[AutoLamellaTaskConfig]:
        """Get the current task configuration."""
        return self.task_config




if __name__ == "__main__":
    import napari

    from fibsem.applications.autolamella.workflows.tasks.tasks import (
        AcquireReferenceImageConfig,
        MillFiducialTaskConfig,
    )

    # Create test config
    test_config = MillFiducialTaskConfig(
        use_fiducial=True
    )
    acquire_config = AcquireReferenceImageConfig()
    # test_config = DEFAULT_PROTOCOL.task_config['MILL_ROUGH']

    # Create widget
    viewer = napari.Viewer()
    widget = AutoLamellaTaskConfigWidget(test_config)
    widget.config_changed.connect(lambda config: print(f"Config changed: {config}"))
    widget.setWindowTitle("AutoLamella Task Config Widget Test")
    viewer.window.add_dock_widget(widget, area="right", add_vertical_stretch=False)


    widget = AutoLamellaTaskConfigWidget(acquire_config)
    widget.config_changed.connect(lambda config: print(f"Config changed: {config}"))
    widget.setWindowTitle(acquire_config.task_name)
    viewer.window.add_dock_widget(widget, area="right", add_vertical_stretch=False)

    napari.run()