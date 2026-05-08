from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, FocusMethod

AUTOFOCUS_CONFIG = {
    "coarse_range": {
        "range": (1.0, 100.0),  # 1 to 100 µm range
        "step": 1.0,
        "decimals": 1,
        "suffix": " µm",
        "default": 20.0,
        "tooltip": "Range for coarse autofocus search (±range/2)",
    },
    "coarse_step": {
        "range": (0.1, 20.0),  # 0.1 to 20 µm step
        "step": 0.1,
        "decimals": 1,
        "suffix": " µm",
        "default": 5.0,
        "tooltip": "Step size for coarse autofocus search",
    },
    "fine_range": {
        "range": (1.0, 50.0),  # 1 to 50 µm range
        "step": 0.5,
        "decimals": 1,
        "suffix": " µm",
        "default": 10.0,
        "tooltip": "Range for fine autofocus search (±range/2)",
    },
    "fine_step": {
        "range": (0.1, 5.0),  # 0.1 to 5 µm step
        "step": 0.1,
        "decimals": 2,
        "suffix": " µm",
        "default": 1.0,
        "tooltip": "Step size for fine autofocus search",
    },
}

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget


class AutofocusWidget(QWidget):

    settings_changed = pyqtSignal(AutoFocusSettings)

    def __init__(self, channel_settings: List[ChannelSettings], parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.parent_widget = parent
        
        # Initialize with default AutoFocusSettings
        self.autofocus_settings = AutoFocusSettings(
            coarse_range=AUTOFOCUS_CONFIG["coarse_range"]["default"] * 1e-6,  # Convert µm to m
            coarse_step=AUTOFOCUS_CONFIG["coarse_step"]["default"] * 1e-6,
            fine_range=AUTOFOCUS_CONFIG["fine_range"]["default"] * 1e-6,
            fine_step=AUTOFOCUS_CONFIG["fine_step"]["default"] * 1e-6,
            method=FocusMethod.LAPLACIAN,
            channel_name=channel_settings[0].name if channel_settings else None
        )
        
        self.initUI()

    def initUI(self):
        
        # Coarse search enabled
        self.checkBox_coarse_enabled = QCheckBox("Coarse Search", self)
        self.checkBox_coarse_enabled.setChecked(self.autofocus_settings.coarse_enabled)
        self.checkBox_coarse_enabled.setToolTip("Enable coarse autofocus search phase")
        self.checkBox_coarse_enabled.toggled.connect(self._on_coarse_enabled_changed)
        
        # Coarse Range
        self.label_coarse_range = QLabel("Coarse Range", self)
        self.doubleSpinBox_coarse_range = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_coarse_range,
            AUTOFOCUS_CONFIG["coarse_range"],
            self.autofocus_settings.coarse_range * 1e6,  # Convert m to µm for display
            self._on_coarse_range_changed
        )
        
        # Coarse Step
        self.label_coarse_step = QLabel("Coarse Step", self)
        self.doubleSpinBox_coarse_step = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_coarse_step,
            AUTOFOCUS_CONFIG["coarse_step"],
            self.autofocus_settings.coarse_step * 1e6,
            self._on_coarse_step_changed
        )
        
        # Fine search enabled
        self.checkBox_fine_enabled = QCheckBox("Fine Search", self)
        self.checkBox_fine_enabled.setChecked(self.autofocus_settings.fine_enabled)
        self.checkBox_fine_enabled.setToolTip("Enable fine autofocus search phase")
        self.checkBox_fine_enabled.toggled.connect(self._on_fine_enabled_changed)
        
        # Fine Range
        self.label_fine_range = QLabel("Fine Range", self)
        self.doubleSpinBox_fine_range = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_fine_range,
            AUTOFOCUS_CONFIG["fine_range"],
            self.autofocus_settings.fine_range * 1e6,
            self._on_fine_range_changed
        )
        
        # Fine Step
        self.label_fine_step = QLabel("Fine Step", self)
        self.doubleSpinBox_fine_step = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_fine_step,
            AUTOFOCUS_CONFIG["fine_step"],
            self.autofocus_settings.fine_step * 1e6,
            self._on_fine_step_changed
        )
        
        # Focus Method Selection
        self.label_method = QLabel("Focus Method", self)
        self.comboBox_method = QComboBox(self)
        self.comboBox_method.setToolTip("Focus measurement method to use")
        for method in FocusMethod:
            self.comboBox_method.addItem(method.value.title(), method)
        # Set current method
        for i in range(self.comboBox_method.count()):
            if self.comboBox_method.itemData(i) == self.autofocus_settings.method:
                self.comboBox_method.setCurrentIndex(i)
                break
        self.comboBox_method.currentIndexChanged.connect(self._on_method_changed)
        
        # Channel Selection
        self.label_channel = QLabel("Focus Channel", self)
        self.comboBox_channel = QComboBox(self)
        self.comboBox_channel.setToolTip("Channel to use for autofocus")
        self._update_channel_list()
        self.comboBox_channel.currentIndexChanged.connect(self._on_channel_changed)
        
        # Create the layout
        layout = QGridLayout()
        row = 0
        layout.addWidget(self.checkBox_coarse_enabled, row, 0, 1, 2)  # Span 2 columns
        row += 1
        layout.addWidget(self.label_coarse_range, row, 0)
        layout.addWidget(self.doubleSpinBox_coarse_range, row, 1)
        row += 1
        layout.addWidget(self.label_coarse_step, row, 0)
        layout.addWidget(self.doubleSpinBox_coarse_step, row, 1)
        row += 1
        layout.addWidget(self.checkBox_fine_enabled, row, 0, 1, 2)  # Span 2 columns
        row += 1
        layout.addWidget(self.label_fine_range, row, 0)
        layout.addWidget(self.doubleSpinBox_fine_range, row, 1)
        row += 1
        layout.addWidget(self.label_fine_step, row, 0)
        layout.addWidget(self.doubleSpinBox_fine_step, row, 1)
        row += 1
        layout.addWidget(self.label_method, row, 0)
        layout.addWidget(self.comboBox_method, row, 1)
        row += 1
        layout.addWidget(self.label_channel, row, 0)
        layout.addWidget(self.comboBox_channel, row, 1)
        
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

    def _setup_spinbox(self, spinbox: QDoubleSpinBox, config: dict, value: float, callback):
        """Setup a spinbox with the given configuration."""
        spinbox.setRange(config["range"][0], config["range"][1])
        spinbox.setValue(value)
        spinbox.setSingleStep(config["step"])
        spinbox.setDecimals(config["decimals"])
        spinbox.setSuffix(config["suffix"])
        spinbox.setToolTip(config["tooltip"])
        spinbox.setKeyboardTracking(False)
        spinbox.valueChanged.connect(callback)

    def _update_channel_list(self):
        """Update the channel selection combobox."""
        self.comboBox_channel.clear()
        for i, channel in enumerate(self.channel_settings):
            self.comboBox_channel.addItem(channel.name, i)
        
        # Select the current channel based on settings
        if self.autofocus_settings.channel_name:
            for i, channel in enumerate(self.channel_settings):
                if channel.name == self.autofocus_settings.channel_name:
                    self.comboBox_channel.setCurrentIndex(i)
                    break

    def _on_coarse_enabled_changed(self, enabled: bool):
        """Handle coarse enabled checkbox change."""
        self.autofocus_settings.coarse_enabled = enabled
        self._emit_settings_changed()

    def _on_coarse_range_changed(self, value: float):
        """Handle coarse range value change."""
        self.autofocus_settings.coarse_range = value * 1e-6  # Convert µm to m
        self._emit_settings_changed()

    def _on_coarse_step_changed(self, value: float):
        """Handle coarse step value change."""
        self.autofocus_settings.coarse_step = value * 1e-6  # Convert µm to m
        self._emit_settings_changed()

    def _on_fine_enabled_changed(self, enabled: bool):
        """Handle fine enabled checkbox change."""
        self.autofocus_settings.fine_enabled = enabled
        self._emit_settings_changed()

    def _on_fine_range_changed(self, value: float):
        """Handle fine range value change."""
        self.autofocus_settings.fine_range = value * 1e-6  # Convert µm to m
        self._emit_settings_changed()

    def _on_fine_step_changed(self, value: float):
        """Handle fine step value change."""
        self.autofocus_settings.fine_step = value * 1e-6  # Convert µm to m
        self._emit_settings_changed()

    def _on_method_changed(self, index: int):
        """Handle focus method selection change."""
        method = self.comboBox_method.itemData(index)
        if method:
            self.autofocus_settings.method = method
            self._emit_settings_changed()

    def _on_channel_changed(self, index: int):
        """Handle channel selection change."""
        if 0 <= index < len(self.channel_settings):
            self.autofocus_settings.channel_name = self.channel_settings[index].name
            self._emit_settings_changed()

    def get_autofocus_settings(self) -> AutoFocusSettings:
        """Get the current autofocus settings."""
        return self.autofocus_settings

    def get_selected_channel(self) -> Optional[ChannelSettings]:
        """Get the currently selected channel."""
        if self.autofocus_settings.channel_name:
            for channel in self.channel_settings:
                if channel.name == self.autofocus_settings.channel_name:
                    return channel
        return None

    def update_channels(self, channel_settings: List[ChannelSettings]):
        """Update the available channels."""
        # Store current selection
        current_selection = self.autofocus_settings.channel_name
        
        # Update channel settings
        self.channel_settings = channel_settings
        
        # Update the UI
        self._update_channel_list()
        
        # Try to restore previous selection
        if current_selection and self.channel_settings:
            for i, channel in enumerate(self.channel_settings):
                if channel.name == current_selection:
                    self.comboBox_channel.setCurrentIndex(i)
                    return
        
        # If we couldn't restore, select first channel
        if self.channel_settings:
            self.comboBox_channel.setCurrentIndex(0)
            self.autofocus_settings.channel_name = self.channel_settings[0].name
        else:
            self.autofocus_settings.channel_name = None

        if current_selection != self.autofocus_settings.channel_name:
            self._emit_settings_changed()

    def set_autofocus_settings(self, settings: AutoFocusSettings):
        """Set autofocus settings from AutoFocusSettings object."""
        self.autofocus_settings = settings
        self._update_ui_from_settings()

    def show_coarse_settings(self, show: bool):
        """Show or hide coarse settings."""
        self.checkBox_coarse_enabled.setVisible(show)
        self.label_coarse_range.setVisible(show)
        self.doubleSpinBox_coarse_range.setVisible(show)
        self.label_coarse_step.setVisible(show)
        self.doubleSpinBox_coarse_step.setVisible(show)

    def show_fine_settings(self, show: bool):
        """Show or hide fine settings."""
        self.checkBox_fine_enabled.setVisible(show)
        self.label_fine_range.setVisible(show)
        self.doubleSpinBox_fine_range.setVisible(show)
        self.label_fine_step.setVisible(show)
        self.doubleSpinBox_fine_step.setVisible(show)

    def single_fine_search_mode(self):
        self.checkBox_fine_enabled.setText("Enable Autofocus")
        self.label_fine_step.setText("Step Size")
        self.label_fine_range.setText("Search Range")
        self.show_coarse_settings(False)

    def set_selected_channel_by_name(self, channel_name: str):
        """Set the selected channel by name."""
        for i, channel in enumerate(self.channel_settings):
            if channel.name == channel_name:
                self.comboBox_channel.setCurrentIndex(i)
                self.autofocus_settings.channel_name = channel_name
                return
    
    def _update_ui_from_settings(self):
        """Update all UI elements from the current autofocus settings."""
        # Update checkboxes
        self.checkBox_coarse_enabled.setChecked(self.autofocus_settings.coarse_enabled)
        self.checkBox_fine_enabled.setChecked(self.autofocus_settings.fine_enabled)
        
        # Update spin boxes (convert m to µm for display)
        self.doubleSpinBox_coarse_range.setValue(self.autofocus_settings.coarse_range * 1e6)
        self.doubleSpinBox_coarse_step.setValue(self.autofocus_settings.coarse_step * 1e6)
        self.doubleSpinBox_fine_range.setValue(self.autofocus_settings.fine_range * 1e6)
        self.doubleSpinBox_fine_step.setValue(self.autofocus_settings.fine_step * 1e6)
        
        # Update method selection
        for i in range(self.comboBox_method.count()):
            if self.comboBox_method.itemData(i) == self.autofocus_settings.method:
                self.comboBox_method.setCurrentIndex(i)
                break
        
        # Update channel selection
        if self.autofocus_settings.channel_name:
            self.set_selected_channel_by_name(self.autofocus_settings.channel_name)

    def _update_channel_names_from_parent(self):
        """Update channel settings from the parent widget's channel settings."""
        try:
            if (self.parent_widget and 
                hasattr(self.parent_widget, 'channelSettingsWidget') and 
                self.parent_widget.channelSettingsWidget):
                channel_settings = self.parent_widget.channelSettingsWidget.channel_settings
                self.update_channels(channel_settings)
        except Exception as e:
            import logging
            logging.warning(f"Error updating channels from parent: {e}")

    def _emit_settings_changed(self):
        """Emit the current autofocus settings."""
        self.settings_changed.emit(self.autofocus_settings)
