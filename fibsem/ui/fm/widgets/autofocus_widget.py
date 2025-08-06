from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)
from fibsem.fm.structures import ChannelSettings

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
    def __init__(self, channel_settings: List[ChannelSettings], parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.parent_widget = parent
        
        # Default autofocus parameters
        self.coarse_range = AUTOFOCUS_CONFIG["coarse_range"]["default"] * 1e-6  # Convert µm to m
        self.coarse_step = AUTOFOCUS_CONFIG["coarse_step"]["default"] * 1e-6
        self.fine_range = AUTOFOCUS_CONFIG["fine_range"]["default"] * 1e-6
        self.fine_step = AUTOFOCUS_CONFIG["fine_step"]["default"] * 1e-6
        self.selected_channel_index = 0  # Default to first channel
        
        self.initUI()

    def initUI(self):
        
        # Coarse Range
        self.label_coarse_range = QLabel("Coarse Range", self)
        self.doubleSpinBox_coarse_range = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_coarse_range,
            AUTOFOCUS_CONFIG["coarse_range"],
            self.coarse_range * 1e6,  # Convert m to µm for display
            self._on_coarse_range_changed
        )
        
        # Coarse Step
        self.label_coarse_step = QLabel("Coarse Step", self)
        self.doubleSpinBox_coarse_step = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_coarse_step,
            AUTOFOCUS_CONFIG["coarse_step"],
            self.coarse_step * 1e6,
            self._on_coarse_step_changed
        )
        
        # Fine Range
        self.label_fine_range = QLabel("Fine Range", self)
        self.doubleSpinBox_fine_range = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_fine_range,
            AUTOFOCUS_CONFIG["fine_range"],
            self.fine_range * 1e6,
            self._on_fine_range_changed
        )
        
        # Fine Step
        self.label_fine_step = QLabel("Fine Step", self)
        self.doubleSpinBox_fine_step = QDoubleSpinBox(self)
        self._setup_spinbox(
            self.doubleSpinBox_fine_step,
            AUTOFOCUS_CONFIG["fine_step"],
            self.fine_step * 1e6,
            self._on_fine_step_changed
        )
        
        # Channel Selection
        self.label_channel = QLabel("AF Channel", self)
        self.comboBox_channel = QComboBox(self)
        self.comboBox_channel.setToolTip("Channel to use for autofocus")
        self._update_channel_list()
        self.comboBox_channel.currentIndexChanged.connect(self._on_channel_changed)
        
        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_coarse_range, 0, 0)
        layout.addWidget(self.doubleSpinBox_coarse_range, 0, 1)
        layout.addWidget(self.label_coarse_step, 1, 0)
        layout.addWidget(self.doubleSpinBox_coarse_step, 1, 1)
        layout.addWidget(self.label_fine_range, 2, 0)
        layout.addWidget(self.doubleSpinBox_fine_range, 2, 1)
        layout.addWidget(self.label_fine_step, 3, 0)
        layout.addWidget(self.doubleSpinBox_fine_step, 3, 1)
        layout.addWidget(self.label_channel, 4, 0)
        layout.addWidget(self.comboBox_channel, 4, 1)
        
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
        
        # Select the current channel
        if self.selected_channel_index < len(self.channel_settings):
            self.comboBox_channel.setCurrentIndex(self.selected_channel_index)

    def _on_coarse_range_changed(self, value: float):
        """Handle coarse range value change."""
        self.coarse_range = value * 1e-6  # Convert µm to m

    def _on_coarse_step_changed(self, value: float):
        """Handle coarse step value change."""
        self.coarse_step = value * 1e-6  # Convert µm to m

    def _on_fine_range_changed(self, value: float):
        """Handle fine range value change."""
        self.fine_range = value * 1e-6  # Convert µm to m

    def _on_fine_step_changed(self, value: float):
        """Handle fine step value change."""
        self.fine_step = value * 1e-6  # Convert µm to m

    def _on_channel_changed(self, index: int):
        """Handle channel selection change."""
        self.selected_channel_index = index

    def get_autofocus_parameters(self) -> dict:
        """Get the current autofocus parameters."""
        return {
            'coarse_range': self.coarse_range,
            'coarse_step': self.coarse_step,
            'fine_range': self.fine_range,
            'fine_step': self.fine_step,
            'channel_settings': self.channel_settings[self.selected_channel_index] if self.channel_settings else None,
        }

    def get_selected_channel(self) -> Optional[ChannelSettings]:
        """Get the currently selected channel."""
        if self.selected_channel_index < len(self.channel_settings):
            return self.channel_settings[self.selected_channel_index]
        return None

    def update_channels(self, channel_settings: List[ChannelSettings]):
        """Update the available channels."""
        # Store current selection
        current_selection = None
        if (self.selected_channel_index < len(self.channel_settings) and 
            self.selected_channel_index >= 0):
            current_selection = self.channel_settings[self.selected_channel_index].name
        
        # Update channel settings
        self.channel_settings = channel_settings
        
        # Update the UI
        self._update_channel_list()
        
        # Try to restore previous selection
        if current_selection and self.channel_settings:
            for i, channel in enumerate(self.channel_settings):
                if channel.name == current_selection:
                    self.selected_channel_index = i
                    self.comboBox_channel.setCurrentIndex(i)
                    return
        
        # If we couldn't restore, select first channel or reset to 0
        if self.channel_settings:
            self.selected_channel_index = 0
            self.comboBox_channel.setCurrentIndex(0)
        else:
            self.selected_channel_index = 0

    def set_autofocus_parameters(self, coarse_range: float, coarse_step: float, fine_range: float, fine_step: float):
        """Set autofocus parameters programmatically."""
        self.doubleSpinBox_coarse_range.setValue(coarse_range * 1e6)  # Convert m to µm
        self.doubleSpinBox_coarse_step.setValue(coarse_step * 1e6)
        self.doubleSpinBox_fine_range.setValue(fine_range * 1e6)
        self.doubleSpinBox_fine_step.setValue(fine_step * 1e6)

    def set_selected_channel_index(self, index: int):
        """Set the selected channel by index."""
        if 0 <= index < len(self.channel_settings):
            self.selected_channel_index = index
            self.comboBox_channel.setCurrentIndex(index)

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