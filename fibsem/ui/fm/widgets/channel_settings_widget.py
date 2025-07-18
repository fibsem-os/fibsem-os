
import logging
from typing import Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
)

CHANNEL_SETTINGS_CONFIG = {
    "power": {
        "range": (0.0, 1.0),
        "step": 0.01,
        "decimals": 3,
        "suffix": " W",
        "tooltip": "Laser power in watts (0.0 to 1.0)",
    },
    "exposure_time": {
        "range": (1.0, 1000.0),  # 1ms to 1000ms
        "step": 1.0,
        "decimals": 1,
        "suffix": " ms",
        "tooltip": "Camera exposure time in milliseconds",
    },
}


class ChannelSettingsWidget(QWidget):
    """Main channel settings widget that wraps MultiChannelSettingsWidget for backward compatibility."""
    
    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: Union[ChannelSettings, List[ChannelSettings]],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fm = fm
        
        # Create the main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Create and add the multi-channel widget
        self._multi_channel_widget = MultiChannelSettingsWidget(
            fm=fm,
            channel_settings=channel_settings,
            parent=self
        )
        layout.addWidget(self._multi_channel_widget)
    
    @property
    def channel_settings(self) -> List[ChannelSettings]:
        """Get channel settings - always returns a list of ChannelSettings."""
        return self._multi_channel_widget.channel_settings
    
    @channel_settings.setter
    def channel_settings(self, value: Union[ChannelSettings, List[ChannelSettings]]):
        """Set channel settings."""
        self._multi_channel_widget.channel_settings = value
    
    # Delegate properties to the multi-channel widget for backward compatibility
    @property
    def channel_name_input(self):
        return self._multi_channel_widget.channel_name_input
    
    @property
    def excitation_wavelength_input(self):
        return self._multi_channel_widget.excitation_wavelength_input
    
    @property
    def emission_wavelength_input(self):
        return self._multi_channel_widget.emission_wavelength_input
    
    @property
    def power_input(self):
        return self._multi_channel_widget.power_input
    
    @property
    def exposure_time_input(self):
        return self._multi_channel_widget.exposure_time_input

    @property
    def selected_channel(self) -> Optional[ChannelSettings]:
        """Get the currently selected channel for live acquisition."""
        return self._multi_channel_widget.selected_channel


class SingleChannelWidget(QWidget):
    """Widget for a single channel's settings."""
    
    remove_requested = pyqtSignal(object)  # Signal to request removal of this channel
    
    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: ChannelSettings,
                 show_remove_button: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.fm = fm
        self.show_remove_button = show_remove_button
        self.initUI()

    def initUI(self):
        """Initialize the UI components for the single channel widget."""
        self.setContentsMargins(5, 5, 5, 5)
        
        # Create a frame to group the channel settings
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame_layout = QGridLayout()
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame.setLayout(frame_layout)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
        
        layout = frame_layout

        # Channel name with optional remove button
        channel_header_layout = QHBoxLayout()
        channel_header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.channel_name_input = QLineEdit(self.channel_settings.name, self)
        self.channel_name_input.setPlaceholderText("Enter channel name")
        channel_header_layout.addWidget(QLabel("Channel:"))
        channel_header_layout.addWidget(self.channel_name_input)
        
        if self.show_remove_button:
            self.remove_button = QPushButton("✕")
            self.remove_button.setStyleSheet(RED_PUSHBUTTON_STYLE)
            self.remove_button.setMaximumWidth(30)
            self.remove_button.setToolTip("Remove this channel")
            self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
            channel_header_layout.addWidget(self.remove_button)
        
        # Convert layout to widget and add to grid
        channel_header_widget = QWidget()
        channel_header_widget.setLayout(channel_header_layout)
        layout.addWidget(channel_header_widget, 0, 0, 1, 2)

        layout.addWidget(QLabel("Excitation Wavelength"), 1, 0)
        self.excitation_wavelength_input = QComboBox()
        for wavelength in self.fm.filter_set.available_excitation_wavelengths:
            self.excitation_wavelength_input.addItem(f"{int(wavelength)} nm", wavelength)

        layout.addWidget(self.excitation_wavelength_input, 1, 1)
        layout.addWidget(QLabel("Emission Wavelength"), 2, 0)
        self.emission_wavelength_input = QComboBox()
        for wavelength in self.fm.filter_set.available_emission_wavelengths:
            if wavelength is None:
                self.emission_wavelength_input.addItem("Reflection", None)
                continue
            self.emission_wavelength_input.addItem(f"{int(wavelength)} nm", wavelength)

        layout.addWidget(self.emission_wavelength_input, 2, 1)
        
        layout.addWidget(QLabel("Power"), 3, 0)
        self.power_input = QDoubleSpinBox()
        self.power_input.setRange(*CHANNEL_SETTINGS_CONFIG["power"]["range"])
        self.power_input.setSingleStep(CHANNEL_SETTINGS_CONFIG["power"]["step"])
        self.power_input.setDecimals(CHANNEL_SETTINGS_CONFIG["power"]["decimals"])
        self.power_input.setSuffix(CHANNEL_SETTINGS_CONFIG["power"]["suffix"])
        self.power_input.setToolTip(CHANNEL_SETTINGS_CONFIG["power"]["tooltip"])

        layout.addWidget(self.power_input, 3, 1)
        
        layout.addWidget(QLabel("Exposure Time"), 4, 0)
        self.exposure_time_input = QDoubleSpinBox()
        self.exposure_time_input.setRange(*CHANNEL_SETTINGS_CONFIG["exposure_time"]["range"])
        self.exposure_time_input.setSingleStep(CHANNEL_SETTINGS_CONFIG["exposure_time"]["step"])
        self.exposure_time_input.setDecimals(CHANNEL_SETTINGS_CONFIG["exposure_time"]["decimals"])
        self.exposure_time_input.setSuffix(CHANNEL_SETTINGS_CONFIG["exposure_time"]["suffix"])
        self.exposure_time_input.setToolTip(CHANNEL_SETTINGS_CONFIG["exposure_time"]["tooltip"])
        layout.addWidget(self.exposure_time_input, 4, 1)

        # Set column stretch factors to make widgets expand properly
        layout.setColumnStretch(0, 1)  # Labels column - expandable
        layout.setColumnStretch(1, 1)  # Input widgets column - expandable

        # connect signals to slots
        self.channel_name_input.textChanged.connect(self.update_channel_name)
        self.channel_name_input.textChanged.connect(self._notify_parent_of_name_change)
        self.excitation_wavelength_input.currentIndexChanged.connect(self.update_excitation_wavelength)
        self.emission_wavelength_input.currentIndexChanged.connect(self.update_emission_wavelength)
        self.power_input.valueChanged.connect(self.update_power)
        self.exposure_time_input.valueChanged.connect(self.update_exposure_time)

        # set keyboard tracking to false for immediate updates
        self.power_input.setKeyboardTracking(False)
        self.exposure_time_input.setKeyboardTracking(False)

        self.power_input.setValue(self.channel_settings.power)
        self.exposure_time_input.setValue(self.channel_settings.exposure_time * 1000)  # Convert seconds to milliseconds

        # get the closest wavelength to the current channel setting
        if self.channel_settings.excitation_wavelength in self.fm.filter_set.available_excitation_wavelengths:
            idx = self.fm.filter_set.available_excitation_wavelengths.index(self.channel_settings.excitation_wavelength)
            self.excitation_wavelength_input.setCurrentIndex(idx)
        else:
            # if the current wavelength is not available, set to the first one
            self.excitation_wavelength_input.setCurrentIndex(0)

        # get the closest wavelength to the current channel setting
        if self.channel_settings.emission_wavelength in self.fm.filter_set.available_emission_wavelengths:
            idx = self.fm.filter_set.available_emission_wavelengths.index(self.channel_settings.emission_wavelength)
            self.emission_wavelength_input.setCurrentIndex(idx)
        else:
            # if the current wavelength is not available, set to the first one
            self.emission_wavelength_input.setCurrentIndex(0)

    @pyqtSlot(int)
    def update_excitation_wavelength(self, idx: int):
        wavelength = self.excitation_wavelength_input.itemData(idx)
        self.channel_settings.excitation_wavelength = wavelength
        logging.info(f"Excitation wavelength updated to: {wavelength} nm")

    @pyqtSlot(int)      
    def update_emission_wavelength(self, idx: int):
        wavelength = self.emission_wavelength_input.itemData(idx)
        self.channel_settings.emission_wavelength = wavelength
        logging.info(f"Emission wavelength updated to: {wavelength} nm")

    @pyqtSlot(float)
    def update_power(self, value: float):
        self.channel_settings.power = value
        logging.info(f"Power updated to: {value} W")

    @pyqtSlot(float)
    def update_exposure_time(self, value: float):
        # Convert milliseconds to seconds for internal storage
        self.channel_settings.exposure_time = value / 1000
        logging.info(f"Exposure time updated to: {value} ms ({value/1000:.3f} s)") 
    
    @pyqtSlot()
    def update_channel_name(self):
        self.channel_settings.name = self.channel_name_input.text()
        logging.info(f"Channel name updated to: {self.channel_settings.name}")
    
    def _notify_parent_of_name_change(self):
        """Notify parent widget that channel name has changed."""
        # Find the parent MultiChannelSettingsWidget and update the channel list
        parent = self.parent()
        while parent:
            if isinstance(parent, MultiChannelSettingsWidget):
                parent._update_channel_list()
                break
            parent = parent.parent()


class MultiChannelSettingsWidget(QWidget):
    """Multi-channel settings widget that manages multiple SingleChannelWidget instances."""
    
    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: Union[ChannelSettings, List[ChannelSettings]],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fm = fm
        
        # Convert single channel to list for consistent handling
        if isinstance(channel_settings, ChannelSettings):
            self._channel_settings_list = [channel_settings]
        else:
            self._channel_settings_list = channel_settings.copy() if channel_settings else []
        
        self.channel_widgets: List[SingleChannelWidget] = []
        self.initUI()
        self._create_channel_widgets()
        self._update_channel_list()
    
    @property
    def channel_settings(self) -> List[ChannelSettings]:
        """Get the list of channel settings."""
        return self._channel_settings_list
    
    @channel_settings.setter
    def channel_settings(self, value: Union[ChannelSettings, List[ChannelSettings]]):
        """Set the channel settings (for backward compatibility)."""
        if isinstance(value, ChannelSettings):
            self._channel_settings_list = [value]
        else:
            self._channel_settings_list = value.copy() if value else []
        self._recreate_channel_widgets()
    
    def initUI(self):
        """Initialize the UI components for the multi-channel settings widget."""
        self.setContentsMargins(0, 0, 0, 0)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        # Header with add channel button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        header_label = QLabel("Channels")
        header_label.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        self.add_channel_button = QPushButton("+ Add Channel")
        self.add_channel_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.add_channel_button.clicked.connect(self.add_channel)
        header_layout.addWidget(self.add_channel_button)
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        main_layout.addWidget(header_widget)
        
        # Scroll area for channel widgets
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(2)  # AsNeeded
        self.scroll_area.setHorizontalScrollBarPolicy(1)  # AlwaysOff
        
        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout()
        self.channels_layout.setContentsMargins(0, 0, 0, 0)
        self.channels_container.setLayout(self.channels_layout)
        
        self.scroll_area.setWidget(self.channels_container)
        main_layout.addWidget(self.scroll_area)
        
        # Channel list for live acquisition selection
        list_header_label = QLabel("Live Acquisition Channel")
        list_header_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #FFFFFF;")
        main_layout.addWidget(list_header_label)
        
        self.channel_list = QListWidget(self)
        self.channel_list.setMaximumHeight(80)
        self.channel_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #262626;
                color: #FFFFFF;
                selection-background-color: #007ACC;
                selection-color: #FFFFFF;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: #FFFFFF;
            }
        """)
        main_layout.addWidget(self.channel_list)
    
    def _create_channel_widgets(self):
        """Create SingleChannelWidget instances for each channel setting."""
        for i, channel_setting in enumerate(self._channel_settings_list):
            # Show remove button only if there's more than one channel
            show_remove = len(self._channel_settings_list) > 1
            
            channel_widget = SingleChannelWidget(
                fm=self.fm,
                channel_settings=channel_setting,
                show_remove_button=show_remove,
                parent=self
            )
            
            channel_widget.remove_requested.connect(self.remove_channel)
            self.channel_widgets.append(channel_widget)
            self.channels_layout.addWidget(channel_widget)
        
        # Add stretch to push everything to the top
        self.channels_layout.addStretch()
    
    def _recreate_channel_widgets(self):
        """Remove all existing channel widgets and create new ones."""
        # Clear existing widgets
        for widget in self.channel_widgets:
            widget.setParent(None)
            widget.deleteLater()
        
        self.channel_widgets.clear()
        
        # Remove all items from layout
        while self.channels_layout.count():
            item = self.channels_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Create new widgets
        self._create_channel_widgets()
        self._update_remove_buttons()
        self._update_channel_list()
    
    def add_channel(self):
        """Add a new channel with default settings."""
        # Create default channel settings
        new_channel = ChannelSettings(
            name=f"Channel-{len(self._channel_settings_list) + 1:02d}",
            excitation_wavelength=self.fm.filter_set.available_excitation_wavelengths[0],
            emission_wavelength=self.fm.filter_set.available_emission_wavelengths[0],
            power=0.03,
            exposure_time=0.005
        )
        
        self._channel_settings_list.append(new_channel)
        
        # Create new widget
        channel_widget = SingleChannelWidget(
            fm=self.fm,
            channel_settings=new_channel,
            show_remove_button=True,
            parent=self
        )
        
        channel_widget.remove_requested.connect(self.remove_channel)
        self.channel_widgets.append(channel_widget)
        
        # Insert before the stretch
        self.channels_layout.insertWidget(len(self.channel_widgets) - 1, channel_widget)
        
        self._update_remove_buttons()
        self._update_channel_list()
        logging.info(f"Added new channel: {new_channel.name}")
    
    def remove_channel(self, channel_widget: SingleChannelWidget):
        """Remove a channel widget and its corresponding settings."""
        if len(self._channel_settings_list) <= 1:
            logging.warning("Cannot remove the last channel")
            return
        
        # Find the index of the widget
        try:
            index = self.channel_widgets.index(channel_widget)
        except ValueError:
            logging.error("Channel widget not found in list")
            return
        
        # Remove from settings list
        removed_channel = self._channel_settings_list.pop(index)
        
        # Remove widget
        self.channel_widgets.remove(channel_widget)
        channel_widget.setParent(None)
        channel_widget.deleteLater()
        
        self._update_remove_buttons()
        self._update_channel_list()
        logging.info(f"Removed channel: {removed_channel.name}")
    
    def _update_remove_buttons(self):
        """Update visibility of remove buttons based on number of channels."""
        show_remove = len(self.channel_widgets) > 1
        
        for widget in self.channel_widgets:
            if hasattr(widget, 'remove_button'):
                widget.remove_button.setVisible(show_remove)
    
    def _update_channel_list(self):
        """Update the channel list widget with current channel names."""
        self.channel_list.clear()
        
        for i, channel_setting in enumerate(self._channel_settings_list):
            # Ensure channel name is properly displayed
            channel_name = channel_setting.name if channel_setting.name else f"Channel-{i+1:02d}"
            item = QListWidgetItem(channel_name, self.channel_list)
            # item.setData(0, channel_name)  # Store channel name for retrieval
            item.setToolTip(f"Excitation: {channel_setting.excitation_wavelength}nm, "
                           f"Emission: {channel_setting.emission_wavelength}nm, "
                           f"Power: {channel_setting.power}W")
            logging.debug(f"Added channel to list: {channel_name} (index {i})")
        
        # Select first channel by default
        if self.channel_list.count() > 0:
            self.channel_list.setCurrentRow(0)
            logging.debug(f"Selected first channel for live acquisition")
    
    @property
    def selected_channel(self) -> Optional[ChannelSettings]:
        """Get the currently selected channel for live acquisition."""
        channel_index = self.channel_list.currentRow()
        if channel_index < 0 or channel_index >= len(self._channel_settings_list):
            return None
        return self._channel_settings_list[channel_index]
    
    def _get_selected_channel_widget(self) -> Optional['SingleChannelWidget']:
        """Get the widget for the currently selected channel."""
        channel_index = self.channel_list.currentRow()
        if 0 <= channel_index < len(self.channel_widgets):
            return self.channel_widgets[channel_index]
        return None
    
    # Properties for backward compatibility with single-channel interface
    @property
    def channel_name_input(self):
        """Get the selected channel's name input."""
        selected_widget = self._get_selected_channel_widget()
        return selected_widget.channel_name_input if selected_widget else None
    
    @property
    def excitation_wavelength_input(self):
        """Get the selected channel's excitation wavelength input."""
        selected_widget = self._get_selected_channel_widget()
        return selected_widget.excitation_wavelength_input if selected_widget else None
    
    @property
    def emission_wavelength_input(self):
        """Get the selected channel's emission wavelength input."""
        selected_widget = self._get_selected_channel_widget()
        return selected_widget.emission_wavelength_input if selected_widget else None
    
    @property
    def power_input(self):
        """Get the selected channel's power input."""
        selected_widget = self._get_selected_channel_widget()
        return selected_widget.power_input if selected_widget else None
    
    @property
    def exposure_time_input(self):
        """Get the selected channel's exposure time input."""
        selected_widget = self._get_selected_channel_widget()
        return selected_widget.exposure_time_input if selected_widget else None