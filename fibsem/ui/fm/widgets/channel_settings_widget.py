
import logging
from typing import List, Optional, Union
from PyQt5.QtCore import pyqtSlot
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
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

CHANNEL_SETTINGS_CONFIG = {
    "power": {
        "range": (0.0, 1.0),
        "step": 0.001,
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

MAX_CHANNELS = 4

class SingleChannelWidget(QWidget):
    """Widget for a single channel's settings."""

    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: ChannelSettings,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.fm = fm
        self.initUI()

    def initUI(self):
        """Initialize the UI components for the single channel widget."""
        self.setContentsMargins(0, 0, 0, 0)

        # Create a frame to group the channel settings
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame_layout = QGridLayout()
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame.setLayout(frame_layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
        
        layout = frame_layout

        # Channel name with optional remove button
        channel_header_layout = QGridLayout()
        channel_header_layout.setContentsMargins(0, 0, 0, 0)

        self.channel_name_input = QLineEdit(self.channel_settings.name, self)
        self.channel_name_input.setPlaceholderText("Enter channel name")
        channel_header_layout.addWidget(QLabel("Channel:"), 0, 0)
        channel_header_layout.addWidget(self.channel_name_input, 0, 1)

        # Remove buttons are now handled at the parent widget level
        channel_header_layout.setColumnStretch(0, 1)  # Labels column - expandable
        channel_header_layout.setColumnStretch(1, 1)  # Input widgets column - expandable

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
            # Handle both numeric and string wavelengths
            if isinstance(wavelength, str):
                self.emission_wavelength_input.addItem(wavelength, wavelength)
            else:
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
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

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
        logging.info(f"Emission wavelength updated to: {wavelength}")

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
        # Find the parent ChannelSettingsWidget and update the channel list
        parent = self.parent()
        while parent:
            if isinstance(parent, ChannelSettingsWidget):
                parent._update_channel_list()
                break
            parent = parent.parent()


class ChannelSettingsWidget(QWidget):
    """Multi-channel settings widget that manages multiple SingleChannelWidget instances."""

    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: Union[ChannelSettings, List[ChannelSettings]],
                 parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent

        # Convert single channel to list for consistent handling
        if isinstance(channel_settings, ChannelSettings):
            channel_settings = [channel_settings]
        self._channel_settings_list = channel_settings.copy()
        
        self.channel_widgets: List[SingleChannelWidget] = []
        self.initUI()
        self._create_channel_widgets()
        self._update_channel_list()
        self._update_button_states()

        # Connect channel selection changes
        self.channel_list.currentRowChanged.connect(self._on_channel_selection_changed)
        self._connect_live_acquisition_signals()

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

        self.add_channel_button = QPushButton("Add Channel")
        self.add_channel_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.add_channel_button.clicked.connect(self.add_channel)
        header_layout.addWidget(self.add_channel_button)

        self.remove_channel_button = QPushButton("Remove Channel")
        self.remove_channel_button.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.remove_channel_button.clicked.connect(self.remove_selected_channel)
        header_layout.addWidget(self.remove_channel_button)

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
            channel_widget = SingleChannelWidget(
                fm=self.fm,
                channel_settings=channel_setting,
                parent=self
            )

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
        self._update_button_states()
        self._update_channel_list()

    def add_channel(self):
        """Add a new channel with default settings."""
        # Check if we've reached the maximum number of channels
        if len(self._channel_settings_list) >= MAX_CHANNELS:
            logging.warning(f"Cannot add more channels. Maximum limit is {MAX_CHANNELS} channels.")
            return

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
            parent=self
        )

        self.channel_widgets.append(channel_widget)

        # Insert before the stretch
        self.channels_layout.insertWidget(len(self.channel_widgets) - 1, channel_widget)
    
        self._update_button_states()
        self._update_channel_list()
        logging.info(f"Added new channel: {new_channel.name}")

    def remove_selected_channel(self):
        """Remove the currently selected channel."""
        if len(self._channel_settings_list) <= 1:
            logging.warning("Cannot remove the last channel")
            return

        # Get the currently selected channel index
        current_index = self.channel_list.currentRow()
        if current_index < 0 or current_index >= len(self.channel_widgets):
            logging.warning("No channel selected for removal")
            return

        self.remove_channel_by_index(current_index)

    def remove_channel_by_index(self, index: int):
        """Remove a channel by its index."""
        if len(self._channel_settings_list) <= 1:
            logging.warning("Cannot remove the last channel")
            return

        if index < 0 or index >= len(self.channel_widgets):
            logging.error(f"Invalid channel index: {index}")
            return

        # Remove from settings list
        removed_channel = self._channel_settings_list.pop(index)

        # Remove widget
        channel_widget = self.channel_widgets.pop(index)
        channel_widget.setParent(None)
        channel_widget.deleteLater()

        self._update_button_states()
        self._update_channel_list()
        logging.info(f"Removed channel: {removed_channel.name}")
    
    def remove_channel(self, channel_widget: SingleChannelWidget):
        """Remove a channel widget and its corresponding settings (legacy method for compatibility)."""
        try:
            index = self.channel_widgets.index(channel_widget)
            self.remove_channel_by_index(index)
        except ValueError:
            logging.error("Channel widget not found in list")
    
    def _update_button_states(self):
        """Update the enabled state of add and remove buttons based on number of channels."""
        num_channels = len(self.channel_widgets)

        # Remove button: enabled only if more than 1 channel
        self.remove_channel_button.setEnabled(num_channels > 1)

        # Add button: enabled only if less than MAX_CHANNELS
        self.add_channel_button.setEnabled(num_channels < MAX_CHANNELS)

    def _update_channel_list(self):
        """Update the channel list widget with current channel names."""
        self.channel_list.clear()

        for i, channel_setting in enumerate(self._channel_settings_list):
            QListWidgetItem(channel_setting.name, self.channel_list)

        # Select first channel by default
        if self.channel_list.count() > 0:
            self.channel_list.setCurrentRow(0)

        # Notify parent FMAcquisitionWidget about channel name changes
        self._notify_parent_of_channel_changes()

    def _notify_parent_of_channel_changes(self):
        """Notify parent FMAcquisitionWidget that channel names have changed."""
        if (self.parent_widget and 
            hasattr(self.parent_widget, 'overviewParametersWidget') and 
            self.parent_widget.overviewParametersWidget):
            self.parent_widget.overviewParametersWidget._update_channel_names_from_parent()
        
        # Also update autofocus widget
        if (self.parent_widget and 
            hasattr(self.parent_widget, 'autofocusWidget') and 
            self.parent_widget.autofocusWidget):
            self.parent_widget.autofocusWidget._update_channel_names_from_parent()

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

    def _on_channel_selection_changed(self, current_row: int):
        """Handle channel selection changes to update live acquisition connections."""
        # Prevent channel switching during any acquisition for safety
        if self.parent_widget:
            if self.parent_widget.fm.is_acquiring or self.parent_widget.is_acquisition_active:
                logging.warning("Channel selection cannot be changed during acquisition")
                return

        if current_row >= 0:
            logging.debug(f"Channel selection changed to row {current_row}")
            # Reconnect live acquisition signals to the newly selected channel
            self._connect_live_acquisition_signals()

    def _connect_live_acquisition_signals(self):
        """Connect live acquisition parameter update signals to the currently selected channel."""
        # Disconnect any existing connections to avoid duplicate signals
        self._disconnect_live_acquisition_signals()

        # Get the currently selected channel widget
        selected_widget = self._get_selected_channel_widget()
        if selected_widget and self.parent_widget and hasattr(self.parent_widget, '_update_exposure_time'):
            # Connect to the selected channel's input widgets
            selected_widget.exposure_time_input.valueChanged.connect(self.parent_widget._update_exposure_time)
            selected_widget.power_input.valueChanged.connect(self.parent_widget._update_power)
            selected_widget.excitation_wavelength_input.currentIndexChanged.connect(self.parent_widget._update_excitation_wavelength)
            selected_widget.emission_wavelength_input.currentIndexChanged.connect(self.parent_widget._update_emission_wavelength)
            logging.info(f"Connected live acquisition signals to channel: {selected_widget.channel_settings.name}")

    def _disconnect_live_acquisition_signals(self):
        """Disconnect live acquisition parameter update signals from all channels."""
        if not self.parent_widget:
            return

        # Disconnect from all channel widgets to avoid stale connections
        for channel_widget in self.channel_widgets:
            try:
                channel_widget.exposure_time_input.valueChanged.disconnect(self.parent_widget._update_exposure_time)
            except (TypeError, AttributeError):
                pass  # Signal wasn't connected or widget doesn't exist
            try:
                channel_widget.power_input.valueChanged.disconnect(self.parent_widget._update_power)
            except (TypeError, AttributeError):
                pass
            try:
                channel_widget.excitation_wavelength_input.currentIndexChanged.disconnect(self.parent_widget._update_excitation_wavelength)
            except (TypeError, AttributeError):
                pass
            try:
                channel_widget.emission_wavelength_input.currentIndexChanged.disconnect(self.parent_widget._update_emission_wavelength)
            except (TypeError, AttributeError):
                pass

if __name__ == "__main__":
    # Example usage
    from fibsem.fm.microscope import FluorescenceMicroscope
    from PyQt5.QtWidgets import QApplication
    fm = FluorescenceMicroscope()
    channel_settings = ChannelSettings(name="Example Channel",
                                       excitation_wavelength=488,
                                       emission_wavelength=520,
                                       power=0.05,
                                       exposure_time=0.01)

    app = QApplication([])
    widget = ChannelSettingsWidget(fm=fm, channel_settings=channel_settings)
    widget.show()
    app.exec_()
