
import logging
from typing import Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings


class ChannelSettingsWidget(QWidget):
    def __init__(self,
                 fm: FluorescenceMicroscope,
                 channel_settings: ChannelSettings,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.fm = fm
        self.initUI()

    def initUI(self):
        """Initialize the UI components for the channel settings widget."""
        self.setContentsMargins(0, 0, 0, 0)
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # add grid layout
        layout.addWidget(QLabel("Channel"), 0, 0)
        self.channel_name_input = QLineEdit(self.channel_settings.name, self)
        self.channel_name_input.setPlaceholderText("Enter channel name")
        layout.addWidget(self.channel_name_input, 0, 1)

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
        self.power_input.setRange(0.0, 1.0)
        self.power_input.setSingleStep(0.01)
        self.power_input.setDecimals(3)
        self.power_input.setSuffix(" W")

        layout.addWidget(self.power_input, 3, 1)
        
        layout.addWidget(QLabel("Exposure Time"), 4, 0)
        self.exposure_time_input = QDoubleSpinBox()
        self.exposure_time_input.setRange(0.01, 10.0)
        self.exposure_time_input.setSingleStep(0.01)
        self.exposure_time_input.setDecimals(3)
        self.exposure_time_input.setSuffix(" s")
        layout.addWidget(self.exposure_time_input, 4, 1)

        # Set column stretch factors to make widgets expand properly
        layout.setColumnStretch(0, 1)  # Labels column - expandable
        layout.setColumnStretch(1, 1)  # Input widgets column - expandable

        # connect signals to slots
        self.channel_name_input.textChanged.connect(self.update_channel_name)
        self.excitation_wavelength_input.currentIndexChanged.connect(self.update_excitation_wavelength)
        self.emission_wavelength_input.currentIndexChanged.connect(self.update_emission_wavelength)
        self.power_input.valueChanged.connect(self.update_power)
        self.exposure_time_input.valueChanged.connect(self.update_exposure_time)

        # set keyboard tracking to false for immediate updates
        self.power_input.setKeyboardTracking(False)
        self.exposure_time_input.setKeyboardTracking(False)

        self.power_input.setValue(self.channel_settings.power)
        self.exposure_time_input.setValue(self.channel_settings.exposure_time)

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
        self.channel_settings.exposure_time = value
        logging.info(f"Exposure time updated to: {value} s") 
    
    @pyqtSlot()
    def update_channel_name(self):
        self.channel_settings.name = self.channel_name_input.text()
        logging.info(f"Channel name updated to: {self.channel_settings.name}")