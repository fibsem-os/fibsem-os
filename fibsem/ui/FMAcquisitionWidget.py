import napari
from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QGridLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox

from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings


class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.fm = fm
        self.viewer = viewer

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("FMA Acquisition Widget", self)
        layout.addWidget(self.label)

        self.startButton = QPushButton("Start Acquisition", self)
        layout.addWidget(self.startButton)

        self.stopButton = QPushButton("Stop Acquisition", self)
        layout.addWidget(self.stopButton)

        self.setLayout(layout)

        # Connect buttons to their respective methods
        self.startButton.clicked.connect(self.start_acquisition)
        self.stopButton.clicked.connect(self.stop_acquisition)
        # Connect the acquisition signal to the handler
        self.fm.acquisition_signal.connect(self.on_acquisition_signal)
        self.update_image_signal.connect(self.update_image)

    # NOTE: not in main thread, so we need to handle signals properly
    def on_acquisition_signal(self, image: FluorescenceImage):
        # Placeholder for handling acquisition signal
        acq_date = image.metadata.acquisition_date if image.metadata else "Unknown"
        self.label.setText(f"Acquisition Signal Received: {acq_date}")
        print(f"Acquisition Date: {acq_date}")
        print(f"Acquisition signal received with image:  {image.data.shape}")

        # Here you would typically process the image or update the UI accordingly
        # For example, you might display the image in a QLabel or similar widget
        self.update_image_signal.emit(image)

    def update_image(self, image: FluorescenceImage):
        # Placeholder for updating the image in the viewer
        print(f"Image updated with shape: {image.data.shape}")

        # Convert structured metadata to dictionary for napari compatibility
        metadata_dict = image.metadata.to_dict() if image.metadata else {}

        # Here you would typically add the image to the napari viewer
        if "Fluorescence Image" in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers["Fluorescence Image"].data = image.data
            self.viewer.layers["Fluorescence Image"].metadata = metadata_dict
        else:
            # If the layer does not exist, create a new one
            self.viewer.add_image(
                data=image.data,
                name="Fluorescence Image",
                metadata=metadata_dict
            )

    def start_acquisition(self):
        # Placeholder for starting acquisition logic
        self.label.setText("Acquisition Started")
        print("Acquisition started")

        self.fm.start_acquisition()

    def stop_acquisition(self):
        # Placeholder for stopping acquisition logic
        self.label.setText("Acquisition Stopped")
        print("Acquisition stopped")

        self.fm.stop_acquisition()



class ChannelSettingsWidget(QWidget):
    def __init__(self, channel_settings: ChannelSettings, parent=None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.fm = FluorescenceMicroscope()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()

        self.setLayout(layout)

        # add grid layout
        layout.addWidget(QLabel("Channel"), 0, 0)
        layout.addWidget(QLineEdit(self.channel_settings.name), 0, 1)

        layout.addWidget(QLabel("Excitation Wavelength"), 1, 0)
        self.excitation_wavelength_input = QComboBox()
        for wavelength in self.fm.filter_set.available_excitation_wavelengths():
            self.excitation_wavelength_input.addItem(f"{wavelength} nm", wavelength)
        self.excitation_wavelength_input.setCurrentText(f"{self.channel_settings.excitation_wavelength} nm")

        layout.addWidget(self.excitation_wavelength_input, 1, 1)
        
        layout.addWidget(QLabel("Emission Wavelength"), 2, 0)
        self.emission_wavelength_input = QComboBox()
        for wavelength in self.fm.filter_set.available_emission_wavelengths():
            self.emission_wavelength_input.addItem(f"{wavelength} nm", wavelength)
        self.emission_wavelength_input.setCurrentText(f"{self.channel_settings.emission_wavelength} nm")
        layout.addWidget(self.emission_wavelength_input, 2, 1)
        
        layout.addWidget(QLabel("Power"), 3, 0)
        self.power_input = QDoubleSpinBox()
        self.power_input.setRange(0.0, 1.0)
        self.power_input.setSingleStep(0.01)
        self.power_input.setSuffix(" W")
        self.power_input.setValue(self.channel_settings.power)
        layout.addWidget(self.power_input, 3, 1)
        
        layout.addWidget(QLabel("Exposure Time"), 4, 0)
        self.exposure_time_input = QDoubleSpinBox()
        self.exposure_time_input.setRange(0.01, 10.0)
        self.exposure_time_input.setSingleStep(0.01)
        self.exposure_time_input.setSuffix(" s")
        self.exposure_time_input.setValue(self.channel_settings.exposure_time)
        layout.addWidget(self.exposure_time_input, 4, 1)

        # connect signals to slots
        self.excitation_wavelength_input.currentIndexChanged.connect(self.update_excitation_wavelength)
        self.emission_wavelength_input.currentIndexChanged.connect(self.update_emission_wavelength)
        self.power_input.valueChanged.connect(self.update_power)
        self.exposure_time_input.valueChanged.connect(self.update_exposure_time)

    @pyqtSlot(int)
    def update_excitation_wavelength(self, idx: int):
        wavelength = self.excitation_wavelength_input.itemData(idx)
        self.channel_settings.excitation_wavelength = wavelength
        print(f"Excitation wavelength updated to: {wavelength} nm")

    @pyqtSlot(int)      
    def update_emission_wavelength(self, idx: int):
        wavelength = self.emission_wavelength_input.itemData(idx)
        self.channel_settings.emission_wavelength = wavelength
        print(f"Emission wavelength updated to: {wavelength} nm")

    @pyqtSlot(float)
    def update_power(self, value: float):
        self.channel_settings.power = value
        print(f"Power updated to: {value} W")

    @pyqtSlot(float)
    def update_exposure_time(self, value: float):
        self.channel_settings.exposure_time = value
        print(f"Exposure time updated to: {value} s")   

def main():
    viewer = napari.Viewer()
    fm = FluorescenceMicroscope()
    widget = FMAcquisitionWidget(fm=fm, viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    return

    # import sys
    # from PyQt5.QtWidgets import QApplication
    # app = QApplication(sys.argv)
    # viewer = napari.Viewer()
    viewer = napari.Viewer()
    channel_settings = ChannelSettings(name="channel-01", 
                                       excitation_wavelength=488, 
                                       emission_wavelength=520, 
                                       power=0.5, exposure_time=0.1)
    channel_widget = ChannelSettingsWidget(channel_settings)

    # channel_widget.show()
    # sys.exit(app.exec_())
    viewer.window.add_dock_widget(channel_widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()




