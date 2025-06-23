import logging

import napari
import numpy as np
from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions, utils
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.structures import BeamType, Point
from fibsem.ui.napari.utilities import is_position_inside_layer


class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.fm = fm
        self.viewer = viewer
        self.channel_name = "Channel-1"  # Default channel name, can be changed later

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("FM Acquisition Widget", self)
        layout.addWidget(self.label)

        self.startButton = QPushButton("Start Acquisition", self)
        layout.addWidget(self.startButton)

        self.stopButton = QPushButton("Stop Acquisition", self)
        layout.addWidget(self.stopButton)

        # add insert and retraact buttons for objective
        self.insertButton = QPushButton("Insert Objective", self)
        self.retractButton = QPushButton("Retract Objective", self)
        layout.addWidget(self.insertButton)
        layout.addWidget(self.retractButton)

        # Connect the insert and retract buttons to the microscope's objective
        self.insertButton.clicked.connect(lambda: self.fm.objective.insert())
        self.retractButton.clicked.connect(lambda: self.fm.objective.retract())

        # add double spin box for objective position
        self.objectivePositionInput = QDoubleSpinBox(self)
        self.objectivePositionInput.setRange(-15, 5)
        self.objectivePositionInput.setSingleStep(0.01)
        self.objectivePositionInput.setSuffix(" mm")
        self.objectivePositionInput.setValue(self.fm.objective.position)
        self.objectivePositionInput.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        layout.addWidget(QLabel("Objective Position (mm):"))
        layout.addWidget(self.objectivePositionInput)

        # Connect the objective position input to the microscope's objective
        self.objectivePositionInput.valueChanged.connect(
            lambda value: self.fm.objective.move_absolute(value*1e-3)  # Convert mm to m
        )
        # Add a label to display the current objective position
        self.objectivePositionLabel = QLabel(f"Current Objective Position: {self.fm.objective.position*1e3:.2f} mm", self)
        layout.addWidget(self.objectivePositionLabel)

        self.setLayout(layout)

        # Connect buttons to their respective methods
        self.startButton.clicked.connect(self.start_acquisition)
        self.stopButton.clicked.connect(self.stop_acquisition)
        # Connect the acquisition signal to the handler
        self.fm.acquisition_signal.connect(self.on_acquisition_signal)
        self.update_image_signal.connect(self.update_image)

        # movement controls
        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)

    def on_mouse_double_click(self, viewer, event):
        """Handle double-click events in the napari viewer."""

        logging.info(f"Mouse double-clicked at {event.position} in viewer {viewer}")
        self.image_layer = self.viewer.layers[self.channel_name]

        if not is_position_inside_layer(event.position, self.image_layer):
            logging.warning("Click position is outside the image layer.")
            return

        resolution = self.fm.camera.resolution
        pixelsize = self.fm.camera.pixel_size[0]

        # convert from image coordinates to microscope coordinates
        coords = self.image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=np.zeros((resolution[1], resolution[0])),  # dummy image for shape
            pixelsize=pixelsize,
        )
        logging.info(f"Mouse double-clicked at coordinates: {point_clicked}")

        self.fm.parent.stable_move(
            dx=point_clicked.x,
            dy=point_clicked.y,
            beam_type=BeamType.ION, 
        )
        # TODO: we need to handle this better as scan rotation is not handled here
        # TODO: handle for the arctis? -> BeamType.ELECTRON?

        logging.info(f"Microscope position: {self.fm.parent.get_stage_position()}")

    # NOTE: not in main thread, so we need to handle signals properly
    def on_acquisition_signal(self, image: FluorescenceImage):
        # Placeholder for handling acquisition signal
        acq_date = image.metadata.acquisition_date if image.metadata else "Unknown"
        self.label.setText(f"Acquisition Signal Received: {acq_date}")
        logging.info(f"Acquisition signal at {acq_date} received with image:  {image.data.shape}")

        # Here you would typically process the image or update the UI accordingly
        # For example, you might display the image in a QLabel or similar widget
        self.update_image_signal.emit(image)

    def update_image(self, image: FluorescenceImage):
        # Placeholder for updating the image in the viewer
        logging.info(f"Image updated with shape: {image.data.shape}, Objective position: {self.fm.objective.position*1e3:.2f} mm")

        # Convert structured metadata to dictionary for napari compatibility
        metadata_dict = image.metadata.to_dict() if image.metadata else {}

        channel_name = image.metadata.channels[0].name # QUERY: is this 

        # Here you would typically add the image to the napari viewer
        if channel_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[channel_name].data = image.data
            self.viewer.layers[channel_name].metadata = metadata_dict
        else:
            # If the layer does not exist, create a new one
            self.image_layer = self.viewer.add_image(
                data=image.data,
                name=channel_name,
                metadata=metadata_dict
            )
        
        self.channel_name = channel_name


    def start_acquisition(self):
        # Placeholder for starting acquisition logic
        self.label.setText("Acquisition Started")
        logging.info("Acquisition started")

        self.fm.start_acquisition()

    def stop_acquisition(self):
        # Placeholder for stopping acquisition logic
        self.label.setText("Acquisition Stopped")
        logging.info("Acquisition stopped")

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

    microscope, settings = utils.setup_session()
    viewer = napari.Viewer()
    widget = FMAcquisitionWidget(fm=microscope.fm, viewer=viewer)
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




