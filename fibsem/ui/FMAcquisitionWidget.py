import logging
from typing import Union, List, Dict, Optional
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
from fibsem.constants import METRE_TO_MILLIMETRE, MILLIMETRE_TO_METRE
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.structures import BeamType, Point
from fibsem.ui.napari.utilities import (
    draw_crosshair_in_napari,
    is_position_inside_layer,
)
from fibsem.ui.FibsemMovementWidget import to_pretty_string_short
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)
from fibsem.ui.utils import message_box_ui

# TODO: allow the user to select the colormap
def wavelength_to_color(wavelength: Union[int, float]) -> str:
    """Convert a wavelength in nm to a color string."""
    if wavelength is None:
        return "gray"
    
    # Simple mapping of wavelengths to colors (this can be extended)
    if wavelength < 400:
        return "violet"
    elif 400 <= wavelength < 450:
        return "blue"
    elif 450 <= wavelength < 500:
        return "cyan"
    elif 500 <= wavelength < 550:
        return "green"
    elif 550 <= wavelength < 600:
        return "yellow"
    elif 600 <= wavelength < 700:
        return "red"
    else:
        return "gray"


OBJECTIVE_CONFIG = {
    "step_size": 0.001,  # mm
    "decimals": 3,  # number of decimal places
    "suffix": " mm",  # unit suffix
}


class ObjectiveControlWidget(QWidget):    
    def __init__(self, fm: FluorescenceMicroscope, parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.fm = fm
        self.parent = parent
        self.initUI()

    def initUI(self):
        
        self.label_header = QLabel("Objective", self)
        self.pushButton_insert_objective = QPushButton("Insert Objective", self)
        self.pushButton_retract_objective = QPushButton("Retract Objective", self)

        # add double spin box for objective position
        self.label_objective_control = QLabel("Position", self)
        self.doubleSpinBox_objective_position = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MILLIMETRE,
                                                        self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MILLIMETRE)  # Convert m to mm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["suffix"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # Add a label to display the current objective position
        self.label_objective_position = QLabel(f"Current Objective Position: {self.fm.objective.position*METRE_TO_MILLIMETRE:.2f} mm", self)

        # TODO: add step-size controls

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 2)
        layout.addWidget(self.pushButton_insert_objective, 1, 0)
        layout.addWidget(self.pushButton_retract_objective, 1, 1)
        layout.addWidget(self.label_objective_control, 2, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 2, 1) 
        layout.addWidget(self.label_objective_position, 3, 0, 1, 2)
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)

        # set stylesheets
        self.pushButton_insert_objective.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_retract_objective.setStyleSheet(RED_PUSHBUTTON_STYLE)

    def insert_objective(self):
        """Insert the objective."""

        # confirmation dialog
        ret = message_box_ui(
            title="Insert Objective",
            text="Are you sure you want to insert the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective insertion cancelled by user.")
            return

        logging.info("Inserting objective...")
        self.fm.objective.insert()
        logging.info("Objective inserted.")
        self.update_objective_position_labels()

    def retract_objective(self):
        """Retract the objective."""
        # confirmation dialog
        ret = message_box_ui(
            title="Retract Objective",
            text="Are you sure you want to retract the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective retraction cancelled by user.")
            return

        self.fm.objective.retract()
        logging.info("Objective retracted.")
        self.update_objective_position_labels()

    def update_objective_position_labels(self):
        """Update the objective position input and label."""
        objective_position = self.fm.objective.position * METRE_TO_MILLIMETRE  # Convert m to mm
        self.doubleSpinBox_objective_position.blockSignals(True)  # Block signals to prevent recursion
        self.doubleSpinBox_objective_position.setValue(objective_position)  # Convert m to mm
        self.doubleSpinBox_objective_position.blockSignals(False)  # Unblock signals
        self.label_objective_position.setText(f"Current Objective Position: {objective_position:.2f} mm")

        self.parent.display_stage_position_overlay()  # Update the stage position overlay in the parent widget

    @pyqtSlot(float)
    def on_objective_position_changed(self, position: float):
        """Handle changes to the objective position."""

        is_large_change = abs(self.fm.objective.position - (position * MILLIMETRE_TO_METRE)) > 1e-3  # 1 mm threshold

        if is_large_change:
            logging.info(f"Large change in objective position requested: {position:.2f} mm")
        
            ret = message_box_ui(
                title="Large Objective Movement",
                text=f"Are you sure you want to change the objective position to {position:.2f} mm?",
                parent=self)
            
            if ret is False:
                logging.info("Objective position change cancelled by user.")
                # Reset the spin box to the current position
                self.update_objective_position_labels()
                return

        logging.info(f"Changing objective position to: {position:.2f} mm")
        self.fm.objective.move_absolute(position * MILLIMETRE_TO_METRE)
        logging.info(f"Objective moved to position: {position:.2f} mm")

        # Update the objective position label
        self.update_objective_position_labels()


class ChannelSettingsWidget(QWidget):
    def __init__(self, fm: FluorescenceMicroscope, channel_settings: ChannelSettings, parent=None):
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.fm = fm
        self.initUI()

        self.setContentsMargins(0, 0, 0, 0)

    def initUI(self):
        layout = QGridLayout()

        self.setLayout(layout)

        # add grid layout
        layout.addWidget(QLabel("Channel"), 0, 0)
        self.channel_name_input = QLineEdit(self.channel_settings.name)
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
        self.power_input.setSuffix(" W")

        layout.addWidget(self.power_input, 3, 1)
        
        layout.addWidget(QLabel("Exposure Time"), 4, 0)
        self.exposure_time_input = QDoubleSpinBox()
        self.exposure_time_input.setRange(0.01, 10.0)
        self.exposure_time_input.setSingleStep(0.01)
        self.exposure_time_input.setSuffix(" s")
        layout.addWidget(self.exposure_time_input, 4, 1)

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
    
    @pyqtSlot()
    def update_channel_name(self):
        self.channel_settings.name = self.channel_name_input.text()
        print(f"Channel name updated to: {self.channel_settings.name}")
   
class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.channel_name: str
        self.fm = fm
        self.viewer = viewer
        self.image_layer = None  # Placeholder for the image layer

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("FM Acquisition Widget", self)

        # Add objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.objectiveControlWidget.setContentsMargins(0, 0, 0, 0)

        # create channel settings widget
        channel_settings=ChannelSettings(
                name="Channel-01",
                excitation_wavelength=450,  # Example wavelength in nm
                emission_wavelength=550,  # Example wavelength in nm
                power=0.5,  # Example power in W
                exposure_time=0.1,  # Example exposure time in seconds
        )
        self.channelSettingsWidget = ChannelSettingsWidget(
            fm=self.fm,
            channel_settings=channel_settings,
            parent=self
        )
    
        self.startButton = QPushButton("Start Acquisition", self)
        self.stopButton = QPushButton("Stop Acquisition", self)

        layout.addWidget(self.label)
        layout.addWidget(self.objectiveControlWidget)
        layout.addWidget(self.channelSettingsWidget)
        layout.addWidget(self.startButton)
        layout.addWidget(self.stopButton)
        self.setLayout(layout)

        # connect signals
        self.channelSettingsWidget.exposure_time_input.valueChanged.connect(self._update_exposure_time)
        self.channelSettingsWidget.power_input.valueChanged.connect(self._update_power)
        self.channelSettingsWidget.excitation_wavelength_input.currentIndexChanged.connect(self._update_excitation_wavelength)
        self.channelSettingsWidget.emission_wavelength_input.currentIndexChanged.connect(self._update_emission_wavelength)
        self.startButton.clicked.connect(self.start_acquisition)
        self.stopButton.clicked.connect(self.stop_acquisition)

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(lambda image: self.update_image_signal.emit(image)) 
        self.update_image_signal.connect(self.update_image)

        # movement controls
        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)
        self.viewer.mouse_wheel_callbacks.append(self.on_mouse_wheel)

        # stylesheets
        self.startButton.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.stopButton.setStyleSheet(RED_PUSHBUTTON_STYLE)

        # draw scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

    def on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events in the napari viewer."""
        # no image layer available yet
        if self.image_layer is None:
            return
        
        # NOTE: scroll wheel events don't seem connected until there is an image layer?

        # Check for Ctrl key to control objective position
        if 'Control' in event.modifiers:
            # TODO: add shift key to change step size?

            # Calculate step size based on wheel delta
            step_mm = np.clip(0.001 * event.delta[1], -0.01, 0.01)  # Adjust sensitivity as needed
            # delta ~= 1
            
            logging.info(f"Mouse wheel event detected with delta: {event.delta}, step size: {step_mm:.4f} mm")

            # Get current position in mm
            current_pos = self.fm.objective.position * METRE_TO_MILLIMETRE
            new_pos = current_pos + step_mm
            
            # Move objective
            logging.info(f"Moving objective by {step_mm:.4f} mm to {new_pos:.4f} mm")
            self.objectiveControlWidget.doubleSpinBox_objective_position.setValue(new_pos)
            
            # Consume the event to prevent default napari behavior
            event.handled = True

    def on_mouse_double_click(self, viewer, event):
        """Handle double-click events in the napari viewer."""

        # no image layer available yet
        if self.image_layer is None:
            return

        logging.info(f"Mouse double-clicked at {event.position} in viewer {viewer}")
        self.image_layer = self.viewer.layers[self.channel_name]

        # NOTE: this doesn't matter, as long as the image layer is present
        # TODO: remove this limitation once tested. We should still put a max range?
        if not is_position_inside_layer(event.position, self.image_layer):
            logging.warning("Click position is outside the image layer.")
            return

        # changed by binning...
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

        # NOTE: this assumes the stage is always at the center of the image... may to not be true
        # need infinite plane -> use absolute coordinates
        if self.fm.parent is None:
            logging.error("FluorescenceMicroscope parent is None. Cannot move stage.")
            return

        # move the stage to the clicked position
        # TODO: we need to handle this better as scan rotation is not handled here
        # TODO: handle for the arctis? -> BeamType.ELECTRON?
        # TODO: create a FM version of this method accounting for these differences
        is_compustage = self.fm.parent.stage_is_compustage
        beam_type = BeamType.ELECTRON if is_compustage else BeamType.ION

        self.fm.parent.stable_move(dx=point_clicked.x, dy=point_clicked.y,beam_type=beam_type)
        logging.info(f"Microscope position: {self.fm.parent.get_stage_position()}")
        # TODO: multi-channels?

        self.display_stage_position_overlay()

    def display_stage_position_overlay(self):
        """Display the stage position as text overlay on the image widget"""
        try:
            # NOTE: this crashes for tescan systems?
            pos = self.fm.parent.get_stage_position()
            orientation = self.fm.parent.get_stage_orientation()
        except Exception as e:
            logging.warning(f"Error getting stage position: {e}")
            return

        if self.image_layer is None:
            return

        # add text layer, showing the stage position in cyan
        scale_y, scale_x = self.image_layer.scale
        points = np.array([[self.image_layer.data.shape[0] * scale_y, 0]])
        text = {
            "string": [
                f"STAGE: {to_pretty_string_short(pos)} [{orientation}]"
                f"\nOBJECTIVE: {self.fm.objective.position*1e3:.2f} mm",
                ],
            "color": "white",
            "font_size": 50,
            "anchor": "lower_left",
            "translation": (5*scale_y, 0),  # Adjust translation if needed
        }
        try:
            self.viewer.layers['stage_position'].data = points
            self.viewer.layers['stage_position'].text = text
        except KeyError:
            self.viewer.add_points(
                data=points,
                name="stage_position",
                size=20,
                text=text,
                border_width=7,
                border_width_is_relative=False,
                border_color="transparent",
                face_color="transparent",
                translate=self.image_layer.translate,
            )   

    def draw_crosshair(self):
        """Draw a crosshair at the center of the image using a single shapes layer."""
        if self.image_layer is None:
            return
            
        # Get image dimensions
        height, width = self.image_layer.data.shape
        
        # Calculate center in data coordinates (pixel coordinates)
        center_y = height / 2
        center_x = width / 2
        
        # Create horizontal line (across width) in data coordinates
        horizontal_line = np.array([
            [center_y, 0],
            [center_y, width]
        ])
        
        # Create vertical line (across height) in data coordinates
        vertical_line = np.array([
            [0, center_x],
            [height, center_x]
        ])
        
        # Combine both lines into a single shapes layer
        crosshair_lines = [horizontal_line, vertical_line]
        
        # Add or update crosshair layer
        try:
            self.viewer.layers['crosshair'].data = crosshair_lines
        except KeyError:
            self.viewer.add_shapes(
                data=crosshair_lines,
                name="crosshair",
                shape_type="line",
                edge_color="yellow",
                edge_width=2,
                face_color="transparent",
                scale=self.image_layer.scale,
                translate=self.image_layer.translate
            )

    # NOTE: not in main thread, so we need to handle signals properly
    @pyqtSlot(FluorescenceImage)
    def update_image(self, image: FluorescenceImage):
        
        acq_date = image.metadata.acquisition_date
        self.label.setText(f"Acquisition Signal Received: {acq_date}")
        logging.info(f"Image updated with shape: {image.data.shape}, Objective position: {self.fm.objective.position*1e3:.2f} mm")
        logging.info(f"Metadata: {image.metadata.channels[0].to_dict()}")

        # Convert structured metadata to dictionary for napari compatibility
        metadata_dict = image.metadata.to_dict() if image.metadata else {}

        channel_name = image.metadata.channels[0].name # QUERY: is this
        wavelength = image.metadata.channels[0].excitation_wavelength
        logging.info(f"Updating image layer with channel name: {channel_name}, wavelength: {wavelength} nm")

        if channel_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[channel_name].data = image.data
            self.viewer.layers[channel_name].metadata = metadata_dict
            self.viewer.layers[channel_name].colormap = wavelength_to_color(wavelength)
        else:
            # If the layer does not exist, create a new one
            self.image_layer = self.viewer.add_image(
                data=image.data,
                name=channel_name,
                metadata=metadata_dict,
                colormap=wavelength_to_color(wavelength),
                scale=(image.metadata.pixel_size_y, image.metadata.pixel_size_x),
            )


        
        self.channel_name = channel_name

        self.display_stage_position_overlay()
        self.draw_crosshair()

    def start_acquisition(self):
        """Start the fluorescence acquisition."""
        if self.fm.is_acquiring:
            logging.warning("Acquisition is already running.")
            return
        
        logging.info("Acquisition started")
        self.startButton.setEnabled(False)
        self.startButton.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.stopButton.setEnabled(True)
        # TODO: handle case where acquisition fails...

        channel_settings = self.channelSettingsWidget.channel_settings
        logging.info(f"Starting acquisition with channel settings: {channel_settings}")

        self.fm.start_acquisition(channel_settings=channel_settings)

    def stop_acquisition(self):
        """Stop the fluorescence acquisition."""

        logging.info("Acquisition stopped")
        self.stopButton.setEnabled(False)
        self.stopButton.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.startButton.setEnabled(True)
        self.startButton.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

        self.fm.stop_acquisition()

    # update methods for live updates
    def _update_exposure_time(self, value: float):
        if self.fm.is_acquiring:
            self.fm.set_exposure_time(value)

    def _update_power(self, value: float):
        if self.fm.is_acquiring:
            self.fm.set_power(value)

    def _update_excitation_wavelength(self, idx: int):
        if self.fm.is_acquiring:
            wavelength = self.channelSettingsWidget.excitation_wavelength_input.itemData(idx)
            logging.info(f"Updating excitation wavelength to: {wavelength} nm")
            self.fm.filter_set.excitation_wavelength = wavelength

    def _update_emission_wavelength(self, idx: int):
        if self.fm.is_acquiring:
            wavelength = self.channelSettingsWidget.emission_wavelength_input.itemData(idx)
            logging.info(f"Updating emission wavelength to: {wavelength} nm")
            self.fm.filter_set.emission_wavelength = wavelength

    # override closeEvent to stop acquisition when the widget is closed
    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info("Closing FMAcquisitionWidget, stopping acquisition if running.")
        if self.fm.is_acquiring:
            try:
                self.fm.acquisition_signal.disconnect()
                self.stop_acquisition()
                print("Acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                print("Acquisition stopped due to widget close.")

        event.accept()

def main():

    microscope, settings = utils.setup_session()
    viewer = napari.Viewer()
    widget = FMAcquisitionWidget(fm=microscope.fm, viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    return


if __name__ == "__main__":
    main()
    # main2()




