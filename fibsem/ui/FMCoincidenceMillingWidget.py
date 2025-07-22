import copy
import logging
import threading
from pprint import pprint
from typing import Optional, Union

import napari
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import acquire, conversions, utils
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.microscope import FluorescenceImage
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.milling.base import FibsemMillingSettings
from fibsem.milling.patterning import TrenchPattern
from fibsem.milling.strategy import CoincidenceMillingStrategy
from fibsem.structures import BeamType, FibsemImage, ImageSettings, Point
from fibsem.ui.FibsemMillingStageEditorWidget import FibsemMillingStageEditorWidget
from fibsem.ui.fm.widgets import LinePlotWidget
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)

milling_stage = FibsemMillingStage(name="Milling Stage",
                                  milling=FibsemMillingSettings(milling_current=60e-12),
                                  pattern=TrenchPattern(),
                                  strategy=CoincidenceMillingStrategy())
milling_stage.alignment.enabled = False



class FMCoincidenceMillingWidget(QWidget):
    """Widget for FM Coincidence Milling with FIB image acquisition, FM image acquisition, and start/stop milling controls."""
    
    fib_image_acquired_signal = pyqtSignal(FibsemImage)
    image_acquired_signal = pyqtSignal(FluorescenceImage)
    milling_state_changed_signal = pyqtSignal(bool)  # True when milling starts, False when stops
    pattern_update_signal = pyqtSignal()

    def __init__(self, microscope: FibsemMicroscope, viewer: napari.Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.microscope = microscope
        self.viewer = viewer
        
        if microscope.fm is None:
            raise ValueError("FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope.")
        
        field_of_view = microscope.fm.camera.field_of_view

        # Default settings
        self.fib_image_settings = ImageSettings(
            resolution=(1536, 1024),
            dwell_time=1e-6,
            hfw=field_of_view[0],
            beam_type=BeamType.ION
        )
        
        self.channel_settings = ChannelSettings(
            name="Channel-01",
            excitation_wavelength=550,
            emission_wavelength=None,
            power=0.003,
            exposure_time=0.1,
        )

        # self.milling_stage = milling_stage

        self.milling_stage_editor = FibsemMillingStageEditorWidget(viewer=self.viewer, 
                                                    microscope=self.microscope, 
                                                    milling_stages=[milling_stage],
                                                    parent=self)
        self.milling_stage_editor.image_layer.translate = (0, self.microscope.fm.camera.resolution[0])
        
        # milling threading
        self._milling_thread: Optional[threading.Thread] = None
        self._milling_stop_event = threading.Event()

        self.initUI()
        self.connect_signals()

        self.line_plot_widget = LinePlotWidget(parent=self)
        self.line_plot_dock = self.viewer.window.add_dock_widget(self.line_plot_widget, name="Line Plot", area='left', tabify=True)

    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Create button layout
        button_layout = QVBoxLayout()
        
        # Acquire FIB Image button
        self.acquire_fib_button = QPushButton("Acquire FIB Image")
        self.acquire_fib_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.acquire_fib_button.clicked.connect(self.acquire_fib_image)
        
        # Acquire FM Image button  
        self.acquire_fm_button = QPushButton("Acquire FM Image")
        self.acquire_fm_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.acquire_fm_button.clicked.connect(self.acquire_fm_image)
        
        # Start/Stop Milling button
        self.milling_button = QPushButton("Start Milling")
        self.milling_button.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.milling_button.clicked.connect(self.start_milling)

        self.stop_milling_button = QPushButton("Stop Milling")
        self.stop_milling_button.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.stop_milling_button.clicked.connect(self.stop_milling)
        
        self.pause_milling_button = QPushButton("Pause Milling")
        self.pause_milling_button.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)
        self.pause_milling_button.clicked.connect(self.pause_milling)
        self.resume_milling_button = QPushButton("Resume Milling")
        self.resume_milling_button.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.resume_milling_button.clicked.connect(self.resume_milling)

        self.label_milling_state = QPushButton("Milling State: Idle")
        self.label_milling_state.setEnabled(False)
        self.label_milling_state.setStyleSheet("background-color: lightgray; color: black;")

        button_layout.addWidget(self.acquire_fib_button)
        button_layout.addWidget(self.acquire_fm_button)
        button_layout.addWidget(self.milling_button)
        button_layout.addWidget(self.stop_milling_button)
        button_layout.addWidget(self.pause_milling_button)
        button_layout.addWidget(self.resume_milling_button)
        button_layout.addWidget(self.label_milling_state)
        layout.addLayout(button_layout)
        layout.addWidget(self.milling_stage_editor)
        self.setLayout(layout)
    
    def connect_signals(self):
        """Connect internal signals."""
        self.microscope.fm.acquisition_signal.connect(lambda image: self.image_acquired_signal.emit(image)) 
        self.image_acquired_signal.connect(self.update_image)
        self.fib_image_acquired_signal.connect(self.update_image)
        # self.pattern_update_signal.connect(self.draw_patterns)
        # self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)

    @property
    def is_milling(self) -> bool:
        """Check if milling is currently running."""
        return self._milling_thread is not None and self._milling_thread.is_alive()

    def update_image(self, image: Union[FibsemImage, FluorescenceImage]):

        if isinstance(image, FluorescenceImage):
            # Add to napari viewer
            layer_name = f"FM Image {self.channel_settings.name}"
            translation = (0, 0)  # No translation for FM images
            if self.is_milling:
                self.line_plot_widget.append_value(float(np.mean(image.data)))

        elif isinstance(image, FibsemImage):
            # Add to napari viewer
            layer_name = "FIB Image"
            translation = (0, self.microscope.fm.camera.resolution[0])  # Adjust translation based on camera resolution
    
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = image.data
            self.viewer.layers[layer_name].metadata = image.metadata.to_dict()
            self.viewer.layers[layer_name].translate = translation
        else:
            self.viewer.add_image(
                image.data,
                name=layer_name,
                metadata=image.metadata.to_dict(),
                colormap='gray',
                translate=translation,
            )

        if isinstance(image, FibsemImage):
            # self.pattern_update_signal.emit()
            self.milling_stage_editor.image_layer = self.viewer.layers[layer_name]
            self.milling_stage_editor.set_image(image)
            self.milling_stage_editor.update_milling_stage_display()

    @pyqtSlot()
    def acquire_fib_image(self):
        """Acquire a FIB image."""
        try:
            self.acquire_fib_button.setEnabled(False)
            
            # Acquire FIB image
            fib_image = acquire.acquire_image(
                microscope=self.microscope,
                settings=self.fib_image_settings
            )
            self.fib_image_acquired_signal.emit(fib_image)
            logging.info("FIB image acquired successfully")
        except Exception as e:
            logging.error(f"Error acquiring FIB image: {e}")
        finally:
            self.acquire_fib_button.setEnabled(True)
    
    @pyqtSlot()
    def acquire_fm_image(self):
        """Acquire a fluorescence microscopy image."""
        try:
            self.acquire_fm_button.setEnabled(False)
            
            if self.microscope.fm is None:
                raise ValueError("Fluorescence microscope is not available")
            
            # Acquire FM image
            fm_image = acquire_image(
                microscope=self.microscope.fm,
                channel_settings=self.channel_settings
            )
            self.image_acquired_signal.emit(fm_image)
            
            logging.info("FM image acquired successfully")
            
        except Exception as e:
            logging.error(f"Error acquiring FM image: {e}")
        finally:
            self.acquire_fm_button.setEnabled(True)
    
    @pyqtSlot()
    def start_milling(self):
        """Start milling."""
        try:
            if not self.is_milling:
                # Start milling
                self.milling_state_changed_signal.emit(True)
                self.label_milling_state.setText("Milling State: Running")
                self.label_milling_state.setStyleSheet("background-color: lightgreen; color: black;")
                logging.info("Milling started")

                milling_stage = copy.deepcopy(self.milling_stage_editor.get_milling_stages()[0])
                pprint(milling_stage.to_dict())

                self._milling_thread = threading.Thread(
                    target=self._milling_worker,
                    args=(milling_stage,),
                    daemon=True
                )
                self._milling_thread.start()
                logging.info("Milling worker thread started")
    
        except Exception as e:
            logging.error(f"Error toggling milling: {e}")

    def stop_milling(self):
        """Stop milling."""
        if not self.is_milling:
            logging.warning("Milling is not currently running.")
            return
        
        try:
            # Stop milling
            self.microscope.stop_milling()
            self.label_milling_state.setText("Milling State: Stopped")
            self.label_milling_state.setStyleSheet("background-color: red; color: white;")
            self.milling_state_changed_signal.emit(False)
            logging.info("Milling stopped")
        except Exception as e:
            logging.error(f"Error stopping milling: {e}")

    def pause_milling(self):
        """Pause milling."""
        if not self.is_milling:
            logging.warning("Milling is not currently running.")
            return
        
        try:
            # Pause milling
            self.microscope.pause_milling()
            self.label_milling_state.setText("Milling State: Paused")
            self.label_milling_state.setStyleSheet("background-color: yellow; color: black;")
            logging.info("Milling paused")
        except Exception as e:
            logging.error(f"Error pausing milling: {e}")

    def resume_milling(self):
        """Resume milling."""
        if not self.is_milling:
            logging.warning("Milling is not currently running.")
            return

        try:
            # Resume milling
            self.microscope.resume_milling()
            self.label_milling_state.setText("Milling State: Running")
            self.label_milling_state.setStyleSheet("background-color: lightgreen; color: black;")
            self.milling_state_changed_signal.emit(True)
            logging.info("Milling resumed")
        except Exception as e:
            logging.error(f"Error resuming milling: {e}")

    def _milling_worker(self, milling_stage: FibsemMillingStage):
        """Worker function to run the milling strategy."""
        try:
            milling_stage.strategy.run(
                microscope=self.microscope,
                stage=milling_stage,
                asynch=False,
                parent_ui=self
            )
        except Exception as e:
            logging.error(f"Error running milling strategy: {e}")
    
    def set_fib_image_settings(self, settings: ImageSettings):
        """Set the FIB image acquisition settings."""
        self.fib_image_settings = settings
    
    def set_channel_settings(self, settings: ChannelSettings):
        """Set the FM channel settings."""
        self.channel_settings = settings
    
    def get_fib_image_settings(self) -> ImageSettings:
        """Get the current FIB image settings."""
        return self.fib_image_settings
    
    def get_channel_settings(self) -> ChannelSettings:
        """Get the current FM channel settings."""
        return self.channel_settings

    # def draw_patterns(self):
    #     """Draw milling patterns in the napari viewer."""
    #     if self.microscope.fm is None:
    #         logging.error("FluorescenceMicroscope is not initialized. Cannot draw patterns.")
    #         return
        
    #     if "FIB Image" not in self.viewer.layers:
    #         return  # No FIB image layer to draw patterns on

    #     fib_image_layer: NapariImageLayer = self.viewer.layers["FIB Image"]

    #     # Draw the milling patterns in napari
    #     try:
    #         milling_layers = draw_milling_patterns_in_napari(
    #             viewer=self.viewer, 
    #             image_layer=fib_image_layer, 
    #             milling_stages= [self.milling_stage],
    #             pixelsize=fib_image_layer.metadata["pixel_size"]["x"]
    #         )
    #     except Exception as e:
    #         logging.error(f"Error drawing milling patterns: {e}")
    #         return

    def on_mouse_click(self, viewer: napari.Viewer, event):
        """Handle mouse click events in the napari viewer."""
        # Check if the click is a left mouse button click with Shift modifier
        if not (event.type == 'mouse_press' and event.button == 1 and "Shift" in event.modifiers):
            return

        if "FIB Image" not in self.viewer.layers:
            logging.error("No FIB Image layer found in the viewer.")
            return
        
        # Get the coordinates of the mouse click in microscope image coordinates
        image_layer = self.viewer.layers["FIB Image"]
        pixelsize = image_layer.metadata["pixel_size"]["x"]
        # convert from image coordinates to microscope coordinates
        coords = image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=image_layer.data,
            pixelsize=pixelsize,
        )
        
        # TODO: validate the point placement
        # update the pattern point, and re-draw the patterns
        self.milling_stage.pattern.point = point_clicked
        self.pattern_update_signal.emit()


def create_widget(viewer: napari.Viewer) -> FMCoincidenceMillingWidget:
    """Create the FMCoincidenceMillingWidget with a demo microscope."""
    CONFIG_PATH = None  # Use default config
    microscope, settings = utils.setup_session(config_path=CONFIG_PATH)
    
    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized. Cannot create FMCoincidenceMillingWidget.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")
    
    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")
    
    # Ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True
    
    widget = FMCoincidenceMillingWidget(
        microscope=microscope,
        viewer=viewer,
        parent=None
    )
    return widget


def main():
    """Main function to run the widget standalone."""
    viewer = napari.Viewer()
    widget = create_widget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
    return


if __name__ == "__main__":
    main()
