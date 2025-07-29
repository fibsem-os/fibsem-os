import copy
import logging
import threading
from pprint import pprint
from typing import List, Optional, Union

import napari
from napari.layers import Shapes as NapariShapesLayer
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QEvent
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from fibsem import acquire, utils
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.microscope import FluorescenceImage
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.milling.base import FibsemMillingSettings
from fibsem.milling.patterning import TrenchPattern
from fibsem.milling.strategy import CoincidenceMillingStrategy
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
)
from fibsem.ui.FibsemMillingStageEditorWidget import FibsemMillingStageEditorWidget
from fibsem.ui.fm.widgets import LinePlotWidget
from fibsem.ui.napari.patterns import (
    convert_reduced_area_to_napari_shape,
    convert_shape_to_image_area,
)
from fibsem.ui.napari.utilities import update_text_overlay
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)

milling_stage = FibsemMillingStage(name="Milling Stage",
                                  milling=FibsemMillingSettings(hfw=50e-6, milling_current=60e-12),
                                  pattern=TrenchPattern(depth=0.5e-6, 
                                                        width=10e-6, 
                                                        spacing=0.25e-6, 
                                                        upper_trench_height=0.5e-6, 
                                                        lower_trench_height=0.5e-6, 
                                                        cross_section=CrossSectionPattern.CleaningCrossSection),
                                  strategy=CoincidenceMillingStrategy())
milling_stage.alignment.enabled = False

# TODO: enable channel settings for FM image acquisition
# TODO: enable objective control
# TODO: control trench milling pattern -> run top trench, then bottom trench

class FMCoincidenceMillingWidget(QWidget):
    """Widget for FM Coincidence Milling with FIB image acquisition, FM image acquisition, and start/stop milling controls."""
    fib_image_acquired_signal = pyqtSignal(FibsemImage)
    image_acquired_signal = pyqtSignal(FluorescenceImage)
    milling_state_changed_signal = pyqtSignal(bool)
    bbox_updated_signal = pyqtSignal(FibsemRectangle)
    update_line_plot_signal = pyqtSignal(float)

    def __init__(self, microscope: FibsemMicroscope,
                 milling_stages: List[FibsemMillingStage],
                 viewer: napari.Viewer,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.microscope = microscope
        self.viewer = viewer
        self._lock = threading.RLock()  # Lock for thread safety

        if microscope.fm is None:
            raise ValueError("FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope.")

        self.fm_resolution = self.microscope.fm.camera.resolution
        self.field_of_view = microscope.fm.camera.field_of_view

        # Default settings
        self.fib_image_settings = ImageSettings(
            resolution=(1536, 1024),
            dwell_time=1e-6,
            hfw=milling_stages[0].milling.hfw,
            beam_type=BeamType.ION
        )

        self.channel_settings = ChannelSettings(
            name="Channel-01",
            excitation_wavelength=550,
            emission_wavelength=None,
            power=0.003,
            exposure_time=0.1,
        )

        if isinstance(milling_stages, FibsemMillingStage):
            milling_stages = [milling_stages]

        self.milling_stage_editor = FibsemMillingStageEditorWidget(viewer=self.viewer, 
                                                    microscope=self.microscope, 
                                                    milling_stages=milling_stages,
                                                    parent=self)
        self.milling_stage_editor.image_layer.translate = (0, self.fm_resolution[0])
        self.alignment_layer: Optional[NapariShapesLayer] = None

        # milling threading
        self._milling_thread: Optional[threading.Thread] = None
        self._milling_stop_event = threading.Event()

        self.line_plot_widget = LinePlotWidget(parent=self)
        self.line_plot_dock = self.viewer.window.add_dock_widget(self.line_plot_widget, name="Line Plot", area='left', tabify=True)

        self.initUI()
        self.connect_signals()

        update_text_overlay(self.viewer, self.microscope)

    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Acquire FIB Image button
        self.acquire_fib_button = QPushButton("Acquire FIB Image")
        self.acquire_fib_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.acquire_fib_button.clicked.connect(self.acquire_fib_image)
        
        # Acquire FM Image button  
        self.acquire_fm_button = QPushButton("Acquire FM Image")
        self.acquire_fm_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.acquire_fm_button.clicked.connect(self.acquire_fm_image)

        self.show_fm_bbox_button = QPushButton("Show FM Region of Interest")
        self.show_fm_bbox_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.show_fm_bbox_button.clicked.connect(lambda: self.set_alignment_layer(editable=True))
        
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

        self.label_milling_state = QLabel("Milling State: Idle")
        self.label_milling_state.setEnabled(False)
        self.label_milling_state.setStyleSheet("color: black;")

        self.label_fm_bbox = QLabel("FM Region of Interest:")
        self.bbox_updated_signal.connect(self.on_bbox_update)
        self.label_fm_intensity = QLabel("Intensity Drop Threshold: 0.75")
        self.label_fm_bbox.setStyleSheet("color: white;")

        # Create button layout
        button_layout = QGridLayout()
        button_layout.addWidget(self.acquire_fm_button, 0, 0)
        button_layout.addWidget(self.acquire_fib_button, 0, 1)
        button_layout.addWidget(self.show_fm_bbox_button, 1, 0)
        button_layout.addWidget(self.milling_button, 2, 0)
        button_layout.addWidget(self.stop_milling_button, 2, 1)
        button_layout.addWidget(self.pause_milling_button, 3, 0)
        button_layout.addWidget(self.resume_milling_button, 3, 1)
        button_layout.addWidget(self.label_milling_state, 4, 0, 1, 2)
        button_layout.addWidget(self.label_fm_bbox, 5, 0, 1, 2)
        button_layout.addWidget(self.label_fm_intensity, 6, 0, 1, 2)
        layout.addWidget(self.milling_stage_editor)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def connect_signals(self):
        """Connect internal signals."""
        self.microscope.fm.acquisition_signal.connect(self._on_acquisition) 
        self.image_acquired_signal.connect(self.update_image)
        self.fib_image_acquired_signal.connect(self.update_image)
        self.update_line_plot_signal.connect(lambda value: self.line_plot_widget.append_value(value))

    def _on_acquisition(self, image: FluorescenceImage):
        """Handle the acquisition of a fluorescence image."""
        try:
            self.image_acquired_signal.emit(image)
        except Exception as e:
            logging.error(f"Error in _on_acquisition: {e}")
            self.microscope.fm.acquisition_signal.disconnect(self._on_acquisition)

    @property
    def is_milling(self) -> bool:
        """Check if milling is currently running."""
        return self._milling_thread is not None and self._milling_thread.is_alive()

    def update_image(self, image: Union[FibsemImage, FluorescenceImage]):

        if isinstance(image, FluorescenceImage):
            # Add to napari viewer
            layer_name = f"FM Image {self.channel_settings.name}"
            colormap = "green"
            translation = (0, 0)

        elif isinstance(image, FibsemImage):
            # Add to napari viewer
            layer_name = "FIB Image"
            translation = (0, self.microscope.fm.camera.resolution[0])  # Adjust translation based on camera resolution
            colormap = "gray"

        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = image.data
            self.viewer.layers[layer_name].metadata = image.metadata.to_dict()
            self.viewer.layers[layer_name].translate = translation
        else:
            self.viewer.add_image(
                image.data,
                name=layer_name,
                metadata=image.metadata.to_dict(),
                colormap=colormap,
                translate=translation,
                blending="additive",
            )
            self.viewer.reset_view()

        if isinstance(image, FibsemImage):
            self.milling_stage_editor.image_layer = self.viewer.layers[layer_name]
            self.milling_stage_editor.set_image(image)

    @pyqtSlot()
    def acquire_fib_image(self):
        """Acquire a FIB image."""
        try:
            self.acquire_fib_button.setEnabled(False)

            # Acquire FIB image
            self.fib_image = acquire.acquire_image(
                microscope=self.microscope,
                settings=self.fib_image_settings
            )
            self.fib_image_acquired_signal.emit(self.fib_image)
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
            self.fm_image = acquire_image(
                microscope=self.microscope.fm,
                channel_settings=self.channel_settings
            )
            self.image_acquired_signal.emit(self.fm_image)

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
                milling_stage.strategy.config.bbox = self.get_alignment_area()
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

    

    def set_alignment_layer(self, reduced_area: FibsemRectangle = FibsemRectangle(0.25, 0.25, 0.5, 0.5), 
                            editable: bool = True):
        """Set the alignment area layer in napari."""
        if self.fm_resolution is None:
            logging.warning("FM resolution is not available. Cannot set alignment area.")
            return

        # add alignment area to napari
        data = convert_reduced_area_to_napari_shape(reduced_area=reduced_area, 
                                                    image_shape=self.fm_resolution, 
                                                   )
        if self.alignment_layer is None or "bbox" not in self.viewer.layers:
            self.alignment_layer = self.viewer.add_shapes(data=data, 
                                              name="bbox", 
                                    shape_type="rectangle", 
                                    edge_color="white", 
                                    edge_width=5, 
                                    face_color="transparent", 
                                    opacity=0.5, 
                                    )
            self.alignment_layer.metadata = {"type": "alignment"}
        else:
            self.alignment_layer.data = data

        if editable:
            self.viewer.layers.selection.active = self.alignment_layer
            self.alignment_layer.mode = "select"
        # TODO: prevent rotation of rectangles?  
        self.alignment_layer.events.data.connect(self.update_alignment) # type: ignore
        self.update_alignment(None)  # Initial update to validate the area

    def update_alignment(self, event):
        """Validate the alignment area, and update the parent ui."""
        try:
            reduced_area = self.get_alignment_area()

            if reduced_area is None:
                return
            is_valid = reduced_area.is_valid_reduced_area
            logging.info(f"Updated alignment area: {reduced_area}, valid: {is_valid}")
            if not is_valid:
                self.label_fm_bbox.setStyleSheet("color: red;")
                self.label_fm_bbox.setText("ROI: Invalid Alignment Area. Please adjust inside FM Image.")
                return

            self.label_fm_bbox.setWordWrap(True)
            self.label_fm_bbox.setStyleSheet("color: white;")
            self.label_fm_bbox.setText(f"ROI: {reduced_area.pretty_string}")

            self.bbox_updated_signal.emit(reduced_area)

        except Exception as e:
            logging.info(f"Error updating alignment area: {e}")

    def get_alignment_area(self) -> Optional[FibsemRectangle]:
        """Get the alignment area from the alignment layer."""
        with self._lock:

            if self.alignment_layer is None or not isinstance(self.alignment_layer, NapariShapesLayer):
                logging.warning("Alignment layer is not set or not a Shapes layer.")
                return None

            data = self.alignment_layer.data
            if data is None or self.fm_resolution is None:
                return None
            data = data[0]
            return convert_shape_to_image_area(data, self.fm_resolution)

    def on_bbox_update(self, bbox: FibsemRectangle):
        """Handle updates to the bounding box."""
        logging.info(f"Bounding Box Updated: {bbox}")

    def on_intensity_drop_signal(self, ddict: dict):
        """Handle intensity drop signal."""
        # logging.info('-'*80)
        # logging.info(f"Intensity Drop Signal Received: {ddict}")
        # logging.info('-'*80)

        # self.stop_milling()  # stop milling if intensity drop is detected

        self.label_fm_intensity.setText(f"Intensity Drop Detected: Mean Intensity: {ddict['mean_intensity']:.2f}")
        self.label_fm_intensity.setStyleSheet("color: orange;")

    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info("Closing FMCoincidenceMillingWidget, stopping acquisition if running.")

        if self.microscope.fm is None:
            event.accept()
            return

        # Stop live acquisition
        self.microscope.fm.acquisition_signal.disconnect(self._on_acquisition)
        if self.microscope.fm.is_acquiring:
            try:
                self.microscope.fm.stop_acquisition()
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                logging.warning("Acquisition stopped due to widget close.")

        event.accept()

def create_widget(microscope: FibsemMicroscope,
                  viewer: napari.Viewer,
                  milling_stages: List[FibsemMillingStage],
                  parent: Optional[QWidget] = None) -> FMCoincidenceMillingWidget:
    """Create the FMCoincidenceMillingWidget with a demo microscope."""

    widget = FMCoincidenceMillingWidget(
        microscope=microscope,
        milling_stages=milling_stages,
        viewer=viewer,
        parent=parent
    )
    return widget


def main():
    """Main function to run the widget standalone."""

    microscope, settings = utils.setup_session()

    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized. Cannot create FMCoincidenceMillingWidget.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")
    
    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")
    
    # Ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True
    
    viewer = napari.Viewer()
    widget = create_widget(microscope, viewer, milling_stages=[milling_stage])
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
    return


if __name__ == "__main__":
    main()
