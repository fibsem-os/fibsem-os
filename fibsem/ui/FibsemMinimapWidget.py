import logging
import os
import sys
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Layer as NapariLayer
from napari.layers import Shapes as NapariShapesLayer
from napari.qt.threading import thread_worker
from napari.utils.events import Event as NapariEvent
from psygnal import EmissionInfo
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QDialog, QMainWindow, QWidget
from superqt import ensure_main_thread

from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
)
from fibsem.applications.autolamella.config import (
    FEATURE_DISPLAY_GRID_CENTER_MARKER,
    FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol, Lamella
from fibsem.imaging import tiled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.ui import FibsemMovementWidget, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.fm.widgets.display_options_dialog import DisplayOptionsDialog
from fibsem.ui.napari.patterns import COLOURS as MILLING_PATTERN_COLOURS
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    remove_all_napari_shapes_layers,
)
from fibsem.ui.napari.properties import (
    CORRELATION_IMAGE_LAYER_PROPERTIES,
    GRIDBAR_IMAGE_LAYER_PROPERTIES,
    OVERVIEW_IMAGE_LAYER_PROPERTIES,
)
from fibsem.ui.napari.utilities import (
    NapariShapeOverlay,
    create_crosshair_shape,
    create_rectangle_shape,
    is_inside_image_bounds,
    update_text_overlay,
)
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.qtdesigner_files import FibsemMinimapWidget as FibsemMinimapWidgetUI

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


COLOURS = CORRELATION_IMAGE_LAYER_PROPERTIES["colours"]

OVERVIEW_IMAGE_PARAMETERS = {
    "nrows": 3,
    "ncols": 3,
    "fov": 500, # um
    "dwell_time": 1.0, # us
    "autocontrast": True,
    "autogamma": False,
}

# Crosshair layer configuration constants
CROSSHAIR_CONFIG = {
    "layer_name": "stage-position",
    "crosshair_size": 50,
    "text_properties": {
        "color": "white",
        "font_size": 50,
        "anchor": "lower_left",
        "translation": (5, 55),
    },
    "line_style": {
        "edge_width": 5,
        "face_color": "transparent",
    },
    "colors": {
        "origin": "red",
        "current": "yellow",
        "saved_selected": "lime",
        "saved_unselected": "cyan",
        "grid": "red",
    },
}

OVERLAY_CONFIG = {
    "layer_name": "overlay-shapes",
    "text_properties": {
        "color": "white",
        "font_size": 50,
        "anchor": "upper_left",
        "translation": (-5, 5),
    },
    "rectangle_style": {
        "edge_width": 5,
        "face_color": "transparent",
        "opacity": 0.7,
    },
    "circle_style": {
        "edge_width": 40,
        "face_color": "transparent", 
        "opacity": 0.7,
    },
}

LABEL_INSTRUCTIONS = {
    "image-available": "Instructions: \nAlt+Click to Add a position, \nShift+Click to Update a position \nor Double Click to Move the Stage...",
    "no-image": "Please take or load an overview image..."
}

def generate_gridbar_image(shape: Tuple[int, int], pixelsize: float, spacing: float, width: float) -> FibsemImage:
    """Generate an synthetic image of cryo gridbars."""
    arr = np.zeros(shape=shape, dtype=np.uint8)

    # create grid, grid bars thickness = 10px
    thickness_px = int(width / pixelsize)
    spacing_px = int(spacing / pixelsize)
    for i in range(0, arr.shape[0], spacing_px):
        arr[i:i+thickness_px, :] = 255
        arr[:, i:i+thickness_px] = 255

    # TODO: add metadata
    return FibsemImage(data=arr)

# TODO: migrate to properly scaled infinite canvas
# TODO: allow acquiring multiple overview images
# TODO: deprecate the need for the movement_widget widgets...
# TODO: update layer name for correlation layers, set from file?
# TODO: set combobox to all images in viewer 
class FibsemMinimapWidget(FibsemMinimapWidgetUI.Ui_MainWindow, QMainWindow):
    tile_acquisition_progress_signal = pyqtSignal(dict)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: 'AutoLamellaUI',
    ):
        super().__init__(parent=parent) # type: ignore
        self.setupUi(self)

        self.parent_widget = parent

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.image: Optional[FibsemImage] = None
        self.image_layer: Optional[NapariImageLayer]  = None
        self.correlation_image_layers: List[str] = []

        self.correlation_mode_enabled: bool = False

        self._thread_stop_event = threading.Event()
        self._acquisition_worker: Optional[threading.Thread] = None

        # display options
        self.show_current_fov: bool = True
        self.show_overview_fov: bool = True
        self.show_saved_positions_fov: bool = True
        self.show_stage_limits: bool = True
        self.show_circle_overlays: bool = True
        self.show_histogram: bool = False

        if (
            self.parent_widget is None
            or self.parent_widget.microscope is None
            or self.parent_widget.experiment is None
        ):
            return

        self.setup_connections()

        self.draw_blank_image() # TMP: disable until better workflow + testing

    def draw_blank_image(self):
        image: Optional[FibsemImage] = None
        orientation = self.microscope.get_stage_orientation()
        if orientation == "SEM": 
            beam_type = BeamType.ELECTRON 
        else: 
            beam_type = BeamType.ION
        ms = self.microscope.get_microscope_state(beam_type=beam_type)
        image = FibsemImage.generate_blank_image(resolution=(4096, 4096), hfw=4000e-6)
        image.metadata.image_settings.beam_type = beam_type  # type: ignore
        image.metadata.microscope_state = ms                # type: ignore
        image.metadata.system = self.microscope.system      # type: ignore
        self.update_viewer(image=image)

    def set_experiment(self):
        if self.parent_widget is None:
            raise ValueError("Parent widget is None, cannot proceed.")
        if self.parent_widget.experiment is None:
            raise ValueError("Experiment in parent widget is None, cannot proceed.")

        self.setup_connections()
        self.draw_blank_image()

    @property
    def microscope(self) -> FibsemMicroscope:
        return self.parent_widget.microscope

    @property
    def movement_widget(self) -> FibsemMovementWidget:
        return self.parent_widget.movement_widget

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        if self.parent_widget is None:
            return None
        if self.parent_widget.experiment is None:
            return None
        return self.parent_widget.experiment.task_protocol

    def setup_connections(self):
        
        # acquisition buttons
        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)
        self.actionLoad_Image.triggered.connect(self.load_image)

        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])
        path = str(self.parent_widget.experiment.path) if self.parent_widget.experiment is not None else os.getcwd()
        self.lineEdit_tile_path.setText(path)
        self.doubleSpinBox_tile_fov.setValue(OVERVIEW_IMAGE_PARAMETERS["fov"])
        self.doubleSpinBox_tile_dwell_time.setValue(OVERVIEW_IMAGE_PARAMETERS["dwell_time"])
        self.comboBox_tile_resolution.addItems(cfg.SQUARE_RESOLUTIONS)
        self.comboBox_tile_resolution.setCurrentText(cfg.DEFAULT_SQUARE_RESOLUTION)
        self.checkBox_tile_autogamma.setChecked(OVERVIEW_IMAGE_PARAMETERS["autogamma"])
        self.checkBox_tile_autocontrast.setChecked(OVERVIEW_IMAGE_PARAMETERS["autocontrast"])
        self.spinBox_tile_nrows.setValue(OVERVIEW_IMAGE_PARAMETERS["nrows"])
        self.spinBox_tile_ncols.setValue(OVERVIEW_IMAGE_PARAMETERS["ncols"])
        self.spinBox_tile_nrows.valueChanged.connect(self.update_imaging_display)
        self.spinBox_tile_ncols.valueChanged.connect(self.update_imaging_display)
        self.spinBox_tile_nrows.setKeyboardTracking(False)
        self.spinBox_tile_ncols.setKeyboardTracking(False)
        self.spinBox_tile_nrows.setRange(1, 15)
        self.spinBox_tile_ncols.setRange(1, 15)
        self.doubleSpinBox_tile_fov.valueChanged.connect(self.update_imaging_display)
        self.update_imaging_display() # update the total fov


        # position buttons
        self.pushButton_move_to_position.clicked.connect(self.move_to_position_pressed)
        self.comboBox_tile_position.currentIndexChanged.connect(self.update_current_selected_position)
        self.pushButton_remove_position.clicked.connect(self.remove_selected_position_pressed)

        # disable updating position name:
        self.label_position_name.setVisible(False)
        self.lineEdit_tile_position_name.setVisible(False)
        self.pushButton_update_position.setVisible(False)

        # signals
        self.tile_acquisition_progress_signal.connect(self.handle_tile_acquisition_progress)
        self.parent_widget.experiment.events.connect(self._on_experiment_position_changed) # type: ignore

        # handle movement progress
        self.microscope.stage_position_changed.connect(self._on_stage_position_changed)

        # pattern overlay
        available_task_names = []
        if self.protocol is not None:
            # TODO: only support tasks that have milling
            available_task_names = [name for name in self.protocol.task_config.keys() if self.protocol.task_config[name].milling]
            self.comboBox_pattern_overlay.addItems(available_task_names)
            if "Trench Milling" in available_task_names:
                self.comboBox_pattern_overlay.setCurrentText("Trench Milling")
            elif "Rough Milling" in available_task_names:
                self.comboBox_pattern_overlay.setCurrentText("Rough Milling")
            self.comboBox_pattern_overlay.currentIndexChanged.connect(self._draw_milling_pattern_overlay)
            self.checkBox_pattern_overlay.stateChanged.connect(self._draw_milling_pattern_overlay)

        if not available_task_names:
            self.checkBox_pattern_overlay.setEnabled(False)
            self.comboBox_pattern_overlay.setToolTip("No milling patterns available.")

        # correlation
        self.actionLoad_Correlation_Image.triggered.connect(self.load_image)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        self.pushButton_enable_correlation.clicked.connect(self._toggle_correlation_mode)
        self.viewer.bind_key("C", self._toggle_correlation_mode)
        self.pushButton_enable_correlation.setEnabled(False) # disabled until correlation images added

        # gridbar controls
        self.groupBox_correlation.setEnabled(True) # only grid-bar overlay enabled
        self.checkBox_gridbar.setEnabled(True)
        self.checkBox_gridbar.stateChanged.connect(self.toggle_gridbar_display)
        self.label_gb_spacing.setVisible(False)
        self.label_gb_width.setVisible(False)
        self.doubleSpinBox_gb_spacing.setVisible(False)
        self.doubleSpinBox_gb_width.setVisible(False)
        self.doubleSpinBox_gb_spacing.setValue(GRIDBAR_IMAGE_LAYER_PROPERTIES["spacing"])
        self.doubleSpinBox_gb_width.setValue(GRIDBAR_IMAGE_LAYER_PROPERTIES["width"])
        self.doubleSpinBox_gb_spacing.setKeyboardTracking(False)
        self.doubleSpinBox_gb_width.setKeyboardTracking(False)
        self.doubleSpinBox_gb_spacing.valueChanged.connect(self.update_gridbar_layer)
        self.doubleSpinBox_gb_width.valueChanged.connect(self.update_gridbar_layer)
        self.groupBox_correlation.setToolTip("Correlation Controls are disabled until an image is acquired or loaded.")

        # set styles
        self.pushButton_update_position.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_run_tile_collection.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_cancel_acquisition.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.progressBar_acquisition.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.pushButton_remove_position.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_move_to_position.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_enable_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        # add a file menu for display options
        self.actionDisplay_Options = QAction("Display Options", self)
        self.actionDisplay_Options.triggered.connect(self.show_display_options_dialog)
        self.menuFile.addAction(self.actionDisplay_Options)

        # set italics for instructions
        self.label_instructions.setStyleSheet("font-style: italic;")
        self.scrollArea.setHorizontalScrollBarPolicy(1)  # always off

        # right-click context menu
        if FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED:
            self.viewer.mouse_drag_callbacks.append(self._on_right_click)

        self._update_position_display()
        self.toggle_interaction(enable=True)


    def show_display_options_dialog(self):
        """Show the display options dialog and apply changes."""
        dialog = DisplayOptionsDialog(self)
        dialog.checkbox_current_fov.hide()  # hide current fov option for minimap
        dialog.checkbox_histogram.hide()  # hide histogram option for minimap
        dialog.checkbox_circle_overlays.hide()  # hide circle overlays option for minimap
        if dialog.exec_() == QDialog.Accepted:
            # Get the new display options
            options = dialog.get_display_options()
    
            # Apply the options
            self.show_current_fov = options['show_current_fov']
            self.show_overview_fov = options['show_overview_fov']
            self.show_saved_positions_fov = options['show_saved_positions_fov']
            self.show_stage_limits = options['show_stage_limits']
            self.show_histogram = options['show_histogram']
            self.show_circle_overlays = options['show_circle_overlays']

            # Refresh the display
            self.draw_current_stage_position()

            logging.info("Display options updated successfully")

    @property
    def positions(self) -> List[FibsemStagePosition]:
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return []
        stage_positions = []
        for p in self.parent_widget.experiment.positions:
            stage_position = p.stage_position
            stage_position.name = p.name
            stage_positions.append(stage_position)
        return stage_positions # TODO: migrate to use Lamella directly?

    @ensure_main_thread
    def _on_experiment_position_changed(self, event: EmissionInfo):
        if self.parent_widget is None:
            return
        try:
            # logging.info(f"Experiment position emitted: {event.signal.name}, {event.path}")
            if event.signal.name not in ["inserted", "removed", "changed"]:
                return
            self._update_position_display()
        except Exception as e:
            logging.error(f"Error logging experiment position change: {e}")
            self.parent_widget.experiment.events.disconnect(self._on_experiment_position_changed) # type: ignore

    def get_imaging_parameters(self) -> Dict[str, Any]:
        """Get the imaging parameters from the UI."""
        return {
            "fov": self.doubleSpinBox_tile_fov.value() * constants.MICRO_TO_SI,
            "resolution": list(map(int, self.comboBox_tile_resolution.currentText().split("x"))),
            "tile_count": (self.spinBox_tile_nrows.value(), self.spinBox_tile_ncols.value()),
            "dwell_time": self.doubleSpinBox_tile_dwell_time.value() * constants.MICRO_TO_SI,
            "beam_type": BeamType[self.comboBox_tile_beam_type.currentText()],
            "cryo": self.checkBox_tile_autogamma.isChecked(),
            "autocontrast": self.checkBox_tile_autocontrast.isChecked(),
            "path": self.lineEdit_tile_path.text(),
            "filename": self.lineEdit_tile_filename.text(),
        }

    def update_imaging_display(self):
        """Update the imaging parameters based on the field of view and tile count."""
        # update imaging parameters based on fov and tile count
        imaging_params = self.get_imaging_parameters()
        fov = imaging_params["fov"] 
        nrows, ncols = imaging_params["tile_count"]
        total_fov = f"{nrows * fov * constants.SI_TO_MICRO:.0f} x {ncols * fov * constants.SI_TO_MICRO:.0f} um"
        self.label_tile_total_fov.setText(f"Total Field of View: {total_fov}")
        self.draw_current_stage_position()
        # TODO: calculate estimate time for acquisition

    def run_tile_collection(self):
        """Run the tiled acquisition."""
        logging.info("running tile collection")

        imaging_params = self.get_imaging_parameters()
        fov = imaging_params["fov"] 
        resolution = imaging_params["resolution"]
        dwell_time = imaging_params["dwell_time"]
        beam_type = imaging_params["beam_type"]
        cryo = imaging_params["cryo"]
        autocontrast = imaging_params["autocontrast"]
        path = imaging_params["path"]
        filename = imaging_params["filename"]
        nrows, ncols = imaging_params["tile_count"]

        image_settings = ImageSettings(
            hfw = fov,
            resolution = resolution,
            dwell_time = dwell_time,
            beam_type = beam_type,
            autocontrast = autocontrast,
            save = True,
            path = path,
            filename = filename,
        )

        # TODO: support overlap, better stitching (non-existent)
        if image_settings.filename == "":
            napari.utils.notifications.show_error("Please enter a filename for the image")
            return

        # ui feedback
        self.toggle_interaction(enable=False)
        self._hide_overlay_layers()

        # TODO: migrate to threading.Thread, rather than napari thread_worker
        self._thread_stop_event.clear()
        self._acquisition_worker = self.run_tile_collection_thread( # type: ignore
            microscope=self.microscope, image_settings=image_settings, 
            nrows=nrows,
            ncols=ncols,
            tile_size=fov, 
            overlap=0, 
            cryo=cryo)

        self._acquisition_worker.finished.connect(self.tile_collection_finished) # type: ignore
        self._acquisition_worker.errored.connect(self.tile_collection_errored) # type: ignore
        self._acquisition_worker.start()

    def tile_collection_finished(self):
        self._acquisition_worker = None
        self._thread_stop_event.clear()
        napari.utils.notifications.show_info("Tile collection finished.")
        self.update_viewer(self.image)
        self.toggle_interaction(enable=True)

    def tile_collection_errored(self):
        logging.error("Tile collection errored.")
        self._thread_stop_event.clear()
        self._acquisition_worker = None
        # TODO: handle when acquisition is cancelled halfway, clear viewer, etc

    @thread_worker
    def run_tile_collection_thread(
        self,
        microscope: FibsemMicroscope,
        image_settings: ImageSettings,
        nrows: int,
        ncols: int,
        tile_size: float,
        overlap: float = 0,
        cryo: bool = True,
    ):
        """Threaded worker for tiled acquisition and stitching."""
        try:
            self.image = tiled.tiled_image_acquisition_and_stitch(
                microscope=microscope,
                image_settings=image_settings,
                nrows=nrows,
                ncols=ncols,
                tile_size=tile_size,
                overlap=overlap,
                cryo=cryo,
                parent_ui=self,
            )
        except Exception as e:
            # TODO: specify the error, user cancelled, or error in acquisition
            logging.error(f"Error in tile collection: {e}")

    def handle_tile_acquisition_progress(self, ddict: dict) -> None:
        """Callback for handling the tile acquisition progress."""

        msg = f"{ddict['msg']} ({ddict['counter']}/{ddict['total']})"
        logging.info(msg)
        napari.utils.notifications.show_info(msg)

        # progress bar
        count, total = ddict["counter"], ddict["total"]
        self.progressBar_acquisition.setMaximum(100)
        self.progressBar_acquisition.setValue(int(count/total*100))

        image = ddict.get("image", None)
        if image is not None:
            self.update_viewer(FibsemImage(data=image), tmp=True) # TODO: this gets too slow when there are lots of tiles, update only the new tile

    def cancel_acquisition(self):
        """Cancel the tiled acquisition."""
        logging.info("Cancelling acquisition...")
        self._thread_stop_event.set()

    @property
    def is_acquiring(self) -> bool:
        """Check if the acquisition thread is running."""
        return self._acquisition_worker is not None # and self._acquisition_worker.isRunning() # type: ignore

    def toggle_gridbar_display(self):
        """Toggle the display of the synthetic grid bar overlay."""
        show_gridbar = self.checkBox_gridbar.isChecked()
        self.label_gb_spacing.setVisible(show_gridbar)
        self.label_gb_width.setVisible(show_gridbar)
        self.doubleSpinBox_gb_spacing.setVisible(show_gridbar)
        self.doubleSpinBox_gb_width.setVisible(show_gridbar)

        if show_gridbar:
            self.update_gridbar_layer()
        else:
            layer_name = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
            if layer_name in self.viewer.layers:
                self.correlation_image_layers.remove(layer_name)
                self.viewer.layers.remove(layer_name)

            self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
            self.comboBox_correlation_selected_layer.clear()
            self.comboBox_correlation_selected_layer.addItems([layer.name for layer in self.viewer.layers if "correlation-image" in layer.name ])
            self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
            # if no correlation layers left, disable enable correlation
            if len(self.comboBox_correlation_selected_layer) == 0:
                self.pushButton_enable_correlation.setEnabled(False)

    def update_gridbar_layer(self):
        """Update the synthetic grid bar overlay."""
        # update gridbars image
        spacing = self.doubleSpinBox_gb_spacing.value() * constants.MICRO_TO_SI
        width = self.doubleSpinBox_gb_width.value() * constants.MICRO_TO_SI
        gridbars_image = generate_gridbar_image(shape=self.image.data.shape,  # type: ignore
                                                pixelsize=self.image.metadata.pixel_size.x, 
                                                spacing=spacing, width=width)

        # update gridbar layer
        gridbar_layer = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
        if gridbar_layer in self.viewer.layers:
            self.viewer.layers[gridbar_layer].data = gridbars_image.data
        else:
            self.add_correlation_image(gridbars_image, is_gridbar=True)

    def toggle_interaction(self, enable: bool = True):
        """Toggle the interactivity of the UI elements."""
        self.pushButton_run_tile_collection.setEnabled(enable)
        self.pushButton_cancel_acquisition.setVisible(not enable)
        self.progressBar_acquisition.setVisible(not enable)
        # reset progress bar
        self.progressBar_acquisition.setValue(0)

        if enable:
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Run Tile Collection")
        else:
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Running Tile Collection...")

        if self.image is None:
            self.label_instructions.setText(LABEL_INSTRUCTIONS["no-image"])

    def load_image(self):
        """Ask the user to select a file to load an image as overview or correlation image."""
        is_correlation = self.sender() == self.actionLoad_Correlation_Image

        filename = ui_utils.open_existing_file_dialog(
            msg="Select image to load", 
            path=str(self.lineEdit_tile_path.text()), 
            _filter="Image Files (*.tif *.tiff)", 
            parent=self)

        if filename == "":
            napari.utils.notifications.show_error("No file selected..")
            return

        # load the image
        image = FibsemImage.load(filename)
        
        if is_correlation:
            self.add_correlation_image(image)
        else:
            self.update_viewer(image)

    def update_viewer(self, image: Optional[FibsemImage] = None, tmp: bool = False):
        """Update the viewer with the image and positions."""
        if image is not None:

            if not tmp:
                self.image = image
            arr = image.data

            try:
                self.image_layer.data = arr
            except Exception as e:
                self.image_layer = self.viewer.add_image(arr, 
                                                         name=OVERVIEW_IMAGE_LAYER_PROPERTIES["name"],
                                                         colormap=OVERVIEW_IMAGE_LAYER_PROPERTIES["colormap"],
                                                         blending=OVERVIEW_IMAGE_LAYER_PROPERTIES["blending"])  # type: ignore

            if tmp:
                return # don't update the rest of the UI, we are just updating the image
            if self.image_layer is None:
                napari.utils.notifications.show_error("Error adding image layer to viewer.")
                return

            self.image_layer.mouse_drag_callbacks.clear()
            self.image_layer.mouse_double_click_callbacks.clear()
            self.image_layer.mouse_drag_callbacks.append(self.on_single_click) # TODO: migrate to use viewer.events, rather than image layer
            self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)
            self.viewer.reset_view()

            # NOTE: how to do respace scaling, convert to infinite canvas
            # px = self.image.metadata.pixel_size.x
            # self.image_layer.scale = [px*constants.SI_TO_MICRO, px*constants.SI_TO_MICRO]
            # self.viewer.scale_bar.visible = True
            # self.viewer.scale_bar.unit = "um"

        if self.image:
            self.draw_current_stage_position()  # draw the current stage position on the image
            self._draw_milling_pattern_overlay()         # draw the reprojected positions on the image
            self.label_instructions.setText(LABEL_INSTRUCTIONS["image-available"])
        update_text_overlay(self.viewer, self.microscope)
        self.set_active_layer_for_movement()

    def get_coordinate_in_microscope_coordinates(self, layer: NapariLayer, event: NapariEvent) -> Tuple[np.ndarray, Point]:
        """Validate if event position is inside image, and convert to microscope coords
        Args:
            layer (NapariLayer): The image layer.
            event (Event): The event object.
        Returns:
            Tuple[np.ndarray, Point]: The coordinates in image and microscope image coordinates.
        """
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")

        # get coords in image coordinates (adjusts for translation, etc)
        coords = layer.world_to_data(event.position)

        # check if clicked point is inside image
        if not is_inside_image_bounds(coords=coords, shape=self.image.data.shape):
            napari.utils.notifications.show_warning(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return False, False

        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), 
            image=self.image.data, 
            pixelsize=self.image.metadata.pixel_size.x,
        )

        return coords, point

    def on_single_click(self, layer: NapariImageLayer, event: NapariEvent) -> None:
        """Callback for single click on the image layer.
        Supports adding and updating positions with Shift and Alt modifiers.
        No modifier: checks closest experiment position.
        Args:
            layer: The image layer.
            event: The event object.
        """

        update_position: bool = "Shift" in event.modifiers
        add_new_position: bool = "Alt" in event.modifiers
        no_modifier: bool = len(event.modifiers) == 0

        # left click only
        if event.button != 1:
            return

        coords, point = self.get_coordinate_in_microscope_coordinates(layer, event)

        if point is False: # clicked outside image
            return

        if self.image is None or self.image.metadata is None:
            return

        # get the stage position (xyzrt) based on the clicked point and projection
        stage_position = self.microscope.project_stable_move(
                    dx=point.x, dy=point.y,
                    beam_type=self.image.metadata.image_settings.beam_type,
                    base_position=self.image.metadata.stage_position)

        # no modifier: check closest position
        if no_modifier:
            self.check_closest_experiment_position(stage_position)
            return

        # handle case where multiple modifiers are pressed
        if update_position and add_new_position:
            napari.utils.notifications.show_warning("Please select either Shift or Alt modifier, not both.")
            return

        if self.parent_widget is None or self.parent_widget.experiment is None:
            return # prevent editing positions directly if not using autolamella

        # check if position is within stage limits
        if not stage_position.is_within_limits(self.microscope._stage.limits, axes=["x", "y"]):
            napari.utils.notifications.show_warning("Position is outside stage limits. Please select a position within the stage limits.")
            return

        if update_position:
            idx = self.comboBox_tile_position.currentIndex()
            if idx == -1:
                logging.debug("No position selected to update.")
                return
            self.parent_widget.experiment.positions[idx].stage_position = stage_position # not evented, so need to manually update
            self.parent_widget.experiment.save()
            self._update_position_display()
        elif add_new_position:
            self.parent_widget.add_new_lamella(stage_position)
            # NOTE: PY_38 doesnt support callback for experiment.events required to refresh the display, so we
            # are hacking it here, by force calling the update display
            if sys.version_info < (3, 9):
                self._update_position_display()

    def check_closest_experiment_position(self, clicked_position: FibsemStagePosition) -> None:
        """Check and print distances to all experiment positions, highlighting the closest one.

        Args:
            clicked_position: The stage position that was clicked on the minimap.
        """
        if not FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED:
            return

        if not self.positions:
            logging.info("No experiment positions to compare.")
            return

        # Calculate distances to all positions
        distances = []
        for pos in self.positions:
            distance = clicked_position.euclidean_distance(pos)
            distances.append((pos.name, distance))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Highlight the closest
        closest_name, closest_dist = distances[0]

        # If closest position is within 5um, select it
        SELECTED_POSITION_THRESHOLD_MICRONS = 5.0
        if closest_dist < SELECTED_POSITION_THRESHOLD_MICRONS * constants.MICRO_TO_SI:
            idx = self.comboBox_tile_position.findText(closest_name)
            if idx != -1:
                self.comboBox_tile_position.setCurrentIndex(idx)
                napari.utils.notifications.show_info(f"Selected: {closest_name} ({closest_dist * constants.SI_TO_MICRO:.1f} um)")
                return

        napari.utils.notifications.show_info(f"Closest: {closest_name} ({closest_dist * constants.SI_TO_MICRO:.1f} um)")

    def on_double_click(self, layer: NapariImageLayer, event: NapariEvent) -> None:
        """Callback for double click on the image layer.
        Moves the stage to the clicked position.
        Args:
            layer: The image layer.
            event: The event object.
        """

        if event.button != 1: # left click only
            return

        coords, point = self.get_coordinate_in_microscope_coordinates(layer, event)

        if point is False: # clicked outside image
            return

        if self.image is None or self.image.metadata is None:
            return

        beam_type = self.image.metadata.image_settings.beam_type
        stage_position = self.microscope.project_stable_move(
            dx=point.x, dy=point.y,
            beam_type=beam_type,
            base_position=self.image.metadata.stage_position)

        # check if position is within stage limits
        if not stage_position.is_within_limits(self.microscope._stage.limits, axes=["x", "y"]):
            napari.utils.notifications.show_warning("Position is outside stage limits. Please select a position within the stage limits.")
            return

        self.move_to_stage_position(stage_position)

    def _on_right_click(self, viewer, event: NapariEvent) -> None:
        """Callback for right-click on the viewer.
        Shows a context menu with options to add a new position or move selected position.
        Args:
            viewer: The napari viewer.
            event: The event object.
        """
        # Only handle right-click press events
        if event.button != 2 or event.type != "mouse_press":
            return

        # Check if we have an image loaded
        if self.image is None or self.image.metadata is None:
            return

        # Check if overview image layer exists
        if self.image_layer is None or self.image_layer not in self.viewer.layers:
            return

        # Get coordinates in image space
        coords = self.image_layer.world_to_data(event.position)

        # Check if clicked point is inside image
        if not is_inside_image_bounds(coords=coords, shape=self.image.data.shape):
            napari.utils.notifications.show_warning("Position is outside image bounds. Please select a position within the image.")
            return

        event.handled = True

        # Convert to microscope coordinates
        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]),
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        # Calculate the stage position for the clicked point
        stage_position = self.microscope.project_stable_move(
            dx=point.x, dy=point.y,
            beam_type=self.image.metadata.image_settings.beam_type,
            base_position=self.image.metadata.stage_position)

        # Check if position is within stage limits
        if not stage_position.is_within_limits(self.microscope._stage.limits, axes=["x", "y"]):
            napari.utils.notifications.show_warning("Position is outside stage limits. Please select a position within the stage limits.")
            return

        # Build context menu
        config = ContextMenuConfig()
        config.add_action(
            "Add New Position Here",
            callback=lambda: self._add_position_at_stage_position(stage_position),
        )

        # Only show "Move Selected Position" if there are positions to move
        if len(self.positions) > 0:
            selected_name = self.comboBox_tile_position.currentText()
            config.add_action(
                f"Move Selected Position Here ({selected_name})",
                callback=lambda: self._update_selected_position(stage_position),
            )

        menu = ContextMenu(config, parent=self)
        menu.show_at_cursor()

    def _add_position_at_stage_position(self, stage_position: FibsemStagePosition) -> None:
        """Add a new position at the given stage position."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return

        self.parent_widget.add_new_lamella(stage_position)
        # NOTE: PY_38 doesnt support callback for experiment.events required to refresh the display
        if sys.version_info < (3, 9):
            self._update_position_display()

    def _update_selected_position(self, stage_position: FibsemStagePosition) -> None:
        """Update the currently selected position to the given stage position."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return

        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            logging.debug("No position selected to update.")
            return

        self.parent_widget.experiment.positions[idx].stage_position = stage_position
        self.parent_widget.experiment.save()
        self._update_position_display()

    def update_current_selected_position(self):
        """Update the currently selected position."""
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1 or len(self.positions) == 0:
            self.lineEdit_tile_position_name.setText("")
            return

        self.lineEdit_tile_position_name.setText(self.positions[idx].name)
        self.pushButton_move_to_position.setText(f"Move to {self.positions[idx].name}")
        self.label_position_info.setText(f"{self.positions[idx].name}: {self.positions[idx].pretty_string}")

        # redraw the positions to show the selected one
        self.draw_current_stage_position()

        # QUERY: should this also update autolamella?

    def update_positions_combobox(self):
        """Update the positions combobox with the current positions."""

        has_positions = len(self.positions) > 0
        self.pushButton_move_to_position.setEnabled(has_positions)
        self.pushButton_remove_position.setEnabled(has_positions)
        self.pushButton_update_position.setEnabled(has_positions)
        self.groupBox_positions.setVisible(has_positions)

        idx = self.comboBox_tile_position.currentIndex()
        self.comboBox_tile_position.clear()
        self.comboBox_tile_position.addItems([pos.name for pos in self.positions])
        if idx == -1:
            return
        if idx < self.comboBox_tile_position.count():
            self.comboBox_tile_position.setCurrentIndex(idx)
        else:
            self.comboBox_tile_position.setCurrentIndex(self.comboBox_tile_position.count() - 1)

    def remove_selected_position_pressed(self):
        """Remove the selected position from the list."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return # prevent editing positions directly if not using autolamella

        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            return

        del self.parent_widget.experiment.positions[idx]
        self.parent_widget.experiment.save()

    def _update_position_display(self):
        """refresh the position display."""
        self.update_positions_combobox()
        self.update_viewer()

    def move_to_position_pressed(self) -> None:
        """Move the stage to the selected position."""
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            return
        stage_position = self.positions[idx] # TODO: migrate to self.selected_stage_position
        self.move_to_stage_position(stage_position)

    def move_to_stage_position(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the selected position via movement widget."""
        self.movement_widget.move_to_position(stage_position)

    @ensure_main_thread
    def _on_stage_position_changed(self, stage_position: FibsemStagePosition):
        """Callback for when the stage position is changed."""
        if self.is_acquiring:
            return # do not update while acquiring
        try:
            self.update_viewer()
        except Exception as e:
            self.microscope.stage_position_changed.disconnect(self._on_stage_position_changed)
            logging.error(f"Error updating viewer on stage position change, signal disconnected: {e}")

    def _hide_overlay_layers(self):
        """Hide all overlay layers."""
        if OVERLAY_CONFIG["layer_name"] in self.viewer.layers:
            self.viewer.layers[OVERLAY_CONFIG["layer_name"]].visible = False
        if CROSSHAIR_CONFIG["layer_name"] in self.viewer.layers:
            self.viewer.layers[CROSSHAIR_CONFIG["layer_name"]].visible = False
        if "Milling Patterns" in self.viewer.layers:
            self.viewer.layers["Milling Patterns"].visible = False
        return

    def draw_current_stage_position(self):
        """Draws the current stage position on the image."""
        if self.image is None or self.image.metadata is None:
            return

        if self.is_acquiring:
            self._hide_overlay_layers()
            return


        self._draw_overlay_shapes()
        self._draw_position_crosshairs()
        self._draw_milling_pattern_overlay()

    def _collect_all_overlays(self) -> List[NapariShapeOverlay]:
        """Collect all overlay shapes to be drawn on the image."""

        if self.image is None or self.image.metadata is None:
            return []

        # If no overlays are to be shown, return empty list
        if not (self.show_current_fov or 
                self.show_overview_fov or 
                self.show_saved_positions_fov or 
                self.show_stage_limits):
            return []

        current_stage_position = deepcopy(self.microscope.get_stage_position())
        current_stage_position.name = "Current Position"
        points = tiled.reproject_stage_positions_onto_image2(self.image, [current_stage_position])

        pixelsize = self.image.metadata.pixel_size.x
        current_position = points[0]

        overlays = []

        # current overview fov
        if self.show_overview_fov:
            imaging_params = self.get_imaging_parameters()
            fov = imaging_params["fov"]
            nrows, ncols = imaging_params["tile_count"]
            width = (ncols * fov) / pixelsize
            height = (nrows * fov) / pixelsize
            rect = create_rectangle_shape(current_position, width, height)
            overlays.append(NapariShapeOverlay(
                shape=rect,
                color="magenta",
                label="Overview FoV",
                shape_type='rectangle')
            )

        # stage limits
        if self.show_stage_limits and self.microscope.stage_is_compustage:
            stage_limits = self.microscope._stage.limits
            xmin, xmax = stage_limits["x"].min, stage_limits["x"].max
            ymin, ymax = stage_limits["y"].min, stage_limits["y"].max
            centre_grid = FibsemStagePosition(name="Grid Centre", x=0, y=0, z=0, r=0, t=0)
            top_limit = FibsemStagePosition(name="Top Limit", x=0, y=ymin, z=0, r=0, t=0)
            bottom_limit = FibsemStagePosition(name="Bottom Limit", x=0, y=ymax, z=0, r=0, t=0)
            points = tiled.reproject_stage_positions_onto_image2(self.image, [centre_grid, top_limit, bottom_limit])
            width = (xmax-xmin) / pixelsize
            height = points[1].y - points[2].y
            grid_centre = points[0]
            rect = create_rectangle_shape(grid_centre, width, height)
            overlays.append(NapariShapeOverlay(
                shape=rect,
                color="yellow",
                label="Stage Limits",
                shape_type='rectangle')
            )

        if self.show_saved_positions_fov:
            points = tiled.reproject_stage_positions_onto_image2(self.image, self.positions)
            width = 80e-6 / pixelsize # TODO: make this match the milling fov
            height = 1024/1536 * width 
            selected_position = self.comboBox_tile_position.currentText()
            for point in points:
                overlays.append(NapariShapeOverlay(
                    shape=create_rectangle_shape(point, width=width, height=height),
                    color="lime" if point.name == selected_position else "cyan",
                    label=point.name,
                    shape_type='rectangle')
                )

        return overlays

    def _draw_overlay_shapes(self, layer_scale: Optional[Tuple[float, float]] = None):
        """Draw all overlay shapes (FOV boxes and circles) on a single layer.

        Creates overlays for:
        - Current position single image FOV (magenta rectangles)
        - Overview acquisition area (orange rectangles, only if not acquiring)
        - Saved positions FOV (cyan rectangles)
        - Circle overlays (various colors based on configuration)

        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
        """
        layer_name = OVERLAY_CONFIG["layer_name"]

        try:
            # Collect all overlay shapes
            overlays = self._collect_all_overlays()

            # Update or create the layer
            if not overlays:
                # Hide layer if no shapes to display
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = False
                return

            # Extract data for napari
            all_shapes = [overlay.shape for overlay in overlays]
            all_colors = [overlay.color for overlay in overlays]
            all_labels = [overlay.label for overlay in overlays]
            all_shape_types = [overlay.shape_type for overlay in overlays]

            # Prepare text properties for labels
            text_properties = {
                "string": all_labels,
                **OVERLAY_CONFIG["text_properties"]
            }

            if layer_name in self.viewer.layers:
                # Update existing layer
                layer: NapariShapesLayer = self.viewer.layers[layer_name]
                layer.data = [] # clear to reset shape type
                layer.data = all_shapes
                layer.shape_type = all_shape_types
                layer.edge_color = all_colors
                layer.edge_width = OVERLAY_CONFIG["rectangle_style"]["edge_width"]
                layer.face_color = OVERLAY_CONFIG["rectangle_style"]["face_color"]
                layer.opacity = OVERLAY_CONFIG["rectangle_style"]["opacity"]
                layer.visible = True
                # Try to update text properties
                try:
                    layer.text = text_properties
                except AttributeError:
                    logging.debug("Could not update text properties for overlay layer")
            else:
                # Create new layer with mixed shape types
                self.viewer.add_shapes(
                    data=all_shapes,
                    name=layer_name,
                    shape_type=all_shape_types,
                    edge_color=all_colors,
                    edge_width=OVERLAY_CONFIG["rectangle_style"]["edge_width"],
                    face_color=OVERLAY_CONFIG["rectangle_style"]["face_color"],
                    scale=layer_scale,
                    opacity=OVERLAY_CONFIG["rectangle_style"]["opacity"],
                    text=text_properties,
                )

        except Exception as e:
            logging.warning(f"Error drawing overlay shapes: {e}")
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

    def _collect_crosshair_overlays(self, layer_scale: Optional[Tuple[float, float]] = None):
        """Collect crosshair overlays for current and saved positions."""
        if self.image is None or self.image.metadata is None:
            return

        current_stage_position = deepcopy(self.microscope.get_stage_position())
        stage_origin = FibsemStagePosition(name="Origin", x=0, y=0, z=0, r=0, t=0)
        points = tiled.reproject_stage_positions_onto_image2(self.image, [current_stage_position, stage_origin])

        origin_point = points[1]
        current_point = points[0]
        crosshair_size = CROSSHAIR_CONFIG["crosshair_size"]
        layer_scale = None
        overlays: List[NapariShapeOverlay] = []

        # grid centre
        origin_lines = create_crosshair_shape(origin_point, crosshair_size, layer_scale)
        for line, txt in zip(origin_lines, ["Origin (0, 0)", ""]):
            overlays.append(NapariShapeOverlay(
                shape=line,
                color=CROSSHAIR_CONFIG["colors"]["origin"],
                label=txt,
                shape_type="line"
            ))

        # grid positions
        if FEATURE_DISPLAY_GRID_CENTER_MARKER:
            grid_positions = [g.position for g in self.microscope._stage.holder.grids.values()]
            grid_points = tiled.reproject_stage_positions_onto_image2(self.image, grid_positions)
            for i, grid_point in enumerate(grid_points):
                grid_lines = create_crosshair_shape(grid_point, crosshair_size, layer_scale)
                for line, txt in zip(grid_lines, [grid_positions[i].name, ""]):
                    overlays.append(NapariShapeOverlay(
                        shape=line,
                        color=CROSSHAIR_CONFIG["colors"]["grid"],
                        label=txt,
                        shape_type="line"
                    ))

        # current stage position
        current_lines = create_crosshair_shape(current_point, crosshair_size, layer_scale)
        for line, txt in zip(current_lines, ["Stage Position", ""]):
            overlays.append(NapariShapeOverlay(
                shape=line,
                color=CROSSHAIR_CONFIG["colors"]["current"],
                label=txt,
                shape_type="line"
            ))

        # saved positions
        selected_index = self.comboBox_tile_position.currentIndex()
        pts = tiled.reproject_stage_positions_onto_image2(self.image, self.positions)
        for i, (saved_pos, saved_point) in enumerate(zip(self.positions, pts)):
            saved_lines = create_crosshair_shape(saved_point, crosshair_size, layer_scale)

            # Use lime for selected position, cyan for others
            color = CROSSHAIR_CONFIG["colors"]["saved_unselected"]
            if i == selected_index:
                color = CROSSHAIR_CONFIG["colors"]["saved_selected"]

            # Show position name on crosshair if saved position FOV is disabled
            label = saved_pos.name if not self.show_saved_positions_fov else ""

            for line, txt in zip(saved_lines, [label, ""]):
                overlays.append(NapariShapeOverlay(
                    shape=line,
                    color=color,
                    label=txt,
                    shape_type="line"
                ))

        return overlays

    def _draw_position_crosshairs(self, layer_scale: Optional[Tuple[float, float]] = None):
        """Draw crosshair overlays for current and saved positions on a single layer. """

        # Collect all crosshair overlays
        layer_name = CROSSHAIR_CONFIG["layer_name"]
        crosshair_overlays = self._collect_crosshair_overlays(layer_scale)

        # Extract data for napari
        crosshair_lines = [overlay.shape for overlay in crosshair_overlays]
        colors = [overlay.color for overlay in crosshair_overlays]
        labels = [overlay.label for overlay in crosshair_overlays]

        # Prepare text properties for labels
        text_properties = {
            "string": labels,
            **CROSSHAIR_CONFIG["text_properties"]
        }

        # Update or create the napari layer
        if layer_name in self.viewer.layers:
            # Update existing layer
            layer: NapariShapesLayer = self.viewer.layers[layer_name]
            layer.data = crosshair_lines
            # Note: edge_color and text updates may not work with all napari versions
            try:
                layer.edge_color = colors
                layer.edge_width = CROSSHAIR_CONFIG["line_style"]["edge_width"]
                layer.text = text_properties
                layer.visible = True
            except AttributeError:
                logging.debug("Could not update layer properties directly")
        else:
            # Create new layer
            self.viewer.add_shapes(
                data=crosshair_lines,
                name=layer_name,
                shape_type="line",
                edge_color=colors,
                edge_width=CROSSHAIR_CONFIG["line_style"]["edge_width"],
                face_color=CROSSHAIR_CONFIG["line_style"]["face_color"],
                scale=layer_scale,
                text=text_properties,
            )

    def _draw_milling_pattern_overlay(self):
        """Draws the milling patterns for all saved positions on the image."""

        # Performance Note: the reason this is slow is because every time a new position is added, we re-draw every position
        # this is not necessary, we can just add the new position to the existing layer
        # almost all the slow down comes from the linked callbacks from fibsem.applications.autolamella. probably saving and re-drawing milling patterns
        # we should delay that until the user requests it

        if not self.checkBox_pattern_overlay.isChecked():
            if "Milling Patterns" in self.viewer.layers:
                self.viewer.layers["Milling Patterns"].visible = False
            return

        if not (self.image
                and self.image.metadata 
                and self.image_layer 
                and self.positions):
            return

        if self.protocol is None:
            napari.utils.notifications.show_warning("No milling patterns found in protocol...")
            return

        selected_pattern = self.comboBox_pattern_overlay.currentText()
        selected_milling_stages: List[FibsemMillingStage] = []
        if selected_pattern == "":
            napari.utils.notifications.show_warning("Please select a milling pattern to overlay...")
            return

        task_config = self.protocol.task_config
        milling_config = task_config[selected_pattern].milling
        if MILL_ROUGH_KEY in milling_config:
            selected_milling_stages = deepcopy(milling_config[MILL_ROUGH_KEY].stages)
        else:
            key = list(milling_config.keys())[0]
            selected_milling_stages = deepcopy(milling_config[key].stages)
        points = tiled.reproject_stage_positions_onto_image2(self.image, self.positions)

        # TODO: this should show the real milling patterns for each lamella, rather than the same one for all.
        # then, changes in the protocol would be reflected in the minimap
        milling_stages: List[FibsemMillingStage] = []
        for point in points:
            pt = conversions.image_to_microscope_image_coordinates(coord=point,
                                                                image=self.image.data,
                                                                pixelsize=self.image.metadata.pixel_size.x)
            stages = deepcopy(selected_milling_stages)
            for i, stage in enumerate(stages):
                stage.name = f"{point.name}-{stage.name}"
                stage.pattern.point += pt
            milling_stages.extend(deepcopy(stages))

        draw_milling_patterns_in_napari(
            viewer=self.viewer,
            image_layer=self.image_layer,
            pixelsize=self.image.metadata.pixel_size.x,
            milling_stages=milling_stages,
            draw_crosshair=False,
            colors=MILLING_PATTERN_COLOURS[:len(selected_milling_stages)],
            )
        if "Milling Patterns" in self.viewer.layers:
            self.viewer.layers["Milling Patterns"].visible = True

        # set the image layer as the active layer for movement
        self.set_active_layer_for_movement()

    def add_correlation_image(self, image: FibsemImage, is_gridbar: bool = False):
        """Add a correlation image to the viewer."""

        basename = CORRELATION_IMAGE_LAYER_PROPERTIES["name"]
        if is_gridbar:
            basename = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
        idx = 1
        layer_name = f"{basename}-{idx:02d}"
        while layer_name in self.viewer.layers:
            idx+=1
            layer_name = f"{basename}-{idx:02d}"

        # if grid bar in _name, idx = 3
        if is_gridbar:
            layer_name = basename
            idx = 3

        # add the image layer
        self.viewer.add_image(image.data, 
                        name=layer_name, 
                        colormap=COLOURS[idx%len(COLOURS)], 
                        blending=CORRELATION_IMAGE_LAYER_PROPERTIES["blending"], 
                        opacity=CORRELATION_IMAGE_LAYER_PROPERTIES["opacity"])
        self.correlation_image_layers.append(layer_name)

        # update the combobox
        self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
        idx = self.comboBox_correlation_selected_layer.currentIndex()
        self.comboBox_correlation_selected_layer.clear()
        self.comboBox_correlation_selected_layer.addItems(self.correlation_image_layers)
        if idx != -1:
            self.comboBox_correlation_selected_layer.setCurrentIndex(idx)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        
        # set the image layer as the active layer
        self.set_active_layer_for_movement()
        self.groupBox_correlation.setEnabled(True) # TODO: allow enabling grid-bar overlay separately
        self.checkBox_gridbar.setEnabled(True)
        self.pushButton_enable_correlation.setEnabled(True)

    # do this when image selected is changed
    def update_correlation_ui(self):

        # set ui
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        self.pushButton_enable_correlation.setEnabled(layer_name != "")
        if layer_name == "":
            napari.utils.notifications.show_info("Please select a layer to correlate with update data...")
            return

    def _toggle_correlation_mode(self, event: NapariEvent = None):
        """Toggle correlation mode on or off."""
        if self.image is None:
            napari.utils.notifications.show_warning("Please acquire an image first...")
            return
        
        if not self.correlation_image_layers:
            napari.utils.notifications.show_warning("Please load a correlation image first...")
            return

        # toggle correlation mode
        self.correlation_mode_enabled = not self.correlation_mode_enabled

        if self.correlation_mode_enabled:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Disable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(False)
        else:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Enable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(True)

        # if no correlation layer selected, disable the button
        if self.comboBox_correlation_selected_layer.currentIndex() == -1:
            self.pushButton_enable_correlation.setEnabled(False)
            return

        # get current correlation layer
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        correlation_layer = self.viewer.layers[layer_name]
        
        # set transformation mode on
        if self.correlation_mode_enabled:
            correlation_layer.mode = 'transform'
            self.viewer.layers.selection.active = correlation_layer
        else:
            correlation_layer.mode = 'pan_zoom'
            self.set_active_layer_for_movement()

    def set_active_layer_for_movement(self) -> None:
        """Set the active layer to the image layer for movement."""
        if self.image_layer is not None and self.image_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.image_layer
