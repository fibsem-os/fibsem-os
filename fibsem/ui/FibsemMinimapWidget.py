import logging
import os
import sys
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Layer as NapariLayer
from napari.layers import Shapes as NapariShapesLayer
from napari.qt.threading import thread_worker
from napari.utils.events import Event as NapariEvent
from psygnal import EmissionInfo
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem import constants, conversions
from fibsem.applications.autolamella.config import (
    FEATURE_DISPLAY_GRID_CENTER_MARKER,
)
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    DefectType,
    Lamella,
)
from fibsem.imaging import tiled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
    Point,
)
from fibsem.ui import FibsemMovementWidget, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.napari.patterns import COLOURS as MILLING_PATTERN_COLOURS
from fibsem.ui.napari.patterns import (
    MILLING_PATTERN_LAYER_NAME,
    draw_milling_patterns_in_napari,
)
from fibsem.ui.napari.properties import (
    CORRELATION_IMAGE_LAYER_PROPERTIES,
    GRIDBAR_IMAGE_LAYER_PROPERTIES,
    OVERVIEW_IMAGE_LAYER_PROPERTIES,
)
from fibsem.ui.napari.utilities import (
    NapariShapeOverlay,
    create_circle_shape,
    create_crosshair_shape,
    create_rectangle_shape,
    is_inside_image_bounds,
    update_text_overlay,
)
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig, LamellaNameListWidget, TitledPanel
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    OverviewAcquisitionSettingsWidget,
)

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
    "image-available": "Instructions: \nRight Click to Add/Move a Lamella Position or Double Click to Move the Stage...",
    "no-image": "Please take or load an overview image..."
}
DEFAULT_OVERVIEW_ACQUISITION_SETTINGS = OverviewAcquisitionSettings(
    image_settings=ImageSettings(
        hfw=OVERVIEW_IMAGE_PARAMETERS["fov"] * constants.MICRO_TO_SI,
        dwell_time=OVERVIEW_IMAGE_PARAMETERS["dwell_time"] * constants.MICRO_TO_SI,
        autocontrast=OVERVIEW_IMAGE_PARAMETERS["autocontrast"],
        autogamma=OVERVIEW_IMAGE_PARAMETERS["autogamma"],
        beam_type=BeamType.ELECTRON,
        save=True,
        path=None,  # will be set to experiment path when overview acquisition widget is initialized
        filename="overview-image",
    ),
    nrows=OVERVIEW_IMAGE_PARAMETERS["nrows"],
    ncols=OVERVIEW_IMAGE_PARAMETERS["ncols"],
)


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
class FibsemMinimapWidget(QWidget):
    tile_acquisition_progress_signal = pyqtSignal(dict)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: 'AutoLamellaUI',
    ):
        super().__init__(parent=parent) # type: ignore
        self._setup_ui()

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

        if (
            self.parent_widget is None
            or self.parent_widget.microscope is None
            or self.parent_widget.experiment is None
        ):
            return

        self.setup_connections()
        self.draw_blank_image()

    def _setup_ui(self):
        # Main layout directly on self
        self.gridLayout = QGridLayout(self)

        # Scroll area — row 0
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.gridLayout_5 = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0)

        # Bottom elements (outside scroll area)
        self.label_instructions = QLabel(LABEL_INSTRUCTIONS["no-image"])
        self.gridLayout.addWidget(self.label_instructions, 1, 0)

        self.pushButton_run_tile_collection = QPushButton("Run Tiled Acquisition")
        self.gridLayout.addWidget(self.pushButton_run_tile_collection, 2, 0)

        self.pushButton_cancel_acquisition = QPushButton("Cancel Acquisition")
        self.gridLayout.addWidget(self.pushButton_cancel_acquisition, 3, 0)

        self.progressBar_acquisition = QProgressBar()
        self.progressBar_acquisition.setValue(24)
        self.progressBar_acquisition.setAlignment(Qt.AlignCenter)
        self.progressBar_acquisition.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
        self.gridLayout.addWidget(self.progressBar_acquisition, 4, 0)

        # --- Acquisition settings widget ---
        self.overview_acquisition_widget = OverviewAcquisitionSettingsWidget(self)
        self.gridLayout_5.addWidget(self.overview_acquisition_widget, 0, 0)

        # ── Positions panel ────────────────────────────────────────
        positions_content = QWidget()
        self.gridLayout_2 = QGridLayout(positions_content)
        self.gridLayout_2.setContentsMargins(4, 4, 4, 4)

        bold_font = QFont()
        bold_font.setBold(True)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_actions_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_remove_button(True)
        self.gridLayout_2.addWidget(self.lamella_list, 0, 0, 1, 2)

        self.label_position_info = QLabel("No Positions saved.")
        self.gridLayout_2.addWidget(self.label_position_info, 2, 0, 1, 2)

        # row 3 skipped (matches original .ui)
        self.label_pattern_overlay = QLabel("Pattern Overlay")
        self.label_pattern_overlay.setFont(bold_font)
        self.gridLayout_2.addWidget(self.label_pattern_overlay, 4, 0)

        self.checkBox_pattern_overlay = QCheckBox("Display Pattern")
        self.comboBox_pattern_overlay = QComboBox()
        self.gridLayout_2.addWidget(self.checkBox_pattern_overlay, 5, 0)
        self.gridLayout_2.addWidget(self.comboBox_pattern_overlay, 5, 1)

        self.positions_panel = TitledPanel("Positions", content=positions_content)
        self.positions_panel._btn_collapse.setChecked(True)
        self.gridLayout_5.addWidget(self.positions_panel, 1, 0)

        # ── Correlation panel ─────────────────────────────────────
        correlation_content = QWidget()
        self.gridLayout_4 = QGridLayout(correlation_content)
        self.gridLayout_4.setContentsMargins(4, 4, 4, 4)

        self.label_correlation_selected_layer = QLabel("Selected Layer")
        self.comboBox_correlation_selected_layer = QComboBox()
        self.gridLayout_4.addWidget(self.label_correlation_selected_layer, 0, 0)
        self.gridLayout_4.addWidget(self.comboBox_correlation_selected_layer, 0, 1, 1, 2)

        # row 1 skipped (matches original .ui)
        self.checkBox_gridbar = QCheckBox("Show Grid Overlay")
        self.label_gb_width = QLabel("Gridbar Width (um)")
        self.label_gb_spacing = QLabel("Gridbar Spacing (um)")
        self.gridLayout_4.addWidget(self.checkBox_gridbar, 2, 0)
        self.gridLayout_4.addWidget(self.label_gb_width, 2, 1)
        self.gridLayout_4.addWidget(self.label_gb_spacing, 2, 2)

        self.doubleSpinBox_gb_width = QDoubleSpinBox()
        self.doubleSpinBox_gb_width.setMaximum(10000.0)
        self.doubleSpinBox_gb_spacing = QDoubleSpinBox()
        self.doubleSpinBox_gb_spacing.setMaximum(10000.0)
        self.gridLayout_4.addWidget(self.doubleSpinBox_gb_width, 3, 1)
        self.gridLayout_4.addWidget(self.doubleSpinBox_gb_spacing, 3, 2)

        self.pushButton_enable_correlation = QPushButton("Enable Correlation Mode")
        self.gridLayout_4.addWidget(self.pushButton_enable_correlation, 4, 0, 1, 3)

        self.correlation_panel = TitledPanel("Correlation", content=correlation_content)
        self.correlation_panel._btn_collapse.setChecked(False)
        self.gridLayout_5.addWidget(self.correlation_panel, 2, 0)

        # ── Display Options panel ─────────────────────────────────
        display_content = QWidget()
        _dlo = QVBoxLayout(display_content)
        _dlo.setContentsMargins(4, 4, 4, 4)
        self.checkBox_show_overview_fov = QCheckBox("Show Overview FOV")
        self.checkBox_show_overview_fov.setChecked(True)
        self.checkBox_show_saved_positions_fov = QCheckBox("Show Saved Positions FOV")
        self.checkBox_show_saved_positions_fov.setChecked(True)
        self.checkBox_show_stage_limits = QCheckBox("Show Stage Limits")
        self.checkBox_show_stage_limits.setChecked(True)
        self.checkBox_show_circle_overlays = QCheckBox("Show Circle Overlays")
        self.checkBox_show_circle_overlays.setChecked(True)
        _dlo.addWidget(self.checkBox_show_overview_fov)
        _dlo.addWidget(self.checkBox_show_saved_positions_fov)
        _dlo.addWidget(self.checkBox_show_stage_limits)
        _dlo.addWidget(self.checkBox_show_circle_overlays)

        display_panel = TitledPanel("Display Options", content=display_content)
        display_panel._btn_collapse.setChecked(False)
        self.gridLayout_5.addWidget(display_panel, 3, 0)

        self.pushButton_load_image = QPushButton("Load Image")
        self.gridLayout_5.addWidget(self.pushButton_load_image, 4, 0)

        self.pushButton_load_correlation_image = QPushButton("Load Correlation Image")
        self.gridLayout_5.addWidget(self.pushButton_load_correlation_image, 5, 0)

        # add strech to end of scroll content
        self.gridLayout_5.setRowStretch(6, 1)

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
        self.pushButton_load_image.clicked.connect(self.load_image)

        # initialise overview acquisition widget with defaults
        path = str(self.parent_widget.experiment.path) if self.parent_widget.experiment is not None else os.getcwd()
        DEFAULT_OVERVIEW_ACQUISITION_SETTINGS.image_settings.path = path
        self.overview_acquisition_widget.update_from_settings(DEFAULT_OVERVIEW_ACQUISITION_SETTINGS)
        self.overview_acquisition_widget.settings_changed.connect(self.update_imaging_display)

        # position list signals
        self.lamella_list.lamella_selected.connect(self.update_current_selected_position)
        self.lamella_list.move_to_requested.connect(self._on_move_to_requested)
        self.lamella_list.remove_requested.connect(self._on_remove_requested)

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
        self.pushButton_load_correlation_image.clicked.connect(self.load_image)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        self.pushButton_enable_correlation.clicked.connect(self._toggle_correlation_mode)
        # self.viewer.bind_key("C", self._toggle_correlation_mode, overwrite=True)
        self.pushButton_enable_correlation.setEnabled(False) # disabled until correlation images added

        # gridbar controls
        self.correlation_panel.setEnabled(True) # only grid-bar overlay enabled
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
        self.correlation_panel.setToolTip("Correlation Controls are disabled until an image is acquired or loaded.")

        # set styles
        self.pushButton_run_tile_collection.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_cancel_acquisition.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.progressBar_acquisition.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.pushButton_enable_correlation.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_load_image.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_load_correlation_image.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        # display option checkboxes
        self.checkBox_show_overview_fov.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_saved_positions_fov.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_stage_limits.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_circle_overlays.toggled.connect(self._on_display_option_toggled)

        # set italics for instructions
        self.label_instructions.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        # right-click context menu
        self.viewer.mouse_drag_callbacks.append(self._on_right_click)

        self._update_position_display()
        self.toggle_interaction(enable=True)

    def _on_display_option_toggled(self):
        self.show_overview_fov = self.checkBox_show_overview_fov.isChecked()
        self.show_saved_positions_fov = self.checkBox_show_saved_positions_fov.isChecked()
        self.show_stage_limits = self.checkBox_show_stage_limits.isChecked()
        self.show_circle_overlays = self.checkBox_show_circle_overlays.isChecked()
        self.draw_current_stage_position()

    @property
    def lamellas(self) -> List[Lamella]:
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return []
        return list(self.parent_widget.experiment.positions)

    @property
    def positions(self) -> List[FibsemStagePosition]:
        stage_positions = []
        for lam in self.lamellas:
            sp = deepcopy(lam.stage_position)
            sp.name = lam.name
            stage_positions.append(sp)
        return stage_positions

    def _lamella_color(self, lamella: Lamella, selected: bool) -> str:
        """Return a display color for a lamella based on its defect state and selection."""
        if selected:
            return CROSSHAIR_CONFIG["colors"]["saved_selected"]  # lime
        if lamella.defect.state == DefectType.FAILURE:
            return "red"
        if lamella.defect.state == DefectType.REWORK:
            return "orange"
        return CROSSHAIR_CONFIG["colors"]["saved_unselected"]  # cyan

    @property
    def selected_lamella(self) -> Optional[Lamella]:
        """Return the currently selected lamella, or None if nothing is selected."""
        return self.lamella_list.selected_lamella

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

    def get_overview_settings(self) -> OverviewAcquisitionSettings:
        """Get the current overview acquisition settings from the UI."""
        return self.overview_acquisition_widget.get_settings()

    def update_imaging_display(self):
        """Refresh overlays whenever acquisition settings change."""
        self.draw_current_stage_position()

    def run_tile_collection(self):
        """Run the tiled acquisition."""
        logging.info("running tile collection")

        overview_settings = self.get_overview_settings()
        image_settings = overview_settings.image_settings
        image_settings.save = True

        if not image_settings.filename:
            napari.utils.notifications.show_error("Please enter a filename for the image")
            return

        # ui feedback
        self.toggle_interaction(enable=False)
        self._hide_overlay_layers()

        # TODO: migrate to threading.Thread, rather than napari thread_worker
        self._thread_stop_event.clear()
        self._acquisition_worker = self.run_tile_collection_thread(  # type: ignore
            microscope=self.microscope,
            overview_settings=overview_settings,
        )

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
        overview_settings: OverviewAcquisitionSettings,
    ):
        """Threaded worker for tiled acquisition and stitching."""
        try:
            image_settings = overview_settings.image_settings
            self.image = tiled.tiled_image_acquisition_and_stitch(
                microscope=microscope,
                image_settings=image_settings,
                nrows=overview_settings.nrows,
                ncols=overview_settings.ncols,
                tile_size=image_settings.hfw,
                overlap=overview_settings.overlap,
                cryo=image_settings.autogamma,
                parent_ui=self,
            )
        except Exception as e:
            # TODO: specify the error, user cancelled, or error in acquisition
            logging.error(f"Error in tile collection: {e}")

    def handle_tile_acquisition_progress(self, ddict: dict) -> None:
        """Callback for handling the tile acquisition progress."""

        # progress bar
        count, total = ddict["counter"], ddict["total"]
        self.progressBar_acquisition.setMaximum(100)
        self.progressBar_acquisition.setValue(int(count/total*100))
        self.progressBar_acquisition.setFormat(f"{ddict['msg']} — {count}/{total} tiles (%p%)")

        image = ddict.get("image", None)
        if image is not None:
            self.update_viewer(image, tmp=True)

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
            if self.comboBox_correlation_selected_layer.count() == 0:
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
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            self.pushButton_run_tile_collection.setText("Run Tile Collection")
        else:
            # self.pushButton_run_tile_collection.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Running Tile Collection...")

        if self.image is None:
            self.label_instructions.setText(LABEL_INSTRUCTIONS["no-image"])

    def load_image(self):
        """Ask the user to select a file to load an image as overview or correlation image."""
        is_correlation = self.sender() == self.pushButton_load_correlation_image

        filename = ui_utils.open_existing_file_dialog(
            msg="Select image to load",
            path=str(self.overview_acquisition_widget.get_settings().image_settings.path or os.getcwd()),
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
                arr = image.filtered_data
            else:
                arr = image # np.array(image)

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
            idx = self.lamella_list.selected_index
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

        # If closest position is within 50um, select it
        SELECTED_POSITION_THRESHOLD_MICRONS = 50.0
        if closest_dist < SELECTED_POSITION_THRESHOLD_MICRONS * constants.MICRO_TO_SI:
            self.lamella_list.select(closest_name)
            return

    def on_double_click(self, layer: NapariImageLayer, event: NapariEvent) -> None:
        """Callback for double click on the image layer.
        Moves the stage to the clicked position.
        Args:
            layer: The image layer.
            event: The event object.
        """

        if self.parent_widget.is_workflow_running:
            napari.utils.notifications.show_warning("Cannot move stage while workflow is running.")
            return

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
        if len(self.lamellas) > 0:
            selected_name = self.lamella_list.selected_name
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

        idx = self.lamella_list.selected_index
        if idx == -1:
            logging.debug("No position selected to update.")
            return

        self.parent_widget.experiment.positions[idx].stage_position = stage_position
        self.parent_widget.experiment.save()
        self._update_position_display()

    def update_current_selected_position(self, _lamella=None):
        """Update the currently selected position."""
        lam = self.selected_lamella
        if lam is None:
            return

        self.label_position_info.setText(f"{lam.name}: {lam.stage_position.pretty_string}")

        # redraw the positions to show the selected one
        self.draw_current_stage_position()

    def update_positions_combobox(self):
        """Update the positions combobox with the current positions."""

        lamellas = self.lamellas
        has_positions = len(lamellas) > 0
        self.positions_panel.setEnabled(has_positions)
        if not has_positions:
            self.positions_panel.setToolTip("No positions available. Please add a position via Right Click on the image.")
        else:
            self.positions_panel.setToolTip("")

        self.lamella_list.set_lamella(lamellas)

    def _on_move_to_requested(self, lamella):
        """Handle move-to request from the list row's actions menu."""
        if lamella is None:
            return
        self.move_to_stage_position(lamella.stage_position)

    def _on_remove_requested(self, lamella):
        """Handle removal from the list row's remove button (confirmation already handled)."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return
        try:
            self.parent_widget.experiment.positions.remove(lamella)
        except ValueError:
            return
        self.parent_widget.experiment.save()
        self._update_position_display()

    def _update_position_display(self):
        """refresh the position display."""
        self.update_positions_combobox()
        self.update_viewer()

    def move_to_stage_position(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the selected position via movement widget."""
        self.movement_widget.move_to_position(stage_position)

    @ensure_main_thread
    def _on_stage_position_changed(self, stage_position: FibsemStagePosition):
        """Callback for when the stage position is changed."""
        if self.is_acquiring:
            return # do not update while acquiring
        try:
            self.draw_current_stage_position(stage_position=stage_position)
            update_text_overlay(self.viewer, self.microscope, stage_position=stage_position)
            self.set_active_layer_for_movement()
        except Exception as e:
            self.microscope.stage_position_changed.disconnect(self._on_stage_position_changed)
            logging.error(f"Error updating viewer on stage position change, signal disconnected: {e}")

    def _hide_overlay_layers(self):
        """Hide all overlay layers."""
        if OVERLAY_CONFIG["layer_name"] in self.viewer.layers:
            self.viewer.layers[OVERLAY_CONFIG["layer_name"]].visible = False
        if CROSSHAIR_CONFIG["layer_name"] in self.viewer.layers:
            self.viewer.layers[CROSSHAIR_CONFIG["layer_name"]].visible = False
        if MILLING_PATTERN_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[MILLING_PATTERN_LAYER_NAME].visible = False
        return

    def draw_current_stage_position(self, stage_position: Optional[FibsemStagePosition] = None):
        """Draws the current stage position on the image."""
        if self.image is None or self.image.metadata is None:
            return

        if self.is_acquiring:
            self._hide_overlay_layers()
            return

        if stage_position is None:
            stage_position = self.microscope._stage_position

        self._draw_overlay_shapes(stage_position=stage_position)
        self._draw_position_crosshairs()
        self._draw_milling_pattern_overlay()

    def _collect_all_overlays(self, stage_position: FibsemStagePosition) -> List[NapariShapeOverlay]:
        """Collect all overlay shapes to be drawn on the image."""

        if self.image is None or self.image.metadata is None:
            return []

        # If no overlays are to be shown, return empty list
        if not (self.show_current_fov or 
                self.show_overview_fov or 
                self.show_saved_positions_fov or 
                self.show_stage_limits or
                self.show_circle_overlays
                ):
            return []

        stage_position.name = "Current Position"
        points = tiled.reproject_stage_positions_onto_image2(self.image, [stage_position])

        pixelsize = self.image.metadata.pixel_size.x
        current_position = points[0]

        overlays = []

        # current overview fov
        if self.show_overview_fov:
            overview_settings = self.get_overview_settings()
            fov = overview_settings.image_settings.hfw
            nrows, ncols = overview_settings.nrows, overview_settings.ncols
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
        if (self.show_stage_limits or self.show_circle_overlays) and self.microscope.stage_is_compustage:
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
            if self.show_stage_limits:
                rect = create_rectangle_shape(grid_centre, width, height)
                overlays.append(NapariShapeOverlay(
                    shape=rect,
                    color="yellow",
                    label="Stage Limits",
                    shape_type='rectangle')
                )
            if self.show_circle_overlays:
                # grid boundary circle (red)
                origin_radius = 1000e-6 / pixelsize  # 1000μm in meters
                origin_circle = create_circle_shape(grid_centre, origin_radius, None)
                overlays.append(NapariShapeOverlay(
                    shape=origin_circle,
                    color="red",
                    label="Grid Boundary",
                    shape_type="ellipse"
                ))

        if self.show_saved_positions_fov:
            points = tiled.reproject_stage_positions_onto_image2(self.image, self.positions)
            width = 80e-6 / pixelsize # TODO: make this match the milling fov
            height = 1024/1536 * width
            selected_index = self.lamella_list.selected_index
            for i, (lam, point) in enumerate(zip(self.lamellas, points)):
                overlays.append(NapariShapeOverlay(
                    shape=create_rectangle_shape(point, width=width, height=height),
                    color=self._lamella_color(lam, selected=i == selected_index),
                    label=point.name,
                    shape_type='rectangle')
                )

        return overlays

    def _draw_overlay_shapes(self, stage_position: FibsemStagePosition,
                             layer_scale: Optional[Tuple[float, float]] = None):
        """Draw all overlay shapes (FOV boxes and circles) on a single layer.

        Creates overlays for:
        - Current position single image FOV (magenta rectangles)
        - Overview acquisition area (orange rectangles, only if not acquiring)
        - Saved positions FOV (cyan rectangles)
        - Circle overlays (various colors based on configuration)

        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
            stage_position: Optional FibsemStagePosition for the current stage position
        """
        layer_name = OVERLAY_CONFIG["layer_name"]

        try:
            # Collect all overlay shapes
            overlays = self._collect_all_overlays(stage_position=stage_position)

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

        current_stage_position = deepcopy(self.microscope._stage_position)
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
        selected_index = self.lamella_list.selected_index
        lamellas = self.lamellas
        positions = self.positions
        pts = tiled.reproject_stage_positions_onto_image2(self.image, positions)
        for i, (lam, saved_point) in enumerate(zip(lamellas, pts)):
            saved_lines = create_crosshair_shape(saved_point, crosshair_size, layer_scale)
            color = self._lamella_color(lam, selected=i == selected_index)

            # Show position name on crosshair if saved position FOV is disabled
            label = lam.name if not self.show_saved_positions_fov else ""

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
            if MILLING_PATTERN_LAYER_NAME in self.viewer.layers:
                self.viewer.layers[MILLING_PATTERN_LAYER_NAME].visible = False
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
        if MILLING_PATTERN_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[MILLING_PATTERN_LAYER_NAME].visible = True

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
        self.correlation_panel.setEnabled(True) # TODO: allow enabling grid-bar overlay separately
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

    def _toggle_correlation_mode(self, event: Optional[NapariEvent] = None):
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
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
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
