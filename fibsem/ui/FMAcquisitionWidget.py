import logging
import os
import threading
import yaml
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import napari
import numpy as np
from napari.layers import Image as NapariImageLayer, Shapes as NapariShapesLayer
from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QVBoxLayout,
    QWidget,
    QMenuBar,
    QAction,
)
from superqt import QCollapsible
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem import conversions, utils
from fibsem.config import LOG_PATH
from fibsem.constants import METRE_TO_MILLIMETRE
from fibsem.fm.acquisition import (
    AutofocusMode,
    acquire_and_stitch_tileset,
    acquire_at_positions,
    acquire_image,
    acquire_z_stack,
    calculate_grid_coverage_area,
)
from fibsem.fm.calibration import run_autofocus
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, FMStagePosition, ZParameters, FluorescenceImageMetadata
from fibsem.structures import FibsemStagePosition, Point
from fibsem.ui.FibsemMovementWidget import to_pretty_string_short
from fibsem.ui.fm.widgets import (
    MultiChannelSettingsWidget,
    HistogramWidget,
    ObjectiveControlWidget,
    OverviewParametersWidget,
    SavedPositionsWidget,
    ZParametersWidget,
)

from fibsem.ui.napari.utilities import (
    create_circle_shape,
    create_crosshair_shape,
    create_rectangle_shape,
)
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)


@dataclass
class NapariShapeOverlay:
    """Represents a shape overlay for napari with its properties."""
    shape: np.ndarray
    color: str
    label: str
    shape_type: str  # "rectangle", "ellipse", or "line"
    
    def __post_init__(self):
        """Validate shape type."""
        if self.shape_type not in ["rectangle", "ellipse", "line"]:
            raise ValueError(f"Invalid shape_type: {self.shape_type}. Must be 'rectangle', 'ellipse', or 'line'")

# Overlay layer configuration constants
OVERLAY_CONFIG = {
    "layer_name": "overlay-shapes",
    "text_properties": {
        "color": "white",
        "font_size": 50,
        "anchor": "upper_left",
        "translation": (5, 5),
    },
    "rectangle_style": {
        "edge_width": 30,
        "face_color": "transparent",
        "opacity": 0.7,
    },
    "circle_style": {
        "edge_width": 40,
        "face_color": "transparent", 
        "opacity": 0.7,
    },
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
        "edge_width": 12,
        "face_color": "transparent",
    },
    "colors": {
        "origin": "red",
        "current": "yellow",
        "saved_selected": "lime",
        "saved_unselected": "cyan",
    },
}


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


class DisplayOptionsDialog(QDialog):
    """Dialog for configuring display overlay options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Display Options")
        self.setModal(True)
        self.initUI()
    
    def initUI(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("Overlay Display Options")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Checkboxes for each display option
        self.checkbox_current_fov = QCheckBox("Show Current FOV")
        self.checkbox_current_fov.setChecked(self.parent_widget.show_current_fov)
        layout.addWidget(self.checkbox_current_fov)
        
        self.checkbox_overview_fov = QCheckBox("Show Overview FOV")
        self.checkbox_overview_fov.setChecked(self.parent_widget.show_overview_fov)
        layout.addWidget(self.checkbox_overview_fov)
        
        self.checkbox_saved_positions_fov = QCheckBox("Show Saved Positions FOV")
        self.checkbox_saved_positions_fov.setChecked(self.parent_widget.show_saved_positions_fov)
        layout.addWidget(self.checkbox_saved_positions_fov)
        
        self.checkbox_stage_limits = QCheckBox("Show Stage Limits")
        self.checkbox_stage_limits.setChecked(self.parent_widget.show_stage_limits)
        layout.addWidget(self.checkbox_stage_limits)
        
        self.checkbox_circle_overlays = QCheckBox("Show Circle Overlays")
        self.checkbox_circle_overlays.setChecked(self.parent_widget.show_circle_overlays)
        layout.addWidget(self.checkbox_circle_overlays)
        
        self.checkbox_histogram = QCheckBox("Show Image Histogram")
        self.checkbox_histogram.setChecked(self.parent_widget.show_histogram)
        layout.addWidget(self.checkbox_histogram)
        
        # Buttons
        button_layout = QGridLayout()
        
        self.button_ok = QPushButton("OK")
        self.button_ok.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.button_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.button_ok, 0, 0)
        
        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.button_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.button_cancel, 0, 1)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_display_options(self) -> dict:
        """Get the selected display options."""
        return {
            'show_current_fov': self.checkbox_current_fov.isChecked(),
            'show_overview_fov': self.checkbox_overview_fov.isChecked(),
            'show_saved_positions_fov': self.checkbox_saved_positions_fov.isChecked(),
            'show_stage_limits': self.checkbox_stage_limits.isChecked(),
            'show_circle_overlays': self.checkbox_circle_overlays.isChecked(),
            'show_histogram': self.checkbox_histogram.isChecked(),
        }

def stage_position_to_napari_image_coordinate(image_shape: Union[Tuple[int, int], Tuple[int, ...]], pos: Optional[FibsemStagePosition], pixelsize: float) -> Point:
    """Convert a sample-stage coordinate to a napari image layer coordinate.
    
    This handles the offset and scaling for positioning images within napari layers,
    accounting for image dimensions and pixel size.
    
    Args:
        image_shape: Shape of the image (height, width)
        pos: Stage position in meters
        pixelsize: Pixel size in meters
        
    Returns:
        Point in napari image layer coordinates
    """
    if pos is None or pos.x is None or pos.y is None:
        raise ValueError("Stage position must have valid x and y coordinates.")

    p = Point(
        x = pos.x - image_shape[1] * pixelsize / 2,
                y = -pos.y - image_shape[0] * pixelsize / 2)
    return p

def napari_image_coordinate_to_stage_position(image_shape: Tuple[int, int], pos: Point, pixelsize: float) -> FibsemStagePosition:
    """Convert a napari image layer coordinate to a sample-stage coordinate.
    
    This handles the reverse conversion from napari image coordinates back to
    stage coordinates, accounting for image dimensions and pixel size.
    
    Args:
        image_shape: Shape of the image (height, width)
        pos: Point in napari image coordinates
        pixelsize: Pixel size in meters
        
    Returns:
        FibsemStagePosition in stage coordinates (meters)
    """
    if pos.x is None or pos.y is None:
        raise ValueError("Stage position must have valid x and y coordinates.")

    p = FibsemStagePosition(
        x = pos.x + image_shape[1] * pixelsize / 2,
        y = -pos.y - image_shape[0] * pixelsize / 2)

    return p

def stage_position_to_napari_world_coordinate(stage_position: FibsemStagePosition) -> Point:
    """Convert from stage coordinates to napari world coordinates.
    
    This performs a simple coordinate system conversion for direct interaction
    with the napari viewer (mouse clicks, overlays, etc.). The conversion
    only involves inverting the Y coordinate to match napari's coordinate system.
    
    Args:
        stage_position: Position in stage coordinates (meters)
        
    Returns:
        Point in napari world coordinates (meters)
        
    Note:
        - X coordinate remains unchanged
        - Y coordinate is inverted (stage_y → -napari_y)
        - Used for mouse interactions and drawing overlays
    """
    if stage_position.x is None or stage_position.y is None:
        raise ValueError("Stage position must have valid x and y coordinates.")

    return Point(stage_position.x, -stage_position.y)

def napari_world_coordinate_to_stage_position(napari_coordinate: Point) -> FibsemStagePosition:
    """Convert from napari world coordinates to stage coordinates.
    
    This performs the reverse conversion from napari viewer coordinates back
    to stage coordinates. Used for processing mouse click events and converting
    viewer interactions back to stage movements.
    
    Args:
        napari_coordinate: Point in napari world coordinates (meters)
        
    Returns:
        FibsemStagePosition in stage coordinates (meters)
        
    Note:
        - X coordinate remains unchanged  
        - Y coordinate is inverted (napari_y → -stage_y)
        - Used for mouse click processing and stage movement commands
    """
    if napari_coordinate.x is None or napari_coordinate.y is None:
        raise ValueError("Napari coordinate must have valid x and y coordinates.")
    return FibsemStagePosition(x=napari_coordinate.x, y=-napari_coordinate.y)

def _image_metadata_to_napari_image_layer(metadata: FluorescenceImageMetadata, 
                                          image_shape: Tuple[int, int], channel_index: int = 0) -> dict:
    """Convert FluorescenceImageMetadata to a dictionary compatible with napari image layer.
    This function extracts relevant metadata from the FluorescenceImageMetadata object
    and formats it into a dictionary that can be used to create a napari image layer.
    Args:
        metadata: FluorescenceImageMetadata object containing image metadata
        image_shape: Shape of the image (height, width)
        channel_index: Index of the channel to extract metadata for (default is 0)
    Returns:
        A dictionary containing the metadata formatted for napari image layer.
    Raises:
        ValueError: If metadata is None or does not contain required fields.
    """
    # Convert structured metadata to dictionary for napari compatibility
    metadata_dict = metadata.to_dict() if metadata else {}

    channel_name = metadata.channels[channel_index].name
    wavelength = metadata.channels[channel_index].excitation_wavelength
    emission_wavelength = metadata.channels[channel_index].emission_wavelength
    stage_position = metadata.stage_position

    pos = stage_position_to_napari_image_coordinate(image_shape, 
                                                    stage_position, 
                                                    metadata.pixel_size_x)

    colormap = "gray"
    if emission_wavelength is not None:
        colormap = wavelength_to_color(wavelength)

    return {
        "name": channel_name,
        "description": metadata.description or channel_name,
        "metadata": metadata_dict,
        "colormap": colormap,
        "scale": (metadata.pixel_size_y, metadata.pixel_size_x),  # yx order for napari
        "translate": (pos.y, pos.x),  # Translate to stage position
        "blending": "additive",
    }


MAX_OBJECTIVE_STEP_SIZE = 0.05  # mm
LIVE_IMAGING_RATE_LIMIT_SECONDS = 0.25  # seconds

z_parameters = ZParameters(
    zmin=-5e-6,     # -5 µm
    zmax=5e-6,      # +5 µm
    zstep=1e-6      # 1 µm step
)
channel_settings=ChannelSettings(
        name="Channel-01",
        excitation_wavelength=550,      # Example wavelength in nm
        emission_wavelength=None,       # Example wavelength in nm
        power=0.03,                      # Example power in W
        exposure_time=0.005,             # Example exposure time in seconds
)

# TODO: add a progress bar for each acquisition
# TODO: add user defined experiment directory + save/load images
# TODO: add user defined protocol (channel, z-stack parameters, overview parameters, etc.)
# TODO: enforce stage limits in the UI
# TODO: menu function to load images
# TODO: integrate with milling workflow
# TODO: multi-overview acquisition
# TODO: disable all controls during acquisition. FIX: can add positions during acquisition

# REFACTORING TODO: Extract common worker exception handling pattern and worker decorator
# REFACTORING TODO: Replace acquisition type magic strings with enum
# REFACTORING TODO: Address repeated "TODO: Show error message to user" comments

class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)
    update_persistent_image_signal = pyqtSignal(FluorescenceImage)
    acquisition_finished_signal = pyqtSignal()

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, experiment_path: str, parent=None):
        super().__init__(parent)

        self.fm = fm
        self.viewer = viewer
        self.image_layer: Optional[NapariImageLayer] = None
        self.stage_positions: List[FMStagePosition] = []
        self.experiment_path = experiment_path

        # widgets
        self.channelSettingsWidget: MultiChannelSettingsWidget
        self.objectiveControlWidget: ObjectiveControlWidget
        self.zParametersWidget: ZParametersWidget
        self.overviewParametersWidget: OverviewParametersWidget
        self.savedPositionsWidget: SavedPositionsWidget
        self.histogramWidget: HistogramWidget

        # Consolidated acquisition threading
        self._acquisition_thread: Optional[threading.Thread] = None
        self._acquisition_stop_event = threading.Event()
        self._current_acquisition_type: Optional[str] = None

        # Rate limiting for update_image
        self._last_updated_at = None
        self.max_update_interval = LIVE_IMAGING_RATE_LIMIT_SECONDS  # seconds
        if isinstance(self.fm.parent, DemoMicroscope):
            self.max_update_interval = 0.11  # faster updates for demo mode

        # Display flags for overlay controls
        self.show_current_fov = True
        self.show_overview_fov = True
        self.show_saved_positions_fov = True
        self.show_stage_limits = True
        self.show_circle_overlays = True
        self.show_histogram = True

        self.initUI()
        self.display_stage_position_overlay()

    @property
    def is_acquisition_active(self) -> bool:
        """Check if any acquisition or operation is currently running.

        Returns:
            True if any acquisition (single image, overview, z-stack, positions) or autofocus is active
        """
        return self._current_acquisition_type is not None

    def _validate_parent_microscope(self) -> bool:
        """Validate that the parent microscope is available for operations requiring stage control.
        
        Returns:
            True if parent microscope is available, False otherwise
        """
        if self.fm.parent is None:
            logging.error("FluorescenceMicroscope parent is None. Cannot perform operation.")
            return False
        return True

    def _get_current_settings(self):
        """Get current settings from all widgets for acquisition operations.
        
        Returns:
            Dictionary containing all current settings for acquisitions
        """
        # Get channel settings (always a list now)
        channel_settings = self.channelSettingsWidget.channel_settings
        
        # Get selected channel for live acquisition
        selected_channel_settings = self.channelSettingsWidget.selected_channel
        
        return {
            'channel_settings': channel_settings,
            'selected_channel_settings': selected_channel_settings,
            'z_parameters': self.zParametersWidget.z_parameters,
            'overview_grid_size': self.overviewParametersWidget.get_grid_size(),
            'overview_overlap': self.overviewParametersWidget.get_overlap(),
            'overview_use_zstack': self.overviewParametersWidget.get_use_zstack(),
            'overview_autofocus_mode': self.overviewParametersWidget.get_autofocus_mode(),
        }

    def initUI(self):
        """Initialize the user interface for the FMAcquisitionWidget."""

        self.label = QLabel("FM Acquisition Widget", self)

        # Add objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.objectiveCollapsible = QCollapsible("Objective Control", self)
        self.objectiveCollapsible.addWidget(self.objectiveControlWidget)

        # create z parameters widget
        self.zParametersWidget = ZParametersWidget(z_parameters=z_parameters, parent=self)
        self.zParametersCollapsible = QCollapsible("Z-Stack Parameters", self)
        self.zParametersCollapsible.addWidget(self.zParametersWidget)

        # create overview parameters widget
        self.overviewParametersWidget = OverviewParametersWidget(parent=self)
        self.overviewCollapsible = QCollapsible("Overview Parameters", self)
        self.overviewCollapsible.addWidget(self.overviewParametersWidget)

        # create saved positions widget
        self.savedPositionsWidget = SavedPositionsWidget(parent=self)
        self.positionsCollapsible = QCollapsible("Saved Positions", self)
        self.positionsCollapsible.addWidget(self.savedPositionsWidget)

        # create channel settings widget
        self.channelSettingsWidget = MultiChannelSettingsWidget(
            fm=self.fm,
            channel_settings=channel_settings,
            parent=self
        )
        self.channelCollapsible = QCollapsible("Channel Settings", self)
        self.channelCollapsible.addWidget(self.channelSettingsWidget)

        # Set initial expanded state for all collapsible widgets
        self.objectiveCollapsible.expand(animate=False)
        self.zParametersCollapsible.expand(animate=False)
        self.overviewCollapsible.expand(animate=False)
        self.positionsCollapsible.expand(animate=False)
        self.channelCollapsible.expand(animate=False)

        # Set content margins to 0 for all collapsible widgets
        self.objectiveCollapsible.setContentsMargins(0, 0, 0, 0)
        self.zParametersCollapsible.setContentsMargins(0, 0, 0, 0)
        self.overviewCollapsible.setContentsMargins(0, 0, 0, 0)
        self.positionsCollapsible.setContentsMargins(0, 0, 0, 0)
        self.channelCollapsible.setContentsMargins(0, 0, 0, 0)

        # create histogram widget
        self.histogramWidget = HistogramWidget(parent=self)
        self.histogram_dock = self.viewer.window.add_dock_widget(self.histogramWidget, name="Image Histogram", area='left')
        
        # Set initial histogram visibility based on display option
        self.histogram_dock.setVisible(self.show_histogram)

        self.pushButton_toggle_acquisition = QPushButton("Start Acquisition", self)
        self.pushButton_acquire_single_image = QPushButton("Acquire Image", self)
        self.pushButton_acquire_zstack = QPushButton("Acquire Z-Stack", self)
        self.pushButton_acquire_overview = QPushButton("Acquire Overview", self)
        self.pushButton_acquire_at_positions = QPushButton("Acquire at Saved Positions (0)", self)
        self.pushButton_run_autofocus = QPushButton("Run Auto-Focus", self)
        self.pushButton_cancel_acquisition = QPushButton("Cancel Acquisition", self)

        # Define button configurations for data-driven state management
        self.button_configs = [
            (self.pushButton_toggle_acquisition, GREEN_PUSHBUTTON_STYLE),
            (self.pushButton_acquire_single_image, BLUE_PUSHBUTTON_STYLE),
            (self.pushButton_acquire_zstack, BLUE_PUSHBUTTON_STYLE),
            (self.pushButton_acquire_overview, BLUE_PUSHBUTTON_STYLE),
            (self.pushButton_run_autofocus, ORANGE_PUSHBUTTON_STYLE),
        ]

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.objectiveCollapsible)
        layout.addWidget(self.zParametersCollapsible)
        layout.addWidget(self.overviewCollapsible)
        layout.addWidget(self.positionsCollapsible)
        layout.addWidget(self.channelCollapsible)
        layout.setContentsMargins(0, 0, 0, 0)

        # create grid layout for buttons
        button_layout = QGridLayout()
        button_layout.addWidget(self.pushButton_toggle_acquisition, 0, 0, 1, 2)
        button_layout.addWidget(self.pushButton_run_autofocus, 1, 0, 1, 2)
        button_layout.addWidget(self.pushButton_acquire_single_image, 2, 0)
        button_layout.addWidget(self.pushButton_acquire_zstack, 2, 1)
        button_layout.addWidget(self.pushButton_acquire_overview, 3, 0, 1, 2)
        button_layout.addWidget(self.pushButton_acquire_at_positions, 4, 0, 1, 2)
        button_layout.addWidget(self.pushButton_cancel_acquisition, 5, 0, 1, 2)
        button_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(button_layout)

        # set layout -> content -> scroll area -> main layout
        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_widget = QWidget(self)
        content_widget.setLayout(layout)
        layout.addStretch()
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)        
        self.setLayout(main_layout)
        scroll_area.setContentsMargins(0, 0, 0, 0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setContentsMargins(0, 0, 0, 0)

        # connect signals
        self.overviewParametersWidget.spinBox_rows.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.spinBox_cols.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.doubleSpinBox_overlap.valueChanged.connect(self._update_overview_bounding_box)
        
        # Connect Z parameter changes to update overview z-stack planes display
        self.zParametersWidget.doubleSpinBox_zmin.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)
        self.zParametersWidget.doubleSpinBox_zmax.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)
        self.zParametersWidget.doubleSpinBox_zstep.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)
        # Initial channel inputs for live acquisition are now handled by the widget itself
        self.pushButton_toggle_acquisition.clicked.connect(self.toggle_acquisition)
        self.pushButton_acquire_single_image.clicked.connect(self.acquire_image)
        self.pushButton_acquire_zstack.clicked.connect(self.acquire_image)
        self.pushButton_acquire_overview.clicked.connect(self.acquire_overview)
        self.pushButton_acquire_at_positions.clicked.connect(self.acquire_at_positions)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(lambda image: self.update_image_signal.emit(image)) 
        self.update_image_signal.connect(self.update_image)
        self.update_persistent_image_signal.connect(self.update_persistent_image)
        self.acquisition_finished_signal.connect(self._on_acquisition_finished)

        # Setup keyboard shortcuts
        self.f6_shortcut = QShortcut(QKeySequence("F6"), self)
        self.f6_shortcut.activated.connect(self.toggle_acquisition)

        self.f7_shortcut = QShortcut(QKeySequence("F7"), self)
        self.f7_shortcut.activated.connect(self.acquire_image)

        self.f8_shortcut = QShortcut(QKeySequence("F8"), self)
        self.f8_shortcut.activated.connect(self.run_autofocus)

        # movement controls
        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)
        self.viewer.mouse_wheel_callbacks.append(self.on_mouse_wheel)
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)

        # Connect layer selection to histogram updates
        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_changed)

        # stylesheets
        self.pushButton_toggle_acquisition.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.pushButton_acquire_single_image.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_acquire_zstack.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_acquire_overview.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_acquire_at_positions.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_run_autofocus.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_cancel_acquisition.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_cancel_acquisition.hide()  # Hide by default, show when acquisition starts
        self.pushButton_toggle_acquisition.setEnabled(True)

        # Explicitly enable acquisition buttons initially (disabled only during live acquisition)
        self.pushButton_acquire_single_image.setEnabled(True)
        self.pushButton_acquire_zstack.setEnabled(True)
        self.pushButton_acquire_overview.setEnabled(True)
        self.pushButton_acquire_at_positions.setEnabled(True)
        self.pushButton_run_autofocus.setEnabled(True)

        # Initialize positions button state
        self._update_positions_button()

        # add file menu
        self.menubar = QMenuBar(self)
        self.file_menu = self.menubar.addMenu("File")
        load_action = QAction("Load Positions", self)
        load_action.triggered.connect(self.savedPositionsWidget._load_positions_from_file)
        self.file_menu.addAction(load_action)
        
        # Add separator and display options
        self.file_menu.addSeparator()
        display_options_action = QAction("Display Options...", self)
        display_options_action.triggered.connect(self.show_display_options_dialog)
        self.file_menu.addAction(display_options_action)
        
        self.layout().setMenuBar(self.menubar)

        # draw scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

    def _on_layer_selection_changed(self, event):
        """Handle napari layer selection changes to update histogram."""
        try:
            added_layers = list(event.added)
            if not added_layers:
                return
            layer = added_layers[0]

            # Check if it's an image layer
            if isinstance(layer.data, np.ndarray) and layer.data.ndim in (2, 3, 4):
                self.histogramWidget.update_histogram(layer.data, layer.name)
                
        except Exception as e:
            logging.warning(f"Error updating histogram from layer selection: {e}")
            self.histogramWidget.clear_histogram()

    def on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events in the napari viewer."""

        if 'Shift' not in event.modifiers:
            event.handled = False  # Let napari handle zooming if Shift is not pressed
            return

        # Prevent objective movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Objective movement disabled during acquisition")
            event.handled = True
            return

        # Calculate step size based on wheel delta
        objective_step_size = self.objectiveControlWidget.doubleSpinBox_objective_step_size.value() * 1e-3 # convert from um to mm    
        step_mm = np.clip(objective_step_size * event.delta[1], -MAX_OBJECTIVE_STEP_SIZE, MAX_OBJECTIVE_STEP_SIZE)  # Adjust sensitivity as needed
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

    def on_mouse_click(self, viewer, event):
        """Handle mouse click events in the napari viewer."""

        # only left mouse button
        if event.button != 1:
            return

        # Check for modifier keys - Alt to add new position, Shift to update existing
        if 'Alt' not in event.modifiers and 'Shift' not in event.modifiers:
            return

        # Prevent stage movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            event.handled = True
            return

        # get event position in world coordinates, convert to stage coordinates
        position_clicked = event.position[-2:]  # yx required
        stage_position = napari_world_coordinate_to_stage_position(Point(x=position_clicked[1], y=position_clicked[0]))
        logging.info(f"Mouse clicked at {event.position}. Stage position: {stage_position}")

        if 'Alt' in event.modifiers:
            # Add new position
            current_objective_position = self.fm.objective.position

            # Create FMStagePosition with automatic name generation
            fm_stage_position = FMStagePosition.create_from_current_position(
                stage_position=stage_position,
                objective_position=current_objective_position,
                num=len(self.stage_positions) + 1
            )
            self.stage_positions.append(fm_stage_position)
            logging.info(f"New stage position saved: {fm_stage_position}")

        elif 'Shift' in event.modifiers:
            # Update existing position
            current_index = self.savedPositionsWidget.comboBox_positions.currentIndex()

            if current_index < 0 or current_index >= len(self.stage_positions):
                logging.warning("No position selected to update. Please select a position first.")
                event.handled = True
                return

            current_position = self.stage_positions[current_index]

            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Update Position",
                f"Update position '{current_position.name}' to new coordinates?\n\n"
                f"Current: X={current_position.stage_position.x*1e6:.1f} μm, "
                f"Y={current_position.stage_position.y*1e6:.1f} μm\n"
                f"New: X={stage_position.x*1e6:.1f} μm, "
                f"Y={stage_position.y*1e6:.1f} μm",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                logging.info(f"Position update cancelled for '{current_position.name}'")
                event.handled = True  # Prevent further processing
                return

            # Update only the stage position, keep name and objective position
            self.stage_positions[current_index].stage_position = stage_position
            logging.info(f"Updated position '{current_position.name}' to new stage coordinates: {stage_position}")

        # Update positions button and widget
        self._save_positions_to_yaml()
        self.savedPositionsWidget.update_positions(self.stage_positions)
        self.draw_stage_position_crosshairs()
        self._update_positions_button()
        event.handled = True  # Prevent further processing by napari

    def on_mouse_double_click(self, viewer, event):
        """Handle double-click events in the napari viewer."""

        # only left clicks
        if event.button != 1:
            return

        # Prevent stage movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            event.handled = True
            return
        
        if not self._validate_parent_microscope():
            return

        logging.info(f"Mouse double-clicked at {event.position} in viewer {viewer}")
        # coords = self.image_layer.world_to_data(event.position)  # PIXEL COORDINATES
        logging.info("-" * 40)
        position_clicked = event.position[-2:]  # yx required
        stage_position = napari_world_coordinate_to_stage_position(Point(x=position_clicked[1], y=position_clicked[0]))  # yx required

        self.fm.parent.move_stage_absolute(stage_position) # TODO: support absolute stable-move

        stage_position = self.fm.parent.get_stage_position()
        logging.info(f"Stage position after move: {stage_position}")

        self.display_stage_position_overlay()
        return

    def update_text_overlay(self):
        """Update the text overlay with current stage position and objective information."""
        try:
            if self.fm.parent is None:
                logging.warning("FluorescenceMicroscope parent is None. Cannot update text overlay.")
                return
            pos = self.fm.parent.get_stage_position()
            orientation = self.fm.parent.get_stage_orientation()
            current_grid = self.fm.parent.current_grid

            # Create combined text for overlay
            overlay_text = (
                f"STAGE: {pos.pretty_string} [{orientation}] [{current_grid}]\n"
                f"OBJECTIVE: {self.fm.objective.position*1e3:.3f} mm"
            )
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.position = "bottom_left"
            self.viewer.text_overlay.text = overlay_text

        except Exception as e:
            logging.warning(f"Error updating text overlay: {e}")
            # Fallback text if stage position unavailable
            self.viewer.text_overlay.text = "Fluorescence Acquisition Widget"

    def display_stage_position_overlay(self):
        """Legacy method for compatibility - redirects to update_text_overlay."""
        self.update_text_overlay()
        self.draw_stage_position_crosshairs()

    def show_display_options_dialog(self):
        """Show the display options dialog and apply changes."""
        dialog = DisplayOptionsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Get the new display options
            options = dialog.get_display_options()
            
            # Apply the options
            self.show_current_fov = options['show_current_fov']
            self.show_overview_fov = options['show_overview_fov']
            self.show_saved_positions_fov = options['show_saved_positions_fov']
            self.show_stage_limits = options['show_stage_limits']
            self.show_circle_overlays = options['show_circle_overlays']
            self.show_histogram = options['show_histogram']
            
            # Apply histogram visibility
            self._update_histogram_visibility()
            
            # Refresh the display
            self.display_stage_position_overlay()
            
            logging.info("Display options updated successfully")

    def _update_histogram_visibility(self):
        """Update the visibility of the histogram dock widget."""
        try:
            if hasattr(self, 'histogram_dock'):
                self.histogram_dock.setVisible(self.show_histogram)
        except Exception as e:
            logging.warning(f"Could not update histogram visibility: {e}")

    def draw_stage_position_crosshairs(self):
        """Draw multiple crosshairs showing various stage positions on a single layer.

        Creates crosshair overlays in the napari viewer showing:
        - Origin crosshair (red) at stage coordinates (0,0) 
        - Current stage position crosshair (yellow)
        - Saved stage positions crosshairs (cyan) from self.stage_positions list

        Each crosshair consists of horizontal and vertical lines intersecting at the 
        respective stage position, with text labels for identification.

        Features:
            - All crosshairs combined on single "stage-position" layer
            - Color-coded: red (origin), yellow (current), cyan (saved positions)
            - Text labels showing position names ("origin", "stage-position", custom names)
            - Automatic coordinate conversion from stage to pixel coordinates
            - Updates existing layer or creates new one if it doesn't exist

        Note:
            - Crosshair size is fixed at 50 pixels in each direction
            - Uses stage coordinate system converted to pixel coordinates via camera pixel size
            - Handles multiple positions dynamically based on self.stage_positions list
            - Uses NapariShapeOverlay dataclass for type safety
        """
        try:

            # Get coordinate conversion scale from camera pixel size
            layer_scale = (self.fm.camera.pixel_size[0], self.fm.camera.pixel_size[1])
                
            # Draw all overlay shapes (FOV boxes and circles) on single layer
            self._draw_overlay_shapes(layer_scale)
            self._draw_crosshair_overlay(layer_scale)

        except Exception as e:
            logging.warning(f"Error drawing stage position crosshairs: {e}")

    def _draw_overlay_shapes(self, layer_scale: Tuple[float, float]):
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
            overlays = self._collect_all_overlays(layer_scale)
            
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
    
    def _collect_all_overlays(self, layer_scale: Tuple[float, float]) -> List[NapariShapeOverlay]:
        """Collect all overlay shapes (FOV rectangles and circles).
        
        Returns:
            List of NapariShapeOverlay objects for all overlay shapes
        """
        overlays = []
        
        # Get camera FOV once for all calculations
        fov_x, fov_y = self.fm.camera.field_of_view
        
        # Add current position single image FOV (magenta)
        if self.fm.parent:
            current_pos = self.fm.parent.get_stage_position()
            center_point = stage_position_to_napari_world_coordinate(current_pos)

            if self.show_current_fov:
                fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
                overlays.append(NapariShapeOverlay(
                    shape=fov_rect,
                    color="magenta",
                    label="Current FOV",
                    shape_type="rectangle"
                ))

            # Add overview acquisition area (orange, only if not acquiring)
            if (self.show_overview_fov and 
                self._current_acquisition_type != "overview" and 
                self._current_acquisition_type != "positions"):

                # Get overview parameters and calculate total area
                settings = self._get_current_settings()
                grid_size = settings['overview_grid_size']
                overlap = settings['overview_overlap']

                total_width, total_height = calculate_grid_coverage_area(
                    ncols=grid_size[1], nrows=grid_size[0],
                    fov_x=fov_x, fov_y=fov_y, overlap=overlap
                )

                overview_rect = create_rectangle_shape(center_point, total_width, total_height, layer_scale)
                overlays.append(NapariShapeOverlay(
                    shape=overview_rect,
                    color="orange",
                    label=f"Overview {grid_size[0]}x{grid_size[1]}",
                    shape_type="rectangle"
                ))

        # Add 1mm bounding box around origin (yellow)
        if self.show_stage_limits:
            origin_point = Point(x=0, y=0)
            origin_size = 0.8e-3  # 0.8mm in meters
            origin_rect = create_rectangle_shape(origin_point, origin_size, origin_size, layer_scale)
            overlays.append(NapariShapeOverlay(
                shape=origin_rect,
                color="yellow",
                label="Stage Limits",
                shape_type="rectangle"
            ))

        # Add saved positions FOVs
        if self.show_saved_positions_fov:
            selected_index = -1
            if self.savedPositionsWidget.comboBox_positions.currentIndex() >= 0:
                selected_index = self.savedPositionsWidget.comboBox_positions.currentIndex()
            
            for i, saved_pos in enumerate(self.stage_positions):
                center_point = stage_position_to_napari_world_coordinate(saved_pos.stage_position)
                fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)

                # Use lime for selected position, cyan for others
                color = "lime" if i == selected_index else "cyan"
                overlays.append(NapariShapeOverlay(
                    shape=fov_rect,
                    color=color,
                    label=saved_pos.name,
                    shape_type="rectangle"
                ))

        # Add circle overlays
        if self.show_circle_overlays:
            # grid boundary circle (red)
            origin_point = Point(x=0, y=0)
            origin_radius = 1000e-6  # 1000μm in meters
            origin_circle = create_circle_shape(origin_point, origin_radius, layer_scale)
            overlays.append(NapariShapeOverlay(
                shape=origin_circle,
                color="red",
                label="Grid Boundary",
                shape_type="ellipse"
            ))

        # You can add more circles here by following the same pattern
        # Example: Add circles at saved positions
        # for i, saved_pos in enumerate(self.stage_positions):
        #     circle_point = Point(x=saved_pos.x, y=-saved_pos.y)
        #     circle_radius = 50e-6  # 50μm radius
        #     circle_shape = create_circle_shape(circle_point, circle_radius, layer_scale)
        #     overlays.append(NapariShapeOverlay(
        #         shape=circle_shape,
        #         color="blue",
        #         label=f"Circle {i+1}",
        #         shape_type="ellipse"
        #     ))

        return overlays

    def _update_overview_bounding_box(self):
        """Update the FOV boxes when parameters change."""
        # Don't update bounding box during overview or position acquisition
        if self._current_acquisition_type == "overview" or self._current_acquisition_type == "positions":
            return

        try:
            layer_scale = (self.fm.camera.pixel_size[0], self.fm.camera.pixel_size[1])
            self._draw_overlay_shapes(layer_scale)
        except Exception as e:
            logging.warning(f"Error updating overlay shapes: {e}")


    def _draw_crosshair_overlay(self, layer_scale: Tuple[float, float]):
        """Draw all crosshair overlays on a single layer.

        Creates crosshair overlays for:
        - Origin crosshair (red) at stage coordinates (0,0) 
        - Current stage position crosshair (yellow)
        - Saved stage positions crosshairs (lime for selected, cyan for others)

        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
        """
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
            layer = self.viewer.layers[layer_name]
            layer.data = crosshair_lines
            # Note: edge_color and text updates may not work with all napari versions
            try:
                layer.edge_color = colors
                layer.edge_width = CROSSHAIR_CONFIG["line_style"]["edge_width"]
                layer.text = text_properties
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

    def _collect_crosshair_overlays(self, layer_scale: Tuple[float, float]) -> List[NapariShapeOverlay]:
        """Collect all crosshair overlays for stage positions.

        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion

        Returns:
            List of NapariShapeOverlay objects for crosshair lines
        """
        overlays = []
        crosshair_size = CROSSHAIR_CONFIG["crosshair_size"]

        # Add origin position (0,0)
        origin_point = Point(x=0, y=0)
        origin_lines = create_crosshair_shape(origin_point, crosshair_size, layer_scale)
        for line, txt in zip(origin_lines, ["origin", ""]):
            overlays.append(NapariShapeOverlay(
                shape=line,
                color=CROSSHAIR_CONFIG["colors"]["origin"],
                label=txt,
                shape_type="line"
            ))

        # Add current stage position if available
        if self.fm.parent:
            current_pos = self.fm.parent.get_stage_position()
            current_point = stage_position_to_napari_world_coordinate(current_pos)
            current_lines = create_crosshair_shape(current_point, crosshair_size, layer_scale)
            for line, txt in zip(current_lines, ["stage-position", ""]):
                overlays.append(NapariShapeOverlay(
                    shape=line,
                    color=CROSSHAIR_CONFIG["colors"]["current"],
                    label=txt,
                    shape_type="line"
                ))

        # Add saved positions
        selected_index = self.savedPositionsWidget.comboBox_positions.currentIndex()
        for i, saved_pos in enumerate(self.stage_positions):
            saved_point = stage_position_to_napari_world_coordinate(saved_pos.stage_position)
            saved_lines = create_crosshair_shape(saved_point, crosshair_size, layer_scale)
            
            # Use lime for selected position, cyan for others
            color = CROSSHAIR_CONFIG["colors"]["saved_selected"] if i == selected_index else CROSSHAIR_CONFIG["colors"]["saved_unselected"]
            
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

    def _update_positions_button(self):
        """Update the positions acquisition button text and state based on saved positions."""
        num_positions = len(self.stage_positions)
        button_text = f"Acquire at Saved Positions ({num_positions})"
        self.pushButton_acquire_at_positions.setText(button_text)
        
        # Enable/disable button based on whether positions exist
        if num_positions == 0:
            self.pushButton_acquire_at_positions.setEnabled(False)
            self.pushButton_acquire_at_positions.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        else:
            self.pushButton_acquire_at_positions.setEnabled(True)
            self.pushButton_acquire_at_positions.setStyleSheet(BLUE_PUSHBUTTON_STYLE)

    def acquire_at_positions(self):
        """Start threaded acquisition at all saved positions."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire at positions while live acquisition is running. Stop acquisition first.")
            return

        if not self.stage_positions:
            logging.warning("No saved positions available for acquisition.")
            return

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info(f"Starting acquisition at {len(self.stage_positions)} saved positions")
        self._current_acquisition_type = "positions"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings['channel_settings']
        z_parameters = settings['z_parameters']

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._positions_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._acquisition_thread.start()
    
    def _positions_worker(self, channel_settings: List[ChannelSettings], z_parameters: Optional[ZParameters]):
        """Worker thread for positions acquisition."""
        try:
            logging.info(f"Acquiring at {len(self.stage_positions)} saved positions")
            
            # Get the parent microscope for stage movement
            if not self._validate_parent_microscope():
                return
            
            # Acquire images at all saved positions (using FMStagePosition directly)
            images = acquire_at_positions(
                microscope=self.fm.parent,
                positions=self.stage_positions,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                use_autofocus=False,
                save_directory=self.experiment_path,
            )
            
            # Check if acquisition was cancelled
            if self._acquisition_stop_event.is_set():
                logging.info("Positions acquisition was cancelled")
                return
            
            # Emit each acquired image
            for image in images:
                self.update_persistent_image_signal.emit(image)
            
            logging.info(f"Positions acquisition completed successfully. Acquired {len(images)} images.")
            
        except Exception as e:
            logging.error(f"Error during positions acquisition: {e}")
            # TODO: Show error message to user
            
        finally:
            # Signal that positions acquisition is finished (thread-safe)
            self.acquisition_finished_signal.emit()
    
    def _on_acquisition_finished(self):
        """Handle consolidated acquisition completion in the main thread."""

        # Clear acquisition state
        self._current_acquisition_type = None
        
        # Re-display overview FOV now that acquisition is complete
        self._update_overview_bounding_box()

        # Update objective position display
        self.objectiveControlWidget.update_objective_position_labels()

        # Update all button states
        self._update_acquisition_button_states()
        
        # Update histogram with current selected layer after acquisition ends
        try:
            selected_layers = list(self.viewer.layers.selection)
            if selected_layers:
                layer = selected_layers[0]
                if hasattr(layer, 'data') and layer.data is not None:
                    layer_name = getattr(layer, 'name', 'Unknown Layer')
                    self.histogramWidget.update_histogram(layer.data, layer_name)
        except Exception as e:
            logging.debug(f"Could not update histogram after acquisition: {e}")

    # NOTE: not in main thread, so we need to handle signals properly
    @pyqtSlot(FluorescenceImage)
    def update_image(self, image: FluorescenceImage):
        """Update the napari image layer with the given image data and metadata. (Live images)"""

        # Rate limiting: only update if enough time has passed since last update
        now = datetime.now()

        if self._last_updated_at is not None:
            time_since_last_update = now - self._last_updated_at
            if time_since_last_update < timedelta(seconds=self.max_update_interval):
                return  # Early return if not enough time has passed

        self._last_updated_at = now

        # acq_date = image.metadata.acquisition_date
        # self.label.setText(f"Acquisition Signal Received: {acq_date}")
        # logging.info(f"Image updated with shape: {image.data.shape}, Objective position: {self.fm.objective.position*1e3:.2f} mm")
        # logging.info(f"Metadata: {image.metadata.channels[0].to_dict()}")

        # Convert metadata to napari image layer format
        image_height, image_width = image.data.shape[-2:]
        metadata_dict = _image_metadata_to_napari_image_layer(image.metadata, (image_height, image_width))
        channel_name = metadata_dict["name"]
        logging.info(f"Updating image layer: {channel_name}, shape: {image.data.shape}, acquisition date: {image.metadata.acquisition_date}")

        self._update_napari_image_layer(channel_name, image.data, metadata_dict)
        self.display_stage_position_overlay()

    @pyqtSlot(FluorescenceImage)
    def update_persistent_image(self, image: FluorescenceImage):
        """Update or create a napari image layer with the given image data and metadata. (Persistent images)"""
        
        # Convert structured metadata to dictionary for napari compatibility
        image_height, image_width = image.data.shape[-2:]
        for channel_index in range(image.data.shape[0]):
            metadata_dict = _image_metadata_to_napari_image_layer(image.metadata, (image_height, image_width), channel_index=channel_index)
            layer_name = f"{metadata_dict['description']}-{metadata_dict['name']}"
            self._update_napari_image_layer(layer_name, image.data[channel_index], metadata_dict)

    def _update_napari_image_layer(self, layer_name: str, image: np.ndarray, metadata_dict: dict):
        """Update or create a napari image layer with the given metadata.
        Args:
            layer_name: Name of the napari image layer
            image: FluorescenceImage object containing the image data and metadata
            metadata_dict: Dictionary containing metadata for the image layer
        Returns:
            None
        """

        # Add a singleton dimension for z if needed
        if image.ndim == 3 and len(metadata_dict["scale"]) == 2:
            metadata_dict["scale"] = (1, *metadata_dict["scale"])

        if layer_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[layer_name].data = image
            self.viewer.layers[layer_name].metadata = metadata_dict["metadata"]
            self.viewer.layers[layer_name].colormap = metadata_dict["colormap"]
            self.viewer.layers[layer_name].translate = metadata_dict["translate"]
        else:
            # If the layer does not exist, create a new one
            self.viewer.add_image(
                data=image,
                name=layer_name,
                metadata=metadata_dict["metadata"],
                colormap=metadata_dict["colormap"],
                scale=metadata_dict["scale"],
                translate=metadata_dict["translate"],
                blending="additive",
            )
        # Update histogram (widget handles visibility checking internally)
        self.histogramWidget.update_histogram(image, layer_name)

    def start_acquisition(self):
        """Start the fluorescence acquisition."""
        if self.fm.is_acquiring:
            logging.warning("Acquisition is already running.")
            return

        # TODO: handle case where acquisition fails...

        # Get current settings
        settings = self._get_current_settings()
        selected_channel_settings = settings['selected_channel_settings']

        if selected_channel_settings is None:
            logging.warning("No channel selected for live acquisition")
            return

        logging.info(f"Starting acquisition with channel settings: {selected_channel_settings}")

        self.fm.start_acquisition(channel_settings=selected_channel_settings)
        self._update_acquisition_button_states()

    def stop_acquisition(self):
        """Stop the fluorescence acquisition."""

        logging.info("Acquisition stopped")

        self.fm.stop_acquisition()
        self._update_acquisition_button_states()

    def cancel_acquisition(self):
        """Cancel all ongoing acquisitions (single image, z-stack, overview, positions, autofocus)."""
        logging.info("Cancelling all acquisitions")

        # Cancel consolidated acquisition (single image or future consolidated types)
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.info(f"Cancelling {self._current_acquisition_type} acquisition")
            self._acquisition_stop_event.set()
            self._acquisition_thread.join(timeout=5)

        logging.info("All acquisitions cancelled")

    def toggle_acquisition(self):
        """Toggle acquisition start/stop with F6 key."""
        if self.fm.is_acquiring:
            logging.info("F6 pressed: Stopping acquisition")
            self.stop_acquisition()
        else:
            logging.info("F6 pressed: Starting acquisition")
            self.start_acquisition()

    def acquire_image(self):
        """Start threaded image acquisition using the current Z parameters and channel settings."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire image while live acquisition is running. Stop acquisition first.")
            return

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info("Starting image acquisition")
        self._current_acquisition_type = "image"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings['channel_settings']

        z_parameters = None
        if self.sender() is self.pushButton_acquire_zstack:
            z_parameters = settings['z_parameters']

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._image_acquistion_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._acquisition_thread.start()
    
    def _image_acquistion_worker(self, channel_settings: List[ChannelSettings], z_parameters: ZParameters):
        """Worker thread for single image acquisition."""
        try:
            # Generate filename for saving
            name = "z-stack" if z_parameters is not None else "image"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}-{timestamp}.ome.tiff"
            filepath = os.path.join(self.experiment_path, filename)
            
            # Acquire image with cancellation support and automatic saving
            image = acquire_image(
                microscope=self.fm,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                filename=filepath
            )

            # Check if acquisition was cancelled
            if self._acquisition_stop_event.is_set() or image is None:
                logging.info("image acquisition was cancelled")
                return

            # Emit the image
            self.update_persistent_image_signal.emit(image)

            logging.info("Image acquisition completed successfully")
            
        except Exception as e:
            logging.error(f"Error during image acquisition: {e}")
            # TODO: Show error message to user
            
        finally:
            # Signal that image acquisition is finished (thread-safe)
            self.acquisition_finished_signal.emit()
        
    def _update_acquisition_button_states(self):
        """Update acquisition button states and control widgets based on live acquisition or specific acquisition status."""
        # Check if any acquisition is active (live or specific acquisitions)
        any_acquisition_active = self.fm.is_acquiring or self.is_acquisition_active
        
        # Special case buttons with unique behavior
        self.pushButton_cancel_acquisition.setVisible(self.is_acquisition_active)
        
        # Update toggle acquisition button text and style based on state
        if self.fm.is_acquiring:
            self.pushButton_toggle_acquisition.setText("Stop Acquisition")
            self.pushButton_toggle_acquisition.setStyleSheet(RED_PUSHBUTTON_STYLE)
        else:
            self.pushButton_toggle_acquisition.setText("Start Acquisition")
            self.pushButton_toggle_acquisition.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

        # Update standard buttons using configuration from initUI (excluding toggle button)
        for button, normal_style in self.button_configs:
            if button == self.pushButton_toggle_acquisition:
                # Toggle button is always enabled, handled separately above
                continue
            if any_acquisition_active:
                button.setEnabled(False)
                button.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
            else:
                button.setEnabled(True)
                button.setStyleSheet(normal_style)

        # Special handling for positions button (depends on saved positions)
        if any_acquisition_active:
            self.pushButton_acquire_at_positions.setEnabled(False)
            self.pushButton_acquire_at_positions.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        else:
            self._update_positions_button()

        # Disable control widgets during acquisition
        self.objectiveControlWidget.setEnabled(not any_acquisition_active)
        self.zParametersWidget.setEnabled(not any_acquisition_active)
        self.overviewParametersWidget.setEnabled(not any_acquisition_active)
        self.savedPositionsWidget.setEnabled(not any_acquisition_active)
        
        # Disable channel settings during specific acquisitions (but allow during live imaging)
        self.channelSettingsWidget.setEnabled(not self.is_acquisition_active)
        
        # Disable channel list selection during any acquisition to prevent switching channels mid-acquisition
        self.channelSettingsWidget.channel_list.setEnabled(not any_acquisition_active)

    def _save_positions_to_yaml(self):
        """Save current stage positions to positions.yaml in the experiment directory."""
        if not self.experiment_path:
            logging.warning("No experiment path set, cannot save positions")
            return

        positions_file = os.path.join(self.experiment_path, "positions.yaml")

        try:
            # Convert positions to dictionary format for YAML serialization
            positions_data = {
                'positions': [pos.to_dict() for pos in self.stage_positions],
                'created_date': datetime.now().isoformat(),
                'num_positions': len(self.stage_positions)
            }

            with open(positions_file, 'w') as f:
                yaml.dump(positions_data, f, default_flow_style=False, indent=2)

            logging.info(f"Saved {len(self.stage_positions)} positions to {positions_file}")

        except Exception as e:
            logging.error(f"Failed to save positions to {positions_file}: {e}")

    def acquire_overview(self):
        """Start threaded overview acquisition using the current channel settings."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire overview while live acquisition is running. Stop acquisition first.")
            return

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info("Starting overview acquisition")

        self._current_acquisition_type = "overview"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings['channel_settings']
        grid_size = settings['overview_grid_size']
        tile_overlap = settings['overview_overlap']
        use_zstack = settings['overview_use_zstack']
        z_parameters = settings['z_parameters'] if use_zstack else None
        autofocus_mode = settings['overview_autofocus_mode']
        # TODO: support positions, autofocus channel and autofocus z parameters
        # TODO: allow setting different sized overviews for each position
        positions = None #self.stage_positions

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._overview_worker,
            args=(channel_settings, grid_size, tile_overlap, z_parameters, autofocus_mode, positions),
            daemon=True
        )
        self._acquisition_thread.start()

    def _overview_worker(self,
                        channel_settings: List[ChannelSettings],
                        grid_size: tuple[int, int],
                        tile_overlap: float,
                        z_parameters: Optional[ZParameters],
                        autofocus_mode: AutofocusMode, 
                        positions: Optional[List[FMStagePosition]] = None):
        """Worker thread for overview acquisition."""
        try:
            # Get the parent microscope for tileset acquisition
            if not self._validate_parent_microscope():
                return
            if positions is None:
                positions = []

            # if positions:
            #     logging.info(f"Acquiring overview at {len(positions)} saved positions")
            #     from fibsem.fm.acquisition import acquire_multiple_overviews

            #     overview_images = acquire_multiple_overviews(
            #         microscope=self.fm.parent,
            #         positions=self.stage_positions,
            #         channel_settings=channel_settings,
            #         grid_size=grid_size,
            #         tile_overlap=tile_overlap,
            #         zparams=z_parameters,
            #         autofocus_mode=autofocus_mode,
            #         save_directory=self.experiment_path,
            #         stop_event=self._acquisition_stop_event
            #     )
            #     # Check if acquisition was cancelled
            #     if self._acquisition_stop_event.is_set() or overview_images is None:
            #         logging.info("Overview acquisition was cancelled")
            #         return

            #     # Process the acquired overview images
            #     for img in overview_images:
            #         self.update_persistent_image_signal.emit(img)
            #     return

            # Acquire and stitch tileset
            overview_image = acquire_and_stitch_tileset(
                microscope=self.fm.parent,
                channel_settings=channel_settings,
                grid_size=grid_size,
                tile_overlap=tile_overlap,
                zparams=z_parameters,
                autofocus_mode=autofocus_mode,
                save_directory=self.experiment_path,
                stop_event=self._acquisition_stop_event
            )

            # Check if acquisition was cancelled
            if self._acquisition_stop_event.is_set() or overview_image is None:
                logging.info("Overview acquisition was cancelled")
                return

            # Emit the overview image
            self.update_persistent_image_signal.emit(overview_image)

            logging.info("Overview acquisition completed successfully")

        except Exception as e:
            logging.error(f"Error during overview acquisition: {e}")
            # TODO: Show error message to user

        finally:
            # Signal that overview acquisition is finished (thread-safe)
            self.acquisition_finished_signal.emit()

    # update methods for live updates
    def _update_exposure_time(self, value: float):
        if self.fm.is_acquiring:
            self.fm.set_exposure_time(value * 1e-3)  # Convert ms to seconds

    def _update_power(self, value: float):
        if self.fm.is_acquiring:
            self.fm.set_power(value)

    def _update_excitation_wavelength(self, idx: int):
        if self.fm.is_acquiring:
            # Get the selected channel widget to retrieve the wavelength value
            selected_widget = self.channelSettingsWidget._get_selected_channel_widget()
            if selected_widget and selected_widget.excitation_wavelength_input:
                wavelength = selected_widget.excitation_wavelength_input.itemData(idx)
                logging.info(f"Updating excitation wavelength to: {wavelength} nm")
                self.fm.filter_set.excitation_wavelength = wavelength

    def _update_emission_wavelength(self, idx: int):
        if self.fm.is_acquiring:
            # Get the selected channel widget to retrieve the wavelength value
            selected_widget = self.channelSettingsWidget._get_selected_channel_widget()
            if selected_widget and selected_widget.emission_wavelength_input:
                wavelength = selected_widget.emission_wavelength_input.itemData(idx)
                logging.info(f"Updating emission wavelength to: {wavelength} nm")
                self.fm.filter_set.emission_wavelength = wavelength

    # override closeEvent to stop acquisition when the widget is closed
    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info("Closing FMAcquisitionWidget, stopping acquisition if running.")
        
        # Stop live acquisition
        if self.fm.is_acquiring:
            try:
                self.fm.acquisition_signal.disconnect()
                self.stop_acquisition()
                print("Acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                print("Acquisition stopped due to widget close.")
        
        # Stop consolidated acquisition
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            try:
                self._acquisition_stop_event.set()
                self._acquisition_thread.join(timeout=5)  # Wait up to 5 seconds
                logging.info(f"{self._current_acquisition_type} acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping {self._current_acquisition_type} acquisition: {e}")

        # Reset acquisition flags
        self._current_acquisition_type = None

        event.accept()

    def run_autofocus(self):
        """Start threaded auto-focus using the current channel settings and Z parameters."""
        if self.fm.is_acquiring:
            logging.warning("Cannot run autofocus while live acquisition is running. Stop acquisition first.")
            return

        if self._current_acquisition_type is not None:
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info("Starting auto-focus")
        self._current_acquisition_type = "autofocus"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings['channel_settings']
        z_parameters = settings['z_parameters']

        # Start auto-focus thread
        self._acquisition_thread = threading.Thread(
            target=self._autofocus_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._acquisition_thread.start()

    def _autofocus_worker(self, channel_settings: List[ChannelSettings], z_parameters: ZParameters):
        """Worker thread for auto-focus."""
        try:
            logging.info("Running auto-focus with laplacian method")
            
            # Run autofocus using laplacian method (default)
            # Use first channel for autofocus
            first_channel = channel_settings[0] if channel_settings else None
            if not first_channel:
                logging.error("No channel settings available for autofocus")
                return
                
            best_z = run_autofocus(
                microscope=self.fm,
                channel_settings=first_channel,
                # z_parameters=z_parameters,
                method='laplacian',
                stop_event=self._acquisition_stop_event
            )

            # Check if auto-focus was cancelled
            if best_z is None or self._acquisition_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return

            logging.info(f"Auto-focus completed successfully. Best focus: {best_z*1e6:.1f} μm")

        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")

        finally:
            # Signal that auto-focus is finished (thread-safe)
            self.acquisition_finished_signal.emit()


def create_widget(viewer: napari.Viewer) -> FMAcquisitionWidget:
    CONFIG_PATH = None#r"C:\Users\User\Documents\github\openfibsem\fibsem-os\fibsem\config\tfs-arctis-configuration.yaml"
    microscope, settings = utils.setup_session(config_path=CONFIG_PATH)

    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized. Cannot create FMAcquisitionWidget.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    from fibsem.microscopes.simulator import DemoMicroscope
    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")

    # ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True

    # Create experiment path with current directory + datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = os.path.join(LOG_PATH, f"fibsem_experiment_{timestamp}")
    
    # Create the experiment directory
    try:
        os.makedirs(experiment_path, exist_ok=True)
        logging.info(f"Created experiment directory: {experiment_path}")
    except Exception as e:
        logging.error(f"Failed to create experiment directory: {e}")
        # Fallback to current directory
        experiment_path = os.getcwd()
    
    widget = FMAcquisitionWidget(fm=microscope.fm,
                                 viewer=viewer,
                                 experiment_path=experiment_path, parent=None)

    return widget

def main():

    viewer = napari.Viewer()
    widget = create_widget(viewer)    
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    return


if __name__ == "__main__":
    main()

