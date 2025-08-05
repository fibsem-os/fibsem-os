import logging
import os
from pathlib import Path
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
from fibsem.microscopes.simulator import DemoMicroscope, FibsemMicroscope
from fibsem import utils
from fibsem.config import LOG_PATH
from fibsem.constants import METRE_TO_MILLIMETRE
from fibsem.fm.acquisition import (
    AutofocusMode,
    acquire_and_stitch_tileset,
    acquire_at_positions,
    acquire_image,
    calculate_grid_coverage_area,
)
from fibsem.fm.calibration import run_autofocus
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, FMStagePosition, ZParameters, FluorescenceImageMetadata
from fibsem.structures import FibsemStagePosition, Point, FibsemImage, FibsemImageMetadata
from fibsem.ui.fm.widgets import (
    ChannelSettingsWidget,
    HistogramWidget,
    ObjectiveControlWidget,
    OverviewParametersWidget,
    SavedPositionsWidget,
    ZParametersWidget,
    LinePlotWidget,
    StagePositionControlWidget,
    SEMAcquisitionWidget,
    ExperimentCreationDialog,
)
from fibsem.applications.autolamella.structures import create_new_experiment, Experiment
from fibsem.ui.napari.utilities import (
    create_circle_shape,
    create_crosshair_shape,
    create_rectangle_shape,
    update_text_overlay,
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


class OverviewConfirmationDialog(QDialog):
    """Small confirmation dialog showing overview acquisition parameters."""

    def __init__(self, settings: dict, fm: FluorescenceMicroscope, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.fm = fm
        self.setWindowTitle("Overview Acquisition")
        self.setModal(True)
        self.initUI()

    def initUI(self):
        """Initialize the confirmation dialog UI."""
        layout = QVBoxLayout()

        # Parameters display
        params_layout = QVBoxLayout()

        # Grid size and total area
        grid_size: Tuple[int, int] = self.settings['overview_grid_size']
        try:
            fov_x, fov_y = self.fm.camera.field_of_view
            from fibsem.fm.acquisition import calculate_grid_coverage_area
            total_width, total_height = calculate_grid_coverage_area(
                ncols=grid_size[1], nrows=grid_size[0],
                fov_x=fov_x, fov_y=fov_y, overlap=self.settings['overview_overlap']
            )
            total_area = f"{total_width*1e6:.1f} x {total_height*1e6:.1f} μm"
        except Exception:
            total_area = "N/A"

        grid_label = QLabel(f"Grid Size: {grid_size[0]} x {grid_size[1]}. (Area: {total_area})")
        params_layout.addWidget(grid_label)

        # Channels
        channel_settings: List[ChannelSettings] = self.settings['channel_settings']
        channels_label = QLabel(f"Channels: {len(channel_settings)}")
        params_layout.addWidget(channels_label)

        for i, channel in enumerate(channel_settings):  # Show all channels
            channel_info = QLabel(f"  • {channel.pretty_name}")
            channel_info.setStyleSheet("font-size: 10px; color: #666666;")
            params_layout.addWidget(channel_info)

        # Z-stack parameters
        use_zstack = self.settings['overview_use_zstack']
        if use_zstack and self.settings['z_parameters']:
            z_params: ZParameters = self.settings['z_parameters']
            z_label = QLabel(f"{z_params.pretty_name}")
            params_layout.addWidget(z_label)

        # Auto-focus
        autofocus_mode: AutofocusMode = self.settings['overview_autofocus_mode']
        af_label = QLabel(f"Auto-Focus: {autofocus_mode.name.replace('_', ' ').title()}")
        params_layout.addWidget(af_label)

        layout.addLayout(params_layout)
        layout.addStretch()

        # Buttons
        button_layout = QGridLayout()
        self.button_start = QPushButton("Start Acquisition")
        self.button_start.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.button_start.clicked.connect(self.accept)
        button_layout.addWidget(self.button_start, 0, 0)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.button_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.button_cancel, 0, 1)

        layout.addLayout(button_layout)
        self.setLayout(layout)


class AcquisitionSummaryDialog(QDialog):
    """Dialog showing a summary of the acquisition before it starts."""

    def __init__(self, checked_positions: List[FMStagePosition], channel_settings: List[ChannelSettings], z_parameters: ZParameters, use_autofocus: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.checked_positions = checked_positions
        self.channel_settings = channel_settings
        self.z_parameters = z_parameters
        self.use_autofocus = use_autofocus
        self.setWindowTitle("Acquisition Summary")
        self.setModal(True)
        self.initUI()
        self.setContentsMargins(0, 0, 0, 0)

    def initUI(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Position information
        num_positions = len(self.checked_positions)
        position_label = QLabel(f"Positions: {num_positions}")
        position_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(position_label)

        # List position names
        position_names = [pos.pretty_name for pos in self.checked_positions]
        position_list_text = "\n".join([f"  • {name}" for name in position_names[:10]])  # Limit to first 10
        if len(position_names) > 10:
            position_list_text += f"\n  • ... and {len(position_names) - 10} more"

        position_details = QLabel(position_list_text)
        position_details.setStyleSheet("font-size: 10px; color: #666666;")
        layout.addWidget(position_details)

        # Channel information
        channel_label = QLabel(f"Channels: {len(self.channel_settings)}")
        channel_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(channel_label)

        for i, channel in enumerate(self.channel_settings):
            channel_details = QLabel(channel.pretty_name)
            channel_details.setStyleSheet("font-size: 10px; color: #666666;")
            layout.addWidget(channel_details)

        # Z-stack information
        if self.z_parameters:
            num_planes = self.z_parameters.num_planes
            z_details = QLabel(self.z_parameters.pretty_name)
        else:
            num_planes = 1
            z_details = QLabel("No Z-Stack")
        zlabel = QLabel("Z-Stack: " + str(num_planes) + " planes")
        zlabel.setStyleSheet("font-weight: bold; font-size: 12px;")
        z_details.setStyleSheet("font-size: 10px; color: #666666;")
        layout.addWidget(zlabel)
        layout.addWidget(z_details)
        
        # Auto-focus information
        autofocus_label = QLabel(f"Auto-focus: {'Enabled' if self.use_autofocus else 'Disabled'}")
        autofocus_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(autofocus_label)
        
        autofocus_details = QLabel("Autofocus will run at each position before acquisition" if self.use_autofocus else "No autofocus will be performed")
        autofocus_details.setStyleSheet("font-size: 10px; color: #666666;")
        layout.addWidget(autofocus_details)

        # Buttons
        button_layout = QGridLayout()
        self.button_start = QPushButton("Start Acquisition")
        self.button_start.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.button_start.clicked.connect(self.accept)
        button_layout.addWidget(self.button_start, 0, 0)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.button_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.button_cancel, 0, 1)

        layout.addLayout(button_layout)
        self.setLayout(layout)


class DisplayOptionsDialog(QDialog):
    """Dialog for configuring display overlay options."""

    def __init__(self, parent: 'FMAcquisitionWidget'):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Display Options")
        self.setModal(True)
        self.initUI()

    def initUI(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

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

def _fibsem_image_metadata_to_napari_image_layer(metadata: FibsemImageMetadata, shape: Tuple[int, int]) -> dict:
    """Convert FibsemImageMetadata to a dictionary suitable for napari image layer."""
    pos = stage_position_to_napari_image_coordinate(shape, metadata.stage_position, metadata.pixel_size.x)
    acq_date = datetime.fromtimestamp(metadata.microscope_state.timestamp).strftime('%Y-%m-%d-%H-%M-%S')
    return {
        "name": f"SEM Overview - {acq_date}",
        "description": "SEM Overview Image",
        "scale": (metadata.pixel_size.y, metadata.pixel_size.x),
        "translate": (pos.y, pos.x),
        "colormap": "gray",
        "metadata": {
            "timestamp": metadata.microscope_state.timestamp,
            "resolution": metadata.image_settings.resolution,
            "hfw": metadata.image_settings.hfw,
            "dwell_time": metadata.image_settings.dwell_time,
            "beam_type": metadata.beam_type.name
        }
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
# TODO: save/load images
# TODO: add user defined protocol (channel, z-stack parameters, overview parameters, etc.)
# TODO: enforce stage limits in the UI
# TODO: menu function to load images
# TODO: integrate with milling workflow
# TODO: Extract common worker exception handling pattern and worker decorator
# TODO: Replace acquisition type magic strings with enum

class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)
    update_persistent_image_signal = pyqtSignal(object)  # Union[FluorescenceImage, FibsemImage]
    acquisition_finished_signal = pyqtSignal()

    def __init__(self, microscope: FibsemMicroscope, viewer: napari.Viewer, experiment: Optional[Experiment] = None, parent=None):
        super().__init__(parent)

        if microscope.fm is None:
            raise ValueError("FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope.")

        self.microscope = microscope
        self.fm: FluorescenceMicroscope = microscope.fm
        self.viewer = viewer
        self.stage_positions: List[FMStagePosition] = []
        self.experiment: Optional[Experiment] = experiment

        # widgets
        self.channelSettingsWidget: ChannelSettingsWidget
        self.objectiveControlWidget: ObjectiveControlWidget
        self.stagePositionControlWidget: StagePositionControlWidget
        self.zParametersWidget: ZParametersWidget
        self.overviewParametersWidget: OverviewParametersWidget
        self.savedPositionsWidget: SavedPositionsWidget
        self.histogramWidget: HistogramWidget
        # self.line_plot_widget: LinePlotWidget

        # Consolidated acquisition threading
        self._acquisition_thread: Optional[threading.Thread] = None
        self._acquisition_stop_event = threading.Event()
        self._current_acquisition_type: Optional[str] = None

        # Rate limiting for update_image
        self._last_updated_at = None
        self.max_update_interval = LIVE_IMAGING_RATE_LIMIT_SECONDS  # seconds
        if isinstance(self.microscope, DemoMicroscope):
            self.max_update_interval = 0.11  # faster updates for demo mode

        # Display flags for overlay controls
        self.show_current_fov = True
        self.show_overview_fov = True
        self.show_saved_positions_fov = True
        self.show_stage_limits = True
        self.show_circle_overlays = True
        self.show_histogram = True

        self.sync_experiment_positions()
        self.initUI()
        self.display_stage_position_overlay()

    def sync_experiment_positions(self):
        """Sync stage positions from the experiment to the widget."""
        # TODO: save the fm state directly, and sync between the uis
        if self.experiment and self.experiment.positions:
            for pos in self.experiment.positions:
                if pos.objective_position is None:
                    logging.warning(f"Position {pos.name} does not have an objective position, skipping.")
                    continue
                if self.microscope.get_stage_orientation(pos.stage_position) != "FM":
                    logging.warning(f"Position {pos.name} is not in FM orientation, skipping.")
                    continue
                self.stage_positions.append(FMStagePosition(name=pos.name,
                                                            stage_position=pos.stage_position,
                                                            objective_position=pos.objective_position))

    def toggle_widgets(self):
        """Toggle widgets based on current stage orientation."""

        is_fm_enabled = self.microscope.get_stage_orientation() == "FM"
        is_sem_enabled = self.microscope.get_stage_orientation() == "SEM"

        self.objectiveControlWidget.setEnabled(is_fm_enabled)
        self.zParametersWidget.setEnabled(is_fm_enabled)
        self.channelSettingsWidget.setEnabled(is_fm_enabled)
        self.savedPositionsWidget.setEnabled(is_fm_enabled)
        self.overviewParametersWidget.setEnabled(is_fm_enabled)
        self.semAcquisitionWidget.setEnabled(is_fm_enabled)
        # toggle acquisition buttons based on orientation
        for btn, _ in self.button_configs:
            btn.setEnabled(is_fm_enabled)

    @property
    def is_acquisition_active(self) -> bool:
        """Check if any acquisition or operation is currently running.

        Returns:
            True if any acquisition (single image, overview, z-stack, positions) or autofocus is active
        """
        return self._current_acquisition_type is not None

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
            'overview_autofocus_channel_name': self.overviewParametersWidget.get_autofocus_channel_name(),
        }

    def initUI(self):
        """Initialize the user interface for the FMAcquisitionWidget."""

        # Add objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.objectiveCollapsible = QCollapsible("Objective Control", self)
        self.objectiveCollapsible.addWidget(self.objectiveControlWidget)

        # Add stage position control widget
        self.stagePositionControlWidget = StagePositionControlWidget(microscope=self.microscope, parent=self)
        self.stagePositionCollapsible = QCollapsible("Stage Position Control", self)
        self.stagePositionCollapsible.addWidget(self.stagePositionControlWidget)

        # add SEM image acquisition widget
        self.semAcquisitionWidget = SEMAcquisitionWidget(microscope=self.microscope, parent=self)
        self.semAcquisitionCollapsible = QCollapsible("SEM Image Acquisition", self)
        self.semAcquisitionCollapsible.addWidget(self.semAcquisitionWidget)

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
        self.channelSettingsWidget = ChannelSettingsWidget(
            fm=self.fm,
            channel_settings=channel_settings,
            parent=self
        )
        self.channelCollapsible = QCollapsible("Channel Settings", self)
        self.channelCollapsible.addWidget(self.channelSettingsWidget)

        # Set initial expanded state for all collapsible widgets
        # self.objectiveCollapsible.expand(animate=False)
        # self.stagePositionCollapsible.expand(animate=False)
        # self.semAcquisitionCollapsible.expand(animate=False)
        # self.zParametersCollapsible.expand(animate=False)
        # self.overviewCollapsible.expand(animate=False)
        # self.positionsCollapsible.expand(animate=False)
        # self.channelCollapsible.expand(animate=False)

        # Set content margins to 0 for all collapsible widgets
        self.objectiveCollapsible.setContentsMargins(0, 0, 0, 0)
        self.stagePositionCollapsible.setContentsMargins(0, 0, 0, 0)
        self.semAcquisitionCollapsible.setContentsMargins(0, 0, 0, 0)
        self.zParametersCollapsible.setContentsMargins(0, 0, 0, 0)
        self.overviewCollapsible.setContentsMargins(0, 0, 0, 0)
        self.positionsCollapsible.setContentsMargins(0, 0, 0, 0)
        self.channelCollapsible.setContentsMargins(0, 0, 0, 0)

        # create histogram widget
        self.histogramWidget = HistogramWidget(parent=self)
        self.histogram_dock = self.viewer.window.add_dock_widget(self.histogramWidget, name="Image Histogram", area='left')
        self.histogram_dock.setVisible(self.show_histogram)
        # self.line_plot_widget = LinePlotWidget(parent=self)
        # self.line_plot_dock = self.viewer.window.add_dock_widget(self.line_plot_widget, name="Line Plot", area='left', tabify=True)

        self.pushButton_toggle_acquisition = QPushButton("Start Acquisition", self)
        self.pushButton_acquire_single_image = QPushButton("Acquire Image", self)
        self.pushButton_acquire_zstack = QPushButton("Acquire Z-Stack", self)
        self.pushButton_acquire_overview = QPushButton("Acquire Overview", self)
        self.pushButton_acquire_at_positions = QPushButton("Acquire at Positions (0/0)", self)
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
        layout.addWidget(self.stagePositionCollapsible)
        layout.addWidget(self.semAcquisitionCollapsible)
        layout.addWidget(self.objectiveCollapsible)
        layout.addWidget(self.channelCollapsible)
        layout.addWidget(self.zParametersCollapsible)
        layout.addWidget(self.positionsCollapsible)
        layout.addWidget(self.overviewCollapsible)
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
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        scroll_area.setContentsMargins(0, 0, 0, 0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setContentsMargins(0, 0, 0, 0)

        # connect signals
        self.overviewParametersWidget.spinBox_rows.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.spinBox_cols.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.doubleSpinBox_overlap.valueChanged.connect(self._update_overview_bounding_box)

        self.zParametersWidget.doubleSpinBox_zmin.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)
        self.zParametersWidget.doubleSpinBox_zmax.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)
        self.zParametersWidget.doubleSpinBox_zstep.valueChanged.connect(self.overviewParametersWidget._update_zstack_planes_visibility)

        self.pushButton_toggle_acquisition.clicked.connect(self.toggle_acquisition)
        self.pushButton_acquire_single_image.clicked.connect(self.acquire_image)
        self.pushButton_acquire_zstack.clicked.connect(self.acquire_image)
        self.pushButton_acquire_overview.clicked.connect(self.acquire_overview)
        self.pushButton_acquire_at_positions.clicked.connect(self.acquire_at_positions)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(self._wrap_update_image)
        self.update_image_signal.connect(self.update_image)
        self.update_persistent_image_signal.connect(self.update_persistent_image)
        self.acquisition_finished_signal.connect(self._on_acquisition_finished)

        # keyboard shortcuts
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

        # draw scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

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
        self.savedPositionsWidget.update_positions(self.stage_positions)
        self._update_positions_button()

        # add file menu
        self.menubar = QMenuBar(self)

        # Experiment creation
        new_experiment_action = QAction("New Experiment...", self)
        new_experiment_action.triggered.connect(self.show_new_experiment_dialog)
        display_options_action = QAction("Display Options...", self)
        display_options_action.triggered.connect(self.show_display_options_dialog)
        run_coincidence_action = QAction("Open Coincidence Milling...", self)
        run_coincidence_action.triggered.connect(self.run_coincidence)
        self.file_menu = self.menubar.addMenu("File")
        if self.file_menu is not None:
            self.file_menu.addAction(new_experiment_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction(display_options_action)
            self.file_menu.addAction(run_coincidence_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction("Exit", self.close)
        main_layout.setMenuBar(self.menubar)

    def run_coincidence(self):
        from fibsem.ui.FMCoincidenceMillingWidget import create_widget, milling_stage
        viewer = napari.Viewer(title="Coincidence Milling")
        self.coincidence_widget = create_widget(self.microscope, viewer, [milling_stage])

        viewer.window.add_dock_widget(self.coincidence_widget, name="Coincidence Milling", area="right")
        napari.run(max_loop_level=2)

    def _on_layer_selection_changed(self, event):
        """Handle napari layer selection changes to update histogram."""
        try:
            added_layers = list(event.added)
            if not added_layers:
                return
            layer: NapariImageLayer = added_layers[0]

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
        step_mm = np.clip(objective_step_size * event.delta[1], -MAX_OBJECTIVE_STEP_SIZE, MAX_OBJECTIVE_STEP_SIZE)
        # delta ~= 1

        logging.info(f"Mouse wheel event detected with delta: {event.delta}, step size: {step_mm:.4f} mm")

        # Get current position in mm
        current_pos = self.fm.objective.position * METRE_TO_MILLIMETRE
        new_pos = current_pos + step_mm

        # Move objective
        logging.info(f"Moving objective by {step_mm:.4f} mm to {new_pos:.4f} mm")
        self.objectiveControlWidget.doubleSpinBox_objective_position.setValue(new_pos)
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

        current_orientation = self.microscope.get_stage_orientation()
        # get event position in world coordinates, convert to stage coordinates
        position_clicked = event.position[-2:]  # yx required
        stage_position = napari_world_coordinate_to_stage_position(Point(x=position_clicked[1], y=position_clicked[0]))
        current_stage_position = self.microscope.get_stage_position()
        stage_position.r = current_stage_position.r  
        stage_position.t = current_stage_position.t
        stage_position.z = current_stage_position.z  # keep current z,r,t
        logging.info(f"Mouse clicked at {event.position}. Stage position: {stage_position}")

        # TODO: QUERY: Does the position need to be flipped when in SEM orientation?

        if 'Alt' in event.modifiers:
            # Add new position
            if current_orientation == "FM":
                objective_position = self.fm.objective.position
            if current_orientation == "SEM":
                objective_position = self.fm.objective.focus_position

            # Create FMStagePosition with automatic name generation
            fm_stage_position = FMStagePosition.create_from_current_position(
                stage_position=stage_position,
                objective_position=objective_position,
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
        event.handled = True

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

        logging.info(f"Mouse double-clicked at {event.position} in viewer {viewer}")
        # coords = layer.world_to_data(event.position)  # PIXEL COORDINATES
        logging.info("-" * 40)
        position_clicked = event.position[-2:]  # yx required
        stage_position = napari_world_coordinate_to_stage_position(Point(x=position_clicked[1], y=position_clicked[0]))  # yx required

        self.microscope.move_stage_absolute(stage_position) # TODO: support absolute stable-move

        stage_position = self.microscope.get_stage_position()
        logging.info(f"Stage position after move: {stage_position}")

        self.display_stage_position_overlay()
        return

    def display_stage_position_overlay(self):
        """Legacy method for compatibility - redirects to update_text_overlay."""
        update_text_overlay(self.viewer, self.microscope)
        self.draw_stage_position_crosshairs()
        self.toggle_widgets()

    def show_new_experiment_dialog(self):
        """Show the experiment setup dialog for creating or loading experiments."""
        dialog = ExperimentCreationDialog(initial_directory=LOG_PATH, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            experiment_info = dialog.get_experiment_info()

            if experiment_info['mode'] == 'create':
                self._handle_create_experiment(experiment_info)
            else:
                self._handle_load_experiment(experiment_info)

    def _handle_create_experiment(self, experiment_info):
        """Handle creating a new experiment."""
        experiment_name = experiment_info['name']

        if not experiment_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid experiment name.")
            return

        # Create the experiment directory
        try:
            self.experiment = create_new_experiment(
                path=experiment_info['directory'],
                name=experiment_name,
                )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Creating Experiment", 
                f"Failed to create experiment directory:\n{str(e)}"
            )

    def _handle_load_experiment(self, experiment_info):
        """Handle loading an existing experiment."""
        positions_file = experiment_info['positions_file']
        experiment_path = experiment_info['full_path']

        try:
            # Load positions from the YAML file
            with open(positions_file, 'r') as f:
                positions_data = yaml.safe_load(f)

            # Convert positions back to FMStagePosition objects
            loaded_positions = []
            for pos_dict in positions_data.get('positions', []):
                try:
                    fm_position = FMStagePosition.from_dict(pos_dict)
                    loaded_positions.append(fm_position)
                except Exception as e:
                    logging.warning(f"Failed to load position {pos_dict}: {e}")

            # Update the widget state
            self.experiment = Experiment.load(Path(os.path.join(experiment_path, "experiment.yaml")))
            self.stage_positions = loaded_positions

            # Update the saved positions widget
            self.savedPositionsWidget.update_positions(self.stage_positions)
            self._update_positions_button()
            self.draw_stage_position_crosshairs()

            # Show success message
            QMessageBox.information(
                self,
                "Experiment Loaded",
                f"Loaded experiment '{os.path.basename(experiment_path)}' with {len(loaded_positions)} positions.\n\nPath: {experiment_path}"
            )

            logging.info(f"Loaded experiment from {experiment_path} with {len(loaded_positions)} positions")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Experiment",
                f"Failed to load experiment:\n{str(e)}"
            )
            logging.error(f"Failed to load experiment from {positions_file}: {e}")

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
        current_pos = self.microscope.get_stage_position()
        current_orientation = self.microscope.get_stage_orientation()
        center_point = stage_position_to_napari_world_coordinate(current_pos)

        if self.show_current_fov:
            fov_rect = None
            if current_orientation == "FM":
                fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
            elif current_orientation == "SEM":
                fov = self.semAcquisitionWidget.image_settings.hfw  # square fov
                fov_rect = create_rectangle_shape(center_point, fov, fov, layer_scale)
            if fov_rect is not None:
                overlays.append(NapariShapeOverlay(
                    shape=fov_rect,
                    color="magenta",
                    label=f"Current FOV ({current_orientation})",
                    shape_type="rectangle"
                ))

        # Add overview acquisition area (orange, only if not acquiring)
        if (self.show_overview_fov and 
            self._current_acquisition_type != "overview" and 
            self._current_acquisition_type != "positions" and
            current_orientation == "FM"):

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
        current_pos = self.microscope.get_stage_position()
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
        """Update the positions acquisition button text and state based on checked positions."""
        num_total_positions = len(self.stage_positions)
        num_checked_positions = len(self.savedPositionsWidget.get_checked_positions()) if hasattr(self, 'savedPositionsWidget') else 0
        button_text = f"Acquire at Positions ({num_checked_positions}/{num_total_positions})"
        self.pushButton_acquire_at_positions.setText(button_text)

        # Enable/disable button based on whether checked positions exist
        if num_checked_positions == 0:
            self.pushButton_acquire_at_positions.setEnabled(False)
            self.pushButton_acquire_at_positions.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        else:
            self.pushButton_acquire_at_positions.setEnabled(True)
            self.pushButton_acquire_at_positions.setStyleSheet(BLUE_PUSHBUTTON_STYLE)

    def acquire_at_positions(self):
        """Start threaded acquisition at all checked saved positions."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire at positions while live acquisition is running. Stop acquisition first.")
            return

        if not self.stage_positions:
            logging.warning("No saved positions available for acquisition.")
            return

        # Get only checked positions from the saved positions widget
        checked_positions = self.savedPositionsWidget.get_checked_positions()
        if not checked_positions:
            logging.warning("No positions are checked for acquisition. Please check at least one position.")
            return

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.warning("Another acquisition is already in progress.")
            return

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings['channel_settings']
        z_parameters = settings['z_parameters']
        
        # Get auto focus setting from the saved positions widget
        use_autofocus = self.savedPositionsWidget.get_auto_focus_enabled()

        # Show acquisition summary dialog
        summary_dialog = AcquisitionSummaryDialog(
            checked_positions=checked_positions,
            channel_settings=channel_settings,
            z_parameters=z_parameters,
            use_autofocus=use_autofocus,
            parent=self
        )

        # Only proceed if user confirms
        if summary_dialog.exec_() != QDialog.Accepted:
            logging.info("Acquisition cancelled by user")
            return

        logging.info(f"Starting acquisition at {len(checked_positions)} checked positions")
        self._current_acquisition_type = "positions"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._positions_worker,
            args=(checked_positions, channel_settings, z_parameters, use_autofocus),
            daemon=True
        )
        self._acquisition_thread.start()

    def _positions_worker(self,
                          checked_positions: List[FMStagePosition],
                          channel_settings: List[ChannelSettings],
                          z_parameters: Optional[ZParameters],
                          use_autofocus: bool):
        """Worker thread for positions acquisition."""
        try:
            logging.info(f"Acquiring at {len(checked_positions)} checked positions")

            # acquire images at only the checked positions
            images = acquire_at_positions(
                microscope=self.microscope,
                positions=checked_positions,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                use_autofocus=use_autofocus,
                save_directory=self.experiment.path,
            )

            if self._acquisition_stop_event.is_set():
                logging.info("Positions acquisition was cancelled")
                return

            # Emit each acquired image
            for image in images:
                self.update_persistent_image_signal.emit(image)

        except Exception as e:
            logging.error(f"Error during positions acquisition: {e}")

        finally:
            self.acquisition_finished_signal.emit()
    
    def _on_acquisition_finished(self):
        """Handle consolidated acquisition completion in the main thread."""

        # clear acquisition state
        self._current_acquisition_type = None

        # refresh overview bbox, objective widget, and button states
        self._update_overview_bounding_box()
        self.objectiveControlWidget.update_objective_position_labels()

        self._update_acquisition_button_states()

    def _wrap_update_image(self, image: FluorescenceImage):
        """Wrap the update image signal emission for re-emission in the main thread."""
        # rate limiting: only update if enough time has passed since last update
        now = datetime.now()
        if self._last_updated_at is not None:
            time_since_last_update = now - self._last_updated_at
            if time_since_last_update < timedelta(seconds=1.0):
                return  # early return if not enough time has passed
        self._last_updated_at = now

        self.update_image_signal.emit(image)

    # NOTE: not in main thread, so we need to handle signals properly
    @pyqtSlot(FluorescenceImage)
    def update_image(self, image: FluorescenceImage):
        """Update the napari image layer with the given image data and metadata. (Live images)"""

        # convert metadata to napari image layer format
        image_height, image_width = image.data.shape[-2:]
        metadata_dict = _image_metadata_to_napari_image_layer(image.metadata, (image_height, image_width))
        channel_name = metadata_dict["name"]
        logging.info(f"Updating image layer: {channel_name}, shape: {image.data.shape}, acquisition date: {image.metadata.acquisition_date}")

        self._update_napari_image_layer(channel_name, image.data, metadata_dict)
        # self.display_stage_position_overlay()

    @pyqtSlot(FluorescenceImage)
    def update_persistent_image(self, image: Union[FibsemImage, FluorescenceImage]):
        """Update or create a napari image layer with the given image data and metadata. (Persistent images)"""

        if isinstance(image, FibsemImage):
            self._update_fibsem_image(image)
            return

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
        # self.line_plot_widget.append_value(float(np.mean(image)), layer_name)

    def _update_fibsem_image(self, image: FibsemImage):
        """Update the napari image layer with the given FibsemImage data and metadata."""
        metadata_dict = _fibsem_image_metadata_to_napari_image_layer(image.metadata, image.data.shape)
        self._update_napari_image_layer(metadata_dict['name'], image.data, metadata_dict)

    def start_acquisition(self):
        """Start the fluorescence acquisition."""
        if self.fm.is_acquiring:
            logging.warning("Acquisition is already running.")
            return

        if self.microscope.get_stage_orientation() != "FM":
            logging.warning("Stage is not in FM orientation. Cannot start acquisition.")
            return
        
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            logging.warning("Another acquisition is already in progress.")
            return

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
            # generate filename for saving
            name = "z-stack" if z_parameters is not None else "image"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}-{timestamp}.ome.tiff"
            filepath = os.path.join(self.experiment.path, filename)

            image = acquire_image(
                microscope=self.fm,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                filename=filepath
            )

            if self._acquisition_stop_event.is_set() or image is None:
                logging.info("image acquisition was cancelled")
                return

            # Emit the image
            self.update_persistent_image_signal.emit(image)
            logging.info("Image acquisition completed successfully")

        except Exception as e:
            logging.error(f"Error during image acquisition: {e}")
            
        finally:
            self.acquisition_finished_signal.emit()
        
    def _update_acquisition_button_states(self):
        """Update acquisition button states and control widgets based on live acquisition or specific acquisition status."""
        if self.microscope.get_stage_orientation() != "FM":
            # If not in FM orientation, disable all acquisition buttons
            self.pushButton_toggle_acquisition.setEnabled(False)
            self.pushButton_acquire_zstack.setEnabled(False)
            self.pushButton_acquire_overview.setEnabled(False)
            self.pushButton_acquire_at_positions.setEnabled(False)
            self.pushButton_cancel_acquisition.setEnabled(False)
            return

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
        self.stagePositionControlWidget.setEnabled(not any_acquisition_active)
        self.zParametersWidget.setEnabled(not any_acquisition_active)
        self.overviewParametersWidget.setEnabled(not any_acquisition_active)
        self.savedPositionsWidget.setEnabled(not any_acquisition_active)

        # Disable channel settings during specific acquisitions (but allow during live imaging)
        self.channelSettingsWidget.setEnabled(not self.is_acquisition_active)

        # Disable channel list selection during any acquisition to prevent switching channels mid-acquisition
        self.channelSettingsWidget.channel_list.setEnabled(not any_acquisition_active)

    def _save_positions_to_yaml(self):
        """Save current stage positions to positions.yaml in the experiment directory."""
        if not self.experiment.path:
            logging.warning("No experiment path set, cannot save positions")
            return

        positions_file = os.path.join(self.experiment.path, "positions.yaml")

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

        # show overview confirmation dialog
        settings = self._get_current_settings()
        confirmation_dialog = OverviewConfirmationDialog(
            settings=settings,
            fm=self.fm,
            parent=self
        )

        # Only proceed if user confirms
        if confirmation_dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled by user")
            return

        logging.info("Starting overview acquisition")

        self._current_acquisition_type = "overview"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        channel_settings = settings['channel_settings']
        grid_size = settings['overview_grid_size']
        tile_overlap = settings['overview_overlap']
        use_zstack = settings['overview_use_zstack']
        z_parameters = settings['z_parameters'] if use_zstack else None
        autofocus_mode = settings['overview_autofocus_mode']
        autofocus_channel_name = settings['overview_autofocus_channel_name']
        # TODO: support positions and autofocus z parameters
        # TODO: allow setting different sized overviews for each position
        positions = None #self.stage_positions

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._overview_worker,
            args=(channel_settings,
                  grid_size,
                  tile_overlap,
                  z_parameters,
                  autofocus_mode,
                  autofocus_channel_name,
                  positions),
            daemon=True
        )
        self._acquisition_thread.start()

    def _overview_worker(self,
                        channel_settings: List[ChannelSettings],
                        grid_size: tuple[int, int],
                        tile_overlap: float,
                        z_parameters: Optional[ZParameters],
                        autofocus_mode: AutofocusMode, 
                        autofocus_channel_name: Optional[str],
                        positions: Optional[List[FMStagePosition]] = None):
        """Worker thread for overview acquisition."""
        try:
            if positions is None:
                positions = []

            # if positions:
            #     logging.info(f"Acquiring overview at {len(positions)} saved positions")
            #     from fibsem.fm.acquisition import acquire_multiple_overviews

            #     overview_images = acquire_multiple_overviews(
            #         microscope=self.microscope,
            #         positions=self.stage_positions,
            #         channel_settings=channel_settings,
            #         grid_size=grid_size,
            #         tile_overlap=tile_overlap,
            #         zparams=z_parameters,
            #         autofocus_mode=autofocus_mode,
            #         save_directory=self.experiment.path,
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
                microscope=self.microscope,
                channel_settings=channel_settings,
                grid_size=grid_size,
                tile_overlap=tile_overlap,
                zparams=z_parameters,
                autofocus_mode=autofocus_mode,
                autofocus_channel_name=autofocus_channel_name,
                save_directory=self.experiment.path,
                stop_event=self._acquisition_stop_event
            )

            if self._acquisition_stop_event.is_set() or overview_image is None:
                logging.info("Overview acquisition was cancelled")
                return

            # Emit the overview image
            self.update_persistent_image_signal.emit(overview_image)

            logging.info("Overview acquisition completed successfully")

        except Exception as e:
            logging.error(f"Error during overview acquisition: {e}")

        finally:
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
            selected_widget = self.channelSettingsWidget._get_selected_channel_widget()
            if selected_widget and selected_widget.excitation_wavelength_input:
                wavelength = selected_widget.excitation_wavelength_input.itemData(idx)
                self.fm.filter_set.excitation_wavelength = wavelength

    def _update_emission_wavelength(self, idx: int):
        if self.fm.is_acquiring:
            selected_widget = self.channelSettingsWidget._get_selected_channel_widget()
            if selected_widget and selected_widget.emission_wavelength_input:
                wavelength = selected_widget.emission_wavelength_input.itemData(idx)
                self.fm.filter_set.emission_wavelength = wavelength

    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info("Closing FMAcquisitionWidget, stopping acquisition if running.")

        # Stop live acquisition
        self.fm.acquisition_signal.disconnect(self._wrap_update_image)
        if self.fm.is_acquiring:
            try:
                self.stop_acquisition()
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                logging.warning("Acquisition stopped due to widget close.")

        # Stop acquisition worker thread if it is running
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            try:
                self._acquisition_stop_event.set()
                self._acquisition_thread.join(timeout=5)
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
        channel_settings = settings['selected_channel_settings']
        z_parameters = settings['z_parameters']

        # Start auto-focus thread
        self._acquisition_thread = threading.Thread(
            target=self._autofocus_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._acquisition_thread.start()

    def _autofocus_worker(self, channel_settings: ChannelSettings, z_parameters: ZParameters):
        """Worker thread for auto-focus."""
        try:
            logging.info("Running auto-focus with laplacian method")

  
            best_z = run_autofocus(
                microscope=self.fm,
                channel_settings=channel_settings,
                # z_parameters=z_parameters,
                method='laplacian',
                stop_event=self._acquisition_stop_event
            )

            if best_z is None or self._acquisition_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return

            logging.info(f"Auto-focus completed successfully. Best focus: {best_z*1e6:.1f} μm")

        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")
        finally:
            self.acquisition_finished_signal.emit()


def create_widget(viewer: napari.Viewer) -> FMAcquisitionWidget:
    # CONFIG_PATH = None #r"C:\Users\User\Documents\github\openfibsem\fibsem-os\fibsem\config\tfs-arctis-configuration.yaml"
    CONFIG_PATH = "/Users/patrickcleeve/Documents/fibsem/fibsem/fibsem/config/sim-arctis-configuration.yaml"
    microscope, settings = utils.setup_session(config_path=CONFIG_PATH)

    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized. Cannot create FMAcquisitionWidget.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")

    # ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True


    widget = FMAcquisitionWidget(microscope=microscope,
                                 viewer=viewer,
                                 parent=None)

    return widget

def main():

    viewer = napari.Viewer()
    widget = create_widget(viewer)    
    viewer.window.add_dock_widget(widget, area="right")
    if widget.experiment is None:
        widget.show_new_experiment_dialog()  # Show experiment creation dialog on startup
    napari.run()

    return


if __name__ == "__main__":
    main()

