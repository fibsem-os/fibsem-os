from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import napari
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Shapes as NapariShapesLayer
from PyQt5.QtCore import QEvent, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import median_filter
from superqt import ensure_main_thread

from fibsem import config as fcfg
from fibsem import constants, utils
from fibsem.applications.autolamella.structures import Experiment
from fibsem.fm.acquisition import (
    acquire_and_stitch_tileset,
    acquire_image,
    calculate_grid_coverage_area,
)
from fibsem.fm.calibration import run_autofocus
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceImage,
    FluorescenceImageMetadata,
    FluorescenceConfiguration,
    FMStagePosition,
    OverviewParameters,
    ZParameters,
)
from fibsem.microscopes.simulator import DemoMicroscope, FibsemMicroscope
from fibsem.structures import (
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    Point,
)
from fibsem.ui.fm.widgets import (
    AutofocusWidget,
    FluorescenceMultiChannelWidget,
    DisplayOptionsDialog,
    HistogramWidget,
    LoadImageDialog,
    ObjectiveControlWidget,
    OverviewConfirmationDialog,
    OverviewParametersWidget,
    ZParametersWidget,
)
from fibsem.ui.napari.utilities import (
    create_circle_shape,
    create_crosshair_shape,
    create_rectangle_shape,
    update_text_overlay,
    NapariShapeOverlay,
)
from fibsem.ui.stylesheets import (
    MILLING_PROGRESS_BAR_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget, TitledPanel
from fibsem.utils import format_duration


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
        "edge_width": 30,
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

# NOTE when orientation is SEM:
# FM views the grid from underside, so Y coordinates are inverted?
# when orientation is FM:
# FM views grid from topside, like SEM, but camera is inverted?
# need to figure out the views


def stage_position_to_napari_image_coordinate(
    image_shape: Union[Tuple[int, int], Tuple[int, ...]],
    pos: Optional[FibsemStagePosition],
    pixelsize: float,
) -> Point:
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

    mod = 1.0
    if np.isclose(pos.t, np.radians(-180)):
        mod = -1.0

    p = Point(
        x=pos.x - image_shape[1] * pixelsize / 2,
        y=mod * pos.y - image_shape[0] * pixelsize / 2,
    )
    return p


def napari_image_coordinate_to_stage_position(
    image_shape: Tuple[int, int], pos: Point, pixelsize: float
) -> FibsemStagePosition:
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
        x=pos.x + image_shape[1] * pixelsize / 2,
        y=-pos.y - image_shape[0] * pixelsize / 2,
    )

    return p


def stage_position_to_napari_world_coordinate(
    stage_position: FibsemStagePosition, inverted: bool = True
) -> Point:
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

    mod = 1.0
    if inverted:
        mod = -1.0

    return Point(stage_position.x, mod * stage_position.y)


def napari_world_coordinate_to_stage_position(
    napari_coordinate: Point, inverted: bool = True
) -> FibsemStagePosition:
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
    mod = 1.0
    if inverted:
        mod = -1.0
    return FibsemStagePosition(x=napari_coordinate.x, y=mod * napari_coordinate.y)


def _image_metadata_to_napari_image_layer(
    metadata: FluorescenceImageMetadata,
    image_shape: Tuple[int, int],
    channel_index: int = 0,
) -> dict:
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
    colormap = metadata.channels[channel_index].color or "gray"
    stage_position = metadata.stage_position
    pixel_size_x = metadata.pixel_size_x
    pixel_size_y = metadata.pixel_size_y

    pos = stage_position_to_napari_image_coordinate(
        image_shape, stage_position, pixel_size_x
    )

    return {
        "name": channel_name,
        "description": metadata.description or channel_name,
        "metadata": metadata_dict,
        "colormap": colormap,
        "scale": (pixel_size_y, pixel_size_x),  # yx order for napari
        "translate": (pos.y, pos.x),  # Translate to stage position
        "blending": "additive",
    }


def _fibsem_image_metadata_to_napari_image_layer(
    metadata: FibsemImageMetadata, shape: Tuple[int, int]
) -> dict:
    """Convert FibsemImageMetadata to a dictionary suitable for napari image layer."""
    pos = stage_position_to_napari_image_coordinate(
        shape, metadata.stage_position, metadata.pixel_size.x
    )
    acq_date = metadata.microscope_state.timestamp
    if isinstance(acq_date, (int, float)):
        acq_date = datetime.fromtimestamp(acq_date).strftime("%Y-%m-%d-%H-%M-%S")
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
            "beam_type": metadata.beam_type.name,
        },
    }


z_parameters = ZParameters(
    zmin=-5e-6,  # -5 µm
    zmax=5e-6,  # +5 µm
    zstep=1e-6,  # 1 µm step
)
channel_settings = ChannelSettings(
    name="Channel-01",
    excitation_wavelength=550,  # Example wavelength in nm
    emission_wavelength=None,  # Example wavelength in nm
    power=0.01,  # Example power in %  TODO: support setting in mW?
    exposure_time=0.001,  # Example exposure time in ms
)

# TODO: add export as png + metadata
# TODO: Extract common worker exception handling pattern and worker decorator
# TODO: Replace acquisition type magic strings with enum
# TODO: add autofocus widget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


class FMAcquisitionWidget(QWidget):
    # update_image_signal = pyqtSignal(FluorescenceImage)
    update_persistent_image_signal = pyqtSignal(
        object
    )  # Union[FluorescenceImage, FibsemImage]
    acquisition_finished_signal = pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope,
        experiment: Optional[Experiment] = None,
        parent: Optional["AutoLamellaUI"] = None,
        parent_dialog: Optional[QDialog] = None,
    ):
        super().__init__(parent_dialog or parent)

        if microscope.fm is None:
            raise ValueError(
                "FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope."
            )

        self.microscope = microscope
        self.fm: FluorescenceMicroscope = microscope.fm
        self.viewer = napari.Viewer(show=False, title="FM Acquisition")
        self.viewer.window._qt_window.menuBar().hide()
        self.viewer.window._qt_window.statusBar().hide()
        self.experiment: Optional[Experiment] = experiment
        self.parent_widget = parent

        # widgets
        self.channelSettingsWidget: FluorescenceMultiChannelWidget
        self.objectiveControlWidget: ObjectiveControlWidget
        self.zParametersWidget: ZParametersWidget
        self.overviewParametersWidget: OverviewParametersWidget
        self.lamella_list: LamellaNameListWidget
        self.label_fm_pose: QLabel
        self.autofocusWidget: AutofocusWidget
        self.histogramWidget: HistogramWidget

        # Consolidated acquisition threading
        self._acquisition_thread: Optional[threading.Thread] = None
        self._acquisition_stop_event = threading.Event()
        self._current_acquisition_type: Optional[str] = None

        # Rate limiting for update_image
        self._last_updated_at = None
        self._last_remaining_time = None

        # Display flags for overlay controls
        self.show_current_fov = True
        self.show_overview_fov = True
        self.show_saved_positions_fov = True
        self.show_stage_limits = True
        self.show_circle_overlays = True
        self.show_histogram = True

        self.initUI()
        self._update_stage_position_display()

    def toggle_widgets(self):
        """Toggle widgets based on current stage orientation."""

        is_fm_enabled = self.microscope.get_stage_orientation() in ["FM", "SEM"]

        self.objectiveControlWidget.setEnabled(is_fm_enabled)
        self.zParametersWidget.setEnabled(is_fm_enabled)
        self.channelSettingsWidget.setEnabled(is_fm_enabled)
        self.autofocusWidget.setEnabled(is_fm_enabled)
        self.lamella_list.setEnabled(is_fm_enabled)
        self.overviewParametersWidget.setEnabled(is_fm_enabled)
        # toggle acquisition buttons based on orientation
        for btn, _ in self.button_configs:
            btn.setEnabled(is_fm_enabled)

    @property
    def is_acquiring(self) -> bool:
        """Check if any acquisition is currently active."""
        return self.fm.is_acquiring or bool(
            self._acquisition_thread and self._acquisition_thread.is_alive()
        )

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
            "channel_settings": channel_settings,
            "selected_channel_settings": selected_channel_settings,
            "z_parameters": self.zParametersWidget.z_parameters,
            "overview_parameters": self.overviewParametersWidget.overview_parameters,
            "overview_autofocus_channel_name": self.overviewParametersWidget.get_autofocus_channel_name(),
        }

    def initUI(self):
        """Initialize the user interface for the FMAcquisitionWidget."""

        # Add objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.objectivePanel = TitledPanel(
            "Objective Control", content=self.objectiveControlWidget, collapsible=True
        )
        self.objectivePanel.expand()

        # create z parameters widget
        self.zParametersWidget = ZParametersWidget(
            z_parameters=z_parameters, parent=self
        )
        self.zParametersPanel = TitledPanel(
            "Z-Stack Parameters", content=self.zParametersWidget, collapsible=True
        )
        self.zParametersPanel.collapse()

        # create overview parameters widget
        self.overviewParametersWidget = OverviewParametersWidget(parent=self)
        self.overviewPanel = TitledPanel(
            "Overview Parameters",
            content=self.overviewParametersWidget,
            collapsible=True,
        )
        self.overviewPanel.collapse()

        # create positions widget
        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_remove_button(True)

        self.label_fm_pose = QLabel("No position selected")
        self.label_fm_pose.setWordWrap(True)
        self.label_fm_pose.setContentsMargins(4, 4, 4, 4)

        _positions_content = QWidget()
        _positions_layout = QVBoxLayout(_positions_content)
        _positions_layout.setContentsMargins(0, 0, 0, 0)
        _positions_layout.addWidget(self.lamella_list)
        _positions_layout.addWidget(self.label_fm_pose)

        self.positionsPanel = TitledPanel(
            "Positions", content=_positions_content, collapsible=True
        )
        self.positionsPanel.collapse()

        # create channel settings widget
        self.channelSettingsWidget = FluorescenceMultiChannelWidget(
            fm=self.fm, channel_settings=channel_settings, parent=self
        )
        self.channelPanel = TitledPanel(
            "Channel Settings", content=self.channelSettingsWidget, collapsible=True
        )
        self.channelPanel.expand()

        # Create autofocus widget
        self.autofocusWidget = AutofocusWidget(
            channel_settings=self.channelSettingsWidget.channel_settings, parent=self
        )
        self.autofocusPanel = TitledPanel(
            "Autofocus Settings", content=self.autofocusWidget, collapsible=True
        )
        self.autofocusPanel.collapse()

        # create histogram widget
        self.histogramWidget = HistogramWidget(parent=self)
        self.histogram_dock = self.viewer.window.add_dock_widget(
            self.histogramWidget, name="Image Histogram", area="left"
        )
        self.histogram_dock.setVisible(self.show_histogram)

        self.pushButton_toggle_acquisition = QPushButton("Start Acquisition", self)
        self.pushButton_acquire_single_image = QPushButton("Acquire Image", self)
        self.pushButton_acquire_zstack = QPushButton("Acquire Z-Stack", self)
        self.pushButton_acquire_overview = QPushButton("Acquire Overview", self)
        self.pushButton_run_autofocus = QPushButton("Run Auto-Focus", self)
        self.pushButton_cancel_acquisition = QPushButton("Cancel Acquisition", self)

        # Create progress bar (hidden by default)
        self.progressBar_current_acquisition = QProgressBar(self)
        self.progressBar_acquisition_task = QProgressBar(self)
        self.progressText = QLabel("Acquisition Progress", self)
        self.progressBar_acquisition_task.setStyleSheet(MILLING_PROGRESS_BAR_STYLESHEET)
        self.progressBar_current_acquisition.setStyleSheet(
            MILLING_PROGRESS_BAR_STYLESHEET
        )
        self.progressBar_current_acquisition.hide()
        self.progressBar_acquisition_task.hide()
        self.progressText.hide()

        # Define button configurations for data-driven state management
        self.button_configs = [
            (self.pushButton_toggle_acquisition, PRIMARY_BUTTON_STYLESHEET),
            (self.pushButton_acquire_single_image, SECONDARY_BUTTON_STYLESHEET),
            (self.pushButton_acquire_zstack, SECONDARY_BUTTON_STYLESHEET),
            (self.pushButton_acquire_overview, SECONDARY_BUTTON_STYLESHEET),
            (self.pushButton_run_autofocus, SECONDARY_BUTTON_STYLESHEET),
        ]

        layout = QVBoxLayout()
        layout.addWidget(self.objectivePanel)
        layout.addWidget(self.channelPanel)
        layout.addWidget(self.autofocusPanel)
        layout.addWidget(self.zParametersPanel)
        layout.addWidget(self.positionsPanel)
        layout.addWidget(self.overviewPanel)
        layout.setContentsMargins(0, 0, 0, 0)

        # create grid layout for buttons
        button_layout = QGridLayout()
        button_layout.addWidget(self.pushButton_toggle_acquisition, 0, 0, 1, 2)
        button_layout.addWidget(self.pushButton_run_autofocus, 1, 0, 1, 2)
        button_layout.addWidget(self.pushButton_acquire_single_image, 2, 0)
        button_layout.addWidget(self.pushButton_acquire_zstack, 2, 1)
        button_layout.addWidget(self.pushButton_acquire_overview, 3, 0, 1, 2)
        button_layout.addWidget(self.pushButton_cancel_acquisition, 4, 0, 1, 2)
        button_layout.addWidget(self.progressText, 5, 0, 1, 2)
        button_layout.addWidget(self.progressBar_current_acquisition, 6, 0, 1, 2)
        button_layout.addWidget(self.progressBar_acquisition_task, 7, 0, 1, 2)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area wrapping the panels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_widget = QWidget()
        content_widget.setLayout(layout)
        layout.addStretch()
        scroll_area.setWidget(content_widget)

        # Side panel: scroll area + button grid
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(scroll_area)
        side_layout.addLayout(button_layout)
        side_panel.setMinimumWidth(450)

        # Splitter: viewer (left) | controls (right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.viewer.window._qt_window)
        splitter.addWidget(side_panel)
        splitter.setSizes([900, 450])

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # connect signals
        self.overviewParametersWidget.spinBox_rows.valueChanged.connect(
            self._update_overview_bounding_box
        )
        self.overviewParametersWidget.spinBox_cols.valueChanged.connect(
            self._update_overview_bounding_box
        )
        self.overviewParametersWidget.doubleSpinBox_overlap.valueChanged.connect(
            self._update_overview_bounding_box
        )

        self.zParametersWidget.doubleSpinBox_zmin.valueChanged.connect(
            self.overviewParametersWidget._update_zstack_planes_visibility
        )
        self.zParametersWidget.doubleSpinBox_zmax.valueChanged.connect(
            self.overviewParametersWidget._update_zstack_planes_visibility
        )
        self.zParametersWidget.doubleSpinBox_zstep.valueChanged.connect(
            self.overviewParametersWidget._update_zstack_planes_visibility
        )

        self.pushButton_toggle_acquisition.clicked.connect(self.toggle_acquisition)
        self.pushButton_acquire_single_image.clicked.connect(self.acquire_image)
        self.pushButton_acquire_zstack.clicked.connect(self.acquire_image)
        self.pushButton_acquire_overview.clicked.connect(self.acquire_overview)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(self.update_image)
        # self.update_image_signal.connect(self.update_image)
        self.update_persistent_image_signal.connect(self.update_persistent_image)
        self.acquisition_finished_signal.connect(self._on_acquisition_finished)
        self.fm.acquisition_progress_signal.connect(self._on_acquisition_progress)
        self.microscope.stage_position_changed.connect(
            self._update_stage_position_display
        )

        # live parameter updates from channel list
        self.channelSettingsWidget.channel_field_changed.connect(
            self._on_channel_field_changed
        )

        # lamella list signals
        self.lamella_list.lamella_selected.connect(self._on_lamella_selected)
        self.lamella_list.move_to_requested.connect(self._on_move_to_fm_position)
        self.lamella_list.remove_requested.connect(self._on_remove_fm_position)

        # keyboard shortcuts
        self.f6_shortcut = QShortcut(QKeySequence("F6"), self)
        self.f6_shortcut.activated.connect(self.toggle_acquisition)

        self.f7_shortcut = QShortcut(QKeySequence("F7"), self)
        self.f7_shortcut.activated.connect(self.acquire_image)

        self.f8_shortcut = QShortcut(QKeySequence("F8"), self)
        self.f8_shortcut.activated.connect(self.run_autofocus)

        # movement controls
        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)

        # draw scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

        # Connect layer selection to histogram updates
        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_changed
        )

        # stylesheets
        self.pushButton_toggle_acquisition.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_single_image.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_zstack.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_overview.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autofocus.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_cancel_acquisition.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_cancel_acquisition.hide()  # Hide by default, show when acquisition starts
        self.pushButton_toggle_acquisition.setEnabled(True)

        # Explicitly enable acquisition buttons initially (disabled only during live acquisition)
        self.pushButton_acquire_single_image.setEnabled(True)
        self.pushButton_acquire_zstack.setEnabled(True)
        self.pushButton_acquire_overview.setEnabled(True)
        self.pushButton_run_autofocus.setEnabled(True)

        # Initialize positions widget state
        if self.experiment is not None:
            self.lamella_list.set_lamella(self.experiment.positions)
            if self.parent_widget is not None:
                self.experiment.events.connect(self._on_experiment_positions_changed)
        self.overviewParametersWidget._update_channel_names_from_parent()
        self.autofocusWidget.update_channels(
            self.channelSettingsWidget.channel_settings
        )

        # add file menu
        self.menubar = QMenuBar(self)
        load_image_action = QAction("Load Image...", self)
        load_image_action.triggered.connect(self.show_load_image_dialog)
        display_options_action = QAction("Display Options...", self)
        display_options_action.triggered.connect(self.show_display_options_dialog)
        save_fm_config_action = QAction("Save Configuration...", self)
        save_fm_config_action.triggered.connect(self.save_fm_configuration)
        load_fm_config_action = QAction("Load Configuration...", self)
        load_fm_config_action.triggered.connect(self.load_fm_configuration)

        self.file_menu = self.menubar.addMenu("File")
        if self.file_menu is not None:
            self.file_menu.addAction(load_image_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction(save_fm_config_action)
            self.file_menu.addAction(load_fm_config_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction(display_options_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction("Exit", self.close)
        main_layout.setMenuBar(self.menubar)

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

    def on_mouse_click(self, viewer, event):
        """Handle mouse click events in the napari viewer."""

        # only left mouse button
        if event.button != 1:
            return

        # Check for modifier keys - Alt to add new position, Shift to update existing
        if "Alt" not in event.modifiers and "Shift" not in event.modifiers:
            return

        # Prevent stage movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            event.handled = True
            return

        if self.experiment is None:
            logging.warning("No experiment loaded. Cannot save or update positions.")
            return

        current_orientation = self.microscope.get_stage_orientation()
        # get event position in world coordinates, convert to stage coordinates
        position_clicked = event.position[-2:]  # yx required
        inverted = self.microscope.get_stage_orientation() == "FM"
        stage_position = napari_world_coordinate_to_stage_position(
            Point(x=position_clicked[1], y=position_clicked[0]), inverted=inverted
        )
        current_stage_position = self.microscope.get_stage_position()
        stage_position.r = current_stage_position.r
        stage_position.t = current_stage_position.t
        stage_position.z = current_stage_position.z  # keep current z,r,t
        logging.info(
            f"Mouse clicked at {event.position}. Stage position: {stage_position}"
        )

        # TODO: QUERY: Does the position need to be flipped when in SEM orientation?

        if "Alt" in event.modifiers:
            # Add new position
            if (
                current_orientation in ["FM", "SEM"]
                and self.fm.objective.state == "Inserted"
            ):
                objective_position = self.fm.objective.position
            elif self.fm.objective.state == "Retracted":
                objective_position = self.fm.objective.focus_position

            # Create Lamella with automatic name generation
            if self.parent_widget is None:
                logging.error("Parent widget is None, cannot add new lamella.")
                return
            lamella = self.parent_widget.add_new_lamella(
                stage_position=stage_position, objective_position=objective_position
            )
            logging.info(
                f"New stage position saved: {lamella.name} at {stage_position}"
            )

        elif "Shift" in event.modifiers:
            # Update existing position
            current_index = self.lamella_list.selected_index

            if current_index < 0 or current_index >= len(self.experiment.positions):
                logging.warning(
                    "No position selected to update. Please select a position first."
                )
                event.handled = True
                return

            current_position = self.experiment.positions[current_index]

            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Update Position",
                f"Update position '{current_position.name}' to new coordinates?\n\n"
                f"Current: X={current_position.stage_position.x * 1e6:.1f} μm, "
                f"Y={current_position.stage_position.y * 1e6:.1f} μm\n"
                f"New: X={stage_position.x * 1e6:.1f} μm, "
                f"Y={stage_position.y * 1e6:.1f} μm",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.No:
                logging.info(f"Position update cancelled for '{current_position.name}'")
                event.handled = True  # Prevent further processing
                return

            # Update only the stage position, keep name and objective position
            self.experiment.positions[
                current_index
            ].fluorescence_pose.stage_position = stage_position
            logging.info(
                f"Updated position '{current_position.name}' to new stage coordinates: {stage_position}"
            )

        # Update positions widget
        self.lamella_list.set_lamella(self.experiment.positions)
        self.draw_stage_position_crosshairs()
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

        position_clicked = event.position[-2:]  # yx required
        inverted = self.microscope.get_stage_orientation() == "FM"
        stage_position = napari_world_coordinate_to_stage_position(
            Point(x=position_clicked[1], y=position_clicked[0]), inverted=inverted
        )  # yx required

        # threaded stage movement
        self.move_to_stage_position(stage_position)

        # self.microscope.move_stage_absolute(stage_position) # TODO: support absolute stable-move
        # stage_position = self.microscope.get_stage_position()

        event.handled = True

        return

    def move_to_stage_position(
        self,
        stage_position: FibsemStagePosition,
        objective_position: Optional[float] = None,
    ):
        logging.info(f"Starting stage movement to {stage_position}")
        self._current_acquisition_type = "stage_movement"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        self._acquisition_thread = threading.Thread(
            target=self._stage_movement_worker,
            args=(stage_position, objective_position),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _stage_movement_worker(
        self,
        stage_position: FibsemStagePosition,
        objective_position: Optional[float] = None,
    ):
        """Worker thread for stage movement."""
        try:
            logging.info("-" * 40)
            self.microscope.move_stage_absolute(stage_position)
            stage_position = self.microscope.get_stage_position()
            logging.info(f"Stage position after move: {stage_position}")
            if objective_position is not None:
                self.fm.objective.move_absolute(objective_position)
                logging.info(f"Objective position after move: {objective_position}")
            logging.info("-" * 40)

        except Exception as e:
            logging.error(f"Error during stage movement: {e}")

        finally:
            self.acquisition_finished_signal.emit()

    def _on_experiment_positions_changed(self, event=None):
        if self.parent_widget is not None:
            try:
                self.lamella_list.set_lamella(self.experiment.positions)
            except Exception as e:
                logging.error(f"Error updating positions: {e}")

    def _on_lamella_selected(self, lamella):
        if lamella is None or lamella.fluorescence_pose is None:
            self.label_fm_pose.setText("No fluorescence pose saved")
        else:
            pose = lamella.fluorescence_pose
            stage = pose.stage_position
            obj = pose.objective_position
            obj_str = f"{obj * 1e3:.3f} mm" if obj is not None else "N/A"
            self.label_fm_pose.setText(f"Stage: {stage.pretty}\nObjective: {obj_str}")
        self.draw_stage_position_crosshairs()

    def _on_move_to_fm_position(self, lamella):
        if lamella is None or lamella.fluorescence_pose is None:
            return
        pose = lamella.fluorescence_pose
        reply = QMessageBox.question(
            self,
            "Move to Position",
            f"Move to '{lamella.name}'?\n\n{pose.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return
        self.move_to_stage_position(pose.stage_position, pose.objective_position)

    def _on_remove_fm_position(self, lamella):
        if lamella is None or self.experiment is None:
            return
        try:
            idx = self.experiment.positions.index(lamella)
        except ValueError:
            return
        del self.experiment.positions[idx]
        self.experiment.save()
        self.lamella_list.set_lamella(self.experiment.positions)
        self.label_fm_pose.setText("No position selected")
        self.draw_stage_position_crosshairs()
        logging.info(f"Deleted position: {lamella.name}")

    def display_stage_position_overlay(self):
        """Legacy method for compatibility - redirects to update_text_overlay."""
        update_text_overlay(self.viewer, self.microscope)
        self.draw_stage_position_crosshairs()
        self.toggle_widgets()

    @ensure_main_thread
    def _update_stage_position_display(self):
        """Update the display of the stage position overlay."""
        try:
            update_text_overlay(self.viewer, self.microscope)
            self.draw_stage_position_crosshairs()
            self.toggle_widgets()
        except Exception as e:
            logging.error(
                f"Error updating stage position display: {e}. Unsubscribing from updates."
            )
            self.microscope.stage_position_changed.disconnect(
                self._update_stage_position_display
            )

    def show_load_image_dialog(self):
        """Show the load image dialog and display the loaded image."""
        # Determine starting directory - use experiment path if available
        start_dir = None
        if (
            self.experiment
            and hasattr(self.experiment, "path")
            and self.experiment.path
        ):
            start_dir = str(self.experiment.path)

        dialog = LoadImageDialog(self, start_directory=start_dir)

        # Connect the dialog's signal to update_persistent_image
        dialog.image_loaded_signal.connect(self.update_persistent_image)
        ret = dialog.exec_()
        if ret == QDialog.Accepted:
            logging.debug("Image loaded successfully")
        else:
            logging.debug("Image loading canceled")

    def show_display_options_dialog(self):
        """Show the display options dialog and apply changes."""
        dialog = DisplayOptionsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Get the new display options
            options = dialog.get_display_options()

            # Apply the options
            self.show_current_fov = options["show_current_fov"]
            self.show_overview_fov = options["show_overview_fov"]
            self.show_saved_positions_fov = options["show_saved_positions_fov"]
            self.show_stage_limits = options["show_stage_limits"]
            self.show_circle_overlays = options["show_circle_overlays"]
            self.show_histogram = options["show_histogram"]

            # Apply histogram visibility
            self._update_histogram_visibility()

            # Refresh the display
            self.display_stage_position_overlay()

            logging.info("Display options updated successfully")

    def save_fm_configuration(self):
        """Save current FM configuration to a YAML file."""
        try:
            # Get filename from user
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save FM Configuration",
                "fm_configuration.yaml",
                "YAML files (*.yaml *.yml);;All files (*.*)",
            )

            if not filename:
                return

            # Gather current settings from UI
            settings = self._get_current_settings()

            # Create autofocus settings if autofocus is enabled
            autofocus_settings = None
            if settings["overview_parameters"].autofocus_mode != AutoFocusMode.NONE:
                autofocus_settings = AutoFocusSettings(
                    channel_name=settings["overview_autofocus_channel_name"]
                )
                # If autofocus widget exists, get its settings
                # if hasattr(self, 'autofocusWidget'):
                # autofocus_settings = self.autofocusWidget.get_autofocus_settings()

            # Create FM configuration
            fm_config = FluorescenceConfiguration(
                channel_settings=settings["channel_settings"],
                z_parameters=settings["z_parameters"],
                overview_parameters=settings["overview_parameters"],
                autofocus_settings=autofocus_settings,
                focus_position=self.fm.objective.focus_position,
            )

            # Export configuration
            fm_config.export(filename)

            QMessageBox.information(
                self, "Configuration Saved", f"FM configuration saved to:\n{filename}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"Failed to save FM configuration:\n{str(e)}",
            )
            logging.error(f"Error saving FM configuration: {e}")

    def load_fm_configuration(self):
        """Load FM configuration from a YAML file."""
        try:
            # Get filename from user
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load FM Configuration",
                fcfg.CONFIG_PATH,
                "YAML files (*.yaml *.yml);;All files (*.*)",
            )

            if not filename:
                return

            # Load configuration
            config = FluorescenceConfiguration.load(filename)

            # Apply settings to UI widgets
            self.channelSettingsWidget.channel_settings = config.channel_settings
            self.zParametersWidget.z_parameters = config.z_parameters

            # Update overview parameters
            self.overviewParametersWidget.overview_parameters = (
                config.overview_parameters
            )

            if config.focus_position is not None:
                self.objectiveControlWidget._set_focus_position(config.focus_position)

            # Apply autofocus settings if available
            # if config.autofocus_settings and hasattr(self.overviewParametersWidget, 'autofocusWidget'):
            # self.overviewParametersWidget.autofocusWidget.set_autofocus_settings(config.autofocus_settings)

            QMessageBox.information(
                self,
                "Configuration Loaded",
                f"FM configuration loaded from:\n{filename}\n\n"
                f"Channels: {len(config.channel_settings)}\n"
                f"Grid: {config.overview_parameters.rows}x{config.overview_parameters.cols}\n"
                f"Autofocus: {config.overview_parameters.autofocus_mode.value}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"Failed to load FM configuration:\n{str(e)}",
            )
            logging.error(f"Error loading FM configuration: {e}")

    def _update_histogram_visibility(self):
        """Update the visibility of the histogram dock widget."""
        try:
            if hasattr(self, "histogram_dock"):
                self.histogram_dock.setVisible(self.show_histogram)
        except Exception as e:
            logging.warning(f"Could not update histogram visibility: {e}")

    def draw_stage_position_crosshairs(self):
        """Draw multiple crosshairs showing various stage positions on a single layer.

        Creates crosshair overlays in the napari viewer showing:
        - Origin crosshair (red) at stage coordinates (0,0)
        - Current stage position crosshair (yellow)
        - Saved stage positions crosshairs (cyan) from self.experiment.positions list

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
            - Handles multiple positions dynamically based on self.experiment.positions list
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
                **OVERLAY_CONFIG["text_properties"],
            }

            if layer_name in self.viewer.layers:
                # Update existing layer
                layer: NapariShapesLayer = self.viewer.layers[layer_name]
                layer.data = []  # clear to reset shape type
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

    def _collect_all_overlays(
        self, layer_scale: Tuple[float, float]
    ) -> List[NapariShapeOverlay]:
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
        inverted = current_orientation == "FM"
        center_point = stage_position_to_napari_world_coordinate(
            current_pos, inverted=inverted
        )

        if self.show_current_fov:
            fov_rect = None
            # if current_orientation == "FM":
            fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
            if fov_rect is not None:
                overlays.append(
                    NapariShapeOverlay(
                        shape=fov_rect,
                        color="magenta",
                        label=f"Current FOV ({current_orientation})",
                        shape_type="rectangle",
                    )
                )

        # Add overview acquisition area (orange, only if not acquiring)
        if (
            self.show_overview_fov
            and self._current_acquisition_type != "overview"
            and self._current_acquisition_type != "positions"
            and current_orientation in ["FM", "SEM"]
        ):
            # Get overview parameters and calculate total area
            settings = self._get_current_settings()
            overview_params = settings["overview_parameters"]

            total_width, total_height = calculate_grid_coverage_area(
                ncols=overview_params.cols,
                nrows=overview_params.rows,
                fov_x=fov_x,
                fov_y=fov_y,
                overlap=overview_params.overlap,
            )

            overview_rect = create_rectangle_shape(
                center_point, total_width, total_height, layer_scale
            )
            overlays.append(
                NapariShapeOverlay(
                    shape=overview_rect,
                    color="orange",
                    label=f"Overview {overview_params.rows}x{overview_params.cols}",
                    shape_type="rectangle",
                )
            )

        # Add 1mm bounding box around origin (yellow)
        if self.show_stage_limits:
            origin_point = Point(x=0, y=0)
            stage_limits = self.microscope._stage.limits
            xmin, xmax = stage_limits["x"].min, stage_limits["x"].max
            ymin, ymax = stage_limits["y"].min, stage_limits["y"].max
            box_width = xmax - xmin
            box_height = ymax - ymin
            origin_size = (box_width, box_height)
            origin_rect = create_rectangle_shape(
                origin_point, origin_size[0], origin_size[1], layer_scale
            )
            overlays.append(
                NapariShapeOverlay(
                    shape=origin_rect,
                    color="yellow",
                    label="Stage Limits",
                    shape_type="rectangle",
                )
            )

        # Add saved positions FOVs
        if self.show_saved_positions_fov and self.experiment is not None:
            selected_index = self.lamella_list.selected_index

            for i, saved_pos in enumerate(self.experiment.positions):
                center_point = stage_position_to_napari_world_coordinate(
                    saved_pos.fluorescence_pose.stage_position, inverted=inverted
                )
                fov_rect = create_rectangle_shape(
                    center_point, fov_x, fov_y, layer_scale
                )

                # Use lime for selected position, cyan for others
                color = "lime" if i == selected_index else "cyan"
                overlays.append(
                    NapariShapeOverlay(
                        shape=fov_rect,
                        color=color,
                        label=saved_pos.name,
                        shape_type="rectangle",
                    )
                )

        # Add circle overlays
        if self.show_circle_overlays:
            # grid boundary circle (red)
            origin_point = Point(x=0, y=0)
            origin_radius = 1000e-6  # 1000μm in meters
            origin_circle = create_circle_shape(
                origin_point, origin_radius, layer_scale
            )
            overlays.append(
                NapariShapeOverlay(
                    shape=origin_circle,
                    color="red",
                    label="Grid Boundary",
                    shape_type="ellipse",
                )
            )

        # You can add more circles here by following the same pattern
        # Example: Add circles at saved positions
        # for i, saved_pos in enumerate(self.experiment.positions):
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
        if (
            self._current_acquisition_type == "overview"
            or self._current_acquisition_type == "positions"
        ):
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
        text_properties = {"string": labels, **CROSSHAIR_CONFIG["text_properties"]}

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

    def _collect_crosshair_overlays(
        self, layer_scale: Tuple[float, float]
    ) -> List[NapariShapeOverlay]:
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
            overlays.append(
                NapariShapeOverlay(
                    shape=line,
                    color=CROSSHAIR_CONFIG["colors"]["origin"],
                    label=txt,
                    shape_type="line",
                )
            )

        # Add current stage position if available
        current_pos = self.microscope.get_stage_position()
        current_ori = self.microscope.get_stage_orientation()
        inverted = current_ori == "FM"
        current_point = stage_position_to_napari_world_coordinate(
            current_pos, inverted=inverted
        )
        current_lines = create_crosshair_shape(
            current_point, crosshair_size, layer_scale
        )
        for line, txt in zip(current_lines, ["stage-position", ""]):
            overlays.append(
                NapariShapeOverlay(
                    shape=line,
                    color=CROSSHAIR_CONFIG["colors"]["current"],
                    label=txt,
                    shape_type="line",
                )
            )

        # Add saved positions
        if self.experiment:
            selected_index = self.lamella_list.selected_index
            for i, saved_pos in enumerate(self.experiment.positions):
                saved_point = stage_position_to_napari_world_coordinate(
                    saved_pos.stage_position, inverted=inverted
                )
                saved_lines = create_crosshair_shape(
                    saved_point, crosshair_size, layer_scale
                )

                # Use lime for selected position, cyan for others
                color = (
                    CROSSHAIR_CONFIG["colors"]["saved_selected"]
                    if i == selected_index
                    else CROSSHAIR_CONFIG["colors"]["saved_unselected"]
                )

                # Show position name on crosshair if saved position FOV is disabled
                label = saved_pos.name if not self.show_saved_positions_fov else ""

                for line, txt in zip(saved_lines, [label, ""]):
                    overlays.append(
                        NapariShapeOverlay(
                            shape=line, color=color, label=txt, shape_type="line"
                        )
                    )

        return overlays

    def _on_acquisition_finished(self):
        """Handle consolidated acquisition completion in the main thread."""

        # clear acquisition state
        self._current_acquisition_type = None
        self._last_remaining_time = None

        # hide progress bar when acquisition finishes
        self.progressBar_acquisition_task.hide()
        self.progressBar_current_acquisition.hide()
        self.progressText.setText("")

        # refresh overview bbox, objective widget, and button states
        self._update_overview_bounding_box()
        self.objectiveControlWidget.update_objective_position_labels()

        self._update_acquisition_button_states()

    @ensure_main_thread
    def _on_acquisition_progress(self, progress: dict):
        logging.info(f"Acquisition progress: {progress}")

        try:
            self.progressBar_acquisition_task.value()
        except RuntimeError:
            logging.warning("Progress bar widget not available for updates.")
            self.fm.acquisition_progress_signal.disconnect(
                self._on_acquisition_progress
            )
            return

        # Show progress bar when acquisition progress is updated
        if self._current_acquisition_type in ["positions", "overview"]:
            # only show acquisition_task progress bar when acquiring overview/positions
            if not self.progressBar_acquisition_task.isVisible():
                self.progressBar_acquisition_task.show()
        if not self.progressBar_current_acquisition.isVisible():
            self.progressBar_current_acquisition.show()
        if not self.progressText.isVisible():
            self.progressText.show()

        progress_zlevels = progress.get("zlevel", None)
        progress_total_zlevels = progress.get("total_zlevels", None)
        progress_current = progress.get("current", None)
        progress_total = progress.get("total", None)
        channel_name = progress.get("channel", None)
        progress_state = progress.get("state", None)

        if progress_state == "moving":
            self.progressText.setText("Moving stage...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        if progress_state == "autofocus":
            self.progressText.setText("Running Autofocus...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        # set progress message
        if channel_name is not None:
            channel_index = progress.get("channel_index", 1)
            total_channels = progress.get("total_channels", 1)
            msg = f"Acquiring {channel_name} ({channel_index}/{total_channels})..."
            self.progressText.setText(msg)

        # set individual image acquisition progress bar
        if progress_zlevels and progress_total_zlevels:
            percentage_zlevel = (
                int((progress_zlevels / progress_total_zlevels) * 100)
                if progress_total_zlevels > 0
                else 0
            )
            self.progressBar_current_acquisition.setValue(percentage_zlevel)
            self.progressBar_current_acquisition.setFormat(
                f"Z-level {progress_zlevels}/{progress_total_zlevels}"
            )

        # set total acquisition task progress
        if progress_current is not None and progress_total is not None:
            # Format time remaining string if available
            time_remaining_str = ""
            remaining_time = progress.get("estimated_remaining_time", 0)
            if remaining_time > 0:
                self._last_remaining_time = remaining_time
            if self._last_remaining_time is not None and self._last_remaining_time > 0:
                time_remaining_str = (
                    f"Remaining Time: {format_duration(self._last_remaining_time)}"
                )

            # Set progress bar/text
            percentage = (
                int((progress_current / progress_total) * 100)
                if progress_total > 0
                else 0
            )
            msg = f"Position {progress_current}/{progress_total} - {time_remaining_str}"
            self.progressBar_acquisition_task.setValue(percentage)
            self.progressBar_acquisition_task.setFormat(msg)

    @ensure_main_thread
    @pyqtSlot(FluorescenceImage)
    def update_image(self, image: FluorescenceImage):
        """Update the napari image layer with the given image data and metadata. (Live images)"""

        try:
            # convert metadata to napari image layer format
            image_height, image_width = image.data.shape[-2:]
            metadata_dict = _image_metadata_to_napari_image_layer(
                image.metadata, (image_height, image_width)
            )
            channel_name = metadata_dict["name"]
            logging.info(
                f"Updating image layer: {channel_name}, shape: {image.data.shape}, acquisition date: {image.metadata.acquisition_date}"
            )

            self._update_napari_image_layer(channel_name, image.data, metadata_dict)
            # self.display_stage_position_overlay()
        except Exception as e:
            logging.error(f"Error updating image layer: {e}")
            self.fm.acquisition_signal.disconnect(self.update_image)

    @ensure_main_thread
    @pyqtSlot(FluorescenceImage)
    def update_persistent_image(self, image: Union[FibsemImage, FluorescenceImage]):
        """Update or create a napari image layer with the given image data and metadata. (Persistent images)"""

        if isinstance(image, FibsemImage):
            self._update_fibsem_image(image)
            return

        # Convert structured metadata to dictionary for napari compatibility
        image_height, image_width = image.data.shape[-2:]
        for channel_index in range(image.data.shape[0]):
            metadata_dict = _image_metadata_to_napari_image_layer(
                image.metadata, (image_height, image_width), channel_index=channel_index
            )
            layer_name = f"{metadata_dict['description']}-{metadata_dict['name']}"
            self._update_napari_image_layer(
                layer_name, image.data[channel_index], metadata_dict
            )

    def _update_napari_image_layer(
        self, layer_name: str, image: np.ndarray, metadata_dict: dict
    ):
        """Update or create a napari image layer with the given metadata.
        Args:
            layer_name: Name of the napari image layer
            image: FluorescenceImage object containing the image data and metadata
            metadata_dict: Dictionary containing metadata for the image layer
        Returns:
            None
        """
        # make sure all images are 3D for napari reasons (required to transform)
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        # Add a singleton dimension for z if needed
        if image.ndim == 3 and len(metadata_dict["scale"]) == 2:
            metadata_dict["scale"] = (1, *metadata_dict["scale"])

        # print(image.shape, metadata_dict["scale"], metadata_dict["translate"])

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

    def _update_fibsem_image(self, image: FibsemImage):
        """Update the napari image layer with the given FibsemImage data and metadata."""
        if image.metadata is None:
            return
        metadata_dict = _fibsem_image_metadata_to_napari_image_layer(
            image.metadata, image.data.shape
        )
        data = median_filter(image.data, 3)

        if self.microscope.get_stage_orientation() == "FM":
            data = np.flip(data, axis=0)
            tf = metadata_dict["translate"]
            metadata_dict["translate"] = (tf[0] - 260e-6, tf[1])

        self._update_napari_image_layer(metadata_dict["name"], data, metadata_dict)

    def start_acquisition(self):
        """Start the fluorescence acquisition."""

        if self.microscope.get_stage_orientation() not in ["FM", "SEM"]:
            logging.warning(
                "Stage is not in FM or SEM orientation. Cannot start acquisition."
            )
            return

        if self.is_acquiring:
            logging.warning("Another acquisition is already in progress.")
            return

        # Get current settings
        settings = self._get_current_settings()
        selected_channel_settings = settings["selected_channel_settings"]

        if selected_channel_settings is None:
            logging.warning("No channel selected for live acquisition")
            return

        logging.info(
            f"Starting acquisition with channel settings: {selected_channel_settings}"
        )
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
        if self.is_acquiring:
            logging.info(f"Cancelling {self._current_acquisition_type} acquisition")
            self._acquisition_stop_event.set()
            self._acquisition_thread.join(timeout=5)  # type: ignore[union-attr]

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

        if self.is_acquiring:
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info("Starting image acquisition")
        self._current_acquisition_type = "image"
        self._last_remaining_time = None
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        settings = self._get_current_settings()

        z_parameters = None
        channel_settings = settings["selected_channel_settings"]
        if self.sender() is self.pushButton_acquire_zstack:
            channel_settings = settings["channel_settings"]
            z_parameters = settings["z_parameters"]

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._image_acquistion_worker,
            args=(channel_settings, z_parameters),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _image_acquistion_worker(
        self, channel_settings: List[ChannelSettings], z_parameters: ZParameters
    ):
        """Worker thread for single image acquisition."""
        try:
            if self.experiment is None:
                logging.error("No experiment set for image acquisition")
                return

            # generate filename for saving
            name = "z-stack" if z_parameters is not None else "image"
            timestamp = utils.current_timestamp_v3(timeonly=True)
            filename = f"{name}-{timestamp}.ome.tiff"
            filepath = os.path.join(self.experiment.path, filename)

            image = acquire_image(
                microscope=self.fm,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                filename=filepath,
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
        if self.microscope.get_stage_orientation() not in ["FM", "SEM"]:
            # If not in FM or SEM orientation, disable all acquisition buttons
            self.pushButton_toggle_acquisition.setEnabled(False)
            self.pushButton_acquire_zstack.setEnabled(False)
            self.pushButton_acquire_overview.setEnabled(False)
            self.pushButton_cancel_acquisition.setEnabled(False)
            return

        # Check if any acquisition is active (live or specific acquisitions)
        any_acquisition_active = self.is_acquiring or self.is_acquisition_active

        # Special case buttons with unique behavior
        self.pushButton_cancel_acquisition.setVisible(self.is_acquisition_active)

        # Update toggle acquisition button text based on state
        if self.fm.is_acquiring:
            self.pushButton_toggle_acquisition.setText("Stop Acquisition")
        else:
            self.pushButton_toggle_acquisition.setText("Start Acquisition")

        # Update standard buttons using configuration from initUI (excluding toggle button)
        for button, _ in self.button_configs:
            if button == self.pushButton_toggle_acquisition:
                continue
            button.setEnabled(not any_acquisition_active)

        # Disable control widgets during acquisition
        self.objectiveControlWidget.setEnabled(not self.is_acquisition_active)
        self.zParametersWidget.setEnabled(not any_acquisition_active)
        self.overviewParametersWidget.setEnabled(not any_acquisition_active)
        self.lamella_list.setEnabled(not any_acquisition_active)

        # Disable channel settings during specific acquisitions (but allow during live imaging)
        self.channelSettingsWidget.setEnabled(not self.is_acquisition_active)
        self.objectiveControlWidget.pushButton_insert_objective.setEnabled(
            not any_acquisition_active
        )
        self.objectiveControlWidget.pushButton_move_to_focus.setEnabled(
            not any_acquisition_active
        )
        self.objectiveControlWidget.pushButton_retract_objective.setEnabled(
            not any_acquisition_active
        )

        # Disable channel list selection during any acquisition to prevent switching channels mid-acquisition
        # self.channelSettingsWidget.channel_list.setEnabled(not any_acquisition_active)

    def acquire_overview(self):
        """Start threaded overview acquisition using the current channel settings."""
        if self.fm.is_acquiring:
            logging.warning(
                "Cannot acquire overview while live acquisition is running. Stop acquisition first."
            )
            return

        if self.is_acquiring:
            logging.warning("Another acquisition is already in progress.")
            return

        # show overview confirmation dialog
        settings = self._get_current_settings()
        confirmation_dialog = OverviewConfirmationDialog(
            settings=settings, fm=self.fm, parent=self
        )

        # Only proceed if user confirms
        if confirmation_dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled by user")
            return

        logging.info("Starting overview acquisition")

        self._current_acquisition_type = "overview"
        self._last_remaining_time = None
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        channel_settings = settings["channel_settings"]
        overview_parameters = settings["overview_parameters"]
        z_parameters = (
            settings["z_parameters"] if overview_parameters.use_zstack else None
        )

        # Create autofocus settings if autofocus is enabled
        autofocus_settings = None
        if overview_parameters.autofocus_mode != AutoFocusMode.NONE:
            autofocus_settings = AutoFocusSettings(
                channel_name=settings["overview_autofocus_channel_name"]
            )
        # TODO: integrate auto-focus settings into overview parameters

        # TODO: support positions and autofocus z parameters
        # TODO: allow setting different sized overviews for each position
        positions = None  # self.experiment.positions

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._overview_worker,
            args=(
                channel_settings,
                overview_parameters,
                z_parameters,
                autofocus_settings,
                positions,
            ),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _overview_worker(
        self,
        channel_settings: List[ChannelSettings],
        overview_parameters: OverviewParameters,
        z_parameters: Optional[ZParameters],
        autofocus_settings: Optional[AutoFocusSettings],
        positions: Optional[List[FMStagePosition]] = None,
    ):
        """Worker thread for overview acquisition."""
        try:
            if positions is None:
                positions = []

            if self.experiment is None:
                logging.error("No experiment set for overview acquisition")
                return

            # if positions:
            #     logging.info(f"Acquiring overview at {len(positions)} saved positions")
            #     from fibsem.fm.acquisition import acquire_multiple_overviews

            #     overview_images = acquire_multiple_overviews(
            #         microscope=self.microscope,
            #         positions=self.experiment.positions,
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
                overview_parameters=overview_parameters,
                zparams=z_parameters,
                autofocus_settings=autofocus_settings,
                save_directory=str(self.experiment.path),
                stop_event=self._acquisition_stop_event,
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

    def _on_channel_field_changed(self, channel, field: str, value) -> None:
        """Update a single microscope parameter during live acquisition."""
        if not self.fm.is_acquiring:
            return
        if channel is not self.channelSettingsWidget.selected_channel:
            return
        if field == "excitation_wavelength":
            self.fm.filter_set.excitation_wavelength = value
        elif field == "emission_wavelength":
            self.fm.filter_set.emission_wavelength = value
        elif field == "exposure_time":
            self.fm.set_exposure_time(value)  # seconds
        elif field == "gain":
            self.fm.set_gain(value)  # fraction
        elif field == "power":
            self.fm.set_power(value)  # fraction
        elif field == "color":
            self.fm.set_channel_color(value)

    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info("Closing FMAcquisitionWidget, stopping acquisition if running.")

        # Stop live acquisition
        self.fm.acquisition_signal.disconnect(self.update_image)
        self.fm.acquisition_progress_signal.disconnect(self._on_acquisition_progress)
        self.microscope.stage_position_changed.disconnect(
            self._update_stage_position_display
        )
        if self.experiment is not None:
            self.experiment.events.disconnect(self._on_experiment_positions_changed)
        if self.fm.is_acquiring:
            try:
                self.stop_acquisition()
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                logging.warning("Acquisition stopped due to widget close.")

        # Stop acquisition worker thread if it is running
        if self.is_acquiring:
            try:
                self._acquisition_stop_event.set()
                self._acquisition_thread.join(timeout=5)  # type: ignore[union-attr]
                logging.info(
                    f"{self._current_acquisition_type} acquisition stopped due to widget close."
                )
            except Exception as e:
                logging.error(
                    f"Error stopping {self._current_acquisition_type} acquisition: {e}"
                )

        # Reset acquisition flags
        self._current_acquisition_type = None
        self._last_remaining_time = None

        self.histogramWidget.close()
        self.viewer.close()
        event.accept()

    def run_autofocus(self):
        """Start threaded auto-focus using the current channel settings and Z parameters."""
        if self.is_acquiring:
            logging.warning(
                "Cannot run auto-focus while another acquisition is in progress. Stop acquisition first."
            )
            return

        logging.info("Starting auto-focus")
        self._current_acquisition_type = "autofocus"
        self._last_remaining_time = None
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings["selected_channel_settings"]
        z_parameters = settings["z_parameters"]

        # Start auto-focus thread
        self._acquisition_thread = threading.Thread(
            target=self._autofocus_worker,
            args=(channel_settings, z_parameters),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _autofocus_worker(
        self, channel_settings: ChannelSettings, z_parameters: ZParameters
    ):
        """Worker thread for auto-focus."""
        try:
            logging.info("Running auto-focus with laplacian method")

            best_z = run_autofocus(
                microscope=self.fm,
                channel_settings=channel_settings,
                # z_parameters=z_parameters,
                method="laplacian",
                stop_event=self._acquisition_stop_event,
            )

            if best_z is None or self._acquisition_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return

            logging.info(
                f"Auto-focus completed successfully. Best focus: {best_z * 1e6:.1f} μm"
            )

        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")
        finally:
            self.acquisition_finished_signal.emit()


def open_fm_acquisition_dialog(
    microscope: "FibsemMicroscope",
    experiment=None,
    parent=None,
) -> QDialog:
    """Open FMAcquisitionWidget in a non-modal QDialog window.

    Returns the dialog — caller must keep a reference to prevent GC.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("FM Acquisition")
    dlg.setWindowFlags(dlg.windowFlags() | Qt.Window)
    dlg.resize(1400, 800)

    widget = FMAcquisitionWidget(
        microscope=microscope, experiment=experiment, parent=parent, parent_dialog=dlg
    )
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)

    dlg.show()
    return dlg


def create_widget() -> FMAcquisitionWidget:
    # CONFIG_PATH = r"C:\Users\User\Documents\github\openfibsem\fibsem-os\fibsem\config\tfs-arctis-configuration.yaml"
    CONFIG_PATH = "/Users/patrickcleeve/Documents/fibsem/fibsem/fibsem/config/sim-arctis-configuration.yaml"
    if not os.path.exists(CONFIG_PATH):
        CONFIG_PATH = None
    microscope, settings = utils.setup_session(config_path=CONFIG_PATH)

    if microscope.fm is None:
        logging.error(
            "FluorescenceMicroscope is not initialized. Cannot create FMAcquisitionWidget."
        )
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")

    # ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True

    return FMAcquisitionWidget(microscope=microscope, experiment=None, parent=None)


def main():
    widget = create_widget()
    widget.show()
    napari.run()
    return


if __name__ == "__main__":
    main()
