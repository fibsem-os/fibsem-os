import logging
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import napari
import numpy as np
from napari.layers import Image as NapariImageLayer
from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions, utils
from fibsem.config import LOG_PATH
from fibsem.constants import METRE_TO_MILLIMETRE
from fibsem.fm.acquisition import (
    AutofocusMode,
    acquire_and_stitch_tileset,
    acquire_at_positions,
    acquire_z_stack,
    calculate_grid_coverage_area,
)
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, FMStagePosition, ZParameters
from fibsem.structures import BeamType, FibsemStagePosition, Point
from fibsem.ui.FibsemMovementWidget import to_pretty_string_short
from fibsem.ui.fm.widgets import (
    ChannelSettingsWidget,
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

# napari coordinate is y down, x right, based at top left
# need to offset ssp, by half image size, and invert y
def to_napari_pos(image_shape, pos: FibsemStagePosition, pixelsize: float) -> Point:
    """Convert a sample-stage coordinate to a napari image coordinate"""
    pos2 = Point(
        x = pos.x - image_shape[1] * pixelsize / 2,
                y = -pos.y - image_shape[0] * pixelsize / 2)
    return pos2

def from_napari_pos(image: np.ndarray, pos: Point, pixelsize: float) -> FibsemStagePosition:
    """Convert a napari image coordinate to a sample-stage coordinate"""
    p = FibsemStagePosition(
        x = pos.x + image.shape[1] * pixelsize / 2,
        y = -pos.y - image.shape[0] * pixelsize / 2)

    return p

MAX_OBJECTIVE_STEP_SIZE = 0.05  # mm

class FMAcquisitionWidget(QWidget):
    update_image_signal = pyqtSignal(FluorescenceImage)
    update_persistent_image_signal = pyqtSignal(FluorescenceImage)
    zstack_finished_signal = pyqtSignal()
    overview_finished_signal = pyqtSignal()
    positions_acquisition_finished_signal = pyqtSignal()
    autofocus_finished_signal = pyqtSignal()

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.channel_name: str
        self.fm = fm
        self.viewer = viewer
        self.image_layer: Optional[NapariImageLayer] = None  # Placeholder for the image layer
        self.stage_positions: List[FMStagePosition] = []  # List to store stage positions
        
        # Create experiment path with current directory + datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = os.path.join(LOG_PATH, f"fibsem_experiment_{timestamp}")
        
        # Create the experiment directory
        try:
            os.makedirs(self.experiment_path, exist_ok=True)
            logging.info(f"Created experiment directory: {self.experiment_path}")
        except Exception as e:
            logging.error(f"Failed to create experiment directory: {e}")
            # Fallback to current directory
            self.experiment_path = os.getcwd()

        # Z-stack acquisition threading
        self._zstack_thread: Optional[threading.Thread] = None
        self._zstack_stop_event = threading.Event()
        self._is_zstack_acquiring = False
        
        # Overview acquisition threading
        self._overview_thread: Optional[threading.Thread] = None
        self._overview_stop_event = threading.Event()
        self._is_overview_acquiring = False
        
        # Positions acquisition threading
        self._positions_thread: Optional[threading.Thread] = None
        self._positions_stop_event = threading.Event()
        self._is_positions_acquiring = False
        
        # Auto-focus threading
        self._autofocus_thread: Optional[threading.Thread] = None
        self._autofocus_stop_event = threading.Event()
        self._is_autofocus_running = False

        self.initUI()
        self.draw_stage_position_crosshairs()
        self.display_stage_position_overlay()

    @property
    def is_acquisition_active(self) -> bool:
        """Check if any acquisition or operation is currently running.
        
        Returns:
            True if any acquisition (overview, z-stack, positions) or autofocus is active
        """
        return (self._is_overview_acquiring or 
                self._is_zstack_acquiring or 
                self._is_positions_acquiring or 
                self._is_autofocus_running)

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("FM Acquisition Widget", self)

        # Add objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.objectiveControlWidget.setContentsMargins(0, 0, 0, 0)

        # create z parameters widget
        z_parameters = ZParameters(
            zmin=-5e-6,     # -5 µm
            zmax=5e-6,      # +5 µm
            zstep=1e-6      # 1 µm step
        )
        self.zParametersWidget = ZParametersWidget(z_parameters=z_parameters, parent=self)
        
        # create overview parameters widget
        self.overviewParametersWidget = OverviewParametersWidget(parent=self)
        
        # Connect overview parameter changes to bounding box updates
        self.overviewParametersWidget.spinBox_rows.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.spinBox_cols.valueChanged.connect(self._update_overview_bounding_box)
        self.overviewParametersWidget.doubleSpinBox_overlap.valueChanged.connect(self._update_overview_bounding_box)
        
        # create saved positions widget
        self.savedPositionsWidget = SavedPositionsWidget(parent=self)

        # create channel settings widget
        channel_settings=ChannelSettings(
                name="Channel-01",
                excitation_wavelength=450,      # Example wavelength in nm
                emission_wavelength=None,       # Example wavelength in nm
                power=0.1,                      # Example power in W
                exposure_time=0.25,             # Example exposure time in seconds
        )
        self.channelSettingsWidget = ChannelSettingsWidget(
            fm=self.fm,
            channel_settings=channel_settings,
            parent=self
        )
    
        self.pushButton_start_acquisition = QPushButton("Start Acquisition", self)
        self.pushButton_stop_acquisition = QPushButton("Stop Acquisition", self)
        self.pushButton_acquire_zstack = QPushButton("Acquire Z-Stack", self)
        self.pushButton_acquire_overview = QPushButton("Acquire Overview", self)
        self.pushButton_acquire_at_positions = QPushButton("Acquire at Saved Positions (0)", self)
        self.pushButton_run_autofocus = QPushButton("Run Auto-Focus", self)

        layout.addWidget(self.label)
        layout.addWidget(self.objectiveControlWidget)
        layout.addWidget(self.zParametersWidget)
        layout.addWidget(self.overviewParametersWidget)
        layout.addWidget(self.savedPositionsWidget)
        layout.addWidget(self.channelSettingsWidget)

        # create grid layout for buttons
        button_layout = QGridLayout()
        button_layout.addWidget(self.pushButton_start_acquisition, 0, 0)
        button_layout.addWidget(self.pushButton_stop_acquisition, 0, 1)
        button_layout.addWidget(self.pushButton_acquire_zstack, 1, 0, 1, 2)  # Span 2 columns
        button_layout.addWidget(self.pushButton_acquire_overview, 2, 0, 1, 2)  # Span 2 columns
        button_layout.addWidget(self.pushButton_acquire_at_positions, 3, 0, 1, 2)  # Span 2 columns
        button_layout.addWidget(self.pushButton_run_autofocus, 4, 0, 1, 2)  # Span 2 columns
        button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the button layout
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # connect signals
        self.channelSettingsWidget.exposure_time_input.valueChanged.connect(self._update_exposure_time)
        self.channelSettingsWidget.power_input.valueChanged.connect(self._update_power)
        self.channelSettingsWidget.excitation_wavelength_input.currentIndexChanged.connect(self._update_excitation_wavelength)
        self.channelSettingsWidget.emission_wavelength_input.currentIndexChanged.connect(self._update_emission_wavelength)
        self.pushButton_start_acquisition.clicked.connect(self.start_acquisition)
        self.pushButton_stop_acquisition.clicked.connect(self.stop_acquisition)
        self.pushButton_acquire_zstack.clicked.connect(self.acquire_zstack)
        self.pushButton_acquire_overview.clicked.connect(self.acquire_overview)
        self.pushButton_acquire_at_positions.clicked.connect(self.acquire_at_positions)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(lambda image: self.update_image_signal.emit(image)) 
        self.update_image_signal.connect(self.update_image)
        self.update_persistent_image_signal.connect(self.update_persistent_image)
        self.zstack_finished_signal.connect(self._on_zstack_finished)
        self.overview_finished_signal.connect(self._on_overview_finished)
        self.positions_acquisition_finished_signal.connect(self._on_positions_finished)
        self.autofocus_finished_signal.connect(self._on_autofocus_finished)
        
        # Setup keyboard shortcuts
        self.f6_shortcut = QShortcut(QKeySequence("F6"), self)
        self.f6_shortcut.activated.connect(self.toggle_acquisition)
        
        self.f7_shortcut = QShortcut(QKeySequence("F7"), self)
        self.f7_shortcut.activated.connect(self.run_autofocus)

        # movement controls
        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)
        self.viewer.mouse_wheel_callbacks.append(self.on_mouse_wheel)
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)

        # stylesheets
        self.pushButton_start_acquisition.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.pushButton_stop_acquisition.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_acquire_zstack.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_acquire_overview.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_acquire_at_positions.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_run_autofocus.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_start_acquisition.setEnabled(True)
        self.pushButton_stop_acquisition.setEnabled(False)
        
        # Initialize positions button state
        self._update_positions_button()

        # draw scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

    def on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events in the napari viewer."""
        # no image layer available yet
        if self.image_layer is None:
            return
        
        # Prevent objective movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Objective movement disabled during acquisition")
            event.handled = True
            return
        
        # NOTE: scroll wheel events don't seem connected until there is an image layer?

        # Check for Ctrl key to control objective position
        if 'Shift' in event.modifiers:
            # TODO: add shift key to change step size?

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

        # only left clicks
        if event.button != 1:  # Left mouse button
            return
        
        # shift key pressed
        if 'Alt' not in event.modifiers:
            return
        
        # get event position in world coordinates, convert to stage coordinates
        position_clicked = event.position[-2:]  # yx required
        stage_position = FibsemStagePosition(x=position_clicked[1], y=-position_clicked[0])  # yx required
        logging.info(f"Mouse clicked at {event.position} in viewer {viewer}")
        logging.info(f"Stage position clicked: {stage_position}")

        # add to saved positions

        # give it a petname
        import petname
        num = len(self.stage_positions) + 1
        name = f"{num:02d}-{petname.generate(2)}"
        
        # Get current objective position
        current_objective_position = self.fm.objective.position
        stage_position.name = name
        
        # Create FMStagePosition with stage position and objective position
        fm_stage_position = FMStagePosition(
            name=name,
            stage_position=stage_position,
            objective_position=current_objective_position
        )
        self.stage_positions.append(fm_stage_position)

        logging.info(f"Stage position saved: {stage_position}")
        # add crosshair at clicked position
        self.draw_stage_position_crosshairs()
        # Update positions button and widget
        self._update_positions_button()
        self.savedPositionsWidget.update_positions(self.stage_positions)

    def on_mouse_double_click(self, viewer, event):
        """Handle double-click events in the napari viewer."""

        # no image layer available yet
        # if self.image_layer is None:
            # return

        # only left clicks
        if event.button != 1:  # Left mouse button
            return

        # Prevent stage movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            event.handled = True
            return

        logging.info(f"Mouse double-clicked at {event.position} in viewer {viewer}")
        # coords = self.image_layer.world_to_data(event.position)  # PIXEL COORDINATES
        logging.info("-" * 40)
        position_clicked = event.position[-2:]  # yx required
        stage_position = FibsemStagePosition(x=position_clicked[1], y=-position_clicked[0])  # yx required

        self.fm.parent.move_stage_absolute(stage_position) # TODO: support absolute stable-move

        stage_position = self.fm.parent.get_stage_position()
        logging.info(f"Stage position after move: {stage_position}")

        self.display_stage_position_overlay()
        return
        self.image_layer = self.viewer.layers[self.channel_name]

        # NOTE: this doesn't matter, as long as the image layer is present
        # TODO: remove this limitation once tested. We should still put a max range?
        # if not is_position_inside_layer(event.position, self.image_layer):
        #     logging.warning("Click position is outside the image layer.")
        #     return

        # changed by binning...
        # resolution = self.fm.camera.resolution
        # pixelsize = self.fm.camera.pixel_size[0]
        resolution = self.image_layer.data.shape[-2:]
        pixelsize = self.image_layer.scale[-1] if len(self.image_layer.scale) > 0 else self.fm.camera.pixel_size[0]
        # NOTE: this doesn't work with overview images as they are composited of multiple images

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

        pixelsize = self.fm.camera.pixel_size[0]  # Assuming square pixels

        points = np.array([[0, 0]])
        text = {
            "string": [
                f"STAGE: {to_pretty_string_short(pos)} [{orientation}]"
                f"\nOBJECTIVE: {self.fm.objective.position*1e3:.3f} mm",
                ],
            "color": "white",
            "font_size": 50,
            "anchor": "lower_left",
            "translation": (20*pixelsize, 5*pixelsize),  # Adjust translation if needed
        }
        try:
            self.viewer.layers["microscope-info"].data = points
            self.viewer.layers["microscope-info"].text = text
        except KeyError:
            self.viewer.add_points(
                data=points,
                name="microscope-info",
                size=20,
                text=text,
                border_width=7,
                border_width_is_relative=False,
                border_color="transparent",
                face_color="transparent",
                # translate=self.image_layer.translate[-2:] if len(self.image_layer.translate) >= 2 else self.image_layer.translate,
            )
        self.draw_stage_position_crosshairs()

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
            - Uses create_crosshair_shape() utility function for coordinate conversion
        """
        try:
            CROSSHAIR_SIZE = 50
            LAYER_NAME = "stage-position"
            
            # Get coordinate conversion scale from camera pixel size
            layer_scale = (self.fm.camera.pixel_size[0], self.fm.camera.pixel_size[1])
            
            # Collect all positions to display
            positions_data = self._collect_stage_positions()
            
            # Generate crosshair line data for all positions
            crosshair_lines = []
            for position in positions_data["positions"]:
                lines = create_crosshair_shape(position, CROSSHAIR_SIZE, layer_scale)
                crosshair_lines.extend(lines)
            
            
            # Prepare text properties for labels
            text_properties = {
                "string": positions_data["labels"],
                "color": "white",
                "font_size": 50,
                "anchor": "lower_left",
                "translation": (5, 55),
            }
            
            # Update or create the napari layer
            self._update_crosshair_layer(
                layer_name=LAYER_NAME,
                crosshair_lines=crosshair_lines,
                colors=positions_data["colors"],
                text_properties=text_properties,
                layer_scale=layer_scale
            )
            
            # Draw all FOV bounding boxes on single layer
            self._draw_fov_boxes(layer_scale)
            
            # Draw all circle overlays on single layer
            self._draw_circle_overlays(layer_scale)

        except Exception as e:
            logging.warning(f"Error drawing stage position crosshairs: {e}")
    
    def _draw_fov_boxes(self, layer_scale: Tuple[float, float]):
        """Draw all FOV bounding boxes on a single layer.
        
        Creates rectangle overlays for:
        - Current position single image FOV (magenta)
        - Overview acquisition area (orange, only if not acquiring)
        - Saved positions FOV (cyan)
        
        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
        """
        LAYER_NAME = "fov-boxes"
        
        try:
            # Collect all FOV rectangles and their colors
            fov_data = self._collect_fov_rectangles(layer_scale)
            
            # Update or create the FOV layer
            if not fov_data["rectangles"]:
                # Hide layer if no rectangles to display
                if LAYER_NAME in self.viewer.layers:
                    self.viewer.layers[LAYER_NAME].visible = False
                return
            
            # Prepare text properties for labels
            text_properties = {
                "string": fov_data["labels"],
                "color": "white",
                "font_size": 50,
                "anchor": "upper_left",
                "translation": (5, 5),
            }
            
            if LAYER_NAME in self.viewer.layers:
                # Update existing layer
                layer = self.viewer.layers[LAYER_NAME]
                layer.data = fov_data["rectangles"]
                layer.edge_color = fov_data["colors"]
                layer.edge_width = 30
                layer.face_color = "transparent"
                layer.opacity = 0.7
                layer.visible = True
                # Try to update text properties
                try:
                    layer.text = text_properties
                except AttributeError:
                    logging.debug("Could not update text properties for FOV layer")
            else:
                # Create new layer
                self.viewer.add_shapes(
                    data=fov_data["rectangles"],
                    name=LAYER_NAME,
                    shape_type="rectangle",
                    edge_color=fov_data["colors"],
                    edge_width=30,
                    face_color="transparent",
                    scale=layer_scale,
                    opacity=0.7,
                    text=text_properties,
                )
                
        except Exception as e:
            logging.warning(f"Error drawing FOV boxes: {e}")
            if LAYER_NAME in self.viewer.layers:
                self.viewer.layers[LAYER_NAME].visible = False
    
    def _collect_fov_rectangles(self, layer_scale: Tuple[float, float]) -> Dict[str, List]:
        """Collect all FOV rectangles with their associated colors.
        
        Returns:
            Dictionary containing:
            - rectangles: List of rectangle arrays for FOV areas
            - colors: List of color strings for each rectangle
            - labels: List of text labels for each rectangle
        """
        rectangles = []
        colors = []
        labels = []
        
        # Get camera FOV once for all calculations
        fov_x, fov_y = self.fm.camera.field_of_view
        
        # Add current position single image FOV (magenta)
        if self.fm.parent:
            current_pos = self.fm.parent.get_stage_position()
            if current_pos:
                center_point = Point(x=current_pos.x, y=-current_pos.y)
                fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
                rectangles.append(fov_rect)
                colors.append("magenta")
                labels.append("Current FOV")
        
        # Add overview acquisition area (orange, only if not acquiring)
        if not self._is_overview_acquiring and not self._is_positions_acquiring and self.fm.parent and hasattr(self, 'overviewParametersWidget'):
            current_pos = self.fm.parent.get_stage_position()
            if current_pos:
                # Get overview parameters and calculate total area
                grid_size = self.overviewParametersWidget.get_grid_size()
                overlap = self.overviewParametersWidget.get_overlap()
                
                total_width, total_height = calculate_grid_coverage_area(
                    ncols=grid_size[1], nrows=grid_size[0],
                    fov_x=fov_x, fov_y=fov_y, overlap=overlap
                )
                
                center_point = Point(x=current_pos.x, y=-current_pos.y)
                overview_rect = create_rectangle_shape(center_point, total_width, total_height, layer_scale)
                rectangles.append(overview_rect)
                colors.append("orange")
                labels.append(f"Overview {grid_size[0]}×{grid_size[1]}")
        
        # Add 1mm bounding box around origin (yellow)
        origin_point = Point(x=0, y=0)
        origin_size = 0.8e-3  # 0.8mm in meters
        origin_rect = create_rectangle_shape(origin_point, origin_size, origin_size, layer_scale)
        rectangles.append(origin_rect)
        colors.append("yellow")
        labels.append("Stage Limits")
        
        # Add saved positions FOV (cyan/lime based on selection)
        # Get currently selected position index from the SavedPositionsWidget
        selected_index = -1
        if hasattr(self, 'savedPositionsWidget') and self.savedPositionsWidget.comboBox_positions.currentIndex() >= 0:
            selected_index = self.savedPositionsWidget.comboBox_positions.currentIndex()
        
        for i, saved_pos in enumerate(self.stage_positions):
            center_point = Point(x=saved_pos.stage_position.x, y=-saved_pos.stage_position.y)
            fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
            rectangles.append(fov_rect)
            
            # Use lime for selected position, cyan for others
            if i == selected_index:
                colors.append("lime")  # Lime for selected position
            else:
                colors.append("cyan")  # Cyan for other saved positions
            
            # Add label with position name
            pos_name = saved_pos.name
            labels.append(pos_name)
        
        return {
            "rectangles": rectangles,
            "colors": colors,
            "labels": labels
        }
    
    def _update_overview_bounding_box(self):
        """Update the FOV boxes when parameters change."""
        # Don't update bounding box during overview or position acquisition
        if self._is_overview_acquiring or self._is_positions_acquiring:
            return
            
        try:
            layer_scale = (self.fm.camera.pixel_size[0], self.fm.camera.pixel_size[1])
            self._draw_fov_boxes(layer_scale)
        except Exception as e:
            logging.warning(f"Error updating FOV boxes: {e}")
    
    def _draw_circle_overlays(self, layer_scale: Tuple[float, float]):
        """Draw circle overlays on a single layer.
        
        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
        """
        LAYER_NAME = "circle-overlays"
        
        try:
            # Collect all circle overlays and their colors
            circle_data = self._collect_circle_overlays(layer_scale)
            
            # Update or create the circle layer
            if not circle_data["circles"]:
                # Hide layer if no circles to display
                if LAYER_NAME in self.viewer.layers:
                    self.viewer.layers[LAYER_NAME].visible = False
                return
            
            # Prepare text properties for labels
            text_properties = {
                "string": circle_data["labels"],
                "color": "white",
                "font_size": 50,
                "anchor": "upper_left",
                "translation": (5, 5),
            }
            
            if LAYER_NAME in self.viewer.layers:
                # Update existing layer
                layer = self.viewer.layers[LAYER_NAME]
                layer.data = circle_data["circles"]
                layer.edge_color = circle_data["colors"]
                layer.edge_width = 40
                layer.face_color = "transparent"
                layer.opacity = 0.7
                layer.visible = True
                # Try to update text properties
                try:
                    layer.text = text_properties
                except AttributeError:
                    logging.debug("Could not update text properties for circle layer")
            else:
                # Create new layer
                self.viewer.add_shapes(
                    data=circle_data["circles"],
                    name=LAYER_NAME,
                    shape_type="polygon",
                    edge_color=circle_data["colors"],
                    edge_width=40,
                    face_color="transparent",
                    scale=layer_scale,
                    opacity=0.7,
                    text=text_properties,
                )
                
        except Exception as e:
            logging.warning(f"Error drawing circle overlays: {e}")
            if LAYER_NAME in self.viewer.layers:
                self.viewer.layers[LAYER_NAME].visible = False
    
    def _collect_circle_overlays(self, layer_scale: Tuple[float, float]) -> Dict[str, List]:
        """Collect all circle overlays with their associated colors and labels.
        
        Args:
            layer_scale: Tuple of (pixel_size_x, pixel_size_y) for coordinate conversion
            
        Returns:
            Dictionary containing:
            - circles: List of circle arrays
            - colors: List of color strings for each circle
            - labels: List of text labels for each circle
        """
        circles = []
        colors = []
        labels = []
        
        # Example: Add a circle at origin with 100μm radius
        origin_point = Point(x=0, y=0)
        origin_radius = 1000e-6  # 100μm in meters
        origin_circle = create_circle_shape(origin_point, origin_radius, layer_scale)
        circles.append(origin_circle)
        colors.append("red")
        labels.append("Origin Circle")
        
        # You can add more circles here by following the same pattern
        # Example: Add circles at saved positions
        # for i, saved_pos in enumerate(self.stage_positions):
        #     circle_point = Point(x=saved_pos.x, y=-saved_pos.y)
        #     circle_radius = 50e-6  # 50μm radius
        #     circle_shape = create_circle_shape(circle_point, circle_radius, layer_scale)
        #     circles.append(circle_shape)
        #     colors.append("blue")
        #     labels.append(f"Circle {i+1}")
        
        return {
            "circles": circles,
            "colors": colors,
            "labels": labels
        }
    
    def _collect_stage_positions(self) -> Dict[str, List]:
        """Collect all stage positions with their associated colors and labels.
        
        Returns:
            Dictionary containing:
            - positions: List of Point objects in stage coordinates
            - colors: List of color strings (2 per position for horizontal/vertical lines)
            - labels: List of label strings (2 per position, with empty string for spacing)
        """
        positions = []
        colors = []
        labels = []
        
        # Get currently selected position index from the SavedPositionsWidget
        selected_index = -1
        if hasattr(self, 'savedPositionsWidget') and self.savedPositionsWidget.comboBox_positions.currentIndex() >= 0:
            selected_index = self.savedPositionsWidget.comboBox_positions.currentIndex()
        
        # Add origin position (0,0)
        positions.append(Point(x=0, y=0))
        colors.extend(["red", "red"])  # Red for origin crosshair
        labels.extend(["origin", ""])  # Label with spacing
        
        # Add current stage position if available
        if self.fm.parent:
            try:
                current_pos = self.fm.parent.get_stage_position()
                if current_pos:
                    # Convert to napari coordinate convention (y inverted)
                    positions.append(Point(x=current_pos.x, y=-current_pos.y))
                    colors.extend(["yellow", "yellow"])  # Yellow for current position
                    labels.extend(["stage-position", ""])
            except Exception as e:
                logging.warning(f"Could not get current stage position: {e}")
        
        # Add saved positions from user clicks
        for i, saved_pos in enumerate(self.stage_positions):
            # Convert to napari coordinate convention (y inverted)
            positions.append(Point(x=saved_pos.stage_position.x, y=-saved_pos.stage_position.y))
            
            # Use lime for selected position, cyan for others
            if i == selected_index:
                colors.extend(["lime", "lime"])  # Lime for selected position
            else:
                colors.extend(["cyan", "cyan"])  # Cyan for other saved positions
                
            # labels.extend([saved_pos.name or "saved", ""])  # Use position name or default
            labels.extend(["", ""])
        return {
            "positions": positions,
            "colors": colors,
            "labels": labels
        }
    
    def _update_crosshair_layer(self, layer_name: str, crosshair_lines: List, 
                               colors: List[str], text_properties: Dict, 
                               layer_scale: Tuple[float, float]):
        """Update or create the crosshair layer in napari viewer.
        
        Args:
            layer_name: Name of the napari layer
            crosshair_lines: List of line arrays for crosshair shapes
            colors: List of color strings for each line
            text_properties: Dictionary of text styling properties
            layer_scale: Tuple of (x_scale, y_scale) for coordinate conversion
        """
        if layer_name in self.viewer.layers:
            # Update existing layer
            layer = self.viewer.layers[layer_name]
            layer.data = crosshair_lines
            # Note: edge_color and text updates may not work with all napari versions
            try:
                layer.edge_color = colors
                layer.edge_width = 12
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
                edge_width=12,
                face_color="transparent",
                scale=layer_scale,
                text=text_properties,
            )

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
        
        if self._positions_thread and self._positions_thread.is_alive():
            logging.warning("Positions acquisition is already in progress.")
            return
        
        logging.info(f"Starting acquisition at {len(self.stage_positions)} saved positions")
        self.pushButton_acquire_at_positions.setEnabled(False)
        self.pushButton_acquire_at_positions.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        
        # Set positions acquisition flag to prevent FOV updates
        self._is_positions_acquiring = True
        
        # Clear stop event
        self._positions_stop_event.clear()
        
        # Get current settings
        channel_settings = self.channelSettingsWidget.channel_settings
        z_parameters = self.zParametersWidget.z_parameters if hasattr(self, 'zParametersWidget') else None
        
        # Start acquisition thread
        self._positions_thread = threading.Thread(
            target=self._positions_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._positions_thread.start()
    
    def _positions_worker(self, channel_settings: ChannelSettings, z_parameters: Optional[ZParameters]):
        """Worker thread for positions acquisition."""
        try:
            logging.info(f"Acquiring at {len(self.stage_positions)} saved positions")
            
            # Get the parent microscope for stage movement
            if self.fm.parent is None:
                logging.error("FluorescenceMicroscope parent is None. Cannot acquire at positions.")
                return
            
            # Extract FibsemStagePosition objects from FMStagePosition objects
            stage_positions = [fm_pos.stage_position for fm_pos in self.stage_positions]
            
            # Acquire images at all saved positions
            images = acquire_at_positions(
                microscope=self.fm.parent,
                positions=stage_positions,
                channel_settings=channel_settings,
                zparams=z_parameters,
                use_autofocus=False,
                save_directory=self.experiment_path,
            )
            
            # Check if acquisition was cancelled
            if self._positions_stop_event.is_set():
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
            self.positions_acquisition_finished_signal.emit()
    
    def _on_positions_finished(self):
        """Handle positions acquisition completion in the main thread."""
        self.pushButton_acquire_at_positions.setEnabled(True)
        self.pushButton_acquire_at_positions.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self._is_positions_acquiring = False
        # Update button text in case positions were modified during acquisition
        self._update_positions_button()
        # Re-display overview FOV now that acquisition is complete
        self._update_overview_bounding_box()

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
        emission_wavelength = image.metadata.channels[0].emission_wavelength
        logging.info(f"Updating image layer with channel name: {channel_name}, wavelength: {wavelength} nm")

        stage_position = image.metadata.stage_position

        pos = to_napari_pos(image.data.shape[-2:], stage_position, image.metadata.pixel_size_x)

        if emission_wavelength is not None:
            colormap = wavelength_to_color(wavelength)
        else:
            colormap = "gray"

        if channel_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[channel_name].data = image.data
            self.viewer.layers[channel_name].metadata = metadata_dict
            self.viewer.layers[channel_name].colormap = colormap
            self.viewer.layers[channel_name].translate = (pos.y, pos.x)  # Translate to stage position
        else:
            # If the layer does not exist, create a new one
            self.image_layer = self.viewer.add_image(
                data=image.data,
                name=channel_name,
                metadata=metadata_dict,
                colormap=colormap,
                scale=(image.metadata.pixel_size_y, image.metadata.pixel_size_x),
                translate=(pos.y, pos.x),  # Translate to stage position,
                blending="additive",
            )
        
        self.channel_name = channel_name

        self.display_stage_position_overlay()

    @pyqtSlot(FluorescenceImage)
    def update_persistent_image(self, image: FluorescenceImage):
        
        acq_date = image.metadata.acquisition_date
       
        # Convert structured metadata to dictionary for napari compatibility
        metadata_dict = image.metadata.to_dict() if image.metadata else {}

        channel_name = image.metadata.channels[0].name
        wavelength = image.metadata.channels[0].excitation_wavelength
        emission_wavelength = image.metadata.channels[0].emission_wavelength

        stage_position = image.metadata.stage_position
        pos = to_napari_pos(image.data.shape[-2:], stage_position, image.metadata.pixel_size_x)

        layer_name = image.metadata.description
        if not layer_name:
            layer_name = f"{channel_name}-{acq_date}"

        scale = (image.metadata.pixel_size_y, image.metadata.pixel_size_x)  # yx order for napari
        if image.data.ndim == 3:
            scale = (1, *scale)  # Add a singleton dimension for time if needed

        if emission_wavelength is not None:
            colormap = wavelength_to_color(wavelength)
        else:
            colormap = "gray"

        if layer_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[layer_name].data = image.data
            self.viewer.layers[layer_name].metadata = metadata_dict
            self.viewer.layers[layer_name].colormap = colormap
            self.viewer.layers[layer_name].translate = (pos.y, pos.x)  # Translate to stage position
        else:
            # If the layer does not exist, create a new one
            self.viewer.add_image(
                data=image.data,
                name=layer_name,
                metadata=metadata_dict,
                colormap=colormap,
                scale=scale,
                translate=(pos.y, pos.x),  # Translate to stage position
                blending="additive",
            )

    def start_acquisition(self):
        """Start the fluorescence acquisition."""
        if self.fm.is_acquiring:
            logging.warning("Acquisition is already running.")
            return
        
        logging.info("Acquisition started")
        self.pushButton_start_acquisition.setEnabled(False)
        self.pushButton_start_acquisition.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_stop_acquisition.setEnabled(True)
        # TODO: handle case where acquisition fails...

        channel_settings = self.channelSettingsWidget.channel_settings
        logging.info(f"Starting acquisition with channel settings: {channel_settings}")

        self.fm.start_acquisition(channel_settings=channel_settings)

    def stop_acquisition(self):
        """Stop the fluorescence acquisition."""

        logging.info("Acquisition stopped")
        self.pushButton_stop_acquisition.setEnabled(False)
        self.pushButton_stop_acquisition.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_start_acquisition.setEnabled(True)
        self.pushButton_start_acquisition.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

        self.fm.stop_acquisition()

    def toggle_acquisition(self):
        """Toggle acquisition start/stop with F6 key."""
        if self.fm.is_acquiring:
            logging.info("F6 pressed: Stopping acquisition")
            self.stop_acquisition()
        else:
            logging.info("F6 pressed: Starting acquisition")
            self.start_acquisition()

    def acquire_zstack(self):
        """Start threaded Z-stack acquisition using the current Z parameters and channel settings."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire Z-stack while live acquisition is running. Stop acquisition first.")
            return
        
        if self._zstack_thread and self._zstack_thread.is_alive():
            logging.warning("Z-stack acquisition is already in progress.")
            return
        
        logging.info("Starting Z-stack acquisition")
        self.pushButton_acquire_zstack.setEnabled(False)
        self.pushButton_acquire_zstack.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        
        # Set z-stack acquisition flag
        self._is_zstack_acquiring = True
        
        # Clear stop event
        self._zstack_stop_event.clear()
        
        # Get current settings
        channel_settings = self.channelSettingsWidget.channel_settings
        z_parameters = self.zParametersWidget.z_parameters
        
        # Start acquisition thread
        self._zstack_thread = threading.Thread(
            target=self._zstack_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._zstack_thread.start()
    
    def _zstack_worker(self, channel_settings: ChannelSettings, z_parameters: ZParameters):
        """Worker thread for Z-stack acquisition."""
        try:
            logging.info(f"Acquiring Z-stack with {len(z_parameters.generate_positions(self.fm.objective.position))} planes")
            logging.info(f"Z parameters: {z_parameters.to_dict()}")
            
            # Acquire z-stack
            zstack_image = acquire_z_stack(
                microscope=self.fm,
                channel_settings=channel_settings,
                zparams=z_parameters
            )
            
            # Check if acquisition was cancelled
            if self._zstack_stop_event.is_set():
                logging.info("Z-stack acquisition was cancelled")
                return
            
            # Save z-stack to experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"z-stack-{timestamp}.ome.tiff"
            filepath = os.path.join(self.experiment_path, filename)
            zstack_image.metadata.description = filename.removesuffix(".ome.tiff")
            
            try:
                zstack_image.save(filepath)
                logging.info(f"Z-stack saved to: {filepath}")
            except Exception as e:
                logging.error(f"Failed to save Z-stack to {filepath}: {e}")
            
            # Emit the z-stack image
            self.update_persistent_image_signal.emit(zstack_image)
            
            logging.info("Z-stack acquisition completed successfully")
            
        except Exception as e:
            logging.error(f"Error during Z-stack acquisition: {e}")
            # TODO: Show error message to user
            
        finally:
            # Signal that Z-stack acquisition is finished (thread-safe)
            self.zstack_finished_signal.emit()
    
    def _on_zstack_finished(self):
        """Handle Z-stack acquisition completion in the main thread."""
        self.pushButton_acquire_zstack.setEnabled(True)
        self.pushButton_acquire_zstack.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        
        # Clear z-stack acquisition flag
        self._is_zstack_acquiring = False

    def acquire_overview(self):
        """Start threaded overview acquisition using the current channel settings."""
        if self.fm.is_acquiring:
            logging.warning("Cannot acquire overview while live acquisition is running. Stop acquisition first.")
            return
        
        if self._overview_thread and self._overview_thread.is_alive():
            logging.warning("Overview acquisition is already in progress.")
            return
        
        logging.info("Starting overview acquisition")
        self.pushButton_acquire_overview.setEnabled(False)
        self.pushButton_acquire_overview.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        
        # Set overview acquisition flag to prevent bounding box updates
        self._is_overview_acquiring = True
        
        # Clear stop event
        self._overview_stop_event.clear()
        
        # Get current settings
        channel_settings = self.channelSettingsWidget.channel_settings
        grid_size = self.overviewParametersWidget.get_grid_size()
        tile_overlap = self.overviewParametersWidget.get_overlap()
        use_zstack = self.overviewParametersWidget.get_use_zstack()
        z_parameters = self.zParametersWidget.z_parameters if use_zstack else None
        autofocus_mode = self.overviewParametersWidget.get_autofocus_mode()
        
        # Start acquisition thread
        self._overview_thread = threading.Thread(
            target=self._overview_worker,
            args=(channel_settings, grid_size, tile_overlap, z_parameters, autofocus_mode),
            daemon=True
        )
        self._overview_thread.start()
    
    def _overview_worker(self, channel_settings: ChannelSettings, grid_size: tuple[int, int], tile_overlap: float, z_parameters: Optional[ZParameters], autofocus_mode: AutofocusMode):
        """Worker thread for overview acquisition."""
        try:
            info_parts = [f"Acquiring overview with {grid_size[0]}x{grid_size[1]} grid, {tile_overlap:.1%} overlap"]
            if z_parameters:
                info_parts.append("with z-stacks")
            if autofocus_mode != AutofocusMode.NONE:
                info_parts.append(f"with auto-focus mode: {autofocus_mode.value}")
            logging.info(", ".join(info_parts))
            
            # Get the parent microscope for tileset acquisition
            if self.fm.parent is None:
                logging.error("FluorescenceMicroscope parent is None. Cannot acquire overview.")
                return
            
            # Create timestamp and subdirectory for tiles
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tiles_directory = os.path.join(self.experiment_path, f"overview-{timestamp}")
            os.makedirs(tiles_directory, exist_ok=True)
            
            # Acquire and stitch tileset
            overview_image = acquire_and_stitch_tileset(
                microscope=self.fm.parent,
                channel_settings=channel_settings,
                grid_size=grid_size,
                tile_overlap=tile_overlap,
                zparams=z_parameters,
                autofocus_mode=autofocus_mode,
                save_directory=tiles_directory
            )
            
            # Check if acquisition was cancelled
            if self._overview_stop_event.is_set():
                logging.info("Overview acquisition was cancelled")
                return
            
            # Save overview to experiment directory
            filename = f"overview-{timestamp}.ome.tiff"
            filepath = os.path.join(self.experiment_path, filename)
            overview_image.metadata.description = filename.removesuffix(".ome.tiff")
            
            try:
                overview_image.save(filepath)
                logging.info(f"Overview saved to: {filepath}")
            except Exception as e:
                logging.error(f"Failed to save overview to {filepath}: {e}")
            
            # Emit the overview image
            self.update_persistent_image_signal.emit(overview_image)
            
            logging.info("Overview acquisition completed successfully")
            
        except Exception as e:
            logging.error(f"Error during overview acquisition: {e}")
            # TODO: Show error message to user
            
        finally:
            # Signal that overview acquisition is finished (thread-safe)
            self.overview_finished_signal.emit()
    
    def _on_overview_finished(self):
        """Handle overview acquisition completion in the main thread."""
        self.pushButton_acquire_overview.setEnabled(True)
        self.pushButton_acquire_overview.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self._is_overview_acquiring = False
        # Re-display overview FOV now that acquisition is complete
        self._update_overview_bounding_box()

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
        
        # Stop Z-stack acquisition
        if self._zstack_thread and self._zstack_thread.is_alive():
            try:
                self._zstack_stop_event.set()
                self._zstack_thread.join(timeout=5)  # Wait up to 5 seconds
                logging.info("Z-stack acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping Z-stack acquisition: {e}")
        
        # Stop overview acquisition
        if self._overview_thread and self._overview_thread.is_alive():
            try:
                self._overview_stop_event.set()
                self._overview_thread.join(timeout=5)  # Wait up to 5 seconds
                logging.info("Overview acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping overview acquisition: {e}")
        
        # Stop positions acquisition
        if self._positions_thread and self._positions_thread.is_alive():
            try:
                self._positions_stop_event.set()
                self._positions_thread.join(timeout=5)  # Wait up to 5 seconds
                logging.info("Positions acquisition stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping positions acquisition: {e}")
        
        # Stop auto-focus
        if self._autofocus_thread and self._autofocus_thread.is_alive():
            try:
                self._autofocus_stop_event.set()
                self._autofocus_thread.join(timeout=5)  # Wait up to 5 seconds
                logging.info("Auto-focus stopped due to widget close.")
            except Exception as e:
                logging.error(f"Error stopping auto-focus: {e}")
        
        # Reset acquisition flags
        self._is_overview_acquiring = False
        self._is_positions_acquiring = False
        self._is_zstack_acquiring = False
        self._is_autofocus_running = False

        event.accept()

    def run_autofocus(self):
        """Start threaded auto-focus using the current channel settings and Z parameters."""
        if self.fm.is_acquiring:
            logging.warning("Cannot run autofocus while live acquisition is running. Stop acquisition first.")
            return
        
        if self._autofocus_thread and self._autofocus_thread.is_alive():
            logging.warning("Auto-focus is already in progress.")
            return
        
        logging.info("Starting auto-focus")
        self.pushButton_run_autofocus.setEnabled(False)
        self.pushButton_run_autofocus.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        
        # Set auto-focus running flag
        self._is_autofocus_running = True
        
        # Clear stop event
        self._autofocus_stop_event.clear()
        
        # Get current settings
        channel_settings = self.channelSettingsWidget.channel_settings
        z_parameters = self.zParametersWidget.z_parameters
        
        # Start auto-focus thread
        self._autofocus_thread = threading.Thread(
            target=self._autofocus_worker,
            args=(channel_settings, z_parameters),
            daemon=True
        )
        self._autofocus_thread.start()
    
    def _autofocus_worker(self, channel_settings: ChannelSettings, z_parameters: ZParameters):
        """Worker thread for auto-focus."""
        try:
            logging.info("Running auto-focus with laplacian method")
            
            from fibsem.fm.calibration import run_autofocus
            
            # Run autofocus using laplacian method (default)
            best_z = run_autofocus(
                microscope=self.fm,
                channel_settings=channel_settings,
                # z_parameters=z_parameters,
                method='laplacian'
            )
            
            # Check if auto-focus was cancelled
            if self._autofocus_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return
            
            logging.info(f"Auto-focus completed successfully. Best focus: {best_z*1e6:.1f} μm")
            
        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")
            
        finally:
            # Signal that auto-focus is finished (thread-safe)
            self.autofocus_finished_signal.emit()
    
    def _on_autofocus_finished(self):
        """Handle auto-focus completion in the main thread."""
        self.pushButton_run_autofocus.setEnabled(True)
        self.pushButton_run_autofocus.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)
        self._is_autofocus_running = False
        
        # Update objective position display
        self.objectiveControlWidget.update_objective_position_labels()

def main():

    microscope, settings = utils.setup_session()
    from fibsem.structures import BeamType
    # microscope.move_flat_to_beam(BeamType.ELECTRON)
    microscope.move_to_microscope("FM")
    
    viewer = napari.Viewer()
    widget = FMAcquisitionWidget(fm=microscope.fm, viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    return


if __name__ == "__main__":
    main()
    # main2()




