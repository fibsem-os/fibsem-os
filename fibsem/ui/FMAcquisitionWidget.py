import logging
import threading
from typing import Union, List, Dict, Optional, Tuple
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

from napari.layers import Image as NapariImageLayer
from fibsem import conversions, utils
from fibsem.constants import METRE_TO_MILLIMETRE, MILLIMETRE_TO_METRE
from fibsem.fm.acquisition import acquire_z_stack, acquire_and_stitch_tileset, calculate_grid_coverage_area, acquire_at_positions
from fibsem.fm.microscope import FluorescenceImage, FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, ZParameters
from fibsem.structures import BeamType, Point, FibsemStagePosition
from fibsem.ui.napari.utilities import (
    create_crosshair_shape,
    create_rectangle_shape,
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


OBJECTIVE_CONFIG = {
    "step_size": 0.001,  # mm
    "decimals": 3,  # number of decimal places
    "suffix": " mm",  # unit suffix
}
MAX_OBJECTIVE_STEP_SIZE = 0.05  # mm

Z_PARAMETERS_CONFIG = {
    "step_size": 0.1,  # µm
    "decimals": 1,  # number of decimal places
    "suffix": " µm",  # unit suffix
}

OVERVIEW_PARAMETERS_CONFIG = {
    "min_grid_size": 1,
    "max_grid_size": 10,
    "default_rows": 3,
    "default_cols": 3,
    "default_overlap": 0.1,
    "overlap_step": 0.01,
    "overlap_decimals": 2,
}


class ZParametersWidget(QWidget):
    def __init__(self, z_parameters: ZParameters, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.z_parameters = z_parameters
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        
        self.label_header = QLabel("Z-Stack Parameters", self)
        
        # Z minimum
        self.label_zmin = QLabel("Z Min", self)
        self.doubleSpinBox_zmin = QDoubleSpinBox(self)
        self.doubleSpinBox_zmin.setRange(-100.0, -0.5)  # ±100 µm range
        self.doubleSpinBox_zmin.setValue(self.z_parameters.zmin * 1e6)  # Convert m to µm
        self.doubleSpinBox_zmin.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zmin.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zmin.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zmin.setToolTip("Minimum Z position relative to current position")
        self.doubleSpinBox_zmin.setKeyboardTracking(False)
        
        # Z maximum
        self.label_zmax = QLabel("Z Max", self)
        self.doubleSpinBox_zmax = QDoubleSpinBox(self)
        self.doubleSpinBox_zmax.setRange(0.5, 100.0)  # ±100 µm range
        self.doubleSpinBox_zmax.setValue(self.z_parameters.zmax * 1e6)  # Convert m to µm
        self.doubleSpinBox_zmax.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zmax.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zmax.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zmax.setToolTip("Maximum Z position relative to current position")
        self.doubleSpinBox_zmax.setKeyboardTracking(False)
        
        # Z step
        self.label_zstep = QLabel("Z Step", self)
        self.doubleSpinBox_zstep = QDoubleSpinBox(self)
        self.doubleSpinBox_zstep.setRange(0.1, 10.0)  # 0.1 to 10 µm range
        self.doubleSpinBox_zstep.setValue(self.z_parameters.zstep * 1e6)  # Convert m to µm
        self.doubleSpinBox_zstep.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zstep.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zstep.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zstep.setToolTip("Step size between Z positions")
        self.doubleSpinBox_zstep.setKeyboardTracking(False)
        
        # Number of planes (calculated, read-only)
        self.label_num_planes = QLabel("Planes", self)
        self.label_num_planes_value = QLabel(self._calculate_num_planes(), self)
        self.label_num_planes_value.setStyleSheet("QLabel { color: #666666; }")
        
        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 2)
        layout.addWidget(self.label_zmin, 1, 0)
        layout.addWidget(self.doubleSpinBox_zmin, 1, 1)
        layout.addWidget(self.label_zmax, 2, 0)
        layout.addWidget(self.doubleSpinBox_zmax, 2, 1)
        layout.addWidget(self.label_zstep, 3, 0)
        layout.addWidget(self.doubleSpinBox_zstep, 3, 1)
        layout.addWidget(self.label_num_planes, 4, 0)
        layout.addWidget(self.label_num_planes_value, 4, 1)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.doubleSpinBox_zmin.valueChanged.connect(self._on_zmin_changed)
        self.doubleSpinBox_zmax.valueChanged.connect(self._on_zmax_changed)
        self.doubleSpinBox_zstep.valueChanged.connect(self._on_zstep_changed)

    def _calculate_num_planes(self) -> str:
        """Calculate the number of planes based on current parameters."""
        try:
            z_range = self.z_parameters.zmax - self.z_parameters.zmin
            if self.z_parameters.zstep <= 0:
                return "Invalid"
            num_planes = int(z_range / self.z_parameters.zstep) + 1
            return f"{num_planes}"
        except (ValueError, ZeroDivisionError):
            return "Invalid"

    def _update_num_planes_display(self):
        """Update the number of planes display."""
        self.label_num_planes_value.setText(self._calculate_num_planes())

    def _on_zmin_changed(self, value: float):
        """Handle Z min value change."""
        self.z_parameters.zmin = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()
        
        # Ensure zmin <= zmax
        if self.z_parameters.zmin > self.z_parameters.zmax:
            self.doubleSpinBox_zmax.setValue(value)

    def _on_zmax_changed(self, value: float):
        """Handle Z max value change."""
        self.z_parameters.zmax = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()
        
        # Ensure zmax >= zmin
        if self.z_parameters.zmax < self.z_parameters.zmin:
            self.doubleSpinBox_zmin.setValue(value)

    def _on_zstep_changed(self, value: float):
        """Handle Z step value change."""
        self.z_parameters.zstep = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()


class OverviewParametersWidget(QWidget):
    def __init__(self, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.parent_widget = parent
        
        # Initialize parameters
        self.rows = OVERVIEW_PARAMETERS_CONFIG["default_rows"]
        self.cols = OVERVIEW_PARAMETERS_CONFIG["default_cols"]
        self.overlap = OVERVIEW_PARAMETERS_CONFIG["default_overlap"]
        
        self.initUI()

    def initUI(self):
        
        self.label_header = QLabel("Overview Parameters", self)
        
        # Number of rows
        self.label_rows = QLabel("Rows", self)
        self.spinBox_rows = QSpinBox(self)
        self.spinBox_rows.setRange(OVERVIEW_PARAMETERS_CONFIG["min_grid_size"], 
                                   OVERVIEW_PARAMETERS_CONFIG["max_grid_size"])
        self.spinBox_rows.setValue(self.rows)
        self.spinBox_rows.setToolTip("Number of rows in the overview grid")
        
        # Number of columns
        self.label_cols = QLabel("Columns", self)
        self.spinBox_cols = QSpinBox(self)
        self.spinBox_cols.setRange(OVERVIEW_PARAMETERS_CONFIG["min_grid_size"], 
                                   OVERVIEW_PARAMETERS_CONFIG["max_grid_size"])
        self.spinBox_cols.setValue(self.cols)
        self.spinBox_cols.setToolTip("Number of columns in the overview grid")
        
        # Tile overlap
        self.label_overlap = QLabel("Overlap", self)
        self.doubleSpinBox_overlap = QDoubleSpinBox(self)
        self.doubleSpinBox_overlap.setRange(0.0, 0.9)
        self.doubleSpinBox_overlap.setValue(self.overlap)
        self.doubleSpinBox_overlap.setSingleStep(OVERVIEW_PARAMETERS_CONFIG["overlap_step"])
        self.doubleSpinBox_overlap.setDecimals(OVERVIEW_PARAMETERS_CONFIG["overlap_decimals"])
        self.doubleSpinBox_overlap.setToolTip("Fraction of overlap between adjacent tiles")
        self.doubleSpinBox_overlap.setKeyboardTracking(False)
        
        # Total area (calculated, read-only)
        self.label_total_area = QLabel("Total Area", self)
        self.label_total_area_value = QLabel(self._calculate_total_area(), self)
        self.label_total_area_value.setStyleSheet("QLabel { color: #666666; }")
        
        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 2)
        layout.addWidget(self.label_rows, 1, 0)
        layout.addWidget(self.spinBox_rows, 1, 1)
        layout.addWidget(self.label_cols, 2, 0)
        layout.addWidget(self.spinBox_cols, 2, 1)
        layout.addWidget(self.label_overlap, 3, 0)
        layout.addWidget(self.doubleSpinBox_overlap, 3, 1)
        layout.addWidget(self.label_total_area, 4, 0)
        layout.addWidget(self.label_total_area_value, 4, 1)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.spinBox_rows.valueChanged.connect(self._on_rows_changed)
        self.spinBox_cols.valueChanged.connect(self._on_cols_changed)
        self.doubleSpinBox_overlap.valueChanged.connect(self._on_overlap_changed)

    def _calculate_total_area(self) -> str:
        """Calculate the total area of the overview grid."""
        try:
            if self.parent_widget is None or self.parent_widget.fm is None:
                return "N/A"
            
            # Get FOV dimensions directly from camera
            fov_x, fov_y = self.parent_widget.fm.camera.field_of_view
            
            # Calculate total coverage area using helper function
            total_width, total_height = calculate_grid_coverage_area(
                ncols=self.cols,
                nrows=self.rows,
                fov_x=fov_x,
                fov_y=fov_y,
                overlap=self.overlap
            )
            
            # Convert to micrometers for display
            width_um = total_width * 1e6
            height_um = total_height * 1e6
            
            return f"{width_um:.1f} × {height_um:.1f} μm"
            
        except (ValueError, TypeError, AttributeError) as e:
            return f"Error: {str(e)}"

    def _update_total_area_display(self):
        """Update the total area display."""
        self.label_total_area_value.setText(self._calculate_total_area())

    def _on_rows_changed(self, value: int):
        """Handle rows value change."""
        self.rows = value
        self._update_total_area_display()

    def _on_cols_changed(self, value: int):
        """Handle columns value change."""
        self.cols = value
        self._update_total_area_display()

    def _on_overlap_changed(self, value: float):
        """Handle overlap value change."""
        self.overlap = value
        self._update_total_area_display()

    def get_grid_size(self) -> tuple[int, int]:
        """Get the current grid size as (rows, cols)."""
        return (self.rows, self.cols)

    def get_overlap(self) -> float:
        """Get the current overlap fraction."""
        return self.overlap


class SavedPositionsWidget(QWidget):
    position_deleted = pyqtSignal(int)  # Signal emitted when a position is deleted (index)
    position_selected = pyqtSignal(int)  # Signal emitted when a position is selected (index)
    
    def __init__(self, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        self.label_header = QLabel("Saved Positions", self)
        
        # Combobox for selecting saved positions
        self.label_positions = QLabel("Select Position", self)
        self.comboBox_positions = QComboBox(self)
        self.comboBox_positions.setToolTip("Select a saved position from the list")
        
        # Buttons for managing positions
        self.pushButton_goto_position = QPushButton("Go To", self)
        self.pushButton_goto_position.setToolTip("Move stage to the selected position")
        self.pushButton_delete_position = QPushButton("Delete", self)
        self.pushButton_delete_position.setToolTip("Delete the selected position")
        self.pushButton_clear_all = QPushButton("Clear All", self)
        self.pushButton_clear_all.setToolTip("Delete all saved positions")
        
        # Position info label
        self.label_position_info = QLabel("No positions saved", self)
        self.label_position_info.setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        self.label_position_info.setWordWrap(True)
        
        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 3)
        layout.addWidget(self.label_positions, 1, 0)
        layout.addWidget(self.comboBox_positions, 1, 1, 1, 2)
        layout.addWidget(self.pushButton_goto_position, 2, 0)
        layout.addWidget(self.pushButton_delete_position, 2, 1)
        layout.addWidget(self.pushButton_clear_all, 2, 2)
        layout.addWidget(self.label_position_info, 3, 0, 1, 3)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Connect signals
        self.comboBox_positions.currentIndexChanged.connect(self._on_position_selected)
        self.pushButton_goto_position.clicked.connect(self._goto_selected_position)
        self.pushButton_delete_position.clicked.connect(self._delete_selected_position)
        self.pushButton_clear_all.clicked.connect(self._clear_all_positions)
        
        # Set initial button states
        self._update_widget_state()
        
        # Set button styles
        self.pushButton_goto_position.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_delete_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_clear_all.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)

    def update_positions(self, positions: List[FibsemStagePosition]):
        """Update the combobox with current saved positions."""
        # Store current selection
        current_text = self.comboBox_positions.currentText()
        
        # Clear and repopulate combobox
        self.comboBox_positions.clear()
        
        if not positions:
            self.comboBox_positions.addItem("No positions saved")
            self.label_position_info.setText("No positions saved")
        else:
            for i, pos in enumerate(positions):
                display_text = f"{pos.name}" if pos.name else f"Position {i+1}"
                self.comboBox_positions.addItem(display_text)
            
            # Try to restore previous selection
            current_index = self.comboBox_positions.findText(current_text)
            if current_index >= 0:
                self.comboBox_positions.setCurrentIndex(current_index)
            
            # Update info for currently selected position
            self._update_position_info()
        
        self._update_widget_state()

    def _update_widget_state(self):
        """Update button enabled/disabled state based on available positions."""
        has_positions = self.parent_widget and len(self.parent_widget.stage_positions) > 0
        
        self.pushButton_goto_position.setEnabled(has_positions)
        self.pushButton_delete_position.setEnabled(has_positions)
        self.pushButton_clear_all.setEnabled(has_positions)
        self.comboBox_positions.setEnabled(has_positions)
        
        if not has_positions:
            self.pushButton_goto_position.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
            self.pushButton_delete_position.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
            self.pushButton_clear_all.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        else:
            self.pushButton_goto_position.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
            self.pushButton_delete_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
            self.pushButton_clear_all.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)

    def _update_position_info(self):
        """Update the position info label with details of the selected position."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            self.label_position_info.setText("No positions saved")
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            pos = self.parent_widget.stage_positions[current_index]
            info_text = f"X: {pos.x*1e6:.1f} μm, Y: {pos.y*1e6:.1f} μm"
            if hasattr(pos, 'z') and pos.z is not None:
                info_text += f", Z: {pos.z*1e6:.1f} μm"
            self.label_position_info.setText(info_text)
        else:
            self.label_position_info.setText("Invalid selection")

    def _on_position_selected(self, index: int):
        """Handle position selection in combobox."""
        self._update_position_info()
        if index >= 0:
            self.position_selected.emit(index)
            # Update crosshairs to highlight selected position
            if self.parent_widget:
                self.parent_widget.draw_stage_position_crosshairs()

    def _goto_selected_position(self):
        """Move stage to the selected position."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            position = self.parent_widget.stage_positions[current_index]
            
            if self.parent_widget.fm.parent:
                try:
                    logging.info(f"Moving to saved position: {position.name} at {position}")
                    self.parent_widget.fm.parent.move_stage_absolute(position)
                    self.parent_widget.display_stage_position_overlay()
                except Exception as e:
                    logging.error(f"Error moving to position: {e}")
            else:
                logging.error("No parent microscope available for stage movement")

    def _delete_selected_position(self):
        """Delete the selected position."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            position = self.parent_widget.stage_positions[current_index]
            
            # Confirmation dialog
            from fibsem.ui.utils import message_box_ui
            ret = message_box_ui(
                title="Delete Position",
                text=f"Are you sure you want to delete position '{position.name}'?",
                parent=self
            )
            if ret:
                # Remove from parent's list
                del self.parent_widget.stage_positions[current_index]
                
                # Update displays
                self.update_positions(self.parent_widget.stage_positions)
                self.parent_widget.draw_stage_position_crosshairs()
                self.parent_widget._update_positions_button()
                
                # Emit signal
                self.position_deleted.emit(current_index)
                
                logging.info(f"Deleted saved position: {position.name}")

    def _clear_all_positions(self):
        """Clear all saved positions."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return
        
        # Confirmation dialog
        from fibsem.ui.utils import message_box_ui
        ret = message_box_ui(
            title="Clear All Positions",
            text=f"Are you sure you want to delete all {len(self.parent_widget.stage_positions)} saved positions?",
            parent=self
        )
        if ret:
            # Clear parent's list
            self.parent_widget.stage_positions.clear()
            
            # Update displays
            self.update_positions(self.parent_widget.stage_positions)
            self.parent_widget.draw_stage_position_crosshairs()
            self.parent_widget._update_positions_button()
            
            logging.info("Cleared all saved positions")


class ObjectiveControlWidget(QWidget):    
    def __init__(self, fm: FluorescenceMicroscope, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        
        self.label_header = QLabel("Objective", self)
        self.pushButton_insert_objective = QPushButton("Insert Objective", self)
        self.pushButton_retract_objective = QPushButton("Retract Objective", self)

        # add double spin box for objective position
        self.label_objective_control = QLabel("Position", self)
        self.label_objective_step_size = QLabel("Step Size", self)
        self.doubleSpinBox_objective_position = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MILLIMETRE,
                                                        self.fm.objective.limits[1] * METRE_TO_MILLIMETRE)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MILLIMETRE)  # Convert m to mm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["suffix"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        self.doubleSpinBox_objective_step_size = QDoubleSpinBox(self)
        self.doubleSpinBox_objective_step_size.setRange(1.0, 50.0)      # Set a reasonable range for step size
        self.doubleSpinBox_objective_step_size.setSingleStep(0.1)       # Set step size for the spin box
        self.doubleSpinBox_objective_step_size.setValue(1.0)            # Default step size (1.0 µm)
        self.doubleSpinBox_objective_step_size.setSuffix(" µm")
        self.doubleSpinBox_objective_step_size.setToolTip("Step size for objective movement in microns")
        self.doubleSpinBox_objective_step_size.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # Add a label to display the current objective position
        self.label_objective_position = QLabel(f"Current Objective Position: {self.fm.objective.position*METRE_TO_MILLIMETRE:.2f} mm", self)

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_header, 0, 0, 1, 2)
        layout.addWidget(self.pushButton_insert_objective, 1, 0)
        layout.addWidget(self.pushButton_retract_objective, 1, 1)
        layout.addWidget(self.label_objective_control, 2, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 2, 1) 
        layout.addWidget(self.label_objective_step_size, 3, 0)
        layout.addWidget(self.doubleSpinBox_objective_step_size, 3, 1)
        layout.addWidget(self.label_objective_position, 4, 0, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value * 1e-3))  # Convert from um to mm

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
        
        if self.parent_widget is not None:
            self.parent_widget.display_stage_position_overlay()  # Update the stage position overlay in the parent widget

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
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout

        self.setLayout(layout)

        # add grid layout
        layout.addWidget(QLabel("Channel"), 0, 0)
        self.channel_name_input = QLineEdit(self.channel_settings.name, self)
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

        # Set column stretch factors to make widgets expand properly
        layout.setColumnStretch(0, 1)  # Labels column - expandable
        layout.setColumnStretch(1, 1)  # Input widgets column - expandable

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
    update_persistent_image_signal = pyqtSignal(FluorescenceImage)
    zstack_finished_signal = pyqtSignal()
    overview_finished_signal = pyqtSignal()
    positions_acquisition_finished_signal = pyqtSignal()

    def __init__(self, fm: FluorescenceMicroscope, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.channel_name: str
        self.fm = fm
        self.viewer = viewer
        self.image_layer: Optional[NapariImageLayer] = None  # Placeholder for the image layer
        self.stage_positions: List[FibsemStagePosition] = []  # List to store stage positions

        # Z-stack acquisition threading
        self._zstack_thread: Optional[threading.Thread] = None
        self._zstack_stop_event = threading.Event()
        
        # Overview acquisition threading
        self._overview_thread: Optional[threading.Thread] = None
        self._overview_stop_event = threading.Event()
        self._is_overview_acquiring = False
        
        # Positions acquisition threading
        self._positions_thread: Optional[threading.Thread] = None
        self._positions_stop_event = threading.Event()

        self.initUI()
        self.draw_stage_position_crosshairs()
        self.display_stage_position_overlay()

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

        # we need to re-emit the signal to ensure it is handled in the main thread
        self.fm.acquisition_signal.connect(lambda image: self.update_image_signal.emit(image)) 
        self.update_image_signal.connect(self.update_image)
        self.update_persistent_image_signal.connect(self.update_persistent_image)
        self.zstack_finished_signal.connect(self._on_zstack_finished)
        self.overview_finished_signal.connect(self._on_overview_finished)
        self.positions_acquisition_finished_signal.connect(self._on_positions_finished)

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
        
        # NOTE: scroll wheel events don't seem connected until there is an image layer?

        # Check for Ctrl key to control objective position
        if 'Control' in event.modifiers:
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
        stage_position.name = name
        self.stage_positions.append(stage_position)

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
                f"\nOBJECTIVE: {self.fm.objective.position*1e3:.2f} mm",
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
            
            if LAYER_NAME in self.viewer.layers:
                # Update existing layer
                layer = self.viewer.layers[LAYER_NAME]
                layer.data = fov_data["rectangles"]
                layer.edge_color = fov_data["colors"]
                layer.edge_width = 10
                layer.face_color = "transparent"
                layer.opacity = 0.7
                layer.visible = True
            else:
                # Create new layer
                self.viewer.add_shapes(
                    data=fov_data["rectangles"],
                    name=LAYER_NAME,
                    shape_type="rectangle",
                    edge_color=fov_data["colors"],
                    edge_width=10,
                    face_color="transparent",
                    scale=layer_scale,
                    opacity=0.7,
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
        """
        rectangles = []
        colors = []
        
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
        
        # Add overview acquisition area (orange, only if not acquiring)
        if not self._is_overview_acquiring and self.fm.parent and hasattr(self, 'overviewParametersWidget'):
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
        
        # Add saved positions FOV (cyan)
        for saved_pos in self.stage_positions:
            center_point = Point(x=saved_pos.x, y=-saved_pos.y)
            fov_rect = create_rectangle_shape(center_point, fov_x, fov_y, layer_scale)
            rectangles.append(fov_rect)
            colors.append("cyan")
        
        return {
            "rectangles": rectangles,
            "colors": colors
        }
    
    def _update_overview_bounding_box(self):
        """Update the FOV boxes when parameters change."""
        # Don't update bounding box during overview acquisition
        if self._is_overview_acquiring:
            return
            
        try:
            layer_scale = (self.fm.camera.pixel_size[0], self.fm.camera.pixel_size[1])
            self._draw_fov_boxes(layer_scale)
        except Exception as e:
            logging.warning(f"Error updating FOV boxes: {e}")
    
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
            positions.append(Point(x=saved_pos.x, y=-saved_pos.y))
            
            # Use lime for selected position, cyan for others
            if i == selected_index:
                colors.extend(["lime", "lime"])  # Lime for selected position
            else:
                colors.extend(["cyan", "cyan"])  # Cyan for other saved positions
                
            labels.extend([saved_pos.name or "saved", ""])  # Use position name or default
        
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
                layer.edge_width = 6
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
                edge_width=6,
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
            
            # Acquire images at all saved positions
            images = acquire_at_positions(
                microscope=self.fm.parent,
                positions=self.stage_positions,
                channel_settings=channel_settings,
                zparams=z_parameters,
                use_autofocus=False
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
        # Update button text in case positions were modified during acquisition
        self._update_positions_button()

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

        stage_position = image.metadata.stage_position

        pos = to_napari_pos(image.data.shape[-2:], stage_position, image.metadata.pixel_size_x)

        if channel_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[channel_name].data = image.data
            self.viewer.layers[channel_name].metadata = metadata_dict
            self.viewer.layers[channel_name].colormap = wavelength_to_color(wavelength)
            self.viewer.layers[channel_name].translate = (pos.y, pos.x)  # Translate to stage position
        else:
            # If the layer does not exist, create a new one
            self.image_layer = self.viewer.add_image(
                data=image.data,
                name=channel_name,
                metadata=metadata_dict,
                colormap=wavelength_to_color(wavelength),
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

        stage_position = image.metadata.stage_position
        pos = to_napari_pos(image.data.shape[-2:], stage_position, image.metadata.pixel_size_x)

        layer_name = f"{channel_name}-{acq_date}"

        scale = (image.metadata.pixel_size_y, image.metadata.pixel_size_x)  # yx order for napari
        if image.data.ndim == 3:
            scale = (1, *scale)  # Add a singleton dimension for time if needed

        if layer_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[layer_name].data = image.data
            self.viewer.layers[layer_name].metadata = metadata_dict
            self.viewer.layers[layer_name].colormap = wavelength_to_color(wavelength)
            self.viewer.layers[layer_name].translate = (pos.y, pos.x)  # Translate to stage position
        else:
            # If the layer does not exist, create a new one
            self.viewer.add_image(
                data=image.data,
                name=layer_name,
                metadata=metadata_dict,
                colormap=wavelength_to_color(wavelength),
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
        
        # Start acquisition thread
        self._overview_thread = threading.Thread(
            target=self._overview_worker,
            args=(channel_settings, grid_size, tile_overlap),
            daemon=True
        )
        self._overview_thread.start()
    
    def _overview_worker(self, channel_settings: ChannelSettings, grid_size: tuple[int, int], tile_overlap: float):
        """Worker thread for overview acquisition."""
        try:
            logging.info(f"Acquiring overview with {grid_size[0]}x{grid_size[1]} grid, {tile_overlap:.1%} overlap")
            
            # Get the parent microscope for tileset acquisition
            if self.fm.parent is None:
                logging.error("FluorescenceMicroscope parent is None. Cannot acquire overview.")
                return
            
            # Acquire and stitch tileset
            overview_image = acquire_and_stitch_tileset(
                microscope=self.fm.parent,
                channel_settings=channel_settings,
                grid_size=grid_size,
                tile_overlap=tile_overlap
            )
            
            # Check if acquisition was cancelled
            if self._overview_stop_event.is_set():
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
            self.overview_finished_signal.emit()
    
    def _on_overview_finished(self):
        """Handle overview acquisition completion in the main thread."""
        self.pushButton_acquire_overview.setEnabled(True)
        self.pushButton_acquire_overview.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self._is_overview_acquiring = False

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

        event.accept()

def main():

    microscope, settings = utils.setup_session()
    # from fibsem.structures import BeamType
    # microscope.move_flat_to_beam(BeamType.ELECTRON)
    viewer = napari.Viewer()
    widget = FMAcquisitionWidget(fm=microscope.fm, viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    return


if __name__ == "__main__":
    main()
    # main2()




