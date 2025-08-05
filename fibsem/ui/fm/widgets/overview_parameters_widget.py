
import logging
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from fibsem.fm.acquisition import AutofocusMode, calculate_grid_coverage_area

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

OVERVIEW_PARAMETERS_CONFIG = {
    "min_grid_size": 1,
    "max_grid_size": 15,
    "default_rows": 3,
    "default_cols": 3,
    "default_overlap": 0.1,
    "default_use_zstack": False,
    "default_autofocus_mode": AutofocusMode.NONE,
    "overlap_range": (0.0, 0.9),
    "overlap_step": 0.01,
    "overlap_decimals": 2,
    "tooltips": {
        "rows": "Number of rows in the overview grid",
        "cols": "Number of columns in the overview grid",
        "overlap": "Fraction of overlap between adjacent tiles",
        "use_zstack": "Acquire z-stacks at each tile position using current Z parameters",
        "autofocus_mode": "Select when to perform auto-focus during tileset acquisition",
    },
}

class OverviewParametersWidget(QWidget):
    def __init__(self, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.parent_widget = parent
        
        # Initialize parameters
        self.rows = OVERVIEW_PARAMETERS_CONFIG["default_rows"]
        self.cols = OVERVIEW_PARAMETERS_CONFIG["default_cols"]
        self.overlap = OVERVIEW_PARAMETERS_CONFIG["default_overlap"]
        self.use_zstack = OVERVIEW_PARAMETERS_CONFIG["default_use_zstack"]
        self.autofocus_mode = OVERVIEW_PARAMETERS_CONFIG["default_autofocus_mode"]
        
        self.initUI()

    def initUI(self):
        
        # Number of rows
        self.label_rows = QLabel("Rows", self)
        self.spinBox_rows = QSpinBox(self)
        self.spinBox_rows.setRange(OVERVIEW_PARAMETERS_CONFIG["min_grid_size"], 
                                   OVERVIEW_PARAMETERS_CONFIG["max_grid_size"])
        self.spinBox_rows.setValue(self.rows)
        self.spinBox_rows.setToolTip(OVERVIEW_PARAMETERS_CONFIG["tooltips"]["rows"])
        
        # Number of columns
        self.label_cols = QLabel("Columns", self)
        self.spinBox_cols = QSpinBox(self)
        self.spinBox_cols.setRange(OVERVIEW_PARAMETERS_CONFIG["min_grid_size"], 
                                   OVERVIEW_PARAMETERS_CONFIG["max_grid_size"])
        self.spinBox_cols.setValue(self.cols)
        self.spinBox_cols.setToolTip(OVERVIEW_PARAMETERS_CONFIG["tooltips"]["cols"])
        
        # Tile overlap
        self.label_overlap = QLabel("Overlap", self)
        self.doubleSpinBox_overlap = QDoubleSpinBox(self)
        self.doubleSpinBox_overlap.setRange(*OVERVIEW_PARAMETERS_CONFIG["overlap_range"])
        self.doubleSpinBox_overlap.setValue(self.overlap)
        self.doubleSpinBox_overlap.setSingleStep(OVERVIEW_PARAMETERS_CONFIG["overlap_step"])
        self.doubleSpinBox_overlap.setDecimals(OVERVIEW_PARAMETERS_CONFIG["overlap_decimals"])
        self.doubleSpinBox_overlap.setToolTip(OVERVIEW_PARAMETERS_CONFIG["tooltips"]["overlap"])
        self.doubleSpinBox_overlap.setKeyboardTracking(False)
        
        # Z-stack checkbox
        self.checkBox_use_zstack = QCheckBox("Use Z-Stack", self)
        self.checkBox_use_zstack.setChecked(self.use_zstack)
        self.checkBox_use_zstack.setToolTip(OVERVIEW_PARAMETERS_CONFIG["tooltips"]["use_zstack"])
        
        # Z-stack planes info (shown when z-stack is enabled)
        self.label_zstack_planes_value = QLabel(self._calculate_zstack_planes(), self)
        self.label_zstack_planes_value.setStyleSheet("QLabel { color: #666666; }")
        self.label_zstack_planes_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Auto-focus mode selection
        self.label_autofocus_mode = QLabel("Auto-Focus Mode", self)
        self.comboBox_autofocus_mode = QComboBox(self)
        self.comboBox_autofocus_mode.addItem("Don't Auto-Focus", AutofocusMode.NONE)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Once", AutofocusMode.ONCE)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Each Row", AutofocusMode.EACH_ROW)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Each Tile", AutofocusMode.EACH_TILE)
        
        # Set default autofocus mode from config
        default_mode = OVERVIEW_PARAMETERS_CONFIG["default_autofocus_mode"]
        for i in range(self.comboBox_autofocus_mode.count()):
            if self.comboBox_autofocus_mode.itemData(i) == default_mode:
                self.comboBox_autofocus_mode.setCurrentIndex(i)
                break
        self.comboBox_autofocus_mode.setToolTip(OVERVIEW_PARAMETERS_CONFIG["tooltips"]["autofocus_mode"])

        # Auto-focus channel selection
        self.label_autofocus_channel = QLabel("Auto-Focus Channel", self)
        self.comboBox_autofocus_channel = QComboBox(self)
        self.comboBox_autofocus_channel.setToolTip("Select the channel to use for auto-focus during overview acquisition")
        self.autofocus_channel_name = None

        # Total area (calculated, read-only)
        self.label_total_area = QLabel("Total Area", self)
        self.label_total_area_value = QLabel(self._calculate_total_area(), self)
        self.label_total_area_value.setStyleSheet("QLabel { color: #666666; }")

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_rows, 0, 0)
        layout.addWidget(self.spinBox_rows, 0, 1)
        layout.addWidget(self.label_cols, 1, 0)
        layout.addWidget(self.spinBox_cols, 1, 1)
        layout.addWidget(self.label_overlap, 2, 0)
        layout.addWidget(self.doubleSpinBox_overlap, 2, 1)
        layout.addWidget(self.checkBox_use_zstack, 3, 0)
        layout.addWidget(self.label_zstack_planes_value, 3, 1)
        layout.addWidget(self.label_autofocus_mode, 4, 0)
        layout.addWidget(self.comboBox_autofocus_mode, 4, 1)
        layout.addWidget(self.label_autofocus_channel, 5, 0)
        layout.addWidget(self.comboBox_autofocus_channel, 5, 1)
        layout.addWidget(self.label_total_area, 6, 0)
        layout.addWidget(self.label_total_area_value, 6, 1)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.spinBox_rows.valueChanged.connect(self._on_rows_changed)
        self.spinBox_cols.valueChanged.connect(self._on_cols_changed)
        self.doubleSpinBox_overlap.valueChanged.connect(self._on_overlap_changed)
        self.checkBox_use_zstack.stateChanged.connect(self._on_zstack_changed)
        self.comboBox_autofocus_mode.currentIndexChanged.connect(self._on_autofocus_mode_changed)
        self.comboBox_autofocus_channel.currentIndexChanged.connect(self._on_autofocus_channel_changed)

        # Set initial z-stack planes visibility
        self._update_zstack_planes_visibility()

        # Initialize channel names from parent widget
        self._update_channel_names_from_parent()

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
            
            return f"{width_um:.1f} x {height_um:.1f} μm"
            
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
    
    def get_use_zstack(self) -> bool:
        """Get whether to use z-stack acquisition."""
        return self.use_zstack
    
    def get_autofocus_mode(self) -> AutofocusMode:
        """Get the selected auto-focus mode."""
        return self.autofocus_mode

    def get_autofocus_channel_name(self) -> str:
        """Get the selected auto-focus channel name."""
        return self.autofocus_channel_name

    def update_channel_names(self, channel_names: list):
        """Update the autofocus channel combobox with available channel names."""
        current_selection = self.autofocus_channel_name
        self.comboBox_autofocus_channel.clear()

        for channel_name in channel_names:
            self.comboBox_autofocus_channel.addItem(channel_name)

        # Restore previous selection if it still exists
        if current_selection and current_selection in channel_names:
            index = self.comboBox_autofocus_channel.findText(current_selection)
            if index >= 0:
                self.comboBox_autofocus_channel.setCurrentIndex(index)
        elif channel_names:
            # Select first channel by default
            self.comboBox_autofocus_channel.setCurrentIndex(0)
            self.autofocus_channel_name = channel_names[0]
    
    def _update_channel_names_from_parent(self):
        """Update channel names from the parent widget's channel settings."""
        try:
            if (self.parent_widget and 
                hasattr(self.parent_widget, 'channelSettingsWidget') and 
                self.parent_widget.channelSettingsWidget):
                channel_names = [channel.name for channel in self.parent_widget.channelSettingsWidget.channel_settings]
                self.update_channel_names(channel_names)
        except Exception as e:
            logging.warning(f"Error updating channel names from parent: {e}")

    def _on_zstack_changed(self, state: int):
        """Handle z-stack checkbox change."""
        self.use_zstack = state == 2  # Qt.Checked
        self._update_zstack_planes_visibility()
    
    def _on_autofocus_mode_changed(self, index: int):
        """Handle auto-focus mode change."""
        self.autofocus_mode = self.comboBox_autofocus_mode.itemData(index)

    def _on_autofocus_channel_changed(self, index: int):
        """Handle auto-focus channel change."""
        if index >= 0:
            self.autofocus_channel_name = self.comboBox_autofocus_channel.itemText(index)
        else:
            self.autofocus_channel_name = None

    def _calculate_zstack_planes(self) -> str:
        """Calculate the number of z-stack planes based on current Z parameters."""
        try:
            # Get Z parameters from parent widget
            if self.parent_widget and hasattr(self.parent_widget, 'zParametersWidget'):
                z_params = self.parent_widget.zParametersWidget.z_parameters
                if z_params:
                    # Use the convenient property
                    num_planes = z_params.num_planes
                    if num_planes <= 0:
                        return "Invalid"
                    return f"{num_planes} planes"
            return "N/A"
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            return "N/A"
    
    def _update_zstack_planes_visibility(self):
        """Update visibility of z-stack planes info based on checkbox state."""
        is_visible = self.use_zstack
        self.label_zstack_planes_value.setVisible(is_visible)
        
        # Update the plane count if visible
        if is_visible:
            self.label_zstack_planes_value.setText(self._calculate_zstack_planes())
        else:
            self.label_zstack_planes_value.setText("")
