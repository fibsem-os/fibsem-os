
from typing import TYPE_CHECKING, Optional

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
    "overlap_step": 0.01,
    "overlap_decimals": 2,
}

class OverviewParametersWidget(QWidget):
    def __init__(self, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.parent_widget = parent
        
        # Initialize parameters
        self.rows = OVERVIEW_PARAMETERS_CONFIG["default_rows"]
        self.cols = OVERVIEW_PARAMETERS_CONFIG["default_cols"]
        self.overlap = OVERVIEW_PARAMETERS_CONFIG["default_overlap"]
        self.use_zstack = False
        self.autofocus_mode = AutofocusMode.NONE
        
        self.initUI()

    def initUI(self):
        
        self.label_header = QLabel("Overview Parameters", self)
        self.label_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        
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
        
        # Z-stack checkbox
        self.checkBox_use_zstack = QCheckBox("Use Z-Stack", self)
        self.checkBox_use_zstack.setChecked(self.use_zstack)
        self.checkBox_use_zstack.setToolTip("Acquire z-stacks at each tile position using current Z parameters")
        
        # Auto-focus mode selection
        self.label_autofocus_mode = QLabel("Auto-Focus Mode", self)
        self.comboBox_autofocus_mode = QComboBox(self)
        self.comboBox_autofocus_mode.addItem("Don't Auto-Focus", AutofocusMode.NONE)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Once", AutofocusMode.ONCE)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Each Row", AutofocusMode.EACH_ROW)
        self.comboBox_autofocus_mode.addItem("Auto-Focus Each Tile", AutofocusMode.EACH_TILE)
        self.comboBox_autofocus_mode.setCurrentIndex(0)  # Default to NONE
        self.comboBox_autofocus_mode.setToolTip("Select when to perform auto-focus during tileset acquisition")
        
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
        layout.addWidget(self.checkBox_use_zstack, 4, 0, 1, 2)  # Span both columns
        layout.addWidget(self.label_autofocus_mode, 5, 0)
        layout.addWidget(self.comboBox_autofocus_mode, 5, 1)
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
    
    def _on_zstack_changed(self, state: int):
        """Handle z-stack checkbox change."""
        self.use_zstack = state == 2  # Qt.Checked
    
    def _on_autofocus_mode_changed(self, index: int):
        """Handle auto-focus mode change."""
        self.autofocus_mode = self.comboBox_autofocus_mode.itemData(index)
