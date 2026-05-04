import logging
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, ZParameters, OverviewParameters
from fibsem.fm.timing import estimate_tileset_acquisition_time
from fibsem.ui.stylesheets import (
    GREEN_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
)
from fibsem.utils import format_duration


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
        overview_params: OverviewParameters = self.settings['overview_parameters']
        try:
            fov_x, fov_y = self.fm.camera.field_of_view
            from fibsem.fm.acquisition import calculate_grid_coverage_area
            total_width, total_height = calculate_grid_coverage_area(
                ncols=overview_params.cols, nrows=overview_params.rows,
                fov_x=fov_x, fov_y=fov_y, overlap=overview_params.overlap
            )
            total_area = f"{total_width*1e6:.1f} x {total_height*1e6:.1f} μm"
        except Exception:
            total_area = "N/A"

        grid_label = QLabel(f"Grid Size: {overview_params.rows} x {overview_params.cols}. (Area: {total_area})")
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
        if overview_params.use_zstack and self.settings['z_parameters']:
            z_params: ZParameters = self.settings['z_parameters']
            z_label = QLabel(f"{z_params.pretty_name}")
            params_layout.addWidget(z_label)

        # Auto-focus
        af_label = QLabel(f"Auto-Focus: {overview_params.autofocus_mode.name.replace('_', ' ').title()}")
        params_layout.addWidget(af_label)

        # Time estimation
        try:
            time_estimate = self._calculate_time_estimate()
            if time_estimate:
                time_label = QLabel(f"Estimated Time: {time_estimate}")
                time_label.setStyleSheet("font-weight: bold; color: #0066cc;")
                params_layout.addWidget(time_label)
        except Exception as e:
            logging.warning(f"Could not calculate time estimate: {e}")

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

    def _calculate_time_estimate(self) -> Optional[str]:
        """Calculate and format the estimated acquisition time for the overview."""
        try:
            # Get settings
            channel_settings: List[ChannelSettings] = self.settings['channel_settings']
            overview_params: OverviewParameters = self.settings['overview_parameters']
            z_parameters = self.settings['z_parameters'] if overview_params.use_zstack else None
            
            # Estimate acquisition time using the tileset function
            timing_result = estimate_tileset_acquisition_time(
                channel_settings=channel_settings,
                grid_size=(overview_params.rows, overview_params.cols),
                zparams=z_parameters,
                autofocus_mode=overview_params.autofocus_mode,  # Pass enum directly
            )
            
            total_time_seconds = timing_result["total_time"]
            
            # Use the fibsem utility function to format duration
            return format_duration(total_time_seconds)
                
        except Exception as e:
            logging.warning(f"Error calculating time estimate: {e}")
            return None