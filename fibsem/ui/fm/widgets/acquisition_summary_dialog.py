import logging
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from fibsem.fm.structures import ChannelSettings, ZParameters, FMStagePosition
from fibsem.fm.timing import estimate_positions_acquisition_time
from fibsem.ui.stylesheets import (
    GREEN_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)
from fibsem.utils import format_duration


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

        # Time estimation
        try:
            time_estimate = self._calculate_time_estimate()
            if time_estimate:
                time_label = QLabel(f"Estimated Time: {time_estimate}")
                time_label.setStyleSheet("font-weight: bold; color: #0066cc; font-size: 12px;")
                layout.addWidget(time_label)
        except Exception as e:
            logging.warning(f"Could not calculate time estimate: {e}")

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
    
    def _calculate_time_estimate(self) -> Optional[str]:
        """Calculate and format the estimated acquisition time for position-based acquisition."""
        try:
            # Use the new estimate_positions_acquisition_time function
            timing_result = estimate_positions_acquisition_time(
                channel_settings=self.channel_settings,
                num_positions=len(self.checked_positions),
                zparams=self.z_parameters,
                use_autofocus=self.use_autofocus
            )
            
            # Use the fibsem utility function to format duration
            return format_duration(timing_result['total_time'])
            
        except Exception as e:
            logging.warning(f"Error calculating time estimate: {e}")
            return None