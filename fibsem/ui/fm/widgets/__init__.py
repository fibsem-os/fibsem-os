"""
Fluorescence Microscopy UI Widgets

This module contains all the individual widget components for the 
fluorescence microscopy acquisition interface.
"""

from .z_parameters_widget import ZParametersWidget
from .overview_parameters_widget import OverviewParametersWidget
from .saved_positions_widget import SavedPositionsWidget
from .objective_control_widget import ObjectiveControlWidget
from .channel_settings_widget import ChannelSettingsWidget, MultiChannelSettingsWidget
from .histogram_widget import HistogramWidget

__all__ = [
    'ZParametersWidget',
    'OverviewParametersWidget', 
    'SavedPositionsWidget',
    'ObjectiveControlWidget',
    'ChannelSettingsWidget',
    'HistogramWidget'
]