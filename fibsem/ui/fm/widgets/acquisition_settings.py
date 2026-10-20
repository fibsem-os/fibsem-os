"""The channels, Z-stack and autofocus settings of an FM acquisition, one panel each."""

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, ZParameters
from fibsem.ui.fm.widgets.autofocus_widget import AutofocusWidget
from fibsem.ui.fm.widgets.fm_multi_channel_widget import (
    FluorescenceMultiChannelWidget,
)
from fibsem.ui.fm.widgets.z_parameters_widget import ZParametersWidget
from fibsem.ui.widgets.custom_widgets import TitledPanel


class FMAcquisitionSettingsWidget(QWidget):
    """Channel, Z and autofocus panels, with the autofocus channels following the
    channel list. The channel panel starts expanded, the other two collapsed."""

    channel_settings_changed = pyqtSignal(list)  # List[ChannelSettings]
    settings_changed = pyqtSignal()

    def __init__(
        self,
        fm: Optional[FluorescenceMicroscope],
        channel_settings: List[ChannelSettings],
        z_parameters: ZParameters,
        autofocus_settings: Optional[AutoFocusSettings] = None,
        z_title: str = "Z Parameters",
        autofocus_before_z: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.channels_widget = FluorescenceMultiChannelWidget(
            fm=fm, channel_settings=channel_settings, parent=self
        )
        self.channels_panel = TitledPanel(
            "Channel Settings", content=self.channels_widget, collapsible=True
        )

        self.z_parameters_widget = ZParametersWidget(z_parameters, parent=self)
        self.z_parameters_panel = TitledPanel(
            z_title, content=self.z_parameters_widget, collapsible=True
        )
        self.z_parameters_panel.collapse()

        self.autofocus_widget = AutofocusWidget(
            channel_settings=self.channels_widget.channel_settings, parent=self
        )
        if autofocus_settings is not None:
            self.autofocus_widget.set_autofocus_settings(autofocus_settings)
        self.autofocus_panel = TitledPanel(
            "Autofocus Settings", content=self.autofocus_widget, collapsible=True
        )
        self.autofocus_panel.collapse()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.channels_panel)
        if autofocus_before_z:
            layout.addWidget(self.autofocus_panel)
            layout.addWidget(self.z_parameters_panel)
        else:
            layout.addWidget(self.z_parameters_panel)
            layout.addWidget(self.autofocus_panel)

        self.channels_widget.settings_changed.connect(self._on_channels_changed)
        self.z_parameters_widget.settings_changed.connect(self.settings_changed)
        self.autofocus_widget.settings_changed.connect(self.settings_changed)

    def set_fm(self, fm: Optional[FluorescenceMicroscope]) -> None:
        """Offer a different microscope's filters in the channel editor."""
        self.channels_widget.set_fm(fm)

    @property
    def channel_settings(self) -> List[ChannelSettings]:
        return self.channels_widget.channel_settings

    @channel_settings.setter
    def channel_settings(self, value: List[ChannelSettings]) -> None:
        self.channels_widget.channel_settings = value

    @property
    def selected_channel(self) -> Optional[ChannelSettings]:
        return self.channels_widget.selected_channel

    @property
    def z_parameters(self) -> ZParameters:
        return self.z_parameters_widget.z_parameters

    @z_parameters.setter
    def z_parameters(self, value: ZParameters) -> None:
        self.z_parameters_widget.z_parameters = value

    @property
    def autofocus_settings(self) -> AutoFocusSettings:
        return self.autofocus_widget.autofocus_settings

    @autofocus_settings.setter
    def autofocus_settings(self, value: AutoFocusSettings) -> None:
        self.autofocus_widget.set_autofocus_settings(value)

    def _on_channels_changed(self, channel_settings: List[ChannelSettings]) -> None:
        self.autofocus_widget.update_channels(channel_settings)
        self.channel_settings_changed.emit(channel_settings)
        self.settings_changed.emit()
