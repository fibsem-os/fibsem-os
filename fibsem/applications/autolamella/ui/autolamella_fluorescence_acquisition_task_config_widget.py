import copy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QWidget
from typing import Optional
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
)
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, ZParameters, FocusMethod
from fibsem.ui.fm.widgets import (
    AutofocusWidget,
    ChannelSettingsWidget,
    ZParametersWidget,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel


class AutoLamellaFluorescenceAcquisitionTaskConfigWidget(QWidget):

    channel_settings_changed = pyqtSignal(list)
    z_parameters_changed = pyqtSignal(ZParameters)
    autofocus_settings_changed = pyqtSignal(AutoFocusSettings)
    settings_changed = pyqtSignal(AcquireFluorescenceImageConfig)

    def __init__(self, 
                 microscope,
                 config: Optional[AcquireFluorescenceImageConfig] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        if config is None:
            config = AcquireFluorescenceImageConfig()
        self.config = config
        self.microscope = microscope
        self.fm = microscope.fm
        self.initUI()

    def initUI(self):
        layout = QGridLayout()

        # Channel Settings
        from fibsem.ui.fm.widgets.fm_multi_channel_widget import FluorescenceMultiChannelWidget
        self.channelSettingsWidget = FluorescenceMultiChannelWidget(fm=self.microscope.fm,
                                                                    channel_settings=self.config.channel_settings,
                                                                    parent=self)
        self.channelPanel = TitledPanel("Channel Settings", content=self.channelSettingsWidget, collapsible=True)

        # Z Parameters
        self.z_parameters_widget = ZParametersWidget(self.config.zparams, parent=self)
        self.zParametersPanel = TitledPanel("Z Parameters", content=self.z_parameters_widget, collapsible=True)

        # Create autofocus widget
        self.autofocusWidget = AutofocusWidget(
            channel_settings=self.channelSettingsWidget.channel_settings,
            parent=self
        )
        self.autofocusWidget.set_autofocus_settings(self.config.autofocus_settings)
        self.autofocusPanel = TitledPanel("Autofocus Settings", content=self.autofocusWidget, collapsible=True)
        self.autofocusWidget.single_fine_search_mode()

        # propagate settings change events
        self.channelSettingsWidget.settings_changed.connect(self._on_channel_settings_changed)
        self.z_parameters_widget.settings_changed.connect(self._on_z_parameters_changed)
        self.autofocusWidget.settings_changed.connect(self._on_autofocus_settings_changed)

        layout.addWidget(self.channelPanel, 0, 0)
        layout.addWidget(self.zParametersPanel, 1, 0)
        layout.addWidget(self.autofocusPanel, 2, 0)

        self.zParametersPanel.collapse()
        self.autofocusPanel.collapse()

        self.channelSettingsWidget.set_live_acquisition_controls(enabled=False)
    
        self.setLayout(layout)

    def get_task_config(self) -> AcquireFluorescenceImageConfig:
        self.config.zparams = self.z_parameters_widget.z_parameters
        self.config.channel_settings = self.channelSettingsWidget.channel_settings
        self.config.autofocus_settings = self.autofocusWidget.get_autofocus_settings()
        return self.config

    def set_task_config(self, config: AcquireFluorescenceImageConfig):
        self.blockSignals(True)
        self.config = copy.deepcopy(config)
        self.z_parameters_widget.z_parameters = config.zparams
        self.channelSettingsWidget.channel_settings = config.channel_settings # this also updates autofocus channels...
        self.autofocusWidget.set_autofocus_settings(config.autofocus_settings)
        self.blockSignals(False)

    def _on_channel_settings_changed(self, channel_settings: list):
        """Relay channel settings updates."""
        self.config.channel_settings = channel_settings
        self.autofocusWidget.update_channels(channel_settings)
        self.channel_settings_changed.emit(channel_settings)
        self._emit_settings_changed()

    def _on_z_parameters_changed(self, zparams: ZParameters):
        """Relay z-parameter updates."""
        self.config.zparams = zparams
        self.z_parameters_changed.emit(zparams)
        self._emit_settings_changed()

    def _on_autofocus_settings_changed(self, autofocus_settings: AutoFocusSettings):
        """Relay autofocus settings updates."""
        self.config.autofocus_settings = autofocus_settings
        self.autofocus_settings_changed.emit(autofocus_settings)
        self._emit_settings_changed()

    def _emit_settings_changed(self):
        """Emit the aggregated task configuration."""
        self.settings_changed.emit(self.get_task_config())


def main():
    import napari
    from fibsem import utils

    microscope, settings = utils.setup_session()
    config = AcquireFluorescenceImageConfig(
            task_name="TASK",
            channel_settings=[
                ChannelSettings(name="Reflection Channel", excitation_wavelength=550, emission_wavelength=None, power=0.02, exposure_time=0.005),
                ChannelSettings(name="Red Channel", excitation_wavelength=550, emission_wavelength="FLUORESCENCE", power=0.3, exposure_time=0.5, color="red")],
            zparams=ZParameters(zmin=-10e-6, zmax=10e-6, zstep=1e-6),
            autofocus_settings=AutoFocusSettings(fine_enabled=False, method=FocusMethod.SOBEL, channel_name="Reflection Channel")
        )

    viewer = napari.Viewer()
    widget = AutoLamellaFluorescenceAcquisitionTaskConfigWidget(microscope=microscope, config=None)
    viewer.window.add_dock_widget(widget, area='right')

    widget.set_task_config(config)

    def _on_settings_changed(updated: AcquireFluorescenceImageConfig):
        print("Task config updated:", updated.autofocus_settings)
    widget.settings_changed.connect(_on_settings_changed)

    napari.run()


if __name__ == "__main__":
    main()
