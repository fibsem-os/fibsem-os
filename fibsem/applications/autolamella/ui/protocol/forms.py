import copy
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QWidget

from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
)
from fibsem.ui.fm.widgets.acquisition_settings import FMAcquisitionSettingsWidget
from fibsem.ui.widgets.custom_widgets import ValueComboBox

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


class AutoLamellaFluorescenceAcquisitionTaskConfigWidget(QWidget):
    channel_settings_changed = pyqtSignal(list)
    settings_changed = pyqtSignal(AcquireFluorescenceImageConfig)

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: Optional[AcquireFluorescenceImageConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if config is None:
            config = AcquireFluorescenceImageConfig()
        self.config = config
        self.microscope = microscope
        self.fm = microscope.fm
        self.initUI()

    def initUI(self):
        layout = QGridLayout()

        # Orientation selector
        orientation_label = QLabel("Orientation:")
        self.orientation_combo = ValueComboBox(
            items=[None, "FM", "SEM"],
            value=self.config.orientation,
            format_fn=lambda v: "Default (use pose)" if v is None else v,
        )
        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)

        self.acquisition_settings = FMAcquisitionSettingsWidget(
            fm=self.microscope.fm,
            channel_settings=self.config.channel_settings,
            z_parameters=self.config.zparams,
            autofocus_settings=self.config.autofocus_settings,
            parent=self,
        )
        self.acquisition_settings.channel_settings_changed.connect(
            self.channel_settings_changed
        )
        self.acquisition_settings.settings_changed.connect(self._emit_settings_changed)

        layout.addWidget(orientation_label, 0, 0)
        layout.addWidget(self.orientation_combo, 0, 1)
        layout.addWidget(self.acquisition_settings, 1, 0, 1, 2)

        self.setLayout(layout)

    def set_microscope(self, microscope: "FibsemMicroscope") -> None:
        """Point this widget at a different microscope after a reconnect.

        The editors that own this widget are built once and survive
        disconnect/reconnect, so without this the widget keeps the microscope it
        was constructed with -- and its channel editor keeps offering that
        microscope's excitation/emission filters. Connect to a non-FM system
        first and it offers none at all, for the life of the process.

        The captured ``fm`` matters as much as the microscope: it is a live
        handle to hardware, not a value.
        """
        self.microscope = microscope
        self.fm = microscope.fm
        self.acquisition_settings.set_fm(microscope.fm)

    def get_task_config(self) -> AcquireFluorescenceImageConfig:
        self.config.orientation = self.orientation_combo.value()
        self.config.zparams = self.acquisition_settings.z_parameters
        self.config.channel_settings = self.acquisition_settings.channel_settings
        self.config.autofocus_settings = self.acquisition_settings.autofocus_settings
        return self.config

    def set_task_config(self, config: AcquireFluorescenceImageConfig):
        self.blockSignals(True)
        self.config = copy.deepcopy(config)
        self.orientation_combo.set_value(config.orientation)
        self.acquisition_settings.z_parameters = config.zparams
        self.acquisition_settings.channel_settings = config.channel_settings
        self.acquisition_settings.autofocus_settings = config.autofocus_settings
        self.blockSignals(False)

    def _on_orientation_changed(self):
        self.config.orientation = self.orientation_combo.value()
        self._emit_settings_changed()

    def _emit_settings_changed(self):
        """Emit the aggregated task configuration."""
        self.settings_changed.emit(self.get_task_config())
