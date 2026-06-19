"""Per-task config editor widgets for grid tasks.

Each grid task type gets a tailored editor (rather than a generic
parameter form), so rich/nested configs (e.g. the overview acquisition
settings) and unit-aware fields are presented properly. The task_type ->
widget mapping lives here in the UI layer so the configs in ``grid_tasks.py``
stay Qt-free. Tasks without a custom editor yet get a placeholder.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (
    AcquireImageGridTaskConfig,
    AcquireOverviewImageGridTaskConfig,
    GridTaskConfig,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    OverviewAcquisitionSettingsWidget,
)

_ORIENTATIONS = ["SEM", "FIB", "MILLING"]
# preset beam currents (amps) for the current selector; SI-formatted (pA/nA/µA)
_BEAM_CURRENTS = [
    1e-12, 3e-12, 10e-12, 30e-12, 0.1e-9, 0.3e-9, 1e-9, 3e-9, 10e-9, 30e-9,
    0.1e-6, 0.3e-6, 1e-6,
]
# preset imaging voltages (volts); SI-formatted (V/kV)
_BEAM_VOLTAGES = [500, 1_000, 2_000, 3_000, 5_000, 10_000, 15_000, 20_000, 30_000]


class GridTaskConfigWidget(QWidget):
    """Base editor for a single GridTaskConfig.

    Subclasses implement ``_setup_ui`` (build controls, connect changes to
    ``self._on_changed``), ``_load`` (populate controls from a config) and
    ``_apply`` (read controls back into a config). Emits ``config_changed`` on
    user edits; ``get_config()`` returns the (live-updated) config.
    """

    config_changed = pyqtSignal()

    def __init__(self, config: GridTaskConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._updating = False
        self._setup_ui()
        self.set_config(config)

    # subclass hooks
    def _setup_ui(self) -> None:
        raise NotImplementedError

    def _load(self, config: GridTaskConfig) -> None:
        raise NotImplementedError

    def _apply(self, config: GridTaskConfig) -> None:
        raise NotImplementedError

    # public API
    def set_config(self, config: GridTaskConfig) -> None:
        self._config = config
        self._updating = True
        try:
            self._load(config)
        finally:
            self._updating = False

    def get_config(self) -> GridTaskConfig:
        self._apply(self._config)
        return self._config

    def _on_changed(self, *args) -> None:
        if self._updating:
            return
        self._apply(self._config)
        self.config_changed.emit()

    def _orientation_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(_ORIENTATIONS)
        combo.currentTextChanged.connect(self._on_changed)
        return combo


class OverviewGridConfigWidget(GridTaskConfigWidget):
    """Editor for AcquireOverviewImageGridTaskConfig: orientation + the (nested)
    overview acquisition settings (reuses OverviewAcquisitionSettingsWidget)."""

    def _setup_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)
        form = QFormLayout()
        self.orientation_combo = self._orientation_combo()
        form.addRow("Orientation", self.orientation_combo)
        v.addLayout(form)
        self.overview_widget = OverviewAcquisitionSettingsWidget()
        self.overview_widget.settings_changed.connect(self._on_changed)
        v.addWidget(self.overview_widget)
        v.addStretch(1)
        self._hide_save_controls()

    def _hide_save_controls(self) -> None:
        # the task manages output paths itself — hide path/filename from the editor
        isw = self.overview_widget.image_settings_widget
        for w in (isw.path_label, isw.path_edit, isw.filename_label, isw.filename_edit):
            w.setVisible(False)

    def _load(self, config) -> None:
        self.orientation_combo.setCurrentText(config.orientation or "SEM")
        self.overview_widget.update_from_settings(config.settings)
        # update_from_settings re-shows the save controls — re-hide them
        self._hide_save_controls()

    def _apply(self, config) -> None:
        config.orientation = self.orientation_combo.currentText()
        config.settings = self.overview_widget.get_settings()


class AcquireImageGridConfigWidget(GridTaskConfigWidget):
    """Editor for AcquireImageGridTaskConfig: orientation + voltage + beam
    current + the per-image acquisition settings."""

    def _setup_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # beam / stage conditions
        conditions = QWidget()
        form = QFormLayout(conditions)
        form.setContentsMargins(4, 4, 4, 4)
        self.orientation_combo = self._orientation_combo()
        self.voltage_combo = ValueComboBox(items=_BEAM_VOLTAGES, unit="V", decimals=1)
        self.voltage_combo.setToolTip("Electron beam imaging voltage")
        self.voltage_combo.currentIndexChanged.connect(self._on_changed)
        self.current_combo = ValueComboBox(items=_BEAM_CURRENTS, unit="A", decimals=1)
        self.current_combo.setToolTip("Electron beam current")
        self.current_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow("Orientation", self.orientation_combo)
        form.addRow("Beam Voltage", self.voltage_combo)
        form.addRow("Beam Current", self.current_combo)
        v.addWidget(TitledPanel("Beam / Stage Conditions", content=conditions, collapsible=True))

        # image settings (expose beam type — the task applies voltage/current to it)
        self.image_settings_widget = ImageSettingsWidget(show_beam_type=True)
        self.image_settings_widget.settings_changed.connect(lambda *_: self._on_changed())
        v.addWidget(TitledPanel("Image Settings", content=self.image_settings_widget, collapsible=True))
        v.addStretch(1)

    def _load(self, config) -> None:
        self.orientation_combo.setCurrentText(config.orientation or "SEM")
        self.voltage_combo.set_value(config.voltage)
        self.current_combo.set_value(config.beam_current)
        self.image_settings_widget.update_from_settings(config.image_settings)

    def _apply(self, config) -> None:
        config.orientation = self.orientation_combo.currentText()
        config.voltage = self.voltage_combo.value()
        config.beam_current = self.current_combo.value()
        config.image_settings = self.image_settings_widget.get_settings()


class _PlaceholderConfigWidget(GridTaskConfigWidget):
    """Shown for task types without a custom editor yet."""

    def _setup_ui(self) -> None:
        v = QVBoxLayout(self)
        name = getattr(type(self._config), "display_name", self._config.task_type)
        lbl = QLabel(f"Custom editor for '{name}' coming soon.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #808080; padding: 24px;")
        v.addWidget(lbl)

    def _load(self, config) -> None:
        pass

    def _apply(self, config) -> None:
        pass


GRID_TASK_CONFIG_WIDGETS: Dict[str, Type[GridTaskConfigWidget]] = {
    AcquireOverviewImageGridTaskConfig.task_type: OverviewGridConfigWidget,
    AcquireImageGridTaskConfig.task_type: AcquireImageGridConfigWidget,
}


def get_grid_config_widget(config: GridTaskConfig) -> GridTaskConfigWidget:
    """Build the editor widget for a grid task config (placeholder if none)."""
    cls = GRID_TASK_CONFIG_WIDGETS.get(config.task_type, _PlaceholderConfigWidget)
    return cls(config)
