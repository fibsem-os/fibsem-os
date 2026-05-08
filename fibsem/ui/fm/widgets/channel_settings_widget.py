"""ChannelSettingsWidget — detail-panel form for a single selected ChannelSettings.

Shown below ChannelListWidget when a channel row is selected. Displays the full
set of channel parameters (excitation, emission, exposure, power, gain).
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox, ValueSpinBox

_MS_TO_S = 1e-3
_S_TO_MS = 1e3
_PCT_TO_FRAC = 1e-2
_FRAC_TO_PCT = 1e2


def _fmt_emission(w) -> str:
    if w is None:
        return "Reflection"
    if isinstance(w, str):
        return w
    return f"{int(w)} nm"


class ChannelSettingsWidget(QWidget):
    """Detail-panel form showing all settings for one selected ChannelSettings.

    Shown by FluorescenceMultiChannelWidget when a row is selected.
    """

    channel_changed = pyqtSignal(object)                    # ChannelSettings
    channel_field_changed = pyqtSignal(object, str, object) # ChannelSettings, field, value

    def __init__(
        self,
        fm: FluorescenceMicroscope,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._channel: Optional[ChannelSettings] = None
        self._emission_items: List = list(fm.filter_set.available_emission_wavelengths)
        self._excitation_items: List[float] = list(fm.filter_set.available_excitation_wavelengths)

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(6)
        form.setLabelAlignment(form.labelAlignment())

        self.excitation_combo = ValueComboBox(
            items=self._excitation_items,
            unit="nm",
            decimals=0,
        )
        self.excitation_combo.setToolTip("Excitation wavelength (nm)")
        form.addRow("Excitation", self.excitation_combo)

        self.emission_combo = ValueComboBox(
            items=self._emission_items,
            format_fn=_fmt_emission,
        )
        self.emission_combo.setToolTip("Emission / filter")
        form.addRow("Emission", self.emission_combo)

        self.exposure_spin = ValueSpinBox(
            suffix="ms",
            minimum=1.0,
            maximum=10000.0,
            step=1.0,
            decimals=1,
        )
        self.exposure_spin.setToolTip("Exposure time (ms)")
        form.addRow("Exposure", self.exposure_spin)

        self.power_spin = ValueSpinBox(
            suffix="%",
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            decimals=1,
        )
        self.power_spin.setToolTip("Light source power (%)")
        form.addRow("Power", self.power_spin)

        self.gain_spin = ValueSpinBox(
            suffix="%",
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            decimals=1,
        )
        self.gain_spin.setToolTip("Detector gain (%)")
        form.addRow("Gain", self.gain_spin)

        self._panel = TitledPanel("Channel", content=form_widget)
        outer.addWidget(self._panel)

    def _connect_signals(self) -> None:
        self.excitation_combo.currentIndexChanged.connect(self._on_excitation_changed)
        self.emission_combo.currentIndexChanged.connect(self._on_emission_changed)
        self.exposure_spin.editingFinished.connect(self._on_exposure_changed)
        self.power_spin.editingFinished.connect(self._on_power_changed)
        self.gain_spin.editingFinished.connect(self._on_gain_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel(self, channel: ChannelSettings) -> None:
        """Load channel data into all controls without emitting signals."""
        self._channel = channel
        self._panel.set_title(channel.name)
        self.refresh()

    def refresh(self) -> None:
        """Re-sync controls from current channel without emitting signals."""
        if self._channel is None:
            return
        self._block_all(True)
        self.excitation_combo.set_value(self._channel.excitation_wavelength)
        idx = self.emission_combo.findData(self._channel.emission_wavelength)
        self.emission_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.exposure_spin.setValue(self._channel.exposure_time * _S_TO_MS)
        power = self._channel.power if self._channel.power is not None else 0.0
        self.power_spin.setValue(power * _FRAC_TO_PCT)
        gain = self._channel.gain if self._channel.gain is not None else 0.0
        self.gain_spin.setValue(gain * _FRAC_TO_PCT)
        self._block_all(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_all(self, block: bool) -> None:
        for w in (self.excitation_combo, self.emission_combo,
                  self.exposure_spin, self.power_spin, self.gain_spin):
            w.blockSignals(block)

    # ------------------------------------------------------------------
    # Mutation handlers
    # ------------------------------------------------------------------

    def _on_excitation_changed(self) -> None:
        if self._channel is None:
            return
        value = self.excitation_combo.value()
        if value is None or value == self._channel.excitation_wavelength:
            return
        self._channel.excitation_wavelength = value
        self.channel_field_changed.emit(self._channel, "excitation_wavelength", value)
        self.channel_changed.emit(self._channel)

    def _on_emission_changed(self) -> None:
        if self._channel is None:
            return
        value = self.emission_combo.value()
        if value == self._channel.emission_wavelength:
            return
        self._channel.emission_wavelength = value
        self.channel_field_changed.emit(self._channel, "emission_wavelength", value)
        self.channel_changed.emit(self._channel)

    def _on_exposure_changed(self) -> None:
        if self._channel is None:
            return
        value_s = self.exposure_spin.value() * _MS_TO_S
        if value_s == self._channel.exposure_time:
            return
        self._channel.exposure_time = value_s
        self.channel_field_changed.emit(self._channel, "exposure_time", value_s)
        self.channel_changed.emit(self._channel)

    def _on_power_changed(self) -> None:
        if self._channel is None:
            return
        value = self.power_spin.value() * _PCT_TO_FRAC
        if value == self._channel.power:
            return
        self._channel.power = value
        self.channel_field_changed.emit(self._channel, "power", value)
        self.channel_changed.emit(self._channel)

    def _on_gain_changed(self) -> None:
        if self._channel is None:
            return
        value = self.gain_spin.value() * _PCT_TO_FRAC
        if value == self._channel.gain:
            return
        self._channel.gain = value
        self.channel_field_changed.emit(self._channel, "gain", value)
        self.channel_changed.emit(self._channel)
