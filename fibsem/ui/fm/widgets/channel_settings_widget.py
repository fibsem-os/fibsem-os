"""ChannelSettingsWidget — detail-panel form for a single selected ChannelSettings.

Shown below ChannelListWidget when a channel row is selected. Displays the full
set of channel parameters (excitation, emission, exposure, power, gain).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.fm.widgets.emission_filter_combo import EmissionFilterComboBox
from fibsem.ui.fm.widgets.fm_limits import (
    channel_range,
    channel_spinbox,
    gain_scale,
    power_scale,
    show_native_units,
    shown_channel_value,
    stored_channel_value,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox


class ChannelSettingsWidget(QWidget):
    """Detail-panel form showing all settings for one selected ChannelSettings.

    Shown by FluorescenceMultiChannelWidget when a row is selected.
    """

    channel_changed = pyqtSignal(object)  # ChannelSettings
    channel_field_changed = pyqtSignal(
        object, str, object
    )  # ChannelSettings, field, value

    def __init__(
        self,
        fm: FluorescenceMicroscope,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._channel: Optional[ChannelSettings] = None
        self._fm = fm
        self._emission_items: List = (
            list(fm.filter_set.available_emission_filters) if fm is not None else []
        )
        self._excitation_items: List[float] = (
            list(fm.filter_set.available_excitation_wavelengths)
            if fm is not None
            else []
        )

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
        # macOS styles forms its own way -- fields frozen at their size hint, labels
        # right-aligned, the whole block centred. That leaves ragged field widths and
        # dead space on the right, so pin all three to the house style.
        # AllNonFixedFieldsGrow, not ExpandingFieldsGrow: the latter only grows fields
        # that already carry an Expanding policy, which none of these do.
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.excitation_combo = ValueComboBox(
            items=self._excitation_items,
            unit="nm",
            decimals=0,
        )
        self.excitation_combo.setToolTip("Excitation wavelength (nm)")
        form.addRow("Excitation", self.excitation_combo)

        self.emission_combo = EmissionFilterComboBox(items=self._emission_items)
        form.addRow("Emission", self.emission_combo)

        # Unit, step and decimals as ChannelSettings declares them (its devices'
        # display hints); the ranges are what the FM's devices allow.
        self.exposure_spin = channel_spinbox(
            "exposure_time", channel_range(self._fm, "exposure_time")
        )
        self.exposure_spin.setToolTip("Exposure time (ms)")
        form.addRow("Exposure", self.exposure_spin)

        self.power_spin = channel_spinbox("power", channel_range(self._fm, "power"))
        self.power_spin.setToolTip("Light source power (%)")
        show_native_units(self.power_spin, power_scale(self._fm))
        form.addRow("Power", self.power_spin)

        self.gain_spin = channel_spinbox("gain", channel_range(self._fm, "gain"))
        self.gain_spin.setToolTip("Detector gain (%)")
        show_native_units(self.gain_spin, gain_scale(self._fm))
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

    def set_fm(self, fm: Optional[FluorescenceMicroscope]) -> None:
        """Point this widget at a different fluorescence microscope.

        The available wavelengths are the microscope's, not the channel's, so they
        have to be re-read when the microscope changes -- this widget keeps no
        ``fm`` reference of its own, only the two lists derived from it, which is
        why nothing looked stale from the outside while the combos were offering
        a disconnected microscope's filters (FIB-525).

        The selected value is preserved where the new microscope still offers it;
        where it does not, ``set_values`` falls back to the closest match rather
        than silently reporting a wavelength the hardware cannot produce.
        """
        self._fm = fm
        self._emission_items = (
            list(fm.filter_set.available_emission_filters) if fm is not None else []
        )
        self._excitation_items = (
            list(fm.filter_set.available_excitation_wavelengths)
            if fm is not None
            else []
        )

        blocked = self.blockSignals(True)
        try:
            self.excitation_combo.set_values(self._excitation_items)
            self.emission_combo.set_values(self._emission_items)
        finally:
            self.blockSignals(blocked)

        # Re-apply the channel so the controls show what it actually holds, not
        # whatever survived repopulating the combos.
        if self._channel is not None:
            self.refresh()

    def refresh(self) -> None:
        """Re-sync controls from current channel without emitting signals."""
        if self._channel is None:
            return
        self._block_all(True)
        self.excitation_combo.set_value(self._channel.excitation_wavelength)
        idx = self.emission_combo.index_of(self._channel.emission_filter)
        self.emission_combo.setCurrentIndex(idx if idx >= 0 else 0)
        for name, spin in (
            ("exposure_time", self.exposure_spin),
            ("power", self.power_spin),
            ("gain", self.gain_spin),
        ):
            spin.setValue(shown_channel_value(self._channel, name))
        self._block_all(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_all(self, block: bool) -> None:
        for w in (
            self.excitation_combo,
            self.emission_combo,
            self.exposure_spin,
            self.power_spin,
            self.gain_spin,
        ):
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
        if value == self._channel.emission_filter:
            return
        self._channel.emission_filter = value
        self.channel_field_changed.emit(self._channel, "emission_filter", value)
        self.channel_changed.emit(self._channel)

    def _on_exposure_changed(self) -> None:
        if self._channel is None:
            return
        value_s = stored_channel_value(self.exposure_spin, "exposure_time")
        if value_s == self._channel.exposure_time:
            return
        self._channel.exposure_time = value_s
        self.channel_field_changed.emit(self._channel, "exposure_time", value_s)
        self.channel_changed.emit(self._channel)

    def _on_power_changed(self) -> None:
        if self._channel is None:
            return
        value = stored_channel_value(self.power_spin, "power")
        if value == self._channel.power:
            return
        self._channel.power = value
        self.channel_field_changed.emit(self._channel, "power", value)
        self.channel_changed.emit(self._channel)

    def _on_gain_changed(self) -> None:
        if self._channel is None:
            return
        value = stored_channel_value(self.gain_spin, "gain")
        if value == self._channel.gain:
            return
        self._channel.gain = value
        self.channel_field_changed.emit(self._channel, "gain", value)
        self.channel_changed.emit(self._channel)
