"""FluorescenceMultiChannelWidget — composes ChannelListWidget + ChannelSettingsWidget.

Selecting a row in the list shows its full settings in a detail panel below,
following the FibsemMillingStagesWidget pattern.
"""
from __future__ import annotations

from typing import List, Optional, Union

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.fm.widgets.channel_list_widget import ChannelListWidget
from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget


class FluorescenceMultiChannelWidget(QWidget):
    """Composes ChannelListWidget + ChannelSettingsWidget.

    Selecting a channel row shows its full settings (excitation, emission,
    exposure, power, gain) in a detail panel below. Both row and detail
    panel edits are aggregated into ``channel_field_changed`` and
    ``settings_changed``.

    Drop-in replacement for ChannelListWidget at call sites.
    """

    channel_field_changed = pyqtSignal(object, str, object)  # ChannelSettings, field, value
    settings_changed = pyqtSignal(list)    # List[ChannelSettings]
    channel_added = pyqtSignal(object)     # ChannelSettings
    channel_removed = pyqtSignal(object)   # ChannelSettings
    channel_changed = pyqtSignal(object)   # ChannelSettings
    enabled_changed = pyqtSignal(list)     # List[ChannelSettings] (enabled only)
    order_changed = pyqtSignal(list)       # List[ChannelSettings]

    def __init__(
        self,
        fm: FluorescenceMicroscope,
        channel_settings: Union[ChannelSettings, List[ChannelSettings]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.fm = fm
        self._selected_channel: Optional[ChannelSettings] = None
        self._pending_inline_channel: Optional[ChannelSettings] = None
        self._pending_inline_update: bool = False

        self._setup_ui()
        self._connect_signals()

        if isinstance(channel_settings, ChannelSettings):
            channel_settings = [channel_settings]
        self.channel_settings = list(channel_settings)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._list = ChannelListWidget(fm=self.fm, channel_settings=[])
        layout.addWidget(self._list)

        self._settings_widget = ChannelSettingsWidget(fm=self.fm)
        self._settings_widget.setVisible(False)
        layout.addWidget(self._settings_widget)

    def _connect_signals(self) -> None:
        self._list.channel_selected.connect(self._on_channel_selected)
        self._list.channel_added.connect(self._on_channel_added)
        self._list.channel_removed.connect(self._on_channel_removed)
        self._list.channel_changed.connect(self._on_inline_channel_changed)
        self._list.channel_field_changed.connect(self.channel_field_changed)
        self._list.order_changed.connect(self._on_order_changed)
        self._list.enabled_changed.connect(self._on_enabled_changed)

        self._settings_widget.channel_field_changed.connect(self.channel_field_changed)
        self._settings_widget.channel_changed.connect(self._on_detail_changed)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_channel_selected(self, channel: ChannelSettings) -> None:
        self._selected_channel = channel
        self._settings_widget.set_channel(channel)
        self._settings_widget.setVisible(True)

    def _on_channel_added(self, channel: ChannelSettings) -> None:
        self.channel_added.emit(channel)
        self.settings_changed.emit(self._list.channel_settings)

    def _on_channel_removed(self, channel: ChannelSettings) -> None:
        if self._selected_channel is channel:
            self._selected_channel = None
            self._settings_widget.setVisible(False)
        self.channel_removed.emit(channel)
        self.settings_changed.emit(self._list.channel_settings)

    def _on_inline_channel_changed(self, channel: ChannelSettings) -> None:
        """Row inline edit (name/color). Defer to avoid re-entrant redraws."""
        self._pending_inline_channel = channel
        if self._pending_inline_update:
            return
        self._pending_inline_update = True
        QTimer.singleShot(0, self._flush_inline_channel_changed)

    def _flush_inline_channel_changed(self) -> None:
        self._pending_inline_update = False
        channel = self._pending_inline_channel
        self._pending_inline_channel = None
        if channel is None:
            return
        if channel not in self._list.channel_settings:
            return
        # Sync detail panel if the edited row is selected (name/color/excitation/emission)
        if self._selected_channel is channel:
            self._settings_widget.set_channel(channel)
        self.channel_changed.emit(channel)
        self.settings_changed.emit(self._list.channel_settings)

    def _on_detail_changed(self, channel: ChannelSettings) -> None:
        """Detail panel edit — sync the row widget and emit settings_changed."""
        self._list.refresh_channel(channel)
        self.channel_changed.emit(channel)
        self.settings_changed.emit(self._list.channel_settings)

    def _on_order_changed(self, channels: List[ChannelSettings]) -> None:
        self.order_changed.emit(channels)
        self.settings_changed.emit(self._list.channel_settings)

    def _on_enabled_changed(self, enabled_channels: List[ChannelSettings]) -> None:
        self.enabled_changed.emit(enabled_channels)
        self.settings_changed.emit(self._list.channel_settings)

    # ------------------------------------------------------------------
    # Public API (drop-in replacement for ChannelListWidget)
    # ------------------------------------------------------------------

    @property
    def selected_channel(self) -> Optional[ChannelSettings]:
        return self._list.selected_channel

    @property
    def channel_settings(self) -> List[ChannelSettings]:
        return self._list.channel_settings

    @channel_settings.setter
    def channel_settings(self, value: Union[ChannelSettings, List[ChannelSettings]]) -> None:
        if isinstance(value, ChannelSettings):
            value = [value]
        channels = list(value)
        self._list.channel_settings = channels
        # Select first channel and show detail
        if channels:
            self._list.select_channel(channels[0])
        else:
            self._selected_channel = None
            self._settings_widget.setVisible(False)

    def set_live_acquisition_controls(self, enabled: bool) -> None:
        """Stub for ChannelSettingsWidget/ChannelListWidget compatibility."""
        pass
