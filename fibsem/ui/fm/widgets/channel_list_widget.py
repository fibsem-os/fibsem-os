"""ChannelListWidget — compact list-style channel settings widget.

Replaces the stacked-form ChannelSettingsWidget with an inline-row design
following the MillingStageListWidget pattern. All controls are visible per row.

Exposes the same public interface as ChannelSettingsWidget so it can be swapped
in without changes to callers.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import List, Optional, Union

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, ValueComboBox, ValueSpinBox

_DRAG_HANDLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "icons", "drag_handle.svg",
)

_NAME_MIN_WIDTH = 130
_EXCITATION_FIXED_WIDTH = 90
_EMISSION_FIXED_WIDTH = 120
_EXPOSURE_FIXED_WIDTH = 120
_GAIN_FIXED_WIDTH = 100
_POWER_FIXED_WIDTH = 90
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 2 + 8  # color + remove

_MS_TO_S = 1e-3
_S_TO_MS = 1e3
_PCT_TO_FRAC = 1e-2
_FRAC_TO_PCT = 1e2

_NAME_EDIT_STYLE = (
    "QLineEdit { background: transparent; border: none; }"
    "QLineEdit:focus { background: #1e2124; border: 1px solid #555; border-radius: 2px; }"
)

AVAILABLE_COLORS = ["violet", "blue", "cyan", "green", "yellow", "red", "gray"]


def _make_color_icon(color_name: str, size: int = 16) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor(color_name))
    return QIcon(px)


def _unique_name(name: str, existing: set) -> str:
    if name not in existing:
        return name
    n = 2
    while f"{name} ({n})" in existing:
        n += 1
    return f"{name} ({n})"


# ---------------------------------------------------------------------------
# Draggable list
# ---------------------------------------------------------------------------

class _DraggableChannelList(QListWidget):
    """QListWidget with InternalMove drag-and-drop that emits the new channel
    order after each drop.  Qt clears itemWidget on move, so the parent must
    listen to ``reordered`` and rebuild row widgets."""

    reordered = pyqtSignal(list)  # List[ChannelSettings]

    def __init__(self, fm, parent=None) -> None:
        super().__init__(parent)
        self._fm = fm

    def mousePressEvent(self, event) -> None:
        if self._fm.is_acquiring:
            return  # swallow — prevent highlight change during live acquisition
        super().mousePressEvent(event)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        channels = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(channels)


# ---------------------------------------------------------------------------
# Row widget
# ---------------------------------------------------------------------------

class ChannelRowWidget(QWidget):
    enabled_changed = pyqtSignal(object, bool)         # ChannelSettings, enabled
    remove_clicked = pyqtSignal(object)                # ChannelSettings
    row_clicked = pyqtSignal(object)                   # ChannelSettings
    channel_changed = pyqtSignal(object)               # ChannelSettings after inline mutation
    channel_field_changed = pyqtSignal(object, str, object)  # ChannelSettings, field, value

    def __init__(
        self,
        channel: ChannelSettings,
        emission_items: List,
        excitation_items: List[float],
        enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.channel = channel
        self._emission_items = emission_items
        self._excitation_items = excitation_items
        self.name_validator: Optional[callable] = None
        self.name_rejected: Optional[callable] = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(enabled)
        self.checkbox.setToolTip("Enable/disable channel")
        self.checkbox.setStyleSheet("background: transparent;")
        self.checkbox.setVisible(False)
        layout.addWidget(self.checkbox)

        self.name_edit = QLineEdit()
        self.name_edit.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_edit.setStyleSheet(_NAME_EDIT_STYLE)
        self.name_edit.setToolTip("Channel name")
        layout.addWidget(self.name_edit, 1)

        self.excitation_combo = ValueComboBox(
            items=excitation_items,
            unit="nm",
            decimals=0,
        )
        self.excitation_combo.setFixedWidth(_EXCITATION_FIXED_WIDTH)
        self.excitation_combo.setToolTip("Excitation wavelength (nm)")
        layout.addWidget(self.excitation_combo)

        def _fmt_emission(w) -> str:
            if w is None:
                return "Reflection"
            if isinstance(w, str):
                return w
            return f"{int(w)} nm"

        self.emission_combo = ValueComboBox(
            items=emission_items,
            format_fn=_fmt_emission,
        )
        self.emission_combo.setFixedWidth(_EMISSION_FIXED_WIDTH)
        self.emission_combo.setToolTip("Emission wavelength")
        layout.addWidget(self.emission_combo)

        self.exposure_spin = ValueSpinBox(
            suffix="ms",
            minimum=1.0,
            maximum=10000.0,
            step=1.0,
            decimals=1,
        )
        self.exposure_spin.setFixedWidth(_EXPOSURE_FIXED_WIDTH)
        self.exposure_spin.setToolTip("Exposure time (ms)")
        self.exposure_spin.setVisible(False)
        layout.addWidget(self.exposure_spin)

        self.gain_spin = ValueSpinBox(
            suffix="%",
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            decimals=1,
        )
        self.gain_spin.setFixedWidth(_GAIN_FIXED_WIDTH)
        self.gain_spin.setToolTip("Gain (%)")
        self.gain_spin.setVisible(False)
        layout.addWidget(self.gain_spin)

        self.power_spin = ValueSpinBox(
            suffix="%",
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            decimals=1,
        )
        self.power_spin.setFixedWidth(_POWER_FIXED_WIDTH)
        self.power_spin.setToolTip("Light source power (%)")
        self.power_spin.setVisible(False)
        layout.addWidget(self.power_spin)

        self.btn_color = QToolButton()
        self.btn_color.setFixedSize(_BTN_SIZE)
        self.btn_color.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_color.setToolTip("Channel color")
        layout.addWidget(self.btn_color)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove channel", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_remove)

        drag_icon = QLabel()
        drag_icon.setFixedSize(10, 16)
        if os.path.exists(_DRAG_HANDLE_PATH):
            drag_icon.setPixmap(
                QPixmap(_DRAG_HANDLE_PATH).scaled(
                    10, 16,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        self.checkbox.stateChanged.connect(
            lambda s: self.enabled_changed.emit(self.channel, bool(s))
        )
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.channel))
        self.btn_color.clicked.connect(self._on_color_clicked)

        for w in (self.name_edit, self.excitation_combo, self.emission_combo, self.exposure_spin, self.gain_spin, self.power_spin):
            w.installEventFilter(self)

        self._connect_signals()
        self.refresh()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.excitation_combo.currentIndexChanged.connect(self._on_excitation_changed)
        self.emission_combo.currentIndexChanged.connect(self._on_emission_changed)
        self.exposure_spin.editingFinished.connect(self._on_exposure_changed)
        self.gain_spin.editingFinished.connect(self._on_gain_changed)
        self.power_spin.editingFinished.connect(self._on_power_changed)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        child = self.childAt(event.pos())
        if child is None or child is self:
            self.row_clicked.emit(self.channel)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FocusIn:
            self.row_clicked.emit(self.channel)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_controls(self, block: bool) -> None:
        for w in (self.name_edit, self.excitation_combo, self.emission_combo,
                  self.exposure_spin, self.gain_spin, self.power_spin):
            w.blockSignals(block)

    def set_detail_controls_visible(self, visible: bool) -> None:
        """Show or hide the exposure/gain/power spinboxes (hidden by default)."""
        for w in (self.exposure_spin, self.gain_spin, self.power_spin):
            w.setVisible(visible)

    def _emission_index(self, value) -> int:
        """Return the combo index for the given emission value."""
        idx = self.emission_combo.findData(value)
        return idx if idx >= 0 else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._block_controls(True)
        self.name_edit.setText(self.channel.name)

        # Excitation
        self.excitation_combo.set_value(self.channel.excitation_wavelength)

        # Emission
        self.emission_combo.setCurrentIndex(self._emission_index(self.channel.emission_wavelength))

        # Exposure time: stored as seconds, show as ms
        self.exposure_spin.setValue(self.channel.exposure_time * _S_TO_MS)

        # Gain: stored as fraction, show as %
        gain = self.channel.gain if self.channel.gain is not None else 0.0
        self.gain_spin.setValue(gain * _FRAC_TO_PCT)

        # Power: stored as fraction, show as %
        power = self.channel.power if self.channel.power is not None else 0.0
        self.power_spin.setValue(power * _FRAC_TO_PCT)

        # Color icon
        color = self.channel.color if self.channel.color in AVAILABLE_COLORS else "gray"
        self.btn_color.setIcon(_make_color_icon(color))

        self._block_controls(False)

    # ------------------------------------------------------------------
    # Inline mutation handlers
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        text = self.name_edit.text().strip()
        if not text:
            self.name_edit.setText(self.channel.name)
            return
        if text == self.channel.name:
            return
        if self.name_validator is not None and not self.name_validator(text, self):
            self.name_edit.setText(self.channel.name)
            if self.name_rejected is not None:
                self.name_rejected(text)
            return
        self.channel.name = text
        self.channel_changed.emit(self.channel)

    def _on_excitation_changed(self) -> None:
        value = self.excitation_combo.value()
        if value is None or value == self.channel.excitation_wavelength:
            return
        self.channel.excitation_wavelength = value
        self.channel_field_changed.emit(self.channel, "excitation_wavelength", value)
        self.channel_changed.emit(self.channel)

    def _on_emission_changed(self) -> None:
        value = self.emission_combo.value()
        if value == self.channel.emission_wavelength:
            return
        self.channel.emission_wavelength = value
        self.channel_field_changed.emit(self.channel, "emission_wavelength", value)
        self.channel_changed.emit(self.channel)

    def _on_exposure_changed(self) -> None:
        value_s = self.exposure_spin.value() * _MS_TO_S
        if value_s == self.channel.exposure_time:
            return
        self.channel.exposure_time = value_s
        self.channel_field_changed.emit(self.channel, "exposure_time", value_s)
        self.channel_changed.emit(self.channel)

    def _on_gain_changed(self) -> None:
        value = self.gain_spin.value() * _PCT_TO_FRAC
        if value == self.channel.gain:
            return
        self.channel.gain = value
        self.channel_field_changed.emit(self.channel, "gain", value)
        self.channel_changed.emit(self.channel)

    def _on_power_changed(self) -> None:
        value = self.power_spin.value() * _PCT_TO_FRAC
        if value == self.channel.power:
            return
        self.channel.power = value
        self.channel_field_changed.emit(self.channel, "power", value)
        self.channel_changed.emit(self.channel)

    def _on_color_clicked(self) -> None:
        menu = QMenu(self)
        current = self.channel.color if self.channel.color in AVAILABLE_COLORS else "gray"
        for color in AVAILABLE_COLORS:
            action = QAction(_make_color_icon(color), color.capitalize(), menu)
            action.setData(color)
            if color == current:
                action.setCheckable(True)
                action.setChecked(True)
            menu.addAction(action)
        chosen = menu.exec_(self.btn_color.mapToGlobal(self.btn_color.rect().bottomLeft()))
        if chosen is None:
            return
        new_color = chosen.data()
        self.channel.color = new_color
        self.btn_color.setIcon(_make_color_icon(new_color))
        self.channel_field_changed.emit(self.channel, "color", new_color)
        self.channel_changed.emit(self.channel)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

class _ChannelListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)
    add_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self.checkbox_all = QCheckBox("Channel")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        self.checkbox_all.setVisible(False)
        layout.addWidget(self.checkbox_all, 1)

        lbl_channel = QLabel("Channel")
        lbl_channel.setStyleSheet("font-weight: bold; background: transparent;")
        lbl_channel.setMinimumWidth(_NAME_MIN_WIDTH)
        layout.addWidget(lbl_channel, 1)

        self._detail_labels = []
        for label_text, fixed_width, is_detail in [
            ("Excitation", _EXCITATION_FIXED_WIDTH, False),
            ("Emission", _EMISSION_FIXED_WIDTH, False),
            ("Exposure", _EXPOSURE_FIXED_WIDTH, True),
            ("Gain", _GAIN_FIXED_WIDTH, True),
            ("Power", _POWER_FIXED_WIDTH, True),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; background: transparent;")
            lbl.setFixedWidth(fixed_width)
            layout.addWidget(lbl)
            if is_detail:
                lbl.setVisible(False)
                self._detail_labels.append(lbl)

        # spacer covers color btn; btn_add aligns with remove btn
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width() - 8)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add channel", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_add)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )
        self.btn_add.clicked.connect(self.add_clicked)

    def set_detail_columns_visible(self, visible: bool) -> None:
        """Show or hide the Exposure/Gain/Power column headers (hidden by default)."""
        for lbl in self._detail_labels:
            lbl.setVisible(visible)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class ChannelListWidget(QWidget):
    """Compact list-style channel settings widget.

    Drop-in replacement for ChannelSettingsWidget with inline-row controls
    following the MillingStageListWidget pattern.
    """

    settings_changed = pyqtSignal(list)          # List[ChannelSettings]
    channel_selected = pyqtSignal(object)        # ChannelSettings (selection changed)
    channel_added = pyqtSignal(object)           # ChannelSettings
    channel_removed = pyqtSignal(object)         # ChannelSettings
    channel_changed = pyqtSignal(object)         # ChannelSettings (inline field edit)
    channel_field_changed = pyqtSignal(object, str, object)  # ChannelSettings, field, value
    enabled_changed = pyqtSignal(list)           # List[ChannelSettings] (enabled only)
    order_changed = pyqtSignal(list)             # List[ChannelSettings] in new order

    MAX_CHANNELS = 6

    def __init__(
        self,
        fm: FluorescenceMicroscope,
        channel_settings: Union[ChannelSettings, List[ChannelSettings]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.fm = fm
        self._selected_channel: Optional[ChannelSettings] = None
        self._pending_channel: Optional[ChannelSettings] = None
        self._channel_change_pending: bool = False
        self._checkbox_states: dict = {}  # id(channel) -> bool

        if isinstance(channel_settings, ChannelSettings):
            channel_settings = [channel_settings]
        self._channel_list: List[ChannelSettings] = list(channel_settings)

        self._emission_items = list(fm.filter_set.available_emission_wavelengths)
        self._excitation_items = list(fm.filter_set.available_excitation_wavelengths)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _ChannelListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableChannelList(fm=self.fm)
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSpacing(0)
        self._list.setMinimumHeight(3 * _ROW_HEIGHT)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._empty_label = QLabel("No channels. Click + to add one.")
        self._empty_label.setStyleSheet("color: #606060; font-style: italic; padding: 12px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #ff9800; font-style: italic; padding: 2px 12px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self._status_label.setVisible(False))

        self._header.select_all_changed.connect(self._on_select_all)
        self._header.add_clicked.connect(self._on_add_channel)
        self._list.reordered.connect(self._on_reordered)

        for ch in self._channel_list:
            self._add_row(ch, enabled=True)
        self._update_empty_state()

    # ------------------------------------------------------------------
    # Public interface (matches ChannelSettingsWidget)
    # ------------------------------------------------------------------

    @property
    def channel_settings(self) -> List[ChannelSettings]:
        return [self._row(i).channel for i in range(self._list.count())]

    @channel_settings.setter
    def channel_settings(self, value: Union[ChannelSettings, List[ChannelSettings]]) -> None:
        if isinstance(value, ChannelSettings):
            value = [value]
        self._list.clear()
        self._selected_channel = None
        self._checkbox_states.clear()
        self._channel_list = list(value)
        for ch in self._channel_list:
            self._add_row(ch, enabled=True)
        self._update_empty_state()
        self._update_add_button()

    @property
    def selected_channel(self) -> Optional[ChannelSettings]:
        if self._selected_channel is not None:
            return self._selected_channel
        if self._list.count() > 0:
            return self._row(0).channel
        return None

    def add_channel(self, channel: Optional[ChannelSettings] = None) -> None:
        if self._list.count() >= self.MAX_CHANNELS:
            return
        if channel is None:
            count = self._list.count()
            source = self.selected_channel
            if source is not None:
                channel = deepcopy(source)
            else:
                channel = ChannelSettings(
                    excitation_wavelength=self._excitation_items[0] if self._excitation_items else 550.0,
                    emission_wavelength=self._emission_items[0] if self._emission_items else None,
                )
            existing = {self._row(i).channel.name for i in range(self._list.count())}
            channel.name = _unique_name(f"Channel-{count + 1:02d}", existing)
        row = self._add_row(channel, enabled=True)
        self._set_selected(channel)
        self.channel_added.emit(channel)
        self._emit_settings_changed()
        self._update_empty_state()
        self._update_add_button()

    def remove_channel_by_index(self, index: int) -> None:
        if self._list.count() <= 1:
            self._show_status_warning("Cannot remove the last channel.")
            return
        if index < 0 or index >= self._list.count():
            return
        row = self._row(index)
        channel = row.channel
        self._list.takeItem(index)
        if self._selected_channel is channel:
            self._selected_channel = None
        self._checkbox_states.pop(id(channel), None)
        self._sync_select_all()
        self._update_empty_state()
        self._update_add_button()
        self.channel_removed.emit(channel)
        self._emit_settings_changed()

    def remove_selected_channel(self) -> None:
        if self._selected_channel is None:
            return
        for i in range(self._list.count()):
            if self._row(i).channel is self._selected_channel:
                self.remove_channel_by_index(i)
                return

    def set_live_acquisition_controls(self, enabled: bool) -> None:
        """Stub for ChannelSettingsWidget compatibility — no separate live list."""
        pass

    # Backward-compat input properties (delegate to selected row widget)

    def _get_selected_row(self) -> Optional[ChannelRowWidget]:
        ch = self.selected_channel
        if ch is None:
            return None
        for i in range(self._list.count()):
            if self._row(i).channel is ch:
                return self._row(i)
        return None

    @property
    def channel_name_input(self):
        r = self._get_selected_row()
        return r.name_edit if r else None

    @property
    def excitation_wavelength_input(self):
        r = self._get_selected_row()
        return r.excitation_combo if r else None

    @property
    def emission_wavelength_input(self):
        r = self._get_selected_row()
        return r.emission_combo if r else None

    @property
    def exposure_time_input(self):
        r = self._get_selected_row()
        return r.exposure_spin if r else None

    @property
    def gain_input(self):
        r = self._get_selected_row()
        return r.gain_spin if r else None

    @property
    def color_input(self):
        r = self._get_selected_row()
        return r.btn_color if r else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_row(self, channel: ChannelSettings, enabled: bool = True) -> ChannelRowWidget:
        row = ChannelRowWidget(
            channel=channel,
            emission_items=self._emission_items,
            excitation_items=self._excitation_items,
            enabled=enabled,
        )
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, channel)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        self._connect_row(row)
        self._sync_select_all()
        return row

    def _connect_row(self, row: ChannelRowWidget) -> None:
        row.name_validator = self._is_name_available
        row.name_rejected = self._show_name_error
        row.enabled_changed.connect(self._on_enabled_changed)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.row_clicked.connect(self._on_row_clicked)
        row.channel_changed.connect(self._on_row_channel_changed, type=Qt.QueuedConnection)
        row.channel_field_changed.connect(self.channel_field_changed)

    def _row(self, i: int) -> ChannelRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def refresh_channel(self, channel: ChannelSettings) -> None:
        """Re-sync the row widget for the given channel from its current data."""
        for i in range(self._list.count()):
            row = self._row(i)
            if row is not None and row.channel is channel:
                row.refresh()
                return

    def _set_selected(self, channel: Optional[ChannelSettings]) -> None:
        self._selected_channel = channel
        for i in range(self._list.count()):
            if self._row(i).channel is channel:
                self._list.setCurrentRow(i)
                self.channel_selected.emit(channel)
                return
        self._list.setCurrentRow(-1)

    def select_channel(self, channel: ChannelSettings) -> None:
        """Programmatically select a channel and emit channel_selected."""
        self._set_selected(channel)

    def _update_empty_state(self) -> None:
        empty = self._list.count() == 0
        self._empty_label.setVisible(empty)

    def _update_add_button(self) -> None:
        at_limit = self._list.count() >= self.MAX_CHANNELS
        self._header.btn_add.setEnabled(not at_limit)
        self._header.btn_add.setToolTip(
            f"Maximum {self.MAX_CHANNELS} channels reached" if at_limit else "Add channel"
        )

    def _is_name_available(self, name: str, row: ChannelRowWidget) -> bool:
        existing = {self._row(i).channel.name for i in range(self._list.count()) if self._row(i) is not row}
        return name not in existing

    def _show_name_error(self, name: str) -> None:
        self._show_status_warning(f'Name "{name}" is already in use.')

    def _show_status_warning(self, message: str) -> None:
        self._status_label.setText(message)
        self._status_label.setVisible(True)
        self._status_timer.start(3000)

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit(self.channel_settings)

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_add_channel(self) -> None:
        self.add_channel()

    def _on_remove_clicked(self, channel: ChannelSettings) -> None:
        if self.fm.is_acquiring and channel is self._selected_channel:
            self._show_status_warning("Cannot remove the active channel during live acquisition.")
            return
        for i in range(self._list.count()):
            if self._row(i).channel is channel:
                self.remove_channel_by_index(i)
                return

    def _on_row_clicked(self, channel: ChannelSettings) -> None:
        if self.fm.is_acquiring and channel is not self._selected_channel:
            self._show_status_warning("Cannot change channel selection during live acquisition.")
            return
        self._set_selected(channel)

    def _on_row_channel_changed(self, channel: ChannelSettings) -> None:
        self._pending_channel = channel
        if self._channel_change_pending:
            return
        self._channel_change_pending = True
        QTimer.singleShot(0, self._flush_pending_channel_change)

    def _flush_pending_channel_change(self) -> None:
        self._channel_change_pending = False
        channel = self._pending_channel
        self._pending_channel = None
        if channel is not None:
            self.channel_changed.emit(channel)
            self._emit_settings_changed()

    def _on_enabled_changed(self, channel: ChannelSettings, enabled: bool) -> None:
        self._checkbox_states[id(channel)] = enabled
        self._sync_select_all()
        self.enabled_changed.emit([
            self._row(i).channel for i in range(self._list.count())
            if self._row(i).checkbox.isChecked()
        ])

    def _on_select_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
        self.enabled_changed.emit(
            [self._row(i).channel for i in range(self._list.count())] if checked else []
        )

    def _sync_select_all(self) -> None:
        count = self._list.count()
        if count == 0:
            return
        n_checked = sum(self._row(i).checkbox.isChecked() for i in range(count))
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if n_checked == 0:
            cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            cb.setCheckState(Qt.Checked)
        else:
            cb.setTristate(True)
            cb.setCheckState(Qt.PartiallyChecked)
        cb.blockSignals(False)

    def _on_reordered(self, channels: List[ChannelSettings]) -> None:
        """Rebuild row widgets after drag-and-drop (Qt clears itemWidget on move)."""
        for i, channel in enumerate(channels):
            item = self._list.item(i)
            if item is None:
                continue
            enabled = self._checkbox_states.get(id(channel), True)
            row = ChannelRowWidget(
                channel=channel,
                emission_items=self._emission_items,
                excitation_items=self._excitation_items,
                enabled=enabled,
            )
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.setItemWidget(item, row)
            self._connect_row(row)
        self._sync_select_all()
        self.order_changed.emit(channels)
