"""AutofocusWidget — dynamic multi-pass autofocus settings widget.

Each sweep pass is a row of [enabled checkbox, range spinbox, step spinbox,
remove button]. A ``+`` header button adds passes. Add/remove controls are
hidden by default and revealed via :meth:`set_pass_editing_enabled`.

Backed directly by ``AutoFocusSettings.passes`` (list of ``FocusSweepPass``).
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.autofunctions.autofocus import FocusSweepPass
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, FocusMethod
from fibsem.ui.widgets.custom_widgets import IconToolButton, ValueSpinBox

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

_M_TO_UM = 1e6
_UM_TO_M = 1e-6
_BTN_SIZE = 28

# fixed widths for column alignment between the pass rows and the
# method/channel controls below.
_CHECK_W = 28      # enabled checkbox column
_STEPS_W = 56      # right-hand "Steps" count column
_LABEL_W = 110     # "Focus Method" / "Focus Channel" label column

# spinbox configuration (values shown in µm, stored in metres)
_RANGE_CFG = dict(suffix="µm", minimum=1.0, maximum=2000.0, step=1.0, decimals=1)
_STEP_CFG = dict(suffix="µm", minimum=0.1, maximum=500.0, step=0.1, decimals=2)

# defaults for a freshly added / default coarse+fine pair (metres)
_DEFAULT_COARSE = dict(search_range=20e-6, step_size=5e-6)
_DEFAULT_FINE = dict(search_range=10e-6, step_size=1e-6)


# ---------------------------------------------------------------------------
# Pass row
# ---------------------------------------------------------------------------

class _PassRowWidget(QWidget):
    """One sweep pass: enabled checkbox + range + step + remove button."""

    changed = pyqtSignal()
    remove_clicked = pyqtSignal(object)  # _PassRowWidget

    def __init__(self, sweep_pass: FocusSweepPass, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sweep_pass = sweep_pass

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.checkbox = QCheckBox()
        self.checkbox.setFixedWidth(_CHECK_W)
        self.checkbox.setChecked(sweep_pass.enabled)
        self.checkbox.setToolTip("Enable this pass")
        layout.addWidget(self.checkbox)

        self.range_spin = ValueSpinBox(tooltip="Search range (±range/2)", **_RANGE_CFG)
        self.range_spin.setValue(sweep_pass.search_range * _M_TO_UM)
        layout.addWidget(self.range_spin, 1)

        self.step_spin = ValueSpinBox(tooltip="Step size between positions", **_STEP_CFG)
        self.step_spin.setValue(sweep_pass.step_size * _M_TO_UM)
        layout.addWidget(self.step_spin, 1)

        self.steps_label = QLabel()
        self.steps_label.setFixedWidth(_STEPS_W)
        self.steps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.steps_label.setStyleSheet("color: #888;")
        self.steps_label.setToolTip("Number of positions sampled in this pass")
        layout.addWidget(self.steps_label)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove pass", size=_BTN_SIZE
        )
        self.btn_remove.setVisible(False)
        layout.addWidget(self.btn_remove)

        self._update_summary()
        self._apply_enabled_state()

        # values are set above before connecting, so no spurious emissions
        self.checkbox.stateChanged.connect(self._on_changed)
        self.range_spin.valueChanged.connect(self._on_changed)
        self.step_spin.valueChanged.connect(self._on_changed)
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self))

    def _update_summary(self) -> None:
        self.steps_label.setText(str(self.sweep_pass.n_steps))

    def _apply_enabled_state(self) -> None:
        """Grey out the range/step spinboxes and step count when the pass is off."""
        enabled = self.checkbox.isChecked()
        self.range_spin.setEnabled(enabled)
        self.step_spin.setEnabled(enabled)
        self.steps_label.setEnabled(enabled)

    def _on_changed(self, *_) -> None:
        self.sweep_pass.enabled = self.checkbox.isChecked()
        self.sweep_pass.search_range = self.range_spin.value() * _UM_TO_M
        self.sweep_pass.step_size = self.step_spin.value() * _UM_TO_M
        self._apply_enabled_state()
        self._update_summary()
        self.changed.emit()

    def set_remove_visible(self, visible: bool) -> None:
        self.btn_remove.setVisible(visible)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

class _PassListHeader(QWidget):
    add_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl_enabled = QLabel("On")
        lbl_enabled.setFixedWidth(_CHECK_W)
        lbl_enabled.setStyleSheet("font-weight: bold;")
        lbl_enabled.setToolTip("Enable/disable each pass")
        layout.addWidget(lbl_enabled)

        lbl_range = QLabel("Range")
        lbl_range.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_range, 1)

        lbl_step = QLabel("Step")
        lbl_step.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_step, 1)

        lbl_steps = QLabel("Steps")
        lbl_steps.setFixedWidth(_STEPS_W)
        lbl_steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_steps.setStyleSheet("font-weight: bold;")
        lbl_steps.setToolTip("Number of positions sampled per pass")
        layout.addWidget(lbl_steps)

        self.btn_add = IconToolButton(icon="mdi:plus", tooltip="Add pass", size=_BTN_SIZE)
        self.btn_add.setVisible(False)
        layout.addWidget(self.btn_add)

        self.btn_add.clicked.connect(self.add_clicked)

    def set_add_visible(self, visible: bool) -> None:
        self.btn_add.setVisible(visible)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class AutofocusWidget(QWidget):
    """Dynamic multi-pass autofocus settings widget.

    Two passes (coarse + fine) by default. Add/remove controls are hidden until
    :meth:`set_pass_editing_enabled(True)` is called.
    """

    settings_changed = pyqtSignal(AutoFocusSettings)

    MAX_PASSES = 5

    def __init__(
        self,
        channel_settings: List[ChannelSettings],
        parent: Optional["FMAcquisitionWidget"] = None,
    ) -> None:
        super().__init__(parent)
        self.channel_settings = channel_settings
        self.parent_widget = parent
        self._editing_enabled = False
        self._rows: List[_PassRowWidget] = []

        self.autofocus_settings = AutoFocusSettings(
            passes=[
                FocusSweepPass(**_DEFAULT_COARSE),
                FocusSweepPass(**_DEFAULT_FINE),
            ],
            method=FocusMethod.TENENGRAD,
            channel_name=channel_settings[0].name if channel_settings else None,
        )

        self.initUI()
        self._rebuild_rows()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def initUI(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._header = _PassListHeader()
        self._header.add_clicked.connect(self._on_add_pass)
        layout.addWidget(self._header)

        self._rows_container = QVBoxLayout()
        self._rows_container.setContentsMargins(0, 0, 0, 0)
        self._rows_container.setSpacing(2)
        layout.addLayout(self._rows_container)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self.comboBox_method = QComboBox()
        self.comboBox_method.setToolTip("Focus measurement method to use")
        for method in FocusMethod:
            self.comboBox_method.addItem(method.value.title(), method)
        self._sync_method_combo()
        self.comboBox_method.currentIndexChanged.connect(self._on_method_changed)
        layout.addLayout(self._aligned_control_row("Focus Method", self.comboBox_method))

        self.comboBox_channel = QComboBox()
        self.comboBox_channel.setToolTip("Channel to use for autofocus")
        self._update_channel_list()
        self.comboBox_channel.currentIndexChanged.connect(self._on_channel_changed)
        layout.addLayout(self._aligned_control_row("Focus Channel", self.comboBox_channel))

    def _aligned_control_row(self, label_text: str, control: QWidget) -> QHBoxLayout:
        """Build a labelled control row: a fixed-width label followed by the
        control stretching to the full right edge of the widget."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        label = QLabel(label_text)
        label.setFixedWidth(_LABEL_W)
        row.addWidget(label)

        row.addWidget(control, 1)
        return row

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._rows_container.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows = []

    def _rebuild_rows(self) -> None:
        self._clear_rows()
        for sweep_pass in self.autofocus_settings.passes:
            row = _PassRowWidget(sweep_pass)
            row.changed.connect(self._on_row_changed)
            row.remove_clicked.connect(self._on_remove_pass)
            row.set_remove_visible(self._editing_enabled)
            self._rows_container.addWidget(row)
            self._rows.append(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pass_editing_enabled(self, enabled: bool) -> None:
        """Show or hide the add (+) and per-row remove controls (hidden by default)."""
        self._editing_enabled = enabled
        self._header.set_add_visible(enabled)
        for row in self._rows:
            row.set_remove_visible(enabled)

    def get_autofocus_settings(self) -> AutoFocusSettings:
        return self.autofocus_settings

    def set_autofocus_settings(self, settings: AutoFocusSettings) -> None:
        self.autofocus_settings = settings
        self._rebuild_rows()
        self._sync_method_combo()
        if settings.channel_name:
            self.set_selected_channel_by_name(settings.channel_name)

    def get_selected_channel(self) -> Optional[ChannelSettings]:
        if self.autofocus_settings.channel_name:
            for channel in self.channel_settings:
                if channel.name == self.autofocus_settings.channel_name:
                    return channel
        return None

    def set_selected_channel_by_name(self, channel_name: str) -> None:
        for i, channel in enumerate(self.channel_settings):
            if channel.name == channel_name:
                self.comboBox_channel.blockSignals(True)
                self.comboBox_channel.setCurrentIndex(i)
                self.comboBox_channel.blockSignals(False)
                self.autofocus_settings.channel_name = channel_name
                return

    def update_channels(self, channel_settings: List[ChannelSettings]) -> None:
        """Update the available channels, preserving the current selection if possible."""
        current = self.autofocus_settings.channel_name
        self.channel_settings = channel_settings
        self._update_channel_list()

        names = [c.name for c in channel_settings]
        if current in names:
            self.set_selected_channel_by_name(current)
        elif channel_settings:
            self.set_selected_channel_by_name(channel_settings[0].name)
            self._emit_settings_changed()
        else:
            self.autofocus_settings.channel_name = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_channel_list(self) -> None:
        self.comboBox_channel.blockSignals(True)
        self.comboBox_channel.clear()
        for channel in self.channel_settings:
            self.comboBox_channel.addItem(channel.name)
        if self.autofocus_settings.channel_name:
            idx = self.comboBox_channel.findText(self.autofocus_settings.channel_name)
            if idx >= 0:
                self.comboBox_channel.setCurrentIndex(idx)
        self.comboBox_channel.blockSignals(False)

    def _sync_method_combo(self) -> None:
        self.comboBox_method.blockSignals(True)
        for i in range(self.comboBox_method.count()):
            if self.comboBox_method.itemData(i) == self.autofocus_settings.method:
                self.comboBox_method.setCurrentIndex(i)
                break
        self.comboBox_method.blockSignals(False)

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit(self.autofocus_settings)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_changed(self) -> None:
        self._emit_settings_changed()

    def _on_add_pass(self) -> None:
        if len(self.autofocus_settings.passes) >= self.MAX_PASSES:
            return
        # copy the last pass as a sensible starting point
        src = self.autofocus_settings.passes[-1] if self.autofocus_settings.passes else FocusSweepPass(**_DEFAULT_FINE)
        self.autofocus_settings.passes.append(
            FocusSweepPass(search_range=src.search_range, step_size=src.step_size, enabled=True)
        )
        self._rebuild_rows()
        self._emit_settings_changed()

    def _on_remove_pass(self, row: _PassRowWidget) -> None:
        if len(self.autofocus_settings.passes) <= 1:
            return
        if row.sweep_pass in self.autofocus_settings.passes:
            self.autofocus_settings.passes.remove(row.sweep_pass)
        self._rebuild_rows()
        self._emit_settings_changed()

    def _on_method_changed(self, index: int) -> None:
        method = self.comboBox_method.itemData(index)
        if method is not None:
            self.autofocus_settings.method = method
            self._emit_settings_changed()

    def _on_channel_changed(self, index: int) -> None:
        if 0 <= index < len(self.channel_settings):
            self.autofocus_settings.channel_name = self.channel_settings[index].name
            self._emit_settings_changed()
