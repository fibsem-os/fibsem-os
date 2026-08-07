"""Every setting an FM overview acquisition takes, in one self-contained widget.

Replaces `OverviewParametersWidget` for the rebuilt overview UI. Two differences that
matter:

* **No parent coupling.** The old widget type-hinted its parent as
  `FMAcquisitionWidget` and reached into it for channel names, so it could not be used
  anywhere else. This one is told what it needs.
* **Complete.** It covers `tile_order` and `tile_mask`, which the acquisition gained
  with sparse tilesets and tile ordering and which nothing could set from the UI.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)
from fibsem.structures import TileOrderStrategy
from fibsem.ui.fm.widgets.autofocus_widget import AutofocusWidget
from fibsem.ui.fm.widgets.tile_mask_widget import TileMaskWidget
from fibsem.ui.fm.widgets.z_parameters_widget import ZParametersWidget
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox
from fibsem.ui.tokens import (
    TEXT_MUTED_COLOR,
)

MUTED = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"

AUTOFOCUS_LABELS = {
    AutoFocusMode.NONE: "Don't auto-focus",
    AutoFocusMode.ONCE: "Once, at the start",
    AutoFocusMode.EACH_ROW: "On each new row",
    AutoFocusMode.EACH_TILE: "On every tile",
}

TILE_ORDER_LABELS = {
    TileOrderStrategy.TYPEWRITER: "Typewriter",
    TileOrderStrategy.SERPENTINE: "Serpentine",
    TileOrderStrategy.SPIRAL: "Spiral",
}

TILE_ORDER_TOOLTIPS = {
    TileOrderStrategy.TYPEWRITER: "Every row runs left to right.",
    TileOrderStrategy.SERPENTINE: "Rows alternate direction, so the stage does not "
                                  "travel back across the grid between them.",
    TileOrderStrategy.SPIRAL: "Outward from the centre tile, so the tiles nearest "
                              "the starting position are acquired first.",
}

SPINBOX_MIN_WIDTH = 92


class FMOverviewSettingsWidget(QWidget):
    """Grid, sampling and autofocus settings for a tileset acquisition."""

    changed = pyqtSignal()

    def __init__(
        self,
        parameters: Optional[OverviewParameters] = None,
        channel_settings: Optional[List[ChannelSettings]] = None,
        z_parameters: Optional[ZParameters] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._tile_fov: Optional[tuple] = None   # (fov_x, fov_y) metres, set externally
        # Owned here rather than by the parent, so every overview setting sits in one
        # widget and their order is this widget's to decide.
        self.z_widget = ZParametersWidget(z_parameters or ZParameters())
        # The sweep itself -- method, channel, and the coarse/fine passes -- is already
        # a solved widget backed by `AutoFocusSettings`. Only *when* to run it is an
        # overview concern, so that is all this widget adds.
        self.autofocus_widget = AutofocusWidget(list(channel_settings or []))
        self.autofocus_widget.set_pass_editing_enabled(True)
        self._init_ui()
        if parameters is not None:
            self.parameters = parameters
        self._refresh_derived()

    # ── construction ─────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(3)

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(3)

        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setDecimals(2)
        self.spin_overlap.setValue(0.1)

        self.combo_tile_order = ValueComboBox(
            items=list(TileOrderStrategy),
            format_fn=lambda s: TILE_ORDER_LABELS.get(s, s.value.title()),
        )
        # What each strategy does belongs in a tooltip, not in the label -- the label
        # is read on every glance and the explanation once.
        for index in range(self.combo_tile_order.count()):
            strategy = self.combo_tile_order.itemData(index)
            self.combo_tile_order.setItemData(
                index, TILE_ORDER_TOOLTIPS.get(strategy, ""), Qt.ToolTipRole
            )
        self.combo_tile_order.setToolTip(
            "\n".join(f"{TILE_ORDER_LABELS[s]} — {TILE_ORDER_TOOLTIPS[s]}"
                      for s in TileOrderStrategy)
        )

        # No label: this sits in the Z-Stack panel header, which already names it.
        self.check_zstack = QCheckBox()
        self.check_zstack.setToolTip("Acquire a z-stack at each tile")

        self.combo_autofocus_mode = ValueComboBox(
            items=list(AutoFocusMode),
            format_fn=lambda m: AUTOFOCUS_LABELS.get(m, m.name),
        )

        # The app's spinboxes carry -/+ buttons, which eat most of a narrow field and
        # leave the value clipped ("0" for an overlap of 0.10). Give them a floor.
        for widget in (self.spin_rows, self.spin_cols, self.spin_overlap,
                       self.combo_tile_order, self.combo_autofocus_mode):
            widget.setMinimumWidth(SPINBOX_MIN_WIDTH)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for widget in (self.spin_rows, self.spin_cols, self.spin_overlap):
            install_wheel_blocker(widget)

        # Rows and columns read as one setting and are always adjusted together, so
        # they share a line -- which is also what gives each of them room.
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_row.addWidget(self.spin_rows)
        times = QLabel("×")
        times.setStyleSheet(MUTED)
        size_row.addWidget(times)
        size_row.addWidget(self.spin_cols)

        grid_form = QFormLayout()
        grid_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        grid_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label_total_fov = QLabel("—")
        self.label_total_fov.setStyleSheet(MUTED)

        grid_form.addRow("Rows × Columns", size_row)
        grid_form.addRow("Overlap", self.spin_overlap)
        grid_form.addRow("Tile order", self.combo_tile_order)
        grid_form.addRow("Total FOV", self.label_total_fov)
        grid_content = QWidget()
        grid_content.setLayout(grid_form)
        self.grid_panel = TitledPanel("Grid", content=grid_content)

        self.tile_mask = TileMaskWidget(rows=3, cols=3)
        self.mask_panel = TitledPanel("Tiles to acquire", content=self.tile_mask)

        # "When to focus" is the overview's own setting; everything below it -- method,
        # channel, and the sweep passes -- belongs to the sweep and is delegated.
        focus_form = QFormLayout()
        focus_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        focus_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        focus_form.addRow("Auto-focus", self.combo_autofocus_mode)

        focus_layout = QVBoxLayout()
        focus_layout.setContentsMargins(6, 6, 6, 6)
        focus_layout.setSpacing(6)
        focus_layout.addLayout(focus_form)
        focus_layout.addWidget(self.autofocus_widget)
        focus_content = QWidget()
        focus_content.setLayout(focus_layout)
        self.focus_panel = TitledPanel("Focus", content=focus_content)

        # The enable checkbox goes in the panel header, not the body: it is the one
        # piece of z-stack state worth seeing when the panel is folded away, and the
        # parameters below it grey out when it is off rather than staying live and
        # implying they apply.
        zstack_layout = QVBoxLayout()
        zstack_layout.setContentsMargins(6, 6, 6, 6)
        zstack_layout.setSpacing(4)
        zstack_layout.addWidget(self.z_widget)
        zstack_content = QWidget()
        zstack_content.setLayout(zstack_layout)
        self.zstack_panel = TitledPanel("Z-Stack", content=zstack_content)
        self.zstack_panel.add_header_widget(self.check_zstack)

        self.label_summary = QLabel()
        self.label_summary.setStyleSheet(MUTED)
        self.label_summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.focus_panel)
        layout.addWidget(self.zstack_panel)
        layout.addWidget(self.grid_panel)
        # No stretch on any panel: a stretched panel keeps claiming vertical space when
        # it is collapsed, so folding it leaves a tall empty box with the title floating
        # in the middle of it. The trailing stretch below takes the slack instead -- kept
        # here rather than left to the host, so the panels stay put in any container.
        layout.addWidget(self.mask_panel)
        layout.addWidget(self.label_summary)
        layout.addStretch()

        # Grid and the tile mask are what you touch every run; focus and the z-stack are
        # set once and then left alone, so they start folded to keep the column short.
        self.focus_panel.collapse()
        self.zstack_panel.collapse()

        self.spin_rows.valueChanged.connect(self._on_grid_size_changed)
        self.spin_cols.valueChanged.connect(self._on_grid_size_changed)
        self.spin_overlap.valueChanged.connect(self._on_any_change)
        self.combo_tile_order.currentIndexChanged.connect(self._on_tile_order_changed)
        self.check_zstack.stateChanged.connect(self._on_zstack_toggled)
        self.combo_autofocus_mode.currentIndexChanged.connect(self._on_autofocus_changed)
        self.autofocus_widget.settings_changed.connect(self._on_any_change)
        self.tile_mask.changed.connect(self._on_any_change)

        self._on_autofocus_changed()
        self._on_zstack_toggled()

    # ── reactions ────────────────────────────────────────────────────────

    def _on_zstack_toggled(self) -> None:
        self.z_widget.setEnabled(self.check_zstack.isChecked())
        self._on_any_change()

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Set both grid dimensions as a single change.

        Setting the two spin boxes one after the other emits `changed` twice and
        passes through a size nobody asked for -- the new row count against the old
        column count. That is invisible when a spin box is nudged by hand, but an edge
        drag on the canvas does it on every motion event.

        Follows the `parameters` setter: block, apply, resize the mask explicitly
        because its handler is blocked with it, then notify once.
        """
        if (rows, cols) == (self.spin_rows.value(), self.spin_cols.value()):
            return

        for widget in (self.spin_rows, self.spin_cols, self.tile_mask):
            widget.blockSignals(True)
        try:
            self.spin_rows.setValue(rows)
            self.spin_cols.setValue(cols)
            self.tile_mask.set_grid_size(rows, cols)
        finally:
            for widget in (self.spin_rows, self.spin_cols, self.tile_mask):
                widget.blockSignals(False)

        self._on_any_change()

    def _on_grid_size_changed(self) -> None:
        self.tile_mask.set_grid_size(self.spin_rows.value(), self.spin_cols.value())
        self._on_any_change()

    def _on_tile_order_changed(self) -> None:
        self._warn_if_spiral_conflicts()
        self._on_any_change()

    def _on_autofocus_changed(self) -> None:
        mode = self.combo_autofocus_mode.value()
        self.autofocus_widget.setEnabled(mode is not AutoFocusMode.NONE)
        self._warn_if_spiral_conflicts()
        self._on_any_change()

    def _warn_if_spiral_conflicts(self) -> None:
        """A spiral revisits rows out of order, so the runner promotes EACH_ROW.

        Surfaced here rather than left as a log line, so the setting the user chose and
        the setting that will run are not quietly different things.
        """
        spiral = self.combo_tile_order.value() is TileOrderStrategy.SPIRAL
        each_row = self.combo_autofocus_mode.value() is AutoFocusMode.EACH_ROW
        self._spiral_conflict = spiral and each_row

    def _on_any_change(self) -> None:
        self._refresh_derived()
        self.changed.emit()

    # ── derived display ──────────────────────────────────────────────────

    def set_tile_fov(self, fov_x: float, fov_y: float) -> None:
        """Tell the widget one tile's field of view, so it can report total area."""
        self._tile_fov = (fov_x, fov_y)
        self._refresh_derived()

    def total_fov(self) -> Optional[tuple]:
        """Width and height of the whole mosaic in metres, or None if the tile FOV
        has not been supplied.

        The grid always spans the full rectangle regardless of the mask: skipped tiles
        keep their place, so the area covered is a property of rows/cols/overlap alone.
        """
        if self._tile_fov is None:
            return None
        fov_x, fov_y = self._tile_fov
        overlap = self.spin_overlap.value()
        return (
            ((self.spin_cols.value() - 1) * (1 - overlap) + 1) * fov_x,
            ((self.spin_rows.value() - 1) * (1 - overlap) + 1) * fov_y,
        )

    def _refresh_derived(self) -> None:
        total = self.total_fov()
        self.label_total_fov.setText(
            "—" if total is None
            else f"{total[0] * constants.SI_TO_MICRO:.0f} × "
                 f"{total[1] * constants.SI_TO_MICRO:.0f} µm"
        )

        # Warnings only -- the tile count is already on the mask panel, and saying it
        # twice just makes the second one look like a different number.
        parts = []
        if getattr(self, "_spiral_conflict", False):
            parts.append("per-row focus becomes per-tile for a spiral")
        if self._focus_without_passes():
            parts.append("auto-focus is on but every sweep pass is disabled")
        self.label_summary.setText(" · ".join(parts))

    def _focus_without_passes(self) -> bool:
        """Focusing scheduled, but nothing for it to sweep.

        `AutoFocusSettings.enabled` is False when every pass is unticked, so the run
        would schedule focusing and then do nothing -- two controls that each look set
        correctly, contradicting each other.
        """
        settings = self.autofocus_settings
        return settings is not None and not settings.enabled

    # ── value ────────────────────────────────────────────────────────────

    def set_channel_names(self, names: List[str]) -> None:
        current = self.combo_autofocus_channel.value()
        self.combo_autofocus_channel.blockSignals(True)
        self.combo_autofocus_channel.clear()
        self.combo_autofocus_channel.add_values(list(names))
        if current in names:
            self.combo_autofocus_channel.set_value(current)
        self.combo_autofocus_channel.blockSignals(False)

    def set_channel_settings(self, channels: List[ChannelSettings]) -> None:
        """Keep the focus-channel choices in step with the channels being acquired."""
        self.autofocus_widget.update_channels(list(channels))

    @property
    def autofocus_channel_name(self) -> Optional[str]:
        channel = self.autofocus_widget.get_selected_channel()
        return channel.name if channel is not None else None

    @property
    def autofocus_settings(self) -> Optional[AutoFocusSettings]:
        """The sweep to run, or None when autofocus is off.

        None rather than settings-behind-a-disabled-control, matching how the
        acquisition already reads `autofocus_settings=None`: one place decides whether
        focusing happens, and it is the mode.
        """
        if self.combo_autofocus_mode.value() is AutoFocusMode.NONE:
            return None
        return self.autofocus_widget.get_autofocus_settings()

    @property
    def z_parameters(self) -> Optional[ZParameters]:
        """The z-stack settings, or None when z-stacking is off.

        None rather than the values behind a disabled checkbox: the acquisition takes
        `zparams=None` to mean "no z-stack", so this hands back exactly what it wants
        and there is no second place deciding whether the setting applies.
        """
        if not self.check_zstack.isChecked():
            return None
        return self.z_widget.z_parameters

    @property
    def parameters(self) -> OverviewParameters:
        return OverviewParameters(
            rows=self.spin_rows.value(),
            cols=self.spin_cols.value(),
            overlap=self.spin_overlap.value(),
            use_zstack=self.check_zstack.isChecked(),
            autofocus_mode=self.combo_autofocus_mode.value(),
            tile_order=self.combo_tile_order.value(),
            tile_mask=self.tile_mask.mask,
        )

    @parameters.setter
    def parameters(self, value: OverviewParameters) -> None:
        for widget in (self.spin_rows, self.spin_cols, self.spin_overlap,
                       self.combo_tile_order, self.check_zstack,
                       self.combo_autofocus_mode, self.tile_mask):
            widget.blockSignals(True)
        try:
            self.spin_rows.setValue(value.rows)
            self.spin_cols.setValue(value.cols)
            self.spin_overlap.setValue(value.overlap)
            self.combo_tile_order.set_value(value.tile_order)
            self.check_zstack.setChecked(value.use_zstack)
            self.combo_autofocus_mode.set_value(value.autofocus_mode)
            self.tile_mask.set_grid_size(value.rows, value.cols)
            self.tile_mask.mask = value.tile_mask
        finally:
            for widget in (self.spin_rows, self.spin_cols, self.spin_overlap,
                           self.combo_tile_order, self.check_zstack,
                           self.combo_autofocus_mode, self.tile_mask):
                widget.blockSignals(False)
        # Both, because the setter blocked the signals that normally trigger them --
        # otherwise loading a z-stacked configuration leaves the z-parameters greyed
        # out while the checkbox says they apply.
        self._on_autofocus_changed()
        self._on_zstack_toggled()
