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

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.fm.structures import AutoFocusMode, OverviewParameters
from fibsem.structures import TileOrderStrategy
from fibsem.ui.fm.widgets.tile_mask_widget import TileMaskWidget
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import ValueComboBox

MUTED = "color: #868e93; font-size: 11px;"

AUTOFOCUS_LABELS = {
    AutoFocusMode.NONE: "Don't auto-focus",
    AutoFocusMode.ONCE: "Once, at the start",
    AutoFocusMode.EACH_ROW: "On each new row",
    AutoFocusMode.EACH_TILE: "On every tile",
}

TILE_ORDER_LABELS = {
    TileOrderStrategy.TYPEWRITER: "Typewriter (rows left to right)",
    TileOrderStrategy.SERPENTINE: "Serpentine (alternating rows)",
    TileOrderStrategy.SPIRAL: "Spiral (outward from centre)",
}


class FMOverviewSettingsWidget(QWidget):
    """Grid, sampling and autofocus settings for a tileset acquisition."""

    changed = pyqtSignal()

    def __init__(
        self,
        parameters: Optional[OverviewParameters] = None,
        channel_names: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._tile_fov: Optional[tuple] = None   # (fov_x, fov_y) metres, set externally
        self._init_ui()
        self.set_channel_names(channel_names or [])
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

        self.check_zstack = QCheckBox("Acquire a z-stack at each tile")

        self.combo_autofocus_mode = ValueComboBox(
            items=list(AutoFocusMode),
            format_fn=lambda m: AUTOFOCUS_LABELS.get(m, m.name),
        )
        self.combo_autofocus_channel = ValueComboBox()

        for widget in (self.spin_rows, self.spin_cols, self.spin_overlap):
            install_wheel_blocker(widget)

        grid_form = QFormLayout()
        grid_form.addRow("Rows", self.spin_rows)
        grid_form.addRow("Columns", self.spin_cols)
        grid_form.addRow("Overlap", self.spin_overlap)
        grid_form.addRow("Tile order", self.combo_tile_order)
        grid_box = QGroupBox("Grid")
        grid_box.setLayout(grid_form)

        self.tile_mask = TileMaskWidget(rows=3, cols=3)
        mask_layout = QVBoxLayout()
        mask_layout.addWidget(self.tile_mask)
        mask_box = QGroupBox("Tiles to acquire")
        mask_box.setLayout(mask_layout)

        focus_form = QFormLayout()
        focus_form.addRow(self.check_zstack)
        focus_form.addRow("Auto-focus", self.combo_autofocus_mode)
        focus_form.addRow("Focus channel", self.combo_autofocus_channel)
        focus_box = QGroupBox("Focus")
        focus_box.setLayout(focus_form)

        self.label_summary = QLabel()
        self.label_summary.setStyleSheet(MUTED)
        self.label_summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(grid_box)
        layout.addWidget(mask_box, stretch=1)
        layout.addWidget(focus_box)
        layout.addWidget(self.label_summary)

        self.spin_rows.valueChanged.connect(self._on_grid_size_changed)
        self.spin_cols.valueChanged.connect(self._on_grid_size_changed)
        self.spin_overlap.valueChanged.connect(self._on_any_change)
        self.combo_tile_order.currentIndexChanged.connect(self._on_tile_order_changed)
        self.check_zstack.stateChanged.connect(self._on_any_change)
        self.combo_autofocus_mode.currentIndexChanged.connect(self._on_autofocus_changed)
        self.combo_autofocus_channel.currentIndexChanged.connect(self._on_any_change)
        self.tile_mask.changed.connect(self._on_any_change)

        self._on_autofocus_changed()

    # ── reactions ────────────────────────────────────────────────────────

    def _on_grid_size_changed(self) -> None:
        self.tile_mask.set_grid_size(self.spin_rows.value(), self.spin_cols.value())
        self._on_any_change()

    def _on_tile_order_changed(self) -> None:
        self._warn_if_spiral_conflicts()
        self._on_any_change()

    def _on_autofocus_changed(self) -> None:
        mode = self.combo_autofocus_mode.value()
        self.combo_autofocus_channel.setEnabled(mode is not AutoFocusMode.NONE)
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

    def _refresh_derived(self) -> None:
        rows, cols = self.spin_rows.value(), self.spin_cols.value()
        overlap = self.spin_overlap.value()
        n_enabled = self.tile_mask.n_enabled

        parts = [f"{n_enabled} of {rows * cols} tiles"]
        if self._tile_fov is not None:
            fov_x, fov_y = self._tile_fov
            width = ((cols - 1) * (1 - overlap) + 1) * fov_x * constants.SI_TO_MICRO
            height = ((rows - 1) * (1 - overlap) + 1) * fov_y * constants.SI_TO_MICRO
            parts.append(f"{width:.0f} × {height:.0f} µm")
        if getattr(self, "_spiral_conflict", False):
            parts.append("per-row focus becomes per-tile for a spiral")
        self.label_summary.setText(" · ".join(parts))

    # ── value ────────────────────────────────────────────────────────────

    def set_channel_names(self, names: List[str]) -> None:
        current = self.combo_autofocus_channel.value()
        self.combo_autofocus_channel.blockSignals(True)
        self.combo_autofocus_channel.clear()
        self.combo_autofocus_channel.add_values(list(names))
        if current in names:
            self.combo_autofocus_channel.set_value(current)
        self.combo_autofocus_channel.blockSignals(False)

    @property
    def autofocus_channel_name(self) -> Optional[str]:
        return self.combo_autofocus_channel.value()

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
        self._on_autofocus_changed()
