"""The part of an overview that is the same whatever is imaging it.

How many tiles, how much they overlap, what order they are visited in, which of them to
acquire, and how much ground that covers. None of it depends on whether the tiles come
from a beam or a camera -- the two tabs had all five, written twice, and had already
started to disagree about the fifth: overlap was a fraction on one side and a percentage
on the other, of the same stored number.

What is *not* here is everything that decides what a tile is: beam, dwell and resolution
on one side; channels, exposure and z on the other. Those stay with their tab.

The tile mask lives inside this panel rather than beside it. It is a grid property --
which of *these* tiles -- and it is the one setting that changes what a run does while
being invisible everywhere else in the column, so hiding it behind a disclosure would
make the thing that is already easy to forget harder to see. Folded away, the header
carries the count instead.
"""
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.structures import TileOrderStrategy
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox, ValueSpinBox
from fibsem.ui.widgets.tile_mask_widget import TileMaskWidget

_MUTED = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"

# The app's spinboxes carry -/+ buttons, which eat most of a narrow field and leave the
# value clipped. The fluorescence column found the same floor for the same reason.
_FIELD_MIN_WIDTH = 90

# How each strategy is named and explained. Here rather than in either tab: they
# describe `TileOrderStrategy`, which is not modality-specific, and the fluorescence
# column had the only copy.
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


class OverviewGridSettingsWidget(QWidget):
    """Grid shape, overlap, order and the per-tile mask, as one titled panel."""

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tile_fov: Optional[Tuple[float, float]] = None
        self._build()
        self._connect()
        self._refresh_derived()

    # ── construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        self.spin_rows = self._field(
            ValueSpinBox(minimum=1, maximum=100, step=1, decimals=0)
        )
        self.spin_rows.setValue(3)
        self.spin_cols = self._field(
            ValueSpinBox(minimum=1, maximum=100, step=1, decimals=0)
        )
        self.spin_cols.setValue(3)

        # Percent, not the stored fraction. `0.10` reads as a coefficient; `10 %` reads
        # as what it is. The two tabs disagreed about this, of one stored number.
        self.spin_overlap = self._field(
            ValueSpinBox(suffix=" %", minimum=0.0, maximum=90.0, step=5.0, decimals=0)
        )

        self.combo_tile_order = self._field(ValueComboBox(
            items=list(TileOrderStrategy),
            format_fn=lambda s: TILE_ORDER_LABELS.get(s, s.value.title()),
        ))
        # What each strategy does belongs in a tooltip, not the label: the label is read
        # on every glance and the explanation once.
        for index in range(self.combo_tile_order.count()):
            strategy = self.combo_tile_order.itemData(index)
            self.combo_tile_order.setItemData(
                index, TILE_ORDER_TOOLTIPS.get(strategy, ""), Qt.ToolTipRole
            )

        self.label_total_fov = QLabel("—")
        self.label_total_fov.setStyleSheet(_MUTED)

        # Rows and columns read as one setting and are always adjusted together, so they
        # share a line -- which is also what gives each of them room.
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_row.addWidget(self.spin_rows)
        times = QLabel("×")
        times.setStyleSheet(_MUTED)
        size_row.addWidget(times)
        size_row.addWidget(self.spin_cols)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.addRow("Rows × Columns", size_row)
        form.addRow("Overlap", self.spin_overlap)
        form.addRow("Tile order", self.combo_tile_order)
        form.addRow("Total FOV", self.label_total_fov)

        self.tile_mask = TileMaskWidget(rows=3, cols=3)

        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(6, 6, 6, 6)
        body.setSpacing(8)
        body.addLayout(form)
        body.addWidget(self.tile_mask)

        # Shown in the header so a sparse run announces itself even folded away. Blank
        # at full coverage: a badge that is always there stops being read.
        self.label_sparse = QLabel("")
        self.label_sparse.setStyleSheet(_MUTED)

        self.panel = TitledPanel("Grid", content=content)
        self.panel.add_header_widget(self.label_sparse)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.panel)

    @staticmethod
    def _field(widget: QWidget) -> QWidget:
        widget.setMinimumWidth(_FIELD_MIN_WIDTH)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        install_wheel_blocker(widget)
        return widget

    def _connect(self) -> None:
        self.spin_rows.valueChanged.connect(self._on_grid_size_changed)
        self.spin_cols.valueChanged.connect(self._on_grid_size_changed)
        self.spin_overlap.valueChanged.connect(self._on_changed)
        self.combo_tile_order.currentIndexChanged.connect(self._on_changed)
        self.tile_mask.changed.connect(self._on_changed)

    # ── reactions ────────────────────────────────────────────────────────

    def _on_grid_size_changed(self) -> None:
        """The mask is positional, so it has to follow the grid it describes.

        **Resizing remaps the mask** -- tiles that still exist keep their state, tiles
        that come into existence start enabled. That is the fluorescence tab's answer.
        The FIB/SEM tab's `OverviewAcquisitionSettingsWidget` answered the opposite,
        deliberately: it *drops* a mask whose shape no longer matches, on the grounds
        that growing has to invent whether the new tiles are in or out, and shrinking
        throws away a choice.

        Remapping wins here because of what resizing now is. Dragging a grid edge on the
        canvas resizes it on **every motion event**, so "drop the mask on any resize"
        means nudging a 3x3 to a 4x3 silently discards a selection built by hand. Having
        to invent one row of enabled tiles is the smaller surprise, and it is what the
        other tab already does -- one behaviour, rather than the same gesture meaning
        two things depending on which tab you learned it on.
        """
        self.tile_mask.set_grid_size(self.rows, self.cols)
        self._on_changed()

    def _on_changed(self) -> None:
        self._refresh_derived()
        self.changed.emit()

    def _refresh_derived(self) -> None:
        enabled, total = self.tile_mask.n_enabled, self.rows * self.cols
        self.label_sparse.setText("" if enabled == total else f"{enabled}/{total} tiles")

        if self._tile_fov is None:
            self.label_total_fov.setText("—")
            return
        fov_x, fov_y = self._tile_fov
        width, height = self.total_fov
        sym = constants.MICRON_SYMBOL
        self.label_total_fov.setText(
            f"{width * constants.SI_TO_MICRO:.0f} × "
            f"{height * constants.SI_TO_MICRO:.0f} {sym}"
        )

    # ── public API ───────────────────────────────────────────────────────

    @property
    def rows(self) -> int:
        return int(self.spin_rows.value())

    @property
    def cols(self) -> int:
        return int(self.spin_cols.value())

    @property
    def overlap(self) -> float:
        """The stored fraction, not the percent shown."""
        return self.spin_overlap.value() / 100.0

    @property
    def tile_order(self) -> TileOrderStrategy:
        return self.combo_tile_order.value()

    @property
    def mask(self) -> Optional[List[List[bool]]]:
        """The mask, or None when every tile is enabled — what the settings mean by
        "acquire everything", so there are not two spellings of it in saved files."""
        return self.tile_mask.mask

    @property
    def total_fov(self) -> Tuple[float, float]:
        """Ground covered, in metres. Zero until a tile field of view is known.

        The same arithmetic as `OverviewAcquisitionSettings.total_fov_x/y`, which is why
        this can live here: overlap and grid shape decide it, and a tile's size is the
        only modality-specific part, which the host supplies.
        """
        if self._tile_fov is None:
            return 0.0, 0.0
        fov_x, fov_y = self._tile_fov
        overlap = self.overlap
        return (
            ((self.cols - 1) * (1 - overlap) + 1) * fov_x,
            ((self.rows - 1) * (1 - overlap) + 1) * fov_y,
        )

    def set_tile_fov(self, fov_x: float, fov_y: float) -> None:
        """Tell the panel how big one tile is, so it can report the total."""
        self._tile_fov = (float(fov_x), float(fov_y))
        self._refresh_derived()

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Both dimensions as one change.

        Setting the two spin boxes one after the other emits twice and passes through a
        size nobody asked for -- the new row count against the old column count. That is
        invisible when a spin box is nudged and constant when an edge is dragged on the
        canvas, which emits on every motion event.
        """
        rows, cols = int(rows), int(cols)
        if (rows, cols) == (self.rows, self.cols):
            return
        for spinbox in (self.spin_rows, self.spin_cols):
            spinbox.blockSignals(True)
        try:
            self.spin_rows.setValue(rows)
            self.spin_cols.setValue(cols)
        finally:
            for spinbox in (self.spin_rows, self.spin_cols):
                spinbox.blockSignals(False)
        self.tile_mask.set_grid_size(rows, cols)
        self._on_changed()

    def set_mask(self, mask: Optional[List[List[bool]]]) -> None:
        self.tile_mask.mask = mask
        self._refresh_derived()

    def apply(
        self,
        rows: int,
        cols: int,
        overlap: float,
        tile_order: TileOrderStrategy,
        mask: Optional[List[List[bool]]],
    ) -> None:
        """Load every value at once, without emitting on the way through."""
        widgets = (self.spin_rows, self.spin_cols, self.spin_overlap,
                   self.combo_tile_order)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.spin_rows.setValue(int(rows))
            self.spin_cols.setValue(int(cols))
            self.spin_overlap.setValue(overlap * 100.0)
            self.combo_tile_order.set_value(tile_order)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.tile_mask.set_grid_size(int(rows), int(cols))
        self.tile_mask.mask = mask
        self._refresh_derived()
