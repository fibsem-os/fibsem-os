"""Interactive per-tile enable mask for a sparse overview.

The acquisition data model is a per-tile boolean mask, deliberately: rectangular
region selection is then an affordance over it rather than a second concept in the
engine. This widget is that affordance -- click a tile to toggle it, drag to sweep a
rectangle, or use the presets.
"""

from typing import List, Optional

from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
)

# napari-dark palette, matching the rest of the app's chrome. Taken from the
# shared tokens rather than restated, so retuning the palette reaches this
# widget too -- these were copies of the same five values.
BACKGROUND = QColor(SURFACE_COLOR)
PANEL = QColor(PANEL_COLOR)
BORDER = QColor(BORDER_COLOR)
ENABLED_FILL = QColor(ACCENT_COLOR)
TEXT_MUTED = QColor(TEXT_MUTED_COLOR)

# No token: the enabled/disabled tile edges are this widget's own, and the two
# other widgets that draw tile state disagree with them (see tiled_overview_widget
# and tile_grid_overlay). Needs one shared decision, not three local ones.
ENABLED_EDGE = QColor("#7cc0ff")
DISABLED_EDGE = QColor("#4a4f5c")

MIN_CELL_PX = 10
MAX_CELL_PX = 44
GAP_PX = 2


class TileMaskGrid(QWidget):
    """The grid itself: rows x cols of togglable cells."""

    changed = pyqtSignal()

    def __init__(self, rows: int = 3, cols: int = 3, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows = rows
        self._cols = cols
        self._mask = [[True] * cols for _ in range(rows)]
        self._drag_value: Optional[bool] = None
        self._hover: Optional[tuple] = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(120, 120)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def mask(self) -> List[List[bool]]:
        """A copy, so callers cannot mutate the widget's state behind its back."""
        return [row[:] for row in self._mask]

    @mask.setter
    def mask(self, value: Optional[List[List[bool]]]) -> None:
        if value is None:
            self._mask = [[True] * self._cols for _ in range(self._rows)]
        else:
            self._mask = [
                [bool(value[r][c]) if r < len(value) and c < len(value[r]) else True
                 for c in range(self._cols)]
                for r in range(self._rows)
            ]
        self.update()
        self.changed.emit()

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Resize the grid, keeping the state of tiles that still exist.

        Nudging the row count should not silently discard a selection the user built
        by hand; tiles that come into existence start enabled.
        """
        if (rows, cols) == (self._rows, self._cols):
            return
        old = self._mask
        self._mask = [
            [old[r][c] if r < len(old) and c < len(old[r]) else True
             for c in range(cols)]
            for r in range(rows)
        ]
        self._rows, self._cols = rows, cols
        self.update()
        self.changed.emit()

    @property
    def n_enabled(self) -> int:
        return sum(v for row in self._mask for v in row)

    def set_all(self, value: bool) -> None:
        self._mask = [[value] * self._cols for _ in range(self._rows)]
        self.update()
        self.changed.emit()

    def invert(self) -> None:
        self._mask = [[not v for v in row] for row in self._mask]
        self.update()
        self.changed.emit()

    # ── geometry ─────────────────────────────────────────────────────────

    def _cell_size(self) -> int:
        avail_w = (self.width() - GAP_PX * (self._cols + 1)) / max(1, self._cols)
        avail_h = (self.height() - GAP_PX * (self._rows + 1)) / max(1, self._rows)
        return int(max(MIN_CELL_PX, min(MAX_CELL_PX, min(avail_w, avail_h))))

    def _origin(self, cell: int) -> tuple:
        grid_w = self._cols * cell + GAP_PX * (self._cols + 1)
        grid_h = self._rows * cell + GAP_PX * (self._rows + 1)
        return (self.width() - grid_w) // 2, (self.height() - grid_h) // 2

    def _cell_rect(self, row: int, col: int) -> QRect:
        cell = self._cell_size()
        ox, oy = self._origin(cell)
        return QRect(
            ox + GAP_PX + col * (cell + GAP_PX),
            oy + GAP_PX + row * (cell + GAP_PX),
            cell, cell,
        )

    def _cell_at(self, x: int, y: int) -> Optional[tuple]:
        for r in range(self._rows):
            for c in range(self._cols):
                if self._cell_rect(r, c).contains(x, y):
                    return r, c
        return None

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), PANEL)

        for r in range(self._rows):
            for c in range(self._cols):
                rect = self._cell_rect(r, c)
                enabled = self._mask[r][c]
                hovered = self._hover == (r, c)

                if enabled:
                    fill = ENABLED_FILL.lighter(115) if hovered else ENABLED_FILL
                    painter.fillRect(rect, fill)
                    painter.setPen(QPen(ENABLED_EDGE, 1))
                else:
                    if hovered:
                        painter.fillRect(rect, BORDER)
                    painter.setPen(QPen(DISABLED_EDGE, 1, Qt.DashLine))
                painter.drawRect(rect)

        painter.end()

    # ── interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        cell = self._cell_at(event.x(), event.y())
        if cell is None:
            return
        r, c = cell
        # A drag paints whatever the first cell became, so a sweep sets a rectangle
        # to one value instead of toggling each cell it crosses.
        self._drag_value = not self._mask[r][c]
        self._mask[r][c] = self._drag_value
        self.update()
        self.changed.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cell = self._cell_at(event.x(), event.y())
        if cell != self._hover:
            self._hover = cell
            self.update()
        if self._drag_value is None or cell is None:
            return
        r, c = cell
        if self._mask[r][c] != self._drag_value:
            self._mask[r][c] = self._drag_value
            self.update()
            self.changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_value = None

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()


class TileMaskWidget(QWidget):
    """The grid plus its presets and a live count."""

    changed = pyqtSignal()

    def __init__(self, rows: int = 3, cols: int = 3, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.grid = TileMaskGrid(rows, cols)
        self.grid.changed.connect(self._on_changed)
        self.grid.setMinimumHeight(130)

        # Row 0 is the top of the mosaic. Stated, because the whole reason the row
        # direction is worth stating is that it was inverted once and nothing noticed.
        hint = QLabel("Row 0 is the top of the mosaic · click or drag to toggle")
        hint.setStyleSheet(f"color: {TEXT_MUTED.name()}; font-size: 10px;")
        hint.setWordWrap(True)

        self.label_count = QLabel()
        self.label_count.setStyleSheet(f"color: {TEXT_MUTED.name()}; font-size: 11px;")

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for text, slot in (
            ("All", lambda: self.grid.set_all(True)),
            ("None", lambda: self.grid.set_all(False)),
            ("Invert", self.grid.invert),
        ):
            button = QPushButton(text)
            button.setFixedHeight(22)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        buttons.addWidget(self.label_count)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(hint)
        layout.addWidget(self.grid, stretch=1)
        layout.addLayout(buttons)

        self._refresh_count()

    def _on_changed(self) -> None:
        self._refresh_count()
        self.changed.emit()

    def _refresh_count(self) -> None:
        total = self.grid._rows * self.grid._cols
        n = self.grid.n_enabled
        self.label_count.setText(f"{n}/{total} tiles")

    # ── public API ───────────────────────────────────────────────────────

    def set_grid_size(self, rows: int, cols: int) -> None:
        self.grid.set_grid_size(rows, cols)
        self._refresh_count()

    @property
    def n_enabled(self) -> int:
        return self.grid.n_enabled

    @property
    def mask(self) -> Optional[List[List[bool]]]:
        """The mask, or None when every tile is enabled.

        None rather than an all-True mask, because that is what
        `OverviewParameters.tile_mask` means by "acquire everything" -- keeping the
        two spellings distinct would leave two ways to say the same thing in saved
        configurations.
        """
        mask = self.grid.mask
        if all(all(row) for row in mask):
            return None
        return mask

    @mask.setter
    def mask(self, value: Optional[List[List[bool]]]) -> None:
        self.grid.mask = value
