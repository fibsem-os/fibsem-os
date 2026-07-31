"""Floating options for the tile grid overlay.

Follows `FMLayersPanel`: a frameless tool window opened from a canvas toolbar button,
positioned under it. A separate top-level window rather than a child widget, for the
same reason -- native controls parented into the matplotlib canvas are forced to
repaint on every canvas redraw.

The settings here are all about *seeing* the grid, not about what will be acquired.
Nothing in this panel changes the acquisition; the tile selection lives in
`TileMaskWidget` and on the canvas itself.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.icon import fibsem_icon

# Ordered by how often an FM sample makes them useless: magenta first, because
# fluorescence channels are usually cyan/green/blue and rarely magenta.
GRID_COLORS = [
    ("Magenta", "#ff3ec8"),
    ("Yellow", "#ffd54f"),
    ("Cyan", "#00e5ff"),
    ("Green", "#69f0ae"),
    ("White", "#ffffff"),
]

def _color_icon(color: str, size: int = 12) -> QIcon:
    """A filled swatch for a combo entry, so the colour is picked by eye not by name."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


_PANEL_QSS = """
QFrame#tileGridPanel { background: #1e2027; border: 1px solid #3d4251; border-radius: 6px; }
QLabel { color: #d6d6d6; font-size: 11px; background: transparent; }
QLabel#panelTitle { color: #9aa0a6; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QCheckBox { color: #d6d6d6; font-size: 11px; background: transparent; }
QLabel#gridSummary { color: #868e93; font-size: 10px; background: transparent; }
"""


class TileGridOptionsPanel(QFrame):
    """Show/hide, colour and fill opacity for the tile grid overlay."""

    visibility_changed = pyqtSignal(bool)
    color_changed = pyqtSignal(str)
    fill_alpha_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("tileGridPanel")
        self.setStyleSheet(_PANEL_QSS)
        self.setFixedWidth(196)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        icon = QLabel()
        icon.setPixmap(fibsem_icon("mdi:grid", color="#9aa0a6").pixmap(QSize(14, 14)))
        title = QLabel("TILE GRID")
        title.setObjectName("panelTitle")
        header.addWidget(icon)
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        # Read-only. This panel does not change what gets acquired -- the grid's
        # parameters are owned by the settings column -- but when you are zoomed into
        # the canvas it is the obvious place to ask what you are looking at.
        self.label_summary = QLabel()
        self.label_summary.setObjectName("gridSummary")
        self.label_summary.setWordWrap(True)
        root.addWidget(self.label_summary)

        self.check_visible = QCheckBox("Show grid")
        self.check_visible.setChecked(True)
        self.check_visible.toggled.connect(self.visibility_changed)
        root.addWidget(self.check_visible)

        root.addWidget(QLabel("Colour"))
        self.combo_color = QComboBox()
        for name, color in GRID_COLORS:
            self.combo_color.addItem(_color_icon(color), name, color)
        self.combo_color.currentIndexChanged.connect(
            lambda _index: self.color_changed.emit(self.combo_color.currentData())
        )
        root.addWidget(self.combo_color)

        # Off by default -- outlines alone show the tile boundaries without hiding the
        # data. Turned up, the fill doubles as an overlap map.
        root.addWidget(QLabel("Fill"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 40)  # per-cent; above ~40 the data is unreadable
        self.slider_alpha.setValue(0)
        self.slider_alpha.valueChanged.connect(
            lambda value: self.fill_alpha_changed.emit(value / 100.0)
        )
        root.addWidget(self.slider_alpha)

    def set_summary(self, text: str) -> None:
        """What the grid currently describes, e.g. `3 x 3 - 10% overlap - 9/9 tiles`."""
        self.label_summary.setText(text)

    def set_color(self, color: str) -> None:
        """Reflect the current colour without emitting -- for syncing from the overlay."""
        index = self.combo_color.findData(color)
        if index < 0:
            return
        self.combo_color.blockSignals(True)
        self.combo_color.setCurrentIndex(index)
        self.combo_color.blockSignals(False)
