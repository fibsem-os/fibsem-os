"""Floating options for the tile grid overlay.

Follows `FMLayersPanel`: a frameless tool window opened from a canvas toolbar button,
positioned under it. A separate top-level window rather than a child widget, for the
same reason -- native controls parented into the matplotlib canvas are forced to
repaint on every canvas redraw.

The settings here are about *seeing* the grid -- what is acquired is decided in the
settings column and on the canvas itself, where the tile selection lives. The one
exception is "Centre on stage", which undoes a drag of the grid: it is here because
this is where you come to ask what the grid currently describes, and the summary
directly above it is what tells you the grid is off-centre in the first place.
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
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    DISABLED_BG_COLOR,
    GRAY_PRIMARY_COLOR,
    PANEL_COLOR,
    ROW_ALT_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    WHITE_ICON_COLOR,
)

# Ordered by how often an FM sample makes them useless: magenta first, because
# fluorescence channels are usually cyan/green/blue and rarely magenta.
GRID_COLORS = [
    ("Magenta", "#ff3ec8"),
    ("Yellow", "#ffd54f"),
    ("Cyan", "#00e5ff"),
    ("Green", "#69f0ae"),
    ("White", WHITE_ICON_COLOR),
]

def _color_icon(color: str, size: int = 12) -> QIcon:
    """A filled swatch for a combo entry, so the colour is picked by eye not by name."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


_PANEL_QSS = f"""
QFrame#tileGridPanel {{ background: {PANEL_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 6px; }}
QLabel {{ color: {TEXT_COLOR}; font-size: 11px; background: transparent; }}
QLabel#panelTitle {{ color: #9aa0a6; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
QCheckBox {{ color: {TEXT_COLOR}; font-size: 11px; background: transparent; }}
QLabel#gridSummary {{ color: {TEXT_MUTED_COLOR}; font-size: 10px; background: transparent; }}
QPushButton#centreButton {{
    background: {ROW_ALT_COLOR}; color: {TEXT_COLOR}; font-size: 11px;
    border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 4px 8px;
}}
QPushButton#centreButton:hover:enabled {{ background: #333744; }}
QPushButton#centreButton:disabled {{ color: {GRAY_PRIMARY_COLOR}; border-color: {DISABLED_BG_COLOR}; }}
"""


class TileGridOptionsPanel(QFrame):
    """Show/hide, colour and fill opacity for the tile grid overlay, and re-centre it."""

    visibility_changed = pyqtSignal(bool)
    color_changed = pyqtSignal(str)
    fill_alpha_changed = pyqtSignal(float)
    centre_requested = pyqtSignal()

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

        # The way back from dragging the grid. Directly under the summary, which is
        # what reports the offset -- disabled until there is an offset to undo, so it
        # neither invites a click that does nothing nor strands the state a drag set.
        self.button_centre = QPushButton("Centre on stage")
        self.button_centre.setObjectName("centreButton")
        self.button_centre.setToolTip("Plan the overview around the stage position again")
        self.button_centre.setEnabled(False)
        self.button_centre.clicked.connect(self.centre_requested)
        root.addWidget(self.button_centre)

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
        """What the grid currently describes, e.g. `3 x 3 - 10% overlap - 9/9 tiles`.

        The minimum height is set from the label's own `heightForWidth` because nothing
        else will. A word-wrapped QLabel's height depends on the width it is given, and
        that dependency does not reach this panel's own size hint -- so as the summary
        grew (dragging the grid adds a line saying how far it sits from the stage, and
        that line wraps) the label kept a height computed for fewer lines and clipped
        the text top and bottom, the overflow split evenly because a QLabel centres its
        text vertically (FIB-510).

        Measured rather than reasoned: at 196px wide the label reported height 36 for
        text needing 48 -- exactly one line short.
        """
        self.label_summary.setText(text)
        label = self.label_summary
        width = label.width() or (self.width() - 24)  # 24 = the layout's l/r margins
        label.setMinimumHeight(label.heightForWidth(width))

    def set_centre_enabled(self, enabled: bool) -> None:
        """Whether the grid is off the stage position, and so worth re-centring."""
        self.button_centre.setEnabled(bool(enabled))

    def set_color(self, color: str) -> None:
        """Reflect the current colour without emitting -- for syncing from the overlay."""
        index = self.combo_color.findData(color)
        if index < 0:
            return
        self.combo_color.blockSignals(True)
        self.combo_color.setCurrentIndex(index)
        self.combo_color.blockSignals(False)
