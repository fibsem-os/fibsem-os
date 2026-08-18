"""Shared furniture for pre-flight confirmation dialogs.

The house style, in one place: a meta line, count chips, a detail block of label/value
rows, and a primary action in the footer. `FMOverviewConfirmationDialog` established it,
`CoincidenceMillingConfirmationDialog` imported the pieces from that module rather than
copying them, and left a note saying to lift them out if a third appeared. The FIB/SEM
overview's dialog is the third.

Only the parts all three agree on live here. The footer does not: milling makes Cancel
the default button because a mill is irreversible, and an overview does not.
"""

from typing import Iterable, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets

BACKGROUND = stylesheets.SURFACE_COLOR
PANEL = stylesheets.PANEL_COLOR
BORDER = stylesheets.BORDER_COLOR
TEXT = stylesheets.TEXT_COLOR
TEXT_STRONG = stylesheets.TEXT_STRONG_COLOR
TEXT_MUTED = stylesheets.TEXT_MUTED_COLOR


def format_duration(seconds: float) -> str:
    """`2m 14s`, `1h 03m`, `45s` — whichever reads best at that magnitude.

    Not `fibsem.utils.format_duration`, which carries two decimal places: that is the
    right shape for a milling estimate quoted to the second, and the wrong one for a
    figure that is already approximate.
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def chip(text: str) -> QWidget:
    """Plain pill label. No status dot: these are counts, not states, and a colour
    that does not encode anything reads as though it does."""
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 10px; }}"
    )
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 3, 10, 3)
    layout.setSpacing(0)

    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
    layout.addWidget(label)
    return frame


def detail_block(
    rows: Iterable[Tuple[str, str]],
    label_width: int = 96,
    wrap_labels: bool = False,
) -> QFrame:
    """The label/value block the three dialogs all show.

    Args:
        rows: (label, value) pairs, in reading order.
        label_width: the left column's width. Fixed rather than laid out, so the values
            line up down the block instead of stepping in and out with each label.
        wrap_labels: whether a long label wraps rather than clipping. Wanted where the
            labels are user-supplied (milling stage names); a truncated name in a
            pre-flight check is worse than two lines.
    """
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 4px; }}"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)
    for label_text, value_text in rows:
        row = QHBoxLayout()
        row.setSpacing(12)
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; border: none;")
        label.setFixedWidth(label_width)
        label.setWordWrap(wrap_labels)
        value = QLabel(value_text)
        value.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
        value.setWordWrap(True)
        row.addWidget(label, alignment=Qt.AlignTop)
        row.addWidget(value, stretch=1)
        layout.addLayout(row)
    return frame


def meta_label(text: str) -> QLabel:
    """The line that leads the dialog: what is about to happen, in one sentence.

    Normal text weight rather than muted -- there is no heading above it, because the
    window title already says what this is and repeating it 8px below costs a line.
    """
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT_STRONG}; font-size: 12px;")
    label.setWordWrap(True)
    return label
