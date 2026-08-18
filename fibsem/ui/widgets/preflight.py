"""Shared furniture for pre-flight confirmation dialogs.

The house style, in one place: a meta line, count chips, a detail block of label/value
rows, and a primary action in the footer. `FMOverviewConfirmationDialog` established it,
`CoincidenceMillingConfirmationDialog` imported the pieces from that module rather than
copying them, and left a note saying to lift them out if a third appeared. The FIB/SEM
overview's dialog is the third.

Only the parts all three agree on live here. The footer does not: milling makes Cancel
the default button because a mill is irreversible, and an overview does not.

The *two overview* dialogs agree on more than that -- on everything except which facts
they list -- so they also share `OverviewPreflightDialog` below. The milling one is not a
third subclass of it and should not become one: it confirms a different kind of thing,
and the seven places it differs are seven constructor arguments a base class would carry
for one caller.
"""

from typing import Iterable, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


class OverviewPreflightDialog(QDialog):
    """What both overview tabs put in front of a run.

    The two tabs describe very different runs -- one lists channels, a z-stack and a
    focus sweep, the other a dwell time and where the grid was dragged to -- and they
    present them identically: the same title, the same width, the same
    acquire/skipped chips, the same footer, and the same refusal when no tile is
    selected. Everything except the facts, in other words, which is exactly the split
    a subclass hook makes.

    Subclasses fill in :meth:`_meta_line`, :meth:`_rows` and :meth:`_tile_counts`, then
    call :meth:`_init_ui` at the end of their own ``__init__`` -- last, because it reads
    all three, and none of them can answer before the subclass has stored what it is
    describing.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Start Overview Acquisition")
        self.setMinimumWidth(430)
        self.setStyleSheet(f"QDialog {{ background: {BACKGROUND}; }}")

    # ── what a subclass supplies ─────────────────────────────────────────

    def _meta_line(self) -> str:
        """The line that leads: what is about to happen, in one sentence."""
        raise NotImplementedError

    def _rows(self) -> List[Tuple[str, str]]:
        """Label/value pairs for the detail block, in reading order."""
        raise NotImplementedError

    def _tile_counts(self) -> Tuple[int, int]:
        """``(to acquire, total)``. Equal means every tile; the chips say the rest."""
        raise NotImplementedError

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        acquired, total = self._tile_counts()

        # No in-dialog heading: the window title already says "Start Overview
        # Acquisition", and repeating it 8px below costs a line and says nothing. The
        # meta line leads instead, so it carries normal text weight rather than muted.
        #
        # Tile counts only in the chips. A channel count was a third one once, and the
        # Channels row below already lists them by name -- it added a number, not
        # information.
        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(chip(f"{acquired} to acquire"))
        if acquired != total:
            chips.addWidget(chip(f"{total - acquired} skipped"))
        chips.addStretch()

        self.button_start = QPushButton("Start Acquisition")
        self.button_start.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_start.setMinimumHeight(30)
        self.button_start.clicked.connect(self.accept)
        button_cancel = QPushButton("Cancel")
        button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        button_cancel.setMinimumHeight(30)
        button_cancel.clicked.connect(self.reject)

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button_cancel)
        footer.addWidget(self.button_start)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(meta_label(self._meta_line()))
        layout.addLayout(chips)
        layout.addWidget(detail_block(self._rows()))
        layout.addLayout(footer)

        if acquired == 0:
            # Nothing to do: both runners refuse this anyway, so say why here rather
            # than letting it fail after the dialog is dismissed.
            self.button_start.setEnabled(False)
            self.button_start.setToolTip("No tiles are selected.")
