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

from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

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

from fibsem.constants import DATETIME_DISPLAY_AMPM, TIME_DISPLAY_AMPM_SHORT
from fibsem.ui import stylesheets, tokens
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.utils import format_time_remaining as utils_format_time_remaining

BACKGROUND = stylesheets.SURFACE_COLOR
PANEL = stylesheets.PANEL_COLOR
BORDER = stylesheets.BORDER_COLOR
TEXT = stylesheets.TEXT_COLOR
TEXT_STRONG = stylesheets.TEXT_STRONG_COLOR
TEXT_MUTED = stylesheets.TEXT_MUTED_COLOR
WARNING = tokens.SEMANTIC_WARNING_COLOR


def format_duration(seconds: float) -> str:
    """`2m 14s`, `1h 03m`, `45s` — whichever reads best at that magnitude.

    The padded form of `fibsem.utils.format_time_remaining`, which was the same three
    branches with the same thresholds written a second time (FIB-701). Kept as a name
    here because every dialog in this module asks for the same shape, and `pad=True` at
    a dozen call sites is a detail none of them should be restating.
    """
    return utils_format_time_remaining(seconds, pad=True)


def format_bytes(count: float) -> str:
    """`3.4 GB`, `880 MB`, `12 kB`. Decimal units, because a disk is sold in them.

    One decimal place above a gigabyte and none below: the number is an estimate of what
    a run will write, and quoting it to four figures would claim a precision the tile
    count does not have.
    """
    for unit, step in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if count >= step:
            value = count / step
            return f"{value:.1f} {unit}" if unit in ("TB", "GB") else f"{value:.0f} {unit}"
    return f"{int(count)} B"


def mosaic_pixels(rows: int, cols: int, overlap: float, width: int, height: int):
    """How big the stitched mosaic comes out, in pixels.

    The same step the runners use -- `fov * (1 - overlap)` per tile, plus one whole tile
    -- so the estimate cannot disagree with what is allocated. The mask does not enter:
    skipped tiles keep their place, so the rectangle is the full grid either way.
    """
    return (
        int(round(((cols - 1) * (1 - overlap) + 1) * width)),
        int(round(((rows - 1) * (1 - overlap) + 1) * height)),
    )


class PathValue(str):
    """A detail-block value that is a filesystem path, and should be shown as one.

    A `str` subclass rather than a flag on the row, so a caller that only wants the text
    -- a test, a log line, the row tuple itself -- is unaffected, and `detail_block`
    needs no second shape of row to carry the distinction.

    What it buys: elided from the *left* with the whole path in the tooltip. A
    destination is an experiment path plus a stamped run directory, which runs well past
    a dialog's width and wrapped to three lines; and the part that answers "am I writing
    where I meant to" is the tail, not the mount point (FIB-660).
    """


def format_clock(when: Optional[datetime], reference: Optional[datetime]) -> str:
    """A time the user can act on: `6:15AM` today, `Tomorrow, 6:15AM` overnight.

    The day is not decoration. An overnight workflow is exactly the case a pre-flight
    estimate exists for, and a bare clock time for one finishing the next morning is
    wrong by a day in the direction that matters -- `_wait_until_scheduled` dates its
    countdown for the same reason. But a date stamp reads as a filename where a person
    would just say "tomorrow", so the two days either side of today get the word instead.

    Anything further out keeps the dated form: "in 3 days" is arithmetic the reader has
    to do, and a workflow reaching that far is rare enough not to optimise for.

    Lives here rather than in the dialog that first needed it because the in-run timeline
    quotes the same finish time; two copies of this rule would drift.
    """
    if when is None:
        return "\u2014"
    # 6:15AM, not 06:15AM -- the leading zero is noise at this size.
    time_str = when.strftime(TIME_DISPLAY_AMPM_SHORT).lstrip("0")
    if reference is None:
        return time_str
    days = (when.date() - reference.date()).days
    if days == 0:
        return time_str
    if days == 1:
        return f"Tomorrow, {time_str}"
    if days == -1:
        return f"Yesterday, {time_str}"
    return when.strftime(DATETIME_DISPLAY_AMPM)


# Anything sitting on a panel has to say it is transparent. The app applies
# NAPARI_STYLE, whose bare `QWidget { background-color: ... }` rule reaches every
# descendant, so a label that only sets `color` paints the *surface* colour onto the
# darker panel behind it -- a lighter block around every word. Only visible with the app
# stylesheet loaded, which is why offscreen renders of it looked clean.
ON_PANEL = "background: transparent; border: none; "


def metric(caption_text: str, value_text: str, sub_text: str = "") -> QWidget:
    """A figure the dialog leads with, in a panel of its own.

    Args:
        sub_text: a smaller line under the value, for the figure's own context -- what a
            number *was* before the change being confirmed, most usefully. Omitted by
            the pre-flight dialog, whose two figures need none.
    """
    frame = QFrame()
    # Scoped to this frame by name. A bare `QFrame { ... }` selector also matches every
    # QFrame *inside* it, and `chip()` is one -- which repainted each chip with the
    # panel's fill on top of its own.
    frame.setObjectName("preflightMetric")
    frame.setStyleSheet(
        f"QFrame#preflightMetric {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 4px; }}"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 9, 12, 9)
    layout.setSpacing(2)

    caption = QLabel(caption_text)
    caption.setStyleSheet(ON_PANEL + f"color: {TEXT_MUTED}; font-size: 11px;")
    value = QLabel(value_text)
    value.setStyleSheet(ON_PANEL + f"color: {TEXT_STRONG}; font-size: 19px;")
    layout.addWidget(caption)
    layout.addWidget(value)
    if sub_text:
        sub = QLabel(sub_text)
        sub.setStyleSheet(ON_PANEL + f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(sub)
    return frame


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
        if isinstance(value_text, PathValue):
            value = ElidedLabel(str(value_text), mode=Qt.ElideLeft)
        else:
            value = QLabel(value_text)
            value.setWordWrap(True)
        value.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
        row.addWidget(label, alignment=Qt.AlignTop)
        row.addWidget(value, stretch=1)
        layout.addLayout(row)
    return frame


def warning_label(text: str) -> QLabel:
    """Why the dialog will not start this run, said where it can be read.

    Visible rather than only a tooltip on the disabled button. A dialog whose primary
    action is greyed out with no reason on screen is a dead end -- the reason is the
    whole point of refusing here rather than letting the runner refuse afterwards, and
    hovering to find it is not reading it.
    """
    label = QLabel(text)
    label.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
    label.setWordWrap(True)
    return label


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

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        unreachable: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        """
        Args:
            unreachable: (row, col) for each tile the stage cannot travel to, from
                `fibsem.imaging.tiling.unreachable_tiles`. Passed in rather than worked
                out here: neither of these dialogs takes a microscope, which is what
                lets them be built and tested on their own. None and empty both mean
                nothing to say -- a caller that could not run the check says nothing
                rather than claiming the grid is fine.
        """
        super().__init__(parent)
        self.unreachable: List[Tuple[int, int]] = list(unreachable or [])
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

    def _refusal(self, acquired: int) -> str:
        """Why this run cannot start, or "" if it can.

        Both cases are grids the runners refuse anyway -- one with "No tiles are
        selected", the other in `raise_if_outside_stage_limits`. Said here instead
        because there they are said after the dialog is dismissed, and the unreachable
        one only after a directory has been made and the stage has started moving.

        Neither is fixable *in* the dialog, which is Start and Cancel. The refusal is
        informational: it stops the run and sends you back to the canvas, where both
        the mask and where the grid sits are set.
        """
        if acquired == 0:
            return "No tiles are selected."
        if self.unreachable:
            count = len(self.unreachable)
            tiles = "tile is" if count == 1 else "tiles are"
            # A count, not the coordinates. The runner's error names them because it has
            # nowhere else to say it; here a grid can have dozens out of range, and a
            # list of them is not something you read and act on.
            return (
                f"{count} {tiles} outside the stage's travel. Turn "
                f"{'it' if count == 1 else 'them'} off, or move the grid, "
                "to acquire the rest."
            )
        return ""

    def _init_ui(self) -> None:
        acquired, total = self._tile_counts()
        refusal = self._refusal(acquired)

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
        if refusal:
            layout.addWidget(warning_label(refusal))
        layout.addLayout(footer)

        if refusal:
            self.button_start.setEnabled(False)
            self.button_start.setToolTip(refusal)
