from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.workflow_estimate import estimate_queue
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    DISABLED_TEXT_COLOR,
    NEUTRAL_500,
    NEUTRAL_700,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.ui.widgets.preflight import format_clock, format_duration

# ── Colours ───────────────────────────────────────────────────────────────────
_DOT_COMPLETED = stylesheets.GREEN_COLOR
_DOT_ACTIVE = stylesheets.ORANGE_COLOR
_DOT_PENDING = NEUTRAL_700
_DOT_FAILED = stylesheets.RED_COLOR
_DOT_SKIPPED = NEUTRAL_500
# Slate, not amber: the old #e0a030 was indistinguishable from the active orange,
# so a cancelled row read as still running — especially when it was the last thing
# to run and there was no active row beside it to compare against. Still not red:
# a user abort is not an error.
_DOT_CANCELLED = "#6f7d8c"

_LINE_COLOR = "#3a3d42"

_SKIP_REASON_LABELS = {
    "not_required": "Not in required list",
    "failure": "Marked as failure",
    "missing_prereqs": "Prerequisites not met",
}
# Selection marks the row you are acting on; it drives nothing else, so it stays
# quiet. Painted rather than set as a stylesheet: a stylesheet set on the row
# cascades to its child labels, which do not span the row, so the fill came out
# as disconnected bands with a gap between label and subtitle.
#
# PRIMARY_COLOR is what selection means across the app — the lamella cards mark
# themselves with a 2px border in it, so the timeline agrees rather than
# inventing a second selection blue.
_SELECTED_BG = QColor(stylesheets.PRIMARY_COLOR)
_SELECTED_BG.setAlpha(30)
_SELECTED_BAR = QColor(stylesheets.PRIMARY_COLOR)
_SELECTED_BAR_W = 2
_LABEL_COLOR = stylesheets.GRAY_TEXT_COLOR
_SUBTITLE_COLOR = stylesheets.GRAY_SECONDARY_COLOR

_ADD_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; border-radius: 3px;"
    " color: #d1d2d4; font-size: 13px; padding: 3px 6px; }"
    "QToolButton:hover { background: #3a3d42; }"
    f"QToolButton:disabled {{ color: {DISABLED_TEXT_COLOR}; }}"
    "QToolButton::menu-indicator { image: none; }"
)

_ACTIONS_BTN = 20  # px — the hover-revealed "..." button
_ACTIONS_STYLE = (
    "QToolButton { background: transparent; border: none; border-radius: 3px; }"
    "QToolButton:hover { background: #3a3d42; }"
    "QToolButton::menu-indicator { image: none; }"
)

_DOT_SIZE = 12
_INNER_DOT_SIZE = 8
_LINE_W = 2
_LEFT_COL = 32  # px — fixed width for dot + connector column
_INNER_ROW_H = 22  # px — minimum height per inner step row

# The estimate column. Fixed width and right-aligned so the times hold one line down the
# panel however long the names beside them are — that column is the thing being scanned,
# and a ragged right edge would break it. The pre-flight dialog reserves 84px for the same
# column against a wider dialog; 78 is what fits here beside the actions button.
_TRAILING_W = 78


# ── Status mapping ───────────────────────────────────────────────────────────
def _queue_status_to_step_status(s) -> "StepStatus":
    """Map AutoLamellaTaskStatus → StepStatus for display."""
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    return {
        AutoLamellaTaskStatus.NotStarted: StepStatus.PENDING,
        AutoLamellaTaskStatus.InProgress: StepStatus.ACTIVE,
        AutoLamellaTaskStatus.Completed: StepStatus.COMPLETED,
        AutoLamellaTaskStatus.Failed: StepStatus.FAILED,
        AutoLamellaTaskStatus.Skipped: StepStatus.SKIPPED,
        AutoLamellaTaskStatus.Cancelled: StepStatus.CANCELLED,
        # Removed items are filtered out before they reach the timeline; mapped
        # anyway so any other caller gets the struck-through look, not a crash.
        AutoLamellaTaskStatus.Removed: StepStatus.SKIPPED,
    }.get(s, StepStatus.PENDING)  # unknown status must never crash the timeline


# ── Data model ────────────────────────────────────────────────────────────────
class StepStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


@dataclass
class TimelineStep:
    label: str
    status: StepStatus = StepStatus.PENDING
    subtitle: str = ""
    trailing: str = ""
    """The estimate column, pre-formatted.

    A display string like `subtitle`, not a number: what belongs here depends on the row
    state (an estimate, a countdown, an elapsed, an actual) and deciding that in the row
    would put the policy in the one place that has no idea what a queue is. Written by
    the caller as the workflow proceeds, and preserved across a sync for the same reason
    subtitles are.
    """


# ── Helpers ───────────────────────────────────────────────────────────────────
def _status_color(status: StepStatus) -> str:
    return {
        StepStatus.COMPLETED: _DOT_COMPLETED,
        StepStatus.ACTIVE: _DOT_ACTIVE,
        StepStatus.PENDING: _DOT_PENDING,
        StepStatus.FAILED: _DOT_FAILED,
        StepStatus.SKIPPED: _DOT_SKIPPED,
        StepStatus.CANCELLED: _DOT_CANCELLED,
    }.get(status, _DOT_PENDING)


def _trailing_style(status: StepStatus) -> str:
    """Style for the estimate column.

    Two shades only, and the split is "not yet" against everything else: muted while a
    row is still an estimate, normal once it is a real measured number.

    The running row is deliberately *not* a third colour. Its dot and its bold label
    already mark it, so a coloured number would be a third marker saying the same thing —
    and the active orange is a shade off the amber that was rejected for looking like a
    warning, which is exactly how it would read once the row outruns its estimate and the
    column starts reporting elapsed. Nothing here changes colour for that (FIB-666).

    `background: transparent` is not decoration: the app applies NAPARI_STYLE, whose bare
    `QWidget { background-color: ... }` rule reaches every descendant, so a label that
    sets only `color` paints a block of the surface colour — invisible against a panel of
    the same colour, and not invisible over the selection tint the row paints underneath.
    """
    muted = status in (StepStatus.PENDING, StepStatus.SKIPPED)
    color = _SUBTITLE_COLOR if muted else _LABEL_COLOR
    return f"background: transparent; border: none; color: {color}; font-size: 11px;"


# ── _DotWidget ────────────────────────────────────────────────────────────────
class _DotWidget(QWidget):
    """A solid coloured antialiased circle."""

    def __init__(self, color: str, size: int = _DOT_SIZE, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self.setFixedSize(size, size)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, self._size, self._size)


# ── _InnerStepRow ─────────────────────────────────────────────────────────────
class _InnerStepRow(QWidget):
    """One inner step embedded inside an active outer row."""

    def __init__(self, label: str, status: StepStatus, parent=None):
        super().__init__(parent)
        self._setup_ui(label, status)

    def _setup_ui(self, label: str, status: StepStatus) -> None:
        self.setMinimumHeight(_INNER_ROW_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(6)

        self._dot = _DotWidget(_status_color(status), size=_INNER_DOT_SIZE)
        root.addWidget(self._dot, 0, Qt.AlignVCenter)

        self._label = QLabel(label)
        self._update_label_style(status)
        root.addWidget(self._label, 1)

    def set_status(self, status: StepStatus) -> None:
        self._dot.set_color(_status_color(status))
        self._update_label_style(status)

    def _update_label_style(self, status: StepStatus) -> None:
        color = _LABEL_COLOR if status == StepStatus.ACTIVE else _SUBTITLE_COLOR
        self._label.setStyleSheet(f"color: {color}; font-size: 11px;")
        f = self._label.font()
        f.setBold(status == StepStatus.ACTIVE)
        f.setStrikeOut(status == StepStatus.SKIPPED)
        self._label.setFont(f)


# ── _OuterRow ─────────────────────────────────────────────────────────────────
class _OuterRow(QWidget):
    """One outer (lamella, task) row with an embedded collapsible inner step list."""

    clicked = pyqtSignal(int)
    menu_requested = pyqtSignal(int, QPoint)  # index, global position

    def __init__(self, step: TimelineStep, index: int, is_last: bool, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._actions_enabled = False
        self._inner_rows: List[_InnerStepRow] = []
        self._setup_ui(step, is_last)

    def set_index(self, index: int) -> None:
        """Update the index reported by :attr:`clicked`, after a reorder."""
        self._index = index

    def set_is_last(self, is_last: bool) -> None:
        """Show or hide the connector line — the last row has nothing to join to."""
        self._line.setVisible(not is_last)

    def _setup_ui(self, step: TimelineStep, is_last: bool) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 0, 8, 0)
        root.setSpacing(8)

        # ── Left column: dot + connector line ─────────────────────────────
        left = QWidget()
        left.setFixedWidth(_LEFT_COL)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins((_LEFT_COL - _DOT_SIZE) // 2, 4, 0, 0)
        left_layout.setSpacing(0)
        left_layout.setAlignment(Qt.AlignTop)

        self._dot = _DotWidget(_status_color(step.status))
        left_layout.addWidget(self._dot, 0, Qt.AlignHCenter)

        # Always built, shown or hidden by position: a row can become (or stop
        # being) the last one when the queue is reordered. A hidden widget is
        # empty to the layout, so this matches never creating it.
        self._line = QFrame()
        self._line.setFrameShape(QFrame.VLine)
        self._line.setFixedWidth(_LINE_W)
        self._line.setStyleSheet(f"background: {_LINE_COLOR}; border: none;")
        self._line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._line.setVisible(not is_last)
        left_layout.addWidget(self._line, 1, Qt.AlignHCenter)

        root.addWidget(left)

        # ── Right column: label + subtitle + inner container ──────────────
        right = QVBoxLayout()
        right.setSpacing(1)
        right.setContentsMargins(0, 4, 0, 4)

        # ElidedLabel, not QLabel: the rows sit in a QScrollArea that is
        # `setWidgetResizable(True)` with the horizontal scroll bar `AlwaysOff`, so the
        # contents widget can never be narrower than its widest row and anything past
        # the viewport is unreachable. A plain label reports its full text width as its
        # minimum, so one long lamella petname pushed the estimate column off *every*
        # row — measured at 280px, the first row's estimate ended 59px off screen.
        self._label = ElidedLabel(step.label)
        self._label.setStyleSheet(f"color: {_LABEL_COLOR};")
        if step.status == StepStatus.ACTIVE:
            f = self._label.font()
            f.setBold(True)
            self._label.setFont(f)
        right.addWidget(self._label)

        self._subtitle = ElidedLabel(step.subtitle)
        self._subtitle.setStyleSheet(f"color: {_SUBTITLE_COLOR}; font-size: 11px;")
        self._subtitle.setVisible(bool(step.subtitle))
        right.addWidget(self._subtitle)

        # Error label — shown on failed rows
        self._error_msg = ""
        self._error_label = QLabel()
        self._error_label.setStyleSheet(f"color: {_DOT_FAILED}; font-size: 11px;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        right.addWidget(self._error_label)

        # Inner steps — hidden until this row is active
        self._inner_container = QWidget()
        self._inner_layout = QVBoxLayout(self._inner_container)
        self._inner_layout.setContentsMargins(0, 4, 0, 4)
        self._inner_layout.setSpacing(2)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._inner_container.setVisible(False)
        right.addWidget(self._inner_container)

        root.addLayout(right, 1)

        # The estimate column. Held in a top-aligned holder rather than added straight to
        # the row so it lines up with the name rather than centring against a row whose
        # height changes with the subtitle, the error text and the inner steps.
        trailing_holder = QWidget()
        trailing_holder.setStyleSheet("background: transparent; border: none;")
        trailing_box = QVBoxLayout(trailing_holder)
        trailing_box.setContentsMargins(0, 4, 0, 0)
        trailing_box.setSpacing(0)
        self._trailing = QLabel(step.trailing)
        self._trailing.setFixedWidth(_TRAILING_W)
        self._trailing.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._trailing_status = step.status
        self._trailing.setStyleSheet(_trailing_style(step.status))
        trailing_box.addWidget(self._trailing)
        trailing_box.addStretch(1)
        root.addWidget(trailing_holder, 0, Qt.AlignTop)

        # Hover-revealed actions button. The timeline is dense enough that a
        # permanent button on every row would be noise, but right-click alone is
        # undiscoverable — so both, and the button only while the pointer is here.
        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(_ACTIONS_BTN, _ACTIONS_BTN)
        # A hidden widget is empty to the layout, so revealing this one on hover used to
        # shove the estimate column 28px left (the button plus the row's spacing) and
        # back again on the way out. Keeping its space reserved costs a gutter the
        # column would otherwise have to itself, and buys a column that holds still.
        policy = self._btn_actions.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self._btn_actions.setSizePolicy(policy)
        self._btn_actions.setStyleSheet(_ACTIONS_STYLE)
        self._btn_actions.setIcon(
            fibsem_icon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR)
        )
        self._btn_actions.setToolTip("Queue actions")
        self._btn_actions.setVisible(False)
        self._btn_actions.clicked.connect(self._on_actions_clicked)
        root.addWidget(self._btn_actions, 0, Qt.AlignTop)

    def set_trailing(self, text: str) -> None:
        """Write the estimate column alone.

        `refresh` redraws the dot, both labels, the font and the error text; the estimate
        column is rewritten far more often than any of those, so it gets its own way in.
        """
        self._trailing.setText(text)

    # ── Queue actions ─────────────────────────────────────────────────────
    def set_actions_enabled(self, enabled: bool) -> None:
        """Whether this row can offer queue actions at all.

        False when no run is live: there is no queue to act on, so the button
        is hidden rather than opening a menu with every entry greyed out. Which
        *individual* actions apply to this row is decided when the menu opens.
        """
        self._actions_enabled = enabled
        if not enabled:
            self._btn_actions.setVisible(False)

    def _on_actions_clicked(self) -> None:
        self.menu_requested.emit(
            self._index,
            self._btn_actions.mapToGlobal(self._btn_actions.rect().bottomLeft()),
        )

    def contextMenuEvent(self, event):
        if self._actions_enabled:
            self.menu_requested.emit(self._index, event.globalPos())

    def enterEvent(self, event):
        self._btn_actions.setVisible(self._actions_enabled)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._btn_actions.setVisible(False)
        super().leaveEvent(event)

    # ── Outer row refresh ─────────────────────────────────────────────────
    def refresh(self, step: TimelineStep) -> None:
        self._dot.set_color(_status_color(step.status))
        f = self._label.font()
        f.setBold(step.status == StepStatus.ACTIVE)
        f.setStrikeOut(step.status == StepStatus.SKIPPED)
        self._label.setFont(f)
        self._label.setText(step.label)
        self._trailing.setText(step.trailing)
        # setStyleSheet re-polishes the widget, which is far too expensive to pay on
        # every row of every sync for a colour that only moves when the status does.
        if step.status is not self._trailing_status:
            self._trailing_status = step.status
            self._trailing.setStyleSheet(_trailing_style(step.status))
        self._subtitle.setText(step.subtitle)
        self._subtitle.setVisible(bool(step.subtitle))
        # The timeline refreshes every row on every update, so an error has to
        # survive a refresh or it would vanish the moment the next task starts.
        # It belongs to a failed row, so a row that is no longer failed (a
        # re-run, say) drops it.
        if step.status is not StepStatus.FAILED:
            self._error_msg = ""
        self._error_label.setText(self._error_msg)
        self._error_label.setVisible(bool(self._error_msg))

    def set_error(self, msg: str) -> None:
        """Show an error message below the subtitle."""
        self._error_msg = msg
        self._error_label.setText(msg)
        self._error_label.setVisible(bool(msg))

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self.update()

    def paintEvent(self, event):
        if not self._selected:
            return
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        # One contiguous rect across the whole row, including the inner steps when
        # the active row is the selected one — they belong to it.
        p.fillRect(self.rect(), _SELECTED_BG)
        if _SELECTED_BAR_W:
            p.fillRect(0, 0, _SELECTED_BAR_W, self.height(), _SELECTED_BAR)

    # ── Inner step management ─────────────────────────────────────────────
    def set_inner_visible(self, visible: bool) -> None:
        self._inner_container.setVisible(visible)

    def add_inner_step(self, label: str, status: StepStatus) -> None:
        row = _InnerStepRow(label, status)
        self._inner_rows.append(row)
        self._inner_layout.addWidget(row)

    def update_last_inner_step(self, status: StepStatus) -> None:
        if self._inner_rows:
            self._inner_rows[-1].set_status(status)

    def clear_inner(self) -> None:
        for row in self._inner_rows:
            self._inner_layout.removeWidget(row)
            row.deleteLater()
        self._inner_rows.clear()

    # ── Interaction ───────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


# ── WorkflowTimelineWidget ────────────────────────────────────────────────────
class WorkflowTimelineWidget(QWidget):
    """Scrollable vertical list of outer workflow rows."""

    step_selected = pyqtSignal(int)
    row_menu_requested = pyqtSignal(int, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: List[TimelineStep] = []
        self._rows: List[_OuterRow] = []
        self._keys: List[str] = []
        self._selected_key: Optional[str] = None
        self._selected_index: Optional[int] = None
        self._actions_enabled = False
        self._setup_ui()

    def set_actions_enabled(self, enabled: bool) -> None:
        """Offer queue actions on every row, or on none of them."""
        self._actions_enabled = enabled
        for row in self._rows:
            row.set_actions_enabled(enabled)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Transparent rather than a hardcoded colour: this widget is embedded in
        # panels that set their own background (workflow_right_panel is #2b2d31),
        # and painting GRAY_BACKGROUND_COLOR here left the rows as a visibly
        # darker rectangle inside a lighter panel, with the header strip above
        # them a third shade again.
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._scroll.viewport().setStyleSheet("background: transparent;")

        self._contents = QWidget()
        self._contents.setStyleSheet("background: transparent;")
        self._contents_layout = QVBoxLayout(self._contents)
        self._contents_layout.setContentsMargins(0, 4, 0, 4)
        self._contents_layout.setSpacing(0)

        self._scroll.setWidget(self._contents)
        root.addWidget(self._scroll)

    # ── Public API ────────────────────────────────────────────────────────
    def set_steps(self, steps: List[TimelineStep]) -> None:
        """Replace the timeline wholesale. Rows are keyed by position."""
        self.clear()
        self.sync_steps([str(i) for i in range(len(steps))], steps)

    def sync_steps(self, keys: List[str], steps: List[TimelineStep]) -> None:
        """Reconcile the rows against *keys*, reusing the widgets already built.

        Rows are matched by key, so a row keeps its identity — and with it its
        inner step list, error text and subtitle — across an insertion, a
        removal or a reorder.

        Only ``status`` and ``label`` are taken from *steps* for a row that
        already exists. Subtitles are written by the caller as the run
        progresses (elapsed time, completion time, skip reason) and would be
        wiped if every sync rebuilt them from scratch.
        """
        by_key: Dict[str, _OuterRow] = dict(zip(self._keys, self._rows))
        step_by_key: Dict[str, TimelineStep] = dict(zip(self._keys, self._steps))

        for key in self._keys:
            if key not in keys:
                row = by_key.pop(key)
                step_by_key.pop(key, None)
                self._contents_layout.removeWidget(row)
                # Unparent before deleting: deleteLater() does not run until the
                # event loop comes back round, and until it does the row keeps
                # its last geometry and paints over whatever moved up into it.
                row.setParent(None)
                row.deleteLater()

        new_steps: List[TimelineStep] = []
        new_rows: List[_OuterRow] = []
        for i, (key, step) in enumerate(zip(keys, steps)):
            row = by_key.get(key)
            if row is None:
                row = _OuterRow(step, i, is_last=False)
                row.clicked.connect(self._on_row_clicked)
                row.menu_requested.connect(self.row_menu_requested)
                row.set_actions_enabled(self._actions_enabled)
                new_steps.append(step)
            else:
                kept = step_by_key[key]
                kept.label = step.label
                kept.status = step.status
                new_steps.append(kept)
            new_rows.append(row)

        reordered = self._keys != keys
        self._keys = list(keys)
        self._steps = new_steps
        self._rows = new_rows

        if reordered:
            self._relayout()

        last = len(self._rows) - 1
        for i, row in enumerate(self._rows):
            row.set_index(i)
            row.set_is_last(i == last)
            row.refresh(self._steps[i])

        # Selection is held by key so it follows its row rather than staying
        # behind on whatever moved into that position.
        self._selected_index = (
            self._keys.index(self._selected_key)
            if self._selected_key in self._keys
            else None
        )
        for i, row in enumerate(self._rows):
            row.set_selected(i == self._selected_index)

    def _relayout(self) -> None:
        """Re-add every row to the layout in ``self._rows`` order.

        Taking a widget out of a layout does not delete or reparent it, and the
        rows go straight back in before the event loop runs again, so nothing
        the rows own (inner steps in particular) is disturbed.
        """
        while self._contents_layout.count():
            self._contents_layout.takeAt(0)
        for row in self._rows:
            self._contents_layout.addWidget(row)
        self._contents_layout.addStretch(1)

    def set_step_status(self, index: int, status: StepStatus) -> None:
        if 0 <= index < len(self._steps):
            self._steps[index].status = status
            self._rows[index].refresh(self._steps[index])

    def set_active_step(self, index: int) -> None:
        for i, step in enumerate(self._steps):
            if i < index:
                step.status = StepStatus.COMPLETED
            elif i == index:
                step.status = StepStatus.ACTIVE
            else:
                step.status = StepStatus.PENDING
            self._rows[i].refresh(step)

    def clear(self) -> None:
        self._steps.clear()
        for row in self._rows:
            self._contents_layout.removeWidget(row)
            row.setParent(None)  # see sync_steps: it paints until the loop runs
            row.deleteLater()
        self._rows.clear()
        self._keys.clear()
        item = self._contents_layout.takeAt(0)
        while item:
            item = self._contents_layout.takeAt(0)
        self._selected_index = None
        self._selected_key = None

    # ── Internal ──────────────────────────────────────────────────────────
    def _on_row_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._rows):
            return
        if self._selected_index is not None and self._selected_index < len(self._rows):
            self._rows[self._selected_index].set_selected(False)
        self._selected_index = index
        self._selected_key = self._keys[index] if index < len(self._keys) else None
        self._rows[index].set_selected(True)
        self.step_selected.emit(index)


# ── WorkflowProgressWidget ────────────────────────────────────────────────────
_HEADER_STYLE = (
    f"color: {stylesheets.GRAY_SECONDARY_COLOR};"
    "font-size: 11px; font-weight: bold; padding: 4px 8px 2px 8px;"
    "text-transform: uppercase; letter-spacing: 1px;"
)


class WorkflowProgressWidget(QWidget):
    """Two-level workflow progress display.

    Outer timeline: one row per (lamella × task) pair.
    Inner steps: revealed inline inside the currently active outer row,
                 hidden automatically when the task completes or fails.
    """

    # (action, WorkItem.id). The widget never mutates the queue itself — it has
    # no business knowing what a queue is — so the owner performs the action and
    # feeds the result back through refresh_queue().
    queue_action_requested = pyqtSignal(str, str)
    # run_next: True to add at the front of the pending region, False for the end.
    # What to add comes from the owner's own selection UI, not from here.
    add_to_queue_requested = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list = []  # visible WorkItems from the last queue snapshot
        # The active row is held by WorkItem id; _outer_index is derived from it
        # on every sync, so it survives rows being added, removed or reordered
        # above it.
        self._active_id: Optional[str] = None
        self._outer_index: int = -1
        self._inner_finished: bool = False
        self._active_start_time: Optional[float] = None
        # Per-(lamella, task) estimates and per-task schedule, pushed in by the owner.
        # Keyed by name pair rather than WorkItem.id: an item removed and re-added gets a
        # new id but is the same work.
        self._estimates: Dict[Tuple[str, str], float] = {}
        self._schedule: Dict[str, datetime] = {}
        # A supervised task's wait is not machine time, so it must not eat the estimate.
        # Only the countdown pauses; the elapsed on the row keeps running, because the
        # duration that replaces it on completion counts the wait too.
        self._waiting_for_user: bool = False
        self._paused_total: float = 0.0
        self._paused_since: Optional[float] = None
        # "A workflow is live" — the same question the actions and the Add button ask.
        self._actions_enabled: bool = False
        self._setup_ui()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header strip: the progress count, and the one control that acts on the
        # queue as a whole. It lives here rather than in the status bar because
        # that bar is already full during a supervised run (message, milling
        # progress, attention, supervised, Stop) and putting an additive button
        # next to Stop invites the wrong click.
        header_row = QHBoxLayout()
        # 2px + the button's own 6px padding = the 8px the header label is inset by,
        # so the two ends of the strip match. This used to reserve 20px for the
        # scroll bar column below, which only appears when the queue overflows -- so
        # on a short queue the button floated in an empty gutter. Nothing is lost by
        # dropping it: the reservation was there to keep the button's menu arrow off
        # the scroll bar's, and _ADD_BTN_STYLE hides that arrow (FIB-584).
        header_row.setContentsMargins(0, 0, 2, 0)
        header_row.setSpacing(4)

        self._header = QLabel("Workflow")
        self._header.setStyleSheet(_HEADER_STYLE)
        header_row.addWidget(self._header, 1)

        self._btn_add = QToolButton()
        self._btn_add.setStyleSheet(_ADD_BTN_STYLE)
        self._btn_add.setText("Add to Queue")
        self._btn_add.setIcon(
            fibsem_icon("mdi:plus", color=stylesheets.GRAY_ICON_COLOR)
        )
        self._btn_add.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_add.setVisible(False)
        # Nothing is selected until the owner says otherwise.
        self._btn_add.setEnabled(False)
        self._btn_add.setToolTip("Select a lamella and a task to add to the queue")
        # The whole button opens the menu, rather than a split button whose arrow
        # is a tiny target and draws its own raised section against a flat header.
        # Both destinations are then equally visible instead of one being hidden
        # behind an arrow most people never press.
        self._btn_add.setPopupMode(QToolButton.InstantPopup)

        add_menu = QMenu(self._btn_add)
        # A disabled action, not addSection(): the styled QMenu does not draw a
        # section's text, so the heading would silently vanish.
        heading = add_menu.addAction("Add to queue:")
        heading.setEnabled(False)
        add_menu.addSeparator()
        act_end = add_menu.addAction("Add to End of Queue")
        act_next = add_menu.addAction("Add to Run Next")
        act_end.triggered.connect(lambda: self.add_to_queue_requested.emit(False))
        act_next.triggered.connect(lambda: self.add_to_queue_requested.emit(True))
        self._btn_add.setMenu(add_menu)
        header_row.addWidget(self._btn_add)

        root.addLayout(header_row)

        # Remaining time and finish clock. A second line rather than a longer first one:
        # the count and the Add button already fill that row, and the two figures answer
        # different questions ("how much is left" / "when do I come back").
        self._summary = QLabel()
        self._summary.setStyleSheet(
            f"background: transparent; border: none; color: {_SUBTITLE_COLOR};"
            " font-size: 11px; padding: 0px 8px 5px 8px;"
        )
        self._summary.setVisible(False)
        root.addWidget(self._summary)

        self._outer = WorkflowTimelineWidget()
        self._outer.row_menu_requested.connect(self._show_row_menu)
        root.addWidget(self._outer, 1)

    # ── Public API ────────────────────────────────────────────────────────
    def set_actions_enabled(self, enabled: bool) -> None:
        """Offer queue actions on the rows and the header, or not at all.

        False when no run is live — the timeline stays on screen after a
        workflow finishes, and there is no queue left to edit. This is the same
        question the Add button asks, so it shares the gate.
        """
        self._actions_enabled = enabled
        self._outer.set_actions_enabled(enabled)
        self._btn_add.setVisible(enabled)
        self._update_summary()

    def set_add_enabled(self, enabled: bool, tooltip: str = "") -> None:
        """Whether there is a selection worth adding.

        Distinct from :meth:`set_actions_enabled`, which asks whether there is a
        live queue at all — the button is shown but greyed when a run is going
        and nothing is ticked, so it says what it needs rather than vanishing.
        """
        self._btn_add.setEnabled(enabled)
        if tooltip:
            self._btn_add.setToolTip(tooltip)

    def set_estimates(self, seconds_by_key: Dict[Tuple[str, str], float]) -> None:
        """Per-(lamella, task) durations, in seconds.

        Pushed in rather than computed here: the estimate comes from a lamella's task
        config, and this widget has no business knowing what an Experiment is. The owner
        recomputes on workflow start and when the queue is edited — not per status
        update, since `estimated_duration` walks every milling stage of every task.

        Anything not in the mapping simply has no estimate; the column stays empty for it
        rather than showing a number that was invented.
        """
        self._estimates = dict(seconds_by_key)
        self._refresh_trailing()
        self._update_summary()

    def set_schedule(self, schedule: Dict[str, datetime]) -> None:
        """`scheduled_at` per task name, for the tasks that have one.

        Without it the finish time would be short by the whole hold: a task scheduled for
        20:00 after work that ends at 15:40 puts four hours of dead time in the middle of
        the workflow, and `_wait_until_scheduled` really does sit there for them.
        """
        self._schedule = dict(schedule)
        self._update_summary()

    def set_waiting_for_user(self, waiting: bool) -> None:
        """Whether the running task is paused for a human.

        Freezes the countdown for as long as it lasts, and swaps the header's finish
        clock for the work remaining: the wait is unbounded, so any clock quoted through
        it would be a guess dressed as a fact. A scheduled hold needs no such treatment —
        it is bounded, so the clock stays true through it.
        """
        if waiting == self._waiting_for_user:
            return
        self._waiting_for_user = waiting
        now = time.time()
        if waiting:
            self._paused_since = now
        elif self._paused_since is not None:
            self._paused_total += now - self._paused_since
            self._paused_since = None
        self._update_elapsed()
        self._update_summary()

    def set_workflow(self, items: list) -> None:
        """Populate the outer timeline from queue items, starting fresh.

        Resets the active row, so this is the entry point for a new run. Use
        :meth:`refresh_queue` to re-sync a run already in progress.
        """
        self._active_id = None
        self._sync(items)

    def refresh_queue(self, items: list) -> None:
        """Re-sync the timeline with a queue snapshot, mid-run.

        Called when the queue itself is edited (added to, reordered, or an item
        removed) rather than when a task starts or finishes. The active row and
        its inner step list are preserved.
        """
        self._sync(items)

    def update_from_status(self, status: dict) -> None:
        """Advance the timeline from a workflow_update_signal payload.

        Reads queue_items snapshot to update all row statuses directly.
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
        from fibsem.applications.autolamella.workflows.tasks.status import (
            WorkflowStatusUpdate,
        )

        # One decode at the boundary: accepts the dict the signal still carries and the
        # typed report the producer will send (FIB-827).
        report = WorkflowStatusUpdate.from_payload(status)

        queue_items = report.queue_items
        if queue_items is None:
            # No snapshot in this payload -- leave the rows as they are. An *empty*
            # snapshot is a different thing and does fall through, to reconcile the
            # timeline down to zero rows.
            return

        task_status = report.status
        task_duration = report.task_duration

        # Reconcile rows with the snapshot; this also refreshes every status.
        self._sync(queue_items)

        active_id = next(
            (i.id for i in self._items if i.status == AutoLamellaTaskStatus.InProgress),
            None,
        )

        # New active row — show inner container, start elapsed timer. Compared by
        # id, so a reorder that moves the running task does not restart it.
        if active_id is not None and active_id != self._active_id:
            self._active_id = active_id
            self._outer_index = next(
                n for n, i in enumerate(self._items) if i.id == active_id
            )
            self._inner_finished = False
            self._active_start_time = (
                report.timestamp if report.timestamp is not None else time.time()
            )
            # The pause accounting belongs to one task, not to the workflow.
            self._waiting_for_user = False
            self._paused_total = 0.0
            self._paused_since = None
            self._elapsed_timer.start(1000)
            self._show_inner_at(self._outer_index)
            # Auto-scroll to the active row
            self._outer._scroll.ensureWidgetVisible(
                self._outer._rows[self._outer_index]
            )

        # Completion/failure/cancel — set subtitle, hide inner, resolve last inner step
        if task_status in (
            AutoLamellaTaskStatus.Completed,
            AutoLamellaTaskStatus.Failed,
            AutoLamellaTaskStatus.Cancelled,
        ):
            self._elapsed_timer.stop()
            self._active_start_time = None
            self._waiting_for_user = False
            self._paused_since = None
            idx = self._outer_index
            if 0 <= idx < len(self._outer._rows):
                task_name = self._items[idx].task_name
                self._set_completion_subtitle(
                    idx, task_duration, task_name, completed_at=report.timestamp
                )
                # Show error message on failed rows, before the refresh that renders it
                if task_status == AutoLamellaTaskStatus.Failed:
                    error_msg = report.error_message
                    if error_msg:
                        self._outer._rows[idx].set_error(error_msg)
                # Refresh the row so the updated subtitle is rendered
                self._outer._rows[idx].refresh(self._outer._steps[idx])
                self._outer._rows[idx].set_inner_visible(False)
            self._finish_inner(failed=(task_status == AutoLamellaTaskStatus.Failed))

        elif task_status == AutoLamellaTaskStatus.Skipped:
            skip_reason = report.skip_reason
            task_name = report.task_name
            item_name = report.item_name
            reason_str = _SKIP_REASON_LABELS.get(skip_reason, skip_reason or "Skipped")
            for i, item in enumerate(self._items):
                if item.lamella_name == item_name and item.task_name == task_name:
                    self._outer._steps[i].subtitle = reason_str
                    self._outer._rows[i].refresh(self._outer._steps[i])
                    break

    def _sync(self, items: list) -> None:
        """Reconcile the timeline against a queue snapshot, keyed by WorkItem id.

        Removed items are dropped from the view rather than struck through in
        place: the user asked for them to go, so leaving the row behind would
        overstate how much work is left and make a move past one look like it
        jumped two rows. They stay in the queue itself for the run summary.
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        self._items = [
            i for i in items if i.status is not AutoLamellaTaskStatus.Removed
        ]
        self._outer.sync_steps(
            [i.id for i in self._items],
            [
                TimelineStep(
                    label=i.lamella_name,
                    subtitle=i.task_name,
                    status=_queue_status_to_step_status(i.status),
                )
                for i in self._items
            ],
        )
        self._outer_index = next(
            (n for n, i in enumerate(self._items) if i.id == self._active_id), -1
        )
        self._update_header(self._items)
        self._refresh_trailing()
        self._update_summary()

    def update_step(self, step_name: str) -> None:
        """Mark the previous inner step completed and append a new ACTIVE one."""
        if not (0 <= self._outer_index < len(self._outer._rows)):
            return
        row = self._outer._rows[self._outer_index]
        row.update_last_inner_step(StepStatus.COMPLETED)
        row.add_inner_step(step_name, StepStatus.ACTIVE)

    def finish_current_step(self, failed: bool = False) -> None:
        """Resolve the last inner step as COMPLETED or FAILED.

        No-op if inner was already finished (prevents double-finish on failure).
        """
        if self._inner_finished:
            return
        self._finish_inner(failed)

    def clear_steps(self) -> None:
        """Clear inner steps for the active outer row and hide the container."""
        if 0 <= self._outer_index < len(self._outer._rows):
            row = self._outer._rows[self._outer_index]
            row.clear_inner()
            row.set_inner_visible(False)

    def clear(self) -> None:
        self._items.clear()
        self._active_id = None
        self._outer_index = -1
        self._inner_finished = False
        self._elapsed_timer.stop()
        self._active_start_time = None
        self._waiting_for_user = False
        self._paused_total = 0.0
        self._paused_since = None
        self._header.setText("Workflow")
        self._summary.setVisible(False)
        self._outer.clear()

    def set_active_outer(self, index: int) -> None:
        """Show the inner container for *index*, hide all others.

        Convenience method for test scripts that drive the widget directly
        rather than through update_from_status().
        """
        self._outer_index = index
        self._active_id = (
            self._items[index].id if 0 <= index < len(self._items) else None
        )
        self._show_inner_at(index)

    # ── Internal ──────────────────────────────────────────────────────────
    _FINISHED = (
        StepStatus.COMPLETED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
        StepStatus.CANCELLED,
    )

    def build_row_menu(self, index: int) -> Optional[QMenu]:
        """The queue-actions menu for a row, or None if the row is unknown.

        Enablement is worked out when the menu is built rather than cached on
        the row: the worker may have consumed the row since the last paint. That
        still leaves a race between opening the menu and clicking it, which is
        why the owner has to handle a refusal rather than assume success.

        Every row gets the same menu with the inapplicable entries greyed out —
        a row with no menu teaches nothing about why it has no menu.
        """
        steps = self._outer._steps
        if not (0 <= index < len(self._items)) or index >= len(steps):
            return None

        item_id = self._items[index].id
        pending = [i for i, s in enumerate(steps) if s.status is StepStatus.PENDING]
        rank = pending.index(index) if index in pending else -1
        is_pending = rank >= 0
        is_finished = steps[index].status in self._FINISHED
        is_active = steps[index].status is StepStatus.ACTIVE

        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        def add(
            action_id: str,
            label: str,
            icon: str,
            enabled: bool,
            colour: Optional[str] = None,
        ):
            act = menu.addAction(
                fibsem_icon(icon, color=colour or stylesheets.GRAY_ICON_COLOR), label
            )
            act.setEnabled(enabled)
            act.setData(action_id)
            # Explains why an entry is greyed, so only on the greyed ones — the
            # running row now has one entry that is not.
            if not enabled and is_active:
                act.setToolTip("This task is already running")
            act.triggered.connect(
                lambda _checked=False, a=action_id: self.queue_action_requested.emit(
                    a, item_id
                )
            )
            return act

        add("move_up", "Move up", "mdi:arrow-up", is_pending and rank > 0)
        add(
            "move_down",
            "Move down",
            "mdi:arrow-down",
            is_pending and rank < len(pending) - 1,
        )
        add("run_next", "Run next", "mdi:skip-next", is_pending and rank > 0)
        menu.addSeparator()
        add("remove", "Remove from queue", "mdi:trash-can-outline", is_pending)
        if is_finished:
            menu.addSeparator()
            add("run_again", "Run again", "mdi:refresh", True)
        if is_active:
            # Red where Remove is grey: Remove edits a list, this interrupts an
            # operation on the sample.
            menu.addSeparator()
            add(
                "stop_task",
                "Stop Task",
                "mdi:stop-circle-outline",
                True,
                colour=_DOT_FAILED,
            )
        return menu

    def _show_row_menu(self, index: int, pos: QPoint) -> None:
        menu = self.build_row_menu(index)
        if menu is None:
            return
        menu.exec_(pos)
        # exec_ returns only after the chosen action has already fired, so the
        # menu can go. Otherwise every right-click leaves one parented to us.
        menu.deleteLater()

    def _show_inner_at(self, index: int) -> None:
        """Clear and show the inner container at *index*; hide all others."""
        for i, row in enumerate(self._outer._rows):
            if i == index:
                row.clear_inner()
                row.set_inner_visible(True)
            else:
                row.set_inner_visible(False)

    def _finish_inner(self, failed: bool) -> None:
        if not (0 <= self._outer_index < len(self._outer._rows)):
            return
        self._inner_finished = True
        final = StepStatus.FAILED if failed else StepStatus.COMPLETED
        self._outer._rows[self._outer_index].update_last_inner_step(final)

    def _estimate_for(self, item) -> Optional[float]:
        """This item's estimate, or None if there is none to offer."""
        if item is None:
            return None
        return self._estimates.get((item.lamella_name, item.task_name))

    def _machine_elapsed(self, elapsed: float) -> float:
        """`elapsed` less any time the task spent waiting for a human.

        The countdown is against work the machine has to do, and a supervised pause is
        not that — left in, a long enough pause would spend the whole estimate with every
        bit of the task's real work still ahead of it.
        """
        paused = self._paused_total
        if self._paused_since is not None:
            paused += time.time() - self._paused_since
        return max(0.0, elapsed - paused)

    def _refresh_trailing(self) -> None:
        """Write the estimate into the column of every row still waiting to run.

        Pending rows only. A running row's column is its countdown and a finished row's
        is what it actually took; both have superseded the estimate, and rewriting it
        over them would undo that.
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        for i, item in enumerate(self._items):
            if i >= len(self._outer._steps):
                break
            if item.status is not AutoLamellaTaskStatus.NotStarted:
                continue
            seconds = self._estimate_for(item)
            step = self._outer._steps[i]
            step.trailing = format_duration(seconds) if seconds is not None else ""
            self._outer._rows[i].set_trailing(step.trailing)

    def _update_summary(self) -> None:
        """The header's second line: what is left, and when it lands."""
        if not self._items or not self._actions_enabled:
            self._summary.setVisible(False)
            return

        active_elapsed = None
        if self._active_start_time is not None:
            active_elapsed = self._machine_elapsed(
                time.time() - self._active_start_time
            )

        estimate = estimate_queue(
            self._items,
            self._estimate_for,
            schedule=self._schedule,
            active_elapsed=active_elapsed,
        )
        if estimate.remaining_seconds <= 0:
            self._summary.setVisible(False)
            return

        if self._waiting_for_user:
            # No clock: the wait is unbounded, so one would be a guess dressed as a fact.
            # The work still to do is knowable either way, and "of work" is what marks
            # that the number has narrowed to machine time.
            text = f"{format_duration(estimate.work_seconds)} of work left · waiting for you"
        else:
            text = (
                f"{format_duration(estimate.remaining_seconds)} left · "
                f"finishes {format_clock(estimate.expected_finish, datetime.now())}"
            )
        self._summary.setText(text)
        self._summary.setVisible(True)

    def _update_header(self, queue_items: list) -> None:
        """Update the header label with progress counts."""
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        total = len(queue_items)
        if total == 0:
            self._header.setText("Workflow")
            return
        # Everything with a terminal status counts as resolved, whether or not it
        # succeeded — otherwise the counter can never reach the total once anything
        # is cancelled. Removed items are already filtered out of the view, so they
        # leave both sides of the fraction rather than counting as work done.
        done = sum(
            1
            for i in queue_items
            if i.status
            in (
                AutoLamellaTaskStatus.Completed,
                AutoLamellaTaskStatus.Skipped,
                AutoLamellaTaskStatus.Cancelled,
            )
        )
        failed = sum(1 for i in queue_items if i.status == AutoLamellaTaskStatus.Failed)
        cancelled = sum(
            1 for i in queue_items if i.status == AutoLamellaTaskStatus.Cancelled
        )
        text = f"Workflow — {done}/{total}"
        # A run that ended with tasks abandoned should not read like one that did not.
        notes = [
            f"{n} {label}"
            for n, label in ((failed, "failed"), (cancelled, "cancelled"))
            if n
        ]
        if notes:
            text += f" ({', '.join(notes)})"
        self._header.setText(text)

    def _update_elapsed(self) -> None:
        """Tick handler — the active row's elapsed and countdown, and the summary."""
        if self._active_start_time is None:
            return
        if not (0 <= self._outer_index < len(self._outer._steps)):
            return
        step = self._outer._steps[self._outer_index]
        item = (
            self._items[self._outer_index]
            if self._outer_index < len(self._items)
            else None
        )
        task_name = item.task_name if item is not None else ""

        # Wall-clock, including any wait for a human: this is the number the recorded
        # duration replaces on completion, and that one counts the wait too. Only the
        # countdown below works in machine time.
        elapsed = time.time() - self._active_start_time
        estimate = self._estimate_for(item)

        if self._waiting_for_user:
            # The subtitle keeps its elapsed rather than repeating the column: the
            # header already says "waiting for you", and how long it has been sitting
            # there is the thing not said anywhere else.
            step.subtitle = f"{task_name} ({format_duration(elapsed)})"
            step.trailing = "waiting"
        elif estimate is None:
            step.subtitle = f"{task_name} ({format_duration(elapsed)})"
            step.trailing = ""
        elif self._machine_elapsed(elapsed) < estimate:
            step.subtitle = f"{task_name} ({format_duration(elapsed)})"
            step.trailing = (
                f"{format_duration(estimate - self._machine_elapsed(elapsed))} left"
            )
        else:
            # The estimate is spent, so the column stops predicting and reports instead —
            # elapsed, which is the same kind of number the row will show once it is
            # done. Nothing goes negative, nothing stalls, and nothing changes colour:
            # a task running long is variance, not an error.
            step.subtitle = task_name
            step.trailing = format_duration(elapsed)

        self._outer._rows[self._outer_index].refresh(step)
        self._update_summary()

    def _set_completion_subtitle(
        self,
        outer_idx: int,
        task_duration: Optional[float],
        task_name: str = "",
        completed_at: Optional[float] = None,
    ) -> None:
        """What a finished row says: where it got to, and what it cost.

        The duration goes in the estimate column rather than in brackets after the task
        name, so a finished row lines up with the pending ones and the whole workflow can
        be read down one column. It is wall-clock and still counts any supervised wait —
        `AutoLamellaTaskState.duration` is end minus start, and the experiment record
        agrees with it.
        """
        if not (0 <= outer_idx < len(self._outer._steps)):
            return
        time_str = ""
        if completed_at is not None:
            # `.lstrip("0")`, as `format_clock` does: the header now quotes a finish
            # time in the same panel, and `06:59PM` beside `7:13PM` reads as two
            # different clocks rather than one convention.
            stamp = datetime.fromtimestamp(completed_at).strftime(
                TIME_DISPLAY_AMPM_SHORT
            )
            time_str = f" · {stamp.lstrip('0')}"
        self._outer._steps[outer_idx].subtitle = f"{task_name}{time_str}"
        if task_duration is not None:
            self._outer._steps[outer_idx].trailing = format_duration(task_duration)


# ── Demo ──────────────────────────────────────────────────────────────────────
DEMO_STEPS = [
    TimelineStep("Setup session", StepStatus.COMPLETED),
    TimelineStep("Move to position", StepStatus.COMPLETED, "0.32 s"),
    TimelineStep("Acquire overview", StepStatus.COMPLETED, "1.1 s"),
    TimelineStep("Detect features", StepStatus.COMPLETED, "2.4 s"),
    TimelineStep("Mill rough trench", StepStatus.ACTIVE, "running…"),
    TimelineStep("Mill fine trench", StepStatus.PENDING),
    TimelineStep("Polish lamella", StepStatus.PENDING),
    TimelineStep("Acquire final image", StepStatus.PENDING),
]

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QWidget()
    win.setWindowTitle("Workflow Timeline — Demo")
    win.resize(320, 400)
    win.setStyleSheet(f"background: {stylesheets.GRAY_BACKGROUND_COLOR};")

    from PyQt5.QtWidgets import QVBoxLayout as VBox

    layout = VBox(win)
    layout.setContentsMargins(0, 0, 0, 0)

    timeline = WorkflowTimelineWidget()
    timeline.set_steps(DEMO_STEPS)
    timeline.step_selected.connect(lambda i: print(f"Selected step {i}"))

    layout.addWidget(timeline)
    win.show()
    sys.exit(app.exec_())
