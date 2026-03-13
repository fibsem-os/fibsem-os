from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets

# ── Colours ───────────────────────────────────────────────────────────────────
_DOT_COMPLETED  = stylesheets.GREEN_COLOR
_DOT_ACTIVE     = "#ff9800"
_DOT_PENDING    = "#606060"
_DOT_FAILED     = "#99121F"
_DOT_SKIPPED    = "#9e9e9e"

_LINE_COLOR     = "#3a3d42"
_SELECTED_BG    = "#2d3f5c"
_LABEL_COLOR    = stylesheets.GRAY_TEXT_COLOR
_SUBTITLE_COLOR = stylesheets.GRAY_SECONDARY_COLOR

_DOT_SIZE       = 12
_INNER_DOT_SIZE = 8
_LINE_W         = 2
_LEFT_COL       = 32   # px — fixed width for dot + connector column
_INNER_ROW_H    = 22   # px — minimum height per inner step row


# ── Status mapping ───────────────────────────────────────────────────────────
def _queue_status_to_step_status(s) -> "StepStatus":
    """Map AutoLamellaTaskStatus → StepStatus for display."""
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
    return {
        AutoLamellaTaskStatus.NotStarted: StepStatus.PENDING,
        AutoLamellaTaskStatus.InProgress: StepStatus.ACTIVE,
        AutoLamellaTaskStatus.Completed:  StepStatus.COMPLETED,
        AutoLamellaTaskStatus.Failed:     StepStatus.FAILED,
        AutoLamellaTaskStatus.Skipped:    StepStatus.SKIPPED,
    }[s]


# ── Data model ────────────────────────────────────────────────────────────────
class StepStatus(Enum):
    PENDING   = auto()
    ACTIVE    = auto()
    COMPLETED = auto()
    FAILED    = auto()
    SKIPPED   = auto()


@dataclass
class TimelineStep:
    label: str
    status: StepStatus = StepStatus.PENDING
    subtitle: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _status_color(status: StepStatus) -> str:
    return {
        StepStatus.COMPLETED: _DOT_COMPLETED,
        StepStatus.ACTIVE:    _DOT_ACTIVE,
        StepStatus.PENDING:   _DOT_PENDING,
        StepStatus.FAILED:    _DOT_FAILED,
        StepStatus.SKIPPED:   _DOT_SKIPPED,
    }[status]


# ── _DotWidget ────────────────────────────────────────────────────────────────
class _DotWidget(QWidget):
    """A solid coloured antialiased circle."""

    def __init__(self, color: str, size: int = _DOT_SIZE, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size  = size
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

    def __init__(self, step: TimelineStep, index: int, is_last: bool, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._inner_rows: List[_InnerStepRow] = []
        self._setup_ui(step, is_last)

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

        if not is_last:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFixedWidth(_LINE_W)
            line.setStyleSheet(f"background: {_LINE_COLOR}; border: none;")
            line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            left_layout.addWidget(line, 1, Qt.AlignHCenter)

        root.addWidget(left)

        # ── Right column: label + subtitle + inner container ──────────────
        right = QVBoxLayout()
        right.setSpacing(1)
        right.setContentsMargins(0, 4, 0, 4)

        self._label = QLabel(step.label)
        self._label.setStyleSheet(f"color: {_LABEL_COLOR};")
        if step.status == StepStatus.ACTIVE:
            f = self._label.font()
            f.setBold(True)
            self._label.setFont(f)
        right.addWidget(self._label)

        self._subtitle = QLabel(step.subtitle)
        self._subtitle.setStyleSheet(f"color: {_SUBTITLE_COLOR}; font-size: 11px;")
        self._subtitle.setVisible(bool(step.subtitle))
        right.addWidget(self._subtitle)

        # Error label — shown on failed rows
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

    # ── Outer row refresh ─────────────────────────────────────────────────
    def refresh(self, step: TimelineStep) -> None:
        self._dot.set_color(_status_color(step.status))
        self._label.setText(step.label)
        f = self._label.font()
        f.setBold(step.status == StepStatus.ACTIVE)
        f.setStrikeOut(step.status == StepStatus.SKIPPED)
        self._label.setFont(f)
        self._subtitle.setText(step.subtitle)
        self._subtitle.setVisible(bool(step.subtitle))
        # Clear error on refresh — errors are set separately via set_error()
        self._error_label.setVisible(False)

    def set_error(self, msg: str) -> None:
        """Show an error message below the subtitle."""
        self._error_label.setText(msg)
        self._error_label.setVisible(bool(msg))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setStyleSheet(f"background: {_SELECTED_BG if selected else 'transparent'};")

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: List[TimelineStep] = []
        self._rows: List[_OuterRow] = []
        self._selected_index: Optional[int] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {stylesheets.GRAY_BACKGROUND_COLOR}; border: none; }}"
        )

        self._contents = QWidget()
        self._contents.setStyleSheet(f"background: {stylesheets.GRAY_BACKGROUND_COLOR};")
        self._contents_layout = QVBoxLayout(self._contents)
        self._contents_layout.setContentsMargins(0, 4, 0, 4)
        self._contents_layout.setSpacing(0)

        self._scroll.setWidget(self._contents)
        root.addWidget(self._scroll)

    # ── Public API ────────────────────────────────────────────────────────
    def set_steps(self, steps: List[TimelineStep]) -> None:
        self.clear()
        self._steps = list(steps)
        for i, step in enumerate(self._steps):
            row = _OuterRow(step, i, is_last=(i == len(self._steps) - 1))
            row.clicked.connect(self._on_row_clicked)
            self._rows.append(row)
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
            row.deleteLater()
        self._rows.clear()
        item = self._contents_layout.takeAt(0)
        while item:
            item = self._contents_layout.takeAt(0)
        self._selected_index = None

    # ── Internal ──────────────────────────────────────────────────────────
    def _on_row_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._rows):
            return
        if self._selected_index is not None and self._selected_index < len(self._rows):
            self._rows[self._selected_index].set_selected(False)
        self._selected_index = index
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list = []  # List[WorkItem] from queue snapshot
        self._outer_index: int = -1
        self._inner_finished: bool = False
        self._active_start_time: Optional[float] = None
        self._setup_ui()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = QLabel("Workflow")
        self._header.setStyleSheet(_HEADER_STYLE)
        root.addWidget(self._header)

        self._outer = WorkflowTimelineWidget()
        root.addWidget(self._outer, 1)

    # ── Public API ────────────────────────────────────────────────────────
    def set_workflow(self, items: list) -> None:
        """Pre-populate the outer timeline from queue items (WorkItem instances)."""
        self._items = list(items)
        self._outer_index = -1
        steps = [
            TimelineStep(
                label=item.lamella_name,
                subtitle=item.task_name,
                status=_queue_status_to_step_status(item.status),
            )
            for item in self._items
        ]
        self._outer.set_steps(steps)
        self._update_header(self._items)

    def update_from_status(self, status: dict) -> None:
        """Advance the timeline from a workflow_update_signal payload.

        Reads queue_items snapshot to update all row statuses directly.
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        queue_items = status.get("queue_items", None)
        if queue_items is None:
            return

        task_status = status.get("status", None)
        task_duration = status.get("task_duration", None)

        # Update all row statuses from queue snapshot
        active_idx = -1
        for i, item in enumerate(queue_items):
            if i >= len(self._outer._rows):
                break
            step_status = _queue_status_to_step_status(item.status)
            self._outer.set_step_status(i, step_status)
            if item.status == AutoLamellaTaskStatus.InProgress:
                active_idx = i

        # New active row — show inner container, start elapsed timer
        if active_idx >= 0 and active_idx != self._outer_index:
            self._outer_index = active_idx
            self._inner_finished = False
            self._active_start_time = status.get("timestamp", time.time())
            self._elapsed_timer.start(1000)
            self._show_inner_at(active_idx)
            # Auto-scroll to the active row
            if active_idx < len(self._outer._rows):
                self._outer._scroll.ensureWidgetVisible(
                    self._outer._rows[active_idx]
                )

        # Completion/failure — set subtitle, hide inner, resolve last inner step
        if task_status in (AutoLamellaTaskStatus.Completed, AutoLamellaTaskStatus.Failed):
            self._elapsed_timer.stop()
            self._active_start_time = None
            idx = self._outer_index
            if 0 <= idx < len(self._outer._rows):
                task_name = queue_items[idx].task_name if idx < len(queue_items) else ""
                self._set_completion_subtitle(idx, task_duration, task_name)
                # Refresh the row so the updated subtitle is rendered
                self._outer._rows[idx].refresh(self._outer._steps[idx])
                self._outer._rows[idx].set_inner_visible(False)
                # Show error message on failed rows
                if task_status == AutoLamellaTaskStatus.Failed:
                    error_msg = status.get("error_message", None)
                    if error_msg:
                        self._outer._rows[idx].set_error(error_msg)
            self._finish_inner(failed=(task_status == AutoLamellaTaskStatus.Failed))

        # Update progress header
        self._update_header(queue_items)

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
        self._outer_index = -1
        self._inner_finished = False
        self._elapsed_timer.stop()
        self._active_start_time = None
        self._header.setText("Workflow")
        self._outer.clear()

    def set_active_outer(self, index: int) -> None:
        """Show the inner container for *index*, hide all others.

        Convenience method for test scripts that drive the widget directly
        rather than through update_from_status().
        """
        self._outer_index = index
        self._show_inner_at(index)

    # ── Internal ──────────────────────────────────────────────────────────
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

    def _update_header(self, queue_items: list) -> None:
        """Update the header label with progress counts."""
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
        total = len(queue_items)
        if total == 0:
            self._header.setText("Workflow")
            return
        done = sum(1 for i in queue_items if i.status in (
            AutoLamellaTaskStatus.Completed, AutoLamellaTaskStatus.Skipped))
        failed = sum(1 for i in queue_items if i.status == AutoLamellaTaskStatus.Failed)
        text = f"Workflow — {done}/{total}"
        if failed:
            text += f" ({failed} failed)"
        self._header.setText(text)

    def _update_elapsed(self) -> None:
        """Tick handler — update the active row subtitle with elapsed time."""
        if self._active_start_time is None:
            return
        if not (0 <= self._outer_index < len(self._outer._steps)):
            return
        from fibsem.utils import format_duration
        elapsed = time.time() - self._active_start_time
        task_name = ""
        if self._outer_index < len(self._items):
            task_name = self._items[self._outer_index].task_name
        duration_str = format_duration(elapsed)
        self._outer._steps[self._outer_index].subtitle = f"{task_name} ({duration_str})"
        self._outer._rows[self._outer_index].refresh(self._outer._steps[self._outer_index])

    def _set_completion_subtitle(self, outer_idx: int, task_duration: Optional[float], task_name: str = "") -> None:
        from fibsem.utils import format_duration
        if not (0 <= outer_idx < len(self._outer._steps)):
            return
        duration_str = f" ({format_duration(task_duration)})" if task_duration is not None else ""
        self._outer._steps[outer_idx].subtitle = f"{task_name}{duration_str}"


# ── Demo ──────────────────────────────────────────────────────────────────────
DEMO_STEPS = [
    TimelineStep("Setup session",        StepStatus.COMPLETED),
    TimelineStep("Move to position",     StepStatus.COMPLETED, "0.32 s"),
    TimelineStep("Acquire overview",     StepStatus.COMPLETED, "1.1 s"),
    TimelineStep("Detect features",      StepStatus.COMPLETED, "2.4 s"),
    TimelineStep("Mill rough trench",    StepStatus.ACTIVE,    "running…"),
    TimelineStep("Mill fine trench",     StepStatus.PENDING),
    TimelineStep("Polish lamella",       StepStatus.PENDING),
    TimelineStep("Acquire final image",  StepStatus.PENDING),
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
