from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
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

_LINE_COLOR     = "#3a3d42"
_SELECTED_BG    = "#2d3f5c"
_LABEL_COLOR    = stylesheets.GRAY_TEXT_COLOR
_SUBTITLE_COLOR = stylesheets.GRAY_SECONDARY_COLOR

_DOT_SIZE       = 12
_INNER_DOT_SIZE = 8
_LINE_W         = 2
_LEFT_COL       = 32   # px — fixed width for dot + connector column
_INNER_ROW_H    = 22   # px — minimum height per inner step row


# ── Data model ────────────────────────────────────────────────────────────────
class StepStatus(Enum):
    PENDING   = auto()
    ACTIVE    = auto()
    COMPLETED = auto()
    FAILED    = auto()


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
        self._label.setFont(f)
        self._subtitle.setText(step.subtitle)
        self._subtitle.setVisible(bool(step.subtitle))

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


# ── WorkflowItem ──────────────────────────────────────────────────────────────
@dataclass
class WorkflowItem:
    """One (lamella, task) pair tracked in the outer timeline."""
    lamella_name: str
    task_name: str
    status: StepStatus = StepStatus.PENDING

    def as_timeline_step(self) -> TimelineStep:
        return TimelineStep(label=self.lamella_name, subtitle=self.task_name, status=self.status)


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
        self._items: List[WorkflowItem] = []
        self._outer_index: int = -1
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # root.setAlignment(Qt.AlignTop)

        header = QLabel("Workflow")
        header.setStyleSheet(_HEADER_STYLE)
        root.addWidget(header)

        self._outer = WorkflowTimelineWidget()
        root.addWidget(self._outer, 1)

    # ── Public API ────────────────────────────────────────────────────────
    def set_workflow(self, task_names: List[str], lamella_names: List[str]) -> None:
        """Pre-populate the outer timeline (task-outer, lamella-inner loop order)."""
        self._items = [
            WorkflowItem(lamella_name=ln, task_name=tn)
            for tn in task_names
            for ln in lamella_names
        ]
        self._outer_index = -1
        self._outer.set_steps([item.as_timeline_step() for item in self._items])

    def update_from_status(self, status: dict) -> None:
        """Advance the timeline from a workflow_update_signal payload.

        Expected keys: task_names, lamella_names, current_task_index,
                       current_lamella_index, status (AutoLamellaTaskStatus),
                       task_duration (float seconds, optional).
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        task_names    = status.get("task_names", [])
        lamella_names = status.get("lamella_names", [])
        task_idx      = status.get("current_task_index", 0)
        lam_idx       = status.get("current_lamella_index", 0)
        task_status   = status.get("status", None)
        task_duration = status.get("task_duration", None)

        n_lamellas = len(lamella_names)
        if n_lamellas == 0:
            return
        outer_idx = task_idx * n_lamellas + lam_idx

        if task_status == AutoLamellaTaskStatus.InProgress:
            self._outer_index = outer_idx
            for i, item in enumerate(self._items):
                if i < outer_idx:
                    item.status = StepStatus.COMPLETED
                elif i == outer_idx:
                    item.status = StepStatus.ACTIVE
                else:
                    item.status = StepStatus.PENDING
                self._outer.set_step_status(i, item.status)
            self._show_inner_at(outer_idx)

        elif task_status == AutoLamellaTaskStatus.Completed:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.COMPLETED
                self._set_completion_subtitle(outer_idx, task_duration)
                self._outer.set_step_status(outer_idx, StepStatus.COMPLETED)
                self._outer._rows[outer_idx].set_inner_visible(False)
            self._finish_inner(failed=False)

        elif task_status == AutoLamellaTaskStatus.Failed:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.FAILED
                self._set_completion_subtitle(outer_idx, task_duration)
                self._outer.set_step_status(outer_idx, StepStatus.FAILED)
                self._outer._rows[outer_idx].set_inner_visible(False)
            self._finish_inner(failed=True)

        elif task_status == AutoLamellaTaskStatus.Skipped:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.PENDING
                self._outer.set_step_status(outer_idx, StepStatus.PENDING)

    def update_step(self, step_name: str) -> None:
        """Mark the previous inner step completed and append a new ACTIVE one."""
        if not (0 <= self._outer_index < len(self._outer._rows)):
            return
        row = self._outer._rows[self._outer_index]
        row.update_last_inner_step(StepStatus.COMPLETED)
        row.add_inner_step(step_name, StepStatus.ACTIVE)

    def finish_current_step(self, failed: bool = False) -> None:
        """Resolve the last inner step as COMPLETED or FAILED."""
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
        final = StepStatus.FAILED if failed else StepStatus.COMPLETED
        self._outer._rows[self._outer_index].update_last_inner_step(final)

    def _set_completion_subtitle(self, outer_idx: int, task_duration: Optional[float]) -> None:
        from fibsem.utils import format_duration
        if not (0 <= outer_idx < len(self._outer._steps)):
            return
        item = self._items[outer_idx]
        duration_str = f" ({format_duration(task_duration)})" if task_duration is not None else ""
        self._outer._steps[outer_idx].subtitle = f"{item.task_name}{duration_str}"


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
