from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter
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

# ── Colours ──────────────────────────────────────────────────────────────────
_DOT_COMPLETED = "#4caf50"   # green
_DOT_ACTIVE    = "#ff9800"   # orange
_DOT_PENDING   = "#606060"   # muted gray
_DOT_FAILED    = "#99121F"   # red

_LINE_COLOR    = "#3a3d42"
_SELECTED_BG   = "#2d3f5c"
_LABEL_COLOR   = stylesheets.GRAY_TEXT_COLOR       # "#F0F1F2"
_SUBTITLE_COLOR = stylesheets.GRAY_SECONDARY_COLOR  # "#868E93"

_DOT_SIZE  = 12   # px diameter
_LINE_W    = 2    # px connector line width
_LEFT_COL  = 32   # px fixed width for dot + line column


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


# ── _DotWidget ────────────────────────────────────────────────────────────────
class _DotWidget(QWidget):
    """A solid coloured circle."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(_DOT_SIZE, _DOT_SIZE)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, _DOT_SIZE, _DOT_SIZE)


# ── _StepRow ─────────────────────────────────────────────────────────────────
class _StepRow(QWidget):
    clicked = pyqtSignal(int)  # emits own index

    def __init__(self, step: TimelineStep, index: int, is_last: bool, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
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

        # ── Right column: label + subtitle ────────────────────────────────
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

        root.addLayout(right, 1)

    # ── Public refresh ────────────────────────────────────────────────────
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
        bg = _SELECTED_BG if selected else "transparent"
        self.setStyleSheet(f"background: {bg};")

    # ── Interaction ───────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


# ── WorkflowTimelineWidget ────────────────────────────────────────────────────
class WorkflowTimelineWidget(QWidget):
    """Vertical timeline showing workflow steps with coloured status dots."""

    step_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: List[TimelineStep] = []
        self._rows: List[_StepRow] = []
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
        self._contents.setStyleSheet(
            f"background: {stylesheets.GRAY_BACKGROUND_COLOR};"
        )
        self._contents_layout = QVBoxLayout(self._contents)
        self._contents_layout.setContentsMargins(0, 4, 0, 4)
        self._contents_layout.setSpacing(0)

        self._spacer_added = False

        self._scroll.setWidget(self._contents)
        root.addWidget(self._scroll)

    # ── Public API ────────────────────────────────────────────────────────
    def set_steps(self, steps: List[TimelineStep]) -> None:
        self.clear()
        self._steps = list(steps)
        for i, step in enumerate(self._steps):
            is_last = i == len(self._steps) - 1
            row = _StepRow(step, i, is_last)
            row.clicked.connect(self._on_row_clicked)
            self._rows.append(row)
            self._contents_layout.addWidget(row)

        # Push rows to top
        self._contents_layout.addStretch(1)

    def set_step_status(self, index: int, status: StepStatus) -> None:
        if 0 <= index < len(self._steps):
            self._steps[index].status = status
            self._rows[index].refresh(self._steps[index])

    def set_active_step(self, index: int) -> None:
        """Mark *index* as ACTIVE, all prior as COMPLETED, the rest as PENDING."""
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
        # Remove stretch if present
        item = self._contents_layout.takeAt(0)
        while item:
            item = self._contents_layout.takeAt(0)
        self._selected_index = None

    def _append_step(self, step: TimelineStep) -> None:
        """Append a single step without clearing existing rows."""
        # Remove stretch first
        count = self._contents_layout.count()
        if count > 0:
            last = self._contents_layout.itemAt(count - 1)
            if last and last.spacerItem():
                self._contents_layout.takeAt(count - 1)

        # Update previous last row to add its connector line now it's no longer last
        if self._rows:
            prev_row = self._rows[-1]
            prev_step = self._steps[-1]
            # Rebuild the left column of the previous row to add the line
            left = prev_row.layout().itemAt(0).widget()
            left_layout = left.layout()
            if left_layout.count() == 1:  # only dot, no line yet
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFixedWidth(_LINE_W)
                line.setStyleSheet(f"background: {_LINE_COLOR}; border: none;")
                line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                left_layout.addWidget(line, 1, Qt.AlignHCenter)

        index = len(self._steps)
        self._steps.append(step)
        row = _StepRow(step, index, is_last=True)
        row.clicked.connect(self._on_row_clicked)
        self._rows.append(row)
        self._contents_layout.addWidget(row)
        self._contents_layout.addStretch(1)

        # Scroll to bottom to follow new steps
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        )

    # ── Internal ──────────────────────────────────────────────────────────
    def _on_row_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._rows):
            return
        if self._selected_index is not None and self._selected_index < len(self._rows):
            self._rows[self._selected_index].set_selected(False)
        self._selected_index = index
        self._rows[index].set_selected(True)
        self.step_selected.emit(index)


# ── WorkflowItem ─────────────────────────────────────────────────────────────
@dataclass
class WorkflowItem:
    """One (lamella, task) pair in the outer workflow timeline."""
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
_DIVIDER_STYLE = f"background: #3a3d42; border: none;"


class WorkflowProgressWidget(QWidget):
    """Two-level workflow progress display.

    Outer timeline: one row per (lamella, task) pair.
    Inner timeline: individual steps within the currently active task,
                    revealed progressively as they execute.
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
        root.setAlignment(Qt.AlignTop)

        outer_header = QLabel("Workflow")
        outer_header.setStyleSheet(_HEADER_STYLE)
        root.addWidget(outer_header)

        self._outer = WorkflowTimelineWidget()
        root.addWidget(self._outer)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(_DIVIDER_STYLE)
        root.addWidget(divider)

        inner_header = QLabel("Current task steps")
        inner_header.setStyleSheet(_HEADER_STYLE)
        root.addWidget(inner_header)

        self._inner = WorkflowTimelineWidget()
        root.addWidget(self._inner)

    # ── Public API ────────────────────────────────────────────────────────
    def set_workflow(self, task_names: List[str], lamella_names: List[str]) -> None:
        """Pre-populate the outer timeline.

        Loop order matches run_tasks: task-outer, lamella-inner.
        """
        self._items = [
            WorkflowItem(lamella_name=ln, task_name=tn)
            for tn in task_names
            for ln in lamella_names
        ]
        self._outer_index = -1
        self._outer.set_steps([item.as_timeline_step() for item in self._items])
        self._inner.clear()

    def update_from_status(self, status: dict) -> None:
        """Feed a workflow_update_signal dict to advance the outer timeline.

        Expected keys: task_names, lamella_names, current_task_index,
                       current_lamella_index, status (AutoLamellaTaskStatus-like).
        """
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        task_names    = status.get("task_names", [])
        lamella_names = status.get("lamella_names", [])
        task_idx      = status.get("current_task_index", 0)
        lam_idx       = status.get("current_lamella_index", 0)
        task_status   = status.get("status", None)

        n_lamellas = len(lamella_names)
        if n_lamellas == 0:
            return
        outer_idx = task_idx * n_lamellas + lam_idx

        if task_status == AutoLamellaTaskStatus.InProgress:
            # Mark all prior outer rows green, this one orange, rest gray
            self._outer_index = outer_idx
            for i, item in enumerate(self._items):
                if i < outer_idx:
                    item.status = StepStatus.COMPLETED
                elif i == outer_idx:
                    item.status = StepStatus.ACTIVE
                else:
                    item.status = StepStatus.PENDING
                self._outer.set_step_status(i, item.status)
            self._inner.clear()

        elif task_status == AutoLamellaTaskStatus.Completed:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.COMPLETED
                self._outer.set_step_status(outer_idx, StepStatus.COMPLETED)
            self._finish_inner(failed=False)

        elif task_status == AutoLamellaTaskStatus.Failed:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.FAILED
                self._outer.set_step_status(outer_idx, StepStatus.FAILED)
            self._finish_inner(failed=True)

        elif task_status == AutoLamellaTaskStatus.Skipped:
            if 0 <= outer_idx < len(self._items):
                self._items[outer_idx].status = StepStatus.PENDING
                self._outer.set_step_status(outer_idx, StepStatus.PENDING)

    def update_step(self, step_name: str) -> None:
        """Append a new ACTIVE step to the inner timeline.

        The previously active inner step is marked COMPLETED automatically.
        """
        steps = self._inner._steps
        if steps and steps[-1].status == StepStatus.ACTIVE:
            self._inner.set_step_status(len(steps) - 1, StepStatus.COMPLETED)
        self._inner._append_step(TimelineStep(label=step_name, status=StepStatus.ACTIVE))

    def finish_current_step(self, failed: bool = False) -> None:
        """Resolve the last inner step as COMPLETED or FAILED."""
        self._finish_inner(failed)

    def clear_steps(self) -> None:
        """Clear the inner (current task steps) timeline only."""
        self._inner.clear()

    def clear(self) -> None:
        self._items.clear()
        self._outer_index = -1
        self._outer.clear()
        self._inner.clear()

    # ── Internal ──────────────────────────────────────────────────────────
    def _finish_inner(self, failed: bool) -> None:
        steps = self._inner._steps
        if steps and steps[-1].status == StepStatus.ACTIVE:
            final = StepStatus.FAILED if failed else StepStatus.COMPLETED
            self._inner.set_step_status(len(steps) - 1, final)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _status_color(status: StepStatus) -> str:
    return {
        StepStatus.COMPLETED: _DOT_COMPLETED,
        StepStatus.ACTIVE:    _DOT_ACTIVE,
        StepStatus.PENDING:   _DOT_PENDING,
        StepStatus.FAILED:    _DOT_FAILED,
    }[status]


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

    layout = QVBoxLayout(win)
    layout.setContentsMargins(0, 0, 0, 0)

    timeline = WorkflowTimelineWidget()
    timeline.set_steps(DEMO_STEPS)
    timeline.step_selected.connect(lambda i: print(f"Selected step {i}"))

    layout.addWidget(timeline)
    win.show()
    sys.exit(app.exec_())
