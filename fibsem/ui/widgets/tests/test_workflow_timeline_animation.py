"""
Animated two-level test for WorkflowProgressWidget.

Outer timeline: 2 lamellas × 3 tasks = 6 (lamella, task) rows.
Inner timeline: each task reveals 3–5 named steps progressively (1–2 s apart).
A random 20 % failure chance applies to each inner step; a failed step stops that task.
"""

import random
import sys
from typing import Dict, List

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem
from fibsem.ui import stylesheets
from fibsem.ui.widgets.workflow_timeline_widget import (
    StepStatus,
    WorkflowProgressWidget,
)

# ── Fake workflow data ────────────────────────────────────────────────────────
LAMELLA_NAMES = ["Lamella-01", "Lamella-02"]
TASK_NAMES    = ["Trench Milling", "Rough Milling", "Polishing"]

# Inner steps per task type
TASK_STEPS: Dict[str, List[str]] = {
    "Trench Milling": [
        "Move to trench position",
        "Align trench reference",
        "Mill trench",
        "Acquire reference image",
    ],
    "Rough Milling": [
        "Move to milling pose",
        "Align reference image",
        "Mill stress relief",
        "Mill lamella",
        "Acquire reference images",
    ],
    "Polishing": [
        "Move to milling pose",
        "Align reference image",
        "Mill polishing pass 1",
        "Mill polishing pass 2",
        "Acquire final image",
    ],
}

_FAIL_CHANCE = 0.15   # probability any inner step fails


class AnimatedDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workflow Progress — Two-Level Demo")
        self.resize(380, 620)
        self.setStyleSheet(f"background: {stylesheets.GRAY_BACKGROUND_COLOR};")

        # Build flat list of (lamella, task) pairs matching run_tasks loop order
        self._outer_items: List[WorkItem] = [
            WorkItem(lamella_name=ln, task_name=tn)
            for tn in TASK_NAMES
            for ln in LAMELLA_NAMES
        ]
        self._outer_idx   = -1   # which outer item is active
        self._inner_steps: List[str] = []
        self._inner_idx   = -1   # which inner step is active

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Top bar: status label + buttons ───────────────────────────────
        top_bar = QWidget()
        top_bar.setStyleSheet(f"background: {stylesheets.GRAY_BACKGROUND_COLOR};")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(6)

        self._status_label = QLabel("Starting…")
        self._status_label.setStyleSheet(
            f"color: {stylesheets.GRAY_SECONDARY_COLOR}; font-size: 11px;"
        )
        top_layout.addWidget(self._status_label, 1)

        self._btn_restart = QPushButton("Restart")
        self._btn_restart.setFixedHeight(24)
        self._btn_restart.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        top_layout.addWidget(self._btn_restart)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedHeight(24)
        self._btn_stop.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        top_layout.addWidget(self._btn_stop)

        layout.addWidget(top_bar)

        self._progress = WorkflowProgressWidget()
        self._progress.set_workflow(self._outer_items)
        layout.addWidget(self._progress)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

        self._btn_restart.clicked.connect(self._restart)
        self._btn_stop.clicked.connect(self._stop)

        self._timer.start(400)   # short initial delay

    # ── Timer tick ────────────────────────────────────────────────────────
    def _tick(self):
        """Each tick either starts the next outer item or advances the inner steps."""

        # Still have inner steps to run (includes -1 = not yet started)
        if self._inner_steps and self._inner_idx < len(self._inner_steps) - 1:
            self._advance_inner()
            return

        # All inner steps done — resolve last one before moving to next outer
        if self._inner_idx >= 0:
            self._progress.finish_current_step(failed=False)

        # Advance to next outer item
        self._outer_idx += 1
        if self._outer_idx >= len(self._outer_items):
            self._status_label.setText("All tasks completed.")
            return

        item = self._outer_items[self._outer_idx]
        self._inner_steps = TASK_STEPS[item.task_name]
        self._inner_idx   = -1

        # Drive the outer timeline directly
        for i in range(len(self._outer_items)):
            if i < self._outer_idx:
                self._progress._outer.set_step_status(i, StepStatus.COMPLETED)
            elif i == self._outer_idx:
                self._progress._outer.set_step_status(i, StepStatus.ACTIVE)
            else:
                self._progress._outer.set_step_status(i, StepStatus.PENDING)

        self._progress.set_active_outer(self._outer_idx)
        self._status_label.setText(
            f"[{item.lamella_name}] {item.task_name} — starting…"
        )

        # Short pause before first inner step
        self._timer.start(600)

    def _advance_inner(self):
        self._inner_idx += 1
        step_name = self._inner_steps[self._inner_idx]

        # Random failure
        if random.random() < _FAIL_CHANCE:
            self._progress.update_step(step_name)
            self._progress.finish_current_step(failed=True)
            # Mark outer row failed too
            self._progress._outer.set_step_status(self._outer_idx, StepStatus.FAILED)
            failed_lamella = self._outer_items[self._outer_idx].lamella_name
            self._status_label.setText(
                f"[{failed_lamella}] FAILED at '{step_name}' — stopping task."
            )
            # Reset inner so next tick moves to next outer
            self._inner_idx = len(self._inner_steps) - 1
            outer_delay = random.randint(5_000, 10_000)
            self._timer.start(outer_delay)
            return

        self._progress.update_step(step_name)

        item = self._outer_items[self._outer_idx]
        self._status_label.setText(
            f"[{item.lamella_name}] {item.task_name} — {step_name}"
        )

        is_last_inner = self._inner_idx == len(self._inner_steps) - 1
        if is_last_inner:
            # Pause at last inner step for a bit before outer advances
            outer_delay = random.randint(5_000, 10_000)
            self._timer.start(outer_delay)
        else:
            self._timer.start(random.randint(1_000, 2_000))


    def _restart(self):
        self._timer.stop()
        self._outer_items = [
            WorkItem(lamella_name=ln, task_name=tn)
            for tn in TASK_NAMES
            for ln in LAMELLA_NAMES
        ]
        self._outer_idx  = -1
        self._inner_steps = []
        self._inner_idx  = -1
        self._progress.set_workflow(self._outer_items)
        self._status_label.setText("Restarted.")
        self._timer.start(400)

    def _stop(self):
        self._timer.stop()
        self._inner_steps = []
        self._inner_idx   = -1
        self._status_label.setText("Stopped.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AnimatedDemo()
    win.show()
    sys.exit(app.exec_())
