"""Standalone test script for WorkflowTaskEditorWidget.

Exercises the task editor's PROPERTIES (optional/required), SCHEDULING
(enable checkbox + datetime) and REQUIREMENTS sections, and prints the
edited AutoLamellaTaskDescription whenever Apply is clicked.

Run directly:
    python fibsem/ui/widgets/tests/test_workflow_task_editor_widget.py
"""
import sys
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription
from fibsem.ui.widgets.workflow_task_editor_widget import WorkflowTaskEditorWidget

# The other tasks in the workflow — offered as selectable requirements.
AVAILABLE_TASKS = ["Setup", "MillRough", "MillPolishing", "Sharpen"]


def _scheduled_task() -> AutoLamellaTaskDescription:
    """A required task scheduled ~2 minutes out, depending on Setup."""
    return AutoLamellaTaskDescription(
        name="MillRough",
        supervise=True,
        required=True,
        requires=["Setup"],
        scheduled_at=(datetime.now() + timedelta(minutes=2)).replace(second=0, microsecond=0),
    )


def _unscheduled_task() -> AutoLamellaTaskDescription:
    """An optional, unscheduled task with no requirements."""
    return AutoLamellaTaskDescription(
        name="Sharpen",
        supervise=False,
        required=False,
        requires=[],
        scheduled_at=None,
    )


def _describe(task: AutoLamellaTaskDescription) -> str:
    sched = task.scheduled_at.strftime("%Y-%m-%d %H:%M") if task.scheduled_at else "None"
    requires = ", ".join(task.requires) if task.requires else "—"
    return (
        f"name={task.name}  required={task.required}  "
        f"supervise={task.supervise}\nrequires=[{requires}]  scheduled_at={sched}"
    )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QWidget()
    win.setWindowTitle("WorkflowTaskEditorWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.resize(440, 640)

    outer = QVBoxLayout(win)
    outer.setContentsMargins(12, 12, 12, 12)
    outer.setSpacing(8)

    task = _scheduled_task()
    widget = WorkflowTaskEditorWidget(task, available_tasks=AVAILABLE_TASKS)
    outer.addWidget(widget, 1)

    status = QLabel("Click Apply to write the edits back and print the task.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    status.setWordWrap(True)
    outer.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_scheduled = QPushButton("Load scheduled task")
    btn_unscheduled = QPushButton("Load unscheduled task")
    btn_print = QPushButton("Print current task")
    btn_layout.addWidget(btn_scheduled)
    btn_layout.addWidget(btn_unscheduled)
    btn_layout.addWidget(btn_print)
    outer.addWidget(btn_row)

    def on_apply(t: AutoLamellaTaskDescription) -> None:
        text = _describe(t)
        status.setText(f"apply_clicked →\n{text}")
        print("apply_clicked ->")
        print(text)
        print("to_dict() ->", t.to_dict())
        print("-" * 60)

    def load(factory) -> None:
        new_task = factory()
        widget.load_task(new_task, available_tasks=AVAILABLE_TASKS)
        # keep apply_clicked operating on the freshly-loaded task
        status.setText(f"Loaded:\n{_describe(new_task)}")

    widget.apply_clicked.connect(on_apply)
    widget.cancel_clicked.connect(lambda: status.setText("cancel_clicked"))
    btn_scheduled.clicked.connect(lambda: load(_scheduled_task))
    btn_unscheduled.clicked.connect(lambda: load(_unscheduled_task))
    btn_print.clicked.connect(lambda: print(_describe(task)))

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
