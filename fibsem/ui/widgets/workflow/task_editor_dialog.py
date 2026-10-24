"""Edit Task: a task's requirements and schedule, in a dialog over the task
list, and where a task is removed from it."""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QWidget

from fibsem.theme.tokens import SURFACE_COLOR, TEXT_COLOR
from fibsem.ui.widgets.workflow.task_editor import WorkflowTaskEditorWidget
from fibsem.workflows.task_config import TaskPlan

__all__ = ["TaskEditorDialog"]


class TaskEditorDialog(QDialog):
    """Modal dialog wrapping WorkflowTaskEditorWidget. Also where a task is
    removed from the workflow: the editor's Remove asks here, this confirms,
    closes, and tells the host which task to drop."""

    remove_requested = pyqtSignal(object)  # TaskPlan

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Task")
        self.setModal(True)
        self.setMinimumWidth(470)
        self.setMinimumHeight(520)
        self.setStyleSheet(f"background: {SURFACE_COLOR}; color: {TEXT_COLOR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.editor = WorkflowTaskEditorWidget(task=TaskPlan(name=""))
        # Use the editor's own styled Apply/Cancel buttons as the dialog actions.
        self.editor.apply_clicked.connect(self.accept)
        self.editor.cancel_clicked.connect(self.reject)
        self.editor.remove_clicked.connect(self._on_remove_clicked)
        layout.addWidget(self.editor, 1)
        self._task: Optional[TaskPlan] = None

    def open_for(
        self,
        task: TaskPlan,
        requires: List[str],
        available_tasks: List[str],
        allow_remove: bool = True,
    ) -> None:
        self._task = task
        self.editor.load_task(task, requires, available_tasks=available_tasks)
        self.editor.set_remove_allowed(allow_remove)
        self.open()

    def _on_remove_clicked(self) -> None:
        task = self._task
        if task is None:
            return
        reply = QMessageBox.question(
            self,
            "Remove Task",
            f"Remove <b>{task.name}</b> from workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.reject()
        self.remove_requested.emit(task)
