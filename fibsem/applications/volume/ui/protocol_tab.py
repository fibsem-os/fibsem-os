"""The volume window's Protocol tab: the protocol's tasks in run order, with
when each needs a person, Add Task, and Edit Task for what a task requires and
when it may start. Ticked tasks are the ones Run runs.

Every edit is a ``workflow_edits`` function, the one the App Server's protocol
edits call, saved with the protocol and recorded. A refusal is said on a toast
and the list is drawn again from the protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QWidget

from fibsem.ui import notification_service
from fibsem.ui.widgets.workflow.add_task_dialog import AddTaskDialog
from fibsem.ui.widgets.workflow.task_editor_dialog import TaskEditorDialog
from fibsem.ui.widgets.workflow.task_list import WorkflowTaskList
from fibsem.workflows import workflow_edits
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import TaskPlan

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession

__all__ = ["ProtocolTab"]

TASK_LIST_WIDTH = 560


class ProtocolTab(QWidget):
    def __init__(self, session: "AppSession", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.tasks = WorkflowTaskList()
        self.tasks.setMaximumWidth(TASK_LIST_WIDTH)
        self._editor = TaskEditorDialog(self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.tasks, 1)
        layout.addStretch(0)

        self.tasks.add_task_clicked.connect(self._on_add)
        self.tasks.attention_toggled.connect(self._on_attention)
        self.tasks.edit_requested.connect(self._on_edit)
        self.tasks.order_changed.connect(self._on_reorder)
        self._editor.editor.apply_clicked.connect(self._on_edited)
        self._editor.remove_requested.connect(self._on_remove)
        self.refresh()

    @property
    def protocol(self):
        experiment = self.session.experiment
        return experiment.task_protocol if experiment is not None else None

    def refresh(self) -> None:
        """Draw the tasks from the protocol, keeping the ticks; every task is
        ticked when there were none."""
        protocol = self.protocol
        self.setEnabled(protocol is not None)
        if protocol is None:
            self.tasks.clear()
            return
        fresh = not self.tasks.get_tasks()
        self.tasks.set_protocol(protocol)
        self.tasks.set_config(protocol.workflow_config)
        if fresh:
            self.tasks.set_all_selected(True)

    def selected_task_names(self) -> List[str]:
        """The ticked tasks, in run order."""
        return [task.name for task in self.tasks.get_selected()]

    def _done(self, result: dict) -> None:
        """Say a refusal, and draw the list from the protocol after the click
        or drop that made the edit has finished."""
        if not result["applied"]:
            notification_service.show_toast(result["error"], "warning")
        QTimer.singleShot(0, self.refresh)

    def _record(self, kind, payload) -> None:
        self.session.record_event(kind, payload)

    def _on_add(self) -> None:
        protocol = self.protocol
        if protocol is None:
            return
        dialog = AddTaskDialog(
            {name: cls.config_cls for name, cls in protocol.task_classes().items()},
            protocol.task_config,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted or dialog.chosen is None:
            return
        task_type, task_name = dialog.chosen
        self._done(
            workflow_edits.add_task(
                self.session.experiment, task_type, task_name, self._record
            )
        )

    def _on_attention(self, task: TaskPlan) -> None:
        supervised = self.protocol.get_attention(task.name) is Attention.supervised
        self._done(
            workflow_edits.set_supervision(
                self.session.experiment, task.name, not supervised, self._record
            )
        )

    def _on_edit(self, task: TaskPlan) -> None:
        # Only the tasks before it: a task cannot require one that runs later.
        names = [t.name for t in self.tasks.get_tasks()]
        self._editor.open_for(
            task,
            self.protocol.requirements(task.name),
            available_tasks=names[: names.index(task.name)],
        )

    def _on_edited(self, edited: TaskPlan, requires: list) -> None:
        experiment = self.session.experiment
        result = workflow_edits.set_requirements(
            experiment, edited.name, requires, edited.required, self._record
        )
        if result["applied"]:
            result = workflow_edits.set_schedule(
                experiment, edited.name, edited.scheduled_at, self._record
            )
        self._done(result)

    def _on_remove(self, task: TaskPlan) -> None:
        self._done(
            workflow_edits.remove_task(self.session.experiment, task.name, self._record)
        )

    def _on_reorder(self, tasks: List[TaskPlan]) -> None:
        self._done(
            workflow_edits.reorder_tasks(
                self.session.experiment, [t.name for t in tasks], self._record
            )
        )
