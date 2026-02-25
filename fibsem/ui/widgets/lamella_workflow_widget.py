from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
    Lamella,
)
from fibsem.ui.widgets.lamella_list_widget import LamellaListWidget
from fibsem.ui.widgets.workflow_config_widget import WorkflowConfigWidget
from fibsem.ui.widgets.workflow_task_editor_widget import WorkflowTaskEditorWidget

_SECTION_LABEL_STYLE = (
    "font-size: 11px; font-weight: bold; color: #a0a0a0;"
    " padding: 4px 6px 2px 6px; background: #1e2124;"
)


class _TaskEditorDialog(QDialog):
    """Modal dialog wrapping WorkflowTaskEditorWidget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Task")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet("background: #2b2d31; color: #d6d6d6;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.editor = WorkflowTaskEditorWidget(
            task=AutoLamellaTaskDescription(name="", supervise=False, required=True),
        )
        # Hide the built-in Apply/Cancel buttons — the dialog provides its own.
        self.editor._apply_btn.hide()
        self.editor._cancel_btn.hide()
        layout.addWidget(self.editor)

        self._btn_box = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        self._btn_box.setStyleSheet("padding: 6px;")
        self._btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self._btn_box.rejected.connect(self.reject)
        layout.addWidget(self._btn_box)

    def open_for(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: List[str],
    ) -> None:
        self.editor.load_task(task, available_tasks=available_tasks)
        self.open()

    def _on_apply(self) -> None:
        self.editor._on_apply()
        self.accept()


class LamellaWorkflowWidget(QWidget):
    """Combined widget: LamellaListWidget (top) + WorkflowConfigWidget (bottom).

    Task editing is handled internally via a modal dialog.  All signals from
    both sub-widgets are re-emitted so callers can connect to one place.
    Sub-widgets are accessible directly via ``self.lamella_list`` and
    ``self.workflow``.
    """

    # ── lamella signals ──────────────────────────────────────────────────
    lamella_move_to_requested = pyqtSignal(object)   # Lamella
    lamella_edit_requested = pyqtSignal(object)      # Lamella
    lamella_remove_requested = pyqtSignal(object)    # Lamella
    lamella_defect_changed = pyqtSignal(object)      # Lamella
    lamella_selection_changed = pyqtSignal(list)     # List[Lamella]

    # ── workflow signals ─────────────────────────────────────────────────
    task_supervised_changed = pyqtSignal(object)     # AutoLamellaTaskDescription
    task_edited = pyqtSignal(object)                 # AutoLamellaTaskDescription (after apply)
    task_remove_requested = pyqtSignal(object)       # AutoLamellaTaskDescription
    task_selection_changed = pyqtSignal(list)        # List[AutoLamellaTaskDescription]
    task_order_changed = pyqtSignal(list)            # List[AutoLamellaTaskDescription]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._editor_dialog = _TaskEditorDialog(self)
        self._editor_dialog.editor.apply_clicked.connect(self._on_task_applied)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── lamella section ──────────────────────────────────────────────
        self._lamella_header = QLabel("Lamella")
        self._lamella_header.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(self._lamella_header)

        self.lamella_list = LamellaListWidget()
        self.lamella_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.lamella_list, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep)

        # ── workflow section ─────────────────────────────────────────────
        self._workflow_header = QLabel("Workflow")
        self._workflow_header.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(self._workflow_header)

        self.workflow = WorkflowConfigWidget()
        self.workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.workflow, 1)

        # ── wire signals ─────────────────────────────────────────────────
        self.lamella_list.move_to_requested.connect(self.lamella_move_to_requested)
        self.lamella_list.edit_requested.connect(self.lamella_edit_requested)
        self.lamella_list.remove_requested.connect(self.lamella_remove_requested)
        self.lamella_list.defect_changed.connect(self.lamella_defect_changed)
        self.lamella_list.selection_changed.connect(self.lamella_selection_changed)

        self.workflow.supervised_changed.connect(self.task_supervised_changed)
        self.workflow.edit_requested.connect(self._on_task_edit_requested)
        self.workflow.remove_requested.connect(self.task_remove_requested)
        self.workflow.selection_changed.connect(self.task_selection_changed)
        self.workflow.order_changed.connect(self.task_order_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_workflow_config(self, config: AutoLamellaWorkflowConfig) -> None:
        self.workflow.set_config(config)

    def add_lamella(self, lamella: Lamella, checked: bool = True):
        return self.lamella_list.add_lamella(lamella, checked)

    def add_task(self, task: AutoLamellaTaskDescription, checked: bool = True):
        return self.workflow.add_task(task, checked)

    def get_selected_lamella(self) -> List[Lamella]:
        return self.lamella_list.get_selected()

    def get_selected_tasks(self) -> List[AutoLamellaTaskDescription]:
        return self.workflow.get_selected()

    def get_tasks(self) -> List[AutoLamellaTaskDescription]:
        return self.workflow.get_tasks()

    def clear(self) -> None:
        self.lamella_list.clear()
        self.workflow.clear()

    def set_lamella_header(self, text: str) -> None:
        self._lamella_header.setText(text)

    def set_workflow_header(self, text: str) -> None:
        self._workflow_header.setText(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_task_edit_requested(self, task: AutoLamellaTaskDescription) -> None:
        available = [t.name for t in self.workflow.get_tasks()]
        self._editor_dialog.open_for(task, available_tasks=available)

    def _on_task_applied(self, task: AutoLamellaTaskDescription) -> None:
        self.workflow.refresh_task(task)
        self.task_edited.emit(task)
