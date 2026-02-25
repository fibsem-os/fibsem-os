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
    AutoLamellaWorkflowOptions,
    Experiment,
    Lamella,
)
from fibsem.ui.widgets.lamella_list_widget import LamellaListWidget
from fibsem.ui.widgets.workflow_config_widget import WorkflowConfigWidget
from fibsem.ui.widgets.workflow_info_widget import WorkflowInfoWidget
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
        self.setMinimumWidth(380)
        self.setMinimumHeight(440)
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
        layout.addWidget(self.editor, 1)

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
    task_added = pyqtSignal(object)                  # AutoLamellaTaskDescription
    task_schedule_changed = pyqtSignal(object)       # AutoLamellaTaskDescription
    task_selection_changed = pyqtSignal(list)        # List[AutoLamellaTaskDescription]
    task_order_changed = pyqtSignal(list)            # List[AutoLamellaTaskDescription]

    # ── workflow info signals ────────────────────────────────────────────
    workflow_name_changed = pyqtSignal(str)
    workflow_description_changed = pyqtSignal(str)
    workflow_options_changed = pyqtSignal(object)    # AutoLamellaWorkflowOptions

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None

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

        # ── workflow info section ────────────────────────────────────────
        self._info_header = QLabel("Workflow")
        self._info_header.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(self._info_header)

        self.info = WorkflowInfoWidget()
        root.addWidget(self.info)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep2)

        # ── workflow section ─────────────────────────────────────────────
        self._workflow_header = QLabel("Tasks")
        self._workflow_header.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(self._workflow_header)

        self.workflow = WorkflowConfigWidget()
        self.workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.workflow, 1)

        # ── summary label ────────────────────────────────────────────────
        self._summary_label = QLabel("0 lamella, 0 tasks selected")
        self._summary_label.setStyleSheet(
            "color: #007ACC; font-size: 11px; padding: 3px 6px;"
        )
        root.addWidget(self._summary_label)

        # ── instructions ─────────────────────────────────────────────────
        self._instructions_label = QLabel(
            "Drag to reorder  \u2022  click supervision icon to toggle  \u2022  use \u270e to edit task details"
        )
        self._instructions_label.setStyleSheet(
            "color: #a0a0a0; font-size: 10px; padding: 2px 6px 4px 6px;"
        )
        root.addWidget(self._instructions_label)

        # ── wire signals ─────────────────────────────────────────────────
        self.lamella_list.move_to_requested.connect(self.lamella_move_to_requested)
        self.lamella_list.edit_requested.connect(self.lamella_edit_requested)
        self.lamella_list.remove_requested.connect(self.lamella_remove_requested)
        self.lamella_list.defect_changed.connect(self.lamella_defect_changed)
        self.lamella_list.selection_changed.connect(self.lamella_selection_changed)

        self.workflow.supervised_changed.connect(self.task_supervised_changed)
        self.workflow.edit_requested.connect(self._on_task_edit_requested)
        self.workflow.remove_requested.connect(self.task_remove_requested)
        self.workflow.schedule_changed.connect(self.task_schedule_changed)
        self.workflow.selection_changed.connect(self.task_selection_changed)
        self.workflow.order_changed.connect(self.task_order_changed)
        self.workflow.add_task_clicked.connect(self._on_add_task_clicked)

        self.lamella_list.selection_changed.connect(lambda _: self._update_summary())
        self.workflow.selection_changed.connect(lambda _: self._update_summary())

        self.info.name_changed.connect(self.workflow_name_changed)
        self.info.description_changed.connect(self.workflow_description_changed)
        self.info.options_changed.connect(self.workflow_options_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self.experiment = experiment

    def set_workflow_config(self, config: AutoLamellaWorkflowConfig) -> None:
        self.workflow.set_config(config)
        self.info.set_config(config)
        self._update_summary()

    def set_options(self, options: AutoLamellaWorkflowOptions) -> None:
        self.info.set_options(options)

    def add_lamella(self, lamella: Lamella, checked: bool = True):
        result = self.lamella_list.add_lamella(lamella, checked)
        self._update_summary()
        return result

    def add_task(self, task: AutoLamellaTaskDescription, checked: bool = True):
        result = self.workflow.add_task(task, checked)
        self._update_summary()
        return result

    def get_selected_lamella(self) -> List[Lamella]:
        return self.lamella_list.get_selected()

    def get_selected_tasks(self) -> List[AutoLamellaTaskDescription]:
        return self.workflow.get_selected()

    def get_tasks(self) -> List[AutoLamellaTaskDescription]:
        return self.workflow.get_tasks()

    def clear(self) -> None:
        self.lamella_list.clear()
        self.workflow.clear()
        self._update_summary()

    def set_lamella_header(self, text: str) -> None:
        self._lamella_header.setText(text)

    def set_workflow_header(self, text: str) -> None:
        self._workflow_header.setText(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _available_task_names(self) -> List[str]:
        if self.experiment is None:
            return []
        return sorted(self.experiment.task_protocol.task_config.keys())

    def _update_summary(self) -> None:
        n_lam = len(self.lamella_list.get_selected())
        n_task = len(self.workflow.get_selected())
        if n_lam == 0 or n_task == 0:
            missing = []
            if n_lam == 0:
                missing.append("a lamella")
            if n_task == 0:
                missing.append("a task")
            self._summary_label.setStyleSheet(
                "color: #f0a040; font-size: 11px; padding: 3px 6px;"
            )
            self._summary_label.setText(
                f"Select {' and '.join(missing)} to run the workflow"
            )
        else:
            self._summary_label.setStyleSheet(
                "color: #007ACC; font-size: 11px; padding: 3px 6px;"
            )
            self._summary_label.setText(
                f"{n_lam} lamella, {n_task} task{'s' if n_task != 1 else ''} selected"
            )

    def _on_task_edit_requested(self, task: AutoLamellaTaskDescription) -> None:
        available = [t.name for t in self.workflow.get_tasks()]
        self._editor_dialog.open_for(task, available_tasks=available)

    def _on_task_applied(self, task: AutoLamellaTaskDescription) -> None:
        self.workflow.refresh_task(task)
        self.task_edited.emit(task)

    def _on_add_task_clicked(self) -> None:
        # Import here to avoid circular imports at module level
        from fibsem.ui.widgets.autolamella_workflow_widget import AddTaskDialog

        available = self._available_task_names()
        dialog = AddTaskDialog(
            available_tasks=available,
            experiment=self.experiment,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            task_name = dialog.get_selected_task()
            if task_name is None:
                return
            task = AutoLamellaTaskDescription(
                name=task_name, supervise=True, required=True
            )
            self.workflow.add_task(task)
            self.task_added.emit(task)
            self._update_summary()
