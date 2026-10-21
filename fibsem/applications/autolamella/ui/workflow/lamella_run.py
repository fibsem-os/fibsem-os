from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella.rows import LamellaListWidget
from fibsem.applications.autolamella.ui.workflow.task_editor import (
    WorkflowTaskEditorWidget,
)
from fibsem.applications.autolamella.ui.workflow.task_list import WorkflowTaskList
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    GRAY_ICON_COLOR,
    NEUTRAL_500,
    PANEL_COLOR,
    PRIMARY_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
)
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    ValueComboBox,
)
from fibsem.ui.widgets.lists import OptionsPopup, help_button
from fibsem.workflows.task_config import TaskPlan, WorkflowOptions

# What the panel's ? says: how to work the task list. It was a line of text under
# the list on every view; it is for the first time, not every time.
TASK_LIST_HINTS = (
    "Drag to reorder  \u2022  click the chip to change when you are involved"
    "  \u2022  use \u270e to edit task details"
)


class PickTaskDialog(QDialog):
    """Dialog for selecting a task to add to the workflow."""

    def __init__(
        self,
        available_tasks: list[str],
        experiment: Experiment | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Task to Workflow")
        self.setMinimumWidth(400)

        self.available_tasks = available_tasks
        self.experiment = experiment
        self.selected_task: str | None = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Instructions
        info_label = QLabel("Select a task to add to the workflow:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Task selector
        self.task_selector = ValueComboBox()
        self.task_selector.addItem("Select a task...", None)

        for name in sorted(available_tasks):
            display = name
            if experiment is not None:
                task_config = experiment.task_protocol.task_config.get(name)
                if task_config is not None and getattr(task_config, "task_type", ""):
                    display = f"{name} ({task_config.task_type})"
            self.task_selector.addItem(display, name)

        layout.addWidget(self.task_selector)

        if not available_tasks:
            no_tasks_label = QLabel("No tasks available to add")
            no_tasks_label.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(no_tasks_label)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel  # type: ignore
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_accept(self) -> None:
        self.selected_task = self.task_selector.currentData()
        if self.selected_task is None:
            QMessageBox.warning(
                self,
                "No Task Selected",
                "Please select a task to add to the workflow.",
            )
            return
        self.accept()

    def get_selected_task(self) -> str | None:
        """Get the selected task name."""
        return self.selected_task


class _TaskEditorDialog(QDialog):
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


class LamellaWorkflowWidget(QWidget):
    """Combined widget: LamellaListWidget (top) + WorkflowTaskList (bottom).

    Task editing is handled internally via a modal dialog.  All signals from
    both sub-widgets are re-emitted so callers can connect to one place.
    Sub-widgets are accessible directly via ``self.lamella_list`` and
    ``self.workflow``.
    """

    # ── lamella signals ──────────────────────────────────────────────────
    lamella_edit_requested = pyqtSignal(object)  # Lamella
    lamella_remove_requested = pyqtSignal(object)  # Lamella
    lamella_defect_changed = pyqtSignal(object)  # Lamella
    lamella_selection_changed = pyqtSignal(list)  # List[Lamella]

    # ── workflow signals ─────────────────────────────────────────────────
    task_attention_toggled = pyqtSignal(object)  # TaskPlan
    task_edited = pyqtSignal(object, list)  # TaskPlan: the edited copy, requires
    task_remove_requested = pyqtSignal(object)  # TaskPlan
    task_added = pyqtSignal(object)  # TaskPlan
    task_selection_changed = pyqtSignal(list)  # List[TaskPlan]
    task_order_changed = pyqtSignal(list)  # List[TaskPlan]

    # ── workflow info signals ────────────────────────────────────────────
    workflow_name_changed = pyqtSignal(str)
    workflow_description_changed = pyqtSignal(str)
    workflow_options_changed = pyqtSignal(str, object)  # option, value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None

        self._editor_dialog = _TaskEditorDialog(self)
        self._editor_dialog.editor.apply_clicked.connect(self.task_edited)
        self._editor_dialog.remove_requested.connect(self._on_task_remove_confirmed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── lamella section ──────────────────────────────────────────────
        # No section label over either list: each list's own header names it
        # (`list_chrome.ListHeader`).
        self.lamella_list = LamellaListWidget()
        self.lamella_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.lamella_list, 1)

        # ── workflow section ─────────────────────────────────────────────
        self.workflow = WorkflowTaskList()
        self.workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.workflow, 1)

        # The workflow's name, description and options, behind ⚙ on the task
        # list's header; how to work the list, behind ?. No footer: the "select
        # a lamella and a task" line is the status line's to say.
        self.info = WorkflowInfoWidget()
        self._settings_popup = OptionsPopup(
            "Workflow settings", self, margins=(8, 8, 8, 6), spacing=4
        )
        self._settings_popup.body.addWidget(self.info)
        self.btn_settings = IconToolButton(
            icon="mdi:cog-outline", tooltip="Workflow settings", size=24
        )
        self.btn_settings.clicked.connect(
            lambda: self._settings_popup.show_under(self.btn_settings)
        )
        self.workflow.add_header_widget(self.btn_settings)
        self.btn_help = help_button(TASK_LIST_HINTS)
        self.workflow.add_header_widget(self.btn_help)

        # ── wire signals ─────────────────────────────────────────────────
        self.lamella_list.edit_requested.connect(self.lamella_edit_requested)
        self.lamella_list.remove_requested.connect(self.lamella_remove_requested)
        self.lamella_list.defect_changed.connect(self.lamella_defect_changed)
        self.lamella_list.selection_changed.connect(self.lamella_selection_changed)

        self.workflow.attention_toggled.connect(self.task_attention_toggled)
        self.workflow.edit_requested.connect(self._on_task_edit_requested)
        self.workflow.remove_requested.connect(self.task_remove_requested)
        self.workflow.selection_changed.connect(self.task_selection_changed)
        self.workflow.order_changed.connect(self.task_order_changed)
        self.workflow.add_task_clicked.connect(self._on_add_task_clicked)

        self.info.name_changed.connect(self.workflow_name_changed)
        self.info.description_changed.connect(self.workflow_description_changed)
        self.info.options_changed.connect(self.workflow_options_changed)
        for changed in (
            self.info.name_changed,
            self.info.description_changed,
            self.info.options_changed,
        ):
            changed.connect(lambda *_: self._refresh_settings_button())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self.experiment = experiment
        self.workflow.set_protocol(
            getattr(experiment, "task_protocol", None) if experiment else None
        )

    def set_workflow_config(self, config: AutoLamellaWorkflowConfig) -> None:
        self.workflow.set_config(config)
        self.info.set_config(config)
        self._refresh_settings_button()

    def set_options(self, options: WorkflowOptions) -> None:
        self.info.set_options(options)
        self._refresh_settings_button()

    def _refresh_settings_button(self) -> None:
        """The ⚙ in the accent colour while something in it is set, so a workflow
        name, "turn beams off" or a summary turned off is not hidden by the
        popover."""
        info = self.info
        changed = bool(
            info.name_edit.text()
            or info.desc_edit.text()
            or info.turn_beams_off_cb.isChecked()
            or not info.show_run_summary_cb.isChecked()
        )
        self.btn_settings.setIcon(
            fibsem_icon(
                "mdi:cog-outline", color=ACCENT_COLOR if changed else GRAY_ICON_COLOR
            )
        )
        self.btn_settings.setToolTip(
            "Workflow settings (some set)" if changed else "Workflow settings"
        )

    def add_lamella(self, lamella: Lamella, checked: bool = False):
        return self.lamella_list.add_lamella(lamella, checked)

    def add_task(self, task: TaskPlan, checked: bool = False):
        return self.workflow.add_task(task, checked)

    def get_selected_lamella(self) -> List[Lamella]:
        return self.lamella_list.get_selected()

    def get_selected_tasks(self) -> List[TaskPlan]:
        return self.workflow.get_selected()

    def get_tasks(self) -> List[TaskPlan]:
        return self.workflow.get_tasks()

    def clear(self) -> None:
        self.lamella_list.clear()
        self.workflow.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _available_task_names(self) -> List[str]:
        if self.experiment is None:
            return []
        return sorted(self.experiment.task_protocol.task_config.keys())

    def _on_task_edit_requested(self, task: TaskPlan) -> None:
        # Only the tasks before it: a task cannot require one that runs later
        # (`AutoLamellaTaskProtocol.set_requirements`, the App Server's rule).
        names = [t.name for t in self.workflow.get_tasks()]
        available = names[: names.index(task.name)] if task.name in names else names
        protocol = getattr(self.experiment, "task_protocol", None)
        self._editor_dialog.open_for(
            task,
            protocol.requirements(task.name) if protocol is not None else [],
            available_tasks=available,
            allow_remove=self.workflow.remove_allowed,
        )

    def _on_task_remove_confirmed(self, task: TaskPlan) -> None:
        # the list drops the row and re-emits remove_requested, which is wired
        # to task_remove_requested above
        self.workflow.request_remove(task)

    def _on_add_task_clicked(self) -> None:
        available = self._available_task_names()
        dialog = PickTaskDialog(
            available_tasks=available,
            experiment=self.experiment,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            task_name = dialog.get_selected_task()
            if task_name is None:
                return
            task = TaskPlan(name=task_name)
            self.workflow.add_task(task)
            self.task_added.emit(task)


_LABEL_STYLE = f"color: {NEUTRAL_500}; min-width: 80px; font-size: 11px;"
_SECTION_STYLE = (
    "font-size: 10px; font-weight: bold; color: #707070;"
    " padding: 4px 0px 2px 0px; letter-spacing: 0.5px;"
)
_LINE_EDIT_STYLE = f"""
QLineEdit {{
    background: {SURFACE_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid #3a3d42;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 11px;
}}
QLineEdit:focus {{ border-color: {PRIMARY_COLOR}; }}
"""


class WorkflowInfoWidget(QWidget):
    """Compact editor for workflow name, description, and run options."""

    name_changed = pyqtSignal(str)
    description_changed = pyqtSignal(str)
    options_changed = pyqtSignal(str, object)  # option, value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {PANEL_COLOR};")

        self._config: Optional[AutoLamellaWorkflowConfig] = None
        self._options: Optional[WorkflowOptions] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # ── name row ─────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_lbl = QLabel("Name")
        name_lbl.setStyleSheet(_LABEL_STYLE)
        name_row.addWidget(name_lbl)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Workflow name…")
        self.name_edit.setStyleSheet(_LINE_EDIT_STYLE)
        name_row.addWidget(self.name_edit)
        root.addLayout(name_row)

        # ── description row ───────────────────────────────────────────────
        desc_row = QHBoxLayout()
        desc_row.setSpacing(6)
        desc_lbl = QLabel("Description")
        desc_lbl.setStyleSheet(_LABEL_STYLE)
        desc_row.addWidget(desc_lbl)
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Workflow description…")
        self.desc_edit.setStyleSheet(_LINE_EDIT_STYLE)
        desc_row.addWidget(self.desc_edit)
        root.addLayout(desc_row)

        # ── options section ───────────────────────────────────────────────
        opt_lbl = QLabel("OPTIONS")
        opt_lbl.setStyleSheet(_SECTION_STYLE)
        root.addWidget(opt_lbl)

        self.turn_beams_off_cb = QCheckBox("Turn beams off after completion")
        self.turn_beams_off_cb.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        root.addWidget(self.turn_beams_off_cb)

        self.show_run_summary_cb = QCheckBox("Show summary after completion")
        self.show_run_summary_cb.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        root.addWidget(self.show_run_summary_cb)

        # ── signals ───────────────────────────────────────────────────────
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.desc_edit.editingFinished.connect(self._on_desc_changed)
        self.turn_beams_off_cb.toggled.connect(
            lambda on: self._on_option_changed("turn_beams_off", on)
        )
        self.show_run_summary_cb.toggled.connect(
            lambda on: self._on_option_changed("show_run_summary", on)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config: AutoLamellaWorkflowConfig) -> None:
        self._config = config
        self.name_edit.blockSignals(True)
        self.desc_edit.blockSignals(True)
        self.name_edit.setText(config.name or "")
        self.desc_edit.setText(config.description or "")
        self.name_edit.blockSignals(False)
        self.desc_edit.blockSignals(False)

    def set_options(self, options: WorkflowOptions) -> None:
        self._options = options
        for checkbox, value in (
            (self.turn_beams_off_cb, options.turn_beams_off),
            (self.show_run_summary_cb, options.show_run_summary),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)

    def showEvent(self, event) -> None:
        # The agent can change an option while this is closed
        # (workflow_set_options); show what is set now, not what was.
        if self._options is not None:
            self.set_options(self._options)
        super().showEvent(event)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        text = self.name_edit.text()
        if self._config is not None:
            self._config.name = text
        self.name_changed.emit(text)

    def _on_desc_changed(self) -> None:
        text = self.desc_edit.text()
        if self._config is not None:
            self._config.description = text
        self.description_changed.emit(text)

    def _on_option_changed(self, option: str, value: bool) -> None:
        if self._options is not None:
            setattr(self._options, option, value)
        self.options_changed.emit(option, value)
