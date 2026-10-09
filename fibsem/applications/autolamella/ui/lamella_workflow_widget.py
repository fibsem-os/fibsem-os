from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_list_widget import LamellaListWidget
from fibsem.applications.autolamella.ui.workflow_config_widget import (
    WorkflowConfigWidget,
)
from fibsem.applications.autolamella.ui.workflow_info_widget import WorkflowInfoWidget
from fibsem.applications.autolamella.ui.workflow_task_editor_widget import (
    WorkflowTaskEditorWidget,
)
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    GRAY_ICON_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    ValueComboBox,
)

# What the panel's ? says: how to work the task list. It was a line of text under
# the list on every view; it is for the first time, not every time.
TASK_LIST_HINTS = (
    "Drag to reorder  \u2022  click the chip to change when you are involved"
    "  \u2022  use \u270e to edit task details"
)


class _WorkflowSettingsPopup(QFrame):
    """The workflow's name, description and run options, behind the ⚙ on the task
    list's header: set once per run if at all, they took a quarter of the panel
    on every view."""

    def __init__(self, info: WorkflowInfoWidget, parent: QWidget) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("workflowSettings")
        self.setStyleSheet(
            f"#workflowSettings {{ background: {PANEL_COLOR}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 4px; }}"
        )
        self.setFixedWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(4)
        title = QLabel("Workflow settings")
        title.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 13px; font-weight: 600; "
            "background: transparent;"
        )
        layout.addWidget(title)
        layout.addWidget(info)

    def show_under(self, button: QWidget) -> None:
        """Open below *button*, its right edge on the button's."""
        self.adjustSize()
        corner = button.mapToGlobal(QPoint(button.width(), button.height()))
        self.move(corner.x() - self.width(), corner.y() + 2)
        self.show()


class AddTaskDialog(QDialog):
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

    remove_requested = pyqtSignal(object)  # AutoLamellaTaskDescription

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

        self.editor = WorkflowTaskEditorWidget(
            task=AutoLamellaTaskDescription(name="", required=True),
        )
        # Use the editor's own styled Apply/Cancel buttons as the dialog actions.
        self.editor.apply_clicked.connect(self.accept)
        self.editor.cancel_clicked.connect(self.reject)
        self.editor.remove_clicked.connect(self._on_remove_clicked)
        layout.addWidget(self.editor, 1)
        self._task: Optional[AutoLamellaTaskDescription] = None

    def open_for(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: List[str],
        allow_remove: bool = True,
    ) -> None:
        self._task = task
        self.editor.load_task(task, available_tasks=available_tasks)
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
    """Combined widget: LamellaListWidget (top) + WorkflowConfigWidget (bottom).

    Task editing is handled internally via a modal dialog.  All signals from
    both sub-widgets are re-emitted so callers can connect to one place.
    Sub-widgets are accessible directly via ``self.lamella_list`` and
    ``self.workflow``.
    """

    # ── lamella signals ──────────────────────────────────────────────────
    lamella_move_to_requested = pyqtSignal(object)  # Lamella
    lamella_edit_requested = pyqtSignal(object)  # Lamella
    lamella_remove_requested = pyqtSignal(object)  # Lamella
    lamella_defect_changed = pyqtSignal(object)  # Lamella
    lamella_selection_changed = pyqtSignal(list)  # List[Lamella]

    # ── workflow signals ─────────────────────────────────────────────────
    task_attention_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    task_edited = pyqtSignal(object)  # AutoLamellaTaskDescription (after apply)
    task_remove_requested = pyqtSignal(object)  # AutoLamellaTaskDescription
    task_added = pyqtSignal(object)  # AutoLamellaTaskDescription
    task_selection_changed = pyqtSignal(list)  # List[AutoLamellaTaskDescription]
    task_order_changed = pyqtSignal(list)  # List[AutoLamellaTaskDescription]

    # ── workflow info signals ────────────────────────────────────────────
    workflow_name_changed = pyqtSignal(str)
    workflow_description_changed = pyqtSignal(str)
    workflow_options_changed = pyqtSignal(object)  # AutoLamellaWorkflowOptions

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None

        self._editor_dialog = _TaskEditorDialog(self)
        self._editor_dialog.editor.apply_clicked.connect(self._on_task_applied)
        self._editor_dialog.remove_requested.connect(self._on_task_remove_confirmed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── lamella section ──────────────────────────────────────────────
        # No section label over either list: each list's own header names it
        # (`list_chrome.ListHeader`).
        self.lamella_list = LamellaListWidget()
        self.lamella_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lamella_list.enable_move_to_action(False)
        root.addWidget(self.lamella_list, 1)

        # ── workflow section ─────────────────────────────────────────────
        self.workflow = WorkflowConfigWidget()
        self.workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.workflow, 1)

        # The workflow's name, description and options, behind ⚙ on the task
        # list's header; how to work the list, behind ?. No footer: the "select
        # a lamella and a task" line is the status line's to say.
        self.info = WorkflowInfoWidget()
        self._settings_popup = _WorkflowSettingsPopup(self.info, self)
        self.btn_settings = IconToolButton(
            icon="mdi:cog-outline", tooltip="Workflow settings", size=24
        )
        self.btn_settings.clicked.connect(
            lambda: self._settings_popup.show_under(self.btn_settings)
        )
        self.workflow.add_header_widget(self.btn_settings)
        self.btn_help = QToolButton()
        self.btn_help.setFixedSize(24, 24)
        self.btn_help.setAutoRaise(True)
        self.btn_help.setIcon(
            fibsem_icon("mdi:help-circle-outline", color=GRAY_ICON_COLOR)
        )
        self.btn_help.setToolTip(TASK_LIST_HINTS)
        self.btn_help.setStyleSheet(
            "QToolButton { border: none; background: transparent; }"
        )
        self.workflow.add_header_widget(self.btn_help)

        # ── wire signals ─────────────────────────────────────────────────
        self.lamella_list.move_to_requested.connect(self.lamella_move_to_requested)
        self.lamella_list.edit_requested.connect(self.lamella_edit_requested)
        self.lamella_list.remove_requested.connect(self.lamella_remove_requested)
        self.lamella_list.defect_changed.connect(self.lamella_defect_changed)
        self.lamella_list.selection_changed.connect(self.lamella_selection_changed)

        self.workflow.attention_changed.connect(self.task_attention_changed)
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

    def set_options(self, options: AutoLamellaWorkflowOptions) -> None:
        self.info.set_options(options)
        self._refresh_settings_button()

    def _refresh_settings_button(self) -> None:
        """The ⚙ in the accent colour while something in it is set, so a workflow
        name or "turn beams off" is not hidden by the popover."""
        info = self.info
        changed = bool(
            info.name_edit.text()
            or info.desc_edit.text()
            or info.turn_beams_off_cb.isChecked()
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

    def add_task(self, task: AutoLamellaTaskDescription, checked: bool = False):
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _available_task_names(self) -> List[str]:
        if self.experiment is None:
            return []
        return sorted(self.experiment.task_protocol.task_config.keys())

    def _on_task_edit_requested(self, task: AutoLamellaTaskDescription) -> None:
        available = [t.name for t in self.workflow.get_tasks()]
        self._editor_dialog.open_for(
            task,
            available_tasks=available,
            allow_remove=self.workflow.remove_allowed,
        )

    def _on_task_remove_confirmed(self, task: AutoLamellaTaskDescription) -> None:
        # the list drops the row and re-emits remove_requested, which is wired
        # to task_remove_requested above
        self.workflow.request_remove(task)

    def _on_task_applied(self, task: AutoLamellaTaskDescription) -> None:
        self.workflow.refresh_task(task)
        self.task_edited.emit(task)

    def _on_add_task_clicked(self) -> None:
        # Import here to avoid circular imports at module level

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
                name=task_name, required=True, attention=Attention.supervised
            )
            self.workflow.add_task(task)
            self.task_added.emit(task)
