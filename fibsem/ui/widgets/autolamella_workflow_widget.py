from __future__ import annotations

from pprint import pprint
from typing import List, Optional, TYPE_CHECKING

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QToolButton,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
    Experiment,
    AutoLamellaTaskProtocol,
)
from fibsem.ui import stylesheets
from superqt import QIconifyIcon

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI

class TaskListWidget(QListWidget):
    """List widget with drag-and-drop reorder signal."""

    order_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAlternatingRowColors(True)
        self.setSpacing(2)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#202020"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2c2c2c"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#dddddd"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#3a6ea5"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        super().dropEvent(event)
        self.order_changed.emit()


class EditRequirementsDialog(QDialog):
    """Dialog for editing task requirements."""

    def __init__(
        self,
        task_name: str,
        current_requirements: List[str],
        available_tasks: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit Requirements for '{task_name}'")
        self.setMinimumWidth(400)

        self.current_requirements = list(current_requirements)
        self.available_tasks = [t for t in available_tasks if t != task_name]

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Instructions
        info_label = QLabel(
            "Select which tasks must be completed before this task can run:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Checkboxes for each available task
        self.checkboxes: dict[str, QCheckBox] = {}
        for task in self.available_tasks:
            checkbox = QCheckBox(task)
            checkbox.setChecked(task in self.current_requirements)
            checkbox.setStyleSheet(stylesheets.CHECKBOX_STYLE)
            self.checkboxes[task] = checkbox
            layout.addWidget(checkbox)

        if not self.available_tasks:
            no_tasks_label = QLabel("No other tasks in workflow to add as requirements")
            no_tasks_label.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(no_tasks_label)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel # type: ignore
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_requirements(self) -> List[str]:
        """Get the selected requirements."""
        return [task for task, checkbox in self.checkboxes.items() if checkbox.isChecked()]


class AddTaskDialog(QDialog):
    """Dialog for selecting a task to add to the workflow."""

    def __init__(
        self,
        available_tasks: List[str],
        experiment: Optional[Experiment] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Task to Workflow")
        self.setMinimumWidth(400)

        self.available_tasks = available_tasks
        self.experiment = experiment
        self.selected_task: Optional[str] = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Instructions
        info_label = QLabel("Select a task to add to the workflow:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Task selector
        self.task_selector = QComboBox()
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
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel # type: ignore
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

    def get_selected_task(self) -> Optional[str]:
        """Get the selected task name."""
        return self.selected_task


class WorkflowTaskItemWidget(QWidget):
    """Widget representing a workflow task entry."""

    task_changed = pyqtSignal(AutoLamellaTaskDescription)
    remove_requested = pyqtSignal(AutoLamellaTaskDescription)
    requirements_changed = pyqtSignal(AutoLamellaTaskDescription)

    def __init__(
        self,
        task_desc: AutoLamellaTaskDescription,
        available_tasks: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.task_desc = task_desc
        self.available_tasks = available_tasks or []

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 4, 8, 4)
        main_layout.setSpacing(2)
        self.setLayout(main_layout)

        # Top row with controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        self.name_label = QLabel(task_desc.name)
        self.name_label.setMinimumWidth(180)
        controls_layout.addWidget(self.name_label)

        self.required_checkbox = QCheckBox("Required")
        self.required_checkbox.setChecked(task_desc.required)
        self.required_checkbox.setStyleSheet(stylesheets.CHECKBOX_STYLE)
        self.required_checkbox.setMinimumWidth(80)
        self.required_checkbox.setMaximumWidth(80)
        controls_layout.addWidget(self.required_checkbox)

        self.supervise_checkbox = QCheckBox("Supervised")
        self.supervise_checkbox.setChecked(task_desc.supervise)
        self.supervise_checkbox.setStyleSheet(stylesheets.CHECKBOX_STYLE)
        self.supervise_checkbox.setMinimumWidth(95)
        self.supervise_checkbox.setMaximumWidth(95)
        controls_layout.addWidget(self.supervise_checkbox)

        self.edit_button = QToolButton()
        self.edit_button.setIcon(QIconifyIcon("mdi:dots-horizontal", color="white")) # type: ignore
        self.edit_button.setIconSize(QSize(10, 10))
        self.edit_button.setToolTip("Edit Task Requirements")
        self.edit_button.clicked.connect(self._on_edit_clicked)
        # self.edit_button.setFixedWidth(30)
        controls_layout.addWidget(self.edit_button)

        controls_layout.addStretch(1)

        self.remove_button = QPushButton()
        # self.remove_button.setFixedWidth(30)
        self.remove_button.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.remove_button.setToolTip("Remove task from workflow")
        self.remove_button.setIcon(QIconifyIcon("mdi:close", color="white")) # type: ignore
        self.remove_button.setIconSize(QSize(10, 10))
        controls_layout.addWidget(self.remove_button)

        main_layout.addLayout(controls_layout)

        # Requirements label
        self.requirements_label = QLabel()
        self.requirements_label.setStyleSheet("color: lightgray; font-size: 10px; font-style: italic;")
        self.requirements_label.setContentsMargins(4, 0, 0, 0)
        self.requirements_label.setWordWrap(True)
        self._update_requirements_label()
        main_layout.addWidget(self.requirements_label)

        self.required_checkbox.toggled.connect(self._on_required_toggled)
        self.supervise_checkbox.toggled.connect(self._on_supervise_toggled)
        self.remove_button.clicked.connect(self._on_remove_clicked)

    def _on_required_toggled(self, checked: bool) -> None:
        self.task_desc.required = checked
        self.task_changed.emit(self.task_desc)

    def _on_supervise_toggled(self, checked: bool) -> None:
        self.task_desc.supervise = checked
        self.task_changed.emit(self.task_desc)

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit(self.task_desc)

    def _on_edit_clicked(self) -> None:
        dialog = EditRequirementsDialog(
            task_name=self.task_desc.name,
            current_requirements=self.task_desc.requires,
            available_tasks=self.available_tasks,
            parent=self,
        )

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            new_requirements = dialog.get_requirements()
            self.task_desc.requires = new_requirements
            self._update_requirements_label()
            self.requirements_changed.emit(self.task_desc)

    def _update_requirements_label(self) -> None:
        if self.task_desc.requires:
            requirements_text = ", ".join(self.task_desc.requires)
            self.requirements_label.setText(f"Requires: {requirements_text}")
        else:
            self.requirements_label.setText("No requirements")

class AutoLamellaWorkflowWidget(QWidget):
    """Widget to edit AutoLamella workflow configuration and options."""

    workflow_config_changed = pyqtSignal(AutoLamellaWorkflowConfig)
    workflow_options_changed = pyqtSignal(AutoLamellaWorkflowOptions)

    def __init__(
        self,
        experiment: Optional[Experiment] = None,
        parent: Optional['AutoLamellaUI'] = None,
    ) -> None:
        super().__init__(parent)
        self.experiment: Optional[Experiment] = experiment
        self.workflow_config: AutoLamellaWorkflowConfig
        self.workflow_options: AutoLamellaWorkflowOptions
        self._updating_ui = False
        self.parent_widget = parent

        self._attach_data_sources()
        self._setup_ui()
        self._connect_signals()
        self._refresh_from_state()

    def _attach_data_sources(self) -> None:
        if self.experiment is not None:
            protocol = self.experiment.task_protocol
            self.workflow_config = protocol.workflow_config
            self.workflow_options = protocol.options
        else:
            self.workflow_config = AutoLamellaWorkflowConfig()
            self.workflow_options = AutoLamellaWorkflowOptions()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(main_layout)

        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("A name for the workflow")
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("An optional description for the workflow")
        form_layout.addRow("Name", self.name_edit)
        form_layout.addRow("Description", self.description_edit)
        main_layout.addLayout(form_layout)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        self.turn_beams_off_checkbox = QCheckBox("Turn beams off after workflow")
        options_layout.addWidget(self.turn_beams_off_checkbox)
        main_layout.addWidget(options_group)

        tasks_group = QGroupBox("Tasks")
        tasks_layout = QVBoxLayout()
        tasks_group.setLayout(tasks_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addStretch(1)
        self.add_task_button = QPushButton("Add Task")
        self.add_task_button.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        controls_layout.addWidget(self.add_task_button)
        tasks_layout.addLayout(controls_layout)

        self.task_list = TaskListWidget()
        self.task_list.setMinimumHeight(300)
        tasks_layout.addWidget(self.task_list)

        help_label = QLabel(
            "Drag to reorder. Toggle checkboxes to update requirements and supervision."
        )
        help_label.setStyleSheet("color: gray; font-style: italic;")
        help_label.setWordWrap(True)
        tasks_layout.addWidget(help_label)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: limegreen; font-style: italic;")
        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: orange; font-style: italic;")
        self.validation_label.setWordWrap(True)
        tasks_layout.addWidget(self.validation_label)
        tasks_layout.addWidget(self.summary_label)

        main_layout.addWidget(tasks_group)

        # Run workflow button
        self.run_workflow_button = QPushButton("Run Workflow")
        self.run_workflow_button.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.run_workflow_button.setToolTip("Run the AutoLamella workflow")
        self.run_workflow_button.setVisible(False) # Hide by default; shown in main UI
        main_layout.addWidget(self.run_workflow_button)

        main_layout.addStretch(1)

    def _connect_signals(self) -> None:
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.description_edit.editingFinished.connect(self._on_description_changed)
        self.turn_beams_off_checkbox.toggled.connect(
            self._on_turn_beams_off_toggled
        )
        self.add_task_button.clicked.connect(self._on_add_task)
        self.task_list.order_changed.connect(self._sync_order_from_list)
        self.run_workflow_button.clicked.connect(self._on_run_workflow)

    def _refresh_from_state(self) -> None:
        self._updating_ui = True

        self.name_edit.setText(self.workflow_config.name)
        self.description_edit.setText(self.workflow_config.description)
        self.turn_beams_off_checkbox.setChecked(
            self.workflow_options.turn_beams_off
        )

        self.task_list.clear()
        for task_desc in self.workflow_config.tasks:
            self._add_task_widget(task_desc)

        self._update_summary()

        self._updating_ui = False

    def _available_task_names(self) -> List[str]:
        if self.experiment is None:
            return []
        return sorted(self.experiment.task_protocol.task_config.keys())

    def _add_task_widget(self, task_desc: AutoLamellaTaskDescription) -> None:
        item = QListWidgetItem()
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        available_tasks = self._get_workflow_task_names()
        widget = WorkflowTaskItemWidget(task_desc, available_tasks, self.task_list)
        widget.task_changed.connect(self._on_task_description_changed)
        widget.remove_requested.connect(self._on_remove_task)
        widget.requirements_changed.connect(self._on_requirements_changed)

        item.setSizeHint(widget.sizeHint())
        self.task_list.addItem(item)
        self.task_list.setItemWidget(item, widget)

    def _get_workflow_task_names(self) -> List[str]:
        """Get list of task names currently in the workflow."""
        return [task.name for task in self.workflow_config.tasks]

    def _update_all_widget_task_lists(self) -> None:
        """Update the available tasks list in all task widgets."""
        available_tasks = self._get_workflow_task_names()
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            widget = self.task_list.itemWidget(item)
            if widget is not None and isinstance(widget, WorkflowTaskItemWidget):
                widget.available_tasks = available_tasks

    def _on_name_changed(self) -> None:
        if self._updating_ui:
            return
        self.workflow_config.name = self.name_edit.text().strip()
        self._emit_config_changed()

    def _on_description_changed(self) -> None:
        if self._updating_ui:
            return
        self.workflow_config.description = self.description_edit.text().strip()
        self._emit_config_changed()

    def _on_turn_beams_off_toggled(self, checked: bool) -> None:
        if self._updating_ui:
            return
        self.workflow_options.turn_beams_off = checked
        self.workflow_options_changed.emit(self.workflow_options)

    def _on_run_workflow(self) -> None:
        """Run the workflow by calling parent widget's run_task_workflow method."""
        if self.parent_widget is not None:
            self.parent_widget.run_task_workflow()

    def _on_add_task(self) -> None:
        available = self._available_task_names()

        dialog = AddTaskDialog(
            available_tasks=available,
            experiment=self.experiment,
            parent=self,
        )

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            task_name = dialog.get_selected_task()
            if task_name is None:
                return

            task_desc = AutoLamellaTaskDescription(
                name=task_name,
                supervise=True,
                required=True,
                requires=[],
            )

            self.workflow_config.tasks.append(task_desc)
            self._add_task_widget(task_desc)
            self._emit_config_changed()
            self._update_summary()
            self._update_all_widget_task_lists()

    def _on_task_description_changed(
        self, task_desc: AutoLamellaTaskDescription
    ) -> None:
        self._emit_config_changed()
        self._update_summary()

    def _on_requirements_changed(
        self, task_desc: AutoLamellaTaskDescription
    ) -> None:
        self._emit_config_changed()
        self._update_summary()
        self._update_all_widget_task_lists()

    def _on_remove_task(self, task_desc: AutoLamellaTaskDescription) -> None:
        reply = QMessageBox.question(
            self,
            "Remove Task",
            f"Remove task '{task_desc.name}' from the workflow?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._remove_task_from_config(task_desc)
        self._remove_task_widget(task_desc)
        self._emit_config_changed()
        self._update_summary()
        self._update_all_widget_task_lists()

    def _remove_task_from_config(self, task_desc: AutoLamellaTaskDescription) -> None:
        tasks = list(self.workflow_config.tasks)
        for idx, existing in enumerate(tasks):
            if existing is task_desc:
                del tasks[idx]
                self._assign_tasks(tasks)
                return

    def _remove_task_widget(self, task_desc: AutoLamellaTaskDescription) -> None:
        for index in range(self.task_list.count() - 1, -1, -1):
            item = self.task_list.item(index)
            widget = self.task_list.itemWidget(item)
            if widget is not None and widget.task_desc is task_desc:
                self.task_list.takeItem(index)
                break

    def _sync_order_from_list(self) -> None:
        ordered_tasks: List[AutoLamellaTaskDescription] = []
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            widget = self.task_list.itemWidget(item)
            if widget is not None:
                ordered_tasks.append(widget.task_desc)

        if len(ordered_tasks) != len(self.workflow_config.tasks):
            return
        self._assign_tasks(ordered_tasks)
        self._emit_config_changed()
        self._update_summary()

    def _assign_tasks(self, tasks: List[AutoLamellaTaskDescription]) -> None:
        if hasattr(self.workflow_config.tasks, "clear"):
            self.workflow_config.tasks.clear()
            self.workflow_config.tasks.extend(tasks)
        else:
            self.workflow_config.tasks = list(tasks)

    def _emit_config_changed(self) -> None:
        self.workflow_config_changed.emit(self.workflow_config)

    def set_experiment(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self._attach_data_sources()
        self._refresh_from_state()
        self._update_summary()

    def set_workflow_config(self, workflow_config: AutoLamellaWorkflowConfig) -> None:
        self.workflow_config = workflow_config
        self._refresh_from_state()
        self._update_summary()

    def set_workflow_options(self, workflow_options: AutoLamellaWorkflowOptions) -> None:
        self.workflow_options = workflow_options
        self._refresh_from_state()
        self._update_summary()

    def get_workflow_config(self) -> AutoLamellaWorkflowConfig:
        return self.workflow_config

    def get_workflow_options(self) -> AutoLamellaWorkflowOptions:
        return self.workflow_options

    def _update_summary(self) -> None:
        total = len(self.workflow_config.tasks)
        required = sum(1 for task in self.workflow_config.tasks if task.required)
        supervised = sum(1 for task in self.workflow_config.tasks if task.supervise)
        self.summary_label.setText(
            f"{total} Task{'s' if total != 1 else ''} in Workflow"
            f" ({required} Required, {supervised} Supervised)"
        )

        # validation 
        issues = self.workflow_config.validate()
        if len(issues) == 0:
            self.validation_label.setVisible(False)
        else:
            self.validation_label.setVisible(True)
            txt = "\n".join(issues)
            self.validation_label.setText(f"Workflow configuration has issues: \n{txt}")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    import os
    # PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-09-25-13-38"
    # exp = Experiment.load(os.path.join(PATH, "experiment.yaml"))
    from fibsem.applications.autolamella.config import TASK_PROTOCOL_PATH
    exp = Experiment.create(path=os.getcwd(), name="Test Experiment")
    exp.task_protocol = AutoLamellaTaskProtocol.load(TASK_PROTOCOL_PATH)
    app = QApplication(sys.argv)
    widget = AutoLamellaWorkflowWidget(experiment=exp)

    def print_config(cfg: AutoLamellaWorkflowConfig):
        print(f"-"* 40)
        print(cfg.name)
        print(cfg.description)
        for t in cfg.tasks:
            print(f"  - {t.name}: required={t.required}, supervise={t.supervise}")
        print(f"-"* 40)

    widget.workflow_config_changed.connect(print_config)
    widget.workflow_options_changed.connect(lambda opts: pprint(f"Options changed: {opts.to_dict()}"))

    widget.show()
    sys.exit(app.exec_())
