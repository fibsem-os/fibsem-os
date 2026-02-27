"""Test script for WorkflowConfigWidget."""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.ui.widgets.workflow_config_widget import WorkflowConfigWidget
from fibsem.ui.widgets.workflow_task_editor_widget import WorkflowTaskEditorWidget

SAMPLE_CONFIG = AutoLamellaWorkflowConfig(
    name="Standard Cryo-Lamella",
    description="Standard workflow for cryo-lamella preparation",
    tasks=[
        AutoLamellaTaskDescription(name="Acquire Reference Image", supervise=True, required=True),
        AutoLamellaTaskDescription(name="Mill Trench", supervise=True, required=True),
        AutoLamellaTaskDescription(name="Mill Rough", supervise=False, required=True),
        AutoLamellaTaskDescription(name="Mill Polishing", supervise=True, required=True, requires=["Mill Rough"]),
        AutoLamellaTaskDescription(name="Acquire Final Image", supervise=False, required=False),
    ],
)


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Workflow Config Widget — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.resize(900, 480)

        self._config = SAMPLE_CONFIG
        self._counter = len(self._config.tasks) + 1

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel(f"Workflow: {self._config.name}")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        # ── main split: list | editor ────────────────────────────────────
        split = QHBoxLayout()
        split.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(6)

        self.workflow_widget = WorkflowConfigWidget()
        self.workflow_widget.set_config(self._config)
        left.addWidget(self.workflow_widget)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_add = QPushButton("Add Task")
        btn_add.clicked.connect(self._add_task)
        btn_row.addWidget(btn_add)

        btn_refresh = QPushButton("Refresh All")
        btn_refresh.clicked.connect(self.workflow_widget.refresh_all)
        btn_row.addWidget(btn_refresh)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.workflow_widget.clear)
        btn_row.addWidget(btn_clear)

        btn_print = QPushButton("Print Workflow")
        btn_print.clicked.connect(self._print_workflow)
        btn_row.addWidget(btn_print)

        btn_print_sel = QPushButton("Print Selected")
        btn_print_sel.clicked.connect(self._print_selected)
        btn_row.addWidget(btn_print_sel)

        btn_row.addStretch()
        left.addLayout(btn_row)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(6)
        for label, fn in [
            ("Supervise", self.workflow_widget.enable_supervise_button),
            ("Edit",      self.workflow_widget.enable_edit_button),
            ("Remove",    self.workflow_widget.enable_remove_button),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(fn)
            toggle_row.addWidget(cb)
        toggle_row.addStretch()
        left.addLayout(toggle_row)

        split.addLayout(left, 1)

        # editor panel (hidden until edit is clicked)
        self.editor = WorkflowTaskEditorWidget(
            task=self._config.tasks[0],
            available_tasks=[t.name for t in self._config.tasks],
        )
        self.editor.hide()
        split.addWidget(self.editor)

        root.addLayout(split)

        self.log_label = QLabel("(events will appear here)")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        root.addWidget(self.log_label)

        self.workflow_widget.supervised_changed.connect(
            lambda t: self._log(f"Supervised toggled: {t.name} → {t.supervise}")
        )
        self.workflow_widget.edit_requested.connect(self._on_edit_requested)
        self.workflow_widget.remove_requested.connect(
            lambda t: self._log(f"Removed: {t.name}")
        )
        self.editor.apply_clicked.connect(self._on_editor_apply)
        self.editor.cancel_clicked.connect(self._on_editor_cancel)
        self.workflow_widget.order_changed.connect(self._on_order_changed)

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)

    def _on_edit_requested(self, task: AutoLamellaTaskDescription) -> None:
        available = [t.name for t in self._config.tasks]
        self.editor.load_task(task, available_tasks=available)
        self.editor.show()
        self._log(f"Editing: {task.name}")

    def _on_editor_apply(self, task: AutoLamellaTaskDescription) -> None:
        self.workflow_widget.refresh_task(task)
        self.editor.hide()
        self._log(f"Applied: {task.name} | required={task.required} | requires={task.requires}")

    def _on_editor_cancel(self) -> None:
        self.editor.hide()
        self._log("Edit cancelled")

    def _on_order_changed(self, tasks: list) -> None:
        self._config.tasks[:] = tasks
        self._log(f"Reordered: {[t.name for t in tasks]}")

    def _print_workflow(self) -> None:
        tasks = self.workflow_widget.get_tasks()
        print("\n── Workflow ──────────────────────────")
        for i, t in enumerate(tasks):
            mode = "supervised" if t.supervise else "automated"
            req = ", ".join(t.requires) if t.requires else "—"
            opt = "" if t.required else " [optional]"
            print(f"  {i+1}. {t.name}{opt}  |  {mode}  |  requires: {req}")
        print("─────────────────────────────────────\n")
        self._log(f"Printed {len(tasks)} tasks to console")

    def _print_selected(self) -> None:
        tasks = self.workflow_widget.get_selected()
        print("\n── Selected ──────────────────────────")
        for i, t in enumerate(tasks):
            print(f"  {i+1}. {t.name}")
        print("─────────────────────────────────────\n")
        self._log(f"Printed {len(tasks)} selected tasks to console")

    def _add_task(self) -> None:
        import random
        verbs = ["Acquire", "Mill", "Align", "Inspect", "Record"]
        nouns = ["Fiducial", "Undercut", "Notch", "Overview", "Tilt Series"]
        name = f"{random.choice(verbs)} {random.choice(nouns)}"
        requires = random.sample([t.name for t in self._config.tasks], k=random.randint(0, 2))
        task = AutoLamellaTaskDescription(
            name=name,
            supervise=random.random() < 0.5,
            required=True,
            requires=requires,
        )
        self._config.tasks.append(task)
        self.workflow_widget.add_task(task)
        self._counter += 1
        self._log(f"Added: {name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
