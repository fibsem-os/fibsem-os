"""Test script for LamellaWorkflowWidget."""
import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    DefectState,
    Lamella,
)
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.lamella_workflow_widget import LamellaWorkflowWidget

# ── sample data ──────────────────────────────────────────────────────────────

SAMPLE_CONFIG = AutoLamellaWorkflowConfig(
    name="Standard Cryo-Lamella",
    description="Standard workflow for cryo-lamella preparation",
    tasks=[
        AutoLamellaTaskDescription(name="Acquire Reference Image", supervise=True, required=True),
        AutoLamellaTaskDescription(name="Mill Rough", supervise=False, required=True),
        AutoLamellaTaskDescription(name="Mill Polishing", supervise=True, required=True, requires=["Mill Rough"]),
        AutoLamellaTaskDescription(name="Acquire Final Image", supervise=False, required=False),
    ],
)


def _make_lamella(number, petname, last_task="", in_progress="", has_defect=False, requires_rework=False):
    lam = Lamella(path=Path(f"/tmp/test/{petname}"), number=number, petname=petname)
    if last_task:
        state = AutoLamellaTaskState(name=last_task, status=AutoLamellaTaskStatus.Completed)
        state.end_timestamp = time.time()
        lam.task_history.append(state)
    if in_progress:
        lam.task_state.name = in_progress
        lam.task_state.status = AutoLamellaTaskStatus.InProgress
    if has_defect:
        lam.defect = DefectState(has_defect=True, requires_rework=requires_rework, description="test defect")
    return lam


SAMPLE_LAMELLA = [
    _make_lamella(1, "01-humble-molly"),
    _make_lamella(2, "01-hearty-wombat", last_task="Acquire Reference Image"),
    _make_lamella(3, "02-jolly-koala", last_task="Mill Rough", in_progress="Mill Polishing"),
    _make_lamella(4, "03-brave-falcon", last_task="Mill Rough", has_defect=True, requires_rework=True),
]


# ── window ────────────────────────────────────────────────────────────────────

class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lamella + Workflow Widget — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.setStyleSheet(NAPARI_STYLE)
        self.resize(700, 640)

        self._config = SAMPLE_CONFIG
        self._lamella_counter = len(SAMPLE_LAMELLA) + 1
        self._task_counter = len(self._config.tasks) + 1

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Lamella + Workflow")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        self.widget = LamellaWorkflowWidget()
        for lam in SAMPLE_LAMELLA:
            self.widget.add_lamella(lam)
        self.widget.set_workflow_config(self._config)
        root.addWidget(self.widget, 1)

        self.log_label = QLabel("(events will appear here)")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        root.addWidget(self.log_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        for label, fn in [
            ("Add Lamella",    self._add_lamella),
            ("Add Task",       self._add_task),
            ("Print Workflow", self._print_workflow),
            ("Print Selected", self._print_selected),
            ("Clear All",      self.widget.clear),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            btn_row.addWidget(btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── signal connections ────────────────────────────────────────────
        self.widget.lamella_move_to_requested.connect(
            lambda lam: self._log(f"Move to: {lam.name}")
        )
        self.widget.lamella_edit_requested.connect(
            lambda lam: self._log(f"Edit lamella: {lam.name}")
        )
        self.widget.lamella_remove_requested.connect(
            lambda lam: self._log(f"Removed lamella: {lam.name}")
        )
        self.widget.lamella_selection_changed.connect(
            lambda sel: self._log(f"Lamella selected ({len(sel)}): {[l.name for l in sel]}")
        )
        self.widget.task_supervised_changed.connect(
            lambda t: self._log(f"Task supervised toggled: {t.name} → {t.supervise}")
        )
        self.widget.task_edited.connect(
            lambda t: self._log(f"Task edited: {t.name} | required={t.required} | requires={t.requires}")
        )
        self.widget.task_remove_requested.connect(
            lambda t: self._log(f"Removed task: {t.name}")
        )
        self.widget.task_order_changed.connect(self._on_task_order_changed)

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)

    def _on_task_order_changed(self, tasks: list) -> None:
        self._config.tasks[:] = tasks
        self._log(f"Reordered: {[t.name for t in tasks]}")

    def _add_lamella(self) -> None:
        import random
        adjectives = ["calm", "deft", "fierce", "gentle", "nimble"]
        animals = ["crane", "viper", "lynx", "bison", "heron"]
        name = f"{self._lamella_counter:02d}-{random.choice(adjectives)}-{random.choice(animals)}"
        lam = _make_lamella(self._lamella_counter, name)
        self.widget.add_lamella(lam)
        self._lamella_counter += 1
        self._log(f"Added lamella: {name}")

    def _add_task(self) -> None:
        import random
        verbs = ["Acquire", "Mill", "Align", "Inspect", "Record"]
        nouns = ["Fiducial", "Undercut", "Notch", "Overview", "Tilt Series"]
        name = f"{random.choice(verbs)} {random.choice(nouns)}"
        task = AutoLamellaTaskDescription(
            name=name,
            supervise=random.random() < 0.5,
            required=True,
        )
        self._config.tasks.append(task)
        self.widget.add_task(task)
        self._task_counter += 1
        self._log(f"Added task: {name}")

    def _print_workflow(self) -> None:
        tasks = self.widget.get_tasks()
        print("\n── Workflow ──────────────────────────")
        for i, t in enumerate(tasks):
            mode = "supervised" if t.supervise else "automated"
            req = ", ".join(t.requires) if t.requires else "—"
            opt = "" if t.required else " [optional]"
            print(f"  {i+1}. {t.name}{opt}  |  {mode}  |  requires: {req}")
        print("─────────────────────────────────────\n")
        self._log(f"Printed {len(tasks)} tasks to console")

    def _print_selected(self) -> None:
        lamella = self.widget.get_selected_lamella()
        tasks = self.widget.get_selected_tasks()
        print("\n── Selected Lamella ──────────────────")
        for i, l in enumerate(lamella):
            print(f"  {i+1}. {l.name}")
        print(f"\n── Selected Tasks ({len(tasks)}) ────────────────")
        for i, t in enumerate(tasks):
            print(f"  {i+1}. {t.name}")
        print("─────────────────────────────────────\n")
        self._log(f"Selected: {len(lamella)} lamella, {len(tasks)} tasks")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
