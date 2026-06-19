"""Grid protocol editor — the Grids-tab Protocol sub-tab.

Configured **task instances** (left) + the selected instance's custom config
editor (right). Unlike a fixed one-row-per-type list, the protocol may hold
multiple instances of the same task type (e.g. two "Acquire Image" tasks at
different magnifications), so the list is driven by the experiment's
``grid_protocol.task_config`` rather than the registry.

The task list mirrors the AutoLamella protocol editor: a styled ``TaskNameList``
with add/remove, plus an Add dialog (task-type + name with duplicate
validation). Each instance is keyed by a unique ``task_name`` (the dict key),
fixed at creation — there is no rename, since the name doubles as the on-disk
results directory (a post-run rename would orphan prior results). The task_type
chooses both the editor widget and the runner class. Run ordering is handled in
the Workflow tab, not here. The grid protocol is global for now (see design doc
s9).
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
)
from fibsem.ui.widgets.custom_widgets import TaskNameListWidget
from fibsem.ui.widgets.grid_task_config_widgets import (
    GridTaskConfigWidget,
    get_grid_config_widget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment


def _unique_name(base: str, existing) -> str:
    """Return ``base``, or ``base (N)`` for the lowest N≥2 not in ``existing``."""
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"


class AddGridTaskDialog(QDialog):
    """Dialog for adding a new grid task instance (task type + unique name).

    When ``lock_task_type`` is set (the duplicate path), the task-type combo is
    pre-selected and disabled, and ``default_name`` seeds the editable name.
    """

    def __init__(
        self,
        existing_names: Dict[str, object],
        parent: Optional[QWidget] = None,
        task_type: Optional[str] = None,
        lock_task_type: bool = False,
        default_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self._existing = existing_names
        self.setWindowTitle("Duplicate Grid Task" if lock_task_type else "Add Grid Task")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.label_task_type = QLabel("Task Type:")
        self.comboBox_task_type = QComboBox()
        for tt, task_cls in GRID_TASK_REGISTRY.items():
            display = getattr(task_cls.config_cls, "display_name", tt)
            self.comboBox_task_type.addItem(f"{display} ({tt})", tt)

        # preset / lock the task type before wiring signals (avoids a premature
        # default-name override)
        if task_type is not None:
            idx = self.comboBox_task_type.findData(task_type)
            if idx >= 0:
                self.comboBox_task_type.setCurrentIndex(idx)
        if lock_task_type:
            self.comboBox_task_type.setEnabled(False)

        self.label_task_name = QLabel("Task Name:")
        self.lineEdit_task_name = QLineEdit()

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange; font-weight: bold;")

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        form = QGridLayout()
        form.addWidget(self.label_task_type, 0, 0)
        form.addWidget(self.comboBox_task_type, 0, 1)
        form.addWidget(self.label_task_name, 1, 0)
        form.addWidget(self.lineEdit_task_name, 1, 1)
        layout.addLayout(form)
        layout.addWidget(self.label_warning)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.lineEdit_task_name.textChanged.connect(self.validate_task_name)
        self.comboBox_task_type.currentIndexChanged.connect(self.update_default_task_name)
        if default_name is not None:
            self.lineEdit_task_name.setText(default_name)
        else:
            self.update_default_task_name()

    def update_default_task_name(self):
        task_type = self.comboBox_task_type.currentData()
        task_cls = GRID_TASK_REGISTRY.get(task_type)
        if task_cls is not None:
            display = getattr(task_cls.config_cls, "display_name", task_type)
            # default to a unique name so repeated adds don't collide
            self.lineEdit_task_name.setText(_unique_name(display, self._existing))

    def validate_task_name(self) -> bool:
        name = self.lineEdit_task_name.text().strip()
        if not name:
            self.label_warning.setText("")
            return False
        if name in self._existing:
            self.label_warning.setText(f"⚠ Warning: Task name '{name}' already exists!")
            return False
        self.label_warning.setText("")
        return True

    def validate_and_accept(self):
        if not self.lineEdit_task_name.text().strip():
            self.label_warning.setText("⚠ Warning: Task name cannot be empty!")
            return
        if not self.validate_task_name():
            return
        self.accept()

    def get_task_info(self) -> Tuple[str, str]:
        return self.comboBox_task_type.currentData(), self.lineEdit_task_name.text().strip()


class GridProtocolEditorWidget(QWidget):
    """Edit the grid protocol's configured task instances (add/duplicate/remove),
    keyed by unique task_name. Run ordering lives in the Workflow tab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional["Experiment"] = None
        self._editor: Optional[GridTaskConfigWidget] = None
        self._task_name: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        # left column: TaskNameList (header with add/duplicate/remove)
        left = QWidget()
        left.setMaximumWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self._task_list = TaskNameListWidget()
        self._task_list.set_buttons_visible(add=True, remove=True, duplicate=True)
        self._task_list.task_selected.connect(self._on_task_selected)
        self._task_list.add_clicked.connect(self._on_add)
        self._task_list.duplicate_clicked.connect(self._on_duplicate)
        self._task_list.remove_clicked.connect(self._on_remove)
        left_layout.addWidget(self._task_list)

        h.addWidget(left)

        # right column: scrollable editor host
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._editor_host = QWidget()
        self._editor_layout = QVBoxLayout(self._editor_host)
        self._editor_layout.setContentsMargins(4, 4, 4, 4)
        self._editor_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._editor_host)
        h.addWidget(self._scroll, 1)

        self._empty = QLabel("Load an experiment, then add a task to edit its config.")
        self._empty.setStyleSheet("color: #808080; padding: 24px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor_layout.addWidget(self._empty)

        self._update_buttons()

    # --- public API ---

    def set_experiment(self, experiment: Optional["Experiment"]) -> None:
        self._experiment = experiment
        self._rebuild_list()

    # --- helpers ---

    @property
    def _task_config(self):
        return self._experiment.grid_protocol.task_config

    def _names(self) -> List[str]:
        return list(self._task_config.keys()) if self._experiment is not None else []

    def _unique_name(self, base: str) -> str:
        return _unique_name(base, set(self._names()))

    # --- list management ---

    def _rebuild_list(self, select: Optional[str] = None) -> None:
        """Repopulate the instance list from the protocol and select ``select``
        (falling back to row 0). Deterministic selection — unlike
        ``TaskNameListWidget.set_tasks``, which would keep the stale selection."""
        if select is None:
            select = self._task_name
        names = self._names()
        target = select if select in names else (names[0] if names else None)
        lst = self._task_list._list
        lst.blockSignals(True)
        lst.clear()
        for name in names:
            lst.addItem(name)
        if target is not None:
            self._task_list.select(target)
        lst.blockSignals(False)
        if target is not None:
            self._on_task_selected(target)
        else:
            # nothing to edit (no experiment, or experiment with no tasks) —
            # show the prompt
            self._task_name = None
            self._clear_editor()
            self._empty.setVisible(True)
        self._update_buttons()

    def _on_task_selected(self, name: str) -> None:
        if self._experiment is None or not name or name not in self._task_config:
            self._update_buttons()
            return
        self._task_name = name
        self._clear_editor()
        self._empty.setVisible(False)

        config = self._task_config[name]
        self._editor = get_grid_config_widget(config)
        self._editor.config_changed.connect(self._persist)
        self._editor_layout.addWidget(self._editor)
        self._update_buttons()

    # --- toolbar actions ---

    def _on_add(self) -> None:
        if self._experiment is None:
            return
        dialog = AddGridTaskDialog(dict(self._task_config), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        task_type, name = dialog.get_task_info()
        if task_type and name:
            self.add_task(task_type, name)

    def add_task(self, task_type: str, name: Optional[str] = None) -> Optional[str]:
        """Add a task instance of ``task_type`` under a unique ``name`` (generated
        from the task's display name if omitted). The name is always uniquified
        so an existing instance is never silently overwritten. Returns the name
        added."""
        if self._experiment is None:
            return None
        config_cls = GRID_TASK_REGISTRY[task_type].config_cls
        # uniquify regardless of source — guards the public API against an
        # explicit colliding name overwriting an existing instance
        name = self._unique_name(name or getattr(config_cls, "display_name", task_type))
        config = config_cls(task_name=name)
        self._task_config[name] = config
        self._experiment.save()
        self._rebuild_list(select=name)
        return name

    def _on_duplicate(self) -> None:
        if self._experiment is None or self._task_name is None:
            return
        src = self._task_config[self._task_name]
        dialog = AddGridTaskDialog(
            dict(self._task_config),
            parent=self,
            task_type=src.task_type,
            lock_task_type=True,
            default_name=self._unique_name(self._task_name),
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        _, name = dialog.get_task_info()
        if name:
            self.duplicate_task(name)

    def duplicate_task(self, name: Optional[str] = None) -> Optional[str]:
        """Clone the selected task instance under a unique ``name`` (auto-generated
        if omitted), copying its config values. Returns the name added."""
        if self._experiment is None or self._task_name is None:
            return None
        src = self._task_config[self._task_name]
        # always uniquify (defaulting to the source name) so the clone never
        # overwrites an existing instance
        name = self._unique_name(name or self._task_name)
        config = deepcopy(src)
        config.task_name = name
        self._task_config[name] = config
        self._experiment.save()
        self._rebuild_list(select=name)
        return name

    def _on_remove(self) -> None:
        if self._experiment is None or self._task_name is None:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove the task '{self._task_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_task()

    def remove_task(self, name: Optional[str] = None) -> None:
        """Remove a task instance (the selected one if ``name`` is omitted)."""
        if self._experiment is None:
            return
        name = name or self._task_name
        if name is None or name not in self._task_config:
            return
        names = self._names()
        idx = names.index(name)
        del self._task_config[name]
        self._experiment.save()
        remaining = self._names()
        nxt = remaining[min(idx, len(remaining) - 1)] if remaining else None
        self._rebuild_list(select=nxt)

    # --- persistence ---

    def _persist(self) -> None:
        if self._experiment is None or self._editor is None or self._task_name is None:
            return
        self._task_config[self._task_name] = self._editor.get_config()
        self._experiment.save()

    # --- ui state ---

    def _update_buttons(self) -> None:
        has_exp = self._experiment is not None
        has_sel = has_exp and self._task_name in self._names()
        self._task_list.btn_add.setEnabled(has_exp)
        self._task_list.btn_remove.setEnabled(has_sel)
        self._task_list.btn_duplicate.setEnabled(has_sel)

    def _clear_editor(self) -> None:
        if self._editor is not None:
            # disconnect + detach now; deleteLater is async and would otherwise
            # leave a second editor still wired to _persist until the event loop
            # processes the deletion
            try:
                self._editor.config_changed.disconnect(self._persist)
            except (TypeError, RuntimeError):
                pass
            self._editor_layout.removeWidget(self._editor)
            self._editor.deleteLater()
            self._editor = None
        # detach the empty placeholder so it can be re-added cleanly
        self._empty.setParent(None)
        self._editor_layout.addWidget(self._empty)
