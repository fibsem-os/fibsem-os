"""Picking a type and a name for a new task, from an app's task registry."""

from typing import Collection, Dict, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

__all__ = ["AddTaskDialog"]


class AddTaskDialog(QDialog):
    """Pick a task type and a name for a new task: an app's tasks on its
    Protocol tab, the grid tasks' on AutoLamella's Grid page.

    *config_classes* is task type -> config class, from the registry, so a new
    task type shows up without a change. The name offered is the type's default
    (*default_names*, else its display name), numbered past the *existing*
    names. ``chosen`` is (task type, name) once accepted.
    """

    def __init__(
        self,
        config_classes: Dict[str, type],
        existing: Collection[str],
        title: str = "Add task",
        default_names: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self._title = title
        self._config_classes = config_classes
        self._existing = existing
        self._default_names = default_names or {}
        self.chosen: Optional[Tuple[str, str]] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.type_combo = QComboBox()
        for task_type, config_cls in config_classes.items():
            self.type_combo.addItem(config_cls.display_name, task_type)
            self.type_combo.setItemData(
                self.type_combo.count() - 1, task_type, Qt.ToolTipRole
            )
        self.type_combo.currentIndexChanged.connect(self._suggest_name)
        form.addRow("Task", self.type_combo)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "Unique within the protocol; the task's outputs are filed under it"
        )
        form.addRow("Name", self.name_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._suggest_name()

    def _suggest_name(self) -> None:
        task_type = self.type_combo.currentData()
        if task_type is None:
            return
        base = self._default_names.get(
            task_type, self._config_classes[task_type].display_name
        )
        # numbered as the name is written: "Mill Rough 2", "overview_sem_2"
        separator = " " if " " in base else "_"
        name, n = base, 2
        while name in self._existing:
            name, n = f"{base}{separator}{n}", n + 1
        self.name_edit.setText(name)

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, self._title, "Give the task a name.")
            return
        if name in self._existing:
            QMessageBox.warning(
                self, self._title, f"There is already a task named {name}."
            )
            return
        self.chosen = (self.type_combo.currentData(), name)
        self.accept()
