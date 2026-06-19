"""Ordered grid-workflow task list: run order + per-task supervise + select.

Grid analogue of the lamella ``WorkflowConfigWidget``, but lean: it auto-mirrors
the protocol's configured task instances (membership is managed in the Protocol
tab), so it only orders them, toggles supervise, and selects which to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        GridTaskDescription,
        GridWorkflowConfig,
    )


class _GridTaskRow(QWidget):
    """One workflow task: select checkbox · name · supervise toggle · ▲/▼."""

    select_changed = pyqtSignal()
    supervised_changed = pyqtSignal(object)   # GridTaskDescription
    move_requested = pyqtSignal(object, int)  # (description, delta)

    def __init__(self, task: "GridTaskDescription", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.task = task
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        self.checkbox = QCheckBox()
        self.checkbox.toggled.connect(lambda _=None: self.select_changed.emit())
        layout.addWidget(self.checkbox)

        self.label = QLabel(task.name)
        layout.addWidget(self.label, 1)

        self.btn_supervise = IconToolButton(
            icon="mdi:lightning-bolt-circle", color=stylesheets.AUTOMATED_COLOR,
            checked_icon="mdi:account-hard-hat", checked_color=stylesheets.PRIMARY_COLOR,
            tooltip="Automated (click to supervise)",
            checked_tooltip="Supervised (click to automate)",
            checkable=True, checked=task.supervise, size=24,
        )
        self.btn_supervise.toggled.connect(self._on_supervise)
        layout.addWidget(self.btn_supervise)

        self.btn_up = IconToolButton("mdi:chevron-up", tooltip="Move up (earlier)", size=24)
        self.btn_up.clicked.connect(lambda: self.move_requested.emit(self.task, -1))
        layout.addWidget(self.btn_up)
        self.btn_down = IconToolButton("mdi:chevron-down", tooltip="Move down (later)", size=24)
        self.btn_down.clicked.connect(lambda: self.move_requested.emit(self.task, 1))
        layout.addWidget(self.btn_down)

    def _on_supervise(self, checked: bool) -> None:
        self.task.supervise = checked
        self.supervised_changed.emit(self.task)


class GridWorkflowConfigWidget(QWidget):
    """Render + edit a GridWorkflowConfig (run order, supervise, run selection)."""

    selection_changed = pyqtSignal(list)     # ordered selected task names
    supervised_changed = pyqtSignal(object)  # GridTaskDescription
    order_changed = pyqtSignal(list)         # new order (task names)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config: Optional["GridWorkflowConfig"] = None
        self._rows: List[_GridTaskRow] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 2)
        self._select_all = QCheckBox("Select all")
        self._select_all.stateChanged.connect(self._on_select_all)
        bar.addWidget(self._select_all)
        bar.addStretch(1)
        v.addLayout(bar)

        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        v.addWidget(self._rows_host)

        self._empty = QLabel("No tasks configured. Add tasks in the Protocol tab.")
        self._empty.setStyleSheet("color: #606060; font-style: italic; padding: 8px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._empty)
        v.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(TitledPanel("Tasks", content=inner, collapsible=False))

    # --- public API ---

    def set_config(self, config: Optional["GridWorkflowConfig"]) -> None:
        self._config = config
        self._rebuild()

    def selected_tasks(self) -> List[str]:
        """Checked task names, in run order."""
        return [r.task.name for r in self._rows if r.checkbox.isChecked()]

    # --- internals ---

    def _rebuild(self) -> None:
        checked = set(self.selected_tasks())  # preserve checks by name
        for r in self._rows:
            r.setParent(None)
            r.deleteLater()
        self._rows = []

        tasks = list(self._config.tasks) if self._config is not None else []
        for i, task in enumerate(tasks):
            row = _GridTaskRow(task)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(task.name in checked)
            row.checkbox.blockSignals(False)
            row.btn_up.setEnabled(i > 0)
            row.btn_down.setEnabled(i < len(tasks) - 1)
            row.select_changed.connect(self._on_row_select)
            row.supervised_changed.connect(self.supervised_changed)
            row.move_requested.connect(self._on_move)
            self._rows_layout.addWidget(row)
            self._rows.append(row)

        self._empty.setVisible(not tasks)
        self._rows_host.setVisible(bool(tasks))
        self._sync_select_all()

    def _on_row_select(self) -> None:
        self._sync_select_all()
        self.selection_changed.emit(self.selected_tasks())

    def _on_move(self, task: "GridTaskDescription", delta: int) -> None:
        if self._config is None:
            return
        tasks = list(self._config.tasks)
        i = next((k for k, t in enumerate(tasks) if t is task), None)
        if i is None:
            return
        j = i + delta
        if j < 0 or j >= len(tasks):
            return
        tasks[i], tasks[j] = tasks[j], tasks[i]
        self._config.tasks = tasks  # evented reassignment
        self._rebuild()
        self.order_changed.emit(self._config.order)

    def _on_select_all(self, state) -> None:
        target = state == Qt.Checked
        for r in self._rows:
            r.checkbox.blockSignals(True)
            r.checkbox.setChecked(target)
            r.checkbox.blockSignals(False)
        self.selection_changed.emit(self.selected_tasks())

    def _sync_select_all(self) -> None:
        n = len(self._rows)
        c = len(self.selected_tasks())
        self._select_all.blockSignals(True)
        if n and c == n:
            self._select_all.setCheckState(Qt.Checked)
        elif c == 0:
            self._select_all.setCheckState(Qt.Unchecked)
        else:
            self._select_all.setCheckState(Qt.PartiallyChecked)
        self._select_all.blockSignals(False)
