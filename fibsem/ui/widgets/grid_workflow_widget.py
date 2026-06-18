"""Grid workflow selection widget — the grid analogue of the lamella workflow
selection UI, but lean: two checklists (grids + tasks) with select-all.

Lives as the "Grids" sub-tab of the main Workflow tab. The host reads the
selections and runs them through GridTaskManager (sharing the Run/Stop +
timeline infra with the lamella workflow). No task ordering / supervise /
scheduling yet — that arrives if/when grid workflows need it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.grid_tasks import GRID_TASK_REGISTRY
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import GridRecord


def _grid_task_choices() -> List[Tuple[str, str]]:
    """(task_type, display_name) for each registered grid task."""
    choices = []
    for task_type, task_cls in GRID_TASK_REGISTRY.items():
        display = getattr(task_cls.config_cls, "display_name", task_type)
        choices.append((task_type, display))
    return choices


class _CheckList(QWidget):
    """A titled list of checkable items with a select-all toggle."""

    changed = pyqtSignal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 2)
        self._select_all = QCheckBox("Select all")
        self._select_all.stateChanged.connect(self._on_select_all)
        bar.addWidget(self._select_all)
        bar.addStretch(1)
        layout.addLayout(bar)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        self._empty = QLabel("Nothing to select.")
        self._empty.setStyleSheet("color: #606060; font-style: italic; padding: 8px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._panel = TitledPanel(title, content=inner, collapsible=False)
        outer.addWidget(self._panel)
        self._update_empty()

    def set_items(self, items: List[Tuple[str, object]]) -> None:
        """Populate with (label, value) pairs, preserving checks by label."""
        checked = self.checked_labels()
        self._list.blockSignals(True)
        self._list.clear()
        for label, value in items:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, value)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if label in checked else Qt.Unchecked)
            self._list.addItem(it)
        self._list.blockSignals(False)
        self._update_empty()
        self._sync_select_all()
        self.changed.emit()

    def checked_values(self) -> List[object]:
        return [
            self._list.item(i).data(Qt.UserRole)
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.Checked
        ]

    def checked_labels(self) -> set:
        return {
            self._list.item(i).text()
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.Checked
        }

    # --- internals ---

    def _on_item_changed(self, _item) -> None:
        self._sync_select_all()
        self.changed.emit()

    def _on_select_all(self, state) -> None:
        # only fires on a genuine user toggle — _sync_select_all blocks signals
        target = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(target)
        self._list.blockSignals(False)
        self.changed.emit()

    def _sync_select_all(self) -> None:
        n = self._list.count()
        checked = len(self.checked_values())
        self._select_all.blockSignals(True)
        if n and checked == n:
            self._select_all.setCheckState(Qt.Checked)
        elif checked == 0:
            self._select_all.setCheckState(Qt.Unchecked)
        else:
            self._select_all.setCheckState(Qt.PartiallyChecked)
        self._select_all.blockSignals(False)

    def _update_empty(self) -> None:
        empty = self._list.count() == 0
        self._empty.setVisible(empty)
        self._list.setVisible(not empty)


class GridWorkflowWidget(QWidget):
    """Grids + tasks selection for running a grid workflow."""

    grid_selection_changed = pyqtSignal(list)  # List[GridRecord]
    task_selection_changed = pyqtSignal(list)  # List[str] task_type keys

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._grids = _CheckList("Grids")
        self._grids.changed.connect(
            lambda: self.grid_selection_changed.emit(self.get_selected_grids())
        )
        layout.addWidget(self._grids)

        self._tasks = _CheckList("Tasks")
        self._tasks.changed.connect(
            lambda: self.task_selection_changed.emit(self.get_selected_tasks())
        )
        layout.addWidget(self._tasks)

        # tasks come from the registry (static)
        self._tasks.set_items([(disp, key) for key, disp in _grid_task_choices()])

    # --- public API ---

    def set_grids(self, grids: List["GridRecord"]) -> None:
        self._grids.set_items([(g.name, g) for g in grids])

    def get_selected_grids(self) -> List["GridRecord"]:
        return list(self._grids.checked_values())

    def get_selected_tasks(self) -> List[str]:
        return list(self._tasks.checked_values())
