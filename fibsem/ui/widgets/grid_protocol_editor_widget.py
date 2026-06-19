"""Grid protocol editor — the Grids-tab Protocol sub-tab.

Task list (left) + the selected task's custom config editor (right). Edits
persist to the experiment's shared ``grid_protocol.task_config`` (keyed by
task_type) and save. The grid protocol is global for now (see design doc s9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (
    GRID_TASK_REGISTRY,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.grid_task_config_widgets import (
    GridTaskConfigWidget,
    get_grid_config_widget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment


class GridProtocolEditorWidget(QWidget):
    """Edit the shared grid protocol's per-task configs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional["Experiment"] = None
        self._editor: Optional[GridTaskConfigWidget] = None
        self._task_type: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self._list = QListWidget()
        self._list.setMaximumWidth(220)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        for task_type, task_cls in GRID_TASK_REGISTRY.items():
            display = getattr(task_cls.config_cls, "display_name", task_type)
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, task_type)
            self._list.addItem(item)
        self._list.currentRowChanged.connect(self._on_task_selected)
        h.addWidget(self._list)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._editor_host = QWidget()
        self._editor_layout = QVBoxLayout(self._editor_host)
        self._editor_layout.setContentsMargins(4, 4, 4, 4)
        self._editor_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._editor_host)
        h.addWidget(self._scroll, 1)

        self._empty = QLabel("Load an experiment to edit the grid protocol.")
        self._empty.setStyleSheet("color: #808080; padding: 24px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor_layout.addWidget(self._empty)

    # --- public API ---

    def set_experiment(self, experiment: Optional["Experiment"]) -> None:
        self._experiment = experiment
        if experiment is None:
            self._clear_editor()
            self._editor_layout.addWidget(self._empty)
            self._empty.setVisible(True)
            return
        if self._list.count() and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)
        else:
            self._on_task_selected(self._list.currentRow())

    # --- internals ---

    def _on_task_selected(self, row: int) -> None:
        if self._experiment is None or row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        self._task_type = item.data(Qt.UserRole)
        self._clear_editor()
        self._empty.setVisible(False)

        config = self._config_for(self._task_type)
        self._editor = get_grid_config_widget(config)
        self._editor.config_changed.connect(self._persist)
        self._editor_layout.addWidget(self._editor)

    def _config_for(self, task_type: str):
        """Return the protocol's saved config for this task, or a fresh default."""
        proto = self._experiment.grid_protocol
        cfg = proto.task_config.get(task_type)
        if cfg is not None and cfg.task_type == task_type:
            return cfg
        return GRID_TASK_REGISTRY[task_type].config_cls(task_name=task_type)

    def _persist(self) -> None:
        if self._experiment is None or self._editor is None or self._task_type is None:
            return
        self._experiment.grid_protocol.task_config[self._task_type] = self._editor.get_config()
        self._experiment.save()

    def _clear_editor(self) -> None:
        if self._editor is not None:
            self._editor.deleteLater()
            self._editor = None
        # detach the empty placeholder so it can be re-added cleanly
        self._empty.setParent(None)
