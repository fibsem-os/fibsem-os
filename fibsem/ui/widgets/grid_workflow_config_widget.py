"""Ordered grid-workflow task list: run order + per-task supervise + select.

Grid analogue of the lamella ``WorkflowConfigWidget`` (and matches its
formatting): a draggable list of task rows — select checkbox · name · supervise
toggle · drag handle. Drag a row to reorder the run order. It auto-mirrors the
protocol's configured task instances (membership is managed in the Protocol
tab), so it only orders them, toggles supervise, and selects which to run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        GridTaskDescription,
        GridWorkflowConfig,
    )

# match the lamella WorkflowConfigWidget formatting
_DRAG_HANDLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "drag_handle.svg")
_NAME_MIN_WIDTH = 180
_BTN_SIZE = 32
_ROW_HEIGHT = 40

# the supervise toggle is checkable, but should read as automated/supervised via
# its icon only — not a checked-state background (matches the lamella button,
# which is a plain non-checkable QToolButton). Keep hover, drop :checked styling.
_SUPERVISE_BTN_STYLE = """
    QToolButton {
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 2px 6px;
        background-color: transparent;
    }
    QToolButton:hover {
        border: 1px solid #6a6a6a;
        background-color: rgba(255, 255, 255, 25);
    }
    QToolButton:checked {
        border: 1px solid transparent;
        background-color: transparent;
    }
"""


class _DraggableTaskList(QListWidget):
    """QListWidget with InternalMove drag-and-drop; emits the new order on drop.

    Qt drops itemWidget associations when items move, so the parent listens to
    ``reordered`` and rebuilds the row widgets.
    """

    reordered = pyqtSignal(list)  # List[GridTaskDescription]

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        tasks = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(tasks)


class _GridTaskRow(QWidget):
    """One workflow task: select checkbox · name · supervise toggle · drag handle."""

    selection_changed = pyqtSignal(object, bool)  # (description, checked)
    supervised_changed = pyqtSignal(object)        # GridTaskDescription

    def __init__(self, task: "GridTaskDescription", checked: bool = False,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.task = task
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self.checkbox)

        self.name_label = QLabel(task.name)
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        layout.addStretch(1)

        self.btn_supervise = IconToolButton(
            icon="mdi:lightning-bolt-circle", color=stylesheets.AUTOMATED_COLOR,
            checked_icon="mdi:account-hard-hat", checked_color=stylesheets.PRIMARY_COLOR,
            tooltip="Automated (click to supervise)",
            checked_tooltip="Supervised (click to automate)",
            checkable=True, checked=task.supervise, size=_BTN_SIZE,
        )
        self.btn_supervise.setStyleSheet(_SUPERVISE_BTN_STYLE)  # no checked-state bg
        self.btn_supervise.toggled.connect(self._on_supervise)
        layout.addWidget(self.btn_supervise)

        drag_icon = QLabel()
        drag_icon.setFixedSize(10, 16)
        drag_icon.setPixmap(QPixmap(_DRAG_HANDLE_PATH).scaled(
            10, 16, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setToolTip("Drag to reorder")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.task, bool(s))
        )

    def _on_supervise(self, checked: bool) -> None:
        self.task.supervise = checked
        self.supervised_changed.emit(self.task)


class _GridWorkflowHeader(QWidget):
    """Header bar with the Select-All checkbox (matches the lamella header)."""

    select_all_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        self.checkbox_all = QCheckBox("Select All")
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)
        layout.addStretch(1)
        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(s == Qt.Checked)
        )


class GridWorkflowConfigWidget(QWidget):
    """Render + edit a GridWorkflowConfig (run order, supervise, run selection)."""

    selection_changed = pyqtSignal(list)     # ordered selected task names
    supervised_changed = pyqtSignal(object)  # GridTaskDescription
    order_changed = pyqtSignal(list)         # new order (task names)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config: Optional["GridWorkflowConfig"] = None
        self._checked: Dict[str, bool] = {}  # task_name -> checked
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _GridWorkflowHeader()
        self._header.select_all_changed.connect(self._on_select_all)
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableTaskList()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.reordered.connect(self._on_reordered)
        layout.addWidget(self._list)

    # --- public API ---

    def set_config(self, config: Optional["GridWorkflowConfig"]) -> None:
        self._config = config
        self._list.clear()
        tasks = list(config.tasks) if config is not None else []
        for task in tasks:
            self._add_row(task)
        self._sync_select_all()

    def selected_tasks(self) -> List[str]:
        """Checked task names, in run order."""
        return [self._row(i).task.name for i in range(self._list.count())
                if self._row(i).checkbox.isChecked()]

    # --- internals ---

    def _row(self, i: int) -> _GridTaskRow:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    @property
    def _rows(self) -> List[_GridTaskRow]:
        return [self._row(i) for i in range(self._list.count())]

    def _add_row(self, task: "GridTaskDescription") -> None:
        row = _GridTaskRow(task, checked=self._checked.get(task.name, False))
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, task)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        self._connect_row(row)

    def _connect_row(self, row: _GridTaskRow) -> None:
        row.selection_changed.connect(self._on_row_select)
        row.supervised_changed.connect(self.supervised_changed)

    def _on_row_select(self, task: "GridTaskDescription", checked: bool) -> None:
        self._checked[task.name] = checked
        self._sync_select_all()
        self.selection_changed.emit(self.selected_tasks())

    def _on_reordered(self, tasks: List["GridTaskDescription"]) -> None:
        """Rebuild row widgets after a drag reorder and update the config order.

        Qt clears itemWidget associations on internal move, so re-create each row
        from the task stored in the item's UserRole data.
        """
        if self._config is not None:
            self._config.tasks = list(tasks)  # evented reassignment
        for i, task in enumerate(tasks):
            item = self._list.item(i)
            if item is None:
                continue
            row = _GridTaskRow(task, checked=self._checked.get(task.name, False))
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.setItemWidget(item, row)
            self._connect_row(row)
        self._sync_select_all()
        self.order_changed.emit([t.name for t in tasks])

    def _on_select_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
            self._checked[row.task.name] = checked
        self.selection_changed.emit(self.selected_tasks())

    def _sync_select_all(self) -> None:
        n = self._list.count()
        c = len(self.selected_tasks())
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        cb.setTristate(True)
        if n and c == n:
            cb.setCheckState(Qt.CheckState.Checked)
        elif c == 0:
            cb.setCheckState(Qt.CheckState.Unchecked)
        else:
            cb.setCheckState(Qt.CheckState.PartiallyChecked)
        cb.setTristate(False)
        cb.blockSignals(False)
