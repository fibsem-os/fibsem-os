from __future__ import annotations

from typing import Dict, List, Optional

from datetime import datetime
from typing import Optional as _Opt

from PyQt5.QtCore import QDateTime, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)

_NAME_MIN_WIDTH = 180
_BTN_SIZE = QSize(26, 26)
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 4 + 8 * 3  # schedule + supervise + edit + remove + 3 gaps

_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 2px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
"""


class _DraggableTaskList(QListWidget):
    """QListWidget with InternalMove drag-and-drop that emits the new task order after each drop.

    Qt removes itemWidget associations when items are moved, so the parent must
    listen to ``reordered`` and rebuild the row widgets.
    """

    reordered = pyqtSignal(list)  # List[AutoLamellaTaskDescription]

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        tasks = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(tasks)


def _supervise_icon(task: AutoLamellaTaskDescription) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the supervised/automated indicator."""
    if task.supervise:
        return "mdi:account-hard-hat", "#6aabdf", "Supervised"
    return "mdi:refresh-auto", "#80c080", "Automated"


class _ScheduleDialog(QDialog):
    """Small dialog for setting or clearing a task's scheduled_at time."""

    def __init__(
        self,
        task: "AutoLamellaTaskDescription",
        parent: _Opt[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Schedule: {task.name}")
        self.setModal(True)
        self.setMinimumWidth(300)
        self.setStyleSheet("background: #2b2d31; color: #d6d6d6;")

        self._cleared = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Schedule task to start at:"))

        self._dt_edit = QDateTimeEdit()
        self._dt_edit.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self._dt_edit.setCalendarPopup(True)
        initial = (
            QDateTime(
                task.scheduled_at.year,
                task.scheduled_at.month,
                task.scheduled_at.day,
                task.scheduled_at.hour,
                task.scheduled_at.minute,
            )
            if task.scheduled_at is not None
            else QDateTime.currentDateTime()
        )
        self._dt_edit.setDateTime(initial)
        layout.addWidget(self._dt_edit)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Remove scheduled time")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        btn_row.addStretch(1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        btn_row.addWidget(box)

        layout.addLayout(btn_row)

    def _on_clear(self) -> None:
        self._cleared = True
        self.accept()

    def get_scheduled_at(self) -> _Opt[datetime]:
        if self._cleared:
            return None
        return self._dt_edit.dateTime().toPyDateTime().replace(second=0, microsecond=0)


class WorkflowTaskRowWidget(QWidget):
    supervised_changed = pyqtSignal(object)       # AutoLamellaTaskDescription
    edit_clicked = pyqtSignal(object)             # AutoLamellaTaskDescription
    remove_clicked = pyqtSignal(object)           # AutoLamellaTaskDescription
    schedule_changed = pyqtSignal(object)         # AutoLamellaTaskDescription
    selection_changed = pyqtSignal(object, bool)  # AutoLamellaTaskDescription, checked

    def __init__(
        self,
        task: AutoLamellaTaskDescription,
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.task = task

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        layout.addWidget(self.checkbox)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        name_col.setContentsMargins(0, 0, 0, 0)

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        name_col.addWidget(self.name_label)

        self.requires_label = QLabel()
        self.requires_label.setStyleSheet("background: transparent; color: #707070; font-size: 10px;")
        name_col.addWidget(self.requires_label)

        layout.addLayout(name_col)

        layout.addStretch(1)

        self.btn_schedule = QToolButton()
        self.btn_schedule.setFixedSize(_BTN_SIZE)
        self.btn_schedule.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_schedule)

        self.btn_supervise = QToolButton()
        self.btn_supervise.setFixedSize(_BTN_SIZE)
        self.btn_supervise.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_supervise)

        self.btn_edit = QToolButton()
        self.btn_edit.setIcon(QIconifyIcon("mdi:pencil", color="#c0c0c0"))
        self.btn_edit.setToolTip("Edit")
        self.btn_edit.setFixedSize(_BTN_SIZE)
        self.btn_edit.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_edit)

        self.btn_remove = QToolButton()
        self.btn_remove.setIcon(QIconifyIcon("mdi:trash-can-outline", color="#c0c0c0"))
        self.btn_remove.setToolTip("Remove")
        self.btn_remove.setFixedSize(_BTN_SIZE)
        self.btn_remove.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_remove)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.task, bool(s))
        )
        self.btn_schedule.clicked.connect(self._on_schedule_clicked)
        self.btn_supervise.clicked.connect(self._on_supervise_clicked)
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.task))
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        self.refresh()

    def _on_schedule_clicked(self) -> None:
        dialog = _ScheduleDialog(self.task, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.task.scheduled_at = dialog.get_scheduled_at()
            self.refresh()
            self.schedule_changed.emit(self.task)

    def _on_supervise_clicked(self) -> None:
        self.task.supervise = not self.task.supervise
        self.refresh()
        self.supervised_changed.emit(self.task)

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove Task",
            f"Remove <b>{self.task.name}</b> from workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_clicked.emit(self.task)

    def refresh(self) -> None:
        """Re-read all display fields from the stored task."""
        self.name_label.setText(self.task.name)
        requires_text = ", ".join(self.task.requires) if self.task.requires else "No requirements"
        self.requires_label.setText(requires_text)
        icon_name, icon_color, tooltip = _supervise_icon(self.task)
        self.btn_supervise.setIcon(QIconifyIcon(icon_name, color=icon_color))
        self.btn_supervise.setToolTip(tooltip)
        if self.task.scheduled_at is not None:
            self.btn_schedule.setIcon(QIconifyIcon("mdi:clock", color="#f0c040"))
            self.btn_schedule.setToolTip(
                f"Scheduled: {self.task.scheduled_at.strftime('%Y-%m-%d  %H:%M')}"
            )
        else:
            self.btn_schedule.setIcon(QIconifyIcon("mdi:clock-outline", color="#606060"))
            self.btn_schedule.setToolTip("Not scheduled — click to set")


class _WorkflowTaskListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)
    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.checkbox_all = QCheckBox("Select All")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)

        layout.addStretch(1)

        # Spacer covers all row buttons except the last, so btn_add aligns with btn_remove
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width() - 8)
        layout.addWidget(spacer)

        self.btn_add = QToolButton()
        self.btn_add.setIcon(QIconifyIcon("mdi:plus", color="#c0c0c0"))
        self.btn_add.setToolTip("Add Task")
        self.btn_add.setFixedSize(_BTN_SIZE)
        self.btn_add.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_add)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )
        self.btn_add.clicked.connect(self.add_task_clicked)


class WorkflowConfigWidget(QWidget):
    """List widget displaying AutoLamellaWorkflowConfig tasks with name, supervised, edit and remove actions."""

    supervised_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    edit_requested = pyqtSignal(object)      # AutoLamellaTaskDescription
    remove_requested = pyqtSignal(object)    # AutoLamellaTaskDescription
    schedule_changed = pyqtSignal(object)    # AutoLamellaTaskDescription
    selection_changed = pyqtSignal(list)     # List[AutoLamellaTaskDescription]
    order_changed = pyqtSignal(list)         # List[AutoLamellaTaskDescription]
    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._btn_visible = {
            "schedule": False,
            "supervise": True,
            "edit": True,
            "remove": True,
        }
        self._checked: Dict[int, bool] = {}  # id(task) -> checked

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _WorkflowTaskListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableTaskList()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setSpacing(1)
        self._list.setStyleSheet("""
            QListWidget {
                background: #2b2d31;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background: #2b2d31;
                border-bottom: 1px solid #3a3d42;
            }
            QListWidget::item:alternate {
                background: #303338;
            }
            QListWidget::item:selected {
                background: rgba(0, 122, 204, 50);
            }
            QListWidget::item:selected:alternate {
                background: rgba(0, 122, 204, 60);
            }
        """)
        self._list.setAlternatingRowColors(True)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._header.select_all_changed.connect(self._on_select_all)
        self._header.add_task_clicked.connect(self.add_task_clicked)
        self._list.reordered.connect(self._on_reordered)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config: AutoLamellaWorkflowConfig) -> None:
        """Populate the list from an AutoLamellaWorkflowConfig."""
        self.clear()
        for task in config.tasks:
            self.add_task(task)

    def add_task(self, task: AutoLamellaTaskDescription, checked: bool = True) -> WorkflowTaskRowWidget:
        self._checked[id(task)] = checked
        row = WorkflowTaskRowWidget(task, checked)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, task)
        item.setSizeHint(row.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        self._connect_row(row)
        self._apply_btn_visibility(row)
        self._sync_select_all()
        return row

    def _connect_row(self, row: WorkflowTaskRowWidget) -> None:
        row.supervised_changed.connect(self.supervised_changed)
        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.schedule_changed.connect(self.schedule_changed)
        row.selection_changed.connect(self._on_row_selection_changed)

    def enable_supervise_button(self, visible: bool) -> None:
        self._btn_visible["supervise"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_supervise.setVisible(visible)

    def enable_edit_button(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_edit.setVisible(visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_remove.setVisible(visible)

    def remove_task(self, task: AutoLamellaTaskDescription) -> None:
        for i in range(self._list.count()):
            if self._row(i).task is task:
                self._list.takeItem(i)
                self._checked.pop(id(task), None)
                break
        self._sync_select_all()

    def refresh_task(self, task: AutoLamellaTaskDescription) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            if row.task is task:
                row.refresh()
                break

    def refresh_all(self) -> None:
        for i in range(self._list.count()):
            self._row(i).refresh()

    def get_tasks(self) -> List[AutoLamellaTaskDescription]:
        """Return tasks in current display order."""
        return [self._row(i).task for i in range(self._list.count())]

    def get_selected(self) -> List[AutoLamellaTaskDescription]:
        return [
            self._row(i).task
            for i in range(self._list.count())
            if self._row(i).checkbox.isChecked()
        ]

    def clear(self) -> None:
        self._list.clear()
        self._checked.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row(self, i: int) -> WorkflowTaskRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _apply_btn_visibility(self, row: WorkflowTaskRowWidget) -> None:
        row.btn_schedule.setVisible(self._btn_visible["schedule"])
        row.btn_supervise.setVisible(self._btn_visible["supervise"])
        row.btn_edit.setVisible(self._btn_visible["edit"])
        row.btn_remove.setVisible(self._btn_visible["remove"])

    def _on_remove_clicked(self, task: AutoLamellaTaskDescription) -> None:
        self.remove_task(task)
        self.remove_requested.emit(task)

    def _on_select_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
        self.selection_changed.emit(self.get_selected())

    def _on_row_selection_changed(self, task: AutoLamellaTaskDescription, checked: bool) -> None:
        self._checked[id(task)] = checked
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _on_reordered(self, tasks: List[AutoLamellaTaskDescription]) -> None:
        """Rebuild all row widgets after a drag-and-drop reorder.

        Qt clears itemWidget associations when items are moved internally, so
        we re-create each row from the task stored in the item's UserRole data.
        """
        for i, task in enumerate(tasks):
            item = self._list.item(i)
            if item is None:
                continue
            checked = self._checked.get(id(task), True)
            row = WorkflowTaskRowWidget(task, checked)
            item.setSizeHint(row.sizeHint())
            self._list.setItemWidget(item, row)
            self._connect_row(row)
            self._apply_btn_visibility(row)
        self._sync_select_all()
        self.order_changed.emit(tasks)

    def _sync_select_all(self) -> None:
        count = self._list.count()
        if count == 0:
            return
        n_checked = sum(
            self._row(i).checkbox.isChecked()
            for i in range(count)
        )
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if n_checked == 0:
            cb.setCheckState(Qt.CheckState.Unchecked)
        elif n_checked == count:
            cb.setCheckState(Qt.CheckState.Checked)
        else:
            cb.setTristate(True)
            cb.setCheckState(Qt.CheckState.PartiallyChecked)
        cb.blockSignals(False)
