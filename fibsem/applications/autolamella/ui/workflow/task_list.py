from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.applications.autolamella.workflows.tasks.attendance import (
    Attendance,
    attendance_for,
)
from fibsem.constants import DATETIME_DISPLAY_AMPM
from fibsem.ui import stylesheets
from fibsem.ui.icon import (
    DRAG_HANDLE_HEIGHT,
    DRAG_HANDLE_WIDTH,
    drag_handle_pixmap,
    fibsem_icon,
)
from fibsem.ui.tokens import (
    NEUTRAL_700,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton
from fibsem.ui.widgets.lists import (
    CHIP_HEIGHT,
    DETAIL_PX,
    NAME_PX,
    ROW_HEIGHT,
    ListHeader,
    ReorderableListWidget,
    tinted_chip_style,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import RunPlan, TaskConfig, TaskPlan

# The dependency column: present, but never competing with the name it belongs to.
# Fixed width, so the entries line up down the list and the text elides inside it
# rather than the row clipping it mid-word.
REQUIRES_COLOUR = NEUTRAL_700
REQUIRES_FONT_PX = DETAIL_PX
REQUIRES_MAX_WIDTH = 210
BTN_SIZE = QSize(24, 24)
_ROW_HEIGHT = ROW_HEIGHT
# The attention chip: the mode's icon (the same ones the lamella rows and
# status chips use) and its name, fixed width so the row does not jump
# between states. "Supervised" is the widest it shows. The schedule
# clock button, which only duplicated the pencil, gave up the width.
CHIP_WIDTH = 104
CHIP_ICONS = {
    "automated": "mdi:lightning-bolt-circle",
    "supervised": "mdi:account-hard-hat",
}
# Long labels on purpose: the short set (Auto / Superv.) reads badly and the
# long ones fit at the chip's width. Two modes: a person decides, or nobody
# is asked. Where the run waits for the person is a property of the task,
# not a third state.
ATTENTION_LABELS = {
    "supervised": "Supervised",
    "automated": "Automated",
}


def attention_state(attention: Attention) -> str:
    """Which of the chip's states a task with *attention* is in: the one
    question the user has about a task, *when am I involved?*

    There is no Agent state (FIB-1035): a supervised question goes to the
    operator and any connected agent alike.
    """
    if attention is Attention.supervised:
        return "supervised"
    return "automated"


def _attention_chip(
    attention: Attention,
    attendance: Optional[Attendance] = None,
) -> tuple[str, str, str]:
    """(label, colour, tooltip) for the chip: the word is the state, the
    colour is the mode's, and the tooltip says what the state means for this
    task -- whether it needs you there while it runs -- when the task's
    type is known (``attendance``)."""
    label, colour, tip = _attention_chip_words(attention)
    if attendance is not None:
        tip = f"{label} — {attendance.line} Click to change."
    return label, colour, tip


def _attention_chip_words(attention: Attention) -> tuple[str, str, str]:
    state = attention_state(attention)
    label = ATTENTION_LABELS[state]
    if state == "supervised":
        tip = (
            "Supervised — you decide. A question the task needs answered is "
            "asked while it runs, on the widget it is answered on, and the "
            "run waits for your answer. Click to change."
        )
        return label, stylesheets.PRIMARY_COLOR, tip
    tip = (
        "Automated — runs without anyone; what it did is recorded for you to "
        "check. Click to change."
    )
    return label, stylesheets.AUTOMATED_COLOR, tip


def chip_style(colour: str, muted: bool = False) -> str:
    """The mode's colour as a faint fill (`list_chrome.tinted_chip_style`); muted,
    the neutral one."""
    return tinted_chip_style(NEUTRAL_700 if muted else colour)


def _requires_text(task: TaskPlan, requires: List[str], font: QFont) -> str:
    """What the task waits for, and when it is scheduled, sized to the
    column that holds them.

    Empty where a task has neither: "No requirements" on every row was what
    buried the two or three that have one. A schedule is set in the edit
    dialog; the row only says when, in the same column, so a scheduled row
    is not a wider row.

    Elided rather than left to run. A task waiting on four others produced a
    label wider than the row, and Qt cut it off mid-word; the row's tooltip
    carries the full list.
    """
    parts = []
    if requires:
        parts.append("after " + requires_phrase(requires))
    if task.scheduled_at is not None:
        parts.append("at " + task.scheduled_at.strftime("%d %b %H:%M"))
    if not parts:
        return ""
    return QFontMetrics(font).elidedText(
        " · ".join(parts), Qt.ElideRight, REQUIRES_MAX_WIDTH
    )


def requires_phrase(requires: List[str]) -> str:
    """The tasks a task waits for, as a row says them."""
    return ", ".join(requires)


class WorkflowTaskRowWidget(QWidget):
    """One task: its name, what it waits for, and one chip for when a person
    is involved (Automated / Supervised), plus schedule, edit, remove.

    Reads the task's attention and requirements from its *config* (None for a
    task the protocol has no config for). The chip changes nothing itself: it
    emits ``attention_toggled`` for the host to apply.
    """

    attention_toggled = pyqtSignal(object)  # TaskPlan
    edit_clicked = pyqtSignal(object)  # TaskPlan
    selection_changed = pyqtSignal(object, bool)  # TaskPlan, checked

    def __init__(
        self,
        task: TaskPlan,
        config: Optional[TaskConfig],
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.task = task
        self.config = config
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self.checkbox)

        # One line, not a name over a requirements line. The second line was blank
        # on most rows -- every task without a dependency -- and a name sitting at
        # the top of a two-line block reads as floating above a gap. Elided, and
        # it takes the slack: at the panel's width a long name was cut mid-word
        # ("Acquire Fluorescence Imag").
        self.name_label = ElidedLabel()
        # Task names come from the protocol, and a QLabel left on AutoText renders
        # anything that looks like markup as markup.
        self.name_label.setTextFormat(Qt.PlainText)
        self.name_label.setStyleSheet(
            f"background: transparent; font-size: {NAME_PX}px;"
        )
        layout.addWidget(self.name_label, 1)

        # Its own column, right-aligned at a fixed width, so the dependencies line
        # up down the list and can be read as a column rather than hunted for at
        # whatever point each name happens to end.
        self.requires_label = QLabel()
        self.requires_label.setFixedWidth(REQUIRES_MAX_WIDTH)
        self.requires_label.setTextFormat(Qt.PlainText)
        self.requires_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.requires_label.setStyleSheet(
            f"background: transparent; color: {REQUIRES_COLOUR}; "
            f"font-size: {REQUIRES_FONT_PX}px;"
        )
        layout.addWidget(self.requires_label)

        # One chip in place of a supervise button and a review button: the
        # word on it is the state, the colour is the run mode's (automated
        # green, supervised blue, review teal, agent the agent border).
        self.btn_attention = QToolButton()
        self.btn_attention.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_attention.setIconSize(QSize(14, 14))
        self.btn_attention.setFixedSize(CHIP_WIDTH, CHIP_HEIGHT)
        self.btn_attention.setCursor(Qt.PointingHandCursor)
        self.btn_attention.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.btn_attention)
        self._attendance: Optional[Attendance] = None

        # Edit opens the dialog, which is also where a task is removed: a
        # trash can on every row was the one thing there nobody pressed.
        self.btn_edit = IconToolButton(
            icon="mdi:pencil",
            tooltip="Edit, schedule or remove",
            size=BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_edit)

        drag_icon = QLabel()
        drag_icon.setFixedSize(DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)
        drag_icon.setPixmap(drag_handle_pixmap())
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.task, bool(s))
        )
        self.btn_attention.clicked.connect(lambda: self.attention_toggled.emit(task))
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.task))

        self.refresh()

    def set_attendance(self, attendance: Optional[Attendance]) -> None:
        """What this task needs from a person, derived from its type: said on
        the chip's tooltip and the row's, so the mode reads as what it does."""
        if attendance != self._attendance:
            self._attendance = attendance
            self.refresh()

    @property
    def attention(self) -> Attention:
        return self.config.attention if self.config is not None else Attention.automated

    @property
    def requires(self) -> List[str]:
        return list(self.config.requires) if self.config is not None else []

    def refresh(self) -> None:
        """Re-read all display fields from the stored task and its config."""
        self.name_label.setText(self.task.name)
        requires = _requires_text(self.task, self.requires, self.requires_label.font())
        self.requires_label.setText(requires)
        # The column only where there is something in it: on the rows without a
        # dependency the name has the room ("Acquire Fluorescence Image").
        self.requires_label.setVisible(bool(requires))
        tips = []
        if self.requires:
            tips.append("Requires: " + requires_phrase(self.requires))
        if self.task.scheduled_at is not None:
            tips.append(
                "Scheduled: "
                + self.task.scheduled_at.strftime(DATETIME_DISPLAY_AMPM)
                + " (set in the edit dialog)"
            )
        if self._attendance is not None:
            tips.append(self._attendance.line)
        self.setToolTip("\n".join(tips))
        label, colour, tooltip = _attention_chip(self.attention, self._attendance)
        self.btn_attention.setText(label)
        self.btn_attention.setIcon(
            fibsem_icon(CHIP_ICONS[attention_state(self.attention)], color=colour)
        )
        self.btn_attention.setToolTip(tooltip)
        self.btn_attention.setStyleSheet(chip_style(colour))
        self.requires_label.setStyleSheet(
            f"background: transparent; color: {REQUIRES_COLOUR}; "
            f"font-size: {REQUIRES_FONT_PX}px;"
        )


class _WorkflowTaskListHeader(ListHeader):
    """The list's header: the shared one (`list_chrome.ListHeader`), with Add over
    the rows' trailing column."""

    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Tasks", parent=parent)
        self.add_stretch()
        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add Task", size=BTN_SIZE.width()
        )
        self.add_widget(self.btn_add)
        self.btn_add.clicked.connect(self.add_task_clicked)


class WorkflowTaskList(QWidget):
    """The tasks of a run plan, each with its name, what it requires, its
    attention chip, and edit; their configs are the protocol's."""

    attention_toggled = pyqtSignal(object)  # TaskPlan
    edit_requested = pyqtSignal(object)  # TaskPlan
    remove_requested = pyqtSignal(object)  # TaskPlan
    selection_changed = pyqtSignal(list)  # List[TaskPlan]
    order_changed = pyqtSignal(list)  # List[TaskPlan]
    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._remove_allowed = True
        self._checked: Dict[int, bool] = {}  # id(task) -> checked
        # The protocol the tasks belong to, for each task's config and what its
        # type asks: the plan alone names tasks.
        self._protocol: Optional[AutoLamellaTaskProtocol] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _WorkflowTaskListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = ReorderableListWidget()
        self._list.setSpacing(0)
        # No item padding, as on the lamella list: the app sheet pads items by 4px,
        # which put the rows' tick boxes 4px in from the header's.
        self._list.setStyleSheet(
            stylesheets.LIST_WIDGET_STYLESHEET + "QListWidget::item { padding: 0; }"
        )
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._header.select_all_changed.connect(self.set_all_selected)
        self._header.add_task_clicked.connect(self.add_task_clicked)
        self._list.reordered.connect(self._on_reordered)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config: RunPlan) -> None:
        """Populate the list from a run plan, keeping the ticks.

        Reached on every protocol edit (``workflow_config_changed``), not only when
        the task set changes, and the ticked rows are the run selection. Rebuilding
        them unticked emptied that selection on any field edit (FIB-967). Ticks are
        matched by task name, so a task that is gone is simply not re-ticked.
        """
        checked = {task.name for task in self.get_selected()}
        self.clear()
        for task in config.tasks:
            self.add_task(task, checked=task.name in checked)

    def add_task(self, task: TaskPlan, checked: bool = False) -> WorkflowTaskRowWidget:
        self._checked[id(task)] = checked
        row = WorkflowTaskRowWidget(task, self._config_for(task), checked)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, task)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        self._connect_row(row)
        self._refresh_attendance()
        self._sync_select_all()
        return row

    def _config_for(self, task: TaskPlan) -> Optional[TaskConfig]:
        protocol = self._protocol
        return protocol.task_config.get(task.name) if protocol is not None else None

    def _refresh_attendance(self) -> None:
        """Tell each row what its task needs from a person."""
        protocol = self._protocol
        for i in range(self._list.count()):
            row = self._row(i)
            row.set_attendance(
                attendance_for(protocol, row.task.name)
                if protocol is not None
                else None
            )

    def _connect_row(self, row: WorkflowTaskRowWidget) -> None:
        row.attention_toggled.connect(self.attention_toggled)
        row.edit_clicked.connect(self.edit_requested)
        row.selection_changed.connect(self._on_row_selection_changed)

    def enable_remove_button(self, visible: bool) -> None:
        """Whether tasks may be removed at all. Removal lives in the edit
        dialog now; the host reads ``remove_allowed`` when it opens one."""
        self._remove_allowed = visible

    def add_header_widget(self, widget: QWidget) -> None:
        """Put *widget* on the header's right, after Add: the panel's ⚙ and ?."""
        self._header.add_widget(widget)

    @property
    def remove_allowed(self) -> bool:
        return self._remove_allowed

    def remove_task(self, task: TaskPlan) -> None:
        for i in range(self._list.count()):
            if self._row(i).task is task:
                self._list.takeItem(i)
                self._checked.pop(id(task), None)
                break
        self._sync_select_all()

    def refresh_task(self, task: TaskPlan) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            if row.task is task:
                row.refresh()
                break

    def set_protocol(self, protocol: Optional[AutoLamellaTaskProtocol]) -> None:
        """The protocol whose tasks these are, for each task's config and what
        it needs from a person. None: nothing is said."""
        self._protocol = protocol
        self.refresh_all()

    def refresh_all(self) -> None:
        """Re-read every row, its config included: a config can be replaced
        (the protocol loaded again, or a lamella's copied onto it)."""
        for i in range(self._list.count()):
            row = self._row(i)
            row.config = self._config_for(row.task)
            row.refresh()
        self._refresh_attendance()

    def get_tasks(self) -> List[TaskPlan]:
        """Return tasks in current display order."""
        return [self._row(i).task for i in range(self._list.count())]

    def get_selected(self) -> List[TaskPlan]:
        return [
            self._row(i).task
            for i in range(self._list.count())
            if self._row(i).checkbox.isChecked()
        ]

    def clear(self) -> None:
        self._list.clear()
        self._checked.clear()
        self._sync_select_all()

    def set_all_selected(self, checked: bool) -> None:
        """Tick or untick every row, and bring the header and the cache with them.

        Public because the header is not the only thing that clears the selection --
        starting a run and adding to the queue both do (FIB-577). `_checked` matters
        as much as the header here: it is what `_on_reordered` rebuilds rows from, so
        leaving it stale made a drag-and-drop resurrect a selection the user had
        already seen cleared.
        """
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
            self._checked[id(row.task)] = checked
        # Redundant when the header emitted into here (it is already right), and
        # load-bearing for every other caller. Cheap enough not to branch on.
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row(self, i: int) -> WorkflowTaskRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def request_remove(self, task: TaskPlan) -> None:
        """Drop the task's row and announce it. The confirmation happened in
        the edit dialog; this is the one removal path."""
        self.remove_task(task)
        self.remove_requested.emit(task)

    def _on_row_selection_changed(self, task: TaskPlan, checked: bool) -> None:
        self._checked[id(task)] = checked
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _on_reordered(self, tasks: List[TaskPlan]) -> None:
        """Rebuild all row widgets after a drag-and-drop reorder.

        Qt clears itemWidget associations when items are moved internally, so
        we re-create each row from the task stored in the item's UserRole data.
        """
        for i, task in enumerate(tasks):
            item = self._list.item(i)
            if item is None:
                continue
            checked = self._checked.get(id(task), True)
            row = WorkflowTaskRowWidget(task, self._config_for(task), checked)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.setItemWidget(item, row)
            self._connect_row(row)
        self._refresh_attendance()
        self._sync_select_all()
        self.order_changed.emit(tasks)

    def _sync_select_all(self) -> None:
        count = self._list.count()
        self._header.set_count(str(count) if count else "")
        if count == 0:
            return
        n_checked = sum(self._row(i).checkbox.isChecked() for i in range(count))
        self._header.sync(n_checked, count)
