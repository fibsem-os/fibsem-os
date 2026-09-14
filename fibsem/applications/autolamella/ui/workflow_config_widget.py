from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
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
    CANVAS_BG,
    NEUTRAL_700,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton

_NAME_MIN_WIDTH = 180
# The dependency column: present, but never competing with the name it belongs to.
# Fixed width, so the entries line up down the list and the text elides inside it
# rather than the row clipping it mid-word.
REQUIRES_COLOUR = NEUTRAL_700
REQUIRES_FONT_PX = 10
REQUIRES_MAX_WIDTH = 170
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
# The attention chip: one word, no icon (the word is the state, the colour
# is the mode), fixed width so the row does not jump between states.
# "Supervised" is the widest word it shows; the old two buttons took 72px.
_CHIP_WIDTH = 78
_BTN_SPACER_WIDTH = (
    _BTN_SIZE.width() * 3 + _CHIP_WIDTH + 8 * 3
)  # schedule + attention chip + edit + remove + 3 gaps
_BTN_STYLE = stylesheets.TOOLBUTTON_ICON_STYLESHEET
# Long labels on purpose: the short set (Auto / Superv. / Review) reads badly
# and the long ones fit at the chip's width.
ATTENTION_LABELS = {
    "automated": "Automated",
    "supervised": "Supervised",
    "agent": "Agent",
    "review": "Review",
}


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


def _agent_supervision_available() -> bool:
    """Whether the Agent option exists at all: the agent-server preference is on.

    The preference rather than server-running — protocols are configured before
    a microscope connects. Without it, the selector is exactly the old
    two-state toggle and a stored ``supervisor: agent`` displays as plain
    Supervised (the same hard-gate rule as the window chrome).
    """
    import fibsem.config as fibsem_cfg

    try:
        return bool(fibsem_cfg.load_user_preferences().features.agent_server_enabled)
    except Exception:
        return False


def _review_available() -> bool:
    """Whether the Review toggle exists at all: the propose-and-review flag.
    Same hard-gate rule as the Agent option -- off, a stored ``review: true``
    is not shown and not honoured."""
    import fibsem.config as fibsem_cfg

    try:
        return bool(
            fibsem_cfg.load_user_preferences().features.proposer_reviewer_workflow_enabled
        )
    except Exception:
        return False


def attention_state(
    task: AutoLamellaTaskDescription,
    agent_available: Optional[bool] = None,
    review_available: Optional[bool] = None,
) -> str:
    """Which of the chip's states the task is in: the one question the user
    has about a task, *when am I involved?*, read from the two protocol fields.

    ``supervise`` wins over ``review`` when a saved protocol has both (a
    combination the chip never writes): if you are at the microscope for it
    you answer there. A stored ``supervisor: agent`` shows as plain Supervised
    while the agent-server preference is off, and a stored ``review: true``
    shows as Automated while interactive review is off -- the state it will
    actually run in, not the one in the file.
    """
    if agent_available is None:
        agent_available = _agent_supervision_available()
    if review_available is None:
        review_available = _review_available()
    if task.supervise:
        if getattr(task, "supervisor", "human") == "agent" and agent_available:
            return "agent"
        return "supervised"
    if task.review and review_available:
        return "review"
    return "automated"


def _attention_chip(
    task: AutoLamellaTaskDescription, has_dependents: bool = True
) -> tuple[str, str, str]:
    """(label, colour, tooltip) for the chip: the word is the state, the
    colour is the mode's."""
    state = attention_state(task)
    label = ATTENTION_LABELS[state]
    if state == "agent":
        return (
            label,
            stylesheets.BORDER_STATE_COLOURS["agent"],
            "Agent — the connected agent answers this task's questions in the "
            "workflow (you can always answer first). Click to change.",
        )
    if state == "supervised":
        return (
            label,
            stylesheets.PRIMARY_COLOR,
            "Supervised — asks you in the workflow, at the microscope; your "
            "answer is the decision on the record. Click to change.",
        )
    if state == "review":
        tip = (
            "Review — the next task waits for your decision in the Review tab. "
            "Click to change."
        )
        if not has_dependents:
            tip = (
                "Review — but nothing requires this task, so nothing waits on "
                "the decision. Add a requirement to a later task, or click to "
                "change."
            )
        return label, stylesheets.REVIEW_COLOR, tip
    tip = (
        "Automated — runs without anyone; what it did is listed in the Review "
        "tab to check. Click to change."
    )
    if task.review and not task.supervise:
        tip = (
            "Runs as Automated: the protocol says Review, but interactive "
            "review is off in Preferences. Click to change."
        )
    return label, stylesheets.AUTOMATED_COLOR, tip


def _chip_style(colour: str, muted: bool = False) -> str:
    fg = NEUTRAL_700 if muted else colour
    border = NEUTRAL_700 if muted else colour
    return (
        "QToolButton { border: 1px solid "
        + border
        + "; border-radius: 3px; padding: 1px 4px; background: transparent; "
        + f"color: {fg}; font-size: 11px; }}"
        "QToolButton:hover { background: rgba(255, 255, 255, 25); }"
    )


def _requires_text(task: AutoLamellaTaskDescription, font: QFont) -> str:
    """What the task waits for, sized to the column that holds it.

    Empty where a task has no dependency: "No requirements" on every row was what
    buried the two or three that have one.

    Elided rather than left to run. A task waiting on four others produced a label
    wider than the row, and Qt cut it off mid-word; the row's tooltip carries the
    full list.
    """
    if not task.requires:
        return ""
    return QFontMetrics(font).elidedText(
        "after " + ", ".join(task.requires), Qt.ElideRight, REQUIRES_MAX_WIDTH
    )


class WorkflowTaskRowWidget(QWidget):
    """One task: its name, what it waits for, and one chip for when a person
    is involved (Automated / Supervised / Review), plus schedule, edit, remove.

    The chip writes two protocol fields, ``supervise`` and ``review``, and
    emits ``supervised_changed`` or ``review_changed`` for whichever changed,
    so the hosts that listened to two buttons need no change.
    """

    supervised_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    review_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    edit_clicked = pyqtSignal(object)  # AutoLamellaTaskDescription
    remove_clicked = pyqtSignal(object)  # AutoLamellaTaskDescription
    selection_changed = pyqtSignal(object, bool)  # AutoLamellaTaskDescription, checked

    def __init__(
        self,
        task: AutoLamellaTaskDescription,
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
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

        # One line, not a name over a requirements line. The second line was blank
        # on most rows -- every task without a dependency -- and a name sitting at
        # the top of a two-line block reads as floating above a gap.
        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        # Task names come from the protocol, and a QLabel left on AutoText renders
        # anything that looks like markup as markup.
        self.name_label.setTextFormat(Qt.PlainText)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        layout.addStretch(1)

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

        self.btn_schedule = QToolButton()
        self.btn_schedule.setFixedSize(_BTN_SIZE)
        self.btn_schedule.setStyleSheet(_BTN_STYLE)
        layout.addWidget(self.btn_schedule)

        # One chip in place of a supervise button and a review button: the
        # word on it is the state, the colour is the run mode's (automated
        # green, supervised blue, review teal, agent the agent border).
        self.btn_attention = QToolButton()
        self.btn_attention.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_attention.setFixedSize(_CHIP_WIDTH, _BTN_SIZE.height() - 8)
        self.btn_attention.setCursor(Qt.PointingHandCursor)
        self.btn_attention.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.btn_attention)
        self._has_dependents = True
        # kept for callers that showed or hid the old buttons
        self.btn_supervise = self.btn_attention
        self.btn_review = self.btn_attention

        self.btn_edit = IconToolButton(
            icon="mdi:pencil", tooltip="Edit", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_edit)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_remove)

        drag_icon = QLabel()
        drag_icon.setFixedSize(DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)
        drag_icon.setPixmap(drag_handle_pixmap())
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.task, bool(s))
        )
        self.btn_schedule.clicked.connect(lambda: self.edit_clicked.emit(self.task))
        self.btn_attention.clicked.connect(self._on_attention_clicked)
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.task))
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        self.refresh()

    def set_has_dependents(self, has_dependents: bool) -> None:
        """Whether any later task requires this one. A Review state on a
        task nothing requires gates nothing, and the row says so."""
        if has_dependents != self._has_dependents:
            self._has_dependents = has_dependents
            self.refresh()

    def _on_attention_clicked(self) -> None:
        """Cycle Automated → Supervised → Agent → Review → Automated.

        The Agent step exists only while the agent-server preference is on,
        the Review step only while interactive review is on; with neither this
        is the old two-state toggle. Each state writes both fields, so no
        hidden combination survives a click: leaving Agent resets
        ``supervisor`` to human, leaving Review clears ``review``.
        """
        task = self.task
        order = ["automated", "supervised"]
        if _agent_supervision_available():
            order.append("agent")
        if _review_available():
            order.append("review")
        current = attention_state(task)
        following = order[(order.index(current) + 1) % len(order)]
        before = (task.supervise, getattr(task, "supervisor", "human"), task.review)
        task.supervise = following in ("supervised", "agent")
        task.supervisor = "agent" if following == "agent" else "human"
        task.review = following == "review"
        self.refresh()
        if before[:2] != (task.supervise, task.supervisor):
            self.supervised_changed.emit(task)
        if before[2] != task.review:
            self.review_changed.emit(task)

    # the old names, for callers that drove the two buttons directly
    _on_supervise_clicked = _on_attention_clicked
    _on_review_clicked = _on_attention_clicked

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
        self.requires_label.setText(
            _requires_text(self.task, self.requires_label.font())
        )
        self.setToolTip(
            "Requires: " + ", ".join(self.task.requires) if self.task.requires else ""
        )
        label, colour, tooltip = _attention_chip(self.task, self._has_dependents)
        # the file says Review but the preference is off: shown as it will run
        downgraded = (
            self.task.review and not self.task.supervise and not _review_available()
        )
        self.btn_attention.setText(label)
        self.btn_attention.setToolTip(tooltip)
        self.btn_attention.setStyleSheet(_chip_style(colour, muted=downgraded))
        if attention_state(self.task) == "review" and not self._has_dependents:
            # the column keeps the task's own dependency when it has one; the
            # colour and the chip's tooltip carry the warning
            if not self.task.requires:
                self.requires_label.setText("nothing waits on this")
            self.requires_label.setStyleSheet(
                f"background: transparent; color: {stylesheets.WARN_COLOR}; "
                f"font-size: {REQUIRES_FONT_PX}px;"
            )
        else:
            self.requires_label.setStyleSheet(
                f"background: transparent; color: {REQUIRES_COLOUR}; "
                f"font-size: {REQUIRES_FONT_PX}px;"
            )
        if self.task.scheduled_at is not None:
            self.btn_schedule.setIcon(
                fibsem_icon("mdi:clock", color=stylesheets.WHITE_ICON_COLOR)
            )
            self.btn_schedule.setToolTip(
                f"Scheduled: {self.task.scheduled_at.strftime(DATETIME_DISPLAY_AMPM)}"
            )
        else:
            self.btn_schedule.setIcon(
                fibsem_icon("mdi:clock-outline", color=NEUTRAL_700)
            )
            self.btn_schedule.setToolTip("Not scheduled — click to set")


class _WorkflowTaskListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)
    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.checkbox_all = QCheckBox("Select All")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)

        layout.addStretch(1)

        # Spacer covers all row buttons except the last, so btn_add aligns with btn_remove
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width() - 8)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add Task", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_add)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )
        self.btn_add.clicked.connect(self.add_task_clicked)


class WorkflowConfigWidget(QWidget):
    """List widget displaying AutoLamellaWorkflowConfig tasks with name, supervised, edit and remove actions."""

    supervised_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    review_changed = pyqtSignal(object)  # AutoLamellaTaskDescription
    edit_requested = pyqtSignal(object)  # AutoLamellaTaskDescription
    remove_requested = pyqtSignal(object)  # AutoLamellaTaskDescription
    selection_changed = pyqtSignal(list)  # List[AutoLamellaTaskDescription]
    order_changed = pyqtSignal(list)  # List[AutoLamellaTaskDescription]
    add_task_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._btn_visible = {
            "schedule": True,
            "supervise": True,  # the attention chip
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
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
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

    def set_config(self, config: AutoLamellaWorkflowConfig) -> None:
        """Populate the list from an AutoLamellaWorkflowConfig, keeping the ticks.

        Reached on every protocol edit (``workflow_config_changed``), not only when
        the task set changes, and the ticked rows are the run selection. Rebuilding
        them unticked emptied that selection on any field edit (FIB-967). Ticks are
        matched by task name, so a task that is gone is simply not re-ticked.
        """
        checked = {task.name for task in self.get_selected()}
        self.clear()
        for task in config.tasks:
            self.add_task(task, checked=task.name in checked)

    def add_task(
        self, task: AutoLamellaTaskDescription, checked: bool = False
    ) -> WorkflowTaskRowWidget:
        self._checked[id(task)] = checked
        row = WorkflowTaskRowWidget(task, checked)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, task)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        self._connect_row(row)
        self._apply_btn_visibility(row)
        self._refresh_dependents()
        self._sync_select_all()
        return row

    def _refresh_dependents(self) -> None:
        """Tell each row whether a later task requires it: the fact the
        Review state needs to be honest about."""
        tasks = self.get_tasks()
        required = {req for task in tasks for req in task.requires}
        for i in range(self._list.count()):
            row = self._row(i)
            row.set_has_dependents(row.task.name in required)

    def _connect_row(self, row: WorkflowTaskRowWidget) -> None:
        row.supervised_changed.connect(self.supervised_changed)
        row.review_changed.connect(self.review_changed)
        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.selection_changed.connect(self._on_row_selection_changed)

    def enable_schedule_button(self, visible: bool) -> None:
        self._btn_visible["schedule"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_schedule.setVisible(visible)

    def enable_supervise_button(self, visible: bool) -> None:
        """Show or hide the attention chip."""
        self._btn_visible["supervise"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_attention.setVisible(visible)

    def enable_review_button(self, _visible: bool) -> None:
        """The Review state is offered by the preference, not by a host; a
        host that flips the preference re-reads the rows through here."""
        self.refresh_all()

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
        self._refresh_dependents()

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

    def _apply_btn_visibility(self, row: WorkflowTaskRowWidget) -> None:
        row.btn_schedule.setVisible(self._btn_visible["schedule"])
        row.btn_attention.setVisible(self._btn_visible["supervise"])
        row.btn_edit.setVisible(self._btn_visible["edit"])
        row.btn_remove.setVisible(self._btn_visible["remove"])

    def _on_remove_clicked(self, task: AutoLamellaTaskDescription) -> None:
        self.remove_task(task)
        self.remove_requested.emit(task)

    def _on_row_selection_changed(
        self, task: AutoLamellaTaskDescription, checked: bool
    ) -> None:
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
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.setItemWidget(item, row)
            self._connect_row(row)
            self._apply_btn_visibility(row)
        self._refresh_dependents()
        self._sync_select_all()
        self.order_changed.emit(tasks)

    def _sync_select_all(self) -> None:
        count = self._list.count()
        if count == 0:
            return
        n_checked = sum(self._row(i).checkbox.isChecked() for i in range(count))
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
