from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QDateTime, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDateTimeEdit,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

import fibsem.config as fcfg
from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel

_ROW_HEIGHT = 40
# Requirements list shows up to this many rows before it starts to scroll.
_MAX_VISIBLE_REQS = 5
# A task may be scheduled from now up to this many days in advance.
_MAX_SCHEDULE_DAYS_AHEAD = 2
_NAME_PILL_STYLE = (
    "font-size: 12px; font-weight: bold; color: #d6d6d6; background: #2d3340;"
    " border: 1px solid #3d4251; border-radius: 10px; padding: 2px 10px;"
)
_HINT_STYLE = "color: #707070; font-size: 11px; background: transparent;"
_HINT_WARN_STYLE = "color: #f0a040; font-size: 11px; background: transparent;"


class _BoundedDateTimeEdit(QDateTimeEdit):
    """QDateTimeEdit that reports when a step is blocked by the min/max bound.

    ``bound_exceeded`` fires with ``"max"`` or ``"min"`` when the user steps
    (arrows / wheel / keyboard) but the value cannot move because it is already
    pinned at a bound — i.e. they tried to schedule beyond the allowed window.
    """

    bound_exceeded = pyqtSignal(str)  # "min" or "max"

    def stepBy(self, steps: int) -> None:
        before = self.dateTime()
        super().stepBy(steps)
        if steps != 0 and self.dateTime() == before:
            self.bound_exceeded.emit("max" if steps > 0 else "min")


class _RequirementRowWidget(QWidget):
    """Simple checkbox row for a single available task."""

    def __init__(self, task_name: str, checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        layout.addWidget(self.checkbox)

        self.name_label = QLabel(task_name)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label, 1)


class WorkflowTaskEditorWidget(QWidget):
    """
    Inline editor for a single AutoLamellaTaskDescription.

    Pass ``available_tasks`` as the names of all other tasks in the workflow
    so the user can select which ones this task depends on.

    Signals
    -------
    apply_clicked  : emits a *copy* of the task with the edited values applied.
    cancel_clicked : emits when the user clicks Cancel.
    """

    apply_clicked = pyqtSignal(object)   # AutoLamellaTaskDescription
    cancel_clicked = pyqtSignal()

    def __init__(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._available = [t for t in (available_tasks or []) if t != task.name]

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── header: pencil icon + title + task-name pill ─────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(QIconifyIcon("mdi:pencil", color="#a0a0a0").pixmap(18, 18))
        icon_lbl.setStyleSheet("background: transparent;")
        header_row.addWidget(icon_lbl)

        title = QLabel("Edit task")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #c8c8c8; background: transparent;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self._name_label = QLabel(task.name)
        self._name_label.setStyleSheet(_NAME_PILL_STYLE)
        header_row.addWidget(self._name_label)
        root.addLayout(header_row)

        # ── Properties panel (optional flag + scheduling) ────────────────
        props_content = QWidget()
        props_content.setStyleSheet("background: transparent;")
        pc = QVBoxLayout(props_content)
        pc.setContentsMargins(6, 6, 6, 6)
        pc.setSpacing(8)

        self._optional_cb = QCheckBox("Optional  (uncheck to make required)")
        self._optional_cb.setChecked(not task.required)
        pc.addWidget(self._optional_cb)

        self._schedule_cb = QCheckBox("Schedule start time")
        self._schedule_cb.setChecked(task.scheduled_at is not None)

        self._dt_edit = _BoundedDateTimeEdit()
        # 12-hour clock with AM/PM (AP). Lowercase hh => 12-hour, zero-padded.
        self._dt_edit.setDisplayFormat("yyyy-MM-dd  hh:mm AP")
        # Calendar popup is intentionally OFF so the up/down step arrows show:
        # click a field (date / hour / minute / AM-PM) and step it with the
        # arrows, the mouse wheel, or the keyboard.
        self._dt_edit.setCalendarPopup(False)
        self._dt_edit.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self._dt_edit.setStyleSheet(stylesheets.DATETIME_EDIT_STYLESHEET)
        # enough room for "yyyy-MM-dd  hh:mm AP" plus the two step buttons
        self._dt_edit.setMinimumWidth(250)
        self._apply_schedule_bounds()
        self._dt_edit.setDateTime(self._qdatetime_for(task))
        self._dt_edit.setEnabled(self._schedule_cb.isChecked())

        # scheduling controls live in one container so the whole block can be
        # hidden when the scheduled-tasks feature is disabled.
        self._schedule_container = QWidget()
        self._schedule_container.setStyleSheet("background: transparent;")
        scl = QVBoxLayout(self._schedule_container)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.setSpacing(4)

        # checkbox label + date/time field share one row
        sched_row = QHBoxLayout()
        sched_row.setContentsMargins(0, 0, 0, 0)
        sched_row.setSpacing(8)
        sched_row.addWidget(self._schedule_cb)
        sched_row.addWidget(self._dt_edit, 1)
        scl.addLayout(sched_row)

        self._sched_hint = QLabel()
        self._sched_hint.setStyleSheet(_HINT_STYLE)
        self._sched_hint.setWordWrap(True)
        scl.addWidget(self._sched_hint)
        pc.addWidget(self._schedule_container)

        self._schedule_cb.toggled.connect(self._dt_edit.setEnabled)
        self._schedule_cb.toggled.connect(self._refresh_schedule_hint)
        # surface a warning when a step is blocked by a bound; clear it once the
        # value moves again (a successful step fires dateTimeChanged).
        self._dt_edit.bound_exceeded.connect(self._on_bound_exceeded)
        self._dt_edit.dateTimeChanged.connect(self._refresh_schedule_hint)
        self._refresh_schedule_hint()

        self._schedule_container.setVisible(fcfg.FEATURE_SCHEDULED_TASKS_ENABLED)

        self._props_panel = TitledPanel("Properties", content=props_content, collapsible=False)
        root.addWidget(self._props_panel)

        # ── Requirements panel ───────────────────────────────────────────
        self._req_list = QListWidget()
        self._req_list.setSpacing(0)
        self._req_list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._req_list.setAlternatingRowColors(False)
        self._req_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._req_list.setFocusPolicy(Qt.NoFocus)
        # height is sized to the row count (capped, then scrolls) by
        # _resize_req_list so the panel hugs its content.
        self._req_list.setSizePolicy(self._req_list.sizePolicy().horizontalPolicy(),
                                     QSizePolicy.Fixed)

        req_content = QWidget()
        req_content.setStyleSheet("background: transparent;")
        rc = QVBoxLayout(req_content)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.addWidget(self._req_list)

        self._req_rows: List[_RequirementRowWidget] = []
        self._populate_requirements(task)

        self._req_panel = TitledPanel("Requirements", content=req_content, collapsible=False)
        root.addWidget(self._req_panel)

        req_hint = QLabel(
            "This task won't run unless the lamella has completed these "
            "pre-requisite tasks."
        )
        req_hint.setStyleSheet(_HINT_STYLE)
        req_hint.setWordWrap(True)
        root.addWidget(req_hint)

        root.addStretch(1)  # absorb extra height below the content

        # ── buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        btn_row.addWidget(self._cancel_btn)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        btn_row.addWidget(self._apply_btn)

        root.addLayout(btn_row)

        self._apply_btn.clicked.connect(self._on_apply)
        self._cancel_btn.clicked.connect(self.cancel_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _qdatetime_for(task: AutoLamellaTaskDescription) -> QDateTime:
        """Return a QDateTime seeded from task.scheduled_at, or 'now' if unset."""
        if task.scheduled_at is None:
            return QDateTime.currentDateTime()
        sa = task.scheduled_at
        return QDateTime(sa.year, sa.month, sa.day, sa.hour, sa.minute)

    def _apply_schedule_bounds(self) -> None:
        """Restrict the picker to [now, now + _MAX_SCHEDULE_DAYS_AHEAD days].

        Refreshed whenever a task loads so the window tracks the current time.
        Qt clamps both stepping and typed input to these bounds.
        """
        now = QDateTime.currentDateTime()
        self._dt_edit.setMinimumDateTime(now)
        self._dt_edit.setMaximumDateTime(now.addDays(_MAX_SCHEDULE_DAYS_AHEAD))

    def _on_bound_exceeded(self, which: str) -> None:
        """Show a warning hint when the user steps past the allowed window."""
        if which == "max":
            self._sched_hint.setText(
                f"That's as far ahead as you can schedule — up to "
                f"{_MAX_SCHEDULE_DAYS_AHEAD} days from now."
            )
        else:
            self._sched_hint.setText(
                "That's the earliest you can schedule — no times in the past."
            )
        self._sched_hint.setStyleSheet(_HINT_WARN_STYLE)

    def _refresh_schedule_hint(self) -> None:
        """Show how far ahead the selected time is, or a cap reminder when off."""
        if self._schedule_cb.isChecked():
            text = (f"Starts {self._offset_from_now()} from now · up to "
                    f"{_MAX_SCHEDULE_DAYS_AHEAD} days ahead.")
        else:
            text = f"Up to {_MAX_SCHEDULE_DAYS_AHEAD} days ahead."
        self._sched_hint.setText(text)
        self._sched_hint.setStyleSheet(_HINT_STYLE)

    def _offset_from_now(self) -> str:
        """Whole-minute offset between now and the selected time (e.g. '8h 30m')."""
        secs = max(0, QDateTime.currentDateTime().secsTo(self._dt_edit.dateTime()))
        mins = (secs + 30) // 60  # round to the nearest minute
        if mins < 1:
            return "less than a minute"
        hours, minutes = divmod(mins, 60)
        if hours and minutes:
            return f"{hours}h {minutes}m"
        if hours:
            return f"{hours}h"
        return f"{minutes}m"

    def load_task(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: Optional[List[str]] = None,
    ) -> None:
        """Reload the editor with a different task."""
        self._task = task
        self._available = [t for t in (available_tasks or []) if t != task.name]
        self._name_label.setText(task.name)
        self._optional_cb.setChecked(not task.required)

        self._schedule_cb.blockSignals(True)
        self._schedule_cb.setChecked(task.scheduled_at is not None)
        self._schedule_cb.blockSignals(False)
        self._apply_schedule_bounds()
        self._dt_edit.setDateTime(self._qdatetime_for(task))
        self._dt_edit.setEnabled(task.scheduled_at is not None)
        self._refresh_schedule_hint()
        # re-read the flag each open so a Preferences toggle takes effect
        # without restarting (the dialog is persistent and reused).
        self._schedule_container.setVisible(fcfg.FEATURE_SCHEDULED_TASKS_ENABLED)

        self._populate_requirements(task)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate_requirements(self, task: AutoLamellaTaskDescription) -> None:
        """Rebuild the requirements list from the available task names."""
        self._req_list.clear()
        self._req_rows.clear()

        if self._available:
            for name in self._available:
                row = _RequirementRowWidget(name, checked=(name in task.requires))
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, _ROW_HEIGHT))
                self._req_list.addItem(item)
                self._req_list.setItemWidget(item, row)
                self._req_rows.append(row)
        else:
            empty = QLabel("No other tasks available")
            empty.setStyleSheet("color: #606060; font-size: 11px; padding: 6px;")
            empty.setAlignment(Qt.AlignCenter)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._req_list.addItem(item)
            self._req_list.setItemWidget(item, empty)

        self._resize_req_list()

    def _resize_req_list(self) -> None:
        """Size the list to its rows (capped at _MAX_VISIBLE_REQS, then scrolls)."""
        visible = min(max(self._req_list.count(), 1), _MAX_VISIBLE_REQS)
        self._req_list.setFixedHeight(visible * _ROW_HEIGHT + 2 * self._req_list.frameWidth())

    def _on_apply(self) -> None:
        self._task.required = not self._optional_cb.isChecked()
        self._task.requires = [
            row.name_label.text()
            for row in self._req_rows
            if row.checkbox.isChecked()
        ]
        # Only touch scheduled_at when the feature is enabled; otherwise leave
        # any previously-saved schedule untouched (the controls are hidden).
        if fcfg.FEATURE_SCHEDULED_TASKS_ENABLED:
            if self._schedule_cb.isChecked():
                self._task.scheduled_at = (
                    self._dt_edit.dateTime().toPyDateTime().replace(second=0, microsecond=0)
                )
            else:
                self._task.scheduled_at = None
        self.apply_clicked.emit(self._task)
