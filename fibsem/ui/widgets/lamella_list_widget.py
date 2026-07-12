from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from fibsem.ui.icon import fibsem_icon

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    DefectState,
    DefectType,
    Lamella,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton

_NAME_MIN_WIDTH = 160
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40



class _LamellaTooltip(QWidget):
    """Frameless popup showing a thumbnail image when hovering a lamella name."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)  # type: ignore[call-overload]
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("background: #1e2124; border: 1px solid #3a3d42;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._img_label = QLabel()
        self._img_label.setFixedSize(256, 170)
        layout.addWidget(self._img_label)

    def set_image(self, arr: np.ndarray) -> None:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
        self._img_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(256, 170, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def show_near_cursor(self) -> None:
        from PyQt5.QtWidgets import QApplication
        cursor = QCursor.pos()
        self.adjustSize()
        w, h = self.width(), self.height()
        x = cursor.x() + 14
        y = cursor.y() + 14  # below the cursor by default
        screen = QApplication.screenAt(cursor)
        if screen is not None:
            geom = screen.availableGeometry()
            x = max(geom.left(), min(x, geom.right() - w))
            y = max(geom.top(), min(y, geom.bottom() - h))
        self.move(x, y)
        self.show()
        self.raise_()


def _status_text(lamella: Lamella) -> tuple[str, str]:
    """Return (text, stylesheet) for the status column."""
    ts = lamella.task_state
    if ts and ts.status == AutoLamellaTaskStatus.InProgress:
        return f"{ts.name}", f"color: {stylesheets.PRIMARY_COLOR}; background: transparent;"
    last = lamella.last_completed_task
    if last:
        return last.completed, "color: #909090; background: transparent;"
    return "", "background: transparent;"


def _defect_icon(lamella: Lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator button."""
    d = lamella.defect
    if d.state == DefectType.NONE:
        return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"
    if d.state == DefectType.REWORK:
        return "mdi:refresh-circle", stylesheets.DEFECT_ORANGE_COLOR, f"Rework required{': ' + d.description if d.description else ''}"
    return "mdi:close-circle", stylesheets.DEFECT_RED_COLOR, f"Failure{': ' + d.description if d.description else ''}"


class LamellaRowWidget(QWidget):
    move_to_clicked = pyqtSignal(object)       # Lamella
    edit_clicked = pyqtSignal(object)          # Lamella
    remove_clicked = pyqtSignal(object)        # Lamella
    defect_changed = pyqtSignal(object)        # Lamella
    selection_changed = pyqtSignal(object, bool)  # Lamella, checked

    def __init__(
        self,
        lamella: Lamella,
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._popup: Optional[_LamellaTooltip] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_popup)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self.checkbox)

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel()
        layout.addWidget(self.status_label, 1)

        self.btn_defect = QToolButton()
        self.btn_defect.setIcon(fibsem_icon("mdi:circle", color=stylesheets.GREEN_COLOR))
        self.btn_defect.setFixedSize(_BTN_SIZE)
        self.btn_defect.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        layout.addWidget(self.btn_defect)

        self.btn_edit = IconToolButton(icon="mdi:pencil", tooltip="Edit Lamella", size=_BTN_SIZE.width())
        layout.addWidget(self.btn_edit)

        self.btn_remove = IconToolButton(icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE.width())
        layout.addWidget(self.btn_remove)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.lamella, bool(s))
        )
        self.btn_defect.installEventFilter(self)
        self.btn_defect.clicked.connect(self._on_defect_clicked)
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        # Connect to evented fields so only this row updates when its data changes.
        # task_history is a plain List so append won't fire; task_state.events covers
        # in-progress and completion transitions. type: ignore because @evented adds
        # .events dynamically and pyright can't see it.
        lamella.task_state.events.name.connect(self.refresh)    # type: ignore[union-attr]
        lamella.task_state.events.status.connect(self.refresh)  # type: ignore[union-attr]
        lamella.events.defect.connect(self.refresh)             # type: ignore[union-attr]

        self.refresh()

    def _on_defect_clicked(self) -> None:
        menu = QMenu(self)
        action_none = menu.addAction(
            fibsem_icon("mdi:check-circle", color=stylesheets.GREEN_COLOR), "No defect"
        )
        action_rework = menu.addAction(
            fibsem_icon("mdi:refresh-circle", color=stylesheets.DEFECT_ORANGE_COLOR), "Rework required"
        )
        action_failure = menu.addAction(
            fibsem_icon("mdi:close-circle", color=stylesheets.DEFECT_RED_COLOR), "Failure"
        )

        chosen = menu.exec_(self.btn_defect.mapToGlobal(
            self.btn_defect.rect().bottomLeft()
        ))

        if chosen == action_none:
            self.lamella.defect = DefectState(state=DefectType.NONE)
        elif chosen == action_rework:
            self.lamella.defect = DefectState(state=DefectType.REWORK)
        elif chosen == action_failure:
            self.lamella.defect = DefectState(state=DefectType.FAILURE)
        else:
            return

        self.refresh()
        self.defect_changed.emit(self.lamella)

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove Lamella",
            f"Remove <b>{self.lamella.name}</b>?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_clicked.emit(self.lamella)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.btn_defect:
            if event.type() == QEvent.Enter:
                self._hover_timer.start(400)
            elif event.type() == QEvent.Leave:
                self._hover_timer.stop()
                if self._popup:
                    self._popup.hide()
        return super().eventFilter(obj, event)

    def _show_popup(self) -> None:
        if self._popup is None:
            self._popup = _LamellaTooltip()
        self._popup.set_image(self.lamella.get_thumbnail())
        self._popup.show_near_cursor()

    def refresh(self) -> None:
        """Re-read all display fields from the stored Lamella."""
        self.name_label.setText(self.lamella.name)

        icon_name, icon_color, tooltip = _defect_icon(self.lamella)
        self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self.btn_defect.setToolTip(tooltip)

        status_text, status_style = _status_text(self.lamella)
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(status_style)


# independent filter axes: task progress and defect
_TASK_STATUS_OPTIONS = [
    ("any", "Any"),
    ("not_started", "Not started"),
    ("in_progress", "In progress"),
    ("completed", "Completed"),
]
_DEFECT_OPTIONS = [
    ("any", "Any"),
    ("none", "No defect"),
    ("failure", "Failure"),
    ("rework", "Rework"),
]


def _task_progress(lamella: Lamella) -> str:
    """Overall workflow progress of a lamella (independent of defect)."""
    if lamella.task_state.status is AutoLamellaTaskStatus.InProgress:
        return "in_progress"
    if lamella.completed_tasks:  # has finished at least one task
        return "completed"
    return "not_started"


def _defect_status(lamella: Lamella) -> str:
    """Defect state key of a lamella (independent of progress)."""
    d = lamella.defect.state
    if d is DefectType.FAILURE:
        return "failure"
    if d is DefectType.REWORK:
        return "rework"
    return "none"


_FILTER_PANEL_STYLE = (
    "QFrame#LamellaFilterPanel { background: rgba(30,33,36,235); border: 1px solid #555;"
    " border-radius: 4px; }"
    "QLabel { color: #d1d2d4; font-size: 11px; background: transparent; border: none; }"
    "QPushButton { background: rgba(60,63,70,200); border: 1px solid #666;"
    " border-radius: 3px; color: #d1d2d4; font-size: 11px; padding: 3px 10px; }"
    "QPushButton:hover { background: rgba(80,83,90,220); }"
)


class _LamellaFilterPanel(QFrame):
    """Floating grid / status / task filter popover for the lamella list.

    Grid and Task options come from the experiment; Status is fixed. A task
    filter is "pending": it shows lamellae that have NOT completed the selected
    task. Anchored under a header button, mirroring the contrast/gamma popover.
    Emits ``filter_changed`` on any change.
    """
    filter_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LamellaFilterPanel")
        self.setStyleSheet(_FILTER_PANEL_STYLE)
        self.setFixedWidth(220)
        self.setVisible(False)
        self._anchor = None

        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        self.combo_grid = QComboBox()
        self.combo_grid.addItem("All grids", None)
        self.combo_task_status = QComboBox()
        for key, label in _TASK_STATUS_OPTIONS:
            self.combo_task_status.addItem(label, key)
        self.combo_defect = QComboBox()
        for key, label in _DEFECT_OPTIONS:
            self.combo_defect.addItem(label, key)
        self.combo_task = QComboBox()
        self.combo_task.addItem("Any task", None)
        self.combo_task.setToolTip("Show lamellae that have completed the selected task")
        for combo in (self.combo_grid, self.combo_task_status,
                      self.combo_defect, self.combo_task):
            combo.currentIndexChanged.connect(lambda *_: self.filter_changed.emit())

        self._grid_label = QLabel("Grid")
        self._task_label = QLabel("Completed task")
        form.addRow(self._grid_label, self.combo_grid)
        form.addRow(QLabel("Task status"), self.combo_task_status)
        form.addRow(QLabel("Defect"), self.combo_defect)
        form.addRow(self._task_label, self.combo_task)
        self.btn_clear = QPushButton("Clear filters")
        self.btn_clear.clicked.connect(self.reset)
        form.addRow("", self.btn_clear)

    def set_options(self, grid_id_to_name: dict, task_names: List[str]) -> None:
        """Repopulate the Grid + Task dropdowns, preserving the current choice."""
        for combo, first_label, items in (
            (self.combo_grid, "All grids", list(grid_id_to_name.items())),
            (self.combo_task, "Any task", [(t, t) for t in task_names]),
        ):
            prev = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first_label, None)
            for data, label in items:
                combo.addItem(label, data)
            idx = combo.findData(prev)
            combo.setCurrentIndex(idx if idx != -1 else 0)
            combo.blockSignals(False)
        # a lamella-only experiment (no grids) hides the grid row; likewise tasks
        for label, combo, present in (
            (self._grid_label, self.combo_grid, bool(grid_id_to_name)),
            (self._task_label, self.combo_task, bool(task_names)),
        ):
            label.setVisible(present)
            combo.setVisible(present)

    def is_active(self) -> bool:
        """True when any filter is narrower than the default (all rows shown)."""
        return (self.grid_id() is not None or self.task_status() != "any"
                or self.defect() != "any" or self.task() is not None)

    def grid_id(self):
        return self.combo_grid.currentData()

    def task_status(self) -> str:
        return self.combo_task_status.currentData()

    def defect(self) -> str:
        return self.combo_defect.currentData()

    def task(self):
        return self.combo_task.currentData()

    def reset(self) -> None:
        for combo in (self.combo_grid, self.combo_task_status,
                      self.combo_defect, self.combo_task):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.filter_changed.emit()

    # --- popover visibility (mirrors ContrastGammaControl) ---

    def set_open(self, open_: bool, anchor: Optional[QWidget] = None) -> None:
        if anchor is not None:
            self._anchor = anchor
        self.setVisible(open_)
        if open_:
            self.reposition()
            self.raise_()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = parent.width() - self.width() - 4
        y = 4
        if self._anchor is not None:
            # map the anchor button's bottom into the panel's parent coords
            btm = self._anchor.mapTo(parent, self._anchor.rect().bottomLeft())
            y = btm.y() + 4
        self.move(max(4, x), y)


class _LamellaListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # spans checkbox indicator + spacing + name column
        self.checkbox_all = QCheckBox("Select All")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)

        status_header = QLabel("Status")
        status_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(status_header, 1)

        # sits over the row action buttons; align with the rightmost (remove/trash)
        self.btn_filter = QToolButton()
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedSize(_BTN_SIZE)
        self.btn_filter.setToolTip("Filter lamellae")
        self.btn_filter.setStyleSheet("background: transparent;")
        self._refresh_filter_icon(active=False)
        # left pad = the other two row buttons (defect + edit) + their gaps, so the
        # filter lines up in the trash column at the right edge
        pad = QWidget()
        pad.setFixedWidth(_BTN_SIZE.width() * 2 + 8 * 2)
        pad.setStyleSheet("background: transparent;")
        layout.addWidget(pad)
        layout.addWidget(self.btn_filter)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )

    def _refresh_filter_icon(self, active: bool) -> None:
        """Tint the filter button blue when a filter is applied, else grey."""
        colour = "#4da3ff" if active else stylesheets.GRAY_ICON_COLOR
        icon = "mdi:filter" if active else "mdi:filter-outline"
        self.btn_filter.setIcon(fibsem_icon(icon, color=colour))


class LamellaListWidget(QWidget):
    """List widget displaying Lamella objects with name, defect, status and actions."""

    move_to_requested = pyqtSignal(object)   # Lamella
    edit_requested = pyqtSignal(object)      # Lamella
    remove_requested = pyqtSignal(object)    # Lamella
    defect_changed = pyqtSignal(object)      # Lamella
    selection_changed = pyqtSignal(list)     # List[Lamella]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _LamellaListHeader()
        layout.addWidget(self._header)

        # filter popover floats over the list, toggled by the header button
        self._filter = _LamellaFilterPanel(self)
        self._filter.filter_changed.connect(self._apply_filter)
        self._header.btn_filter.toggled.connect(
            lambda on: self._filter.set_open(on, self._header.btn_filter)
        )

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._header.select_all_changed.connect(self._on_select_all)

        self._btn_visible = {
            "edit": True,
            "remove": True,
            "defect": True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_lamella(self, lamella: Lamella, checked: bool = False) -> LamellaRowWidget:
        row = LamellaRowWidget(lamella, checked)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        row.move_to_clicked.connect(self.move_to_requested)
        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.defect_changed.connect(self.defect_changed)
        row.selection_changed.connect(self._on_row_selection_changed)

        self._apply_btn_visibility(row)
        self._sync_select_all()
        return row

    def set_filter_options(self, grid_id_to_name: dict, task_names: List[str]) -> None:
        """Populate the Grid + Task filter dropdowns and re-apply the filter."""
        self._filter.set_options(grid_id_to_name, task_names)
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Hide rows that don't match the active grid/task-status/defect/task filter."""
        gid = self._filter.grid_id()
        task_status = self._filter.task_status()
        defect = self._filter.defect()
        task = self._filter.task()
        for i in range(self._list.count()):
            lam = self._row(i).lamella
            visible = True
            if gid is not None and lam.grid_id != gid:
                visible = False
            elif task_status != "any" and _task_progress(lam) != task_status:
                visible = False
            elif defect != "any" and _defect_status(lam) != defect:
                visible = False
            elif task is not None and not lam.has_completed_task(task):
                visible = False  # "completed task": show only lamellae that did it
            self._list.item(i).setHidden(not visible)
        self._header._refresh_filter_icon(active=self._filter.is_active())
        self._sync_select_all()
        # the effective (visible + checked) selection may have changed
        self.selection_changed.emit(self.get_selected())

    def enable_actions_button(self, visible: bool) -> None:
        self.enable_edit_action(visible)

    def enable_move_to_action(self, visible: bool) -> None:
        pass  # not available

    def enable_edit_action(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for i in range(self._list.count()):
            row = self._row(i)
            if row is not None:
                row.btn_edit.setVisible(visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_remove.setVisible(visible)

    def enable_defect_button(self, visible: bool) -> None:
        self._btn_visible["defect"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_defect.setVisible(visible)

    def remove_lamella(self, lamella: Lamella) -> None:
        for i in range(self._list.count()):
            if self._row(i).lamella is lamella:
                self._list.takeItem(i)
                break
        self._sync_select_all()

    def refresh_lamella(self, lamella: Lamella) -> None:
        """Refresh the display of a single lamella row."""
        for i in range(self._list.count()):
            row = self._row(i)
            if row.lamella is lamella:
                row.refresh()
                break

    def refresh_all(self) -> None:
        """Refresh every row from its current Lamella state."""
        for i in range(self._list.count()):
            self._row(i).refresh()

    def get_selected(self) -> List[Lamella]:
        # only visible + checked rows count — a filter scopes the selection
        return [
            self._row(i).lamella
            for i in range(self._list.count())
            if self._row(i).checkbox.isChecked() and not self._list.item(i).isHidden()
        ]

    def clear(self) -> None:
        self._list.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row(self, i: int) -> LamellaRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _apply_btn_visibility(self, row: LamellaRowWidget) -> None:
        row.btn_edit.setVisible(self._btn_visible["edit"])
        row.btn_remove.setVisible(self._btn_visible["remove"])
        row.btn_defect.setVisible(self._btn_visible["defect"])

    def _on_remove_clicked(self, lamella: Lamella) -> None:
        self.remove_lamella(lamella)
        self.remove_requested.emit(lamella)

    def _on_select_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).isHidden():
                continue  # select-all only touches visible (filtered-in) rows
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
        self.selection_changed.emit(self.get_selected())

    def _on_row_selection_changed(self, *_) -> None:
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _sync_select_all(self) -> None:
        # reflect only visible (filtered-in) rows
        visible = [i for i in range(self._list.count())
                   if not self._list.item(i).isHidden()]
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if not visible:
            cb.setCheckState(Qt.Unchecked)
        else:
            n_checked = sum(self._row(i).checkbox.isChecked() for i in visible)
            if n_checked == 0:
                cb.setCheckState(Qt.Unchecked)
            elif n_checked == len(visible):
                cb.setCheckState(Qt.Checked)
            else:
                cb.setTristate(True)
                cb.setCheckState(Qt.PartiallyChecked)
        cb.blockSignals(False)
