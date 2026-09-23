from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    DefectState,
    DefectType,
    Lamella,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    CANVAS_BG,
    NEUTRAL_550,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton

_NAME_MIN_WIDTH = 160

# What the lamella displays know about grids: `GridRecord.id` -> (name, loaded).
# Built by the window from the experiment's records and the stage's inventory and
# pushed to every list and card; the rows never read the stage themselves.
GridContext = Dict[str, Tuple[str, bool]]

# The rows' type scale: name / row text / secondary line.
NAME_FONT_PX = 12
ROW_FONT_PX = 11
DETAIL_FONT_PX = 10


def grid_of(lamella, context: Optional[GridContext]) -> Optional[Tuple[str, bool]]:
    """(grid name, on the stage) for a lamella linked to a grid the context
    knows, else None -- an unlinked lamella, or one whose grid has no record,
    shows no grid and has no stage actions withheld."""
    grid_id = getattr(lamella, "grid_id", None)
    if not grid_id or not context:
        return None
    return context.get(grid_id)


def grids_named(context: Optional[GridContext]) -> bool:
    """Whether the rows name their grid at all: only when the experiment has
    more than one, since with one grid the name says nothing."""
    return bool(context) and len(context) > 1


def not_on_stage_reason(grid: Optional[Tuple[str, bool]]) -> str:
    """Why a stage action is withheld, or "" when it is not."""
    if grid is None or grid[1]:
        return ""
    return f"{grid[0]} is not on the stage"


def apply_grid_label(
    label: QLabel,
    grid: Optional[Tuple[str, bool]],
    shown: bool,
    followed: bool = True,
) -> None:
    """The grid's name ahead of the status: accent while that grid is on the
    stage, muted while it is not; hidden when there is nothing to say. The
    separator is drawn only when a status follows (*followed*)."""
    if grid is None or not shown:
        label.setVisible(False)
        label.setText("")
        return
    name, loaded = grid
    label.setText(f"{name} ·" if followed else name)
    label.setStyleSheet(
        f"font-size: {DETAIL_FONT_PX}px; background: transparent; "
        f"color: {ACCENT_COLOR if loaded else NEUTRAL_550};"
    )
    label.setToolTip(
        f"{name} is on the stage" if loaded else f"{name} is not on the stage"
    )
    label.setVisible(True)


def short_status(text: str) -> str:
    """The task name without its time stamp, for a line with no room for it."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", text)


def has_defect(lamella) -> bool:
    defect = getattr(lamella, "defect", None)
    return defect is not None and defect.state != DefectType.NONE


def add_defect_menu(menu: QMenu, lamella, on_changed) -> QMenu:
    """A "Defect" submenu on *menu* writing `lamella.defect`; *on_changed* is
    called after a write. The one place the state is set from, now that the
    icon only shows once there is one."""
    sub = menu.addMenu("Defect")
    for text, icon, colour, state in (
        ("No defect", "mdi:check-circle", stylesheets.GREEN_COLOR, DefectType.NONE),
        (
            "Rework required",
            "mdi:refresh-circle",
            stylesheets.DEFECT_ORANGE_COLOR,
            DefectType.REWORK,
        ),
        (
            "Failure",
            "mdi:close-circle",
            stylesheets.DEFECT_RED_COLOR,
            DefectType.FAILURE,
        ),
    ):
        action = sub.addAction(fibsem_icon(icon, color=colour), text)

        def _set(_checked=False, state=state):
            lamella.defect = DefectState(state=state)
            on_changed()

        action.triggered.connect(_set)
    return sub


_BTN_SIZE = QSize(24, 24)
_ROW_HEIGHT = 34


class _LamellaTooltip(QWidget):
    """Frameless popup showing a thumbnail image when hovering a lamella name."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)  # type: ignore[call-overload]
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(f"background: {CANVAS_BG}; border: 1px solid #3a3d42;")

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
            QPixmap.fromImage(qimg).scaled(
                256,
                170,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
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
        return (
            f"{ts.name}",
            f"color: {stylesheets.PRIMARY_COLOR}; background: transparent;",
        )
    last = lamella.last_completed_task
    if last:
        return last.completed, f"color: {NEUTRAL_550}; background: transparent;"
    return "", "background: transparent;"


def _defect_icon(lamella: Lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator button."""
    d = lamella.defect
    if d.state == DefectType.NONE:
        return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"
    if d.state == DefectType.REWORK:
        return (
            "mdi:refresh-circle",
            stylesheets.DEFECT_ORANGE_COLOR,
            f"Rework required{': ' + d.description if d.description else ''}",
        )
    return (
        "mdi:close-circle",
        stylesheets.DEFECT_RED_COLOR,
        f"Failure{': ' + d.description if d.description else ''}",
    )


class LamellaRowWidget(QWidget):
    """One lamella for the run selection: tick box, name, the grid it is on
    (when the experiment has more than one), a short status, a defect icon
    only once there is a defect, and one actions menu."""

    move_to_clicked = pyqtSignal(object)  # Lamella
    edit_clicked = pyqtSignal(object)  # Lamella
    remove_clicked = pyqtSignal(object)  # Lamella
    defect_changed = pyqtSignal(object)  # Lamella
    selection_changed = pyqtSignal(object, bool)  # Lamella, checked

    def __init__(
        self,
        lamella: Lamella,
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self._grid_context: Optional[GridContext] = None
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
        self.name_label.setStyleSheet(
            f"font-size: {ROW_FONT_PX}px; background: transparent;"
        )
        layout.addWidget(self.name_label)

        self.grid_label = QLabel()
        self.grid_label.setVisible(False)
        layout.addWidget(self.grid_label)

        self.status_label = ElidedLabel()
        layout.addWidget(self.status_label, 1)

        # Only drawn once there is a defect; a tick on every healthy row says
        # nothing. Clicking it opens the same defect menu the actions carry.
        self.btn_defect = QToolButton()
        self.btn_defect.setFixedSize(_BTN_SIZE)
        self.btn_defect.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self._defect_icon_menu = QMenu(self)
        add_defect_menu(self._defect_icon_menu, lamella, self._on_defect_written)
        self.btn_defect.clicked.connect(
            lambda: self._defect_icon_menu.popup(
                self.btn_defect.mapToGlobal(self.btn_defect.rect().bottomLeft())
            )
        )
        self.btn_defect.setVisible(False)
        layout.addWidget(self.btn_defect)

        self.btn_actions = QToolButton()
        self.btn_actions.setFixedSize(_BTN_SIZE)
        self.btn_actions.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self.btn_actions.setIcon(
            fibsem_icon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_actions.setToolTip("Actions")
        self.btn_actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        self.action_edit = menu.addAction(
            fibsem_icon("mdi:pencil", color=stylesheets.GRAY_ICON_COLOR), "Edit"
        )
        self.action_remove = menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR),
            "Remove",
        )
        self.defect_menu = add_defect_menu(menu, lamella, self._on_defect_written)
        self.btn_actions.setMenu(menu)
        layout.addWidget(self.btn_actions)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.lamella, bool(s))
        )
        self.btn_defect.installEventFilter(self)
        self.action_edit.triggered.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.action_remove.triggered.connect(self._on_remove_clicked)

        # `defect` and `description` are written by GUI widgets, on the GUI
        # thread. `task_state.name`/`.status` are deliberately not subscribed:
        # they are written from the workflow's worker thread, the marshalled
        # refresh arrives after `clear()` may have destroyed this row, and an
        # exception escaping a Qt slot aborts the process (FIB-604, FIB-329).
        # `_on_workflow_update` refreshes this row by name instead.
        lamella.events.defect.connect(self.refresh)  # type: ignore[union-attr]
        lamella.events.description.connect(self.refresh)  # type: ignore[union-attr]

        self.refresh()

    def _on_defect_written(self) -> None:
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
        thumb = self.lamella.get_thumbnail()
        self._popup.set_image(thumb)
        self._popup.show_near_cursor()

    def set_grid_context(self, context: Optional[GridContext]) -> None:
        self._grid_context = context
        self.refresh()

    @ensure_main_thread
    def refresh(self) -> None:
        """Re-read all display fields from the stored Lamella."""
        self.name_label.setText(self.lamella.name)
        self.setToolTip(self.lamella.description or "")

        icon_name, icon_color, tooltip = _defect_icon(self.lamella)
        self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self.btn_defect.setToolTip(tooltip)
        self.btn_defect.setVisible(has_defect(self.lamella))

        grid = grid_of(self.lamella, self._grid_context)
        status_text, status_style = _status_text(self.lamella)
        apply_grid_label(
            self.grid_label, grid, grids_named(self._grid_context), bool(status_text)
        )
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(
            f"font-size: {DETAIL_FONT_PX}px; "
            + (status_style or f"color: {NEUTRAL_550}; background: transparent;")
        )


class _ToggleLabel(QLabel):
    """A label that toggles the tick box it captions, as a checkbox's own text
    would."""

    def __init__(self, text: str, checkbox: QCheckBox) -> None:
        super().__init__(text)
        self._checkbox = checkbox

    def mousePressEvent(self, event) -> None:
        self._checkbox.toggle()
        super().mousePressEvent(event)


# The filter's two fixed choices; a grid's own key is its record id.
FILTER_ALL = "all"
FILTER_NO_GRID = "none"


class _GridFilterButton(QToolButton):
    """A filter icon whose menu narrows the list to one grid's lamellae (FIB-667).

    Shown only when the experiment has more than one grid: with one there is
    nothing to choose between. "All grids" is the resting state; "No grid"
    appears when some lamellae are linked to none. The icon takes the accent
    while a filter is on, so a narrowed list is never mistaken for the whole.
    """

    filter_changed = pyqtSignal(str)  # FILTER_ALL, FILTER_NO_GRID or a grid id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(_BTN_SIZE)
        self.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self.setPopupMode(QToolButton.InstantPopup)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._actions: Dict[str, QAction] = {}
        self._names: Dict[str, str] = {}
        self._selected = FILTER_ALL
        self._paint()
        self.setVisible(False)

    @property
    def selected(self) -> str:
        return self._selected

    def set_choices(self, grids: List[Tuple[str, str]], unlinked: bool) -> None:
        """*grids* as (id, name); *unlinked* whether a "No grid" entry is needed.
        Keeps the current choice when it is still on offer."""
        self._menu.clear()
        self._actions = {}
        choices = [(FILTER_ALL, "All grids")] + list(grids)
        if unlinked:
            choices.append((FILTER_NO_GRID, "No grid"))
        self._names = dict(choices)
        if self._selected not in self._names:
            self._selected = FILTER_ALL
        for key, text in choices:
            action = self._menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(key == self._selected)
            action.triggered.connect(lambda _c=False, k=key: self.select(k))
            self._actions[key] = action
        self._paint()
        self.setVisible(len(grids) > 1)

    def select(self, key: str) -> None:
        if key not in self._actions:
            key = FILTER_ALL
        self._selected = key
        for k, action in self._actions.items():
            action.setChecked(k == key)
        self._paint()
        self.filter_changed.emit(key)

    def action(self, key: str):
        return self._actions.get(key)

    def _paint(self) -> None:
        on = self._selected != FILTER_ALL
        self.setIcon(
            fibsem_icon(
                "mdi:filter-variant",
                color=ACCENT_COLOR if on else stylesheets.GRAY_ICON_COLOR,
            )
        )
        self.setToolTip(
            f"Showing {self._names.get(self._selected, 'all grids')}"
            if on
            else "Filter by grid"
        )


class _LamellaListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # The same shape as a row -- a textless tick box, then the name column
        # -- so the style sizes the tick box here and there alike and "Status"
        # starts where the rows' status does.
        self.checkbox_all = QCheckBox()
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("background: transparent;")
        self.checkbox_all.setToolTip("Select all")
        layout.addWidget(self.checkbox_all)
        self.label_all = _ToggleLabel("Select All", self.checkbox_all)
        self.label_all.setMinimumWidth(_NAME_MIN_WIDTH)
        self.label_all.setStyleSheet(
            f"font-weight: bold; font-size: {ROW_FONT_PX}px; background: transparent;"
        )
        layout.addWidget(self.label_all)

        status_header = QLabel("Status")
        status_header.setStyleSheet(
            f"font-weight: bold; font-size: {ROW_FONT_PX}px; background: transparent;"
        )
        layout.addWidget(status_header, 1)

        # Over the rows' two trailing buttons: a blank where the defect icon
        # sits, and the grid filter over the actions column.
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SIZE.width())
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)
        self.grid_filter = _GridFilterButton()
        layout.addWidget(self.grid_filter)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )


class LamellaListWidget(QWidget):
    """List widget displaying Lamella objects with name, defect, status and actions."""

    move_to_requested = pyqtSignal(object)  # Lamella
    edit_requested = pyqtSignal(object)  # Lamella
    remove_requested = pyqtSignal(object)  # Lamella
    defect_changed = pyqtSignal(object)  # Lamella
    selection_changed = pyqtSignal(list)  # List[Lamella]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _LamellaListHeader()
        layout.addWidget(self._header)
        self.grid_filter = self._header.grid_filter
        self.grid_filter.filter_changed.connect(self._apply_grid_filter)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.setSpacing(0)
        # The application sheet pads items by 4px; the row widgets carry their
        # own margins, and the padding put them 4px in from the header.
        self._list.setStyleSheet(
            stylesheets.LIST_WIDGET_STYLESHEET + "QListWidget::item { padding: 0; }"
        )
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._header.select_all_changed.connect(self.set_all_selected)

        self._btn_visible = {
            "edit": True,
            "remove": True,
            "defect": True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_grid_context(self, context: Optional[GridContext]) -> None:
        """Which grid each lamella is on and whether it is on the stage; the
        rows name the grid on their status line, and the filter offers the
        grids. Kept for rows added later."""
        self._grid_context = context
        for i in range(self._list.count()):
            self._row(i).set_grid_context(context)
        self._refresh_grid_filter()

    # ------------------------------------------------------------------
    # Filtering by grid (FIB-667)
    # ------------------------------------------------------------------

    def _refresh_grid_filter(self) -> None:
        context = getattr(self, "_grid_context", None) or {}
        grids = [(grid_id, name) for grid_id, (name, _) in context.items()]
        unlinked = any(
            grid_of(self._row(i).lamella, context) is None
            for i in range(self._list.count())
        )
        self.grid_filter.set_choices(grids, unlinked)
        self._apply_grid_filter(self.grid_filter.selected)

    def _row_passes(self, row: "LamellaRowWidget", key: str) -> bool:
        if key == FILTER_ALL:
            return True
        grid_id = getattr(row.lamella, "grid_id", None)
        context = getattr(self, "_grid_context", None) or {}
        known = grid_id if grid_id in context else None
        if key == FILTER_NO_GRID:
            return known is None
        return known == key

    def _apply_grid_filter(self, key: str) -> None:
        """Hide the rows not on the chosen grid, and untick them: a hidden
        tick would run a lamella the list is not showing."""
        for i in range(self._list.count()):
            row = self._row(i)
            shown = self._row_passes(row, key)
            self._list.item(i).setHidden(not shown)
            if not shown and row.checkbox.isChecked():
                row.checkbox.blockSignals(True)
                row.checkbox.setChecked(False)
                row.checkbox.blockSignals(False)
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _visible_rows(self) -> List["LamellaRowWidget"]:
        return [
            self._row(i)
            for i in range(self._list.count())
            if not self._list.item(i).isHidden()
        ]

    def add_lamella(self, lamella: Lamella, checked: bool = False) -> LamellaRowWidget:
        row = LamellaRowWidget(lamella, checked)
        row.set_grid_context(getattr(self, "_grid_context", None))
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        item.setHidden(not self._row_passes(row, self.grid_filter.selected))

        row.move_to_clicked.connect(self.move_to_requested)
        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.defect_changed.connect(self.defect_changed)
        row.selection_changed.connect(self._on_row_selection_changed)

        self._apply_btn_visibility(row)
        self._sync_select_all()
        return row

    def enable_actions_button(self, visible: bool) -> None:
        self.enable_edit_action(visible)

    def enable_move_to_action(self, visible: bool) -> None:
        pass  # not available

    def enable_edit_action(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for i in range(self._list.count()):
            self._apply_btn_visibility(self._row(i))

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for i in range(self._list.count()):
            self._apply_btn_visibility(self._row(i))

    def enable_defect_button(self, visible: bool) -> None:
        self._btn_visible["defect"] = visible
        for i in range(self._list.count()):
            self._apply_btn_visibility(self._row(i))

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
        """The ticked lamellae among those the list is showing."""
        return [row.lamella for row in self._visible_rows() if row.checkbox.isChecked()]

    def set_lamellae(self, lamellae: List[Lamella]) -> None:
        """Rebuild the rows from *lamellae*, keeping the ticks the user already has.

        The host rebuilds on every insert or removal in the experiment, and the
        ticked rows are its run selection. Rebuilding them unticked emptied that
        selection, and silently disabled Run, every time a position was added
        (FIB-966). Ticks are matched by lamella id, so a lamella that is gone is
        simply not re-ticked.
        """
        checked = {lamella.id for lamella in self.get_selected()}
        self.clear()
        for lamella in lamellae:
            self.add_lamella(lamella, checked=lamella.id in checked)
        # The filter's choices depend on the rows (whether any is unlinked).
        self._refresh_grid_filter()

    def clear(self) -> None:
        self._list.clear()

    def set_all_selected(self, checked: bool) -> None:
        """Tick or untick every row, and bring the header checkbox with them.

        Public because the header is not the only thing that clears the selection --
        starting a run and adding to the queue both do. Those callers used to invoke
        the header's slot directly, which left the box reading "Select All" over a
        list where nothing was ticked (FIB-577).
        """
        for row in self._visible_rows():
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
        # Redundant when the header emitted into here (it is already right), and
        # load-bearing for every other caller. Cheap enough not to branch on.
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row(self, i: int) -> LamellaRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _apply_btn_visibility(self, row: LamellaRowWidget) -> None:
        row.action_edit.setVisible(self._btn_visible["edit"])
        row.action_remove.setVisible(self._btn_visible["remove"])
        row.defect_menu.menuAction().setVisible(self._btn_visible["defect"])
        if not self._btn_visible["defect"]:
            row.btn_defect.setVisible(False)

    def _on_remove_clicked(self, lamella: Lamella) -> None:
        self.remove_lamella(lamella)
        self.remove_requested.emit(lamella)

    def _on_row_selection_changed(self, *_) -> None:
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _sync_select_all(self) -> None:
        rows = self._visible_rows()
        count = len(rows)
        if count == 0:
            return
        n_checked = sum(row.checkbox.isChecked() for row in rows)
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if n_checked == 0:
            cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            cb.setCheckState(Qt.Checked)
        else:
            cb.setTristate(True)
            cb.setCheckState(Qt.PartiallyChecked)
        cb.blockSignals(False)
