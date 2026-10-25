"""The lamella list: a name-per-row list with status and a defect indicator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QPixmap
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

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.common.lamella_display import (
    GridContext,
    add_defect_menu,
    apply_grid_label,
    defect_icon,
    grid_of,
    grids_named,
    has_verdict,
    not_on_stage_reason,
    status_look,
)
from fibsem.theme import tokens
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    CANVAS_BG,
    NEUTRAL_550,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION, fibsem_icon
from fibsem.ui.qt.connections import connect_for_lifetime
from fibsem.ui.utils import arr_to_qimage
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton
from fibsem.ui.widgets.lists import (
    DETAIL_PX,
    NAME_PX,
    ROW_HEIGHT,
    ListHeader,
    column_label,
)

_NAME_MIN_WIDTH = 160


_BTN_SIZE = QSize(24, 24)


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
        self._img_label.setPixmap(
            QPixmap.fromImage(arr_to_qimage(arr)).scaled(
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


class LamellaRow(QWidget):
    """One lamella in a list: name, the grid it is on (when the experiment has
    more than one), a short status, a defect icon only once there is a defect,
    and one actions menu.

    *checkable* adds the tick box the run selection uses; *stage_actions* adds
    Move to and Update Position, withheld while the lamella's grid is off the
    stage; *thumbnail* shows the lamella's thumbnail on hovering its name."""

    move_to_clicked = pyqtSignal(object)  # Lamella
    edit_clicked = pyqtSignal(object)  # Lamella
    update_clicked = pyqtSignal(object)  # Lamella
    remove_clicked = pyqtSignal(object)  # Lamella
    defect_changed = pyqtSignal(object)  # Lamella
    selection_changed = pyqtSignal(object, bool)  # Lamella, checked

    def __init__(
        self,
        lamella: Lamella,
        checkable: bool = False,
        checked: bool = False,
        stage_actions: bool = False,
        thumbnail: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self._grid_context: Optional[GridContext] = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox: Optional[QCheckBox] = None
        if checkable:
            self.checkbox = QCheckBox()
            self.checkbox.setChecked(checked)
            self.checkbox.setStyleSheet("background: transparent;")
            self.checkbox.stateChanged.connect(
                lambda s: self.selection_changed.emit(self.lamella, bool(s))
            )
            layout.addWidget(self.checkbox)

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet(
            f"font-size: {NAME_PX}px; background: transparent;"
        )
        layout.addWidget(self.name_label)

        self.grid_label = QLabel()
        self.grid_label.setVisible(False)
        layout.addWidget(self.grid_label)

        self.status_label = ElidedLabel()
        layout.addWidget(self.status_label, 1)

        # Only drawn once someone has judged the lamella; a mark on every row nobody
        # has looked at says nothing. Clicking it opens the same verdict menu the
        # actions carry.
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
            fibsem_icon("mdi:dots-horizontal", color=tokens.GRAY_ICON_COLOR)
        )
        self.btn_actions.setToolTip("Actions")
        self.btn_actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        self.action_move_to: Optional[QAction] = None
        self.action_update: Optional[QAction] = None
        if stage_actions:
            self.action_move_to = menu.addAction(
                fibsem_icon(ICON_MOVE_TO_POSITION, color=tokens.GRAY_ICON_COLOR),
                "Move to Position",
            )
            self.action_update = menu.addAction(
                fibsem_icon(ICON_UPDATE_POSITION, color=tokens.GRAY_ICON_COLOR),
                "Update Position",
            )
            self.action_move_to.triggered.connect(
                lambda: self.move_to_clicked.emit(self.lamella)
            )
            self.action_update.triggered.connect(
                lambda: self.update_clicked.emit(self.lamella)
            )
        self.action_edit = menu.addAction(
            fibsem_icon("mdi:pencil", color=tokens.GRAY_ICON_COLOR), "Edit"
        )
        self.action_remove = menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=tokens.GRAY_ICON_COLOR),
            "Remove",
        )
        self.defect_menu = add_defect_menu(menu, lamella, self._on_defect_written)
        self.btn_actions.setMenu(menu)
        layout.addWidget(self.btn_actions)

        self.action_edit.triggered.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.action_remove.triggered.connect(self._on_remove_clicked)

        self._popup: Optional[_LamellaTooltip] = None
        if thumbnail:
            self._hover_timer = QTimer(self)
            self._hover_timer.setSingleShot(True)
            self._hover_timer.timeout.connect(self._show_popup)
            # Hovering the name shows the thumbnail; the defect icon is drawn only
            # once there is a verdict, so it can't be the only way in.
            self.name_label.installEventFilter(self)
            self.btn_defect.installEventFilter(self)

        # `defect` and `description` are written on the GUI thread (by the
        # window, and by agent edits through the session). `task_state.name`/`.status` are deliberately not subscribed:
        # they are written from the workflow's worker thread, the marshalled
        # refresh arrives after `clear()` may have destroyed this row, and an
        # exception escaping a Qt slot aborts the process.
        # `_on_workflow_update` refreshes this row by name instead.
        connect_for_lifetime(self, lamella.events.defect, self.refresh)  # type: ignore[union-attr]
        connect_for_lifetime(self, lamella.events.description, self.refresh)  # type: ignore[union-attr]

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
        if obj is self.btn_defect or obj is self.name_label:
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

        icon_name, icon_color, tooltip = defect_icon(self.lamella)
        self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self.btn_defect.setToolTip(tooltip)
        self.btn_defect.setVisible(
            self.defect_menu.menuAction().isVisible() and has_verdict(self.lamella)
        )

        grid = grid_of(self.lamella, self._grid_context)
        status_text, status_style = status_look(self.lamella)
        apply_grid_label(
            self.grid_label,
            grid,
            grids_named(self._grid_context),
            bool(status_text),
            font_px=DETAIL_PX,
        )
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(
            f"font-size: {DETAIL_PX}px; "
            + (status_style or f"color: {NEUTRAL_550}; background: transparent;")
        )

        if self.action_move_to is None or self.action_update is None:
            return
        # Moving to or re-recording a lamella whose grid is in the magazine would
        # act on whatever grid *is* on the stage: withheld, with the reason.
        reason = not_on_stage_reason(grid)
        for action, label in (
            (self.action_move_to, "Move to Position"),
            (self.action_update, "Update Position"),
        ):
            action.setEnabled(not reason)
            action.setText(f"{label} ({reason})" if reason else label)


# The filter's two fixed choices; a grid's own key is its record id.
FILTER_ALL = "all"
FILTER_NO_GRID = "none"


class _GridFilterButton(QToolButton):
    """A filter icon whose menu narrows the list to one grid's lamellae.

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
                color=ACCENT_COLOR if on else tokens.GRAY_ICON_COLOR,
            )
        )
        self.setToolTip(
            f"Showing {self._names.get(self._selected, 'all grids')}"
            if on
            else "Filter by grid"
        )


class _LamellaListHeader(ListHeader):
    """The list's header: the shared one (`list_chrome.ListHeader`), with the rows'
    Status column named and the grid filter over the actions column."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Lamella", title_width=_NAME_MIN_WIDTH, parent=parent)
        self.add_widget(column_label("Status"), 1)
        # Over the rows' two trailing buttons: a blank where the defect icon
        # sits, and the grid filter over the actions column.
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SIZE.width())
        spacer.setStyleSheet("background: transparent;")
        self.add_widget(spacer)
        self.grid_filter = _GridFilterButton()
        self.add_widget(self.grid_filter)


class LamellaListWidget(QWidget):
    """List widget displaying Lamella objects with name, defect, status and actions."""

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
    # Filtering by grid
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

    def _row_passes(self, row: "LamellaRow", key: str) -> bool:
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

    def _visible_rows(self) -> List["LamellaRow"]:
        return [
            self._row(i)
            for i in range(self._list.count())
            if not self._list.item(i).isHidden()
        ]

    def add_lamella(self, lamella: Lamella, checked: bool = False) -> LamellaRow:
        row = LamellaRow(lamella, checkable=True, checked=checked, thumbnail=True)
        row.set_grid_context(getattr(self, "_grid_context", None))
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        item.setHidden(not self._row_passes(row, self.grid_filter.selected))

        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.defect_changed.connect(self.defect_changed)
        row.selection_changed.connect(self._on_row_selection_changed)

        self._sync_select_all()
        return row

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
        selection, and silently disabled Run, every time a position was added.
        Ticks are matched by lamella id, so a lamella that is gone is
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
        self._sync_select_all()

    def set_all_selected(self, checked: bool) -> None:
        """Tick or untick every row, and bring the header checkbox with them.

        Public because the header is not the only thing that clears the selection --
        starting a run and adding to the queue both do. Calling the header's slot
        directly would leave the box reading "Select All" over a list where nothing
        was ticked.
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

    def _row(self, i: int) -> LamellaRow:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _on_remove_clicked(self, lamella: Lamella) -> None:
        self.remove_lamella(lamella)
        self.remove_requested.emit(lamella)

    def _on_row_selection_changed(self, *_) -> None:
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _sync_select_all(self) -> None:
        rows = self._visible_rows()
        count = len(rows)
        self._header.set_count(str(count) if count else "")
        if count == 0:
            return
        n_checked = sum(row.checkbox.isChecked() for row in rows)
        self._header.sync(n_checked, count)


class LamellaNameListWidget(QWidget):
    """Single-selection list widget for lamella names with smart restore logic.

    Shows two columns: lamella name and last completed task status.
    Stores the associated object as ``Qt.UserRole`` data on each item so callers
    can retrieve it via ``selected_lamella``.

    Emits ``lamella_selected(object)`` when the selection changes.
    Call ``set_lamella()`` to repopulate; the current selection is preserved by
    name if still present, otherwise falls back to the first row.
    """

    lamella_selected = pyqtSignal(object)
    # the list as the ItemSelection the Image and Fluorescence tabs follow
    item_selected = lamella_selected
    item_noun = "Lamella"
    add_requested = pyqtSignal()
    move_to_requested = pyqtSignal(object)
    update_requested = pyqtSignal(object)
    remove_requested = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._btn_visible = {
            "defect": False,
            "move_to": False,
            "update": False,
            "remove": False,
        }

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Not `ListHeader`: that carries a tick box, and this list picks one.
        header = QWidget()
        header.setStyleSheet(f"background: {CANVAS_BG};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 4, 3)
        header_layout.setSpacing(8)
        lbl_name = QLabel("Lamella")
        lbl_name.setStyleSheet(
            f"font-weight: bold; font-size: {NAME_PX}px; background: transparent;"
        )
        lbl_name.setMinimumWidth(_NAME_MIN_WIDTH)
        header_layout.addWidget(lbl_name)
        lbl_status = QLabel("Status")
        lbl_status.setStyleSheet(
            f"font-weight: bold; font-size: {NAME_PX}px; background: transparent;"
        )
        header_layout.addWidget(lbl_status, stretch=1)
        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add", size=_BTN_SIZE.width()
        )
        self.btn_add.setVisible(False)
        self.btn_add.clicked.connect(self.add_requested.emit)
        header_layout.addWidget(self.btn_add)
        outer.addWidget(header)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._list)

        # Wire signals
        self._list.itemSelectionChanged.connect(
            lambda: self.lamella_selected.emit(self.selected_lamella)
        )

    @property
    def selected_lamella(self) -> Any:
        """Return the ``Qt.UserRole`` data of the current item, or ``None``."""
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None  # type: ignore

    @property
    def selected_item(self) -> Any:
        return self.selected_lamella

    @property
    def selected_name(self) -> str:
        """Return the display name of the current item, or ``""``."""
        lamella = self.selected_lamella
        return lamella.name if lamella is not None else ""

    @property
    def selected_index(self) -> int:
        """Return the current row index, or -1 if nothing is selected."""
        return self._list.currentRow()

    def set_grid_context(self, context) -> None:
        """`GridRecord.id -> (name, on the stage)`, or None; rows name the
        grid ahead of their status when there is more than one, and withhold
        stage actions for a grid off the stage. Kept for rows added later."""
        self._grid_context = context
        for row in self._rows():
            row.set_grid_context(context)

    def set_lamella(self, positions, preferred_name: str = "") -> None:
        """Populate the list from *positions*, restoring selection by name.

        Priority: current selection → *preferred_name* → first row.
        Signals are suppressed during population.
        """
        current = self.selected_name or preferred_name
        self._list.blockSignals(True)
        self._list.clear()
        for pos in positions:
            row = LamellaRow(pos, stage_actions=True)
            row.set_grid_context(getattr(self, "_grid_context", None))
            row.move_to_clicked.connect(self.move_to_requested)
            row.update_clicked.connect(self.update_requested)
            row.remove_clicked.connect(self.remove_requested)
            row.defect_changed.connect(self.defect_changed)
            self._apply_btn_visibility(row)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, pos)  # type: ignore
            item.setSizeHint(QSize(0, ROW_HEIGHT))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._restore_selection(current)
        self._list.blockSignals(False)

    def refresh_lamella(self, lamella: Any) -> None:
        """Refresh the one row displaying *lamella*, matched by identity.

        The workflow path calls this rather than ``refresh_all``: a running workflow
        changes one lamella at a time, and ``refresh_all`` is O(rows) of GUI-thread work
        for it. A row repaint measures ~0.03 ms, two thirds of it the ``setStyleSheet``
        re-parse, so 70 lamellae cost ~2.4 ms per update against ~0.05 ms here.

        Identity, like ``LamellaListWidget.refresh_lamella``: rows and the caller's
        ``get_lamella_by_name`` both come from ``experiment.positions``, and nothing
        replaces an entry in it.
        """
        for row in self._rows():
            if row.lamella is lamella:
                row.refresh()
                return

    def refresh_all(self) -> None:
        """Refresh the display of all rows (e.g. after status changes)."""
        for row in self._rows():
            row.refresh()

    # ------------------------------------------------------------------
    # Button visibility
    # ------------------------------------------------------------------

    def enable_add_button(self, visible: bool) -> None:
        self.btn_add.setVisible(visible)

    def enable_defect_button(self, visible: bool) -> None:
        self._set_visible("defect", visible)

    def enable_move_to_action(self, visible: bool) -> None:
        self._set_visible("move_to", visible)

    def enable_update_action(self, visible: bool) -> None:
        self._set_visible("update", visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._set_visible("remove", visible)

    def _set_visible(self, key: str, visible: bool) -> None:
        self._btn_visible[key] = visible
        for row in self._rows():
            self._apply_btn_visibility(row)

    def _rows(self):
        """Yield all LamellaRow widgets."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, LamellaRow):
                yield row

    def _apply_btn_visibility(self, row: LamellaRow) -> None:
        v = self._btn_visible
        row.defect_menu.menuAction().setVisible(v["defect"])
        row.action_move_to.setVisible(v["move_to"])
        row.action_edit.setVisible(False)
        row.action_update.setVisible(v["update"])
        row.action_remove.setVisible(v["remove"])
        row.btn_actions.setVisible(
            any(v[k] for k in ("defect", "move_to", "update", "remove"))
        )
        row.refresh()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, name: str) -> None:
        """Select the item with the given name (exact match)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            obj = item.data(Qt.UserRole)
            if obj is not None and obj.name == name:
                self._list.setCurrentItem(item)
                return

    def _restore_selection(self, preferred: str) -> None:
        if preferred:
            for i in range(self._list.count()):
                item = self._list.item(i)
                obj = item.data(Qt.UserRole)
                if obj is not None and obj.name == preferred:
                    self._list.setCurrentItem(item)
                    return
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
