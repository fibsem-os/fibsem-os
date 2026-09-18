"""The lamella list: a name-per-row list with status and a defect indicator.

Lifted out of ``custom_widgets`` (FIB-554). It was the only thing in that module
that knew about AutoLamella, and ``custom_widgets`` is imported by 78 files -- so
the shared widget library dragged the application into every one of them. The
imports were deferred into function bodies to soften that, which worked, but left
an application widget living in the generic library.

Nothing here is generic: a row renders ``lamella.task_state`` and ``lamella.defect``
and the defect menu writes ``DefectState`` back. So the imports are plain
module-level ones now -- there is no cycle (``autolamella.structures`` does not
import ``fibsem.ui``) and several sibling widgets already do the same.

This belongs under ``fibsem/applications/autolamella/ui/`` eventually. It cannot go
there yet: three of its consumers (the minimap, the coincidence viewer and the spot
burn widget) live in ``fibsem/ui/``, so moving it now would create three new
generic-to-application imports rather than removing any. That move is FIB-559.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
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
    PoseProvenance,
)
from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION, fibsem_icon
from fibsem.ui.tokens import CANVAS_BG, NEUTRAL_550
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton

_LAMELLA_NAME_MIN_WIDTH = 160
_LAMELLA_ROW_HEIGHT = 30
_LAMELLA_BTN_SIZE = QSize(24, 24)
_ROW_FONT_PX = 11
_DETAIL_FONT_PX = 10


def _lamella_status_text(lamella) -> tuple[str, str]:
    """Return (text, stylesheet) for the status column.

    Gracefully handles objects without task_state (e.g. plain positions).
    """
    ts = getattr(lamella, "task_state", None)
    if ts and getattr(ts, "status", None) is not None:
        if ts.status == AutoLamellaTaskStatus.InProgress:
            return f"{ts.name}", "color: #6aabdf; background: transparent;"
    last = getattr(lamella, "last_completed_task", None)
    if last:
        return last.completed, f"color: {NEUTRAL_550}; background: transparent;"
    return "", "background: transparent;"


def _lamella_defect_icon(lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator.

    Gracefully handles objects without defect field.
    """
    defect = getattr(lamella, "defect", None)
    if defect is None:
        return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"
    if defect.state == DefectType.REWORK:
        desc = f": {defect.description}" if defect.description else ""
        return (
            "mdi:refresh-circle",
            stylesheets.DEFECT_ORANGE_COLOR,
            f"Rework required{desc}",
        )
    if defect.state == DefectType.FAILURE:
        desc = f": {defect.description}" if defect.description else ""
        return "mdi:close-circle", stylesheets.DEFECT_RED_COLOR, f"Failure{desc}"
    return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"


def _lamella_pose_icon(lamella) -> tuple[str, str, str] | None:
    """(icon_name, colour, tooltip) for the pose-provenance mark, or None for nothing.

    A stale pose is the thing to see at a glance -- somebody centred it by hand and
    the other pose has moved since. A derived pose gets a quieter mark, so the rows
    still to be centred by hand can be found. Two observed poses say nothing.
    """
    provenance_of = getattr(lamella, "provenance_of", None)
    poses = getattr(lamella, "poses", None) or {}
    if provenance_of is None or not poses:
        return None
    states = {name: provenance_of(name) for name in poses}
    stale = [name.lower() for name, p in states.items() if p is PoseProvenance.STALE]
    if stale:
        return (
            "mdi:link-variant-off",
            stylesheets.SEMANTIC_WARNING_COLOR,
            f"The {' and '.join(stale)} pose may be stale: the other pose has moved "
            f"since it was set by hand.",
        )
    derived = [
        name.lower() for name, p in states.items() if p is PoseProvenance.DERIVED
    ]
    if derived:
        return (
            "mdi:link-variant",
            stylesheets.GRAY_ICON_COLOR,
            f"The {' and '.join(derived)} pose is derived from the other, not yet "
            f"centred by hand.",
        )
    return None


class _LamellaRow(QWidget):
    """One row: name, the grid it is on (when there is more than one), a short
    status, a pose-provenance mark when a pose is derived or stale, a defect icon
    only once there is a defect, and one actions menu."""

    move_to_clicked = pyqtSignal(object)
    edit_clicked = pyqtSignal(object)
    update_clicked = pyqtSignal(object)
    remove_clicked = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, lamella, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.lamella = lamella
        # (grid name, on the stage, named at all), from the host; see
        # `lamella_list_widget.GridContext`. That module is the application's
        # and this one is still in the shared layer (FIB-559), so the small
        # helpers it needs are imported inside the methods that use them.
        self._grid: Optional[Tuple[str, bool]] = None
        self._grid_named = False
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_label = QLabel(lamella.name)
        self.name_label.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet(
            f"font-size: {_ROW_FONT_PX}px; background: transparent;"
        )
        layout.addWidget(self.name_label)

        self.grid_label = QLabel()
        self.grid_label.setVisible(False)
        layout.addWidget(self.grid_label)

        self.status_label = ElidedLabel()
        layout.addWidget(self.status_label, stretch=1)

        # Pose provenance: a mark, not a control. Hidden when there is nothing to say.
        self.pose_icon = QLabel()
        self.pose_icon.setFixedSize(_LAMELLA_BTN_SIZE)
        self.pose_icon.setAlignment(Qt.AlignCenter)
        self.pose_icon.setStyleSheet("background: transparent;")
        self.pose_icon.setVisible(False)
        layout.addWidget(self.pose_icon)

        from fibsem.applications.autolamella.ui.lamella_list_widget import (
            add_defect_menu,
        )

        # Only drawn once there is a defect. Clicking it opens the defect menu.
        self.btn_defect = QToolButton()
        self.btn_defect.setFixedSize(_LAMELLA_BTN_SIZE)
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

        # One actions menu in place of four inline buttons.
        self.btn_actions = QToolButton()
        self.btn_actions.setFixedSize(_LAMELLA_BTN_SIZE)
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
        self.action_move_to = menu.addAction(
            fibsem_icon(ICON_MOVE_TO_POSITION, color=stylesheets.GRAY_ICON_COLOR),
            "Move to Position",
        )
        self.action_update = menu.addAction(
            fibsem_icon(ICON_UPDATE_POSITION, color=stylesheets.GRAY_ICON_COLOR),
            "Update Position",
        )
        self.action_edit = menu.addAction(
            fibsem_icon("mdi:pencil", color=stylesheets.GRAY_ICON_COLOR), "Edit"
        )
        self.action_remove = menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR),
            "Remove",
        )
        self.defect_menu = add_defect_menu(menu, lamella, self._on_defect_written)
        self.btn_actions.setMenu(menu)
        self.btn_actions.setVisible(False)
        layout.addWidget(self.btn_actions)

        self.action_move_to.triggered.connect(
            lambda: self.move_to_clicked.emit(self.lamella)
        )
        self.action_update.triggered.connect(
            lambda: self.update_clicked.emit(self.lamella)
        )
        self.action_edit.triggered.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.action_remove.triggered.connect(self._on_remove_clicked)

        # `defect` and `description` only: both are written by GUI widgets, on
        # the GUI thread. Not `task_state.name`/`.status`: those are written
        # from the workflow's worker thread, the marshalled refresh could land
        # on a row Qt has already destroyed, and an exception escaping a slot
        # aborts the process (FIB-565, FIB-329). `_on_workflow_update` drives
        # this list by name instead.
        self.lamella.events.description.connect(self.refresh)  # type: ignore[union-attr]
        self.lamella.events.defect.connect(self.refresh)  # type: ignore[union-attr]

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

    def set_grid(self, grid: Optional[Tuple[str, bool]], named: bool) -> None:
        """(grid name, on the stage), or None for a lamella on no known grid;
        *named* says whether rows name their grid at all (more than one grid)."""
        self._grid = grid
        self._grid_named = named
        self.refresh()

    @ensure_main_thread
    def refresh(self) -> None:
        from fibsem.applications.autolamella.ui.lamella_list_widget import (
            apply_grid_label,
            not_on_stage_reason,
        )

        self.name_label.setText(self.lamella.name)
        self.setToolTip(self.lamella.description or "")
        text, style = _lamella_status_text(self.lamella)
        apply_grid_label(self.grid_label, self._grid, self._grid_named, bool(text))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size: {_DETAIL_FONT_PX}px; "
            + (style or f"color: {NEUTRAL_550}; background: transparent;")
        )
        mark = _lamella_pose_icon(self.lamella)
        if mark is None:
            self.pose_icon.setVisible(False)
        else:
            icon_name, icon_color, tooltip = mark
            self.pose_icon.setPixmap(
                fibsem_icon(icon_name, color=icon_color).pixmap(16, 16)
            )
            self.pose_icon.setToolTip(tooltip)
            self.pose_icon.setVisible(True)
        icon_name, icon_color, tooltip = _lamella_defect_icon(self.lamella)
        self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self.btn_defect.setToolTip(tooltip)
        defect = getattr(self.lamella, "defect", None)
        self.btn_defect.setVisible(
            self.defect_menu.menuAction().isVisible()
            and defect is not None
            and defect.state != DefectType.NONE
        )
        # Moving to or re-recording a lamella whose grid is in the magazine would
        # act on whatever grid *is* on the stage: withheld, with the reason.
        reason = not_on_stage_reason(self._grid)
        for action, label in (
            (self.action_move_to, "Move to Position"),
            (self.action_update, "Update Position"),
        ):
            action.setEnabled(not reason)
            action.setText(f"{label} ({reason})" if reason else label)


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
    add_requested = pyqtSignal()
    move_to_requested = pyqtSignal(object)
    edit_requested = pyqtSignal(object)
    update_requested = pyqtSignal(object)
    remove_requested = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._btn_visible = {
            "defect": False,
            "move_to": False,
            "edit": False,
            "update": False,
            "remove": False,
        }

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"background: {CANVAS_BG};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 4, 3)
        header_layout.setSpacing(8)
        lbl_name = QLabel("Lamella")
        lbl_name.setStyleSheet(
            f"font-weight: bold; font-size: {_ROW_FONT_PX}px; background: transparent;"
        )
        lbl_name.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        header_layout.addWidget(lbl_name)
        lbl_status = QLabel("Status")
        lbl_status.setStyleSheet(
            f"font-weight: bold; font-size: {_ROW_FONT_PX}px; background: transparent;"
        )
        header_layout.addWidget(lbl_status, stretch=1)
        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add", size=_LAMELLA_BTN_SIZE.width()
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
            row.set_grid(self._grid_of(row.lamella), self._grids_named())

    def _grids_named(self) -> bool:
        context = getattr(self, "_grid_context", None)
        return bool(context) and len(context) > 1

    def _grid_of(self, lamella) -> Optional[Tuple[str, bool]]:
        context = getattr(self, "_grid_context", None)
        grid_id = getattr(lamella, "grid_id", None)
        if not context or not grid_id:
            return None
        return context.get(grid_id)

    def set_lamella(self, positions, preferred_name: str = "") -> None:
        """Populate the list from *positions*, restoring selection by name.

        Priority: current selection → *preferred_name* → first row.
        Signals are suppressed during population.
        """
        current = self.selected_name or preferred_name
        self._list.blockSignals(True)
        self._list.clear()
        for pos in positions:
            row = _LamellaRow(pos)
            row.set_grid(self._grid_of(pos), self._grids_named())
            row.move_to_clicked.connect(self.move_to_requested)
            row.edit_clicked.connect(self.edit_requested)
            row.update_clicked.connect(self.update_requested)
            row.remove_clicked.connect(self.remove_requested)
            row.defect_changed.connect(self.defect_changed)
            self._apply_btn_visibility(row)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, pos)  # type: ignore
            item.setSizeHint(QSize(0, _LAMELLA_ROW_HEIGHT))
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

    def enable_actions_button(self, visible: bool) -> None:
        """Kept for callers: every action lives on the one actions menu, which
        shows whenever any of them does."""

    def enable_move_to_action(self, visible: bool) -> None:
        self._set_visible("move_to", visible)

    def enable_edit_action(self, visible: bool) -> None:
        self._set_visible("edit", visible)

    def enable_update_action(self, visible: bool) -> None:
        self._set_visible("update", visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._set_visible("remove", visible)

    def _set_visible(self, key: str, visible: bool) -> None:
        self._btn_visible[key] = visible
        for row in self._rows():
            self._apply_btn_visibility(row)

    def _rows(self):
        """Yield all _LamellaRow widgets."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, _LamellaRow):
                yield row

    def _apply_btn_visibility(self, row: _LamellaRow) -> None:
        v = self._btn_visible
        row.defect_menu.menuAction().setVisible(v["defect"])
        row.action_move_to.setVisible(v["move_to"])
        row.action_edit.setVisible(v["edit"])
        row.action_update.setVisible(v["update"])
        row.action_remove.setVisible(v["remove"])
        row.btn_actions.setVisible(
            any(v[k] for k in ("defect", "move_to", "edit", "update", "remove"))
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
