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

from typing import Any, List, Optional

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
)
from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION, fibsem_icon
from fibsem.ui.tokens import CANVAS_BG, NEUTRAL_550
from fibsem.ui.widgets.custom_widgets import IconToolButton

_LAMELLA_NAME_MIN_WIDTH = 160
_LAMELLA_ROW_HEIGHT = 30
_LAMELLA_BTN_SIZE = QSize(24, 24)


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
        return "mdi:refresh-circle", stylesheets.DEFECT_ORANGE_COLOR, f"Rework required{desc}"
    if defect.state == DefectType.FAILURE:
        desc = f": {defect.description}" if defect.description else ""
        return "mdi:close-circle", stylesheets.DEFECT_RED_COLOR, f"Failure{desc}"
    return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"


class _LamellaRow(QWidget):
    """Single row: name + status labels + optional toolbuttons."""

    move_to_clicked = pyqtSignal(object)
    edit_clicked = pyqtSignal(object)
    update_clicked = pyqtSignal(object)
    remove_clicked = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, lamella, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_label = QLabel(lamella.name)
        self.name_label.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel()
        layout.addWidget(self.status_label, stretch=1)

        # Defect button
        self.btn_defect = QToolButton()
        self.btn_defect.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_defect.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_defect.clicked.connect(self._on_defect_clicked)
        self.btn_defect.setVisible(False)
        layout.addWidget(self.btn_defect)

        # Direct action buttons (replaces "…" dropdown)
        self.btn_move_to = QToolButton()
        self.btn_move_to.setIcon(fibsem_icon(ICON_MOVE_TO_POSITION, color=stylesheets.GRAY_ICON_COLOR))
        self.btn_move_to.setToolTip("Move to Position")
        self.btn_move_to.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_move_to.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_move_to.setVisible(False)
        layout.addWidget(self.btn_move_to)

        self.btn_update = QToolButton()
        self.btn_update.setIcon(fibsem_icon(ICON_UPDATE_POSITION, color=stylesheets.GRAY_ICON_COLOR))
        self.btn_update.setToolTip("Update Position")
        self.btn_update.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_update.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_update.setVisible(False)
        layout.addWidget(self.btn_update)

        self.btn_edit = QToolButton()
        self.btn_edit.setIcon(fibsem_icon("mdi:pencil", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_edit.setToolTip("Edit Lamella")
        self.btn_edit.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_edit.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_edit.setVisible(False)
        layout.addWidget(self.btn_edit)

        self.btn_move_to.clicked.connect(lambda: self.move_to_clicked.emit(self.lamella))
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.btn_update.clicked.connect(lambda: self.update_clicked.emit(self.lamella))

        # Remove button
        self.btn_remove = QToolButton()
        self.btn_remove.setIcon(fibsem_icon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_remove.setToolTip("Remove")
        self.btn_remove.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_remove.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_remove.setVisible(False)
        layout.addWidget(self.btn_remove)

        # Redraw when the model changes, whichever widget changed it. Only `description`
        # was subscribed, so a defect set anywhere else left a stale icon here (FIB-564).
        #
        # `defect` and `description` only: both are written by GUI widgets, on the GUI
        # thread, and nothing else refreshes this list when one changes.
        #
        # **Not** `task_state.name`/`.status`. FIB-565 added those so the row would
        # follow a running workflow, and they crashed it:
        #
        #     RuntimeError: wrapped C/C++ object of type QLabel has been deleted
        #       lamella_name_list_widget.py in refresh -> name_label.setText(...)
        #
        # They are written from the workflow's FunctionWorker thread, so `refresh` is
        # marshalled and arrives *later* -- by which point `set_lamella` may have
        # replaced this row and Qt may have destroyed its widgets. psygnal holds bound
        # methods weakly, which was the argument for not disconnecting, but that only
        # covers the *Python* object: Qt's teardown does not wait for the collector, so
        # a live weakref over a destroyed QLabel is exactly the reachable state.
        #
        # `AutoLamellaMainUI._on_workflow_update` drives this list instead, over a Qt
        # signal, and it names the lamella that changed -- so it costs one
        # `refresh_lamella` rather than a subscription per row, and the row follows a
        # running workflow either way.
        # type: ignore because @evented adds .events dynamically.
        self.lamella.events.description.connect(self.refresh)  # type: ignore[union-attr]
        self.lamella.events.defect.connect(self.refresh)  # type: ignore[union-attr]

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
        chosen = menu.exec_(self.btn_defect.mapToGlobal(self.btn_defect.rect().bottomLeft()))
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
            self, "Remove Lamella", f"Remove <b>{self.lamella.name}</b>?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_clicked.emit(self.lamella)

    @ensure_main_thread
    def refresh(self) -> None:
        self.name_label.setText(self.lamella.name)
        self.setToolTip(self.lamella.description or "")
        text, style = _lamella_status_text(self.lamella)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)
        # Update defect icon if visible
        if self.btn_defect.isVisible():
            icon_name, icon_color, tooltip = _lamella_defect_icon(self.lamella)
            self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
            self.btn_defect.setToolTip(tooltip)


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
        lbl_name.setStyleSheet("font-weight: bold; background: transparent;")
        lbl_name.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        header_layout.addWidget(lbl_name)
        lbl_status = QLabel("Status")
        lbl_status.setStyleSheet("font-weight: bold; background: transparent;")
        header_layout.addWidget(lbl_status, stretch=1)
        self.btn_add = IconToolButton(icon="mdi:plus", tooltip="Add", size=_LAMELLA_BTN_SIZE.width())
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
        self._btn_visible["defect"] = visible
        for row in self._rows():
            row.btn_defect.setVisible(visible)
            if visible:
                row.refresh()  # ensure icon is up to date

    def enable_actions_button(self, visible: bool) -> None:
        pass  # no-op: actions are now direct icon buttons; use enable_move_to_action / enable_edit_action / enable_update_action

    def enable_move_to_action(self, visible: bool) -> None:
        self._btn_visible["move_to"] = visible
        for row in self._rows():
            row.btn_move_to.setVisible(visible)

    def enable_edit_action(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for row in self._rows():
            row.btn_edit.setVisible(visible)

    def enable_update_action(self, visible: bool) -> None:
        self._btn_visible["update"] = visible
        for row in self._rows():
            row.btn_update.setVisible(visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for row in self._rows():
            row.btn_remove.setVisible(visible)

    def _rows(self):
        """Yield all _LamellaRow widgets."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, _LamellaRow):
                yield row

    def _apply_btn_visibility(self, row: _LamellaRow) -> None:
        row.btn_defect.setVisible(self._btn_visible["defect"])
        row.btn_move_to.setVisible(self._btn_visible["move_to"])
        row.btn_edit.setVisible(self._btn_visible["edit"])
        row.btn_update.setVisible(self._btn_visible["update"])
        row.btn_remove.setVisible(self._btn_visible["remove"])
        if self._btn_visible["defect"]:
            # Set icon directly — row.refresh() won't work here because
            # isVisible() returns False before the row is parented via setItemWidget
            icon_name, icon_color, tooltip = _lamella_defect_icon(row.lamella)
            row.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
            row.btn_defect.setToolTip(tooltip)

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
