"""The overview areas planned for the next run, one row each.

An overview run used to be one region: whatever the grid was drawn around. A session
usually wants several — a few grid squares, planned up front and left to run (FIB-532).
This is the list of them, and the selection in it is the area the canvas draws in
colour and the settings panel edits.

Shaped after `OverviewListWidget` above it in the same column, deliberately: two lists
in one narrow column that behaved differently would be worse than either. The
differences are what the two lists are *for*. This one is ordered — a batch runs top to
bottom — its rows carry the state of a run in progress, and its identity is positional,
because an area can be renamed and there is nothing else stable to key on.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.structures import OverviewArea
from fibsem.ui import stylesheets
from fibsem.ui.tokens import ACCENT_COLOR, ERROR_COLOR, OK_COLOR, TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import IconToolButton

_BTN_SIZE = 26
_ROW_HEIGHT = 30

# What a row says about an area while a batch is running. Absent means queued, which is
# every area's state when nothing is running -- so a resting list shows no state at all
# rather than three rows all saying the same word.
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

_STATE_COLORS = {
    STATE_RUNNING: ACCENT_COLOR,
    STATE_DONE: OK_COLOR,
    STATE_FAILED: ERROR_COLOR,
}


def describe_area(area: OverviewArea) -> str:
    """The grid, in the few characters a narrow column has for it.

    The enabled count is here rather than only on the canvas because the canvas is
    often zoomed somewhere else, and an area quietly masked down to four tiles is
    exactly the thing you want to notice before starting an unattended run.
    """
    total = area.rows * area.cols
    enabled = area.n_enabled_tiles
    grid = f"{area.rows}×{area.cols}"
    return grid if enabled == total else f"{grid} · {enabled}/{total} tiles"


class OverviewAreaRowWidget(QWidget):
    """One area: its name, its grid, and what to do with it."""

    rename_clicked = pyqtSignal(int)
    remove_clicked = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        area: OverviewArea,
        state: Optional[str] = None,
        removable: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(6)

        self.name_label = QLabel(area.name)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label, 1)

        self.state_label = QLabel()
        self.state_label.setStyleSheet("background: transparent; font-size: 11px;")
        layout.addWidget(self.state_label)

        # The grid and the buttons share the end of the row rather than sitting side by
        # side, as they do in the overviews list: a settings column is narrow, and a
        # name long enough to matter is exactly when both would not fit.
        self.detail_label = QLabel(describe_area(area))
        self.detail_label.setStyleSheet(
            f"background: transparent; color: {TEXT_MUTED_COLOR}; font-size: 11px;"
        )
        layout.addWidget(self.detail_label)

        self.btn_rename = IconToolButton(
            icon="mdi:pencil-outline", tooltip="Rename", size=_BTN_SIZE
        )
        layout.addWidget(self.btn_rename)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE
        )
        # A run needs somewhere to happen, so the last area cannot be dropped. Shown
        # disabled rather than hidden: a button that vanishes when a list gets short
        # reads as a bug, and its tooltip is where the rule gets explained.
        self.btn_remove.setEnabled(removable)
        if not removable:
            self.btn_remove.setToolTip("A run needs at least one area")
        layout.addWidget(self.btn_remove)

        self.btn_rename.clicked.connect(lambda: self.rename_clicked.emit(self.index))
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.index))

        self._selected = False
        self.set_state(state)

    # ── hover and selection ───────────────────────────────────────────────

    def set_selected(self, selected: bool) -> None:
        """Selected rows keep their buttons whether or not the pointer is on them."""
        self._selected = selected
        self._set_actions_shown(selected or self.underMouse())

    def enterEvent(self, event) -> None:
        self._set_actions_shown(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_actions_shown(self._selected)
        super().leaveEvent(event)

    def _set_actions_shown(self, shown: bool) -> None:
        self.btn_rename.setVisible(shown)
        self.btn_remove.setVisible(shown)
        self.detail_label.setVisible(not shown)

    def set_state(self, state: Optional[str]) -> None:
        self.state_label.setText(state or "")
        self.state_label.setVisible(bool(state))
        color = _STATE_COLORS.get(state or "", TEXT_MUTED_COLOR)
        self.state_label.setStyleSheet(
            f"background: transparent; font-size: 11px; color: {color};"
        )
        self._set_actions_shown(self._selected or self.underMouse())


class OverviewAreaListWidget(QWidget):
    """Every planned area, in the order a batch will acquire them."""

    selection_changed = pyqtSignal(int)
    remove_requested = pyqtSignal(int)
    rename_requested = pyqtSignal(int)
    reordered = pyqtSignal(int, int)  # from index, to index

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setFrameShape(QFrame.NoFrame)
        # A batch runs top to bottom, so the order is part of the plan and worth being
        # able to change. Internal moves only -- there is nowhere else to drop a row.
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        self._rows: Dict[int, OverviewAreaRowWidget] = {}
        # Set while rebuilding, so the selection changes Qt emits on the way through do
        # not read as the user picking a different area.
        self._rebuilding = False
        self.set_areas([], 0)

    # ── contents ──────────────────────────────────────────────────────────

    def set_areas(
        self,
        areas: List[OverviewArea],
        selected: int = 0,
        states: Optional[Dict[int, str]] = None,
    ) -> None:
        """Rebuild the list from *areas*, selecting the one at *selected*.

        Rebuilt rather than diffed, as the overviews list is: a handful of rows, changed
        only when one is added, dropped or reordered. The caller passes the selection in
        rather than it being restored here, because the caller is what moved it -- an
        area removed takes the selection with it, and where it lands is a decision this
        widget has no business making.
        """
        states = states or {}
        self._rebuilding = True
        try:
            self._list.clear()
            self._rows = {}

            for index, area in enumerate(areas):
                row = OverviewAreaRowWidget(
                    index=index,
                    area=area,
                    state=states.get(index),
                    removable=len(areas) > 1,
                )
                row.rename_clicked.connect(self.rename_requested)
                row.remove_clicked.connect(self.remove_requested)
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, _ROW_HEIGHT))
                self._list.addItem(item)
                self._list.setItemWidget(item, row)
                self._rows[index] = row

            # Sized to its contents: the settings column is a stack, and a list keeping
            # its full height would push everything below it off the bottom.
            self._list.setFixedHeight(max(len(areas), 1) * _ROW_HEIGHT + 4)
            self.select(selected)
        finally:
            self._rebuilding = False
        # Once, after the rebuild, so a caller that redraws the canvas from this gets
        # one signal for one rebuild rather than one per row Qt touched on the way.
        self._apply_selection_styling()

    def set_states(self, states: Dict[int, str]) -> None:
        """Update what the rows say about a run in progress, without rebuilding."""
        for index, row in self._rows.items():
            row.set_state(states.get(index))

    def selected_index(self) -> int:
        """Which area is selected, or -1 when there are none."""
        return self._list.currentRow()

    def select(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)
        else:
            self._list.clearSelection()

    def set_reorder_enabled(self, enabled: bool) -> None:
        """Reordering is off while a batch runs: the queue being walked is fixed."""
        self._list.setDragDropMode(
            QAbstractItemView.InternalMove if enabled else QAbstractItemView.NoDragDrop
        )

    # ── reactions ─────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        self._apply_selection_styling()
        if not self._rebuilding:
            self.selection_changed.emit(self.selected_index())

    def _apply_selection_styling(self) -> None:
        selected = self.selected_index()
        for index, row in self._rows.items():
            row.set_selected(index == selected)

    def _on_rows_moved(self, parent, start, end, destination, row) -> None:
        """A row was dragged. Report the move; the owner reorders and rebuilds.

        Qt's `row` is where the item was inserted *before* the source was taken out, so
        a downward move is reported one past where it lands. Corrected here rather than
        by every listener.
        """
        if self._rebuilding:
            return
        destination_index = row - 1 if row > start else row
        self.reordered.emit(start, destination_index)
