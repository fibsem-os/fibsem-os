"""The overviews currently on the canvas, one row each.

The canvas can hold several overviews at once, and until this existed nothing said
which ones — you could see them but not name, hide or drop one (FIB-543).

Shaped after `LamellaRowWidget`: a name, its details, and icon buttons that appear
when the row is under the cursor or selected. The difference is where the reveal
lives — the lamella list shows and hides its buttons for every row at once through
`_btn_visible`, because there the choice is which *columns* the table has. Here it is
per row and follows the pointer, so a resting list stays quiet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

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

from fibsem.ui import stylesheets
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import IconToolButton

if TYPE_CHECKING:
    from fibsem.ui.fm.widgets.fm_overview_widget import PlacedOverviewImageRecord

# Imported only for typing: the record lives on `FMOverviewWidget`, which builds this
# widget, so a runtime import would close a cycle. Nothing here needs more than the
# four fields it reads.

_BTN_SIZE = 26
_ROW_HEIGHT = 30


class OverviewRowWidget(QWidget):
    """One overview: its name, and what to do with it."""

    visibility_toggled = pyqtSignal(str, bool)
    remove_clicked = pyqtSignal(str)

    def __init__(
        self, record: "PlacedOverviewImageRecord", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.record_id = record.id
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(6)

        self.name_label = QLabel(record.label)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label, 1)

        # The detail and the buttons occupy the same end of the row rather than sitting
        # side by side: a settings column is narrow, and a name long enough to matter is
        # exactly when both would not fit.
        self.detail_label = QLabel(record.detail)
        self.detail_label.setStyleSheet(
            f"background: transparent; color: {TEXT_MUTED_COLOR}; font-size: 11px;"
        )
        layout.addWidget(self.detail_label)

        # Checked means *hidden*: the icon that shows is the state you would move to,
        # which is how the eye reads in every layer panel.
        self.btn_visible = IconToolButton(
            icon="mdi:eye-outline",
            checked_icon="mdi:eye-off-outline",
            tooltip="Hide",
            checked_tooltip="Show",
            checkable=True,
            checked=not record.visible,
            size=_BTN_SIZE,
        )
        layout.addWidget(self.btn_visible)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE
        )
        layout.addWidget(self.btn_remove)

        self.btn_visible.toggled.connect(
            lambda hidden: self.visibility_toggled.emit(self.record_id, not hidden)
        )
        self.btn_remove.clicked.connect(
            lambda: self.remove_clicked.emit(self.record_id)
        )

        self._selected = False
        # Through `refresh` rather than setting the resting state by hand, so a row
        # built from an already-hidden record looks the same as one that was hidden
        # after the fact -- which it did not, when the two paths were written separately.
        self.refresh(record)

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
        # A hidden overview keeps its eye visible at rest: the row has to say *why* it
        # looks different, and an unexplained greyed-out name is worse than an icon.
        self.btn_visible.setVisible(shown or self.btn_visible.isChecked())
        self.btn_remove.setVisible(shown)
        self.detail_label.setVisible(not shown)

    def refresh(self, record: "PlacedOverviewImageRecord") -> None:
        self.name_label.setText(record.label)
        self.detail_label.setText(record.detail)
        # Only when it differs. `IconToolButton` drives its own icon off `toggled`, so
        # blocking signals to set it silently would leave the icon showing the old
        # state -- and setting it unconditionally would re-emit on every refresh.
        if self.btn_visible.isChecked() is record.visible:
            self.btn_visible.setChecked(not record.visible)
        self.name_label.setEnabled(record.visible)
        self._set_actions_shown(self._selected or self.underMouse())


class OverviewListWidget(QWidget):
    """Every overview on the canvas, oldest first."""

    visibility_toggled = pyqtSignal(str, bool)
    remove_requested = pyqtSignal(str)
    selection_changed = pyqtSignal(object)  # record id, or None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.empty_label = QLabel("No overviews yet.")
        self.empty_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self.empty_label)

        self._list = QListWidget()
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setFrameShape(QFrame.NoFrame)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        self._rows: dict = {}
        self.set_records([])

    # ── contents ──────────────────────────────────────────────────────────

    def set_records(self, records: List["PlacedOverviewImageRecord"]) -> None:
        """Rebuild the list from *records*, keeping the selection where it can be kept.

        Rebuilt rather than diffed: the list is a handful of rows and is refreshed only
        when one is acquired or dropped, so the bookkeeping a diff would need costs more
        than it saves. The selection is restored by id, because the row that held it is
        gone by then.
        """
        previously_selected = self.selected_id()
        self._list.clear()
        self._rows = {}

        for record in records:
            row = OverviewRowWidget(record)
            row.visibility_toggled.connect(self.visibility_toggled)
            row.remove_clicked.connect(self.remove_requested)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            item.setData(Qt.UserRole, record.id)
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            self._rows[record.id] = row

        self.empty_label.setVisible(not records)
        self._list.setVisible(bool(records))
        # Sized to its contents: the settings column is a stack, and a list that keeps
        # its full height empty pushes everything below it off the bottom.
        self._list.setFixedHeight(max(len(records), 1) * _ROW_HEIGHT + 4)

        if previously_selected in self._rows:
            self.select(previously_selected)

    def selected_id(self) -> Optional[str]:
        items = self._list.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def select(self, record_id: Optional[str]) -> None:
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.data(Qt.UserRole) == record_id:
                self._list.setCurrentItem(item)
                return
        self._list.clearSelection()

    def refresh(self, records: List["PlacedOverviewImageRecord"]) -> None:
        """Update the rows in place, for a change that does not add or remove one."""
        for record in records:
            row = self._rows.get(record.id)
            if row is not None:
                row.refresh(record)

    def _on_selection_changed(self) -> None:
        selected = self.selected_id()
        for record_id, row in self._rows.items():
            row.set_selected(record_id == selected)
        self.selection_changed.emit(selected)
