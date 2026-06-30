"""Grid list widget — the experiment's grid workflow records.

Shows ``Experiment.grids`` (``GridRecord``) as name + status rows. This is the
*experiment* side of the grid workflow (workflow units, persisted), distinct
from the hardware-staging LoaderMagazineWidget. The one bridge from hardware to
experiment is the "Add Grids from Loader" action, which imports loaded grids via
``Experiment.sync_grids_from_holder``.

Phase 4a: list + add/remove only. Protocol / Run / Results arrive later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.ui import stylesheets

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import GridRecord

_NAME_COL_WIDTH = 150   # name column — fixed so name/status align across rows
_STATUS_COL_WIDTH = 130  # status column


def _status_text(record: "GridRecord") -> str:
    """Short status summary for a grid record."""
    if record.is_failure:
        return "Failed"
    n = len(record.completed_tasks)
    if n:
        return f"{n} task{'s' if n != 1 else ''} complete"
    return "Not started"


class _GridRowWidget(QWidget):
    """A grid row: name | status | remove."""

    remove_clicked = pyqtSignal(object)  # GridRecord

    def __init__(self, record: "GridRecord", slot_label: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(8)

        # loader magazine slot number for this grid (blank if not in the magazine)
        self.slot_label = QLabel(slot_label or "—")
        self.slot_label.setFixedWidth(28)
        self.slot_label.setStyleSheet("background: transparent; color: #808080;")
        self.slot_label.setToolTip("Loader magazine slot")
        layout.addWidget(self.slot_label)

        self.name_label = QLabel(record.name)
        self.name_label.setFixedWidth(_NAME_COL_WIDTH)
        self.name_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel(_status_text(record))
        self.status_label.setFixedWidth(_STATUS_COL_WIDTH)
        self.status_label.setStyleSheet("background: transparent; color: #a0a0a0;")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        self.btn_remove = QToolButton()
        self.btn_remove.setIcon(
            QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_remove.setToolTip("Remove this grid from the experiment")
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.record))
        layout.addWidget(self.btn_remove)


class GridListWidget(QWidget):
    """Single-selection list of the experiment's grid records (name + status)."""

    grid_selected = pyqtSignal(object)            # GridRecord (or None)
    add_from_loader_requested = pyqtSignal()      # import loaded grids
    remove_requested = pyqtSignal(object)         # GridRecord

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # header bar: "Experiment Grids · N"  +  add
        header = QWidget()
        header.setStyleSheet("background: #1e2124;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 3, 4, 3)
        hl.setSpacing(4)

        self._title = QLabel("Experiment Grids")
        self._title.setStyleSheet("font-weight: bold; background: transparent;")
        self._count = QLabel("· 0")
        self._count.setStyleSheet("background: transparent; color: #a0a0a0;")
        hl.addWidget(self._title)
        hl.addWidget(self._count)
        hl.addStretch(1)

        self.btn_add = QToolButton()
        self.btn_add.setText("Add from Loader")
        self.btn_add.setIcon(
            QIconifyIcon("mdi:tray-arrow-down", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_add.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_add.setToolTip("Import grids loaded in the magazine / working slot")
        self.btn_add.clicked.connect(self.add_from_loader_requested)
        hl.addWidget(self.btn_add)

        outer.addWidget(header)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        outer.addWidget(self._list)

        self._empty = QLabel("No grids yet — add them from the loader.")
        self._empty.setStyleSheet("color: #606060; font-style: italic; padding: 8px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._empty)
        self._update_empty()

    # --- public API ---

    def set_grids(self, grids: List["GridRecord"],
                  slot_labels: Optional[dict] = None) -> None:
        """Repopulate the list, preserving the current selection by name.

        ``slot_labels`` optionally maps grid name → loader magazine slot label
        (e.g. ``"01"``) for display; the widget itself stays hardware-agnostic.
        """
        slot_labels = slot_labels or {}
        prev = self.selected_grid
        prev_name = prev.name if prev is not None else None

        self._list.blockSignals(True)
        self._list.clear()
        for record in grids:
            item = QListWidgetItem(self._list)
            item.setData(Qt.UserRole, record)
            row = _GridRowWidget(record, slot_label=slot_labels.get(record.name, ""))
            row.remove_clicked.connect(self.remove_requested)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._list.blockSignals(False)

        self._count.setText(f"· {len(grids)}")
        self._update_empty()

        # restore selection by name, else first row, else none
        if self._list.count():
            target = 0
            if prev_name is not None:
                for i in range(self._list.count()):
                    rec = self._list.item(i).data(Qt.UserRole)
                    if rec.name == prev_name:
                        target = i
                        break
            self._list.setCurrentRow(target)
        else:
            self._on_selection_changed()  # emit None

    @property
    def selected_grid(self) -> Optional["GridRecord"]:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    # --- handlers ---

    def _on_selection_changed(self) -> None:
        self.grid_selected.emit(self.selected_grid)

    def _update_empty(self) -> None:
        empty = self._list.count() == 0
        self._empty.setVisible(empty)
        self._list.setVisible(not empty)
