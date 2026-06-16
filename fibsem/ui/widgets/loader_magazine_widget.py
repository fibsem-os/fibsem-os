"""Loader magazine widget — the autoloader's storage inventory.

Mirrors :class:`SampleHolderWidget`'s form + slot-list shape, but for the loader
*magazine* (``microscope._stage.loader``) rather than the holder working slots.
Each row edits its grid's name + description **inline** (magazine slots are
storage, not beam-accessible, so there is no stage position to show).

The status dot beside Load is both indicator and control:
  - gray  : empty — click to mark available
  - white : available (grid present) — click to clear; type a name to enable Load
  - green : loaded in the working slot (beam) — use Unload to retract
Typing a name also marks the slot available (no separate toggle needed).

Phase 4: in-memory only (no persistence).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleGridLoader
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.sample_holder_widget import (
    _ACTIONS_BTN_SIZE,
    _BTN_STYLE,
    _ROW_HEIGHT,
    _GridListHeader,
)

_STATUS_COLORS = {
    "gray": "#606060",   # empty
    "white": "#d6d6d6",  # available
    "green": "#4caf50",  # loaded in the working slot (beam)
}
_STATUS_TOOLTIPS = {
    "gray": "Empty — click to mark available",
    "white": "Available — click to clear",
    "green": "Loaded on Microscope",
}


def _dot_style(color: str) -> str:
    return (
        f"QToolButton {{ background: transparent; border: none; font-size: 16px; color: {color}; }}"
        " QToolButton:hover { background: rgba(255,255,255,25); border-radius: 4px; }"
    )


class _MagazineSlotRowWidget(QWidget):
    """A magazine slot row: number | name | description | status dot | load.

    The status dot toggles availability (empty <-> available) on click. Typing a
    name also marks the slot available. Naming gates the load action.
    """

    presence_toggled = pyqtSignal(object, bool)  # GridSlot, available
    grid_changed = pyqtSignal(object)            # GridSlot (name/desc edited)
    load_clicked = pyqtSignal(object)            # GridSlot

    def __init__(self, slot: GridSlot,
                 beam_check: Optional[Callable[[SampleGrid], bool]] = None,
                 name_in_use: Optional[Callable[[GridSlot, str], bool]] = None,
                 parent=None):
        super().__init__(parent)
        self.slot = slot
        self._beam_check = beam_check
        self._name_in_use = name_in_use
        self._updating = False
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        self.slot_label = QLabel()
        self.slot_label.setFixedWidth(28)
        self.slot_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self.slot_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.editingFinished.connect(self._on_name_edited)
        layout.addWidget(self.name_edit, 2)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Description")
        self.desc_edit.editingFinished.connect(self._on_desc_edited)
        layout.addWidget(self.desc_edit, 3)

        self.status_dot = QToolButton()
        self.status_dot.setText("●")  # ●
        self.status_dot.setFixedSize(22, 22)
        self.status_dot.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_dot.clicked.connect(self._on_dot_clicked)
        layout.addWidget(self.status_dot)

        self.btn_load = QToolButton()
        self.btn_load.setFixedSize(_ACTIONS_BTN_SIZE, _ACTIONS_BTN_SIZE)
        self.btn_load.setStyleSheet(_BTN_STYLE)
        self.btn_load.setIcon(QIconifyIcon("mdi:login-variant", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_load.setToolTip("Load this grid into the beam")
        self.btn_load.clicked.connect(lambda: self.load_clicked.emit(self.slot))
        layout.addWidget(self.btn_load)

        self.refresh()

    # --- model mutations (row is self-contained) ---

    def _default_name(self) -> str:
        """A slot-based default grid name (e.g. 'Grid-01'), kept unique."""
        base = f"Grid-{self.slot.index + 1:02d}"
        if self._name_in_use is None or not self._name_in_use(self.slot, base):
            return base
        i = 1  # rare: base already taken by another slot's manual name
        while self._name_in_use(self.slot, f"{base}-{i}"):
            i += 1
        return f"{base}-{i}"

    def _ensure_grid(self) -> None:
        if self.slot.loaded_grid is None:
            self.slot.loaded_grid = SampleGrid(name=self._default_name())

    def _on_dot_clicked(self) -> None:
        # toggle availability; green (in beam) is controlled by Load/Unload, not here
        if self.status() == "green":
            return
        available = self.slot.loaded_grid is None
        if available:
            self._ensure_grid()
        else:
            self.slot.loaded_grid = None
        self.refresh()
        self.presence_toggled.emit(self.slot, available)

    def _on_name_edited(self) -> None:
        if self._updating:
            return
        text = self.name_edit.text()
        if text.strip() and self._name_in_use is not None and self._name_in_use(self.slot, text):
            # duplicate name → reject: revert the field and flag it
            self.refresh()
            self.name_edit.setStyleSheet("border: 1px solid #c0392b;")
            self.name_edit.setToolTip("Name already in use")
            return
        if text.strip():
            self._ensure_grid()
        if self.slot.loaded_grid is not None:
            self.slot.loaded_grid.name = text
        self.refresh()
        self.grid_changed.emit(self.slot)

    def _on_desc_edited(self) -> None:
        if self._updating:
            return
        text = self.desc_edit.text()
        if text.strip():
            self._ensure_grid()
        if self.slot.loaded_grid is not None:
            self.slot.loaded_grid.description = text
            self.refresh()
            self.grid_changed.emit(self.slot)

    # --- status ---

    def status(self) -> str:
        """Beam state: 'gray' (empty), 'white' (available), 'green' (loaded)."""
        grid = self.slot.loaded_grid
        if grid is None:
            return "gray"
        if self._beam_check is not None and self._beam_check(grid):
            return "green"
        return "white"

    def refresh(self) -> None:
        loaded = self.slot.loaded_grid is not None
        self._updating = True
        self.slot_label.setText(f"{self.slot.index + 1:02d}")  # just the slot number
        self.name_edit.setText(self.slot.loaded_grid.name if loaded else "")
        self.name_edit.setStyleSheet("")  # clear any duplicate-name flag
        self.name_edit.setToolTip("")
        self.desc_edit.setText(self.slot.loaded_grid.description if loaded else "")
        self._updating = False

        status = self.status()
        self.status_dot.setStyleSheet(_dot_style(_STATUS_COLORS[status]))
        self.status_dot.setToolTip(_STATUS_TOOLTIPS[status])
        # available + named + not already loaded → can load
        named = loaded and bool(self.slot.loaded_grid.name.strip())
        self.btn_load.setEnabled(status == "white" and named)
        if status == "green":
            self.btn_load.setToolTip("Already loaded on the microscope")
        elif status == "gray":
            self.btn_load.setToolTip("Empty slot")
        elif not named:
            self.btn_load.setToolTip("Name the grid before loading")
        else:
            self.btn_load.setToolTip("Load this grid onto the microscope")


class LoaderMagazineWidget(QWidget):
    """Magazine inventory for the autoloader (``microscope._stage.loader``)."""

    magazine_changed = pyqtSignal()
    presence_toggled = pyqtSignal(str, bool)   # slot_name, available
    load_requested = pyqtSignal(str)           # grid_name
    unload_requested = pyqtSignal()            # retract the working-slot grid

    def __init__(self, microscope=None, parent=None):
        super().__init__(parent)
        self._microscope = microscope
        self._loader: Optional[SampleGridLoader] = None
        self._setup_ui()
        self.setEnabled(False)

    # --- setup ---

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(4, 4, 4, 0)
        form.setSpacing(4)
        self.name_label = QLabel("Autoloader Magazine")
        self.capacity_label = QLabel("—")
        form.addRow("Loader", self.name_label)
        form.addRow("Capacity", self.capacity_label)

        self._header = _GridListHeader("Magazine")
        self.btn_unload = QToolButton()
        self.btn_unload.setText("Unload")
        self.btn_unload.setToolTip("Unload the grid currently in the working slot")
        self.btn_unload.clicked.connect(self.unload_requested)
        self._header.layout().addWidget(self.btn_unload)

        self.btn_inventory = QToolButton()
        self.btn_inventory.setText("Run Inventory")
        self.btn_inventory.setToolTip("Scan the magazine for loaded grids")
        self.btn_inventory.clicked.connect(self._on_run_inventory)
        self._header.layout().addWidget(self.btn_inventory)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setMinimumHeight(3 * _ROW_HEIGHT)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._empty_label = QLabel("No loader present.")
        self._empty_label.setStyleSheet("color: #606060; font-style: italic; padding: 8px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 4)
        inner_layout.setSpacing(4)
        inner_layout.addWidget(form_widget)
        inner_layout.addWidget(self._header)
        inner_layout.addWidget(self._list)
        inner_layout.addWidget(self._empty_label)

        self._panel = TitledPanel("Loader Magazine", content=inner, collapsible=True)
        layout.addWidget(self._panel)

    # --- public API ---

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        loader = getattr(getattr(microscope, "_stage", None), "loader", None)
        self.set_loader(loader)

    def set_loader(self, loader: Optional[SampleGridLoader]) -> None:
        self._loader = loader
        self.setEnabled(loader is not None)
        self._empty_label.setVisible(loader is None)
        self.name_label.setText(getattr(loader, "name", "Autoloader Magazine") if loader else "—")
        self.capacity_label.setText(str(loader.capacity) if loader is not None else "—")
        self._populate()
        self._update_unload_enabled()

    def refresh_rows(self) -> None:
        """Refresh every row (e.g. after a load/unload changes beam state)."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if row is not None:
                row.refresh()
        self._update_unload_enabled()

    def _update_unload_enabled(self) -> None:
        """Unload is only meaningful when a grid sits in the working slot."""
        self.btn_unload.setEnabled(bool(self._loader and self._loader.loaded_slots))

    # --- beam state ---

    def _beam_loaded(self, grid: SampleGrid) -> bool:
        """True if this exact grid object currently sits in a holder working slot."""
        holder = getattr(getattr(self._microscope, "_stage", None), "holder", None)
        if holder is None or grid is None:
            return False
        return any(s.loaded_grid is grid for s in holder.slots.values())

    def _name_in_use(self, slot: GridSlot, name: str) -> bool:
        """True if another magazine slot already holds a grid with this name.

        Names are the grid identity (used for load + GridRecord lookup), so they
        must be unique within the magazine.
        """
        if self._loader is None:
            return False
        return any(
            s is not slot and s.loaded_grid is not None and s.loaded_grid.name == name
            for s in self._loader.slots.values()
        )

    # --- population ---

    def _populate(self) -> None:
        self._list.clear()
        if self._loader is None:
            return
        for slot in self._loader.slots.values():
            row = _MagazineSlotRowWidget(
                slot, beam_check=self._beam_loaded, name_in_use=self._name_in_use)
            row.presence_toggled.connect(self._on_presence_toggled)
            row.grid_changed.connect(lambda _slot: self.magazine_changed.emit())
            row.load_clicked.connect(self._on_load_clicked)
            item = QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

    # --- handlers ---

    def _on_run_inventory(self) -> None:
        if self._loader is None:
            return
        loaded = self._loader.run_inventory()
        logging.info(f"Magazine inventory: {len(loaded)} slot(s) loaded.")
        self.refresh_rows()

    def _on_presence_toggled(self, slot: GridSlot, available: bool) -> None:
        self.presence_toggled.emit(slot.name, available)
        self.magazine_changed.emit()

    def _on_load_clicked(self, slot: GridSlot) -> None:
        if slot.loaded_grid is not None and slot.loaded_grid.name.strip():
            self.load_requested.emit(slot.loaded_grid.name)
