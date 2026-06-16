"""Loader magazine widget — the autoloader's storage inventory.

Mirrors :class:`SampleHolderWidget` (form + list of slot rows + edit panel) but
for the loader *magazine* (``microscope._stage.loader``) rather than the holder
working slots. Magazine slots are storage, not beam-accessible, so the row
actions differ: a presence toggle (loaded/empty), a Load → beam button (emits
``load_requested`` for the host to run an exchange), and Clear. Naming a grid is
a second step, enabled only for loaded slots.

Phase 4: in-memory only (no persistence).
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
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
    _EMPTY_STYLE,
    _LOADED_STYLE,
    _ROW_HEIGHT,
    _SLOT_LABEL_WIDTH,
    _GridListHeader,
    _GridSlotEditPanel,
)


class _MagazineSlotRowWidget(QWidget):
    """A single magazine slot row: label | presence toggle | grid | load | clear."""

    presence_toggled = pyqtSignal(object, bool)  # GridSlot, loaded
    load_clicked = pyqtSignal(object)            # GridSlot
    clear_clicked = pyqtSignal(object)           # GridSlot
    row_clicked = pyqtSignal(object)             # GridSlot

    def __init__(self, slot: GridSlot, parent=None):
        super().__init__(parent)
        self.slot = slot
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        self.slot_label = QLabel()
        self.slot_label.setFixedWidth(_SLOT_LABEL_WIDTH)
        self.slot_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self.slot_label)

        # presence toggle — loaded / empty (independent of naming)
        self.presence_check = QCheckBox()
        self.presence_check.setToolTip("Slot loaded / empty")
        self.presence_check.toggled.connect(
            lambda checked: self.presence_toggled.emit(self.slot, checked)
        )
        layout.addWidget(self.presence_check)

        self.grid_label = QLabel()
        layout.addWidget(self.grid_label, 1)

        self.btn_load = QToolButton()
        self.btn_load.setFixedSize(_ACTIONS_BTN_SIZE, _ACTIONS_BTN_SIZE)
        self.btn_load.setStyleSheet(_BTN_STYLE)
        self.btn_load.setIcon(QIconifyIcon("mdi:tray-arrow-up", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_load.setToolTip("Load this grid into the beam")
        self.btn_load.clicked.connect(lambda: self.load_clicked.emit(self.slot))
        self.btn_load.installEventFilter(self)
        layout.addWidget(self.btn_load)

        self.btn_clear = QToolButton()
        self.btn_clear.setFixedSize(_ACTIONS_BTN_SIZE, _ACTIONS_BTN_SIZE)
        self.btn_clear.setStyleSheet(_BTN_STYLE)
        self.btn_clear.setIcon(
            QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_clear.setToolTip("Remove grid from this magazine slot")
        self.btn_clear.clicked.connect(lambda: self.clear_clicked.emit(self.slot))
        self.btn_clear.installEventFilter(self)
        layout.addWidget(self.btn_clear)

        self.refresh()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FocusIn:
            self.row_clicked.emit(self.slot)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:
        child = self.childAt(event.pos())
        if child is None or child is self:
            self.row_clicked.emit(self.slot)
        super().mousePressEvent(event)

    def refresh(self) -> None:
        loaded = self.slot.loaded_grid is not None
        self.slot_label.setText(self.slot.name)

        # presence checkbox reflects loaded state without re-emitting
        self.presence_check.blockSignals(True)
        self.presence_check.setChecked(loaded)
        self.presence_check.blockSignals(False)

        if loaded:
            self.grid_label.setText(self.slot.loaded_grid.name or "(unnamed)")
            self.grid_label.setStyleSheet(_LOADED_STYLE)
        else:
            self.grid_label.setText("Empty")
            self.grid_label.setStyleSheet(_EMPTY_STYLE)

        # load / clear only meaningful when a grid is present
        self.btn_load.setEnabled(loaded)
        self.btn_clear.setEnabled(loaded)


class LoaderMagazineWidget(QWidget):
    """Magazine inventory for the autoloader (``microscope._stage.loader``)."""

    magazine_changed = pyqtSignal()
    presence_toggled = pyqtSignal(str, bool)   # slot_name, loaded
    grid_selected = pyqtSignal(str, object)    # slot_name, GridSlot
    load_requested = pyqtSignal(str)           # grid_name

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

        self._edit_panel = _GridSlotEditPanel()
        self._edit_panel.slot_changed.connect(self._on_grid_edited)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 4)
        inner_layout.setSpacing(4)
        inner_layout.addWidget(form_widget)
        inner_layout.addWidget(self._header)
        inner_layout.addWidget(self._list)
        inner_layout.addWidget(self._empty_label)
        inner_layout.addWidget(self._edit_panel)

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
        self.capacity_label.setText(str(loader.capacity) if loader is not None else "—")
        self._populate()

    # --- population ---

    def _populate(self) -> None:
        self._list.clear()
        self._edit_panel.set_slot(None)
        if self._loader is None:
            return
        for slot in self._loader.slots.values():
            row = _MagazineSlotRowWidget(slot)
            row.presence_toggled.connect(self._on_presence_toggled)
            row.load_clicked.connect(self._on_load_clicked)
            row.clear_clicked.connect(self._on_clear_clicked)
            row.row_clicked.connect(self._on_row_clicked)
            item = QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

    def _row_for(self, slot: GridSlot) -> Optional[_MagazineSlotRowWidget]:
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if row is not None and row.slot is slot:
                return row
        return None

    # --- handlers ---

    def _on_run_inventory(self) -> None:
        if self._loader is None:
            return
        loaded = self._loader.run_inventory()
        logging.info(f"Magazine inventory: {len(loaded)} slot(s) loaded.")
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if row is not None:
                row.refresh()

    def _on_presence_toggled(self, slot: GridSlot, loaded: bool) -> None:
        if loaded and slot.loaded_grid is None:
            slot.loaded_grid = SampleGrid(name=slot.name)  # unnamed default; rename in edit panel
        elif not loaded:
            slot.loaded_grid = None
        row = self._row_for(slot)
        if row is not None:
            row.refresh()
        # naming is enabled only when loaded
        self._edit_panel.set_slot(slot if slot.loaded_grid is not None else None)
        self.presence_toggled.emit(slot.name, loaded)
        self.magazine_changed.emit()

    def _on_load_clicked(self, slot: GridSlot) -> None:
        if slot.loaded_grid is not None:
            self.load_requested.emit(slot.loaded_grid.name)

    def _on_clear_clicked(self, slot: GridSlot) -> None:
        slot.loaded_grid = None
        row = self._row_for(slot)
        if row is not None:
            row.refresh()
        self._edit_panel.set_slot(None)
        self.magazine_changed.emit()

    def _on_row_clicked(self, slot: GridSlot) -> None:
        # naming panel only for loaded slots
        self._edit_panel.set_slot(slot if slot.loaded_grid is not None else None)
        self.grid_selected.emit(slot.name, slot)

    def _on_grid_edited(self, slot: GridSlot) -> None:
        row = self._row_for(slot)
        if row is not None:
            row.refresh()
        self.magazine_changed.emit()
