from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import List, Optional

import yaml
from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem.structures import FibsemStagePosition
from fibsem.ui import stylesheets
from fibsem.ui.utils import message_box_ui
from fibsem.ui.widgets.custom_widgets import IconToolButton

_NAME_MIN_WIDTH = 160
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_COUNT = 3  # move, update, remove (row buttons)
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * _BTN_COUNT + 8 * (_BTN_COUNT - 1)  # 112px

_NAME_EDIT_STYLE = (
    "QLineEdit { background: transparent; border: none; }"
    "QLineEdit:focus { background: #1e2124; border: 1px solid #555; border-radius: 2px; }"
)


class SavedPositionRowWidget(QWidget):
    name_changed = pyqtSignal(object, str)  # (FibsemStagePosition, name)
    move_to_clicked = pyqtSignal(object)    # FibsemStagePosition
    update_clicked = pyqtSignal(object)     # FibsemStagePosition
    remove_clicked = pyqtSignal(object)     # FibsemStagePosition
    row_clicked = pyqtSignal(object)        # FibsemStagePosition

    def __init__(self, position: FibsemStagePosition, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.position = position
        self._selected = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_edit = QLineEdit(self.position.name or "")
        self.name_edit.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_edit.setStyleSheet(_NAME_EDIT_STYLE)
        self.name_edit.installEventFilter(self)
        layout.addWidget(self.name_edit)

        self.position_label = QLabel(self.position.pretty_string)
        self.position_label.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        layout.addWidget(self.position_label, 1)

        self.btn_move = IconToolButton(
            icon="mdi:crosshairs-gps",
            tooltip="Move to position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_move)

        self.btn_update = IconToolButton(
            icon="mdi:map-marker-check",
            tooltip="Update to current position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_update)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can",
            tooltip="Remove position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_remove)

        self.name_edit.editingFinished.connect(
            lambda: self.name_changed.emit(self.position, self.name_edit.text())
        )
        self.btn_move.clicked.connect(lambda: self.move_to_clicked.emit(self.position))
        self.btn_update.clicked.connect(lambda: self.update_clicked.emit(self.position))
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.position))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.name_edit and event.type() == QEvent.Type.FocusIn:
            self.row_clicked.emit(self.position)
        return super().eventFilter(obj, event)

    def refresh(self, position: FibsemStagePosition) -> None:
        self.position = position
        self.name_edit.setText(position.name or "")
        self.position_label.setText(position.pretty_string)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        bg = "#2d3f5c" if selected else "transparent"
        self.setStyleSheet(f"background: {bg};")


class _SavedPositionListHeader(QWidget):
    add_clicked = pyqtSignal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        lbl_name = QLabel("Name")
        lbl_name.setMinimumWidth(_NAME_MIN_WIDTH)
        lbl_name.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(lbl_name)

        lbl_position = QLabel("Position")
        lbl_position.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(lbl_position, 1)

        # Spacer aligns btn_add with the btn_remove column (rightmost row button)
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width())
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        self.btn_add = IconToolButton(
            icon="mdi:plus",
            tooltip="Add current stage position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_add)

        self.btn_add.clicked.connect(self.add_clicked)


class SavedPositionListWidget(QWidget):
    move_to_requested = pyqtSignal(object)  # FibsemStagePosition
    positions_updated = pyqtSignal(list)    # List[FibsemStagePosition]
    position_selected = pyqtSignal(object)  # FibsemStagePosition

    def __init__(self, microscope=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self.positions: List[FibsemStagePosition] = []
        self._selected_position: Optional[FibsemStagePosition] = None
        self._setup_ui()
        self._load_default_positions()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _SavedPositionListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #3a3d42; border: none;")
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setSpacing(0)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setAlternatingRowColors(False)
        layout.addWidget(self._list)

        self._empty_label = QLabel("No saved positions")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-size: 11px; padding: 8px;")
        self._empty_label.setVisible(False)
        layout.addWidget(self._empty_label)

        self._header.add_clicked.connect(self._on_add_clicked)

        self.setToolTip(
            "Saved Positions — store named stage positions for quick recall.\n"
            f"Positions are automatically saved to:\n{cfg.POSITION_PATH}"
        )

        if self.microscope is None:
            self._header.btn_add.setEnabled(False)

    def _load_default_positions(self) -> None:
        path = cfg.POSITION_PATH
        if path and os.path.exists(path):
            self._load_positions_from_path(path)

    def _load_positions_from_path(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data:
                positions = [FibsemStagePosition.from_dict(d) for d in data]
                self.set_positions(positions)
        except Exception as e:
            logging.warning(f"Failed to load positions from {path}: {e}")

    def _autosave(self) -> None:
        try:
            data = [p.to_dict() for p in self.positions]
            with open(cfg.POSITION_PATH, "w") as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            logging.warning(f"Failed to autosave positions: {e}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_position(self, position: FibsemStagePosition) -> None:
        self.positions.append(position)
        self._add_row(position)
        self._update_empty_state()
        self._autosave()
        self.positions_updated.emit(self.positions)
        logging.info(f"Saved position added: {position.name}")

    def remove_position(self, position: FibsemStagePosition) -> None:
        for i in range(self._list.count()):
            if self._row(i).position is position:
                self._list.takeItem(i)
                break
        if position in self.positions:
            self.positions.remove(position)
        if self._selected_position is position:
            self._selected_position = None
        self._update_empty_state()
        self._autosave()
        self.positions_updated.emit(self.positions)
        logging.info(f"Saved position removed: {position.name}")

    def update_position(self, position: FibsemStagePosition) -> None:
        if self.microscope is None:
            return
        current = self.microscope.get_stage_position()
        position.x = current.x
        position.y = current.y
        position.z = current.z
        position.r = current.r
        position.t = current.t
        position.coordinate_system = current.coordinate_system
        for i in range(self._list.count()):
            row = self._row(i)
            if row.position is position:
                row.refresh(position)
                break
        self._autosave()
        self.positions_updated.emit(self.positions)
        logging.info(f"Saved position updated: {position.name}")

    def get_positions(self) -> List[FibsemStagePosition]:
        return list(self.positions)

    def set_positions(self, positions: List[FibsemStagePosition]) -> None:
        self._list.clear()
        self.positions = []
        self._selected_position = None
        for position in positions:
            self.positions.append(position)
            self._add_row(position)
        self._update_empty_state()
        self._autosave()

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _add_row(self, position: FibsemStagePosition) -> None:
        row = SavedPositionRowWidget(position)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        self._connect_row(row)

    def _connect_row(self, row: SavedPositionRowWidget) -> None:
        row.name_changed.connect(self._on_row_name_changed)
        row.move_to_clicked.connect(self._on_row_move_to_clicked)
        row.update_clicked.connect(self._on_row_update_clicked)
        row.remove_clicked.connect(self._on_row_remove_clicked)
        row.row_clicked.connect(self._on_row_clicked)

    def _row(self, index: int) -> SavedPositionRowWidget:
        return self._list.itemWidget(self._list.item(index))

    def _update_empty_state(self) -> None:
        empty = self._list.count() == 0
        self._list.setVisible(not empty)
        self._empty_label.setVisible(empty)

    def _on_add_clicked(self) -> None:
        if self.microscope is None:
            return
        position = self.microscope.get_stage_position()
        position.name = f"Position {len(self.positions):02d}"
        self.add_position(deepcopy(position))

    def _on_row_name_changed(self, position: FibsemStagePosition, name: str) -> None:
        if not name.strip():
            return
        position.name = name
        self._autosave()
        self.positions_updated.emit(self.positions)

    def _on_row_remove_clicked(self, position: FibsemStagePosition) -> None:
        confirmed = message_box_ui(
            text=f"Remove '{position.name}'? This cannot be undone.",
            title="Remove Position",
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=self,
        )
        if confirmed:
            self.remove_position(position)

    def _on_row_update_clicked(self, position: FibsemStagePosition) -> None:
        if self.microscope is None:
            return
        confirmed = message_box_ui(
            text=f"Update '{position.name}' to the current stage position?",
            title="Update Position",
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=self,
        )
        if confirmed:
            self.update_position(position)

    def _on_row_move_to_clicked(self, position: FibsemStagePosition) -> None:
        confirmed = message_box_ui(
            text=f"Move stage to '{position.name}'?",
            title="Move to Position",
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=self,
        )
        if confirmed:
            self.move_to_requested.emit(position)

    def _on_row_clicked(self, position: FibsemStagePosition) -> None:
        if self._selected_position is not None:
            for i in range(self._list.count()):
                row = self._row(i)
                if row is not None and row.position is self._selected_position:
                    row.set_selected(False)
                    break
        self._selected_position = position
        for i in range(self._list.count()):
            row = self._row(i)
            if row is not None and row.position is position:
                row.set_selected(True)
                break
        self.position_selected.emit(position)
