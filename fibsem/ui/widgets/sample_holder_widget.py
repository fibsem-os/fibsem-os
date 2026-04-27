from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleHolder
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueSpinBox

_ROW_HEIGHT = 40
_BTN_SIZE = QSize(32, 32)
_ACTIONS_BTN_SIZE = 24
_SLOT_LABEL_WIDTH = 72

_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
"""

_EMPTY_STYLE = "color: #606060; font-style: italic; background: transparent;"
_LOADED_STYLE = "background: transparent;"


class _GridSlotRowWidget(QWidget):
    move_clicked = pyqtSignal(object)    # GridSlot
    capture_clicked = pyqtSignal(object) # GridSlot
    clear_clicked = pyqtSignal(object)   # GridSlot
    row_clicked = pyqtSignal(object)     # GridSlot

    def __init__(
        self,
        slot: GridSlot,
        has_microscope: bool = False,
        show_move: bool = True,
        show_grid_edit: bool = False,
        parent=None,
    ):
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

        self.grid_label = QLabel()
        layout.addWidget(self.grid_label, 1)

        # Stage position actions in "..." dropdown
        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(_ACTIONS_BTN_SIZE, _ACTIONS_BTN_SIZE)
        self._btn_actions.setStyleSheet(
            _BTN_STYLE + " QToolButton::menu-indicator { image: none; }"
        )
        self._btn_actions.setIcon(
            QIconifyIcon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR)
        )
        self._btn_actions.setToolTip("Actions")
        self._btn_actions.setPopupMode(QToolButton.InstantPopup)
        self._btn_actions.setVisible(has_microscope and show_move)

        _menu = QMenu(self)
        if has_microscope and show_move:
            _action_capture = _menu.addAction(
                QIconifyIcon("mdi:map-marker-plus", color=stylesheets.GRAY_ICON_COLOR),
                "Update Position",
            )
            _action_move = _menu.addAction(
                QIconifyIcon("mdi:crosshairs-gps", color=stylesheets.GRAY_ICON_COLOR),
                "Move to Position",
            )
            _action_capture.triggered.connect(lambda: self.capture_clicked.emit(self.slot))
            _action_move.triggered.connect(lambda: self.move_clicked.emit(self.slot))

        self._btn_actions.setMenu(_menu)
        self._btn_actions.installEventFilter(self)
        layout.addWidget(self._btn_actions)

        # Clear button — always visible, enabled only when a grid is loaded
        self.btn_clear = QToolButton()
        self.btn_clear.setFixedSize(_ACTIONS_BTN_SIZE, _ACTIONS_BTN_SIZE)
        self.btn_clear.setStyleSheet(_BTN_STYLE)
        self.btn_clear.setIcon(
            QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_clear.setToolTip("Remove grid from this slot")
        self.btn_clear.setVisible(show_grid_edit)
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
        self.slot_label.setText(self.slot.name)
        if self.slot.loaded_grid is not None:
            self.grid_label.setText(self.slot.loaded_grid.name)
            self.grid_label.setStyleSheet(_LOADED_STYLE)
        else:
            self.grid_label.setText("Empty")
            self.grid_label.setStyleSheet(_EMPTY_STYLE)
        if self.btn_clear.isVisible():
            self.btn_clear.setEnabled(self.slot.loaded_grid is not None)


class _GridListHeader(QWidget):
    def __init__(self, title: str = "Slots", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")
        self.setFixedHeight(_BTN_SIZE.height() + 8)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(4)

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(label, 1)


# ---------------------------------------------------------------------------
# Edit panel — shown whenever a slot row is selected
# ---------------------------------------------------------------------------


class _GridSlotEditPanel(QWidget):
    """Inline edit panel for the SampleGrid assigned to a GridSlot.
    Changes auto-apply when the user leaves a field (editingFinished)."""

    slot_changed = pyqtSignal(object)  # GridSlot

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slot: Optional[GridSlot] = None
        self._updating = False
        self._setup_ui()
        self._connect_signals()
        self.setVisible(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        self.slot_name_label = QLabel()
        self.slot_name_label.setStyleSheet("font-weight: bold;")
        self.position_label = QLabel()
        self.position_label.setStyleSheet("color: #a0a0a0;")
        header_row.addWidget(self.slot_name_label)
        header_row.addWidget(self.position_label, 1)
        layout.addLayout(header_row)

        form = QFormLayout()
        form.setSpacing(4)
        self.grid_name_edit = QLineEdit()
        self.grid_name_edit.setPlaceholderText("Grid name")
        self.grid_description_edit = QLineEdit()
        self.grid_description_edit.setPlaceholderText("Description (optional)")
        form.addRow("Name", self.grid_name_edit)
        form.addRow("Description", self.grid_description_edit)
        layout.addLayout(form)

    def _connect_signals(self) -> None:
        self.grid_name_edit.editingFinished.connect(self._handle_apply)
        self.grid_description_edit.editingFinished.connect(self._handle_apply)

    def set_slot(self, slot: Optional[GridSlot]) -> None:
        self._slot = slot
        self.setVisible(slot is not None)
        if slot is None:
            return
        self._updating = True
        self.slot_name_label.setText(slot.name)
        self.position_label.setText(slot.position.pretty if slot.position else "—")
        loaded = slot.loaded_grid
        self.grid_name_edit.setText(loaded.name if loaded else "")
        self.grid_description_edit.setText(loaded.description if loaded else "")
        self._updating = False

    def _handle_apply(self) -> None:
        if self._slot is None or self._updating:
            return
        name = self.grid_name_edit.text().strip() or "Grid"
        description = self.grid_description_edit.text()
        if self._slot.loaded_grid is None:
            self._slot.loaded_grid = SampleGrid(name=name, description=description)
        else:
            self._slot.loaded_grid.name = name
            self._slot.loaded_grid.description = description
        self.slot_changed.emit(self._slot)


# ---------------------------------------------------------------------------
# SampleHolderWidget — hardware configuration + grid assignment
# ---------------------------------------------------------------------------


class SampleHolderWidget(QWidget):
    """Sample holder configuration: metadata, slot positions, and grid assignment."""

    holder_changed = pyqtSignal(object)  # SampleHolder
    grid_selected = pyqtSignal(str, object)  # slot_name, GridSlot

    def __init__(self, microscope=None, parent=None):
        super().__init__(parent)
        self._microscope = microscope
        self._holder: Optional[SampleHolder] = None
        self._updating = False
        self._setup_ui()
        self._connect_signals()
        self.setEnabled(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(4, 4, 4, 0)
        form.setSpacing(4)

        self.name_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.capacity_spin = ValueSpinBox(
            minimum=1.0,
            maximum=12.0,
            step=1.0,
            decimals=0,
            tooltip="Number of slots on this holder",
        )
        self.pre_tilt_spin = ValueSpinBox(
            suffix="°",
            minimum=0.0,
            maximum=90.0,
            step=1.0,
            decimals=0,
            tooltip="Pre-tilt angle of the sample holder",
        )
        self.reference_rotation_spin = ValueSpinBox(
            suffix="°",
            minimum=0.0,
            maximum=360.0,
            step=1.0,
            decimals=0,
            tooltip="Reference rotation angle of the sample holder",
        )
        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("Capacity", self.capacity_spin)
        form.addRow("Pre-Tilt", self.pre_tilt_spin)
        form.addRow("Ref. Rotation", self.reference_rotation_spin)

        save_load_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Holder")
        self.btn_save.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_load_file = QPushButton("Load Holder")
        self.btn_load_file.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        save_load_row.addWidget(self.btn_save)
        save_load_row.addWidget(self.btn_load_file)

        self._header = _GridListHeader("Slots")

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setMinimumHeight(3 * _ROW_HEIGHT)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._empty_label = QLabel("No slots defined.")
        self._empty_label.setStyleSheet(
            "color: #606060; font-style: italic; padding: 8px;"
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._edit_panel = _GridSlotEditPanel()

        holder_inner = QWidget()
        inner_layout = QVBoxLayout(holder_inner)
        inner_layout.setContentsMargins(0, 0, 0, 4)
        inner_layout.setSpacing(4)
        inner_layout.addWidget(form_widget)
        inner_layout.addLayout(save_load_row)
        inner_layout.addWidget(self._header)
        inner_layout.addWidget(self._list)
        inner_layout.addWidget(self._empty_label)
        inner_layout.addWidget(self._edit_panel)

        self._holder_panel = TitledPanel(
            "Sample Holder", content=holder_inner, collapsible=True
        )
        layout.addWidget(self._holder_panel)

        self._update_empty_state()

    def _connect_signals(self) -> None:
        self.name_edit.editingFinished.connect(self._on_holder_form_changed)
        self.description_edit.editingFinished.connect(self._on_holder_form_changed)
        self.capacity_spin.valueChanged.connect(self._on_capacity_changed)
        self.pre_tilt_spin.valueChanged.connect(self._on_holder_form_changed)
        self.reference_rotation_spin.valueChanged.connect(self._on_holder_form_changed)

        self.btn_save.clicked.connect(self._handle_save_holder)
        self.btn_load_file.clicked.connect(self._handle_load_holder)

        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._edit_panel.slot_changed.connect(self._on_slot_changed)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_holder(self, holder: Optional[SampleHolder]) -> None:
        self._holder = holder
        self.setEnabled(holder is not None)
        self._edit_panel.setVisible(False)

        self._updating = True
        if holder is None:
            self.name_edit.clear()
            self.description_edit.clear()
            self.capacity_spin.setValue(2.0)
            self.pre_tilt_spin.setValue(0.0)
            self.reference_rotation_spin.setValue(0.0)
        else:
            self.name_edit.setText(holder.name)
            self.description_edit.setText(holder.description or "")
            self.capacity_spin.setValue(float(holder.capacity))
            self.pre_tilt_spin.setValue(holder.pre_tilt)
            self.reference_rotation_spin.setValue(holder.reference_rotation)
        self._updating = False

        self._refresh_slot_list()

    def current_holder(self) -> Optional[SampleHolder]:
        return self._holder

    def refresh(self) -> None:
        """Refresh row labels in-place after external slot mutations."""
        for i in range(self._list.count()):
            row = self._row_widget(i)
            if row:
                row.refresh()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _refresh_slot_list(self) -> None:
        self._list.clear()
        if self._holder is None:
            self._update_empty_state()
            return

        for slot in sorted(self._holder.slots.values(), key=lambda s: s.index):
            row = _GridSlotRowWidget(
                slot,
                has_microscope=self._microscope is not None,
                show_grid_edit=True,
            )
            row.row_clicked.connect(self._on_row_clicked)
            row.move_clicked.connect(self._on_move_slot)
            row.capture_clicked.connect(self._on_capture_slot)
            row.clear_clicked.connect(self._on_clear_slot)

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, slot)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

        self._update_empty_state()

    def _update_empty_state(self) -> None:
        self._empty_label.setVisible(self._list.count() == 0)

    def _row_widget(self, i: int) -> Optional[_GridSlotRowWidget]:
        item = self._list.item(i)
        return self._list.itemWidget(item) if item else None  # type: ignore

    def _on_list_row_changed(self, row: int) -> None:
        if row < 0:
            self._edit_panel.setVisible(False)
            return
        item = self._list.item(row)
        if item is None:
            return
        slot = item.data(Qt.ItemDataRole.UserRole)
        if slot is not None:
            self._edit_panel.set_slot(slot)
            self.grid_selected.emit(slot.name, slot)

    def _on_row_clicked(self, slot: GridSlot) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) is slot:
                self._list.setCurrentRow(i)
                return

    def _on_clear_slot(self, slot: GridSlot) -> None:
        slot.loaded_grid = None
        self._on_slot_changed(slot)
        if self._edit_panel._slot is slot:
            self._edit_panel.set_slot(slot)

    def _on_slot_changed(self, slot: GridSlot) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) is slot:
                row = self._row_widget(i)
                if row:
                    row.refresh()
                break
        if self._holder is not None:
            self.holder_changed.emit(self._holder)

    def _on_holder_form_changed(self) -> None:
        if self._holder is None or self._updating:
            return
        self._holder.name = self.name_edit.text()
        self._holder.description = self.description_edit.text()
        self._holder.pre_tilt = self.pre_tilt_spin.value()
        self._holder.reference_rotation = self.reference_rotation_spin.value()
        self.holder_changed.emit(self._holder)

    def _on_capacity_changed(self) -> None:
        if self._holder is None or self._updating:
            return
        self._holder.capacity = int(self.capacity_spin.value())
        self._holder._ensure_slots()
        self._refresh_slot_list()
        self.holder_changed.emit(self._holder)

    def _on_capture_slot(self, slot: GridSlot) -> None:
        if self._microscope is None:
            return
        try:
            pos = self._microscope.get_stage_position()
            pos.name = slot.name
            slot.position = pos
            if self._holder is not None:
                self.holder_changed.emit(self._holder)
        except Exception as e:
            logging.warning(f"Failed to capture stage position: {e}")

    def _on_move_slot(self, slot: GridSlot) -> None:
        if self._microscope is None:
            return
        try:
            self._microscope._stage.move_to_slot(slot.name)
        except Exception as e:
            logging.warning(f"Failed to move to slot '{slot.name}': {e}")

    def _handle_save_holder(self) -> None:
        if self._holder is None:
            return
        from fibsem import config as cfg

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Sample Holder", cfg.CONFIG_PATH, "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        if not path.endswith((".yaml", ".yml")):
            path += ".yaml"
        try:
            self._holder.save(path)
            logging.info(f"Saved sample holder to {path}")
        except Exception as e:
            logging.error(f"Failed to save holder: {e}")

    def _handle_load_holder(self) -> None:
        from fibsem import config as cfg

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Sample Holder", cfg.CONFIG_PATH, "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            holder = SampleHolder.load(path)
            self.set_holder(holder)
            self.holder_changed.emit(holder)
        except Exception as e:
            logging.error(f"Failed to load holder: {e}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from fibsem import utils

    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, settings = utils.setup_session(config_path=None)
    holder = microscope._stage.holder

    widget = SampleHolderWidget(microscope=microscope)
    widget.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    widget.set_holder(holder)

    def on_holder_changed(h: SampleHolder) -> None:
        for slot in h.slots.values():
            g = slot.loaded_grid.name if slot.loaded_grid else "Empty"
            print(f"  {slot.name}: {g}")

    widget.holder_changed.connect(on_holder_changed)
    widget.resize(500, 700)
    widget.show()
    sys.exit(app.exec_())
