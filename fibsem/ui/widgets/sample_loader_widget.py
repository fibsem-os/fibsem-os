"""The autoloader's magazine: every slot, what is in it, and the exchange controls.

The hardware view of grids on a system with a loader. Empty and unscanned slots
are shown here and nowhere else: the Grids tab shows the experiment's records,
and a magazine slot with nothing in it is not one. Names typed here go to the
autoloader's slot description through ``Stage.assign_grid``, so they survive a
reconnect and an inventory reads them back.

Every hardware call -- inventory, load, unload -- runs off the GUI thread and the
panel repaints when it returns. Nothing here polls the loader.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleGridLoader
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.tokens import (
    ERROR_COLOR,
    NEUTRAL_700,
    OK_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel, style_with_tooltip
from fibsem.ui.widgets.sample_holder_widget import _NAME_FIELD_STYLE

_ROW_HEIGHT = 34
_SLOT_LABEL_WIDTH = 28
_STATE_LABEL_WIDTH = 64
_ICON_BTN = 26
# An arrow into a bracket for load, out of it for unload: the grid going into,
# and coming out of, the beam.
ICON_LOAD = "mdi:login"
ICON_UNLOAD = "mdi:logout"
ICON_INVENTORY = "mdi:refresh"
ICON_SCAN = "mdi:magnify-scan"
_ICON_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
QToolButton:disabled { background: transparent; }
"""

# A magazine slot is one of three things. `unknown` (a slot the hardware has not
# scanned since the magazine was opened) reads as empty until an inventory says
# otherwise: the loader model does not carry the hardware's Unknown state.
_STATE_COLOUR = {
    "unknown": NEUTRAL_700,
    "loaded": OK_COLOR,
    "occupied": TEXT_COLOR,
    "empty": NEUTRAL_700,
}
_STATE_TEXT = {
    "unknown": "unknown",
    "loaded": "loaded",
    "occupied": "occupied",
    "empty": "empty",
}
_STATE_TIP = {
    "unknown": "Not read yet: run an inventory to find out what is in this slot",
    "loaded": "This grid is in the holder's working slot right now",
    "occupied": "A grid is in this magazine slot; Load brings it into the beam",
    "empty": "Nothing in this magazine slot (or not scanned since the magazine was opened)",
}


def slot_state(slot: GridSlot, loaded: bool) -> str:
    """The row state for a slot the stage has answered for. UNKNOWN rows are
    handed their state directly, from the inventory."""
    if slot.loaded_grid is None:
        return "empty"
    return "loaded" if loaded else "occupied"


class _MagazineRow(QWidget):
    """One magazine slot: number, the grid's name (editable when there is one),
    its state, and Load."""

    load_clicked = pyqtSignal(object)  # GridSlot
    unload_clicked = pyqtSignal(object)  # GridSlot
    grid_named = pyqtSignal(object, str)  # GridSlot, new name

    def __init__(self, slot: GridSlot, state: str, parent=None) -> None:
        super().__init__(parent)
        self.slot = slot
        self.state = state
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(_ROW_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 6, 0)
        layout.setSpacing(8)

        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        layout.addWidget(self.dot)

        self.slot_label = QLabel(f"{slot.index + 1:02d}")
        self.slot_label.setFixedWidth(_SLOT_LABEL_WIDTH)
        style_with_tooltip(
            self.slot_label,
            f"font-family: monospace; color: {TEXT_MUTED_COLOR}; background: transparent;",
        )
        layout.addWidget(self.slot_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("empty")
        self.name_edit.setStyleSheet(_NAME_FIELD_STYLE)
        self.name_edit.setFixedHeight(24)
        self.name_edit.editingFinished.connect(self._on_name_edited)
        layout.addWidget(self.name_edit, 1)

        self.state_label = QLabel()
        self.state_label.setFixedWidth(_STATE_LABEL_WIDTH)
        layout.addWidget(self.state_label)

        # One action, contextual: load an occupied slot's grid, unload the one in
        # the beam. Nothing on an empty slot.
        self.btn_action = QToolButton()
        self.btn_action.setFixedSize(_ICON_BTN, _ICON_BTN)
        self.btn_action.setIconSize(QSize(18, 18))
        self.btn_action.setStyleSheet(_ICON_BTN_STYLE)
        self.btn_action.clicked.connect(self._on_action)
        layout.addWidget(self.btn_action)

        self.refresh(state)

    def refresh(self, state: str, controls_enabled: bool = True) -> None:
        slot = self.slot
        self.state = state
        grid = slot.loaded_grid
        name = grid.name if grid is not None else ""
        if self.name_edit.text() != name:
            self.name_edit.setText(name)
        # A name means a grid: only a slot with one in it can be named. Naming an
        # empty slot would be declaring a grid the hardware says is not there.
        self.name_edit.setReadOnly(grid is None)
        self.name_edit.setToolTip(
            "The grid in this slot; written to the autoloader's slot description"
            if grid is not None
            else _STATE_TIP[self.state if self.state == "unknown" else "empty"]
        )
        style_with_tooltip(
            self.dot,
            f"background: {_STATE_COLOUR[self.state]}; border-radius: 5px;",
        )
        self.dot.setToolTip(_STATE_TIP[self.state])
        colour = TEXT_STRONG_COLOR if self.state == "loaded" else TEXT_MUTED_COLOR
        style_with_tooltip(
            self.state_label,
            f"color: {colour}; font-size: 11px; background: transparent;",
        )
        self.state_label.setText(_STATE_TEXT[self.state])
        self.state_label.setToolTip(_STATE_TIP[self.state])

        if self.state == "loaded":
            self.btn_action.setIcon(
                fibsem_icon(ICON_UNLOAD, color=stylesheets.GRAY_ICON_COLOR)
            )
            self.btn_action.setToolTip(
                f"Return {name} to the magazine"
                if controls_enabled
                else "Not while the loader is busy or a workflow is running"
            )
        else:
            self.btn_action.setIcon(
                fibsem_icon(ICON_LOAD, color=stylesheets.GRAY_ICON_COLOR)
            )
            self.btn_action.setToolTip(
                f"Bring {name} into the beam"
                if self.state == "occupied" and controls_enabled
                else "Not while the loader is busy or a workflow is running"
                if self.state == "occupied"
                else "Nothing to load"
            )
        # An empty slot keeps the icon's space, blank, so the state column lines
        # up down the list.
        if self.state == "empty":
            self.btn_action.setIcon(QIcon())
        self.btn_action.setEnabled(self.state != "empty" and controls_enabled)

    def _on_action(self) -> None:
        if self.state == "loaded":
            self.unload_clicked.emit(self.slot)
        elif self.state == "occupied":
            self.load_clicked.emit(self.slot)

    def _on_name_edited(self) -> None:
        name = self.name_edit.text().strip()
        current = self.slot.loaded_grid.name if self.slot.loaded_grid else ""
        if name and name != current:
            self.grid_named.emit(self.slot, name)
        elif not name:
            # A grid cannot be un-named; put the name back.
            self.name_edit.setText(current)


class SampleLoaderWidget(QWidget):
    """The magazine, Run inventory, Load per slot, Unload.

    ``loader_changed`` fires after any of those finish, whether or not they
    succeeded, so a host drawing the holder's working slot repaints. ``busy_changed``
    tracks an exchange or scan in flight. ``set_controls_enabled(False)`` is the
    host's lockout while a workflow owns the loader.

    ``synchronous`` runs the hardware calls on the calling thread, for tests.
    """

    loader_changed = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, microscope=None, parent=None, synchronous: bool = False):
        super().__init__(parent)
        self._microscope = microscope
        self._synchronous = synchronous
        self._rows: List[_MagazineRow] = []
        self._busy = False
        self._controls_enabled = True
        self._scanned_at: Optional[datetime] = None
        self._read_at: Optional[datetime] = None
        self._worker = None
        self._setup_ui()
        self.refresh()

    # -- layout ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 6)
        inner_layout.setSpacing(6)

        # The scan in the title bar, as an icon, where the app keeps a panel's
        # actions. It moves the magazine, so it asks first.
        self.btn_inventory = QToolButton()
        self.btn_inventory.setFixedSize(_ICON_BTN, _ICON_BTN)
        self.btn_inventory.setIconSize(QSize(18, 18))
        self.btn_inventory.setStyleSheet(_ICON_BTN_STYLE)
        self.btn_inventory.setIcon(
            fibsem_icon(ICON_INVENTORY, color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_inventory.setToolTip(
            "Refresh: read what the autoloader knows about its magazine. Instant; "
            "as fresh as its last scan"
        )
        self.btn_inventory.clicked.connect(self._on_get_inventory)
        # The scan is the slow one, and moves the magazine: it asks first.
        self.btn_scan = QToolButton()
        self.btn_scan.setFixedSize(_ICON_BTN, _ICON_BTN)
        self.btn_scan.setIconSize(QSize(18, 18))
        self.btn_scan.setStyleSheet(_ICON_BTN_STYLE)
        self.btn_scan.setIcon(fibsem_icon(ICON_SCAN, color=stylesheets.GRAY_ICON_COLOR))
        self.btn_scan.setToolTip(
            "Scan magazine: the autoloader checks every slot for a grid. Takes a "
            "while; the magazine must stay shut"
        )
        self.btn_scan.clicked.connect(self._on_run_inventory)

        self.facts_label = QLabel()
        self.facts_label.setWordWrap(True)
        self.facts_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; background: transparent;"
        )
        inner_layout.addWidget(self.facts_label)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setSelectionMode(QListWidget.NoSelection)
        # Fixed (the default) lays item widgets out once, at whatever width the
        # list had then; a later resize or restyle leaves them clipped.
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner_layout.addWidget(self._list)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; background: transparent;"
        )
        inner_layout.addWidget(self.status_label)

        self._panel = TitledPanel("Loader", content=inner, collapsible=True)
        self._panel.add_header_widget(self.btn_inventory)
        self._panel.add_header_widget(self.btn_scan)
        layout.addWidget(self._panel)

    # -- model -----------------------------------------------------------------

    @property
    def loader(self) -> Optional[SampleGridLoader]:
        stage = getattr(self._microscope, "_stage", None)
        return getattr(stage, "loader", None)

    @property
    def busy(self) -> bool:
        return self._busy

    def refresh(self) -> None:
        loader = self.loader
        self._list.clear()
        self._rows = []
        self.setEnabled(loader is not None)
        if loader is None:
            self.facts_label.setText("No autoloader on this system.")
            return

        states = {
            e.slot_name: e.state.value for e in self._microscope._stage.grid_inventory()
        }
        slots = sorted(loader.slots.values(), key=lambda s: s.index)
        occupied = sum(1 for s in slots if s.loaded_grid is not None)
        scanned = (
            f"scanned {self._scanned_at.strftime('%H:%M')}"
            if self._scanned_at is not None
            else f"read {self._read_at.strftime('%H:%M')}, not scanned this session"
            if self._read_at is not None
            else "not scanned this session"
        )
        plural = "" if occupied == 1 else "s"
        self.facts_label.setText(
            f"Magazine · {len(slots)} slots · {occupied} grid{plural} · {scanned}"
        )

        controls = self._controls_enabled and not self._busy
        for slot in slots:
            row = _MagazineRow(slot, states.get(slot.name, "unknown"))
            row.refresh(row.state, controls_enabled=controls)
            row.load_clicked.connect(self._on_load)
            row.unload_clicked.connect(self._on_unload)
            row.grid_named.connect(self._on_grid_named)
            item = QListWidgetItem(self._list)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            self._rows.append(row)
        self._list.setFixedHeight(
            max(2, len(slots)) * _ROW_HEIGHT + 2 * self._list.frameWidth()
        )
        self._apply_controls()

    def _row_widget(self, i: int) -> Optional[_MagazineRow]:
        return self._rows[i] if 0 <= i < len(self._rows) else None

    def set_controls_enabled(self, enabled: bool) -> None:
        """The host's lockout: nothing here may touch the loader while a workflow
        owns it. Names stay editable; they are not hardware motion."""
        self._controls_enabled = enabled
        self._apply_controls()

    def _apply_controls(self) -> None:
        active = self._controls_enabled and not self._busy and self.loader is not None
        self.btn_inventory.setEnabled(active)
        self.btn_scan.setEnabled(active)
        for row in self._rows:
            row.refresh(row.state, controls_enabled=active)

    # -- edits -----------------------------------------------------------------

    def _on_grid_named(self, slot: GridSlot, name: str) -> None:
        grid = slot.loaded_grid
        if grid is None or self._microscope is None:
            return
        grid = SampleGrid(name=name, description=grid.description, radius=grid.radius)
        try:
            self._microscope._stage.assign_grid(slot.name, grid)
        except Exception as e:  # noqa: BLE001 - keep the in-memory change, say so
            logging.warning(f"Could not write the name of {slot.name}: {e}")
            slot.loaded_grid = grid
        self.refresh()
        self.loader_changed.emit()

    # -- hardware, off the GUI thread --------------------------------------------

    def _on_get_inventory(self) -> None:
        if self.loader is None:
            return

        def job() -> None:
            self._microscope._stage.get_inventory()
            self._read_at = datetime.now()

        self._start(job, "Reading the magazine…", "Inventory read.")

    def _on_run_inventory(self) -> None:
        loader = self.loader
        if loader is None:
            return
        if not self._confirm(
            "Scan magazine",
            "Scan the magazine? The autoloader checks every slot for a grid and "
            "reads the names on the slot descriptions; it takes a while and the "
            "magazine must not be opened while it runs.",
        ):
            return

        def job() -> None:
            self._microscope._stage.run_inventory()
            self._scanned_at = datetime.now()

        self._start(job, "Scanning the magazine…", "Scan complete.")

    def _on_load(self, slot: GridSlot) -> None:
        loader = self.loader
        if loader is None or slot.loaded_grid is None:
            return
        name = slot.loaded_grid.name

        def job() -> None:
            loader.load_grid(slot.name)

        self._start(job, f"Loading {name}…", f"{name} is loaded.")

    def _on_unload(self, slot: GridSlot) -> None:
        loader = self.loader
        if loader is None or slot.loaded_grid is None:
            return
        name = slot.loaded_grid.name

        def job() -> None:
            loader.unload_grid()

        self._start(job, f"Unloading {name}…", f"{name} returned to the magazine.")

    def _confirm(self, title: str, text: str) -> bool:
        """Yes/No before an action that moves the loader. Answered without a
        dialog in synchronous (test) mode."""
        if self._synchronous:
            return True
        return (
            QMessageBox.question(
                self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            == QMessageBox.Yes
        )

    def _start(self, job: Callable[[], None], doing: str, done: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._say(doing)
        self._done_text = done
        if self._synchronous:
            try:
                job()
                self._on_returned()
            except Exception as e:  # noqa: BLE001 - reported on the panel
                self._on_errored(e)
            finally:
                self._on_finished()
            return
        worker = self._job_worker(job)
        worker.returned.connect(self._on_returned)
        worker.errored.connect(self._on_errored)
        worker.finished.connect(self._on_finished)
        self._worker = worker  # keep it alive; only signals cross back
        worker.start()

    @thread_worker
    def _job_worker(self, job: Callable[[], None]) -> None:
        job()

    def _on_returned(self, _result: object = None) -> None:
        self._say(self._done_text)

    def _on_errored(self, error: Exception) -> None:
        logging.warning(f"Autoloader operation failed: {error}")
        self._say(str(error), error=True)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        self.refresh()
        self.loader_changed.emit()

    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        self._apply_controls()
        self.busy_changed.emit(busy)

    def _say(self, text: str, error: bool = False) -> None:
        colour = ERROR_COLOR if error else TEXT_MUTED_COLOR
        self.status_label.setStyleSheet(
            f"color: {colour}; font-size: 11px; background: transparent;"
        )
        self.status_label.setText(text)
