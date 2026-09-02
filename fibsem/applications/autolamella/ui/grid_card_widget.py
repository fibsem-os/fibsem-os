"""A card per grid record: the experiment's view of a grid.

The same card as a lamella's, so the two tabs read alike: a thumbnail, the name
over a status line, a small icon for the person's verdict, and an actions menu.
What the hardware says (in the beam, or gone from the magazine) is a chip. The
card's object is the ``GridRecord``, not a hardware slot: the inventory entry
arrives on every refresh and is only drawn.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    GridQuality,
    GridRecord,
)
from fibsem.applications.autolamella.task_outputs import latest_grid_output
from fibsem.config import CARD_MODES, MODE_COMPACT, MODE_COZY, MODE_STANDARD
from fibsem.microscopes._stage import GridInventoryEntry
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    ERROR_COLOR,
    NEUTRAL_200,
    NEUTRAL_550,
    NEUTRAL_700,
    NEUTRAL_900,
    OK_COLOR,
    PRIMARY_COLOR,
    SURFACE_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel, chip
from fibsem.ui.widgets.sample_loader_widget import ICON_LOAD, ICON_UNLOAD

_CARD_WIDTH = 300
_THUMB_PADDING = 6
# The lamella card's sizes, so the two strips line up: a small thumbnail beside
# the name in the standard row, a big one above it in the cozy tile.
_THUMB_W, _THUMB_H = 66, 44
_COZY_THUMB_W = _CARD_WIDTH - 8 - _THUMB_PADDING * 2
_COZY_THUMB_H = 170
_BTN_SIZE = 24
# The load step's history entry, as the grid manager writes it.
_LOAD_ENTRY_NAME = "load"
# Which overview a card shows, in order of preference.
_THUMBNAIL_ROLES = (
    "overview_sem_thumbnail",
    "overview_fib_thumbnail",
    "overview_fm_thumbnail",
)

_CARD_STYLE = f"""
QFrame#GridCard {{
    background: {SURFACE_COLOR};
    border: 1px solid #3a3d42;
    border-radius: 8px;
}}
"""
_CARD_SELECTED_STYLE = f"""
QFrame#GridCard {{
    background: {SURFACE_COLOR};
    border: 2px solid {PRIMARY_COLOR};
    border-radius: 8px;
}}
"""
_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
QToolButton::menu-indicator { image: none; }
"""

_QUALITY_ICON = {
    GridQuality.UNASSESSED: ("mdi:help-circle-outline", NEUTRAL_550, "Unassessed"),
    GridQuality.GOOD: ("mdi:check-circle", stylesheets.GREEN_COLOR, "Good"),
    GridQuality.POOR: ("mdi:close-circle", stylesheets.DEFECT_RED_COLOR, "Poor"),
}


def grid_headline(grid: GridRecord) -> Tuple[str, str]:
    """One line on how the grid's last run went, and its colour.

    Read off the history, never off the quality: whether the tasks ran is a
    different question from whether the grid is any good. The most recent load
    entry starts the run being described; task entries after it are the run.
    """
    state = grid.task_state
    if state.status is AutoLamellaTaskStatus.InProgress:
        return f"Running {state.name}", ACCENT_COLOR
    history = grid.task_history
    if not history:
        return "Not run", NEUTRAL_550
    start = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i].name == _LOAD_ENTRY_NAME:
            start = i
            break
    load = history[start] if history[start].name == _LOAD_ENTRY_NAME else None
    tasks = [t for t in history[start:] if t.name != _LOAD_ENTRY_NAME]
    if load is not None and load.status is AutoLamellaTaskStatus.Failed:
        return "Load failed", ERROR_COLOR
    if not tasks:
        return "Not run", NEUTRAL_550
    failed = sum(1 for t in tasks if t.status is AutoLamellaTaskStatus.Failed)
    cancelled = sum(1 for t in tasks if t.status is AutoLamellaTaskStatus.Cancelled)
    if failed:
        return f"{failed} task{'s' if failed != 1 else ''} failed", ERROR_COLOR
    if cancelled:
        return "Cancelled", stylesheets.DEFECT_ORANGE_COLOR
    # The lamella card's line: the last completed task and when, not a verdict.
    last = tasks[-1]
    return f"{last.name} ({last.completed_at})", NEUTRAL_550


# -- thumbnails, decoded once per file -----------------------------------------

_THUMBNAIL_CACHE: "OrderedDict[tuple, QImage]" = OrderedDict()
_THUMBNAIL_CACHE_MAX = 200


def _stamp(path: str) -> Optional[tuple]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def grid_thumbnail_path(experiment: Experiment, grid: GridRecord) -> Optional[str]:
    """The latest overview thumbnail a run recorded for the grid, SEM first."""
    for role in _THUMBNAIL_ROLES:
        path = latest_grid_output(experiment, grid, role)
        if path is not None:
            return path
    return None


def _thumbnail_image(path: Optional[str], w: int, h: int) -> Optional[QImage]:
    """The thumbnail scaled for a `w` x `h` label, decoded at most once per file
    (the lamella card's lesson, FIB-681: a strip of cards must not re-decode on
    every refresh)."""
    if path is None:
        return None
    key = (path, _stamp(path), w, h)
    image = _THUMBNAIL_CACHE.get(key)
    if image is not None:
        _THUMBNAIL_CACHE.move_to_end(key)
        return image
    image = QImage(path)
    if image.isNull():
        return None
    image = image.scaled(
        w,
        h,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    _THUMBNAIL_CACHE[key] = image
    if len(_THUMBNAIL_CACHE) > _THUMBNAIL_CACHE_MAX:
        _THUMBNAIL_CACHE.popitem(last=False)
    return image


class GridCardWidget(QWidget):
    """One grid record: thumbnail | name over status, chips | quality | actions."""

    clicked = pyqtSignal(object)  # GridRecord
    quality_changed = pyqtSignal(object)  # GridRecord
    load_requested = pyqtSignal(object)  # GridRecord
    unload_requested = pyqtSignal(object)  # GridRecord
    rename_requested = pyqtSignal(object, str)  # GridRecord, new name
    remove_requested = pyqtSignal(object)  # GridRecord

    def __init__(
        self,
        grid: GridRecord,
        experiment: Optional[Experiment] = None,
        mode: str = MODE_COZY,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.grid = grid
        self.experiment = experiment
        self._mode = mode if mode in CARD_MODES else MODE_COZY
        self._entry: Optional[GridInventoryEntry] = None
        self._has_loader = False
        self._controls_enabled = True
        self.setFixedWidth(_CARD_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)
        self._card = QFrame()
        self._card.setObjectName("GridCard")
        self._card.setStyleSheet(_CARD_STYLE)
        self._card.setFixedWidth(_CARD_WIDTH - 8)
        # One content widget the arrangement swaps; the children below are built
        # once and re-parented, so signals and menus survive a mode switch.
        self._frame_layout = QVBoxLayout(self._card)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(0)
        self._content: Optional[QWidget] = None

        self._thumb_label = QLabel()
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setStyleSheet(
            f"background: {NEUTRAL_900}; border-radius: 4px;"
        )
        self._name_label = ElidedLabel()
        self._status_label = ElidedLabel()
        self._chips = QHBoxLayout()
        self._chips.setSpacing(4)
        self._chip_widgets: List[QLabel] = []

        self._btn_quality = QToolButton()
        self._btn_quality.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_quality.setStyleSheet(_BTN_STYLE)
        self._btn_quality.clicked.connect(self._on_quality_clicked)

        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_actions.setStyleSheet(_BTN_STYLE)
        self._btn_actions.setIcon(
            fibsem_icon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR)
        )
        self._btn_actions.setToolTip("Actions")
        self._btn_actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        self._action_load = menu.addAction(
            fibsem_icon(ICON_LOAD, color=stylesheets.GRAY_ICON_COLOR), "Load"
        )
        self._action_unload = menu.addAction(
            fibsem_icon(ICON_UNLOAD, color=stylesheets.GRAY_ICON_COLOR), "Unload"
        )
        self._action_rename = menu.addAction(
            fibsem_icon("mdi:pencil-outline", color=stylesheets.GRAY_ICON_COLOR),
            "Rename…",
        )
        self._action_remove = menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR),
            "Remove",
        )
        self._action_load.triggered.connect(lambda: self.load_requested.emit(self.grid))
        self._action_unload.triggered.connect(
            lambda: self.unload_requested.emit(self.grid)
        )
        self._action_rename.triggered.connect(self._on_rename)
        self._action_remove.triggered.connect(self._on_remove)
        self._btn_actions.setMenu(menu)

        outer.addWidget(self._card)
        self._apply_layout()
        self.refresh()

    # -- arrangement -----------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Cozy tile (big thumbnail), standard row, or compact line."""
        if mode not in CARD_MODES or mode == self._mode:
            return
        self._mode = mode
        self._apply_layout()
        self.refresh()

    @property
    def mode(self) -> str:
        return self._mode

    def _info_block(self) -> QWidget:
        """Name over status and chips; the same block in every arrangement."""
        info = QWidget()
        info.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(info)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addStretch(1)
        layout.addWidget(self._name_label)
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self._status_label, 1)
        status_row.addLayout(self._chips)
        layout.addLayout(status_row)
        layout.addStretch(1)
        return info

    def _apply_layout(self) -> None:
        if self._content is not None:
            self._frame_layout.removeWidget(self._content)
            self._content.setParent(None)
            self._content.deleteLater()
        if self._mode == MODE_COZY:
            # The lamella card's cozy tile, measure for measure: the thumbnail,
            # a rule, then a name row with the buttons and the status under it.
            self._thumb_label.setFixedSize(_COZY_THUMB_W, _COZY_THUMB_H)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(_THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING, 0)
            layout.setSpacing(0)
            layout.addWidget(self._thumb_label)
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #3a3d42;")
            layout.addWidget(sep)
            info = QWidget()
            info.setStyleSheet("background: transparent;")
            info_layout = QVBoxLayout(info)
            info_layout.setContentsMargins(10, 8, 10, 10)
            info_layout.setSpacing(3)
            name_row = QHBoxLayout()
            name_row.setSpacing(4)
            name_row.addWidget(self._name_label, 1)
            name_row.addWidget(self._btn_actions)
            name_row.addWidget(self._btn_quality)
            info_layout.addLayout(name_row)
            status_row = QHBoxLayout()
            status_row.setSpacing(6)
            status_row.addWidget(self._status_label, 1)
            status_row.addLayout(self._chips)
            info_layout.addLayout(status_row)
            layout.addWidget(info)
        else:
            self._thumb_label.setFixedSize(_THUMB_W, _THUMB_H)
            self._thumb_label.setVisible(self._mode != MODE_COMPACT)
            content = QWidget()
            row = QHBoxLayout(content)
            row.setContentsMargins(
                _THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING
            )
            row.setSpacing(6)
            row.addWidget(self._thumb_label)
            row.addWidget(self._info_block(), 1)
            row.addWidget(self._btn_quality, 0, Qt.AlignVCenter)
            row.addWidget(self._btn_actions, 0, Qt.AlignVCenter)
        self._thumb_label.setVisible(self._mode != MODE_COMPACT)
        content.setStyleSheet("background: transparent;")
        self._content = content
        self._frame_layout.addWidget(content)

    # -- what the hardware says ------------------------------------------------

    def set_inventory(
        self, entry: Optional[GridInventoryEntry], has_loader: bool
    ) -> None:
        """The inventory's row for this grid, or None when the hardware has no
        grid of this name any more."""
        self._entry = entry
        self._has_loader = has_loader
        self.refresh()

    @property
    def is_present(self) -> bool:
        return self._entry is not None and self._entry.present

    @property
    def in_beam(self) -> bool:
        return self._entry is not None and self._entry.in_beam

    def set_controls_enabled(self, enabled: bool) -> None:
        """The host's lockout while a workflow owns the loader."""
        self._controls_enabled = enabled
        self.refresh()

    # -- drawing ---------------------------------------------------------------

    def refresh(self) -> None:
        grid = self.grid
        present = self.is_present
        self._name_label.setText(grid.name)
        self._name_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; background: transparent; "
            f"color: {NEUTRAL_200 if present else NEUTRAL_550};"
        )
        self._card.setToolTip(grid.description or "")

        text, colour = grid_headline(grid)
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"font-size: 11px; color: {colour}; background: transparent;"
        )

        for old in self._chip_widgets:
            self._chips.removeWidget(old)
            old.deleteLater()
        self._chip_widgets = []
        chips: List[Tuple[str, str]] = []
        if not present:
            chips.append(("not present", NEUTRAL_700))
        elif self.in_beam:
            chips.append(("Loaded", OK_COLOR))
        for label, chip_colour in chips:
            widget = chip(label, chip_colour)
            # A pill: the radius the helper draws is half of this height.
            widget.setFixedHeight(20)
            self._chips.addWidget(widget)
            self._chip_widgets.append(widget)
        slot = f"slot {self._entry.index + 1:02d}" if present else "not in the holder"
        self._status_label.setToolTip(f"{text} · {slot}")

        icon, icon_colour, verdict = _QUALITY_ICON[grid.quality]
        self._btn_quality.setIcon(fibsem_icon(icon, color=icon_colour))
        self._btn_quality.setToolTip(
            f"Quality: {verdict}. A person's verdict; no task sets it."
        )

        image = _thumbnail_image(
            grid_thumbnail_path(self.experiment, grid) if self.experiment else None,
            self._thumb_label.width(),
            self._thumb_label.height(),
        )
        if image is not None:
            self._thumb_label.setText("")
            self._thumb_label.setPixmap(QPixmap.fromImage(image))
        else:
            self._thumb_label.setPixmap(QPixmap())
            self._thumb_label.setText("")

        # Load brings a grid into the beam; only with a loader, and only for a grid
        # that is present and not already there. Unload for the one that is.
        can = self._controls_enabled
        self._action_load.setVisible(self._has_loader and not self.in_beam)
        self._action_load.setEnabled(present and can)
        self._action_unload.setVisible(self._has_loader and self.in_beam)
        self._action_unload.setEnabled(can)
        self._action_rename.setEnabled(can)

    @property
    def status_text(self) -> str:
        return self._status_label.text()

    def set_selected(self, selected: bool) -> None:
        self._card.setStyleSheet(_CARD_SELECTED_STYLE if selected else _CARD_STYLE)

    # -- events ----------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.grid)
        super().mousePressEvent(event)

    def _on_quality_clicked(self) -> None:
        menu = QMenu(self)
        actions = {}
        for quality, (icon, colour, verdict) in _QUALITY_ICON.items():
            actions[menu.addAction(fibsem_icon(icon, color=colour), verdict)] = quality
        chosen = menu.exec_(
            self._btn_quality.mapToGlobal(self._btn_quality.rect().bottomLeft())
        )
        if chosen in actions:
            self.set_quality(actions[chosen])

    def set_quality(self, quality: GridQuality) -> None:
        if quality is self.grid.quality:
            return
        self.grid.quality = quality
        self.refresh()
        self.quality_changed.emit(self.grid)

    def _on_rename(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Rename grid", "Name:", text=self.grid.name
        )
        name = name.strip()
        if ok and name and name != self.grid.name:
            self.rename_requested.emit(self.grid, name)

    def _on_remove(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove grid",
            f"Remove '{self.grid.name}' from the experiment? Its files are kept; "
            "its lamellae lose their grid.",
            QMessageBox.Yes,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_requested.emit(self.grid)


class GridCardContainer(QWidget):
    """A one-column strip of grid cards, in the order the records were added."""

    grid_selected = pyqtSignal(object)  # GridRecord | None
    quality_changed = pyqtSignal(object)  # GridRecord
    load_requested = pyqtSignal(object)  # GridRecord
    unload_requested = pyqtSignal(object)  # GridRecord
    rename_requested = pyqtSignal(object, str)  # GridRecord, new name
    remove_requested = pyqtSignal(object)  # GridRecord

    def __init__(self, parent: Optional[QWidget] = None, mode: str = MODE_COZY) -> None:
        super().__init__(parent)
        self._cards: Dict[str, GridCardWidget] = {}  # grid.id -> card
        self._selected_id: Optional[str] = None
        self._mode = mode if mode in CARD_MODES else MODE_COZY
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(12)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def add_grid(
        self, grid: GridRecord, experiment: Optional[Experiment] = None
    ) -> GridCardWidget:
        card = GridCardWidget(grid, experiment, mode=self._mode)
        card.clicked.connect(self._on_card_clicked)
        card.quality_changed.connect(self.quality_changed)
        card.load_requested.connect(self.load_requested)
        card.unload_requested.connect(self.unload_requested)
        card.rename_requested.connect(self.rename_requested)
        card.remove_requested.connect(self.remove_requested)
        self._cards[grid.id] = card
        self._layout.addWidget(card)
        return card

    def set_mode(self, mode: str) -> None:
        """Switch every card, and any added later."""
        if mode not in CARD_MODES:
            return
        self._mode = mode
        for card in self._cards.values():
            card.set_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def remove_grid(self, grid: GridRecord) -> None:
        card = self._cards.pop(grid.id, None)
        if card is not None:
            self._layout.removeWidget(card)
            card.deleteLater()
            if self._selected_id == grid.id:
                self._selected_id = None

    def card_for(self, grid: GridRecord) -> Optional[GridCardWidget]:
        return self._cards.get(grid.id)

    @property
    def cards(self) -> List[GridCardWidget]:
        return list(self._cards.values())

    @property
    def selected_grid(self) -> Optional[GridRecord]:
        card = self._cards.get(self._selected_id) if self._selected_id else None
        return card.grid if card is not None else None

    def clear(self) -> None:
        for card in self._cards.values():
            self._layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        # Ids are stable and the host clears then re-adds the same set, so a
        # surviving id would name a card drawn unselected while this thinks it is
        # selected (the lamella strip's FIB-578).
        self._selected_id = None

    def select_grid(self, grid: Optional[GridRecord]) -> None:
        """Select programmatically. Does not emit grid_selected."""
        if self._selected_id and self._selected_id in self._cards:
            self._cards[self._selected_id].set_selected(False)
        self._selected_id = None
        if grid is not None and grid.id in self._cards:
            self._selected_id = grid.id
            self._cards[grid.id].set_selected(True)

    def refresh_all(self) -> None:
        for card in self._cards.values():
            card.refresh()

    def set_controls_enabled(self, enabled: bool) -> None:
        for card in self._cards.values():
            card.set_controls_enabled(enabled)

    def _on_card_clicked(self, grid: GridRecord) -> None:
        if self._selected_id == grid.id:
            self.select_grid(None)
            self.grid_selected.emit(None)
        else:
            self.select_grid(grid)
            self.grid_selected.emit(grid)
