"""A card per grid record: the experiment's view of a grid.

The same card as a lamella's, so the two tabs read alike: a thumbnail, the name
over a status line, a small icon for the person's verdict, and an actions menu.
What the hardware says (loaded, or gone from the magazine) is a chip. The
card's object is the ``GridRecord``, not a hardware slot: the inventory entry
arrives on every refresh and is only drawn.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui.common.cards import (
    BTN_SIZE,
    CARD_BUTTON_STYLE,
    COZY_THUMB_H,
    COZY_THUMB_W,
    THUMB_H,
    THUMB_PADDING,
    THUMB_W,
    CardContainer,
    RecordCard,
)
from fibsem.applications.autolamella.ui.common.lamella_display import (
    VERDICT_LOOK,
    add_verdict_actions,
    mark_current_verdict,
)
from fibsem.config import MODE_COMPACT, MODE_COZY
from fibsem.structures.settings import GridInventoryEntry, GridSlotState
from fibsem.theme import tokens
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    ERROR_COLOR,
    NEUTRAL_200,
    NEUTRAL_550,
    NEUTRAL_700,
    NEUTRAL_900,
    OK_COLOR,
)
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.thumbnails import cached_thumbnail
from fibsem.ui.widgets.custom_widgets import ElidedLabel, chip
from fibsem.ui.widgets.sample_loader_widget import ICON_LOAD, ICON_UNLOAD
from fibsem.workflows.grid.manager import (
    EXCHANGE_ENTRY_NAMES as _EXCHANGE_ENTRY_NAMES,
)
from fibsem.workflows.grid.manager import (
    LOAD_ENTRY_NAME as _LOAD_ENTRY_NAME,
)
from fibsem.workflows.grid.naming import NAME_FIXED_REASON, grid_has_run
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.outputs import latest_item_output
from fibsem.workflows.quality import Verdict
from fibsem.workflows.status import TaskStatus

# Which overview a card shows, in order of preference.
_THUMBNAIL_ROLES = (
    "overview_sem_thumbnail",
    "overview_fib_thumbnail",
    "overview_fm_thumbnail",
)


def grid_headline(grid: GridRecord) -> Tuple[str, str]:
    """One line on how the grid's last run went, and its colour.

    Read off the history, never off the quality: whether the tasks ran is a
    different question from whether the grid is any good. The most recent load
    entry starts the run being described; task entries after it are the run.

    Empty when no task has run since the last load: a card with nothing to say
    says nothing, rather than "Not run" on every grid of a fresh magazine.
    """
    state = grid.task_state
    if state.status is TaskStatus.InProgress:
        return f"Running {state.name}", ACCENT_COLOR
    history = grid.task_history
    if not history:
        return "", NEUTRAL_550
    start = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i].name == _LOAD_ENTRY_NAME:
            start = i
            break
    load = history[start] if history[start].name == _LOAD_ENTRY_NAME else None
    tasks = [t for t in history[start:] if t.name not in _EXCHANGE_ENTRY_NAMES]
    if load is not None and load.status is TaskStatus.Failed:
        return "Load failed", ERROR_COLOR
    if not tasks:
        return "", NEUTRAL_550
    failed = sum(1 for t in tasks if t.status is TaskStatus.Failed)
    cancelled = sum(1 for t in tasks if t.status is TaskStatus.Cancelled)
    if failed:
        return f"{failed} task{'s' if failed != 1 else ''} failed", ERROR_COLOR
    if cancelled:
        return "Cancelled", tokens.DEFECT_ORANGE_COLOR
    # The lamella card's line: the last completed task and when, not a verdict.
    last = tasks[-1]
    return f"{last.name} ({last.completed_at})", NEUTRAL_550


def grid_thumbnail_path(experiment: Experiment, grid: GridRecord) -> Optional[str]:
    """The latest overview thumbnail a run recorded for the grid, SEM first."""
    for role in _THUMBNAIL_ROLES:
        path = latest_item_output(experiment, grid, role)
        if path is not None:
            return path
    return None


class GridCardWidget(RecordCard):
    """One grid record: thumbnail | name over status, chips | quality | actions."""

    quality_changed = pyqtSignal(object)  # GridRecord
    load_requested = pyqtSignal(object)  # GridRecord
    unload_requested = pyqtSignal(object)  # GridRecord
    rename_requested = pyqtSignal(object, str)  # GridRecord, new name
    # The operator's note on the grid: what the report prints beside the verdict.
    note_requested = pyqtSignal(object, str)  # GridRecord, the note
    remove_requested = pyqtSignal(object)  # GridRecord

    def __init__(
        self,
        grid: GridRecord,
        experiment: Optional[Experiment] = None,
        mode: str = MODE_COZY,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(grid, "GridCard", mode, parent)
        self.grid = grid
        self.experiment = experiment
        self._entry: Optional[GridInventoryEntry] = None
        self._has_loader = False
        self._controls_enabled = True
        self._edits_enabled = True

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
        self._btn_quality.setFixedSize(BTN_SIZE, BTN_SIZE)
        self._btn_quality.setStyleSheet(CARD_BUTTON_STYLE)
        self._btn_quality.clicked.connect(self._on_quality_clicked)

        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(BTN_SIZE, BTN_SIZE)
        self._btn_actions.setStyleSheet(CARD_BUTTON_STYLE)
        self._btn_actions.setIcon(
            fibsem_icon("mdi:dots-horizontal", color=tokens.GRAY_ICON_COLOR)
        )
        self._btn_actions.setToolTip("Actions")
        self._btn_actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        # Rename says why when it is greyed out (a grid that has run).
        menu.setToolTipsVisible(True)
        self._action_load = menu.addAction(
            fibsem_icon(ICON_LOAD, color=tokens.GRAY_ICON_COLOR), "Load"
        )
        self._action_unload = menu.addAction(
            fibsem_icon(ICON_UNLOAD, color=tokens.GRAY_ICON_COLOR), "Unload"
        )
        self._action_rename = menu.addAction(
            fibsem_icon("mdi:pencil-outline", color=tokens.GRAY_ICON_COLOR),
            "Rename…",
        )
        self._action_note = menu.addAction(
            fibsem_icon("mdi:note-edit-outline", color=tokens.GRAY_ICON_COLOR),
            "Edit note…",
        )
        self._action_remove = menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=tokens.GRAY_ICON_COLOR),
            "Remove",
        )
        self._action_load.triggered.connect(lambda: self.load_requested.emit(self.grid))
        self._action_unload.triggered.connect(
            lambda: self.unload_requested.emit(self.grid)
        )
        self._action_rename.triggered.connect(self._on_rename)
        self._action_note.triggered.connect(self._on_edit_note)
        self._action_remove.triggered.connect(self._on_remove)
        self._btn_actions.setMenu(menu)

        self._apply_layout()
        self.refresh()

    # -- arrangement -----------------------------------------------------------

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
            self._thumb_label.setFixedSize(COZY_THUMB_W, COZY_THUMB_H)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(THUMB_PADDING, THUMB_PADDING, THUMB_PADDING, 0)
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
            self._thumb_label.setFixedSize(THUMB_W, THUMB_H)
            self._thumb_label.setVisible(self._mode != MODE_COMPACT)
            content = QWidget()
            row = QHBoxLayout(content)
            row.setContentsMargins(
                THUMB_PADDING, THUMB_PADDING, THUMB_PADDING, THUMB_PADDING
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
    def loaded(self) -> bool:
        return self._entry is not None and self._entry.loaded

    def set_controls_enabled(self, enabled: bool) -> None:
        """The host's lockout while a workflow owns the loader."""
        self._controls_enabled = enabled
        self.refresh()

    def set_edits_enabled(self, enabled: bool) -> None:
        """Whether Rename is on. A run doesn't turn it off: the host refuses
        only the grid the run is on (run_edits)."""
        self._edits_enabled = enabled
        self._action_rename.setEnabled(enabled and not grid_has_run(self.grid))

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
        if self._entry is not None and self._entry.state is GridSlotState.UNKNOWN:
            chips.append(("not scanned", NEUTRAL_700))
        elif not present:
            chips.append(("not present", NEUTRAL_700))
        elif self.loaded:
            chips.append(("Loaded", OK_COLOR))
        for label, chip_colour in chips:
            widget = chip(label, chip_colour)
            # A pill: the radius the helper draws is half of this height.
            widget.setFixedHeight(20)
            self._chips.addWidget(widget)
            self._chip_widgets.append(widget)
        slot = f"slot {self._entry.index + 1:02d}" if present else "not in the holder"
        self._status_label.setToolTip(" · ".join(part for part in (text, slot) if part))

        icon, icon_colour, verdict = VERDICT_LOOK[grid.quality.verdict]
        self._btn_quality.setIcon(fibsem_icon(icon, color=icon_colour))
        self._btn_quality.setToolTip(
            f"Quality: {verdict}. A person's verdict; no task sets it."
        )

        path = grid_thumbnail_path(self.experiment, grid) if self.experiment else None
        image = (
            None
            if path is None
            else cached_thumbnail(
                path,
                self._thumb_label.width(),
                self._thumb_label.height(),
                lambda: QImage(path),
            )
        )
        if image is not None:
            self._thumb_label.setText("")
            self._thumb_label.setPixmap(QPixmap.fromImage(image))
        else:
            self._thumb_label.setPixmap(QPixmap())
            self._thumb_label.setText("")

        # Load brings a grid onto the stage; only with a loader, and only for a grid
        # that is present and not already there. Unload for the one that is.
        can = self._controls_enabled
        self._action_load.setVisible(self._has_loader and not self.loaded)
        self._action_load.setEnabled(present and can)
        self._action_unload.setVisible(self._has_loader and self.loaded)
        self._action_unload.setEnabled(can)
        fixed = grid_has_run(grid)
        self._action_rename.setEnabled(self._edits_enabled and not fixed)
        self._action_rename.setToolTip(NAME_FIXED_REASON if fixed else "")

    @property
    def status_text(self) -> str:
        return self._status_label.text()

    # -- events ----------------------------------------------------------------

    def _on_quality_clicked(self) -> None:
        menu = QMenu(self)
        actions = add_verdict_actions(menu)
        mark_current_verdict(actions, self.grid.quality.verdict)
        chosen = menu.exec_(
            self._btn_quality.mapToGlobal(self._btn_quality.rect().bottomLeft())
        )
        for verdict, action in actions.items():
            if chosen is action:
                self.set_quality(verdict)

    def set_quality(self, verdict: Verdict) -> None:
        if verdict is self.grid.quality.verdict:
            return
        self.grid.quality.set_defect(state=verdict)
        self.refresh()
        self.quality_changed.emit(self.grid)

    def _on_rename(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Rename grid", "Name:", text=self.grid.name
        )
        name = name.strip()
        if ok and name and name != self.grid.name:
            self.rename_requested.emit(self.grid, name)

    def _on_edit_note(self) -> None:
        """What the operator makes of the grid, in their words: the note the
        screening report prints beside the verdict, and the card's tooltip."""
        note, ok = QInputDialog.getMultiLineText(
            self,
            "Grid note",
            f"Note on {self.grid.name}, printed in the grid screening report:",
            self.grid.description,
        )
        note = note.strip()
        if ok and note != self.grid.description:
            self.note_requested.emit(self.grid, note)

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


class GridCardContainer(CardContainer):
    """A one-column strip of grid cards, in the order the records were added."""

    grid_selected = pyqtSignal(object)  # GridRecord | None
    quality_changed = pyqtSignal(object)  # GridRecord
    load_requested = pyqtSignal(object)  # GridRecord
    unload_requested = pyqtSignal(object)  # GridRecord
    rename_requested = pyqtSignal(object, str)  # GridRecord, new name
    # The operator's note on the grid: what the report prints beside the verdict.
    note_requested = pyqtSignal(object, str)  # GridRecord, the note
    remove_requested = pyqtSignal(object)  # GridRecord

    def __init__(self, parent: Optional[QWidget] = None, mode: str = MODE_COZY) -> None:
        super().__init__(parent, mode)
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
        card.note_requested.connect(self.note_requested)
        card.remove_requested.connect(self.remove_requested)
        self._cards[grid.id] = card
        self._layout.addWidget(card)
        return card

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

    def select_grid(self, grid: Optional[GridRecord]) -> None:
        """Select programmatically. Does not emit grid_selected."""
        if self._selected_id and self._selected_id in self._cards:
            self._cards[self._selected_id].set_selected(False)
        self._selected_id = None
        if grid is not None and grid.id in self._cards:
            self._selected_id = grid.id
            self._cards[grid.id].set_selected(True)

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
