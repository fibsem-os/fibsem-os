"""Grid card widget — the experiment's grid records as cards.

The card analogue of LamellaCardWidget, for grid workflow records
(``Experiment.grids``). A card shows an overview thumbnail (an "empty grid"
placeholder until the overview task runs), the loader magazine slot badge, the
grid name, an "in beam" indicator, and a status line. GridCardContainer lays the
cards out and owns selection, mirroring LamellaCardContainer.

GridListWidget (compact rows) is kept alongside this for callers that prefer a
list; the Grids tab uses the cards.

Phase 4a: thumbnail placeholder + slot + status + remove. View/Run/overview
image arrive with later phases.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.ui import stylesheets

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import GridRecord

_CARD_WIDTH = 300
_THUMB_HEIGHT = 150
_BTN_SIZE = 24

_GREEN = "#4caf50"
_RED = "#e24b4a"
_GRAY = "#808080"
_AMBER = "#e0a030"

_CARD_STYLE = "QFrame#GridCard { background: #2b2d31; border: 1px solid #3a3d42; border-radius: 8px; }"
_CARD_SELECTED_STYLE = "QFrame#GridCard { background: #2b2d31; border: 2px solid #007ACC; border-radius: 8px; }"
_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; border-radius: 4px; padding: 1px; }"
    " QToolButton:hover { background: rgba(255,255,255,30); }"
)


def _status(record: "GridRecord"):
    """(text, color) status summary for a grid record."""
    if record.task_state.status.name == "InProgress":
        task = record.task_state.name
        return (f"Running — {task}" if task else "Running…"), _AMBER
    if record.is_failure:
        task = record.task_state.name
        return (f"Failed — {task}" if task else "Failed"), _RED
    n = len(record.completed_tasks)
    if n:
        return f"{n} task{'s' if n != 1 else ''} complete", _GREEN
    return "Not started", _GRAY


class _GridThumbnail(QWidget):
    """Overview thumbnail; draws an 'empty grid' (circular mesh) until a real
    overview image is set."""

    def __init__(self, w: int, h: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(w, h)
        self._pixmap: Optional[QPixmap] = None

        # "in beam" pill, top-left over the thumbnail (hidden unless loaded)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self._beam_pill = QLabel("● In microscope")
        self._beam_pill.setStyleSheet(
            f"background: rgba(27,58,42,220); color: {_GREEN}; font-size: 11px;"
            " font-weight: 500; padding: 1px 7px; border-radius: 8px;"
        )
        self._beam_pill.setVisible(False)
        row.addWidget(self._beam_pill)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        self.update()

    def set_in_beam(self, in_beam: bool) -> None:
        self._beam_pill.setVisible(in_beam)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())

        bg = QPainterPath()
        bg.addRoundedRect(rect, 4, 4)
        p.fillPath(bg, QColor("#1a1b1e"))

        if self._pixmap is not None:
            p.setClipPath(bg)
            p.drawPixmap(self.rect(), self._pixmap)
            p.setClipping(False)
            p.end()
            return

        # empty-grid placeholder: a circular EM-grid mesh
        d = min(rect.width(), rect.height()) - 28
        circle = QRectF(
            rect.center().x() - d / 2, rect.center().y() - d / 2, d, d
        )
        clip = QPainterPath()
        clip.addEllipse(circle)
        p.save()
        p.setClipPath(clip)
        p.setPen(QPen(QColor(127, 127, 127, 48), 1))
        step = 8
        x = circle.left()
        while x <= circle.right():
            p.drawLine(QPointF(x, circle.top()), QPointF(x, circle.bottom()))
            x += step
        y = circle.top()
        while y <= circle.bottom():
            p.drawLine(QPointF(circle.left(), y), QPointF(circle.right(), y))
            y += step
        p.restore()
        p.setPen(QPen(QColor("#5a5d62"), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(circle)
        p.end()


class GridCardWidget(QWidget):
    """A single grid record card: thumbnail, slot badge, name, status, actions."""

    clicked = pyqtSignal(object)           # GridRecord
    remove_requested = pyqtSignal(object)  # GridRecord
    load_requested = pyqtSignal(object)    # GridRecord (load into the working slot)
    unload_requested = pyqtSignal(object)  # GridRecord (retract from the beam)

    def __init__(self, record: "GridRecord", slot_label: str = "",
                 in_beam: bool = False, loader_present: bool = True,
                 thumbnail_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.record = record
        self._in_beam = in_beam
        self._loader_present = loader_present
        self.setFixedWidth(_CARD_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self._card = QFrame()
        self._card.setObjectName("GridCard")
        self._card.setStyleSheet(_CARD_STYLE)
        self._card.setFixedWidth(_CARD_WIDTH - 8)
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        self._thumb = _GridThumbnail(_CARD_WIDTH - 8 - 16, _THUMB_HEIGHT)
        self._thumb.set_in_beam(in_beam)
        # show the persisted overview thumbnail (small PNG) if present, else mesh
        if thumbnail_path and os.path.exists(thumbnail_path):
            pm = QPixmap(thumbnail_path)
            if not pm.isNull():
                self._thumb.set_pixmap(pm)
        card_layout.addWidget(self._thumb)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        card_layout.addWidget(sep)

        # name row: slot badge + name + actions
        name_row = QHBoxLayout()
        name_row.setSpacing(6)

        self._slot_badge = QLabel(slot_label or "—")
        self._slot_badge.setToolTip("Loader magazine slot")
        badge_color = "#2d3f5c" if in_beam else "#3a3d42"
        self._slot_badge.setStyleSheet(
            f"background: {badge_color}; color: #d6d6d6; font-size: 11px;"
            " font-weight: 500; padding: 1px 7px; border-radius: 4px;"
        )
        name_row.addWidget(self._slot_badge)

        self._name_label = QLabel(record.name)
        self._name_label.setStyleSheet("font-weight: 500; font-size: 14px; background: transparent;")
        name_row.addWidget(self._name_label, 1)

        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_actions.setStyleSheet(_BTN_STYLE + " QToolButton::menu-indicator { image: none; }")
        self._btn_actions.setIcon(
            QIconifyIcon("mdi:dots-vertical", color=stylesheets.GRAY_ICON_COLOR)
        )
        self._btn_actions.clicked.connect(self._on_actions_clicked)
        name_row.addWidget(self._btn_actions)

        card_layout.addLayout(name_row)

        text, color = _status(record)
        self._status_label = QLabel(text)
        self._status_label.setStyleSheet(f"font-size: 11px; background: transparent; color: {color};")
        card_layout.addWidget(self._status_label)

        outer.addWidget(self._card)

    # --- selection ---

    def set_selected(self, selected: bool) -> None:
        self._card.setStyleSheet(_CARD_SELECTED_STYLE if selected else _CARD_STYLE)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.record)
        super().mousePressEvent(event)

    # --- actions ---

    def _on_actions_clicked(self) -> None:
        menu = QMenu(self)
        act_load = None
        if self._loader_present:  # load/unload only meaningful with an autoloader
            if self._in_beam:
                act_load = menu.addAction(
                    QIconifyIcon("mdi:logout-variant", color=stylesheets.GRAY_ICON_COLOR),
                    "Unload from microscope",
                )
            else:
                act_load = menu.addAction(
                    QIconifyIcon("mdi:login", color=stylesheets.GRAY_ICON_COLOR),
                    "Load into microscope",
                )
            menu.addSeparator()
        act_remove = menu.addAction(
            QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR),
            "Remove from experiment",
        )
        chosen = menu.exec_(
            self._btn_actions.mapToGlobal(self._btn_actions.rect().bottomLeft())
        )
        if chosen == act_remove:
            self._on_remove_clicked()
        elif act_load is not None and chosen == act_load:
            signal = self.unload_requested if self._in_beam else self.load_requested
            signal.emit(self.record)

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self, "Remove Grid", f"Remove '{self.record.name}' from the experiment?",
            QMessageBox.Yes, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_requested.emit(self.record)


class GridCardContainer(QWidget):
    """Header (title + count + Add from Loader) over a scrollable strip of grid
    cards. Drop-in for GridListWidget: same signals + set_grids()."""

    grid_selected = pyqtSignal(object)            # GridRecord | None
    add_from_loader_requested = pyqtSignal()
    remove_requested = pyqtSignal(object)         # GridRecord
    load_requested = pyqtSignal(object)           # GridRecord
    unload_requested = pyqtSignal(object)         # GridRecord

    def __init__(self, columns: int = 1, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._n_cols = max(1, columns)
        self._cards: List[GridCardWidget] = []
        self._selected_name: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # header
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

        # scrollable card grid
        self._inner = QWidget()
        self._grid = QGridLayout(self._inner)
        self._grid.setSpacing(12)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._inner)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        outer.addWidget(self._scroll)

        self._empty = QLabel("No grids yet — add them from the loader.")
        self._empty.setStyleSheet("color: #606060; font-style: italic; padding: 8px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._empty)
        self._update_empty()

    # --- public API (mirrors GridListWidget) ---

    def set_grids(self, grids: List["GridRecord"], slot_labels: Optional[dict] = None,
                  beam_names: Optional[set] = None, loader_present: bool = True,
                  thumbnails: Optional[dict] = None) -> None:
        """Rebuild the cards, preserving selection by name."""
        slot_labels = slot_labels or {}
        beam_names = beam_names or set()
        thumbnails = thumbnails or {}
        prev = self._selected_name

        for card in self._cards:
            card.deleteLater()
        self._cards.clear()

        for record in grids:
            card = GridCardWidget(
                record,
                slot_label=slot_labels.get(record.name, ""),
                in_beam=record.name in beam_names,
                loader_present=loader_present,
                thumbnail_path=thumbnails.get(record.name),
            )
            card.clicked.connect(self._on_card_clicked)
            card.remove_requested.connect(self.remove_requested)
            card.load_requested.connect(self.load_requested)
            card.unload_requested.connect(self.unload_requested)
            self._cards.append(card)
        self._rebuild_grid()

        self._count.setText(f"· {len(grids)}")
        self._update_empty()

        # restore selection by name, else select first, else none
        names = [c.record.name for c in self._cards]
        if names:
            target = prev if prev in names else names[0]
            self.select_grid(target, emit=True)
        else:
            self._selected_name = None
            self.grid_selected.emit(None)

    @property
    def selected_grid(self) -> Optional["GridRecord"]:
        for card in self._cards:
            if card.record.name == self._selected_name:
                return card.record
        return None

    def select_grid(self, name: Optional[str], emit: bool = False) -> None:
        self._selected_name = name
        for card in self._cards:
            card.set_selected(card.record.name == name)
        if emit:
            self.grid_selected.emit(self.selected_grid)

    # --- internals ---

    def _on_card_clicked(self, record: "GridRecord") -> None:
        if self._selected_name == record.name:
            self.select_grid(None)
            self.grid_selected.emit(None)
        else:
            self.select_grid(record.name)
            self.grid_selected.emit(record)

    def _rebuild_grid(self) -> None:
        while self._grid.count():
            self._grid.takeAt(0)
        for i, card in enumerate(self._cards):
            row, col = divmod(i, self._n_cols)
            self._grid.addWidget(card, row, col)

    def _update_empty(self) -> None:
        empty = not self._cards
        self._empty.setVisible(empty)
        self._scroll.setVisible(not empty)
