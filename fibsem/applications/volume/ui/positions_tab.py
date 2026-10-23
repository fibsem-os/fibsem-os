"""The volume window's Positions tab: a card per position, and the selected
position's stored fields (Position) and its slices as one stack, scrubbed with
a slider (Slices).

What the cards and slices show comes from the App Server's own reads on ``VolumeAppContext``,
the ones the web Positions page makes: the experiment summary for the cards
and ``item_stack`` for the slices.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.volume.server import VolumeAppContext
from fibsem.applications.volume.tasks.acquisition import SLICE_ROLE
from fibsem.applications.volume.tasks.milling import SLICE_FIB_ROLE
from fibsem.applications.volume.ui.position_editor import PositionEditor
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    ERROR_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.thumbnails import cached_thumbnail, load_and_resize
from fibsem.ui.utils import arr_to_qimage
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.util.timestamps import format_time
from fibsem.workflows.outputs import StackSlice, item_outputs
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession

__all__ = ["PositionsTab", "SliceScrubber"]

THUMB_W, THUMB_H = 120, 80
CARD_HEIGHT = 96
LIST_WIDTH = 340
# The full view's width: enough to judge a slice, small enough to step through.
VIEW_WIDTH = 1024
PLAY_MS = 250
TOGGLE_STYLE = (
    f"QPushButton:checked {{ background: {ACCENT_COLOR}; color: {TEXT_STRONG_COLOR};"
    f" border: 1px solid {ACCENT_COLOR}; }}"
)


def _decode(path: str, width: int) -> QImage:
    try:
        data, _px, _info = load_and_resize(path, width)
    except Exception:  # noqa: BLE001 - a file that can't be read shows nothing
        logging.debug(f"could not load {path}", exc_info=True)
        return QImage()
    return arr_to_qimage(data)


def state_line(position) -> tuple:
    """What a card says the position is doing, and its colour: the task
    running and its slice, else why it failed, else its last task and when."""
    state = position.task_state
    if state is not None and state.status is TaskStatus.InProgress:
        where = f" · slice {state.iteration + 1}" if state.iteration is not None else ""
        return f"{state.name}{where}", ACCENT_COLOR
    failed = next(
        (s for s in reversed(position.task_history) if s.status is TaskStatus.Failed),
        None,
    )
    if failed is not None:
        return failed.status_message or f"{failed.name} failed", ERROR_COLOR
    last = position.task_history[-1] if position.task_history else None
    if last is None:
        return "Not started", TEXT_MUTED_COLOR
    when = format_time(last.end_timestamp, "%H:%M")
    return f"{last.name} · {when}" if when else last.name, TEXT_MUTED_COLOR


class PositionCard(QWidget):
    """One position: its latest slice, name, slices cut and state."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setStyleSheet("background: #000;")
        self.name = QLabel()
        self.name.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {TEXT_STRONG_COLOR};"
            " background: transparent;"
        )
        self.slices = QLabel()
        self.state = ElidedLabel()
        text = QVBoxLayout()
        text.setSpacing(2)
        for label in (self.name, self.slices, self.state):
            text.addWidget(label)
        for label in (self.slices, self.state):
            label.setStyleSheet("background: transparent;")
        text.addStretch(1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.thumb)
        layout.addLayout(text, 1)

    def show_position(self, position, row, latest: Optional[str]) -> None:
        self.name.setText(position.name)
        self.slices.setText(f"{row.slices_cut} of {row.iterations} slices")
        words, colour = state_line(position)
        self.state.setText(words)
        self.state.setStyleSheet(f"color: {colour}; background: transparent;")
        image = (
            cached_thumbnail(
                latest, THUMB_W, THUMB_H, lambda: _decode(latest, THUMB_W * 2)
            )
            if latest
            else None
        )
        self.thumb.setPixmap(QPixmap.fromImage(image) if image else QPixmap())


class SliceScrubber(QWidget):
    """A position's slices as one stack: the slice shown, a slider with a step
    per slice, play, and Follow to stay on the newest as they arrive. SEM
    Image and FIB Image switch stacks when the milling kept a FIB image."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stacks: Dict[str, List[StackSlice]] = {}
        self._paths: Dict[str, str] = {}
        self._iterations = 0
        self._role = SLICE_ROLE

        self.sem_button = QPushButton("SEM Image")
        self.fib_button = QPushButton("FIB Image")
        roles = QButtonGroup(self)
        for button, role in (
            (self.sem_button, SLICE_ROLE),
            (self.fib_button, SLICE_FIB_ROLE),
        ):
            button.setCheckable(True)
            button.setStyleSheet(TOGGLE_STYLE)
            roles.addButton(button)
            button.clicked.connect(lambda _c, role=role: self.show_role(role))
        self.sem_button.setChecked(True)

        self.image = QLabel("No slices yet")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(320, 240)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image.setStyleSheet(f"background: #000; color: {TEXT_MUTED_COLOR};")
        self.hfw = QLabel(self.image)
        self.hfw.setStyleSheet(f"color: {TEXT_STRONG_COLOR}; background: transparent;")

        self.prev_button = QPushButton("‹")
        self.play_button = QPushButton("▶")
        self.next_button = QPushButton("›")
        for button, tip in (
            (self.prev_button, "Previous slice"),
            (self.play_button, "Play"),
            (self.next_button, "Next slice"),
        ):
            button.setFixedWidth(32)
            button.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setPageStep(1)
        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.readout = QLabel()

        toggles = QHBoxLayout()
        toggles.addStretch(1)
        toggles.addWidget(self.sem_button)
        toggles.addWidget(self.fib_button)
        controls = QHBoxLayout()
        for widget in (self.prev_button, self.play_button, self.next_button):
            controls.addWidget(widget)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.follow)
        controls.addWidget(self.readout)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toggles)
        layout.addWidget(self.image, 1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(PLAY_MS)
        self._timer.timeout.connect(self._play_step)
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
        self.play_button.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._show_slice)
        self.slider.sliderMoved.connect(lambda _v: self.follow.setChecked(False))
        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def slices(self) -> List[StackSlice]:
        return self._stacks.get(self._role, [])

    @property
    def index(self) -> int:
        return self.slider.value()

    def set_stacks(
        self,
        stacks: Dict[str, List[StackSlice]],
        paths: Dict[str, str],
        iterations: int,
        keep: bool = False,
    ) -> None:
        """Show these stacks, by role, with each filename's path. *keep* stays
        on the slice shown (a refresh of the same position); Follow moves to
        the newest."""
        index = self.index if keep else 0
        self._stacks = stacks
        self._paths = paths
        self._iterations = iterations
        self.fib_button.setVisible(bool(stacks.get(SLICE_FIB_ROLE)))
        self.sem_button.setVisible(bool(stacks.get(SLICE_FIB_ROLE)))
        if not stacks.get(SLICE_FIB_ROLE):
            self._role = SLICE_ROLE
            self.sem_button.setChecked(True)
        self._set_range(index)

    def show_role(self, role: str) -> None:
        self._role = role
        self._set_range(self.index)

    def _set_range(self, index: int) -> None:
        count = len(self.slices)
        if self.follow.isChecked():
            index = count - 1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(count - 1, 0))
        self.slider.setValue(min(max(index, 0), max(count - 1, 0)))
        self.slider.blockSignals(False)
        for widget in (
            self.slider,
            self.prev_button,
            self.play_button,
            self.next_button,
        ):
            widget.setEnabled(count > 1)
        self._show_slice(self.slider.value())

    def step(self, delta: int) -> None:
        self.follow.setChecked(False)
        self.slider.setValue(self.index + delta)

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("▶")
            return
        self.follow.setChecked(False)
        if self.index >= self.slider.maximum():
            self.slider.setValue(0)
        self._timer.start()
        self.play_button.setText("❚❚")

    def _play_step(self) -> None:
        if self.index >= self.slider.maximum():
            self._toggle_play()
            return
        self.slider.setValue(self.index + 1)

    def _show_slice(self, index: int) -> None:
        slices = self.slices
        if not slices:
            self.image.setPixmap(QPixmap())
            self.image.setText("No slices yet")
            self.hfw.clear()
            self.readout.clear()
            return
        entry = slices[index]
        path = self._paths.get(entry.filename)
        image = (
            cached_thumbnail(
                path, *self._view_size(), lambda: _decode(path, VIEW_WIDTH)
            )
            if path
            else None
        )
        self.image.setPixmap(QPixmap.fromImage(image) if image else QPixmap())
        self.hfw.setText(f"HFW {entry.hfw * 1e6:.1f} µm" if entry.hfw else "")
        self.hfw.adjustSize()
        self.hfw.move(
            self.image.width() - self.hfw.width() - 8,
            self.image.height() - self.hfw.height() - 6,
        )
        self.readout.setText(self.readout_text(entry))

    def _view_size(self) -> tuple:
        # cached_thumbnail fills its box; the view fits the image inside it
        size = self.image.size()
        width = min(size.width(), int(size.height() * 1.5))
        return width, int(width / 1.5)

    def readout_text(self, entry: StackSlice) -> str:
        number = entry.iteration + 1 if entry.iteration is not None else self.index + 1
        parts = [f"Slice {number} of {self._iterations}"]
        if entry.z is not None:
            parts.append(f"z {entry.z * 1e6:.1f} µm")
        when = format_time(entry.acquired_at, "%H:%M")
        if when:
            parts.append(when)
        return " · ".join(parts)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self.step(-1 if event.key() == Qt.Key_Left else 1)
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        self.step(-1 if event.angleDelta().y() > 0 else 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show_slice(self.index)


class PositionsTab(QWidget):
    def __init__(self, session: "AppSession", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.context = VolumeAppContext(session)
        self.list = QListWidget()
        self.list.setFixedWidth(LIST_WIDTH)
        self.list.setSpacing(3)
        self.title = QLabel()
        self.title.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {TEXT_STRONG_COLOR};"
        )
        self.editor = PositionEditor(session)
        self.scrubber = SliceScrubber()
        self.views = QTabWidget()
        self.views.addTab(self.editor, "Position")
        self.views.addTab(self.scrubber, "Slices")

        right = QVBoxLayout()
        right.addWidget(self.title)
        right.addWidget(self.views, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(right, 1)
        self.list.currentItemChanged.connect(lambda *_: self._show_selected(False))
        self.editor.edited.connect(self.refresh)
        self.refresh()

    @property
    def selected(self) -> Optional[str]:
        item = self.list.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def refresh(self) -> None:
        """Draw the cards from the experiment summary, keeping the selection,
        and the selected position's stack."""
        experiment = self.session.experiment
        selected = self.selected
        summary = self.context.experiment_summary()
        self.list.blockSignals(True)
        self.list.clear()
        for row in summary.items:
            position = experiment.get_position_by_name(row.item_name)
            card = PositionCard()
            card.show_position(position, row, self._latest(position))
            item = QListWidgetItem()
            item.setData(Qt.UserRole, row.item_name)
            item.setSizeHint(QSize(0, CARD_HEIGHT))
            self.list.addItem(item)
            self.list.setItemWidget(item, card)
        names = [row.item_name for row in summary.items]
        if names:
            self.list.setCurrentRow(names.index(selected) if selected in names else 0)
        self.list.blockSignals(False)
        self._show_selected(keep=True)

    def _latest(self, position) -> Optional[str]:
        files = item_outputs(self.session.experiment, position, SLICE_ROLE)
        return files[-1] if files else None

    def _show_selected(self, keep: bool) -> None:
        name = self.selected
        self.title.setText(name or "")
        self.editor.show_position(name)
        if name is None:
            self.scrubber.set_stacks({}, {}, 0)
            return
        experiment = self.session.experiment
        position = experiment.get_position_by_name(name)
        stacks = {
            role: self.context.item_stack(name, role).slices
            for role in (SLICE_ROLE, SLICE_FIB_ROLE)
        }
        paths = {
            os.path.basename(path): path
            for path in item_outputs(experiment, position, SLICE_ROLE, SLICE_FIB_ROLE)
        }
        self.scrubber.set_stacks(stacks, paths, position.iterations, keep=keep)
