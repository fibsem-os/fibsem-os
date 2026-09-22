"""Replay an experiment: step or play through everything it recorded.

Laid out like the app's quad view. The SEM and FIB panes show the images the
run acquired, as it acquired them, with milling patterns and spot-burn points
drawn over the frame they were placed on; the FM pane shows its z-stacks; the
stage pane places the experiment's stored overviews and marks where the stage
was. Beside them, the recorded actions as a list: click one to go there, or
play them back at a multiple of real time.

Everything comes from :mod:`fibsem.applications.autolamella.tools.replay`,
which reads the experiment's ``events.jsonl``, or its log if it has none. Nothing here talks to a microscope, and
nothing here writes to the experiment.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, TypeVar

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME
from fibsem.applications.autolamella.tools.replay import (
    EventKind,
    ExperimentReplay,
    ReplayEvent,
    ReplayScene,
    load_replay,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BODY_MUTED_STYLE,
    BODY_STYLE,
    BORDER_COLOR,
    CAPTION_STYLE,
    CONTROL_STYLE,
    DRAFT_POSITION_COLOUR,
    ERROR_COLOR,
    NUMBER_STYLE,
    OK_COLOR,
    ORANGE_COLOR,
    PANEL_COLOR,
    PANEL_TITLE_STYLE,
    PRIMARY_ACCENT,
    PURPLE_COLOR,
    ROW_ALT_COLOR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.milling_overlay import MillingPatternOverlay
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.custom_widgets import ElidedLabel, chip
from fibsem.ui.widgets.stored_overview_canvas import StoredOverviewCanvas

logger = logging.getLogger(__name__)

# Playback speeds, as multiples of real time. The wait between two events is
# their real gap divided by this, then clamped: long idle stretches (a night
# between sessions) would otherwise stall playback, and bursts would blur.
SPEEDS = {"1×": 1.0, "10×": 10.0, "60×": 60.0, "600×": 600.0}
DEFAULT_SPEED = "60×"
_MIN_STEP_MS = 40
_MAX_STEP_MS = 1500

_KIND_LABEL = {
    EventKind.TASK: "Task",
    EventKind.PROMPT: "Prompt",
    EventKind.IMAGE: "Image",
    EventKind.FLUORESCENCE: "FM",
    EventKind.STAGE: "Stage",
    EventKind.MILLING: "Milling",
    EventKind.ALIGNMENT: "Alignment",
    EventKind.CORRELATION: "Correlation",
    EventKind.EDIT: "Edit",
    EventKind.MESSAGE: "Message",
}
_KIND_COLOUR = {
    EventKind.TASK: ACCENT_COLOR,
    EventKind.PROMPT: DRAFT_POSITION_COLOUR,
    EventKind.IMAGE: OK_COLOR,
    EventKind.FLUORESCENCE: SAVED_POSITION_COLOUR,
    EventKind.STAGE: WARN_COLOR,
    EventKind.MILLING: ORANGE_COLOR,
    EventKind.ALIGNMENT: PURPLE_COLOR,
    EventKind.CORRELATION: SELECTED_POSITION_COLOUR,
    EventKind.EDIT: TEXT_STRONG_COLOR,
    EventKind.MESSAGE: TEXT_MUTED_COLOR,
}
_STAGE_MARK = "Stage"
# The item filter's two entries that are not an item. An item is what a
# workflow works on: a lamella, or a grid in a grid workflow.
_ALL_ITEMS = "All items"
_NO_ITEM = "Outside a workflow"
_IMAGE_CACHE_SIZE = 24
_FM_CACHE_SIZE = 3  # a z-stack is ~100 MB

# Columns of the event table. The task is in the header and the row's tooltip.
_COL_TIME, _COL_ITEM, _COL_KIND, _COL_SUMMARY = range(4)

_TABLE_STYLE = f"""
QTableWidget {{
    background-color: {SURFACE_COLOR};
    alternate-background-color: {ROW_ALT_COLOR};
    border: 1px solid {BORDER_COLOR};
    color: {TEXT_COLOR};
    font-size: 11px;
    outline: none;
}}
QTableWidget::item {{ padding: 2px 6px; border: none; }}
QTableWidget::item:selected {{ background-color: {PRIMARY_ACCENT}; color: {TEXT_STRONG_COLOR}; }}
QHeaderView::section {{
    background-color: {PANEL_COLOR};
    color: {TEXT_MUTED_COLOR};
    font-size: 11px;
    padding: 3px 6px;
    border: none;
    border-bottom: 1px solid {BORDER_COLOR};
}}
"""


T = TypeVar("T")


class _ImageCache(Generic[T]):
    """The last few images read, so stepping back and forth does not reread them."""

    def __init__(self, load: Callable[[str], T], size: int) -> None:
        self._load = load
        self._size = size
        self._images: "OrderedDict[Path, Optional[T]]" = OrderedDict()

    def get(self, path: Path) -> Optional[T]:
        if path in self._images:
            self._images.move_to_end(path)
            return self._images[path]
        try:
            image: Optional[T] = self._load(str(path))
        except Exception as e:
            logger.warning(f"Replay could not read {path}: {e}")
            image = None
        self._images[path] = image
        if len(self._images) > self._size:
            self._images.popitem(last=False)
        return image


def _stage_position(pos: Optional[dict]) -> Optional[FibsemStagePosition]:
    if not isinstance(pos, dict):
        return None
    try:
        return FibsemStagePosition(
            name=_STAGE_MARK,
            x=float(pos["x"]),
            y=float(pos["y"]),
            z=float(pos["z"]) if pos.get("z") is not None else None,
            r=float(pos["r"]) if pos.get("r") is not None else None,
            t=float(pos["t"]) if pos.get("t") is not None else None,
            coordinate_system=pos.get("coordinate_system"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _pixel_size(image: Optional[FibsemImage]) -> Optional[float]:
    try:
        return float(image.metadata.pixel_size.x)
    except (AttributeError, TypeError, ValueError):
        return None


def _describe_stage(pos: Optional[FibsemStagePosition]) -> Optional[str]:
    if pos is None:
        return None

    def um(v: Optional[float]) -> str:
        return "–" if v is None else f"{v * 1e6:.1f}"

    def deg(v: Optional[float]) -> str:
        return "–" if v is None else f"{math.degrees(v):.1f}"

    return (
        f"x {um(pos.x)}  y {um(pos.y)}  z {um(pos.z)} µm   "
        f"r {deg(pos.r)}°  t {deg(pos.t)}°"
    )


def _elapsed(start: Optional[datetime], now: datetime) -> str:
    if start is None:
        return ""
    seconds = int((now - start).total_seconds())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    text = f"+{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"+{days}d {text[1:]}" if days else text


def _source(image: ReplayEvent, item: Optional[str]) -> str:
    """Name the item an image was taken of, when it is not the one in view.

    A pane keeps the last image until the next one, so the first steps on a
    new lamella still show the previous lamella's frame.
    """
    return f"{image.item}/" if image.item and image.item != item else ""


def _fixed_height(layout) -> QWidget:
    """A row whose height does not follow its contents (see `event_label`)."""
    row = QWidget()
    row.setLayout(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    row.setFixedHeight(max(row.sizeHint().height(), 28))
    row.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    return row


def _clear_layout(layout) -> None:
    # Hidden as well as deleted: a widget taken out of a layout still paints at
    # its old place until the deferred delete runs, and the next chip lands on it.
    while layout.count():
        w = layout.takeAt(0).widget()
        if w is not None:
            w.hide()
            w.deleteLater()


def find_overviews(root: Path) -> List[Path]:
    """The stitched beam overviews an experiment keeps, oldest first.

    Grid overviews live under ``grids/<grid>/<task>/``; older experiments keep
    ``overview-image-*.tif`` at the root. Fluorescence overviews
    (``*.ome.tiff``) are not included.
    """
    found = list(root.glob("overview*.tif")) + list(
        root.glob("grids/*/*/overview*.tif")
    )
    return sorted(found, key=lambda p: p.stat().st_mtime)


class ExperimentReplayWidget(QWidget):
    """Plays back a recorded experiment. Build with :meth:`from_directory`."""

    def __init__(
        self, replay: ExperimentReplay, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.replay = replay
        self._images: _ImageCache[FibsemImage] = _ImageCache(
            FibsemImage.load, _IMAGE_CACHE_SIZE
        )
        self._fm_images: _ImageCache[FluorescenceImage] = _ImageCache(
            FluorescenceImage.load, _FM_CACHE_SIZE
        )
        self._stages_cache: Dict[object, List[FibsemMillingStage]] = {}
        self._index = -1
        self._visible: List[int] = list(range(len(replay.events)))
        self._shown: Dict[str, Optional[Path]] = {"sem": None, "fib": None, "fm": None}
        self._playing = False
        # What each pane was last given, so a seek touches only the panes whose
        # content changed: every touch is a full matplotlib redraw of that pane.
        self._memo: Dict[str, object] = {}

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

        self._build()
        self._load_overviews()
        self._fill_table()
        if replay.events:
            self.seek(0)

    @classmethod
    def from_directory(
        cls, path: Path, parent: Optional[QWidget] = None
    ) -> "ExperimentReplayWidget":
        return cls(load_replay(Path(path)), parent=parent)

    # ── layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.setWindowTitle(f"Replay — {self.replay.root.name}")
        # Its own window, so nothing above it passes the app's theme down.
        self.setStyleSheet(NAPARI_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header: what this is, and where the playhead is.
        header = QHBoxLayout()
        title = QLabel(self.replay.root.name)
        title.setStyleSheet(PANEL_TITLE_STYLE)
        header.addWidget(title)
        self.meta_label = QLabel(self._meta_text())
        self.meta_label.setStyleSheet(BODY_MUTED_STYLE)
        header.addWidget(self.meta_label)
        header.addStretch(1)
        self.context_row = QHBoxLayout()
        self.context_row.setSpacing(4)
        header.addLayout(self.context_row)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(NUMBER_STYLE)
        header.addWidget(self.time_label)
        layout.addWidget(_fixed_height(header))

        # The instrument, as the app's quad view lays it out: SEM | FIB over FM | stage.
        self.sem_canvas = FibsemImageCanvas()
        self.fib_canvas = FibsemImageCanvas()
        self.milling_overlay = MillingPatternOverlay()
        self.fib_canvas.add_overlay(self.milling_overlay)
        self.spot_overlay = PointsOverlay(color=ORANGE_COLOR, marker="o", size=6)
        self.fib_canvas.add_overlay(self.spot_overlay)
        self.fm_widget = FMCanvasWidget()

        self.stage_view = StoredOverviewCanvas()
        self.stage_view.set_placing_enabled(False)
        self.view_combo = QComboBox()
        self.view_combo.setToolTip("Which stored overview the stage is shown on")
        self.view_combo.currentTextChanged.connect(self._on_view_selected)
        stage_pane = QWidget()
        stage_layout = QVBoxLayout(stage_pane)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(2)
        stage_layout.addWidget(self.view_combo)
        stage_layout.addWidget(self.stage_view, 1)

        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.sem_canvas)
        left.addWidget(self.fm_widget)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.fib_canvas)
        right.addWidget(stage_pane)
        panes = QSplitter(Qt.Orientation.Horizontal)
        panes.addWidget(left)
        panes.addWidget(right)

        # What happened at the playhead, then the transport.
        # One elided line: a label that wraps changes height with the text, and
        # every change of height resizes -- so fully redraws -- all four panes.
        self.event_label = ElidedLabel("")
        self.event_label.setStyleSheet(BODY_STYLE)
        self.event_kind_chip = QHBoxLayout()
        event_row = QHBoxLayout()
        event_row.addLayout(self.event_kind_chip)
        event_row.addWidget(self.event_label, 1)

        style = self.style()
        self.btn_first = self._button(
            style.standardIcon(QStyle.SP_MediaSkipBackward),
            "First event",
            lambda: self._seek_visible(0),
        )
        self.btn_prev = self._button(
            style.standardIcon(QStyle.SP_MediaSeekBackward),
            "Previous event (Left)",
            lambda: self.step(-1),
        )
        self.btn_play = self._button(
            style.standardIcon(QStyle.SP_MediaPlay),
            "Play / pause (Space)",
            self.toggle_play,
        )
        self.btn_next = self._button(
            style.standardIcon(QStyle.SP_MediaSeekForward),
            "Next event (Right)",
            lambda: self.step(1),
        )
        self.btn_last = self._button(
            style.standardIcon(QStyle.SP_MediaSkipForward),
            "Last event",
            lambda: self._seek_visible(len(self._visible) - 1),
        )
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(list(SPEEDS))
        self.speed_combo.setCurrentText(DEFAULT_SPEED)
        self.speed_combo.setToolTip("Playback speed, as a multiple of real time")
        self.speed_combo.setStyleSheet(CONTROL_STYLE)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.position_label = QLabel("")
        self.position_label.setStyleSheet(NUMBER_STYLE)
        self.position_label.setMinimumWidth(90)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        transport = QHBoxLayout()
        for w in (
            self.btn_first,
            self.btn_prev,
            self.btn_play,
            self.btn_next,
            self.btn_last,
        ):
            transport.addWidget(w)
        transport.addWidget(self.speed_combo)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.position_label)

        # The recorded actions: which lamella, then which kinds.
        caption = QLabel("Show")
        caption.setStyleSheet(CAPTION_STYLE)
        self.item_combo = QComboBox()
        self.item_combo.setStyleSheet(CONTROL_STYLE)
        self.item_combo.setToolTip("Only what happened to one lamella or grid")
        self.item_combo.addItem(_ALL_ITEMS, None)
        items: Dict[str, Optional[str]] = {}  # in the order the run reached them
        for e in self.replay.events:
            if e.item and e.item not in items:
                items[e.item] = e.item_type
        for item, item_type in items.items():
            label = f"{item_type.title()} · {item}" if item_type else item
            self.item_combo.addItem(label, item)
        if any(e.item is None for e in self.replay.events):
            self.item_combo.addItem(_NO_ITEM, _NO_ITEM)
        self.item_combo.currentIndexChanged.connect(self._on_filter_changed)
        show_row = QHBoxLayout()
        show_row.addWidget(caption)
        show_row.addWidget(self.item_combo)
        show_row.addStretch(1)

        filters = QGridLayout()
        filters.setHorizontalSpacing(10)
        filters.setVerticalSpacing(2)
        self.filter_boxes: Dict[str, QCheckBox] = {}
        for n, kind in enumerate(EventKind.ALL):
            box = QCheckBox(_KIND_LABEL[kind])  # the count is set by _apply_filter
            box.setStyleSheet(CONTROL_STYLE)
            box.setChecked(True)
            box.toggled.connect(self._on_filter_changed)
            self.filter_boxes[kind] = box
            filters.addWidget(box, n // 4, n % 4)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Item", "Kind", "Action"])
        self.table.setStyleSheet(_TABLE_STYLE)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)  # one line a row; the tooltip has the rest
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head = self.table.horizontalHeader()
        for col, width in ((_COL_TIME, 78), (_COL_ITEM, 110), (_COL_KIND, 80)):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, width)
        head.setSectionResizeMode(_COL_SUMMARY, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(lambda row, _col: self.seek(row))

        player = QWidget()
        player_layout = QVBoxLayout(player)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.addWidget(panes, 1)
        player_layout.addWidget(_fixed_height(event_row))
        player_layout.addWidget(_fixed_height(transport))

        actions = QWidget()
        actions_layout = QVBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addLayout(show_row)
        actions_layout.addLayout(filters)
        actions_layout.addWidget(self.table, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(player)
        split.addWidget(actions)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1100, 560])
        layout.addWidget(split, 1)

        for key, slot in (
            ("Space", self.toggle_play),
            ("Left", lambda: self.step(-1)),
            ("Right", lambda: self.step(1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        if self.replay.records_unreadable:
            self.meta_label.setToolTip(
                f"{self.replay.records_unreadable} records in {self.replay.source} "
                "could not be read and are not shown."
            )

    def _button(self, icon, tooltip: str, slot) -> QPushButton:
        button = QPushButton()
        button.setIcon(icon)
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(slot)
        return button

    def _meta_text(self) -> str:
        events = self.replay.events
        if not events:
            if self.replay.source == EVENTS_FILENAME:
                return (
                    f"Nothing recorded — {EVENTS_FILENAME} has no replayable actions."
                )
            return "Nothing recorded — the log has no replayable actions (it is written at DEBUG level)."
        images = [e for e in events if e.kind == EventKind.IMAGE]
        on_disk = sum(e.image_on_disk for e in images)
        span = _elapsed(self.replay.start, self.replay.end or self.replay.start).lstrip(
            "+"
        )
        return (
            f"{len(events)} actions · {span} · {on_disk} of {len(images)} images on disk"
            f" · from {self.replay.source}"
        )

    def _fill_table(self) -> None:
        events = self.replay.events
        self.table.setRowCount(len(events))
        for row, e in enumerate(events):
            cells = (
                e.time.strftime("%H:%M:%S"),
                e.item or "",
                _KIND_LABEL.get(e.kind, e.kind),
                e.summary,
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == _COL_KIND:
                    item.setForeground(
                        QColor(_KIND_COLOUR.get(e.kind, TEXT_MUTED_COLOR))
                    )
                if e.kind == EventKind.MESSAGE and e.data.get("level") in (
                    "ERROR",
                    "CRITICAL",
                ):
                    item.setForeground(QColor(ERROR_COLOR))
                if col == _COL_SUMMARY:
                    where = " · ".join(t for t in (e.item, e.task) if t)
                    item.setToolTip(f"{where}\n{text}" if where else text)
                self.table.setItem(row, col, item)
        self._apply_filter()

    def _load_overviews(self) -> None:
        # FIB overviews first: the canvas switches to the view of the image
        # added last, and the SEM's is the one a stage position reads best on.
        paths = find_overviews(self.replay.root)
        paths.sort(key=lambda p: 0 if "FIB" in str(p) else 1)
        for path in paths:
            image = self._images.get(path)
            if image is not None:
                self.stage_view.set_image(image, label=path.stem)
        views = self.stage_view.views
        self.view_combo.blockSignals(True)
        self.view_combo.addItems(views)
        if self.stage_view.view:
            self.view_combo.setCurrentText(self.stage_view.view)
        self.view_combo.blockSignals(False)
        self.view_combo.setVisible(len(views) > 1)
        if not views:
            self.stage_view.canvas.set_hint("No stored overview in this experiment")

    # ── playback ──────────────────────────────────────────────────────────

    @property
    def index(self) -> int:
        """The event at the playhead."""
        return self._index

    @property
    def is_playing(self) -> bool:
        return self._playing

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if not self._visible:
            return
        if self._position() >= len(self._visible) - 1:
            self._seek_visible(0)  # at the end: play from the start
        self._playing = True
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self._schedule()

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def step(self, delta: int) -> None:
        self.pause()
        self._seek_visible(self._position() + delta)

    def _schedule(self) -> None:
        pos = self._position()
        if not self._playing or pos >= len(self._visible) - 1:
            self.pause()
            return
        now = self.replay.events[self._visible[pos]].time
        nxt = self.replay.events[self._visible[pos + 1]].time
        speed = SPEEDS.get(self.speed_combo.currentText(), 60.0)
        wait_ms = (nxt - now).total_seconds() * 1000.0 / speed
        self._timer.start(int(min(max(wait_ms, _MIN_STEP_MS), _MAX_STEP_MS)))

    def _advance(self) -> None:
        self._seek_visible(self._position() + 1)
        self._schedule()

    def _position(self) -> int:
        """Where the playhead sits among the shown events."""
        i = bisect.bisect_left(self._visible, self._index)
        return min(i, len(self._visible) - 1) if self._visible else 0

    def _seek_visible(self, pos: int) -> None:
        if not self._visible:
            return
        pos = min(max(pos, 0), len(self._visible) - 1)
        self.seek(self._visible[pos])

    def _on_slider(self, value: int) -> None:
        if 0 <= value < len(self._visible) and self._visible[value] != self._index:
            self.seek(self._visible[value])

    def _scoped_item(self) -> Optional[str]:
        """The item the panes are scoped to, or None for the whole instrument."""
        selected = self.item_combo.currentData()
        return None if selected in (None, _NO_ITEM) else selected

    def _on_filter_changed(self) -> None:
        self._apply_filter()
        if self._visible and self._index not in self._visible:
            self._seek_visible(self._position())
        else:
            self.seek(self._index)  # the panes follow the item, too

    def _of_selected_item(self, event: ReplayEvent) -> bool:
        selected = self.item_combo.currentData()
        if selected is None:
            return True
        if selected == _NO_ITEM:
            return event.item is None
        return event.item == selected

    def _apply_filter(self) -> None:
        """Show the chosen lamella's actions of the chosen kinds.

        With an item chosen, the panes show only its images too (see
        ``ExperimentReplay.scene``); otherwise a lamella's first steps would
        show the frame the previous lamella left on screen.
        """
        kinds = {k for k, box in self.filter_boxes.items() if box.isChecked()}
        of_item = [self._of_selected_item(e) for e in self.replay.events]
        self._visible = [
            i
            for i, e in enumerate(self.replay.events)
            if of_item[i] and e.kind in kinds
        ]
        for row, e in enumerate(self.replay.events):
            self.table.setRowHidden(row, not (of_item[row] and e.kind in kinds))
        counts = {k: 0 for k in EventKind.ALL}
        for e, keep in zip(self.replay.events, of_item):
            if keep:
                counts[e.kind] = counts.get(e.kind, 0) + 1
        for kind, box in self.filter_boxes.items():
            box.setText(f"{_KIND_LABEL[kind]} ({counts.get(kind, 0)})")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self._visible) - 1, 0))
        self.slider.blockSignals(False)

    # ── showing an event ──────────────────────────────────────────────────

    def seek(self, index: int) -> None:
        """Show the instrument as it was at event *index*."""
        if not 0 <= index < len(self.replay.events):
            return
        self._index = index
        scene = self.replay.scene(index, item=self._scoped_item())
        self._show_header(scene)
        item = scene.event.item
        self._show_image(
            "sem", self.sem_canvas, scene.sem, scene.sem_unsaved_since, item
        )
        self._show_image(
            "fib", self.fib_canvas, scene.fib, scene.fib_unsaved_since, item
        )
        self._show_fm(scene.fm, item)
        self._show_milling(scene)
        self._show_stage(scene)
        self._flash(scene.event)
        self._sync_transport()

    def _sync_transport(self) -> None:
        pos = self._position()
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.position_label.setText(
            f"{pos + 1} / {len(self._visible)}" if self._visible else "0 / 0"
        )
        if 0 <= self._index < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(self._index)
            self.table.blockSignals(False)
            self.table.scrollToItem(
                self.table.item(self._index, 0),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    def _show_header(self, scene: ReplayScene) -> None:
        e = scene.event
        self.time_label.setText(
            f"{e.time:%Y-%m-%d %H:%M:%S}  {_elapsed(self.replay.start, e.time)}"
        )
        _clear_layout(self.context_row)
        for text, colour in (
            (e.item, ACCENT_COLOR),
            (e.task, TEXT_MUTED_COLOR),
            ((e.step or "").replace("_", " ").title(), TEXT_MUTED_COLOR),
        ):
            if text:
                self.context_row.addWidget(chip(text, colour))
        if not e.item and not e.task:
            self.context_row.addWidget(chip("Outside a workflow", TEXT_MUTED_COLOR))

        _clear_layout(self.event_kind_chip)
        self.event_kind_chip.addWidget(
            chip(
                _KIND_LABEL.get(e.kind, e.kind),
                _KIND_COLOUR.get(e.kind, TEXT_MUTED_COLOR),
            )
        )
        self.event_label.setText(e.summary)

    def _show_image(
        self,
        key: str,
        canvas: FibsemImageCanvas,
        event: Optional[ReplayEvent],
        unsaved_since: int,
        item: Optional[str],
    ) -> None:
        beam = "SEM" if key == "sem" else "FIB"
        path = event.image_path if event is not None else None
        if path is None:
            if self._shown[key] is not None:
                self._clear(key, canvas)
            self._set_hint(
                key,
                canvas,
                f"No {beam} image saved yet"
                + (f" ({unsaved_since} taken, not saved)" if unsaved_since else ""),
            )
            return
        if self._shown[key] != path:
            image = self._images.get(path)
            if image is None:
                self._clear(key, canvas)
                self._set_hint(key, canvas, f"Could not read {path.name}")
                return
            canvas.set_image(image)
            self._set_hint(key, canvas, None)
            self._shown[key] = path
        title = f"{beam} · {event.time:%H:%M:%S} · {_source(event, item)}{path.name}"
        if unsaved_since:
            title += f" · {unsaved_since} newer not saved"
        self._set_title(key, canvas, title)

    def _changed(self, key: str, value: object) -> bool:
        """Whether *value* differs from what *key* last showed; remembers it."""
        if key in self._memo and self._memo[key] == value:
            return False
        self._memo[key] = value
        return True

    def _set_title(self, key: str, canvas, text: Optional[str]) -> None:
        if self._changed(f"{key}.title", text):
            canvas.set_title(text)

    def _set_hint(self, key: str, canvas, text: Optional[str]) -> None:
        if self._changed(f"{key}.hint", text):
            canvas.set_hint(text)

    def _clear(self, key: str, canvas) -> None:
        """Empty a pane; `clear()` drops its title and hint, so forget them too."""
        canvas.clear()
        self._shown[key] = None
        self._memo.pop(f"{key}.title", None)
        self._memo.pop(f"{key}.hint", None)

    def _show_fm(self, event: Optional[ReplayEvent], item: Optional[str]) -> None:
        path = event.image_path if event is not None else None
        canvas = self.fm_widget.canvas
        if path is None:
            if self._shown["fm"] is not None:
                self.fm_widget.clear()
                self._clear("fm", canvas)
            return
        if self._shown["fm"] != path:
            image = self._fm_images.get(path)
            if image is None:
                self.fm_widget.clear()
                self._clear("fm", canvas)
                self._set_hint("fm", canvas, f"Could not read {path.name}")
                return
            self.fm_widget.set_fm_image(image)
            self._set_hint("fm", canvas, None)
            self._shown["fm"] = path
        self._set_title(
            "fm",
            canvas,
            f"FM · {event.time:%H:%M:%S} · {_source(event, item)}{path.name}",
        )

    def _show_milling(self, scene: ReplayScene) -> None:
        fib_path = scene.fib.image_path if scene.fib else None
        milling_id = id(scene.milling) if scene.milling is not None else None
        if not self._changed("milling", (milling_id, tuple(scene.spots), fib_path)):
            return
        fib_image = (
            self._images.get(scene.fib.image_path)
            if (scene.fib and scene.fib.image_path)
            else None
        )
        if scene.milling is not None and scene.milling_stages and fib_image is not None:
            stages = self._milling_stages(scene)
            current = (scene.milling.data.get("stage") or {}).get("name")
            selected = next(
                (i for i, s in enumerate(stages) if s.name == current), None
            )
            self.milling_overlay.set_stages(stages, fib_image, selected_index=selected)
        else:
            self.milling_overlay.clear()
        pixel_size = _pixel_size(fib_image)
        if scene.spots and fib_image is not None and pixel_size:
            h, w = fib_image.data.shape[:2]
            self.spot_overlay.set_points(
                [
                    (w / 2 + dx / pixel_size, h / 2 + dy / pixel_size)
                    for dx, dy in scene.spots
                ]
            )
        else:
            self.spot_overlay.set_points([])

    def _milling_stages(self, scene: ReplayScene) -> List[FibsemMillingStage]:
        key = scene.milling.data.get("milling_task_id") if scene.milling else None
        if key not in self._stages_cache:
            stages = []
            for d in scene.milling_stages:
                try:
                    stages.append(FibsemMillingStage.from_dict(d))
                except Exception as e:
                    logger.debug(
                        f"Replay could not rebuild milling stage {d.get('name')}: {e}"
                    )
            self._stages_cache[key] = stages
        return self._stages_cache[key]

    def _show_stage(self, scene: ReplayScene) -> None:
        position = _stage_position(scene.stage_position)
        if not self._changed("stage", position):
            return
        self.stage_view.set_positions([position] if position else [], movable=False)
        self.stage_view.set_selected_position(_STAGE_MARK if position else None)
        self.stage_view.canvas.set_info_text(_describe_stage(position))

    def _flash(self, event: ReplayEvent) -> None:
        if event.kind == EventKind.STAGE:
            self.stage_view.canvas.flash_message(
                event.summary.split(" to ")[0].split(" by ")[0]
            )
        elif event.kind == EventKind.ALIGNMENT and "beam shift" in event.summary:
            canvas = (
                self.fib_canvas if event.summary.startswith("FIB") else self.sem_canvas
            )
            canvas.flash_message("Beam shift")
        elif event.kind == EventKind.MILLING:
            self.fib_canvas.flash_message(event.summary.split(" (")[0])

    def _on_view_selected(self, view: str) -> None:
        if view and view != self.stage_view.view:
            self.stage_view.show_view(view)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        self.pause()
        super().closeEvent(event)


def open_replay_window(
    path: Path, parent: Optional[QWidget] = None
) -> Optional[ExperimentReplayWidget]:
    """Open a replay of the experiment at *path* in its own window.

    Returns None (after logging why) when there is nothing to replay. The
    caller keeps the returned widget alive; it has no parent.
    """
    try:
        widget = ExperimentReplayWidget.from_directory(Path(path))
    except FileNotFoundError as e:
        logger.warning(f"Cannot replay {path}: {e}")
        return None
    widget.resize(1500, 950)
    widget.show()
    widget.activateWindow()
    return widget


def main() -> None:
    """``python -m fibsem.applications.autolamella.ui.experiment_replay_widget [DIR]``"""
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else QFileDialog.getExistingDirectory(None, "Choose an experiment to replay")
    )
    if not path:
        return
    widget = open_replay_window(Path(path))
    if widget is None:
        print(f"Nothing to replay in {path}: no logfile.log")
        sys.exit(1)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
