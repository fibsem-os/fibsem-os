"""The History tab: one row per task run on a lamella, with its operations and images."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImageReader, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.history_rows import (
    OPERATION_KINDS,
    HistoryFilter,
    HistoryOperation,
    HistoryRun,
    filter_runs,
    history_rows,
    milling_stages,
    operation_facts,
    operations_need_a_look,
    operations_summary,
)
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.common.thumbnails import (
    PLACEHOLDER_HEIGHT,
    TARGET_WIDTH,
    ClickableLabel,
    file_stamp,
    load_and_resize,
)
from fibsem.cancellation import CANCELLED, COMPLETED, FAILED, SKIPPED
from fibsem.config import load_user_preferences, update_user_preferences
from fibsem.fm.preview import is_fluorescence_image
from fibsem.imaging.drawing import draw_image_overlays
from fibsem.imaging.export import ImageFields, image_caption, image_summary
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    GRAY_ICON_COLOR,
    NEUTRAL_200,
    NEUTRAL_400,
    NEUTRAL_550,
    NEUTRAL_900,
    NUMBER_FONT,
    PANEL_COLOR,
    SURFACE_COLOR,
)
from fibsem.ui.utils import arr_to_qimage
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.ui.widgets.image_viewer_dialog import ViewerItem, open_image_viewer
from fibsem.ui.widgets.task_summary_formatting import STATUS_BADGE_COLORS
from fibsem.util.timestamps import format_time
from fibsem.workflows.event_recording import (
    EVENTS_FILENAME,
    read_new_events,
)
from fibsem.workflows.status import TaskStatus

_MIN_THUMB_WIDTH = 120
_MAX_THUMB_WIDTH = TARGET_WIDTH
_MARGIN = 16
_SPACING = 8
# The runs stay as wide as a pair of thumbnails rather than spreading across a
# wider panel, so a line's duration sits above the pictures it timed.
_MAX_CONTENT_WIDTH = 2 * _MAX_THUMB_WIDTH + _SPACING + 2 * _MARGIN
_MAX_IMAGES_PER_TASK = 2  # last 2 files = highest-res SEM + FIB
_IMAGES_PER_LINE = 2  # tiles per line before wrapping; matches the SEM/FIB pair
_CAPTION_STYLE = (
    f"color: {NEUTRAL_550}; font-family: {NUMBER_FONT}; font-size: 11px;"
    " background: transparent;"
)


class _ClickableRow(QWidget):
    """A line that emits clicked() when left-clicked: an operation, to expand it."""

    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def _open_path(path: str) -> None:
    """In the system's own viewer: a folder in the file browser, a figure in
    whatever opens PNGs."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


# The widest a figure is drawn on hover: readable, and well inside a screen.
_FIGURE_TOOLTIP_WIDTH = 640
# An opened operation's figure, small at the right of its values and no taller
# than they are; shown only when the thumbnails are at least
# _MIN_WIDTH_FOR_MINI_FIGURE wide, so the values keep their room.
_MINI_FIGURE_SIZE = QSize(220, 140)
_MIN_WIDTH_FOR_MINI_FIGURE = 300


def _figure_tooltip(path: str) -> str:
    """A tooltip that is the figure at ``path``, no wider than the cap. Qt draws
    rich-text tooltips, images included, and reads the file only when shown."""
    width = QImageReader(path).size().width()
    if width <= 0:
        return path
    width = min(width, _FIGURE_TOOLTIP_WIDTH)
    return f'<img src="{QUrl.fromLocalFile(path).toString()}" width="{width}">'


def _link(text: str, path: str) -> QPushButton:
    """A flat accent link that opens ``path``, the path on hover."""
    link = QPushButton(text)
    link.setFlat(True)
    link.setCursor(Qt.CursorShape.PointingHandCursor)
    link.setToolTip(path)
    link.setStyleSheet(
        f"QPushButton {{ color: {ACCENT_COLOR}; background: transparent;"
        " border: none; padding: 0; text-align: left; font-size: 11px; }"
    )
    link.clicked.connect(lambda _c=False: _open_path(path))
    return link


class _ImageLoaderWorker(QThread):
    """Background worker that loads images one at a time."""

    # filepath, array, pixel_size_x, ImageFields
    image_loaded = pyqtSignal(str, np.ndarray, float, object)

    def __init__(self, filepaths: List[str], target_width: int):
        # No parent, and kept in _WORKERS until it has finished: a cancelled
        # worker ends the image it is on by itself, and Qt aborts the app when a
        # QThread is destroyed while it runs, as a parent's deletion would do.
        super().__init__()
        self._filepaths = filepaths
        self._target_width = target_width
        self._cancel = threading.Event()
        _WORKERS.difference_update([w for w in _WORKERS if w.isFinished()])
        _WORKERS.add(self)

    def cancel(self):
        self._cancel.set()

    def run(self):
        for fpath in self._filepaths:
            if self._cancel.is_set():
                return
            try:
                arr, pixel_size_x, info = load_and_resize(fpath, self._target_width)
                if self._cancel.is_set():
                    return
                self.image_loaded.emit(fpath, arr, pixel_size_x, info)
            except Exception as e:
                logging.warning(f"Failed to load image {fpath}: {e}")


_WORKERS: Set[_ImageLoaderWorker] = set()


# Status words and colours, as the run's header shows them. The colours are the
# task-summary badges, so a run reads the same here as in the workflow summary.
_RUN_STATUS = {
    TaskStatus.Completed: ("Completed", "mdi:check", "Completed"),
    TaskStatus.Failed: ("Failed", "mdi:close", "Failed"),
    TaskStatus.Cancelled: ("Cancelled", "mdi:stop", "Cancelled"),
    TaskStatus.InProgress: (
        "In progress",
        "mdi:progress-clock",
        "InProgress",
    ),
    TaskStatus.Skipped: ("Skipped", "mdi:skip-next", "Skipped"),
}
# Which of a run's operation endings its folded line shows: the worst.
_SEVERITY = {FAILED: 0, CANCELLED: 1, SKIPPED: 2}

_OPERATION_ICON = {
    COMPLETED: ("mdi:check", "Completed"),
    SKIPPED: ("mdi:skip-next", "Skipped"),
    FAILED: ("mdi:close", "Failed"),
    CANCELLED: ("mdi:stop", "Cancelled"),
}
_SHOW = (
    ("all", "Everything"),
    ("images", "Images only"),
    ("operations", "Operations only"),
)
_RUNS = (("all", "All runs"), ("failed", "Failed"), ("cancelled", "Cancelled"))


def _clock(value) -> str:
    return format_time(value, "%H:%M") or ""


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} m {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} m"


def _icon_label(key: str, color: str, size: int = 14) -> QLabel:
    label = QLabel()
    label.setPixmap(fibsem_icon(key, color=color).pixmap(QSize(size, size)))
    label.setFixedSize(size + 2, size + 2)
    label.setStyleSheet("background: transparent;")
    return label


def _text(text: str, style: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(style + " background: transparent;")
    return label


class _HistoryFilterButton(QToolButton):
    """The filter menu: what each run shows, which runs, which task.

    Behind an icon: set once and left, so not worth a row
    of controls. The icon takes the accent while anything is narrowed, so a
    filtered history is never mistaken for the whole one. Each section is one of
    several, so its items read as radio buttons -- the mark is the action's icon.
    """

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(26, 26))
        self.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self.setPopupMode(QToolButton.InstantPopup)
        self.setFocusPolicy(Qt.NoFocus)
        # No parent: a child of this button inherits the stylesheets above it, and
        # the header row's unscoped `background: transparent` made the menu
        # see-through. Unparented, it is styled as every other menu in the app,
        # and is not deleted with the header either. self._menu keeps it alive.
        self._menu = QMenu()
        self.setMenu(self._menu)
        self._filter = HistoryFilter()
        self._tasks: List[str] = []
        self._actions: Dict[Tuple[str, Any], QAction] = {}
        self._build_menu()

    @property
    def filter(self) -> HistoryFilter:
        return self._filter

    def set_filter(self, flt: HistoryFilter) -> None:
        self._filter = flt
        self._paint()

    def set_tasks(self, names: List[str]) -> None:
        """The lamella's task names, for the Task section. A filtered task the
        lamella has not run stays listed, so it can be seen and cleared."""
        names = list(dict.fromkeys(names))
        if self._filter.task is not None and self._filter.task not in names:
            names.append(self._filter.task)
        if names != self._tasks:
            self._tasks = names
            self._build_menu()

    def _build_menu(self) -> None:
        self._menu.clear()
        self._actions.clear()
        self._heading("Show")
        for value, text in _SHOW:
            self._add("show", value, text)
        self._menu.addSeparator()
        self._heading("Runs")
        for value, text in _RUNS:
            self._add("status", value, text)
        self._menu.addSeparator()
        self._heading("Task")
        self._add("task", None, "All tasks")
        for name in self._tasks:
            self._add("task", name, name)
        self._menu.addSeparator()
        reset = self._menu.addAction("Reset filters")
        reset.triggered.connect(lambda _c=False: self._pick(HistoryFilter()))
        self._paint()

    def _heading(self, text: str) -> None:
        """A section's name, as a disabled item: ``QMenu.addSection`` draws a
        bare separator under several platform styles, losing the name."""
        self._menu.addAction(text.upper()).setEnabled(False)

    def _add(self, key: str, value: Any, text: str) -> None:
        action = self._menu.addAction(text)
        action.triggered.connect(
            lambda _c=False, k=key, v=value: self._pick(replace(self._filter, **{k: v}))
        )
        self._actions[(key, value)] = action

    def _pick(self, flt: HistoryFilter) -> None:
        self._filter = flt
        self._paint()
        self.changed.emit()

    def _paint(self) -> None:
        chosen = {
            ("show", self._filter.show),
            ("status", self._filter.status),
            ("task", self._filter.task),
        }
        for key, action in self._actions.items():
            on = key in chosen
            action.setIcon(
                fibsem_icon(
                    "mdi:radiobox-marked" if on else "mdi:radiobox-blank",
                    color=ACCENT_COLOR if on else GRAY_ICON_COLOR,
                )
            )
        active = self._filter.active
        self.setIcon(
            fibsem_icon(
                "mdi:filter-variant", color=ACCENT_COLOR if active else GRAY_ICON_COLOR
            )
        )
        self.setToolTip("Filtered: change or reset" if active else "Filter runs")


class LamellaTaskImageWidget(QWidget):
    """The History tab: one row per task run on a lamella, in the order they ran.

    Each row says how the run ended, what its operations did (an alignment, an
    autofocus, each with what it measured or why it did not run to an end), and
    shows its images as thumbnails that open in the image viewer. Rows come from
    ``history_rows``: the lamella's task history joined to the operation events in
    the experiment's ``events.jsonl``. An experiment without that file still shows
    its runs and images.

    A filter behind the icon at the top right narrows what is shown; it is
    remembered per user. Images load in a background thread, so the rows appear at
    once with placeholders that fill in.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lamella: Optional[Lamella] = None
        self._lamella_id: Optional[str] = None
        self._pixmap_cache: Dict[str, QPixmap] = {}
        # image path -> the file's (mtime, size) when its pixmap was read: a
        # refresh keeps the pixmaps whose files have not changed since.
        self._stamps: Dict[str, Tuple[int, int]] = {}
        self._placeholder_labels: Dict[str, QLabel] = {}
        self._caption_labels: Dict[str, QLabel] = {}
        self._task_names: Dict[str, str] = {}  # image path -> the task that made it
        self._kinds: Dict[str, str] = {}  # image path -> SEM / FIB / FM, once read
        # image path -> (caption, tooltip), kept with the pixmaps they came with
        self._captions: Dict[str, Tuple[str, str]] = {}
        self._worker: Optional[_ImageLoaderWorker] = None
        self._runs: List[HistoryRun] = []
        # The operations shown expanded, as (run, index in the run): kept across
        # a rebuild, so a refresh after a task does not fold what was open.
        self._expanded: Set[Tuple[str, int]] = set()
        # A run's operations opened or folded by a click, by run: kept across a
        # rebuild like the lines. A run nobody clicked is open only when one of
        # its operations needs a look (FIB-1361).
        self._operations_open: Dict[str, bool] = {}
        # Whether the experiment records its events: without the file nothing
        # was ever recorded, so "no operations" would be a claim it cannot make.
        self._has_events = False
        # The experiment's operation events read so far, and where in its file
        # the next read starts: a rebuild reads only what was written since,
        # not the whole file again (it grows through a run).
        self._events_path: Optional[Path] = None
        self._events_offset = 0
        self._operation_events: List[Dict[str, Any]] = []
        self._thumb_width = TARGET_WIDTH
        # Re-fit the thumbnails once a resize settles, not on every pixel of a drag.
        self._refit = QTimer(self)
        self._refit.setSingleShot(True)
        self._refit.setInterval(150)
        self._refit.timeout.connect(self._refit_thumbnails)

        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {SURFACE_COLOR}; }}"
        )
        outer.addWidget(self._scroll)

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {SURFACE_COLOR};")
        self._content.setMaximumWidth(_MAX_CONTENT_WIDTH)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(_MARGIN, _MARGIN, _MARGIN, _MARGIN)
        self._content_layout.setSpacing(12)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._filter_button = _HistoryFilterButton(self)
        self._filter_button.set_filter(
            HistoryFilter.from_dict(load_user_preferences().display.history_filter)
        )
        self._filter_button.changed.connect(self._on_filter_changed)
        self._filter_button.hide()

        # What can be done with the lamella's history, behind "...": open its
        # folder, and export the workflow GIF made of the runs and images this tab
        # shows (FIB-1311). Kept by the widget and moved into each header, as the
        # filter is; the menu is unparented for the same reason as the filter's.
        self._actions_button = QToolButton(self)
        self._actions_button.setFixedSize(QSize(26, 26))
        self._actions_button.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self._actions_button.setPopupMode(QToolButton.InstantPopup)
        self._actions_button.setFocusPolicy(Qt.NoFocus)
        self._actions_button.setIcon(
            fibsem_icon("mdi:dots-horizontal", color=GRAY_ICON_COLOR)
        )
        self._actions_button.setToolTip("Actions")
        self._actions_menu = QMenu()
        self._action_open_folder = self._actions_menu.addAction(
            fibsem_icon("mdi:folder", color=GRAY_ICON_COLOR), "Open Lamella Folder"
        )
        self._action_open_folder.triggered.connect(self._on_open_folder)
        self._action_export_gif = self._actions_menu.addAction(
            fibsem_icon("mdi:file-gif-box", color=GRAY_ICON_COLOR), "Export GIF..."
        )
        self._action_export_gif.triggered.connect(self._on_export_gif)
        self._actions_button.setMenu(self._actions_menu)
        self._actions_button.hide()

        self._empty_label = QLabel("Select a lamella card to view task images.")
        self._empty_label.setStyleSheet(f"color: {NEUTRAL_550}; font-size: 12px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addWidget(self._empty_label)

        self._scroll.setWidget(self._content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Set the lamella to display. Skips reload if same lamella."""
        new_id = lamella.id if lamella is not None else None
        if new_id == self._lamella_id:
            return

        self._cancel_worker()
        self._lamella = lamella
        self._lamella_id = new_id
        self._pixmap_cache.clear()
        self._stamps.clear()
        self._captions.clear()
        self._placeholder_labels.clear()
        self._expanded.clear()
        self._operations_open.clear()
        self._rebuild()

    def refresh(self) -> None:
        """Rebuild for the lamella already shown, after a task on it finished.

        `set_lamella` skips the lamella it already shows, so without this the
        panel kept the history it was opened with until another lamella was
        selected and back (FIB-1111). A cached image goes when its file has
        changed: a re-run writes its reference images to the same filenames, so
        its pixmap would be the previous run's picture. The others stay, so a
        refresh after each task reads only the images that task wrote.
        """
        if self._lamella is None:
            return
        self._cancel_worker()
        for path in list(self._pixmap_cache):
            if file_stamp(path) != self._stamps.get(path):
                self._forget_image(path)
        self._rebuild()

    def _forget_image(self, path: str) -> None:
        self._pixmap_cache.pop(path, None)
        self._stamps.pop(path, None)
        self._captions.pop(path, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cancel_worker(self) -> None:
        """Stop the loader without waiting for it: it finishes the image it is on
        in the background, and nothing it loads reaches the panel. Waiting held
        the GUI for up to two seconds, as a large image takes that to resize."""
        if self._worker is not None:
            self._worker.image_loaded.disconnect(self._on_image_loaded)
            self._worker.cancel()
            self._worker = None

    def _clear_layout(self) -> None:
        """Remove all widgets from the content layout, keeping the filter button.

        The button lives in the header row, which is deleted here: it is taken out
        first, or Qt deletes it and its menu with the header. A rebuild usually
        moves it into the new header before that happens, but clearing the
        selection makes no new header, and the next rebuild then used a deleted
        QMenu and aborted the app.
        """
        self._filter_button.hide()
        self._filter_button.setParent(self)
        self._actions_button.hide()
        self._actions_button.setParent(self)
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _fitted_thumb_width(self) -> int:
        """Two thumbnails to a line in the panel's width, within the bounds."""
        available = min(self._scroll.viewport().width(), _MAX_CONTENT_WIDTH)
        width = (available - 2 * _MARGIN - _SPACING) // 2
        return max(_MIN_THUMB_WIDTH, min(_MAX_THUMB_WIDTH, width))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._lamella is not None:
            self._refit.start()

    def _refit_thumbnails(self) -> None:
        """Rebuild at the new size when the thumbnails would change by more than a
        little: the cached pixmaps are the old size, so they go too."""
        if self._lamella is None:
            return
        if abs(self._fitted_thumb_width() - self._thumb_width) < 16:
            return
        self._cancel_worker()
        self._pixmap_cache.clear()
        self._stamps.clear()
        self._captions.clear()
        self._rebuild()

    def _on_filter_changed(self) -> None:
        flt = self._filter_button.filter
        update_user_preferences(
            lambda prefs: setattr(prefs.display, "history_filter", flt.to_dict())
        )
        self._cancel_worker()
        self._rebuild()

    def _events(self) -> List[Dict[str, Any]]:
        """The experiment's recorded events, or none: an experiment from before
        the event stream has no file, and still shows its runs and images."""
        if self._lamella is None:
            return []
        path = Path(self._lamella.path).parent / EVENTS_FILENAME
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        self._has_events = size is not None
        if size is None:
            return []
        if path != self._events_path or size < self._events_offset:
            # another experiment, or a file that was started again
            self._events_path = path
            self._events_offset = 0
            self._operation_events = []
        try:
            new, self._events_offset = read_new_events(
                path, self._events_offset, OPERATION_KINDS
            )
        except OSError as e:
            logging.warning(f"Could not read {path}: {e}")
            return self._operation_events
        self._operation_events.extend(new)
        return self._operation_events

    def _rebuild(self) -> None:
        """Build the rows with placeholders, then load the images in the background."""
        self._clear_layout()
        self._placeholder_labels.clear()
        self._caption_labels.clear()
        self._task_names.clear()

        if self._lamella is None:
            self._filter_button.hide()
            self._actions_button.hide()
            label = QLabel("Select a lamella card to view task images.")
            label.setStyleSheet(f"color: {NEUTRAL_550}; font-size: 12px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._content_layout.addWidget(label)
            return

        lamella = self._lamella
        self._thumb_width = self._fitted_thumb_width()
        self._runs = history_rows(lamella, self._events())
        flt = self._filter_button.filter
        self._filter_button.set_tasks([run.task_name for run in self._runs])
        runs = filter_runs(self._runs, flt)

        self._content_layout.addWidget(self._header(lamella.name, runs, flt))

        if not self._runs:
            self._content_layout.addWidget(
                _text("No task runs yet.", f"color: {NEUTRAL_550}; font-size: 11px;")
            )
            self._content_layout.addStretch(1)
            return
        if not runs:
            self._content_layout.addWidget(self._nothing_matches(flt))
            self._content_layout.addStretch(1)
            return

        all_filepaths: List[str] = []
        for run in runs:
            # cap the reference images *before* appending fluorescence: the cap picks
            # the highest-magnification SEM/FIB pair out of a multi-FOV set, and
            # applying it to a merged list would let a z-stack displace the FIB image.
            finals = [p for p in run.images if not is_fluorescence_image(p)]
            stacks = [p for p in run.images if is_fluorescence_image(p)]
            filenames = finals[-_MAX_IMAGES_PER_TASK:] + stacks
            if flt.show == "operations":
                filenames = []
            self._content_layout.addWidget(self._run_row(run, filenames, flt))
            all_filepaths.extend(filenames)
            self._task_names.update(dict.fromkeys(filenames, run.task_name))

        self._content_layout.addStretch(1)

        to_load = []
        for fpath in all_filepaths:
            if fpath in self._pixmap_cache:
                label = self._placeholder_labels.get(fpath)
                if label is not None:
                    label.setPixmap(self._pixmap_cache[fpath])
                self._show_caption(fpath)
            else:
                to_load.append(fpath)
                self._stamps[fpath] = file_stamp(fpath)

        if to_load:
            self._worker = _ImageLoaderWorker(to_load, self._thumb_width)
            self._worker.image_loaded.connect(self._on_image_loaded)
            self._worker.start()

    def _header(self, name: str, runs: List[HistoryRun], flt: HistoryFilter) -> QWidget:
        """The lamella's name, a line about its runs, and the filter and the
        actions at the right."""
        header = QWidget()
        header.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(
            _text(name, f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200};")
        )
        # the lamella's runs: operations outside a task are not one
        own = [run for run in self._runs if not run.outside_task]
        if flt.active:
            shown = sum(1 for run in runs if not run.outside_task)
            line = f"Filtered · {shown} of {len(own)} runs"
            color = ACCENT_COLOR
        else:
            total = sum(run.duration or 0.0 for run in own)
            count = len(own)
            line = f"{count} run{'' if count == 1 else 's'}"
            if total:
                line += f" · {_duration(total)}"
            color = NEUTRAL_550
        self._subtitle = _text(line, f"font-size: 11px; color: {color};")
        text.addWidget(self._subtitle)
        layout.addLayout(text, 1)
        layout.addWidget(self._filter_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._actions_button, 0, Qt.AlignmentFlag.AlignTop)
        self._filter_button.show()
        self._actions_button.show()
        return header

    def _on_open_folder(self) -> None:
        """The lamella's folder in the system's file browser."""
        lamella = self._lamella
        if lamella is None:
            return
        path = os.fspath(lamella.path)
        if not os.path.isdir(path):
            from fibsem.ui import notification_service

            notification_service.show_toast(f"Folder not found: {path}", "error")
            return
        _open_path(path)

    def _on_export_gif(self) -> None:
        """Open the workflow GIF export on the images the lamella's tasks saved:
        each task's latest run, whatever the filter shows.

        Needs no microscope: it reads the files the task history recorded.
        """
        from PyQt5.QtWidgets import QApplication

        from fibsem.applications.autolamella.task_outputs import images_by_task
        from fibsem.imaging.animation import AnimationStep, load_frames
        from fibsem.ui import notification_service
        from fibsem.ui.widgets.animation_export_dialog import AnimationExportDialog

        lamella = self._lamella
        if lamella is None:
            return
        steps = [
            AnimationStep(title=name, paths=finals, fluorescence=stacks)
            for name, finals, stacks in images_by_task(lamella)
        ]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frames = load_frames(steps)
        finally:
            QApplication.restoreOverrideCursor()
        if not frames:
            notification_service.show_toast(
                f"{lamella.name} has no task images to animate yet", "info"
            )
            return
        AnimationExportDialog(
            frames, lamella.name, os.fspath(lamella.path), self
        ).exec_()

    def _nothing_matches(self, flt: HistoryFilter) -> QWidget:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 8, 0, 0)
        which = "" if flt.status == "all" else f"{flt.status} "
        of = "" if flt.task is None else f" of {flt.task}"
        layout.addWidget(
            _text(
                f"No {which}runs{of} to show on this lamella.",
                f"color: {NEUTRAL_400}; font-size: 12px;",
            )
        )
        reset = QPushButton("Show all runs")
        reset.setFlat(True)
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.setStyleSheet(
            f"QPushButton {{ color: {ACCENT_COLOR}; background: transparent;"
            " border: none; padding: 0; text-align: left; font-size: 11px; }"
        )
        reset.clicked.connect(lambda: self._filter_button._pick(HistoryFilter()))
        layout.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        return box

    def _run_row(
        self, run: HistoryRun, filenames: List[str], flt: HistoryFilter
    ) -> QWidget:
        """One run: name and how it ended, when and how long, its operations, its
        images."""
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        word, icon, badge = _RUN_STATUS.get(
            run.status, (run.status.name, "mdi:circle-outline", "Skipped")
        )
        colour = STATUS_BADGE_COLORS.get(badge, (NEUTRAL_550, NEUTRAL_550))[1]
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(
            _text(
                run.task_name,
                f"font-size: 12px; font-weight: 600; color: {NEUTRAL_400};",
            )
        )
        when = " · ".join(
            part for part in (_clock(run.started_at), _duration(run.duration)) if part
        )
        if when:
            top.addWidget(_text(when, _CAPTION_STYLE))
        top.addStretch(1)
        if not run.outside_task:  # not a run: nothing ended
            top.addWidget(_icon_label(icon, colour, 12))
            top.addWidget(_text(word, f"font-size: 11px; color: {colour};"))
        layout.addLayout(top)
        if run.status_message and run.status in (
            TaskStatus.Failed,
            TaskStatus.Cancelled,
        ):
            message = _text(run.status_message, f"font-size: 11px; color: {colour};")
            message.setWordWrap(True)
            layout.addWidget(message)

        if flt.show != "images":
            if len(run.operations) > 1 and flt.show == "all":
                layout.addWidget(self._operations_fold(run))
            else:  # one line, or operations asked for: no fold to open
                for index, operation in enumerate(run.operations):
                    layout.addWidget(self._operation(run.task_id, index, operation))
            if not run.operations and flt.show == "all" and self._has_events:
                layout.addWidget(
                    _text(
                        "No operations recorded",
                        f"font-size: 11px; color: {NEUTRAL_550};",
                    )
                )

        if flt.show != "operations" and not run.outside_task:
            if filenames:
                layout.addWidget(self._image_grid(filenames))
            elif run.images_replaced_by is not None:
                later = next(
                    (r for r in self._runs if r.task_id == run.images_replaced_by), None
                )
                at = f" {_clock(later.started_at)}" if later is not None else " later"
                layout.addWidget(
                    _text(
                        f"Images replaced by the{at} run.",
                        f"font-size: 11px; color: {NEUTRAL_550};",
                    )
                )
            else:
                layout.addWidget(
                    _text(
                        "No images recorded.",
                        "font-size: 11px; color: #808080;",
                    )
                )
        return container

    def _operations_fold(self, run: HistoryRun) -> QWidget:
        """``› ✓ 3 operations · all completed   6 s``, the run's operations
        under it. Folded unless one needs a look or it was opened (FIB-1361):
        a run that went to plan is one line, and one that did not shows why."""
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        line = _ClickableRow()
        line.setStyleSheet("background: transparent;")
        row = QHBoxLayout(line)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        chevron = _icon_label("mdi:chevron-right", NEUTRAL_550)
        row.addWidget(chevron)
        # the worst way one ended: a failure before a stop before a skip
        worst = min(
            (op.status for op in run.operations), key=lambda s: _SEVERITY.get(s, 3)
        )
        icon, badge = _OPERATION_ICON.get(worst, ("mdi:circle-outline", ""))
        colour = STATUS_BADGE_COLORS.get(badge, (NEUTRAL_550, NEUTRAL_550))[0]
        row.addWidget(_icon_label(icon, colour))
        summary = ElidedLabel(operations_summary(run))
        summary.setStyleSheet(_CAPTION_STYLE)
        row.addWidget(summary, 1)
        durations = [op.duration for op in run.operations if op.duration is not None]
        row.addWidget(
            _text(_duration(sum(durations)) if durations else "", _CAPTION_STYLE)
        )
        line.setToolTip("Click to show or hide the operations")
        layout.addWidget(line)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        lines = QVBoxLayout(body)
        lines.setContentsMargins(22, 0, 0, 0)  # under the summary, past its chevron
        lines.setSpacing(4)
        for index, operation in enumerate(run.operations):
            lines.addWidget(self._operation(run.task_id, index, operation))
        layout.addWidget(body)

        def show(opened: bool) -> None:
            body.setVisible(opened)
            chevron.setPixmap(
                fibsem_icon(
                    "mdi:chevron-down" if opened else "mdi:chevron-right",
                    color=NEUTRAL_550,
                ).pixmap(QSize(14, 14))
            )

        def toggle() -> None:
            opened = not self._operations_opened(run)
            self._operations_open[run.task_id] = opened
            show(opened)

        line.clicked.connect(toggle)
        show(self._operations_opened(run))
        return box

    def _operations_opened(self, run: HistoryRun) -> bool:
        return self._operations_open.get(run.task_id, operations_need_a_look(run))

    def _operation(
        self, task_id: str, index: int, operation: HistoryOperation
    ) -> QWidget:
        """The operation's line, and under it, once clicked, what it ran with and
        measured and the figure it saved. Built on first opening: the figure is
        read from disk."""
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        line, chevron = self._operation_line(operation)
        layout.addWidget(line)
        key = (task_id, index)
        details: List[QWidget] = []

        def show(expanded: bool) -> None:
            if expanded and not details:
                details.append(self._operation_details(operation))
                layout.addWidget(details[0])
            if details:
                details[0].setVisible(expanded)
            chevron.setPixmap(
                fibsem_icon(
                    "mdi:chevron-down" if expanded else "mdi:chevron-right",
                    color=NEUTRAL_550,
                ).pixmap(QSize(14, 14))
            )

        def toggle() -> None:
            expanded = key not in self._expanded
            if expanded:
                self._expanded.add(key)
            else:
                self._expanded.discard(key)
            show(expanded)

        line.clicked.connect(toggle)
        show(key in self._expanded)
        return box

    def _operation_line(
        self, operation: HistoryOperation
    ) -> Tuple[_ClickableRow, QLabel]:
        """``› ✓ Align   1.11 µm · 3 steps   6 s``, and its chevron."""
        line = _ClickableRow()
        line.setStyleSheet("background: transparent;")
        line.setToolTip(f"{operation.detail}\nClick for details")
        layout = QHBoxLayout(line)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        chevron = _icon_label("mdi:chevron-right", NEUTRAL_550)
        layout.addWidget(chevron)
        icon, badge = _OPERATION_ICON.get(operation.status, ("mdi:circle-outline", ""))
        colour = STATUS_BADGE_COLORS.get(badge, (NEUTRAL_550, NEUTRAL_550))[0]
        layout.addWidget(_icon_label(icon, colour))
        layout.addWidget(
            _text(operation.label, f"font-size: 12px; color: {NEUTRAL_200};")
        )
        detail = ElidedLabel(operation.detail)
        detail.setStyleSheet(_CAPTION_STYLE)
        detail.setToolTip(line.toolTip())
        layout.addWidget(detail, 1)
        layout.addWidget(_text(_duration(operation.duration), _CAPTION_STYLE))
        return line, chevron

    def _operation_details(self, operation: HistoryOperation) -> QWidget:
        """What the operation ran with and measured, one fact a line, then links
        to the figure its run saved and its folder; the figure itself small at
        the right, full size on hover. Indented under its line."""
        panel = QFrame()
        panel.setObjectName("operationDetails")
        panel.setStyleSheet(
            f"QFrame#operationDetails {{ background: {PANEL_COLOR}; border-radius: 4px; }}"
        )
        columns = QHBoxLayout(panel)
        columns.setContentsMargins(10, 8, 10, 8)
        columns.setSpacing(12)
        layout = QVBoxLayout()
        layout.setSpacing(6)
        columns.addLayout(layout, 1)

        facts = QGridLayout()
        facts.setHorizontalSpacing(12)
        facts.setVerticalSpacing(2)
        # (name, text, indented, figure): the facts, then for a mill each stage
        # with, indented under it, what its strategy (FIB-1257) and monitors
        # (FIB-1481) said and the drift corrections it ran (FIB-1351)
        rows: List[Tuple[str, str, bool, Optional[str]]] = [
            (name, value, False, None) for name, value in operation_facts(operation)
        ]
        if operation.kind == "milling":
            for stage in milling_stages(operation):
                rows.append((stage.name, stage.text, False, None))
                if stage.strategy:
                    rows.append((stage.strategy, stage.said, True, stage.figure))
                rows.extend((m.name, m.said, True, m.figure) for m in stage.monitors)
                rows.extend(
                    ("alignment", run.detail, True, run.plot) for run in stage.drift
                )
        for row, (name, value, nested, figure) in enumerate(rows):
            label = _text(name, f"font-size: 11px; color: {NEUTRAL_550};")
            if nested:
                label.setContentsMargins(12, 0, 0, 0)
            facts.addWidget(label, row, 0, Qt.AlignmentFlag.AlignTop)
            text = _text(
                value,
                f"font-family: {NUMBER_FONT}; font-size: 11px; color: {NEUTRAL_400};",
            )
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            facts.addWidget(text, row, 1)
            if figure is not None:
                link = _link("Figure", figure)
                link.setToolTip(_figure_tooltip(figure))
                facts.addWidget(link, row, 2, Qt.AlignmentFlag.AlignTop)
        facts.setColumnStretch(1, 1)
        layout.addLayout(facts)

        # The figure small, not full width: a full-width plot under every opened
        # line drowned the run's own images. Full size on hover, or opened.
        links = QHBoxLayout()
        links.setSpacing(16)
        if operation.plot is not None:
            figure = _figure_tooltip(operation.plot)
            link = _link("Open figure", operation.plot)
            link.setToolTip(figure)
            links.addWidget(link)
            mini = self._mini_figure(operation.plot, figure)
            if mini is not None:
                columns.addWidget(mini, 0, Qt.AlignmentFlag.AlignTop)
            else:  # too narrow for it: the hover is the only look
                panel.setToolTip(figure)
        if operation.path and os.path.isdir(operation.path):
            links.addWidget(_link("Open folder", operation.path))
        elif operation.path:
            gone = ElidedLabel(
                f"Saved to {operation.path}, no longer there", mode=Qt.ElideLeft
            )
            gone.setStyleSheet(f"font-size: 11px; color: {NEUTRAL_550};")
            links.addWidget(gone, 1)
        if links.count():
            links.addStretch(1)
            layout.addLayout(links)
        layout.addStretch(1)
        # in line with the operation's name, past its chevron
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        indent = QHBoxLayout(holder)
        indent.setContentsMargins(24, 0, 0, 0)
        indent.addWidget(panel)
        return holder

    def _mini_figure(self, path: str, tooltip: str) -> Optional[QLabel]:
        """The figure at the panel's right, or None when the panel is too narrow to
        give it room beside the facts."""
        if self._thumb_width < _MIN_WIDTH_FOR_MINI_FIGURE:
            return None
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        mini = ClickableLabel(path)
        mini.setPixmap(
            pixmap.scaled(
                _MINI_FIGURE_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        mini.setToolTip(tooltip)
        mini.clicked.connect(_open_path)
        return mini

    def _image_grid(self, filenames: List[str]) -> QWidget:
        """The run's images as thumbnails, two to a line, each with its caption."""
        img_row = QWidget()
        img_row.setStyleSheet("background: transparent;")
        img_layout = QGridLayout(img_row)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(_SPACING)
        height = self._thumb_width * PLACEHOLDER_HEIGHT // TARGET_WIDTH

        for index, fpath in enumerate(filenames):
            img_label = ClickableLabel(fpath)
            # Connected here, where every thumbnail is made, not when its image
            # loads: a filter change rebuilds the rows from images already
            # loaded, and those never load again.
            img_label.clicked.connect(self._open_expanded)
            img_label.setFixedSize(self._thumb_width, height)
            img_label.setStyleSheet(f"background: {NEUTRAL_900}; border-radius: 4px;")
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setText("Loading...")
            caption = QLabel()
            caption.setStyleSheet(_CAPTION_STYLE)
            tile = QWidget()
            tile.setStyleSheet("background: transparent;")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(0, 0, 0, 0)
            tile_layout.setSpacing(2)
            tile_layout.addWidget(img_label)
            tile_layout.addWidget(caption)
            row, column = divmod(index, _IMAGES_PER_LINE)
            img_layout.addWidget(tile, row, column, Qt.AlignmentFlag.AlignTop)
            self._placeholder_labels[fpath] = img_label
            self._caption_labels[fpath] = caption

        # trailing stretch column, so tiles stay left-aligned
        img_layout.setColumnStretch(_IMAGES_PER_LINE, 1)
        return img_row

    def _on_image_loaded(
        self, filepath: str, arr: np.ndarray, pixel_size_x: float, info: ImageFields
    ) -> None:
        """Slot called on main thread when a background image finishes loading."""
        if info.kind != "Image":
            self._kinds[filepath] = info.kind
        self._captions[filepath] = (image_caption(info), image_summary(info))
        self._show_caption(filepath)
        pixmap = QPixmap.fromImage(
            arr_to_qimage(draw_image_overlays(arr, pixel_size_x))
        )
        self._pixmap_cache[filepath] = pixmap

        label = self._placeholder_labels.get(filepath)
        if label is not None:
            label.setText("")
            label.setFixedSize(pixmap.size())
            label.setPixmap(pixmap)

    def _show_caption(self, filepath: str) -> None:
        """Put the image's caption under its tile, and everything its bar would say
        on hover over either."""
        text, summary = self._captions.get(filepath, ("", ""))
        caption = self._caption_labels.get(filepath)
        if caption is not None:
            caption.setText(text)
            caption.setToolTip(summary)
        tile = self._placeholder_labels.get(filepath)
        if tile is not None:
            tile.setToolTip(summary)

    def _open_expanded(self, filepath: str) -> None:
        """Open the image at full resolution in the image viewer (FIB-1189), with all
        of this lamella's images to step through in task order. Each tile stands in
        for its image until the file is read."""
        paths = list(self._task_names) or [filepath]
        name = self._lamella.name if self._lamella is not None else ""
        items = []
        for path in paths:
            task_name = self._task_names.get(path, "")
            items.append(
                ViewerItem(
                    path=path,
                    title=" › ".join(t for t in (name, task_name) if t)
                    or os.path.basename(path),
                    label=self._kinds.get(path, ""),
                    thumbnail=self._pixmap_cache.get(path),
                )
            )
        open_image_viewer(
            self, items, paths.index(filepath) if filepath in paths else 0
        )
