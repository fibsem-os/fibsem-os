"""The History tab: one row per task run on a lamella, with its operations and images."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
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
from skimage.transform import resize

from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME, read_events
from fibsem.applications.autolamella.history_rows import (
    HistoryFilter,
    HistoryOperation,
    HistoryRun,
    filter_runs,
    history_rows,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, Lamella
from fibsem.cancellation import CANCELLED, COMPLETED, FAILED, SKIPPED
from fibsem.config import load_user_preferences, update_user_preferences
from fibsem.fm.preview import composite_projection, is_fluorescence_image
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.drawing import draw_image_overlays
from fibsem.imaging.export import (
    ImageFields,
    image_caption,
    image_fields,
    image_summary,
)
from fibsem.structures import FibsemImage
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
    SURFACE_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.ui.widgets.image_viewer_dialog import ViewerItem, open_image_viewer
from fibsem.ui.widgets.task_summary_formatting import STATUS_BADGE_COLORS
from fibsem.util.timestamps import format_time

# Thumbnails, two to a line in the panel's width: a run is a few lines and a
# pair of pictures, and the full image is a click away in the viewer. The width
# follows the panel between these bounds; _TARGET_WIDTH is also the loader's
# default.
_TARGET_WIDTH = 190
_PLACEHOLDER_HEIGHT = 142  # 3:4 of the width, the box a 3:2 beam image fits
_MIN_THUMB_WIDTH = 120
_MAX_THUMB_WIDTH = 280
_MARGIN = 16
_SPACING = 8
# The runs stay a column rather than spreading across a wide panel, so a line's
# duration sits beside what it times.
_MAX_CONTENT_WIDTH = 2 * _MAX_THUMB_WIDTH + _SPACING + 2 * _MARGIN
_MAX_IMAGES_PER_TASK = 2  # last 2 files = highest-res SEM + FIB
_IMAGES_PER_LINE = 2  # tiles per line before wrapping; matches the SEM/FIB pair
_CAPTION_STYLE = (
    f"color: {NEUTRAL_550}; font-family: {NUMBER_FONT}; font-size: 11px;"
    " background: transparent;"
)


def _arr_to_pixmap(arr: np.ndarray, w: int, h: int) -> QPixmap:
    """Convert a numpy array to a QPixmap scaled to (w, h)."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    ih, iw, c = arr.shape
    qimg = QImage(arr.data, iw, ih, iw * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w,
        h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _load_and_resize(
    filepath: str, target_width: int = _TARGET_WIDTH
) -> Tuple[np.ndarray, float, ImageFields]:
    """Load an image and resize to target width, preserving aspect ratio.

    Handles both a plain .tif and a fluorescence z-stack, which becomes an RGB
    channel composite. resize() preserves a trailing channel axis on its own, so
    both shapes go through the same path below.

    Returns:
        Tuple of (resized array, pixel_size_x in metres adjusted for resize, the
        image's fields, as its bar would show them). The fields come from the file
        already open for the pixels, so a caption costs no second read.
    """
    if is_fluorescence_image(filepath):
        stack = FluorescenceImage.load(filepath)
        pixel_size_x = stack.metadata.pixel_size_x
        info = image_fields(stack)
        try:
            data = composite_projection(stack)
        except ValueError as e:
            raise ValueError(f"{e}: {filepath}") from e
        finally:
            # A real METEOR stack is ~530 MB in memory where its projection is
            # ~25 MB, and the loader reads several in a row: keep only the latter.
            del stack
    else:
        img = FibsemImage.load(filepath)
        data = img.data
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., :3].mean(axis=2).astype(data.dtype)
        pixel_size_x = img.metadata.pixel_size.x
        info = image_fields(img)
    h, w = data.shape[:2]
    # Fit inside the tile box rather than filling its width. Beam images are 3:2 and
    # are width-limited, but a fluorescence stack is square: scaling it to the full
    # width makes it half again as tall as its neighbours, and taller than the
    # placeholder it replaces, so the row jumps when it finishes loading.
    max_height = target_width * _PLACEHOLDER_HEIGHT / _TARGET_WIDTH
    scale = min(target_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = resize(data, (new_h, new_w), preserve_range=True).astype(np.uint8)
    return resized, pixel_size_x / scale, info


class ClickableLabel(QLabel):
    """QLabel that emits clicked(filepath) when left-clicked."""

    clicked = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None) -> None:
        super().__init__(parent)
        self._filepath = filepath
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._filepath)
        super().mousePressEvent(event)


class _ImageLoaderWorker(QThread):
    """Background worker that loads images one at a time."""

    # filepath, array, pixel_size_x, ImageFields
    image_loaded = pyqtSignal(str, np.ndarray, float, object)

    def __init__(self, filepaths: List[str], target_width: int, parent=None):
        super().__init__(parent)
        self._filepaths = filepaths
        self._target_width = target_width
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        for fpath in self._filepaths:
            if self._cancel.is_set():
                return
            try:
                arr, pixel_size_x, info = _load_and_resize(fpath, self._target_width)
                if self._cancel.is_set():
                    return
                self.image_loaded.emit(fpath, arr, pixel_size_x, info)
            except Exception as e:
                logging.warning(f"Failed to load image {fpath}: {e}")


# Status words and colours, as the run's header shows them. The colours are the
# task-summary badges, so a run reads the same here as in the workflow summary.
_RUN_STATUS = {
    AutoLamellaTaskStatus.Completed: ("Completed", "mdi:check", "Completed"),
    AutoLamellaTaskStatus.Failed: ("Failed", "mdi:close", "Failed"),
    AutoLamellaTaskStatus.Cancelled: ("Cancelled", "mdi:stop", "Cancelled"),
    AutoLamellaTaskStatus.InProgress: (
        "In progress",
        "mdi:progress-clock",
        "InProgress",
    ),
    AutoLamellaTaskStatus.AwaitingDecision: (
        "Awaiting decision",
        "mdi:progress-clock",
        "AwaitingDecision",
    ),
    AutoLamellaTaskStatus.Skipped: ("Skipped", "mdi:skip-next", "Skipped"),
}
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

    Behind an icon, as in the Review tab: set once and left, so not worth a row
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
        self._menu = QMenu(self)
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
        self._placeholder_labels: Dict[str, QLabel] = {}
        self._caption_labels: Dict[str, QLabel] = {}
        self._task_names: Dict[str, str] = {}  # image path -> the task that made it
        self._kinds: Dict[str, str] = {}  # image path -> SEM / FIB / FM, once read
        # image path -> (caption, tooltip), kept with the pixmaps they came with
        self._captions: Dict[str, Tuple[str, str]] = {}
        self._worker: Optional[_ImageLoaderWorker] = None
        self._runs: List[HistoryRun] = []
        # Whether the experiment records its events: without the file nothing
        # was ever recorded, so "no operations" would be a claim it cannot make.
        self._has_events = False
        self._thumb_width = _TARGET_WIDTH
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
        self._captions.clear()
        self._placeholder_labels.clear()
        self._rebuild()

    def refresh(self) -> None:
        """Rebuild for the lamella already shown, after a task on it finished.

        `set_lamella` skips the lamella it already shows, so without this the
        panel kept the history it was opened with until another lamella was
        selected and back (FIB-1111). The image cache goes too: a re-run writes
        its reference images to the same filenames, so a cached pixmap would be
        the previous run's picture.
        """
        if self._lamella is None:
            return
        self._cancel_worker()
        self._pixmap_cache.clear()
        self._captions.clear()
        self._rebuild()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.image_loaded.disconnect(self._on_image_loaded)
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait(2000)
            self._worker = None

    def _clear_layout(self) -> None:
        """Remove all widgets from the content layout, keeping the filter button."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is self._filter_button:
                continue
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
        self.refresh()

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
        self._has_events = path.exists()
        if not self._has_events:
            return []
        try:
            return list(read_events(path))
        except OSError as e:
            logging.warning(f"Could not read {path}: {e}")
            return []

    def _rebuild(self) -> None:
        """Build the rows with placeholders, then load the images in the background."""
        self._clear_layout()
        self._placeholder_labels.clear()
        self._caption_labels.clear()
        self._task_names.clear()

        if self._lamella is None:
            self._filter_button.hide()
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

        if to_load:
            self._worker = _ImageLoaderWorker(to_load, self._thumb_width, parent=self)
            self._worker.image_loaded.connect(self._on_image_loaded)
            self._worker.start()

    def _header(self, name: str, runs: List[HistoryRun], flt: HistoryFilter) -> QWidget:
        """The lamella's name, a line about its runs, and the filter at the right."""
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
        if flt.active:
            line = f"Filtered · {len(runs)} of {len(self._runs)} runs"
            color = ACCENT_COLOR
        else:
            total = sum(run.duration or 0.0 for run in self._runs)
            count = len(self._runs)
            line = f"{count} run{'' if count == 1 else 's'}"
            if total:
                line += f" · {_duration(total)}"
            color = NEUTRAL_550
        self._subtitle = _text(line, f"font-size: 11px; color: {color};")
        text.addWidget(self._subtitle)
        layout.addLayout(text, 1)
        layout.addWidget(self._filter_button, 0, Qt.AlignmentFlag.AlignTop)
        self._filter_button.show()
        return header

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
        top.setSpacing(6)
        top.addWidget(
            _text(
                run.task_name,
                f"font-size: 12px; font-weight: 600; color: {NEUTRAL_400};",
            ),
            1,
        )
        top.addWidget(_icon_label(icon, colour, 12))
        top.addWidget(_text(word, f"font-size: 11px; color: {colour};"))
        layout.addLayout(top)
        when = " · ".join(
            part for part in (_clock(run.started_at), _duration(run.duration)) if part
        )
        if when:
            layout.addWidget(_text(when, _CAPTION_STYLE))
        if run.status_message and run.status in (
            AutoLamellaTaskStatus.Failed,
            AutoLamellaTaskStatus.Cancelled,
        ):
            message = _text(run.status_message, f"font-size: 11px; color: {colour};")
            message.setWordWrap(True)
            layout.addWidget(message)

        if flt.show != "images":
            for operation in run.operations:
                layout.addWidget(self._operation_line(operation))
            if not run.operations and flt.show == "all" and self._has_events:
                layout.addWidget(
                    _text(
                        "No operations recorded",
                        f"font-size: 11px; color: {NEUTRAL_550};",
                    )
                )

        if flt.show != "operations":
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

    def _operation_line(self, operation: HistoryOperation) -> QWidget:
        """``✓ Align   1.11 µm · 3 steps   6 s``, its full text on hover."""
        line = QWidget()
        line.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(line)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        icon, badge = _OPERATION_ICON.get(operation.status, ("mdi:circle-outline", ""))
        colour = STATUS_BADGE_COLORS.get(badge, (NEUTRAL_550, NEUTRAL_550))[0]
        layout.addWidget(_icon_label(icon, colour))
        layout.addWidget(
            _text(operation.label, f"font-size: 12px; color: {NEUTRAL_200};")
        )
        detail = ElidedLabel(operation.detail)
        detail.setStyleSheet(_CAPTION_STYLE)
        detail.setToolTip(operation.detail)
        layout.addWidget(detail, 1)
        layout.addWidget(_text(_duration(operation.duration), _CAPTION_STYLE))
        return line

    def _image_grid(self, filenames: List[str]) -> QWidget:
        """The run's images as thumbnails, two to a line, each with its caption."""
        img_row = QWidget()
        img_row.setStyleSheet("background: transparent;")
        img_layout = QGridLayout(img_row)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(_SPACING)
        height = self._thumb_width * _PLACEHOLDER_HEIGHT // _TARGET_WIDTH

        for index, fpath in enumerate(filenames):
            img_label = ClickableLabel(fpath)
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
        arr = draw_image_overlays(arr, pixel_size_x)
        h, w = arr.shape[:2]
        pixmap = _arr_to_pixmap(arr, w, h)
        self._pixmap_cache[filepath] = pixmap

        label = self._placeholder_labels.get(filepath)
        if label is not None:
            label.setText("")
            label.setFixedSize(pixmap.size())
            label.setPixmap(pixmap)
            if isinstance(label, ClickableLabel):
                label.clicked.connect(self._open_expanded)

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
