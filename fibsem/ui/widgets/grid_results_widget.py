"""Per-grid results view — the Results sub-tab of the Grids tab.

Shows, for the selected GridRecord: a card-style summary header, the stitched
overview image (hero, click to zoom), the task history, and a gallery of
per-task artifacts. Image paths come from GridRecord.results (written by tasks
via GridTask.record_result); task history from GridRecord.task_history.

Reuses the lamella Review widget's image helpers (lazy-load worker + zoom dialog)
and the app's TitledPanel for consistent structure. Loaded image arrays are
cached by (path, mtime) so the (large) overview isn't re-read on every refresh
during a workflow run.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.widgets.grid_card_widget import _GridThumbnail
from fibsem.ui.widgets.lamella_task_image_widget import (
    ClickableLabel,
    ExpandedImageDialog,
    _ImageLoaderWorker,
    _arr_to_pixmap,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, GridRecord

_HERO_WIDTH = 520
_THUMB_WIDTH = 150
_GALLERY_COLS = max(1, _HERO_WIDTH // (_THUMB_WIDTH + 12))
_IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

_MUTED = "#a0a0a0"
_HINT = "#808080"
_IMG_BG = "#1a1b1e"
_GREEN = "#4caf50"

_STATUS_COLORS = {
    "Completed": "#4caf50",
    "Failed": "#e24b4a",
    "Skipped": "#a0a0a0",
    "InProgress": "#e0a030",
    "NotStarted": "#808080",
}


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; padding: 4px 2px 2px; background: transparent;")
    return lbl


class _ClickableRow(QWidget):
    """A selectable task-history row that emits its task key on click."""

    clicked = pyqtSignal(str)

    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # a plain QWidget subclass won't paint its stylesheet background otherwise
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # super() first, emit last — the click can rebuild + delete this widget
        # (selection refresh / modal exec_), so don't touch self after emitting.
        super().mousePressEvent(event)
        self.clicked.emit(self._key)


class _HeroImage(ClickableLabel):
    """Clickable overview image that rescales to fill its space (keeps aspect)."""

    def __init__(self, filepath: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(filepath, parent)
        self._arr: Optional[np.ndarray] = None
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {_IMG_BG}; border-radius: 4px;")
        self.setText("loading…")

    def set_array(self, arr: np.ndarray) -> None:
        self._arr = arr
        self.setText("")
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._arr is None:
            return
        w, h = max(1, self.width()), max(1, self.height())
        self.setPixmap(_arr_to_pixmap(self._arr, w, h))


def _task_display_name(ts) -> str:
    """Friendly task name (e.g. 'Acquire Overview Image') from the registry,
    falling back to the stored task name."""
    from fibsem.applications.autolamella.workflows.tasks.grid import (
        GRID_TASK_REGISTRY,
    )

    task_cls = GRID_TASK_REGISTRY.get(ts.task_type)
    if task_cls is not None:
        return getattr(task_cls.config_cls, "display_name", ts.name) or ts.name
    return ts.name or "—"


class GridResultsWidget(QWidget):
    """Summary + overview + task history + artifacts for one GridRecord."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._record: Optional["GridRecord"] = None
        self._experiment: Optional["Experiment"] = None
        self._slot_label: str = ""
        self._in_beam: bool = False
        self._worker: Optional[_ImageLoaderWorker] = None
        # filepath -> [(label, display_width), ...] to fill when the image loads
        self._targets: Dict[str, List[Tuple[QLabel, int]]] = {}
        # path -> (mtime, array) so the overview isn't re-read on every refresh
        self._img_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        # the responsive overview image + its path (loaded via the worker too)
        self._hero: Optional[_HeroImage] = None
        self._hero_path: Optional[str] = None
        # selected task (task_name) → filters the artifacts to that task
        self._selected_task: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        self._content_layout.setSpacing(10)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)

        self._empty = QLabel("Select a grid to view results.")
        self._empty.setStyleSheet(f"color: {_HINT}; padding: 24px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addWidget(self._empty)

    # --- public API ---

    def set_grid(self, record: Optional["GridRecord"],
                 experiment: Optional["Experiment"] = None,
                 slot_label: str = "", in_beam: bool = False) -> None:
        # reset the task filter only when switching to a different grid (not on
        # the live refreshes that happen while a grid is running)
        prev = self._record.name if self._record is not None else None
        new = record.name if record is not None else None
        if prev != new:
            self._selected_task = None
        self._record = record
        if experiment is not None:
            self._experiment = experiment
        self._slot_label = slot_label
        self._in_beam = in_beam
        self._rebuild()

    # --- rebuild ---

    def _rebuild(self) -> None:
        self._cancel_worker()
        self._clear()
        self._targets = {}
        self._hero = None
        self._hero_path = None

        if self._record is None:
            self._content_layout.addWidget(self._empty)
            self._empty.setVisible(True)
            return
        self._empty.setVisible(False)

        self._content_layout.addWidget(self._build_header())
        self._content_layout.addWidget(self._build_overview_section())

        # task history (left, natural width) + artifacts (right, fills) on one row
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(14)
        h.setAlignment(Qt.AlignmentFlag.AlignTop)
        h.addWidget(self._build_history_section(), 0)
        h.addWidget(self._build_gallery_section(), 1)
        self._content_layout.addWidget(row)

        self._fill_or_queue()

    # --- header (echoes the card) ---

    def _build_header(self) -> QWidget:
        rec = self._record
        wrap = QWidget()
        wrap.setStyleSheet(
            "background: #2b2d31; border: 1px solid #3a3d42; border-radius: 8px;"
        )
        v = QVBoxLayout(wrap)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)
        badge = QLabel(self._slot_label or "—")
        badge.setToolTip("Loader magazine slot")
        badge_color = "#2d3f5c" if self._in_beam else "#3a3d42"
        badge.setStyleSheet(
            f"background: {badge_color}; color: #d6d6d6; font-size: 11px;"
            " font-weight: 500; padding: 1px 7px; border-radius: 4px;"
        )
        row.addWidget(badge)
        name = QLabel(rec.name)
        name.setStyleSheet("font-size: 16px; font-weight: 500; background: transparent;")
        row.addWidget(name)
        row.addStretch(1)
        if self._in_beam:
            pill = QLabel("● In microscope")
            pill.setStyleSheet(
                f"background: rgba(27,58,42,220); color: {_GREEN}; font-size: 11px;"
                " font-weight: 500; padding: 1px 8px; border-radius: 8px;"
            )
            row.addWidget(pill)
        v.addLayout(row)

        sub = QLabel(self._summary_text())
        sub.setStyleSheet(f"color: {_MUTED}; background: transparent;")
        v.addWidget(sub)
        return wrap

    def _summary_text(self) -> str:
        rec = self._record
        completed = sum(1 for ts in rec.task_history if ts.status.name == "Completed")
        failed = sum(1 for ts in rec.task_history if ts.status.name == "Failed")
        if not rec.task_history:
            return "Not started"
        parts = [f"{completed} task{'s' if completed != 1 else ''} complete"]
        if failed:
            parts.append(f"{failed} failed")
        last = rec.task_history[-1]
        if last.end_timestamp:
            parts.append("last run " + datetime.fromtimestamp(last.end_timestamp).strftime("%H:%M:%S"))
        return " · ".join(parts)

    # --- overview ---

    def _build_overview_section(self) -> QWidget:
        col = QWidget()
        v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.addWidget(_section_label("Overview"))
        overview = self._overview_path()
        if overview:
            self._hero = _HeroImage(overview)
            self._hero.setMaximumHeight(380)
            self._hero.setToolTip("Click to zoom")
            self._hero.clicked.connect(self._open_image)
            self._hero_path = overview
            v.addWidget(self._hero)
        else:
            thumb = _GridThumbnail(_HERO_WIDTH, int(_HERO_WIDTH * 0.55))
            v.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignLeft)
        return col

    # --- task history ---

    def _latest_per_task(self) -> List:
        """Latest task_history entry per task (task_name key), in first-seen order."""
        latest: Dict[str, object] = {}
        for ts in self._record.task_history:
            latest[ts.name] = ts  # later entries overwrite → latest wins
        return list(latest.values())

    def _build_history_section(self) -> QWidget:
        col = QWidget()
        v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(_section_label("Task history"))
        rows = self._latest_per_task()
        if not rows:
            empty = QLabel("No tasks run yet.")
            empty.setStyleSheet(f"color: {_HINT}; padding: 8px 2px;")
            v.addWidget(empty)
        else:
            for ts in rows:
                v.addWidget(self._history_row(ts))
        v.addStretch(1)
        return col

    def _history_row(self, ts) -> QWidget:
        selected = ts.name == self._selected_task
        row = _ClickableRow(ts.name)
        row.clicked.connect(self._on_task_clicked)
        if selected:
            row.setStyleSheet("_ClickableRow { background: #2d3f5c; border-radius: 3px; }")
        else:
            row.setStyleSheet(
                "_ClickableRow:hover { background: rgba(255,255,255,18); border-radius: 3px; }"
            )
        # surface the (long) error message as a tooltip rather than inline text
        tip = ts.status_message if (ts.status.name == "Failed" and ts.status_message) else ""
        if tip:
            row.setToolTip(tip)
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)

        color = _STATUS_COLORS.get(ts.status.name, _HINT)
        dot = QLabel("●")
        dot.setAttribute(Qt.WA_TransparentForMouseEvents)
        dot.setStyleSheet(f"color: {color}; background: transparent;")
        h.addWidget(dot)

        name = QLabel(_task_display_name(ts))
        name.setFixedWidth(180)
        name.setAttribute(Qt.WA_TransparentForMouseEvents)
        name.setStyleSheet("background: transparent;")
        h.addWidget(name)

        status = QLabel(ts.status.name)
        status.setFixedWidth(90)
        status.setAttribute(Qt.WA_TransparentForMouseEvents)
        status.setStyleSheet(f"color: {color}; background: transparent;")
        h.addWidget(status)

        meta = []
        try:
            if ts.duration:
                meta.append(f"{ts.duration:.1f}s")
        except Exception:
            pass
        if ts.end_timestamp:
            meta.append(datetime.fromtimestamp(ts.end_timestamp).strftime("%H:%M:%S"))
        info = QLabel(" · ".join(meta))
        info.setAttribute(Qt.WA_TransparentForMouseEvents)
        info.setStyleSheet(f"color: {_HINT}; background: transparent;")
        h.addWidget(info)
        h.addStretch(1)
        return row

    def _on_task_clicked(self, task_name: str) -> None:
        # toggle the filter; rebuild (cached images apply instantly)
        self._selected_task = None if self._selected_task == task_name else task_name
        self._rebuild()

    # --- artifacts gallery (wrapping grid) ---

    def _build_gallery_section(self) -> QWidget:
        col = QWidget()
        v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(_section_label("Artifacts"))
        if self._selected_task is not None:
            scope = QLabel("— click the task again to show all")
            scope.setStyleSheet(f"color: {_HINT}; padding: 4px 2px 2px;")
            hdr.addWidget(scope)
        hdr.addStretch(1)
        v.addLayout(hdr)

        items = self._gallery_items(self._selected_task)
        if not items:
            msg = ("No additional artifacts for this task."
                   if self._selected_task is not None else "No artifacts yet.")
            empty = QLabel(msg)
            empty.setStyleSheet(f"color: {_HINT}; padding: 8px 2px;")
            v.addWidget(empty)
            v.addStretch(1)
            return col

        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for i, (label, path) in enumerate(items):
            cell = QWidget()
            c = QVBoxLayout(cell)
            c.setContentsMargins(0, 0, 0, 0)
            c.setSpacing(2)
            c.addWidget(self._image_label(path, _THUMB_WIDTH, clickable=True))
            cap = QLabel(label)
            cap.setStyleSheet(f"color: {_HINT}; font-size: 11px;")
            cap.setMaximumWidth(_THUMB_WIDTH)
            cap.setWordWrap(True)
            c.addWidget(cap)
            grid.addWidget(cell, i // _GALLERY_COLS, i % _GALLERY_COLS)
        v.addWidget(grid_w)
        v.addStretch(1)
        return col

    # --- image label + caching ---

    def _image_label(self, filepath: str, width: int, clickable: bool) -> QLabel:
        lbl = ClickableLabel(filepath) if clickable else QLabel()
        lbl.setFixedWidth(width)
        lbl.setMinimumHeight(int(width * 0.55))
        lbl.setStyleSheet(f"background: {_IMG_BG}; border-radius: 4px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setText("loading…")
        if clickable:
            lbl.clicked.connect(self._open_image)
        self._targets.setdefault(filepath, []).append((lbl, width))
        return lbl

    def _fill_or_queue(self) -> None:
        """Fill targets from the cache; queue uncached/changed images for loading."""
        paths = set(self._targets)
        if self._hero_path:
            paths.add(self._hero_path)
        to_load: List[str] = []
        for path in paths:
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
            cached = self._img_cache.get(path)
            if cached is not None and mtime is not None and cached[0] == mtime:
                self._apply_array(path, cached[1])
            else:
                to_load.append(path)
        if to_load:
            self._worker = _ImageLoaderWorker(to_load, _HERO_WIDTH, parent=self)
            self._worker.image_loaded.connect(self._on_image_loaded)
            self._worker.start()

    def _apply_array(self, path: str, arr: np.ndarray) -> None:
        if self._hero is not None and path == self._hero_path:
            self._hero.set_array(arr)
        for lbl, width in self._targets.get(path, []):
            w = min(width, arr.shape[1]) if arr.shape[1] else width
            h = int(arr.shape[0] * w / arr.shape[1]) if arr.shape[1] else w
            lbl.setText("")
            lbl.setPixmap(_arr_to_pixmap(arr, w, h))
            lbl.setFixedHeight(h)

    def _on_image_loaded(self, filepath: str, arr: np.ndarray, _pixel_size: float) -> None:
        mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0
        self._img_cache[filepath] = (mtime, arr)
        self._apply_array(filepath, arr)

    # --- helpers ---

    def _overview_path(self) -> Optional[str]:
        for art in self._record.results.values():
            if isinstance(art, dict) and art.get("overview"):
                return art["overview"]
        return None

    def _gallery_items(self, task_filter: Optional[str] = None) -> List[Tuple[str, str]]:
        from fibsem.applications.autolamella.workflows.tasks.grid import (
            GRID_TASK_REGISTRY,
        )

        overview = self._overview_path()
        items: List[Tuple[str, str]] = []
        for task_name, art in self._record.results.items():
            if not isinstance(art, dict):
                continue
            if task_filter is not None and task_name != task_filter:
                continue
            task_cls = GRID_TASK_REGISTRY.get(task_name)
            display = getattr(task_cls.config_cls, "display_name", task_name) if task_cls else task_name
            for key, val in art.items():
                if key == "thumbnail" or val == overview:
                    continue
                if isinstance(val, str) and val.lower().endswith(_IMAGE_EXTS):
                    items.append((f"{display} · {key}", val))
        return items

    def _open_image(self, filepath: str) -> None:
        ExpandedImageDialog(filepath, parent=self).exec_()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.image_loaded.disconnect(self._on_image_loaded)
            except Exception:
                pass
            self._worker.cancel()
            self._worker.wait()  # block until the thread stops (avoids teardown races)
            self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel_worker()
        super().closeEvent(event)

    def _clear(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.deleteLater()
            elif item.layout() is not None:
                self._delete_layout(item.layout())

    def _delete_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self._delete_layout(item.layout())
