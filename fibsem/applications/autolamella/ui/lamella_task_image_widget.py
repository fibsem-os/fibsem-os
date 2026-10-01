"""Widget for displaying saved task images for a single lamella."""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from skimage.transform import resize

from fibsem.applications.autolamella.structures import AutoLamellaTaskState, Lamella
from fibsem.applications.autolamella.task_outputs import (
    final_reference_images,
    fluorescence_images,
)
from fibsem.fm.preview import is_fluorescence_image, load_projection
from fibsem.imaging.drawing import draw_image_overlays
from fibsem.structures import FibsemImage
from fibsem.ui.tokens import (
    NEUTRAL_200,
    NEUTRAL_400,
    NEUTRAL_550,
    NEUTRAL_900,
    SURFACE_COLOR,
)

_TARGET_WIDTH = 1024 // 2
_PLACEHOLDER_HEIGHT = 768 // 2  # estimated height for placeholder labels
_MAX_IMAGES_PER_TASK = 2  # last 2 files = highest-res SEM + FIB
_IMAGES_PER_LINE = 2  # tiles per line before wrapping; matches the SEM/FIB pair


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
) -> Tuple[np.ndarray, float]:
    """Load an image and resize to target width, preserving aspect ratio.

    Handles both a plain .tif and a fluorescence z-stack, which becomes an RGB
    channel composite. resize() preserves a trailing channel axis on its own, so
    both shapes go through the same path below.

    Returns:
        Tuple of (resized array, pixel_size_x in metres adjusted for resize).
    """
    if is_fluorescence_image(filepath):
        data, pixel_size_x = load_projection(filepath)
    else:
        img = FibsemImage.load(filepath)
        data = img.data
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., :3].mean(axis=2).astype(data.dtype)
        pixel_size_x = img.metadata.pixel_size.x
    h, w = data.shape[:2]
    # Fit inside the tile box rather than filling its width. Beam images are 3:2 and
    # are width-limited, but a fluorescence stack is square: scaling it to the full
    # width makes it half again as tall as its neighbours, and taller than the
    # placeholder it replaces, so the row jumps when it finishes loading.
    max_height = target_width * _PLACEHOLDER_HEIGHT / _TARGET_WIDTH
    scale = min(target_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = resize(data, (new_h, new_w), preserve_range=True).astype(np.uint8)
    return resized, pixel_size_x / scale


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


class ZoomableImageView(QGraphicsView):
    """QGraphicsView with scroll-to-zoom and drag-to-pan."""

    _ZOOM_FACTOR = 1.05

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._scene.items():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = (
            self._ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self._ZOOM_FACTOR
        )
        self.scale(factor, factor)


class ExpandedImageDialog(QDialog):
    """Modal dialog showing a zoomable/pannable expanded image."""

    _EXPANDED_WIDTH = _TARGET_WIDTH * 2

    def __init__(self, filepath: str, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or os.path.basename(filepath))
        self.setModal(True)
        self.setStyleSheet("background: black;")

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = ZoomableImageView(self)
        layout.addWidget(self._view)

        try:
            arr, pixel_size_x = _load_and_resize(filepath, self._EXPANDED_WIDTH)
            arr = draw_image_overlays(arr, pixel_size_x)
            h, w = arr.shape[:2]
            self._view.set_pixmap(_arr_to_pixmap(arr, w, h))
        except Exception as e:
            logging.warning(f"ExpandedImageDialog: failed to load {filepath}: {e}")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class _ImageLoaderWorker(QThread):
    """Background worker that loads images one at a time."""

    image_loaded = pyqtSignal(str, np.ndarray, float)  # filepath, array, pixel_size_x

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
                arr, pixel_size_x = _load_and_resize(fpath, self._target_width)
                if self._cancel.is_set():
                    return
                self.image_loaded.emit(fpath, arr, pixel_size_x)
            except Exception as e:
                logging.warning(f"Failed to load image {fpath}: {e}")


class LamellaTaskImageWidget(QWidget):
    """Displays final SEM/FIB images for each completed task of a lamella.

    Images are loaded progressively in a background thread so the layout
    appears immediately with gray placeholders that fill in as images load.

    Layout:
        Lamella Name (bold)
        Last Completed Task, completed at timestamp

        Task 1
        [SEM Image]  [FIB Image]

        Task 2
        [SEM Image]  [FIB Image]
        ...
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lamella: Optional[Lamella] = None
        self._lamella_id: Optional[str] = None
        self._pixmap_cache: Dict[str, QPixmap] = {}
        self._placeholder_labels: Dict[str, QLabel] = {}
        self._worker: Optional[_ImageLoaderWorker] = None

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
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 16, 16, 16)
        self._content_layout.setSpacing(12)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

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
        self._placeholder_labels.clear()
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
        """Remove all widgets from the content layout."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild(self) -> None:
        """Build layout with placeholders, then kick off background image loading."""
        self._clear_layout()
        self._placeholder_labels.clear()

        if self._lamella is None:
            label = QLabel("Select a lamella card to view task images.")
            label.setStyleSheet(f"color: {NEUTRAL_550}; font-size: 12px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._content_layout.addWidget(label)
            return

        lamella = self._lamella

        # Header: lamella name
        name_label = QLabel(lamella.name)
        name_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200}; background: transparent;"
        )
        self._content_layout.addWidget(name_label)

        # Subtitle: last completed task + timestamp
        last_task = lamella.last_completed_task
        if last_task is not None:
            subtitle = f"{last_task.name}, completed at {last_task.completed_at}"
        else:
            subtitle = "No completed tasks"
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        self._content_layout.addWidget(subtitle_label)

        # One row per task, accumulating every run of it. Discovery returns the union
        # of what those runs produced, which needs no special-casing: reference images
        # are written to the same filename each run so repeated runs collapse to one
        # set, while fluorescence stacks are uniquely named so every acquisition is
        # kept. Showing only the last run would silently hide earlier FM images.
        runs: Dict[str, List[AutoLamellaTaskState]] = {}
        for t in lamella.task_history:
            runs.setdefault(t.name, []).append(t)
        if not runs:
            no_images = QLabel("No task images available.")
            no_images.setStyleSheet(f"color: {NEUTRAL_550}; font-size: 11px;")
            self._content_layout.addWidget(no_images)
            self._content_layout.addStretch(1)
            return

        # Collect all filepaths to load and build placeholder rows
        all_filepaths: List[str] = []
        for task_name, task_runs in runs.items():
            # cap the reference images *before* appending fluorescence: the cap picks
            # the highest-magnification SEM/FIB pair out of a multi-FOV set, and
            # applying it to a merged list would let a z-stack displace the FIB image.
            filenames = final_reference_images(lamella, *task_runs)[
                -_MAX_IMAGES_PER_TASK:
            ]
            filenames += fluorescence_images(lamella, *task_runs)
            # a task that produced nothing still gets a row saying so: silently
            # omitting it is indistinguishable from the task never having run,
            # which is the confusion this whole feature exists to remove.
            row = self._build_task_row_with_placeholders(task_name, filenames)
            self._content_layout.addWidget(row)
            all_filepaths.extend(filenames)

        self._content_layout.addStretch(1)

        # Filter out already-cached images (set pixmap immediately)
        to_load = []
        for fpath in all_filepaths:
            if fpath in self._pixmap_cache:
                label = self._placeholder_labels.get(fpath)
                if label is not None:
                    label.setPixmap(self._pixmap_cache[fpath])
            else:
                to_load.append(fpath)

        # Start background loader for remaining images
        if to_load:
            self._worker = _ImageLoaderWorker(to_load, _TARGET_WIDTH, parent=self)
            self._worker.image_loaded.connect(self._on_image_loaded)
            self._worker.start()

    def _build_task_row_with_placeholders(
        self, task_name: str, filenames: List[str]
    ) -> QWidget:
        """Build a task row with gray placeholder labels for each image."""
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        # Task label
        task_label = QLabel(task_name)
        task_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {NEUTRAL_400}; background: transparent;"
        )
        layout.addWidget(task_label)

        # Say so explicitly rather than rendering an empty row: a task with a bare
        # heading reads as "still loading", not "produced nothing".
        if not filenames:
            note = QLabel("No images recorded for this task.")
            note.setStyleSheet(
                "font-size: 11px; color: #808080; background: transparent;"
            )
            layout.addWidget(note)
            return container

        # Images wrap onto further lines rather than running off the edge: a task can
        # now produce more than the SEM/FIB pair (a fluorescence stack as well), and
        # the panel scrolls vertically only, so anything past the width is unreachable.
        img_row = QWidget()
        img_row.setStyleSheet("background: transparent;")
        img_layout = QGridLayout(img_row)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(8)

        for index, fpath in enumerate(filenames):
            img_label = ClickableLabel(fpath)
            img_label.setFixedSize(_TARGET_WIDTH, _PLACEHOLDER_HEIGHT)
            img_label.setStyleSheet(f"background: {NEUTRAL_900}; border-radius: 4px;")
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setText("Loading...")
            row, column = divmod(index, _IMAGES_PER_LINE)
            img_layout.addWidget(img_label, row, column)
            self._placeholder_labels[fpath] = img_label

        # trailing stretch column, so tiles stay left-aligned as before
        img_layout.setColumnStretch(_IMAGES_PER_LINE, 1)
        layout.addWidget(img_row)

        return container

    def _on_image_loaded(
        self, filepath: str, arr: np.ndarray, pixel_size_x: float
    ) -> None:
        """Slot called on main thread when a background image finishes loading."""
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

    def _open_expanded(self, filepath: str) -> None:
        """Open an expanded, zoomable/pannable view of the image."""
        dlg = ExpandedImageDialog(
            filepath, title=os.path.basename(filepath), parent=self
        )
        dlg.exec_()
