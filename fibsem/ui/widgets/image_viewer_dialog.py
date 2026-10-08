"""One window to look at a saved image, on the real canvas at full resolution (FIB-1189).

A beam image opens on a :class:`FibsemImageCanvas`, which brings the microscope tab's
zoom, fit, reset, live scalebar, contrast, ruler and crosshair. A fluorescence stack
opens on the :class:`FMCanvasWidget`, with its z-slider, max projection and per-channel
controls; up and down step through the planes. Under either sits the view's bar
(FIB-1186), so the image says what it is in the same words as the quad view, with the
same field choices.

The full-resolution file loads off the GUI thread. Until it arrives the window shows
the caller's thumbnail, so a click answers at once even for a stitched overview.

:func:`open_image_viewer` keeps one non-modal window per application window and
reuses it, so it can stay open beside the app while you click through tiles.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Set, Union

from PyQt5 import sip
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.preview import is_fluorescence_image
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.export import (
    from_fibsem_image,
    from_fluorescence_image,
    image_fields,
    z_stack,
)
from fibsem.structures import FibsemImage
from fibsem.ui.stylesheets import NAPARI_STYLE, SECONDARY_BUTTON_STYLESHEET
from fibsem.ui.tokens import CANVAS_BG, TEXT_MUTED_COLOR, TEXT_STRONG_COLOR
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.view_info_bar import ViewInfoBar, show_fm_plane

ViewerImage = Union[FibsemImage, FluorescenceImage]

_TITLE_STYLE = f"color: {TEXT_STRONG_COLOR}; font-size: 13px; font-weight: 600;"
_LOADING_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 12px; background: {CANVAS_BG};"

_LOADING, _BEAM, _FM = 0, 1, 2  # the stack's pages


def load_viewer_image(path: str) -> ViewerImage:
    """The image at *path*: a fluorescence stack as a stack, anything else as a beam
    image. Full resolution; nothing is resized."""
    if is_fluorescence_image(path):
        return FluorescenceImage.load(path)
    return FibsemImage.load(path)


class _Loader(QThread):
    """Reads one file off the GUI thread; emits the image, or the error."""

    loaded = pyqtSignal(str, object)  # path, image or Exception

    def __init__(self, path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            result = load_viewer_image(self.path)
        except Exception as e:  # reported on the GUI thread, by the viewer
            result = e
        self.loaded.emit(self.path, result)


class ImageViewer(QWidget):
    """A saved image on the real canvas, with its bar, a title and Export."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[ViewerImage] = None
        self._path: Optional[str] = None
        self._pending: Optional[str] = None  # the load whose result we still want
        self._loaders: Set[_Loader] = set()
        self._fm_stack = None
        self._placeholder: Optional[QPixmap] = None
        # A read still running when the app quits must finish first: Qt aborts on a
        # thread destroyed while it runs.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_loaders)

        self.title_label = QLabel()
        self.title_label.setStyleSheet(_TITLE_STYLE)
        self.export_button = QPushButton("Export…")
        self.export_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.export_button.setToolTip("Export this image with its metadata")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 10, 6)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.export_button)

        self.loading_label = QLabel(alignment=Qt.AlignCenter)
        self.loading_label.setStyleSheet(_LOADING_STYLE)

        # A beam image: the canvas, then the bar of its kind. The field picker's
        # choices are kept per kind, so SEM and FIB each keep their own bar.
        self.canvas = FibsemImageCanvas()
        self.sem_bar = ViewInfoBar("SEM")
        self.fib_bar = ViewInfoBar("FIB")
        beam_page = QWidget()
        beam_layout = QVBoxLayout(beam_page)
        beam_layout.setContentsMargins(0, 0, 0, 0)
        beam_layout.setSpacing(0)
        beam_layout.addWidget(self.canvas, 1)
        beam_layout.addWidget(self.sem_bar)
        beam_layout.addWidget(self.fib_bar)

        # A stack: the FM viewer, then its bar under the z row, as in the quad view.
        self.fm_widget = FMCanvasWidget()
        self.fm_bar = ViewInfoBar("FM")
        self.fm_widget.z_display_changed.connect(self._refresh_fm_plane)
        fm_page = QWidget()
        fm_layout = QVBoxLayout(fm_page)
        fm_layout.setContentsMargins(0, 0, 0, 0)
        fm_layout.setSpacing(0)
        fm_layout.addWidget(self.fm_widget, 1)
        fm_layout.addWidget(self.fm_bar)

        self.stack = QStackedWidget()
        self.stack.insertWidget(_LOADING, self.loading_label)
        self.stack.insertWidget(_BEAM, beam_page)
        self.stack.insertWidget(_FM, fm_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.stack, 1)

    # ── what is shown ──────────────────────────────────────────────────────
    @property
    def image(self) -> Optional[ViewerImage]:
        return self._image

    @property
    def path(self) -> Optional[str]:
        """The file the image came from, or None for one handed over in memory."""
        return self._path

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def show_path(self, path: str, placeholder: Optional[QPixmap] = None) -> None:
        """Open the image at *path*, at full resolution, read off the GUI thread.

        *placeholder* -- the tile that was clicked -- shows until the file arrives.
        A later call wins: a slow read that lands after it is dropped.
        """
        self._pending = path
        self.export_button.setEnabled(False)
        has_placeholder = placeholder is not None and not placeholder.isNull()
        self._placeholder = placeholder if has_placeholder else None
        self.loading_label.setText(
            "" if has_placeholder else (f"Loading {os.path.basename(path)}…")
        )
        self._fit_placeholder()
        self.stack.setCurrentIndex(_LOADING)

        loader = _Loader(path, self)
        loader.loaded.connect(self._on_loaded)
        loader.finished.connect(lambda: self._loaders.discard(loader))
        loader.finished.connect(loader.deleteLater)
        self._loaders.add(loader)
        loader.start()

    def _fit_placeholder(self) -> None:
        """Scale the waiting thumbnail to the room it has, which a window not yet
        shown does not know."""
        if self._placeholder is None:
            self.loading_label.setPixmap(QPixmap())
            return
        self.loading_label.setPixmap(
            self._placeholder.scaled(
                self.stack.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self.stack.currentIndex() == _LOADING:
            self._fit_placeholder()

    def _wait_for_loaders(self) -> None:
        for loader in list(self._loaders):
            loader.wait(5000)

    def _on_loaded(self, path: str, result) -> None:
        if path != self._pending:
            return  # superseded by a later click
        self._pending = None
        if isinstance(result, Exception):
            logging.warning("Could not open %s: %s", path, result)
            self._placeholder = None
            self.loading_label.setPixmap(QPixmap())
            self.loading_label.setText(
                f"Couldn't open {os.path.basename(path)}: {result}"
            )
            return
        self.show_image(result, path=path)

    def show_image(self, image: ViewerImage, path: Optional[str] = None) -> None:
        """Show *image*: a stack on the FM viewer, anything else on the beam canvas."""
        self._image, self._path = image, path
        if isinstance(image, FluorescenceImage):
            self.fm_widget.set_fm_image(image)
            self.fm_bar.set_image_fields(image_fields(image))
            self._fm_stack = z_stack(image)
            self._refresh_fm_plane()
            self.stack.setCurrentIndex(_FM)
        else:
            self.canvas.set_image(image)
            info = image_fields(image)
            # A file that does not say what it is gets no bar: "SEM" would be a guess.
            self.sem_bar.setVisible(info.kind == "SEM")
            self.fib_bar.setVisible(info.kind == "FIB")
            if info.kind in ("SEM", "FIB"):
                bar = self.fib_bar if info.kind == "FIB" else self.sem_bar
                bar.set_image_fields(info)
            self.stack.setCurrentIndex(_BEAM)
        self.export_button.setEnabled(True)

    def _refresh_fm_plane(self) -> None:
        show_fm_plane(self.fm_bar, self.fm_widget, self._fm_stack)

    # ── actions ────────────────────────────────────────────────────────────
    def export(self) -> None:
        """Open the image export dialog for the image shown."""
        from fibsem.ui.widgets.image_export_dialog import ImageExportDialog

        image = self._image
        if image is None:
            return
        if isinstance(image, FluorescenceImage):
            export_image = from_fluorescence_image(image, self._path)
        else:
            export_image = from_fibsem_image(image, self._path)
        ImageExportDialog(export_image, self).exec_()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Up and down step through a stack's planes."""
        if self.stack.currentIndex() == _FM and event.key() in (Qt.Key_Up, Qt.Key_Down):
            self.fm_widget.step_z(1 if event.key() == Qt.Key_Up else -1)
            return
        super().keyPressEvent(event)


class ImageViewerDialog(QDialog):
    """The viewer in its own window. Non-modal: the app stays usable beside it."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Image")
        # Its own window: carry the dark theme rather than rely on the parent's sheet.
        self.setStyleSheet(NAPARI_STYLE)
        self.viewer = ImageViewer(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.7), int(screen.height() * 0.75))

    def show_path(
        self, path: str, title: str = "", placeholder: Optional[QPixmap] = None
    ) -> None:
        """Show the image at *path* and bring the window forward."""
        self.setWindowTitle(title or os.path.basename(path))
        self.viewer.set_title(title or os.path.basename(path))
        self.viewer.show_path(path, placeholder)
        self.show()
        self.raise_()
        self.activateWindow()
        self.viewer.setFocus()


_DIALOG_ATTRIBUTE = "_fibsem_image_viewer"


def open_image_viewer(
    parent: QWidget, path: str, title: str = "", placeholder: Optional[QPixmap] = None
) -> ImageViewerDialog:
    """Open *path* in the viewer of *parent*'s window, making it on first use.

    One per application window, reused: a second click shows the new image in the
    window already open rather than stacking another on top of it.
    """
    window = parent.window()
    dialog = getattr(window, _DIALOG_ATTRIBUTE, None)
    if dialog is None or sip.isdeleted(dialog):
        dialog = ImageViewerDialog(window)
        setattr(window, _DIALOG_ATTRIBUTE, dialog)
    dialog.show_path(path, title, placeholder)
    return dialog
