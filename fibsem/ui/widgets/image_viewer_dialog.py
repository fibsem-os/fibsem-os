"""One window to look at a saved image, on the real canvas at full resolution (FIB-1189).

A beam image opens on a :class:`FibsemImageCanvas`, which brings the microscope tab's
zoom, fit, reset, live scalebar, contrast, ruler and crosshair. A fluorescence stack
opens on the :class:`FMCanvasWidget`, with its z-slider, max projection and per-channel
controls; up and down step through the planes. Under either sits the view's bar
(FIB-1186), so the image says what it is in the same words as the quad view, with the
same field choices.

The full-resolution file loads off the GUI thread. Until it arrives the window shows
the caller's thumbnail, so a click answers at once even for a stitched overview.

A viewer that opens arbitrary files (the FM Image Viewer) can also take them dropped
from the desktop: :meth:`ImageViewer.set_accepts_drops`. One that shows a lamella's or a
grid's own images leaves it off, so nothing foreign joins their set.

The caller hands over the images it has -- a lamella's, in task order -- and the one
clicked. Left and right, or the arrows in the header, step through them; a filmstrip
along the bottom shows them all. Stepping swaps between the beam canvas and the FM
viewer as the kind of image changes.

:func:`open_image_viewer` keeps one non-modal window per application window and
reuses it, so it can stay open beside the app while you click through tiles.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Union

from PyQt5 import sip
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QShortcut,
    QStackedWidget,
    QToolButton,
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
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    CANVAS_BG,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.view_info_bar import ViewInfoBar, show_fm_plane

ViewerImage = Union[FibsemImage, FluorescenceImage]

_TITLE_STYLE = f"color: {TEXT_STRONG_COLOR}; font-size: 13px; font-weight: 600;"
_LOADING_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 12px; background: {CANVAS_BG};"
_POSITION_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 12px;"
_HINT_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"
_STEP_STYLE = (
    f"QToolButton {{ background: transparent; color: {TEXT_COLOR}; border: none;"
    " font-size: 16px; padding: 0 6px; }"
    "QToolButton:disabled { color: #444; }"
)
_DROP_SUFFIXES = (".tif", ".tiff")  # beam images, and .ome.tiff stacks
_DROP_HINT_STYLE = (
    f"background: rgba(30, 33, 36, 0.85); color: {TEXT_STRONG_COLOR};"
    f" border: 2px dashed {ACCENT_COLOR}; border-radius: 6px; font-size: 14px;"
)
_FILM_TILE = QSize(96, 64)
_FILM_STYLE = (
    f"QToolButton {{ background: transparent; color: {TEXT_MUTED_COLOR};"
    f" font-size: 11px; border: 2px solid transparent; border-radius: 4px; padding: 2px; }}"
    f"QToolButton:hover {{ border-color: {BORDER_COLOR}; }}"
    f"QToolButton:checked {{ border-color: {ACCENT_COLOR}; color: {TEXT_STRONG_COLOR}; }}"
)

_LOADING, _BEAM, _FM = 0, 1, 2  # the stack's pages


@dataclass
class ViewerItem:
    """One image the viewer can step to: its file, and what to call it.

    *title* heads the window while it is shown (`01-fancy-mite › Rough Milling`);
    *label* names it in the filmstrip and the position (`FIB`), and is the image's
    own kind once it has been read if nothing else named it; *thumbnail* stands in
    while the file loads. *image* is one already in memory, shown without a read --
    an FM stack handed over by the load dialog, which may have no file at all.
    """

    path: str = ""
    title: str = ""
    label: str = ""
    thumbnail: Optional[QPixmap] = None
    image: Optional["ViewerImage"] = None


def dropped_image_paths(mime) -> List[str]:
    """The local image files in a drag's *mime* data, in the order dragged."""
    if not mime.hasUrls():
        return []
    paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
    return [p for p in paths if p.lower().endswith(_DROP_SUFFIXES)]


def load_viewer_image(path: str) -> ViewerImage:
    """The image at *path*: a fluorescence stack as a stack, anything else as a beam
    image. Full resolution; nothing is resized."""
    if is_fluorescence_image(path):
        return FluorescenceImage.load(path)
    return FibsemImage.load(path)


# Every read still running, whichever viewer asked for it. A read belongs to no viewer:
# a viewer can close, or be replaced -- the FM Image Viewer is, on each reopen -- while
# it reads, and Qt aborts on a thread destroyed while it runs. Held here until it ends,
# its result goes nowhere if the viewer is gone.
_RUNNING: Set["_Loader"] = set()


def wait_for_reads(timeout_ms: int = 5000) -> None:
    """Let every running read finish: at quit, before the threads are torn down."""
    for loader in list(_RUNNING):
        loader.wait(timeout_ms)


_QUIT_HOOKED = []  # the application whose quit waits for reads, once hooked


def _start(loader: "_Loader") -> None:
    app = QApplication.instance()
    if app is not None and app not in _QUIT_HOOKED:
        app.aboutToQuit.connect(wait_for_reads)
        _QUIT_HOOKED.append(app)
    _RUNNING.add(loader)
    loader.finished.connect(lambda: _RUNNING.discard(loader))
    loader.finished.connect(loader.deleteLater)
    loader.start()


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
    """Saved images on the real canvas, one at a time, with the bar, a title, the
    position among them, a filmstrip and Export."""

    title_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items: List[ViewerItem] = []
        self._index = -1
        self._actions: List[QWidget] = []  # the current caller's, from set_actions
        self._image: Optional[ViewerImage] = None
        self._path: Optional[str] = None
        self._pending: Optional[str] = None  # the load whose result we still want
        self._loaders: Set[_Loader] = set()  # this viewer's reads, still running
        self._fm_stack = None
        self._placeholder: Optional[QPixmap] = None

        self.title_label = QLabel()
        self.title_label.setStyleSheet(_TITLE_STYLE)
        self.previous_button = self._step_button("‹", "Previous image (←)", -1)
        self.next_button = self._step_button("›", "Next image (→)", 1)
        self.position_label = QLabel()
        self.position_label.setStyleSheet(_POSITION_STYLE)
        self.export_button = QPushButton("Export…")
        self.export_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.export_button.setToolTip("Export this image with its metadata")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 10, 6)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.previous_button)
        header.addWidget(self.position_label)
        header.addWidget(self.next_button)
        header.addSpacing(8)
        header.addWidget(self.export_button)
        self._header = header

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

        # The filmstrip: every image handed over, the one shown checked.
        self._film_buttons: List[QToolButton] = []
        self._film_row = QHBoxLayout()
        self._film_row.setContentsMargins(8, 4, 8, 4)
        self._film_row.setSpacing(6)
        self._film_row.addStretch(1)
        film_content = QWidget()
        film_content.setLayout(self._film_row)
        self.filmstrip = QScrollArea()
        self.filmstrip.setWidget(film_content)
        self.filmstrip.setWidgetResizable(True)
        self.filmstrip.setFrameShape(QScrollArea.NoFrame)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.filmstrip.setFixedHeight(_FILM_TILE.height() + 44)
        self.hint_label = QLabel("← → to step · Esc to close")
        self.hint_label.setStyleSheet(_HINT_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.filmstrip)
        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(10, 0, 10, 6)
        hint_row.addStretch(1)
        hint_row.addWidget(self.hint_label)
        layout.addLayout(hint_row)

        # Shortcuts, not a key handler: the canvases are matplotlib widgets, which
        # take the arrow keys for themselves before a parent hears of them. They
        # follow the focus, so the viewer takes it when nothing inside has it.
        self.setFocusPolicy(Qt.StrongFocus)
        for key, slot in (
            (Qt.Key_Left, lambda: self.step(-1)),
            (Qt.Key_Right, lambda: self.step(1)),
            (Qt.Key_Up, lambda: self._step_z(1)),
            (Qt.Key_Down, lambda: self._step_z(-1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        # Shown over the image while files are dragged over a viewer that takes them.
        self.drop_hint = QLabel("Drop images to open", self, alignment=Qt.AlignCenter)
        self.drop_hint.setStyleSheet(_DROP_HINT_STYLE)
        self.drop_hint.hide()
        self._update_navigation()

    def _step_button(self, text: str, tooltip: str, step: int) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setStyleSheet(_STEP_STYLE)
        button.clicked.connect(lambda: self.step(step))
        return button

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
        self.title_changed.emit(title)

    # ── the images handed over ─────────────────────────────────────────────
    @property
    def items(self) -> List[ViewerItem]:
        return list(self._items)

    @property
    def index(self) -> int:
        """Which of the items is shown; -1 before any."""
        return self._index

    def set_items(self, items: Sequence[ViewerItem], index: int = 0) -> None:
        """Hand over the images to step through, and show the one at *index*."""
        self._items = list(items)
        self._rebuild_filmstrip()
        self._index = -1
        if self._items:
            self.go_to(max(0, min(index, len(self._items) - 1)))
        else:
            self._update_navigation()

    def add_item(self, item: ViewerItem) -> None:
        """Append *item* to the images and show it."""
        self.add_items([item])

    def add_items(self, items: Sequence[ViewerItem]) -> None:
        """Append *items* to the images and show the last: one read, not one each."""
        if not items:
            return
        self._items.extend(items)
        self._rebuild_filmstrip()
        self.go_to(len(self._items) - 1)

    # ── files dropped from the desktop ─────────────────────────────────────
    def set_accepts_drops(self, on: bool) -> None:
        """Take image files dropped onto the window into the filmstrip.

        For a viewer that opens arbitrary files. A dropped file is read directly,
        without the FM load dialog: OME stacks and fibsem images say what they are;
        a plain TIFF that needs its axes set still wants Open….
        """
        self.setAcceptDrops(on)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.acceptDrops() and dropped_image_paths(event.mimeData()):
            event.acceptProposedAction()
            self._show_drop_hint(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.acceptDrops() and dropped_image_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._show_drop_hint(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._show_drop_hint(False)
        paths = dropped_image_paths(event.mimeData()) if self.acceptDrops() else []
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.add_items(
            [ViewerItem(path=path, title=os.path.basename(path)) for path in paths]
        )

    def _show_drop_hint(self, on: bool) -> None:
        if on:
            self.drop_hint.setGeometry(self.stack.geometry().adjusted(12, 12, -12, -12))
            self.drop_hint.raise_()
        self.drop_hint.setVisible(on)

    def add_header_widget(self, widget: QWidget) -> None:
        """Put a caller's action -- Open… -- in the header for good, before Export."""
        self._header.insertWidget(self._header.indexOf(self.export_button), widget)

    def set_actions(self, widgets: Sequence[QWidget]) -> None:
        """Replace the previous caller's actions with *widgets*, before Export.

        For a viewer shared between callers: Grids › Results brings Mark positions,
        the History tab brings none, and each open shows only its own.
        """
        for widget in self._actions:
            self._header.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        self._actions = list(widgets)
        for widget in self._actions:
            self.add_header_widget(widget)

    @property
    def current_item(self) -> Optional[ViewerItem]:
        """The item shown, or None before any."""
        return self._items[self._index] if 0 <= self._index < len(self._items) else None

    def go_to(self, index: int) -> None:
        """Show the item at *index*; out of range does nothing."""
        if not 0 <= index < len(self._items) or index == self._index:
            return
        self._index = index
        item = self._items[index]
        self.set_title(item.title or os.path.basename(item.path) or item.label)
        self._update_navigation()
        if item.image is not None:
            self._pending = None  # a read still running for another item is moot
            self.show_image(item.image, path=item.path or None)
        else:
            self.show_path(item.path, item.thumbnail)

    def step(self, delta: int) -> None:
        """The next (+1) or previous (-1) image; it stops at either end."""
        self.go_to(self._index + delta)

    def _step_z(self, delta: int) -> None:
        if self.stack.currentIndex() == _FM:
            self.fm_widget.step_z(delta)

    def _update_navigation(self) -> None:
        count, index = len(self._items), self._index
        several = count > 1
        for widget in (self.previous_button, self.next_button, self.filmstrip):
            widget.setVisible(several)
        self.hint_label.setText(
            "← → to step · Esc to close" if several else "Esc to close"
        )
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(0 <= index < count - 1)
        for i, button in enumerate(self._film_buttons):
            button.setChecked(i == index)
        if 0 <= index < count:
            item = self._items[index]
            # Not twice: a file's name can be both its title and its label.
            label = item.label if item.label != item.title else ""
            position = f"{index + 1} of {count}" if several else ""
            self.position_label.setText(" · ".join(t for t in (label, position) if t))
            if several:
                self.filmstrip.ensureWidgetVisible(self._film_buttons[index])
        else:
            self.position_label.setText("")

    def _rebuild_filmstrip(self) -> None:
        for button in self._film_buttons:
            self._film_row.removeWidget(button)
            button.hide()
            button.deleteLater()
        self._film_buttons = []
        for i, item in enumerate(self._items):
            button = QToolButton()
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setIconSize(_FILM_TILE)
            button.setStyleSheet(_FILM_STYLE)
            button.setText(item.label or str(i + 1))
            button.setToolTip(item.title or os.path.basename(item.path))
            if item.thumbnail is not None and not item.thumbnail.isNull():
                button.setIcon(
                    QIcon(
                        item.thumbnail.scaled(
                            _FILM_TILE, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
                )
            button.clicked.connect(lambda _=False, i=i: self.go_to(i))
            self._film_row.insertWidget(i, button)
            self._film_buttons.append(button)

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

        loader = _Loader(path)  # no parent: see _RUNNING
        loader.loaded.connect(self._on_loaded)
        loader.finished.connect(lambda: self._loaders.discard(loader))
        self._loaders.add(loader)
        _start(loader)

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
        self._learn_label(path, image_fields(image).kind)

    def _learn_label(self, path: Optional[str], kind: str) -> None:
        """Name the shown item by what the file says it is, once it has been read."""
        if kind == "Image" or not 0 <= self._index < len(self._items):
            return
        item = self._items[self._index]
        if item.label or (item.path or None) != path:
            return  # named already, by the caller or an earlier read
        item.label = kind
        self._film_buttons[self._index].setText(kind)
        self._update_navigation()

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
        self.viewer.title_changed.connect(self.setWindowTitle)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.7), int(screen.height() * 0.75))

    def show_items(
        self,
        items: Sequence[ViewerItem],
        index: int = 0,
        actions: Sequence[QWidget] = (),
    ) -> None:
        """Show *items*, starting at *index*, with the caller's *actions* in the
        header, and bring the window forward."""
        self.viewer.set_actions(actions)
        self.viewer.set_items(items, index)
        self.show()
        self.raise_()
        self.activateWindow()
        self.viewer.setFocus()


_DIALOG_ATTRIBUTE = "_fibsem_image_viewer"


def open_image_viewer(
    parent: QWidget,
    items: Sequence[ViewerItem],
    index: int = 0,
    actions: Sequence[QWidget] = (),
) -> ImageViewerDialog:
    """Show *items* in the viewer of *parent*'s window, at *index*, making the
    viewer on first use. *actions* are the caller's buttons for the header, which
    replace any an earlier caller put there.

    One per application window, reused: a second click shows the new images in the
    window already open rather than stacking another on top of it.
    """
    window = parent.window()
    dialog = getattr(window, _DIALOG_ATTRIBUTE, None)
    if dialog is None or sip.isdeleted(dialog):
        dialog = ImageViewerDialog(window)
        setattr(window, _DIALOG_ATTRIBUTE, dialog)
    dialog.show_items(items, index, actions)
    return dialog
