"""Standalone viewer for loading and displaying FluorescenceImages from file."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.fm.structures import FluorescenceImage
from fibsem.ui.fm.widgets.load_image_dialog import LoadImageDialog
from fibsem.ui.stylesheets import NAPARI_STYLE, PRIMARY_BUTTON_STYLESHEET
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget

# Qt.UserRole payload on each list row: the FluorescenceImage that row displays.
_IMAGE_ROLE = int(Qt.UserRole)


class FMImageViewerWidget(QWidget):
    """Load FluorescenceImages from file and display them on the FM canvas.

    One image is displayed at a time. Loading appends to the list rather than replacing,
    so several files can be held open and switched between; ``FMCanvasWidget.set_fm_image``
    resets the channel set on each call, and blending two *different* images was never a
    workflow, so a list beats stacking them onto one canvas.

    The canvas brings the per-channel colour/visibility/contrast popover, z-scrubbing,
    max-projection and the scalebar with it — this widget only has to load files and
    choose which one is shown.
    """

    image_loaded_signal = pyqtSignal(FluorescenceImage)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        start_directory: Optional[Union[str, Path]] = None,
    ):
        super().__init__(parent)

        self.start_directory = str(start_directory) if start_directory else None
        self._images: List[FluorescenceImage] = []

        self.setWindowTitle("FM Image Viewer")
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        # This opens as a top-level window with no parent, and a Qt stylesheet only
        # cascades to children — so it inherits nothing from the main window and has to
        # carry the dark theme itself. Same as the coincidence viewer and FibsemUI.
        self.setStyleSheet(NAPARI_STYLE)

        self.canvas = FMCanvasWidget()

        self.pushButton_load_image = QPushButton("Load Images...")
        self.pushButton_load_image.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)

        self.label_images = QLabel("Loaded Images")
        self.label_images.setStyleSheet("font-weight: bold; margin-top: 6px;")

        self.listWidget_images = QListWidget()
        self.listWidget_images.setToolTip(
            "Loaded fluorescence images. Selecting one displays it on the canvas."
        )

        self.label_metadata = QLabel("No image loaded.")
        self.label_metadata.setWordWrap(True)
        self.label_metadata.setStyleSheet("color: #a0a0a0; font-size: 11px;")

        sidebar = QWidget()
        sidebar.setMaximumWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)
        sidebar_layout.addWidget(self.pushButton_load_image)
        sidebar_layout.addWidget(self.label_images)
        sidebar_layout.addWidget(self.listWidget_images, 1)
        sidebar_layout.addWidget(self.label_metadata)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.pushButton_load_image.clicked.connect(self.show_load_image_dialog)
        self.listWidget_images.currentRowChanged.connect(self._on_row_changed)
        self.image_loaded_signal.connect(self.add_image)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def show_load_image_dialog(self) -> None:
        """Show the load dialog; every image it emits is appended to the list."""
        dialog = LoadImageDialog(self, start_directory=self.start_directory)
        dialog.image_loaded_signal.connect(self.image_loaded_signal.emit)

        if dialog.exec_():
            logging.info("Image loaded successfully")
        else:
            logging.info("Image loading canceled")

    @ensure_main_thread
    @pyqtSlot(FluorescenceImage)
    def add_image(self, image: FluorescenceImage) -> None:
        """Append *image* to the list and show it.

        The dialog emits once per file, so a multi-file load lands here repeatedly and
        the last one ends up displayed.
        """
        self._images.append(image)

        item = QListWidgetItem(self._display_name(image))
        item.setData(_IMAGE_ROLE, len(self._images) - 1)
        item.setToolTip(image.filepath or "")
        self.listWidget_images.addItem(item)

        # Selecting the new row drives _on_row_changed, which does the display.
        self.listWidget_images.setCurrentRow(self.listWidget_images.count() - 1)

    def display_image(self, image: FluorescenceImage) -> None:
        """Display *image* on the canvas, replacing whatever was shown."""
        try:
            self.canvas.set_fm_image(image)
            self.label_metadata.setText(self._describe(image))
            logging.info(f"Displayed image: {image.metadata.description}")
        except Exception as e:
            logging.error(f"Error displaying image: {e}")

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._images):
            self.display_image(self._images[row])

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _display_name(image: FluorescenceImage) -> str:
        """A filename where there is one, else the image's own description."""
        if image.filepath:
            return os.path.basename(image.filepath)
        return getattr(image.metadata, "description", None) or "Untitled image"

    @staticmethod
    def _describe(image: FluorescenceImage) -> str:
        """The summary shown under the list. Everything here is best-effort: images
        loaded from disk vary in how much metadata they carry."""
        md = image.metadata
        parts: List[str] = [f"<b>{FMImageViewerWidget._display_name(image)}</b>"]

        channels = getattr(md, "channels", None) or []
        shape = getattr(image.data, "shape", ())
        if channels:
            n_z = shape[-3] if len(shape) >= 3 else 1
            parts.append(
                f"{len(channels)} channel{'s' if len(channels) != 1 else ''} · {n_z} z-slice{'s' if n_z != 1 else ''}"
            )
            parts.append(", ".join(c.name for c in channels))
        if len(shape) >= 2:
            parts.append(f"{shape[-1]} × {shape[-2]} px")

        pixel_size = getattr(md, "pixel_size_x", None)
        if pixel_size:
            parts.append(f"{pixel_size * 1e9:.0f} nm/px")
        acquired = getattr(md, "acquisition_date", None)
        if acquired:
            parts.append(FMImageViewerWidget._format_acquired(acquired))

        return "<br>".join(parts)

    @staticmethod
    def _format_acquired(acquired: str) -> str:
        """ISO timestamps read badly in a sidebar; show a date and time instead.

        The field is a free-form string, and images from other sources need not carry a
        parseable one — anything unrecognised is shown as-is rather than dropped.
        """
        try:
            return datetime.fromisoformat(str(acquired)).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            return str(acquired)


def main() -> None:
    """Standalone harness: the viewer in a plain Qt window."""
    import sys

    from PyQt5.QtWidgets import QApplication

    from fibsem.config import LOG_PATH
    from fibsem.ui.stylesheets import NAPARI_STYLE

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(NAPARI_STYLE)

    widget = FMImageViewerWidget(start_directory=LOG_PATH)
    widget.resize(1180, 700)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
