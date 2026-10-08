"""Standalone viewer for loading and displaying FluorescenceImages from file.

The image viewer (FIB-1189) with an Open… button: files picked in the load dialog join
its filmstrip, and the newest is shown. The FM viewer brings the per-channel colour,
visibility and contrast, the z-slider, max projection and the scalebar; the bar under it
says what the image is, in the same words as the quad view.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from fibsem.fm.structures import FluorescenceImage
from fibsem.ui.fm.widgets.load_image_dialog import LoadImageDialog
from fibsem.ui.stylesheets import NAPARI_STYLE, PRIMARY_BUTTON_STYLESHEET
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.image_viewer_dialog import ImageViewer, ViewerItem


class FMImageViewerWidget(QWidget):
    """Load FluorescenceImages from file and look at them, one at a time.

    Loading appends to the filmstrip rather than replacing, so several files can be held
    open and stepped between with the arrow keys; ``FMCanvasWidget.set_fm_image`` resets
    the channel set on each call, and blending two *different* images was never a
    workflow, so stepping beats stacking them onto one canvas.
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
        # This opens as a top-level window with no parent, and a Qt stylesheet only
        # cascades to children — so it inherits nothing from the main window and has to
        # carry the dark theme itself. Same as the coincidence viewer and FibsemUI.
        self.setStyleSheet(NAPARI_STYLE)

        self.image_viewer = ImageViewer(self)
        self.pushButton_load_image = QPushButton("Open…")
        self.pushButton_load_image.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_load_image.setToolTip("Open fluorescence images from file")
        self.image_viewer.add_header_widget(self.pushButton_load_image)
        self.image_viewer.set_title("No image loaded")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_viewer)

        self.pushButton_load_image.clicked.connect(self.show_load_image_dialog)
        self.image_loaded_signal.connect(self.add_image)

    @property
    def canvas(self) -> FMCanvasWidget:
        """The FM viewer the images are shown on."""
        return self.image_viewer.fm_widget

    @property
    def images(self) -> List[FluorescenceImage]:
        return list(self._images)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def show_load_image_dialog(self) -> None:
        """Show the load dialog; every image it emits joins the filmstrip."""
        dialog = LoadImageDialog(self, start_directory=self.start_directory)
        dialog.image_loaded_signal.connect(self.image_loaded_signal.emit)

        if dialog.exec_():
            logging.info("Image loaded successfully")
        else:
            logging.info("Image loading canceled")

    @ensure_main_thread
    @pyqtSlot(FluorescenceImage)
    def add_image(self, image: FluorescenceImage) -> None:
        """Append *image* to the filmstrip and show it.

        The dialog emits once per file, so a multi-file load lands here repeatedly and
        the last one ends up displayed.
        """
        self._images.append(image)
        name = self._display_name(image)
        self.image_viewer.add_item(
            ViewerItem(path=image.filepath or "", title=name, label=name, image=image)
        )

    def display_image(self, image: FluorescenceImage) -> None:
        """Show *image*, one already in the filmstrip or not."""
        # By identity: images compare by their arrays, which have no single truth value.
        index = next((i for i, held in enumerate(self._images) if held is image), None)
        if index is None:
            self.add_image(image)
        else:
            self.image_viewer.go_to(index)

    @staticmethod
    def _display_name(image: FluorescenceImage) -> str:
        """A filename where there is one, else the image's own description."""
        if image.filepath:
            return os.path.basename(image.filepath)
        return getattr(image.metadata, "description", None) or "Untitled image"


def main() -> None:
    import sys

    from PyQt5.QtWidgets import QApplication

    from fibsem.config import LOG_PATH

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(NAPARI_STYLE)

    widget = FMImageViewerWidget(start_directory=LOG_PATH)
    widget.resize(1180, 700)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
