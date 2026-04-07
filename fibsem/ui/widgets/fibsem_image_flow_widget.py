"""Flow-layout widget for displaying a collection of FibsemImages."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QFlowLayout, QIconifyIcon

from fibsem.structures import FibsemImage

_CARD_WIDTH = 320
_CARD_HEIGHT = 240


def _arr_to_pixmap(arr: np.ndarray, w: int, h: int) -> QPixmap:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    ih, iw, _ = arr.shape
    qimg = QImage(arr.data, iw, ih, iw * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class _FullscreenDialog(QDialog):
    """Simple fullscreen dialog showing an image."""

    def __init__(self, pixmap: QPixmap, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setStyleSheet("background: black;")

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.85), int(screen.height() * 0.85))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background: black;")
        scaled = pixmap.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        layout.addWidget(label)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class FibsemImageCard(QWidget):
    """A single image card: thumbnail + zoom button in top-right corner."""

    def __init__(self, image: FibsemImage, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self.setFixedSize(_CARD_WIDTH, _CARD_HEIGHT + 24)
        self.setStyleSheet("background: #1a1b1e; border-radius: 6px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Image container — lets us overlay the zoom button
        img_container = QWidget()
        img_container.setFixedSize(_CARD_WIDTH - 8, _CARD_HEIGHT - 8)
        img_container.setStyleSheet("background: #111; border-radius: 4px;")
        layout.addWidget(img_container)

        self._img_label = QLabel(img_container)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background: transparent;")
        self._img_label.setGeometry(0, 0, _CARD_WIDTH - 8, _CARD_HEIGHT - 8)

        # Zoom button — top-right of the image container
        self._zoom_btn = QPushButton(img_container)
        self._zoom_btn.setIcon(QIconifyIcon("mdi:magnify", color="#cccccc"))
        self._zoom_btn.setFixedSize(28, 28)
        self._zoom_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(0,0,0,160);"
            "  border: none;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover { background: rgba(80,80,80,200); }"
        )
        self._zoom_btn.move(_CARD_WIDTH - 8 - 32, 4)
        self._zoom_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._zoom_btn.clicked.connect(self._open_fullscreen)

        if title:
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("color: #cccccc; font-size: 11px; background: transparent;")
            title_label.setFixedHeight(20)
            layout.addWidget(title_label)

        self._pixmap = _arr_to_pixmap(image.data, _CARD_WIDTH - 8, _CARD_HEIGHT - 8)
        self._img_label.setPixmap(self._pixmap)

    def _open_fullscreen(self) -> None:
        dlg = _FullscreenDialog(self._pixmap, title=self._title, parent=self)
        dlg.exec_()


class FibsemImageFlowWidget(QWidget):
    """Displays a list of FibsemImages in a reflowing grid.

    Images are shown as fixed-size cards that reflow as the widget is resized.
    With the default card width of 320px and typical window widths, 4 images
    will appear as a 2x2 grid initially.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #2b2d31; }")
        outer.addWidget(self._scroll)

        self._content = QWidget()
        self._content.setStyleSheet("background: #2b2d31;")
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._flow = QFlowLayout(self._content)
        self._flow.setContentsMargins(12, 12, 12, 12)
        self._flow.setSpacing(8)

        self._scroll.setWidget(self._content)

    def set_images(self, images: List[FibsemImage], titles: Optional[List[str]] = None) -> None:
        """Replace displayed images."""
        while self._flow.count():
            item = self._flow.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for i, img in enumerate(images):
            title = titles[i] if titles and i < len(titles) else f"Image {i + 1}"
            card = FibsemImageCard(img, title=title, parent=self._content)
            self._flow.addWidget(card)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _make_demo_image(value: int, shape=(256, 256)) -> FibsemImage:
    """Create a fake grayscale FibsemImage filled with a gradient for demo."""
    arr = np.full(shape, value, dtype=np.uint8)
    grad = np.linspace(0, value, shape[1], dtype=np.uint8)
    arr = np.clip(arr + grad[np.newaxis, :], 0, 255).astype(np.uint8)
    return FibsemImage.generate_blank_image(random=True)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    demo_images = [_make_demo_image(v) for v in (60, 120, 180, 240)]
    demo_titles = ["SEM - Task 1", "FIB - Task 1", "SEM - Task 2", "FIB - Task 2"]

    widget = FibsemImageFlowWidget()
    widget.set_images(demo_images, titles=demo_titles)
    widget.setWindowTitle("FibsemImage Flow Layout")
    widget.resize(700, 600)
    widget.show()

    sys.exit(app.exec())
