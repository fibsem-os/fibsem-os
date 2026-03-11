"""Standalone test for drawing overlay functions displayed on QLabels."""

import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from fibsem.imaging.drawing import draw_crosshair, draw_image_overlays, draw_scalebar


def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _make_test_image(width: int = 512, height: int = 384) -> np.ndarray:
    """Generate a synthetic grayscale test image with some structure."""
    y, x = np.mgrid[:height, :width]
    # gradient + circular pattern
    img = ((np.sin(x / 30.0) * np.cos(y / 30.0) + 1) * 60 + 40).astype(np.uint8)
    # add some noise
    rng = np.random.default_rng(42)
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
    return img


def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Drawing Overlay Test")
    window.setStyleSheet("background: #2b2d31;")

    layout = QVBoxLayout(window)

    test_cases = [
        ("Original", None, None),
        ("Scalebar only (10 μm FOV)", 20e-9, {"show_crosshair": False}),
        ("Crosshair only", 20e-9, {"show_scalebar": False}),
        ("Both overlays (10 μm FOV)", 20e-9, {}),
        ("Both overlays (100 μm FOV)", 200e-9, {}),
        ("Both overlays (1 mm FOV)", 2e-6, {}),
    ]

    for i in range(0, len(test_cases), 3):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setSpacing(16)

        for title, pixel_size, kwargs in test_cases[i:i + 3]:
            img = _make_test_image()

            if pixel_size is not None:
                img = draw_image_overlays(img, pixel_size, **(kwargs or {}))

            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(4)

            title_label = QLabel(title)
            title_label.setStyleSheet("color: #e0e0e0; font-size: 11px; font-weight: bold;")
            title_label.setAlignment(Qt.AlignCenter)
            col_layout.addWidget(title_label)

            img_label = QLabel()
            img_label.setPixmap(_arr_to_pixmap(img))
            img_label.setStyleSheet("background: #1a1b1e; border-radius: 4px;")
            img_label.setAlignment(Qt.AlignCenter)
            col_layout.addWidget(img_label)

            row_layout.addWidget(col)

        layout.addWidget(row_widget)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
