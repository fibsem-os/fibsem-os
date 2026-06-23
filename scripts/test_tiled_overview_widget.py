"""Quick test for TiledOverviewWidget — run standalone.

Tab 1: grid-unit mode (no image)
Tab 2: physical mode with a synthetic SEM background image
"""

import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import ImageSettings, OverviewAcquisitionSettings
from fibsem.ui.widgets.tiled_overview_widget import TiledOverviewWidget


def make_synthetic_sem(width=512, height=384) -> np.ndarray:
    """Return a fake greyscale SEM image with some texture."""
    rng = np.random.default_rng(42)
    base = rng.integers(60, 140, (height, width), dtype=np.uint8)
    # add a few bright blobs to look vaguely like a sample
    for _ in range(12):
        cx = rng.integers(50, width - 50)
        cy = rng.integers(50, height - 50)
        r = rng.integers(15, 45)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        base[mask] = np.clip(base[mask].astype(int) + rng.integers(40, 90), 0, 255)
    return base


# ── Tab 1: plain grid mode ────────────────────────────────────────────────────

class GridModeTab(QWidget):
    def __init__(self):
        super().__init__()
        vbox = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Rows:"))
        self._rows = QSpinBox(); self._rows.setRange(1, 10); self._rows.setValue(3)
        ctrl.addWidget(self._rows)
        ctrl.addWidget(QLabel("Cols:"))
        self._cols = QSpinBox(); self._cols.setRange(1, 10); self._cols.setValue(4)
        ctrl.addWidget(self._cols)
        btn = QPushButton("Apply")
        btn.clicked.connect(self._apply)
        ctrl.addWidget(btn)
        ctrl.addStretch()
        vbox.addLayout(ctrl)

        self._widget = TiledOverviewWidget(nrows=3, ncols=4)
        self._widget.tiles_changed.connect(lambda e: print(f"[grid] enabled: {e}"))
        vbox.addWidget(self._widget)

    def _apply(self):
        self._widget.set_grid(self._rows.value(), self._cols.value())


# ── Tab 2: physical mode + background image ───────────────────────────────────

class PhysicalModeTab(QWidget):
    def __init__(self):
        super().__init__()
        vbox = QVBoxLayout(self)

        ctrl = QHBoxLayout()

        # Overview image HFW control
        ctrl.addWidget(QLabel("Overview HFW (µm):"))
        self._overview_hfw = QSpinBox()
        self._overview_hfw.setRange(50, 2000)
        self._overview_hfw.setValue(400)
        ctrl.addWidget(self._overview_hfw)

        # Tile HFW control
        ctrl.addWidget(QLabel("Tile HFW (µm):"))
        self._tile_hfw = QSpinBox()
        self._tile_hfw.setRange(10, 500)
        self._tile_hfw.setValue(100)
        ctrl.addWidget(self._tile_hfw)

        ctrl.addWidget(QLabel("Rows:"))
        self._rows = QSpinBox(); self._rows.setRange(1, 8); self._rows.setValue(2)
        ctrl.addWidget(self._rows)
        ctrl.addWidget(QLabel("Cols:"))
        self._cols = QSpinBox(); self._cols.setRange(1, 8); self._cols.setValue(3)
        ctrl.addWidget(self._cols)

        btn = QPushButton("Apply")
        btn.clicked.connect(self._apply)
        ctrl.addWidget(btn)

        toggle_img_btn = QPushButton("Toggle image")
        toggle_img_btn.setCheckable(True)
        toggle_img_btn.setChecked(True)
        toggle_img_btn.toggled.connect(self._toggle_image)
        ctrl.addWidget(toggle_img_btn)

        ctrl.addStretch()
        vbox.addLayout(ctrl)

        self._widget = TiledOverviewWidget()
        self._widget.tiles_changed.connect(lambda e: print(f"[physical] enabled: {e}"))
        vbox.addWidget(self._widget)

        self._sem_image = make_synthetic_sem()
        self._show_image = True
        self._apply()

    def _apply(self):
        nrows = self._rows.value()
        ncols = self._cols.value()
        tile_hfw_m = self._tile_hfw.value() * 1e-6

        settings = OverviewAcquisitionSettings(
            image_settings=ImageSettings(hfw=tile_hfw_m, resolution=(512, 384)),
            nrows=nrows,
            ncols=ncols,
            overlap=0.1,
        )
        self._widget.set_tile_settings(settings)

        if self._show_image:
            self._widget.set_image(self._sem_image, hfw_um=self._overview_hfw.value())

    def _toggle_image(self, checked: bool):
        self._show_image = checked
        if checked:
            self._widget.set_image(self._sem_image, hfw_um=self._overview_hfw.value())
        else:
            self._widget.clear_image()


# ── Main window ───────────────────────────────────────────────────────────────

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TiledOverviewWidget — test")
        self.resize(750, 580)

        tabs = QTabWidget()
        tabs.addTab(GridModeTab(), "Grid mode")
        tabs.addTab(PhysicalModeTab(), "Physical mode + image")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = TestWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
