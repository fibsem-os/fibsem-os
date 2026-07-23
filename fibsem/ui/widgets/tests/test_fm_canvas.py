"""Standalone demo: multi-channel FM canvas + per-channel layer controls.

Run:
    PYTHONPATH=<worktree> python fibsem/ui/widgets/tests/test_fm_canvas.py

An FMCanvasWidget showing a synthetic 3-channel fluorescence image composited
(per-channel colour, additive blend). Click the **layers** button (top-right of
the canvas) to open the controls and toggle visibility, change colormap, opacity,
or per-channel contrast — the composite updates live. "Re-acquire" jitters the
intensities to mimic a live update.
"""
import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget

H = W = 512
_RNG_SEED = 0


def _blobs(centres, r, peak) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    out = np.zeros((H, W), dtype=np.float32)
    for cy, cx in centres:
        out += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))) * peak
    return out


def _channels(scale: float = 1.0):
    # DAPI: scattered nuclei; GFP: a band; RFP: a cluster
    dapi = _blobs([(120, 140), (200, 380), (340, 180), (400, 410), (260, 260)], 28, 4000 * scale)
    gfp = _blobs([(256, 200), (256, 256), (256, 312)], 40, 3000 * scale)
    rfp = _blobs([(360, 360), (390, 330), (330, 390)], 34, 3500 * scale)
    return {"DAPI": ("blue", dapi), "GFP": ("green", gfp), "RFP": ("red", rfp)}


def main() -> None:
    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(820, 720)

    fm = FMCanvasWidget()
    fm.set_pixel_size(100e-9)
    for name, (color, data) in _channels().items():
        fm.set_channel(name, data, color)

    info = QLabel("Click the layers button (top-right) to control channels.")

    def reacquire():
        rng = np.random.default_rng()
        for name, (color, data) in _channels(scale=float(rng.uniform(0.6, 1.4))).items():
            fm.set_channel(name, data)  # keep colour/display props, swap data
        info.setText("Re-acquired (intensities jittered).")

    btn = QPushButton("Re-acquire")
    btn.clicked.connect(reacquire)

    bar = QHBoxLayout()
    bar.addWidget(btn)
    bar.addStretch()
    lay = QVBoxLayout(win)
    lay.addWidget(fm, 1)
    lay.addLayout(bar)
    lay.addWidget(info)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
