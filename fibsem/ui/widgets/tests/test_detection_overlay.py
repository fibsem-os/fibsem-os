"""Standalone demo: detection display (MaskOverlay + per-feature PointOverlay).

Run:
    PYTHONPATH=<worktree> python fibsem/ui/widgets/tests/test_detection_overlay.py

A FibsemImageCanvas showing a synthetic image + an alpha-blended segmentation
mask (MaskOverlay) + draggable, per-feature-coloured, named feature points
(PointOverlay, move-only) in an active "Detection" mode — the matplotlib
equivalent of FibsemEmbeddedDetectionWidget's napari view (display-only mask).

Drag the points to correct them (positions are logged). Toggle the mask, or drop
to Move via the top-right toolbar toggle (the feature points go inert).
"""
import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.overlays.mask_overlay import MaskOverlay
from fibsem.ui.widgets.overlays.point_overlay import PointOverlay

H = W = 512


def _synthetic_image() -> FibsemImage:
    img = (np.random.rand(H, W) * 60 + 40)
    img[H // 2 - 40:H // 2 + 40, W // 4:3 * W // 4] += 120  # bright lamella band
    return FibsemImage(data=np.clip(img, 0, 255).astype(np.uint8))


def _synthetic_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[H // 2 - 40:H // 2 + 40, W // 4:3 * W // 4] = 1  # class 1 = lamella
    yy, xx = np.ogrid[:H, :W]
    mask[(yy - 120) ** 2 + (xx - 400) ** 2 < 45 ** 2] = 2  # class 2 = manipulator blob
    return mask


# (name, colour, (col, row)) — mirrors detection Feature.name / .color / .px
FEATURES = [
    ("LamellaLeftEdge", "red", (W // 4, H // 2)),
    ("LamellaRightEdge", "red", (3 * W // 4, H // 2)),
    ("LamellaCentre", "yellow", (W // 2, H // 2)),
    ("NeedleTip", "lime", (400, 120)),
]


def main() -> None:
    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(820, 680)

    canvas = FibsemImageCanvas()
    canvas.set_image(_synthetic_image())

    mask_overlay = MaskOverlay()
    canvas.add_overlay(mask_overlay)
    mask_overlay.set_mask(_synthetic_mask())

    feats = PointOverlay(marker="+", size=16, removable=False, add_on_right_click=False)
    canvas.add_overlay(feats)
    feats.set_points(
        [f[2] for f in FEATURES],
        colors=[f[1] for f in FEATURES],
        labels=[f[0] for f in FEATURES],
    )
    canvas.enter_overlay_mode(feats, "Detection", icon="mdi:vector-point")
    canvas.set_hint("drag features to correct  ·  Continue when done")

    log = QLabel("Drag a feature point to correct it.")
    feats.point_moved.connect(
        lambda i, x, y: log.setText(f"{FEATURES[i][0]} → ({x:.0f}, {y:.0f}) px")
    )

    chk = QCheckBox("Show mask")
    chk.setChecked(True)
    chk.toggled.connect(lambda on: mask_overlay.set_mask(_synthetic_mask() if on else None))

    bar = QHBoxLayout()
    bar.addWidget(chk)
    bar.addStretch()
    lay = QVBoxLayout(win)
    lay.addWidget(canvas, 1)
    lay.addLayout(bar)
    lay.addWidget(log)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
