"""Hands-on harness for CorrelationCanvasWidget — the FIB correlation surface.

Was a side-by-side against `ImagePointCanvas` while the migration was in flight;
that class is gone, so this is the single-widget form. It survives because the
widget is otherwise only reachable through a full correlation session with real
data, and a passing Qt suite has said nothing useful about whether a canvas
actually works (925 of them once did, on a widget whose setup had been halved).

Usage
-----
    PYTHONPATH=$PWD python fibsem/ui/correlation/widgets/test_correlation_canvas_demo.py

What to try
-----------
* **Right-click** for the add menu, **drag** a point, **click** to select,
  **Delete** to remove. The log shows what the widget emits — a click that does
  not move a point must not report a move, because downstream that counts as an
  edit and invalidates the correlation result.
* **Show result** — the three marker groups the tab widget draws after a run.
  Check the legend picks up all three, the uncorrected-POI ghost stays a visible
  ring under the marker on top of it, and that clicking a result marker does
  nothing: they are output, not points.
* **A SURFACE point** draws the dashed datum row across the image. Drag it and
  the row should track the marker, not lag it.
* **Contrast / gamma**, the scalebar, zoom, pan and fit-to-view live on the
  toolbar top-right. Contrast is what the old canvas never had, and the reason
  for the migration.
* **Crosshair** is off by default here — correlation draws SURFACE as an orange
  `+` and the shared canvas's yellow one is easy to mistake for it. The toolbar
  toggle brings it back.
"""

from __future__ import annotations

import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    OK_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

_FIB_TYPES = [PointType.FIB, PointType.SURFACE]


def _synthetic_image(h: int = 512, w: int = 512) -> np.ndarray:
    """Gradient + blobs, deliberately low-contrast so the contrast control has
    something to do."""
    rng = np.random.default_rng(42)
    base = np.linspace(0.0, 0.3, w)[None, :] + np.linspace(0.0, 0.3, h)[:, None]
    img = base + rng.normal(0, 0.04, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx, amp, sigma in (
        (150, 180, 0.5, 40),
        (330, 300, 0.4, 55),
        (200, 400, 0.3, 30),
    ):
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    return (np.clip(img, 0, None) / img.max() * 255).astype(np.uint8)


def _seed_points() -> list:
    return [
        Coordinate(PointXYZ(180.0, 150.0, 0.0), PointType.FIB),
        Coordinate(PointXYZ(300.0, 330.0, 0.0), PointType.FIB),
        Coordinate(PointXYZ(400.0, 200.0, 0.0), PointType.SURFACE),
    ]


# The three groups CorrelationTabWidget._overlay_result_on_fib draws, verbatim.
# The reprojected fiducials sit a few pixels off the seed points, as a real fit
# would leave them; the ghost sits under the corrected POI, which is the case
# `hollow` exists for.
_RESULT_GROUPS = (
    (
        [(186.0, 156.0), (295.0, 337.0)],
        dict(
            color="#ff4444",
            label_prefix="E",
            size=4,
            marker="o",
            legend_label="FM reprojected (E)",
        ),
    ),
    (
        [(250.0, 250.0)],
        dict(
            color="#ff00ff",
            size=7,
            marker="o",
            alpha=0.7,
            show_labels=False,
            hollow=True,
            legend_label="POI uncorrected",
        ),
    ),
    (
        [(256.0, 254.0)],
        dict(
            color="#ff00ff",
            label_prefix="P",
            size=5,
            marker="o",
            legend_label="POI (P)",
        ),
    ),
)


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CorrelationCanvasWidget — FIB correlation surface")
        self.resize(1000, 860)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.canvas = CorrelationCanvasWidget(allowed_point_types=_FIB_TYPES)
        self.canvas.set_image(_synthetic_image())
        self.canvas.set_pixel_size(2e-8)
        self.canvas.point_selected.connect(lambda c: self._say("selected", c))
        self.canvas.point_moved.connect(lambda c: self._say("moved", c))
        self.canvas.point_removed.connect(lambda c: self._say("removed", c))
        self.canvas.point_add_requested.connect(self._added)
        root_layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        for label, slot in (
            ("Reset points", self._reset),
            ("Clear points", self._clear),
            ("Show result", self._show_result),
            ("Clear result", self.canvas.clear_overlay),
            ("Clear log", lambda: self.log.clear()),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        for label, setter in (
            ("Legend", "set_legend_visible"),
            ("Labels", "set_labels_visible"),
            ("Scalebar", "set_scalebar_visible"),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(
                lambda on, s=setter, name=label: self._toggle(s, on, name)
            )
            controls.addWidget(btn)

        controls.addStretch()
        controls.addWidget(
            QLabel(
                f"<span style='color:{TEXT_MUTED_COLOR}'>right-click to add · drag to move · "
                f"Delete to remove · contrast/gamma on the toolbar</span>"
            )
        )
        root_layout.addLayout(controls)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(170)
        self.log.setStyleSheet(
            f"background: {SURFACE_COLOR}; color: {TEXT_STRONG_COLOR}; "
            "font-family: monospace; font-size: 11px;"
        )
        root_layout.addWidget(self.log)

        self._reset()

    # ── behaviour ─────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self.canvas.set_coordinates(_seed_points())
        self._log("reset to 3 seed points")

    def _clear(self) -> None:
        self.canvas.set_coordinates([])
        self._log("cleared")

    def _show_result(self) -> None:
        """The same call sequence as CorrelationTabWidget._overlay_result_on_fib."""
        self.canvas.clear_overlay()
        for points, style in _RESULT_GROUPS:
            self.canvas.add_overlay_points(points, **style)
        self._log("drew the reprojected result", OK_COLOR)

    def _toggle(self, setter: str, on: bool, name: str) -> None:
        getattr(self.canvas, setter)(on)
        self._log(f"{name.lower()} {'shown' if on else 'hidden'}")

    def _added(self, x: float, y: float, pt: PointType) -> None:
        """The widget only *requests* an add — the tab widget builds the
        Coordinate, because it owns the z. Mirror that here."""
        coord = Coordinate(PointXYZ(x, y, 0.0), pt)
        self.canvas.set_coordinates(list(self.canvas.points.coordinates()) + [coord])
        self._log(f"add_requested  {pt.value:12} ({x:7.1f}, {y:7.1f})", ACCENT_COLOR)

    def _say(self, what: str, coord: Coordinate) -> None:
        self._log(
            f"{what:9} {coord.point_type.value:12} "
            f"({coord.point.x:7.1f}, {coord.point.y:7.1f})"
        )

    def _log(self, message: str, color: str = None) -> None:
        self.log.append(
            f"<span style='color:{color}'>{message}</span>" if color else message
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
