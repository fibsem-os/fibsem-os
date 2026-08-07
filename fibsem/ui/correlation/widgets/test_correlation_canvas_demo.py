"""Side-by-side harness: ImagePointCanvas (old) vs CorrelationCanvasWidget (new).

Same synthetic image, same starting points, on both. Use it to check the
replacement behaves like the thing it replaces before anything is wired in --
and to see what the shared canvas adds, which is the point of FIB-535.

Usage
-----
    PYTHONPATH=$PWD python fibsem/ui/correlation/widgets/test_correlation_canvas_demo.py

What to try
-----------
* **Right-click** either canvas -> the add menu. Both should offer the same
  types and drop a point in the same place.
* **Drag** a point, **click** to select, **Delete** to remove. The log shows
  what each side emits; the payloads should match.
* **Contrast / gamma** -- the toolbar button on the RIGHT canvas only. The old
  canvas has never had it, and on FM data it is the control you actually want
  while picking points.
* **Scalebar, zoom, pan, fit-to-view** -- right canvas toolbar, also new.
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
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.correlation.widgets.image_point_canvas import ImagePointCanvas
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    OK_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

_FIB_TYPES = [PointType.FIB, PointType.SURFACE]


def _synthetic_image(h: int = 512, w: int = 512) -> np.ndarray:
    """Gradient + blobs, deliberately low-contrast so the contrast control on the
    new canvas has something to do."""
    rng = np.random.default_rng(42)
    base = np.linspace(0.0, 0.3, w)[None, :] + np.linspace(0.0, 0.3, h)[:, None]
    img = base + rng.normal(0, 0.04, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx, amp, sigma in ((150, 180, 0.5, 40), (330, 300, 0.4, 55), (200, 400, 0.3, 30)):
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    img = np.clip(img, 0, None)
    return (img / img.max() * 255).astype(np.uint8)


def _seed_points() -> list:
    return [
        Coordinate(PointXYZ(180.0, 150.0, 0.0), PointType.FIB),
        Coordinate(PointXYZ(300.0, 330.0, 0.0), PointType.FIB),
        Coordinate(PointXYZ(400.0, 200.0, 0.0), PointType.SURFACE),
    ]


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FIB-535 — ImagePointCanvas (old) vs CorrelationCanvasWidget (new)")
        self.resize(1500, 820)

        image = _synthetic_image()

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        canvases = QSplitter(Qt.Orientation.Horizontal)
        canvases.addWidget(self._titled("OLD — ImagePointCanvas", self._build_old(image)))
        canvases.addWidget(
            self._titled(
                "NEW — CorrelationCanvasWidget  (contrast/gamma + toolbar)",
                self._build_new(image),
            )
        )
        canvases.setSizes([700, 700])
        root_layout.addWidget(canvases, 1)

        controls = QHBoxLayout()
        for label, slot in (
            ("Reset both", self._reset),
            ("Clear both", self._clear),
            ("Clear log", lambda: self.log.clear()),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            controls.addWidget(btn)
        controls.addStretch()
        controls.addWidget(
            QLabel(
                f"<span style='color:{TEXT_MUTED_COLOR}'>right-click to add · drag to move · "
                f"Delete to remove · contrast button on the right canvas only</span>"
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

    # ── construction ──────────────────────────────────────────────────────

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"<b>{title}</b>")
        label.setStyleSheet(f"color: {TEXT_STRONG_COLOR}; padding: 4px;")
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return box

    def _build_old(self, image: np.ndarray) -> QWidget:
        self.old = ImagePointCanvas(allowed_point_types=_FIB_TYPES)
        self.old.set_image(image)
        self.old.point_selected.connect(lambda c: self._say("old", "selected", c))
        self.old.point_moved.connect(lambda c: self._say("old", "moved", c))
        self.old.point_removed.connect(lambda c: self._say("old", "removed", c))
        self.old.point_add_requested.connect(
            lambda x, y, pt: self._added("old", x, y, pt)
        )
        return self.old

    def _build_new(self, image: np.ndarray) -> QWidget:
        self.new = CorrelationCanvasWidget(allowed_point_types=_FIB_TYPES)
        self.new.canvas.set_array(image)
        self.new.point_selected.connect(lambda c: self._say("new", "selected", c))
        self.new.point_moved.connect(lambda c: self._say("new", "moved", c))
        self.new.point_removed.connect(lambda c: self._say("new", "removed", c))
        self.new.point_add_requested.connect(
            lambda x, y, pt: self._added("new", x, y, pt)
        )
        return self.new

    # ── behaviour ─────────────────────────────────────────────────────────

    def _reset(self) -> None:
        # Separate Coordinate objects per side: they are held by reference, so
        # sharing one list would let a drag on one canvas silently move the other
        # and hide exactly the difference this harness exists to show.
        self.old.set_coordinates(_seed_points())
        self.new.set_coordinates(_seed_points())
        self._log("reset both canvases to 3 seed points")

    def _clear(self) -> None:
        self.old.set_coordinates([])
        self.new.set_coordinates([])
        self._log("cleared both")

    def _added(self, side: str, x: float, y: float, pt: PointType) -> None:
        """Both canvases only *request* an add -- the tab widget builds the
        Coordinate, because it owns the z. Mirror that here."""
        surface = self.old if side == "old" else self.new
        coord = Coordinate(PointXYZ(x, y, 0.0), pt)
        current = list(
            surface.points.coordinates() if side == "new" else surface._coordinates
        )
        surface.set_coordinates(current + [coord])
        self._log(f"[{side}] add_requested  {pt.value:12} ({x:7.1f}, {y:7.1f})", ACCENT_COLOR)

    def _say(self, side: str, what: str, coord: Coordinate) -> None:
        self._log(
            f"[{side}] {what:9} {coord.point_type.value:12} "
            f"({coord.point.x:7.1f}, {coord.point.y:7.1f})",
            OK_COLOR if side == "new" else None,
        )

    def _log(self, message: str, color: str = None) -> None:
        if color:
            self.log.append(f"<span style='color:{color}'>{message}</span>")
        else:
            self.log.append(message)


def main() -> None:
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
