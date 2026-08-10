"""Hands-on harness for CorrelationFMCanvasWidget — the FM correlation surface.

Was a side-by-side against `FMImageDisplayWidget` while the swap was in flight;
that widget is gone, so this is the single-widget form. It survives because a
multi-channel z-stack is otherwise only reachable with real data, and because in
its side-by-side days it caught two things no test did: the max-projection
default (which would have given every picked point z=0) and a `set_pixel_size`
override that shadowed its parent.

Usage
-----
    PYTHONPATH=$PWD python fibsem/ui/correlation/widgets/test_correlation_fm_canvas_demo.py

What to try
-----------
* **Points** — right-click to add, drag to move, click to select, Delete to
  remove. The log shows what is emitted, including the z each new point takes.
* **Z** — slider, `‹` / `›` step buttons, an ``n/N`` readout, and Shift+scroll on
  the image. Loading a stack starts you **mid-stack**, not at plane 0, because
  plane 0 is the out-of-focus edge of the volume. Shift+scroll is dead on a macOS
  mouse (FIB-552, parked); the arrows and slider work everywhere.
* **Max projection** — the toolbar button. Off by default here: a projection has
  no plane, so a point picked on one has no z to take.
* **Channels** — the layers popover (``mdi:layers``): visibility, colormap,
  opacity, gamma and a dual-handle contrast, per channel. The same controls the
  quad view and FM overview use.
* **Crosshair** — off, because correlation draws SURFACE as an orange `+`.
  Toolbar toggle brings it back.
* **Show result** — the marker groups the tab widget draws after a run.
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
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.ui.correlation.widgets.correlation_fm_canvas_widget import (
    CorrelationFMCanvasWidget,
)
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    OK_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

# The types _POINT_TYPE_SIDES assigns to the FM side, so the add menu matches production.
_FM_TYPES = [PointType.FM, PointType.POI, PointType.SURFACE_FM]
_CHANNELS = (("GFP", "#00ff00"), ("RFP", "#ff3030"), ("DAPI", "#4090ff"))
_NZ, _SIZE = 9, 256


def _fm_image() -> FluorescenceImage:
    """A 3-channel z-stack whose blobs come into focus on different planes, so the
    z-slider and max-projection visibly do something."""
    rng = np.random.default_rng(7)
    data = np.zeros((len(_CHANNELS), _NZ, _SIZE, _SIZE), dtype=np.uint16)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    for c, (_, _colour) in enumerate(_CHANNELS):
        cy, cx = 80 + 45 * c, 90 + 40 * c
        best_z = 2 + 3 * c  # each channel sharpest on a different plane
        for z in range(_NZ):
            spread = 18 + 9 * abs(z - best_z)
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * spread**2)))
            frame = 900 * blob / (1 + 0.35 * abs(z - best_z))
            frame += rng.normal(60, 12, (_SIZE, _SIZE))
            data[c, z] = np.clip(frame, 0, 65535).astype(np.uint16)

    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-08-10T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(_SIZE, _SIZE),
        channels=[
            FluorescenceChannelMetadata(
                name=name, color=colour, excitation_wavelength=488,
                emission_wavelength=509, power=1.0, exposure_time=0.1,
                gain=1.0, offset=0.0,
            )
            for name, colour in _CHANNELS
        ],
        filename="synthetic_zstack",
    )
    return FluorescenceImage(data=data, metadata=metadata)


def _seed_points() -> list:
    return [
        Coordinate(PointXYZ(90.0, 80.0, 2.0), PointType.FM),
        Coordinate(PointXYZ(130.0, 125.0, 5.0), PointType.FM),
        Coordinate(PointXYZ(170.0, 170.0, 8.0), PointType.SURFACE_FM),
    ]


# The groups CorrelationTabWidget._overlay_result_on_fib draws, in FM colours.
_RESULT_GROUPS = (
    (
        [(96.0, 86.0), (125.0, 132.0)],
        dict(color="#ff4444", label_prefix="E", size=4, legend_label="reprojected (E)"),
    ),
    (
        [(150.0, 150.0)],
        dict(color="#ff00ff", size=7, alpha=0.7, show_labels=False, hollow=True,
             legend_label="POI uncorrected"),
    ),
    (
        [(156.0, 154.0)],
        dict(color="#ff00ff", label_prefix="P", size=5, legend_label="POI (P)"),
    ),
)


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "CorrelationFMCanvasWidget — FM correlation surface"
        )
        self.resize(1100, 900)

        image = _fm_image()

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.fm = CorrelationFMCanvasWidget(allowed_point_types=_FM_TYPES)
        self.fm.set_fm_image(image)
        self._wire(self.fm)
        root_layout.addWidget(self.fm, 1)

        controls = QHBoxLayout()
        for label, slot in (
            ("Reset both", self._reset),
            ("Clear both", self._clear),
            ("Show result", self._show_result),
            ("Clear result", self._clear_result),
            ("Clear log", lambda: self.log.clear()),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        for label, setter in (("Legend", "set_legend_visible"), ("Labels", "set_labels_visible")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(lambda on, s=setter, n=label: self._toggle_both(s, on, n))
            controls.addWidget(btn)

        controls.addStretch()
        controls.addWidget(
            QLabel(
                f"<span style='color:{TEXT_MUTED_COLOR}'>right-click to add · drag to move · "
                f"Delete to remove · uncheck Max Projection to reach the z controls</span>"
            )
        )
        root_layout.addLayout(controls)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        self.log.setStyleSheet(
            f"background: {SURFACE_COLOR}; color: {TEXT_STRONG_COLOR}; "
            "font-family: monospace; font-size: 11px;"
        )
        root_layout.addWidget(self.log)

        self._reset()

    # ── construction ──────────────────────────────────────────────────────

    def _wire(self, widget) -> None:
        widget.point_selected.connect(lambda c: self._say("selected", c))
        widget.point_moved.connect(lambda c: self._say("moved", c))
        widget.point_removed.connect(lambda c: self._say("removed", c))
        widget.point_add_requested.connect(self._added)

    # ── behaviour ─────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self.fm.set_coordinates(_seed_points())
        self._log("reset to 3 seed points")

    def _clear(self) -> None:
        self.fm.set_coordinates([])
        self._log("cleared")

    def _show_result(self) -> None:
        self.fm.clear_overlay()
        for points, style in _RESULT_GROUPS:
            self.fm.add_overlay_points(points, **style)
        self._log("drew the reprojected result", OK_COLOR)

    def _clear_result(self) -> None:
        self.fm.clear_overlay()
        self._log("cleared the result")

    def _toggle_both(self, setter: str, on: bool, name: str) -> None:
        getattr(self.fm, setter)(on)
        self._log(f"{name.lower()} {'shown' if on else 'hidden'}")

    def _added(self, x: float, y: float, pt: PointType) -> None:
        """The widget only *requests* an add — the tab widget builds the
        Coordinate, because it owns the z. Mirror that, reading z from the
        slider, which is the number a real correlation depends on."""
        coord = Coordinate(PointXYZ(x, y, float(self.fm.current_z)), pt)
        self.fm.set_coordinates(list(self.fm.points.coordinates()) + [coord])
        self._log(
            f"add_requested  {pt.value:12} ({x:7.1f}, {y:7.1f})  z={self.fm.current_z}",
            ACCENT_COLOR,
        )

    def _say(self, what: str, coord: Coordinate) -> None:
        self._log(
            f"{what:9} {coord.point_type.value:12} "
            f"({coord.point.x:7.1f}, {coord.point.y:7.1f}, z={coord.point.z:5.1f})"
        )

    def _log(self, message: str, color: str = None) -> None:
        self.log.append(f"<span style='color:{color}'>{message}</span>" if color else message)


def main() -> None:
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
