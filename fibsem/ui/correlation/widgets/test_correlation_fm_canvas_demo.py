"""Side-by-side harness: FMImageDisplayWidget (old) vs CorrelationFMCanvasWidget (new).

Same synthetic 3-channel z-stack and the same starting points on both. Built
before the swap, deliberately — the FIB side had `test_correlation_canvas_demo.py`
to compare against and this is the FM equivalent, because the correlation tab is
in production and 925 passing Qt tests have previously said nothing useful about
whether a widget actually works.

Usage
-----
    PYTHONPATH=$PWD python fibsem/ui/correlation/widgets/test_correlation_fm_canvas_demo.py

What to compare
---------------
* **Points** — right-click to add, drag to move, click to select, Delete to
  remove. The log shows what each side emits; the payloads should match.
* **Z** — both have a slider, `‹` / `›` step buttons and an ``n/N`` readout. The
  arrows are new on the shared widget, added here so the swap does not lose a
  control the correlation display has today (the quad view and FM overview gain
  them too).
* **Max projection** — an inline checkbox on the left, a toolbar button on the
  right. Deliberate: it keeps the z row thin.
* **Channels** — an always-visible pane on the left, the layers popover (the
  ``mdi:layers`` toolbar button) on the right. The popover offers colormap,
  opacity, gamma and a dual-handle contrast, against the left's clip percentage.
  This is the agreed UX trade; check it is liveable while actually picking points.
* **Contrast / gamma** — per channel, right side only.
* **Crosshair** — off on both correlation canvases, on elsewhere. Toolbar toggle
  brings it back.
* **Show result** — the three marker groups the tab widget draws after a run.
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
from fibsem.ui.correlation.widgets.fm_image_display_widget import FMImageDisplayWidget
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
            "FIB-535 — FMImageDisplayWidget (old) vs CorrelationFMCanvasWidget (new)"
        )
        self.resize(1600, 900)

        image = _fm_image()

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        panes = QSplitter(Qt.Orientation.Horizontal)
        panes.addWidget(self._titled("OLD — FMImageDisplayWidget", self._build_old(image)))
        panes.addWidget(
            self._titled(
                "NEW — CorrelationFMCanvasWidget  (layers popover + contrast/gamma)",
                self._build_new(image),
            )
        )
        panes.setSizes([780, 780])
        root_layout.addWidget(panes, 1)

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

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"<b>{title}</b>")
        label.setStyleSheet(f"color: {TEXT_STRONG_COLOR}; padding: 4px;")
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return box

    def _wire(self, widget, side: str) -> None:
        widget.point_selected.connect(lambda c: self._say(side, "selected", c))
        widget.point_moved.connect(lambda c: self._say(side, "moved", c))
        widget.point_removed.connect(lambda c: self._say(side, "removed", c))
        widget.point_add_requested.connect(lambda x, y, pt: self._added(side, x, y, pt))

    def _build_old(self, image: FluorescenceImage) -> QWidget:
        self.old = FMImageDisplayWidget(allowed_point_types=_FM_TYPES)
        self.old.set_fm_image(image)
        self._wire(self.old, "old")
        return self.old

    def _build_new(self, image: FluorescenceImage) -> QWidget:
        self.new = CorrelationFMCanvasWidget(allowed_point_types=_FM_TYPES)
        self.new.set_fm_image(image)
        self._wire(self.new, "new")
        return self.new

    # ── behaviour ─────────────────────────────────────────────────────────

    def _reset(self) -> None:
        # Separate Coordinate objects per side: they are held by reference, so
        # sharing one list would let a drag on one widget silently move the other
        # and hide exactly the difference this harness exists to show.
        self.old.set_coordinates(_seed_points())
        self.new.set_coordinates(_seed_points())
        self._log("reset both to 3 seed points")

    def _clear(self) -> None:
        for w in (self.old, self.new):
            w.set_coordinates([])
        self._log("cleared both")

    @staticmethod
    def _surface(widget, method: str):
        """Whichever of the widget or its canvas answers to *method*.

        FMImageDisplayWidget exposes almost nothing at widget level -- the tab
        widget reaches through `.canvas` for the chrome toggles, and result
        markers were never on the FM side at all (only the FIB canvas drew
        them). CorrelationFMCanvasWidget offers all of it directly, which is
        both why the swap touches those call sites and what FIB-289 (overlay the
        correlated POI on the FM display) needs.
        """
        return widget if hasattr(widget, method) else widget.canvas

    def _show_result(self) -> None:
        for w in (self.old, self.new):
            surface = self._surface(w, "add_overlay_points")
            surface.clear_overlay()
            for points, style in _RESULT_GROUPS:
                surface.add_overlay_points(points, **style)
        self._log("drew the reprojected result on both", OK_COLOR)

    def _clear_result(self) -> None:
        for w in (self.old, self.new):
            self._surface(w, "clear_overlay").clear_overlay()
        self._log("cleared the result on both")

    def _toggle_both(self, setter: str, on: bool, name: str) -> None:
        for w in (self.old, self.new):
            getattr(self._surface(w, setter), setter)(on)
        self._log(f"{name.lower()} {'shown' if on else 'hidden'} on both")

    def _added(self, side: str, x: float, y: float, pt: PointType) -> None:
        """Both widgets only *request* an add — the tab widget builds the
        Coordinate, because it owns the z. Mirror that here, reading z from
        whichever widget was clicked."""
        widget = self.old if side == "old" else self.new
        coord = Coordinate(PointXYZ(x, y, float(widget.current_z)), pt)
        widget.set_coordinates(list(widget.points.coordinates()) + [coord])
        self._log(
            f"[{side}] add_requested  {pt.value:12} ({x:7.1f}, {y:7.1f})  z={widget.current_z}",
            ACCENT_COLOR,
        )

    def _say(self, side: str, what: str, coord: Coordinate) -> None:
        self._log(
            f"[{side}] {what:9} {coord.point_type.value:12} "
            f"({coord.point.x:7.1f}, {coord.point.y:7.1f}, z={coord.point.z:5.1f})",
            OK_COLOR if side == "new" else None,
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
