"""Standalone demo for FibsemRealSpaceCanvas.

Not a pytest module despite the name — this directory holds GUI harnesses, which pytest
never collects (``testpaths = ["tests"]``). The automated tests live in
``tests/ui/test_real_space_canvas.py``.

Run directly:
    python fibsem/ui/widgets/tests/test_real_space_canvas_demo.py

Shows a 3x3 tileset placed at true stage offsets on a black backdrop, so overlap and
alignment are visible. The cursor readout gives the position in canvas-frame metres.

The second button adds *the same array* at 4x the pixel size, which must therefore cover
4x the ground. That is the property a canvas can only get right by honouring pixel size
per image: resample everything onto one raster, or ignore pixel size, and it would draw
the same size as a grid tile. It is what a decimated live preview or a binned overview
looks like beside a full-resolution tile.

Pan/zoom, the scalebar and the ruler all work as on any canvas.
"""

import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

PIXEL_SIZE = 1e-7  # 100 nm/px
TILE = 256  # px -> 25.6 um per tile
OVERLAP = 0.1
STEP = TILE * PIXEL_SIZE * (1 - OVERLAP)  # centre-to-centre tile spacing, metres
WORLD_EXTENT = 2000e-6  # declared working area, so framing is stable as tiles arrive

# Stage positions to mark, in metres. The second is the centre of the top-left tile, so
# its marker should sit dead-centre on that tile — the check that overlay geometry and
# image placement agree rather than merely looking plausible together.
STAGE_MARKERS = [
    (0.0, 0.0),
    (-STEP, -STEP),
]


def _tile(seed: int, size: int = TILE) -> np.ndarray:
    """A distinguishable tile: noise plus a bright border to show seams.

    Brightness cycles over a narrow band rather than climbing with the seed — scaling it
    linearly saturated every tile past seed ~11 to solid white, which drew as an opaque
    block hiding whatever it overlapped.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=85 + 22 * (seed % 5), scale=20, size=(size, size))
    data[:4, :] = data[-4:, :] = data[:, :4] = data[:, -4:] = 255
    return np.clip(data, 0, 255).astype(np.uint8)


class RealSpaceCanvasDemo(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Real-space canvas — demo")
        self.resize(1000, 900)

        self.canvas = FibsemRealSpaceCanvas()
        self.label = QLabel()
        self.label.setStyleSheet(
            "color: #d1d2d4; padding: 6px; font-family: monospace;"
        )

        # Live cursor position, which is the readout that actually shows the canvas is
        # in real space: hovering the same feature on two tiles should give the same
        # metres, and the numbers should track the tile centres the demo places.
        self.cursor_label = QLabel()
        self.cursor_label.setStyleSheet(
            "color: #ffd23f; padding: 0px 6px 6px 6px; font-family: monospace;"
        )
        self.canvas.mpl_connect("motion_notify_event", self._on_cursor_moved)
        self._on_cursor_moved(None)

        # Stage-position markers, placed through the same metres-to-canvas conversion
        # the images use — so if the two ever disagree, the marker misses its tile.
        self.markers = PointsOverlay(color="#00e5ff", marker="+", size=14)
        self.canvas.add_overlay(self.markers)

        self._n_coarse = 0
        self._add_tileset()

        btn_reset = QPushButton("Reload tileset")
        btn_reset.clicked.connect(self._reload)
        btn_coarse = QPushButton(
            "Add same tile at 4x pixel size (covers 4x the ground)"
        )
        btn_coarse.clicked.connect(self._add_coarse)
        btn_clear = QPushButton("Clear images")
        btn_clear.clicked.connect(self._clear)
        self.btn_world = QPushButton()
        self.btn_world.clicked.connect(self._toggle_world)

        row = QHBoxLayout()
        for b in (btn_reset, btn_coarse, btn_clear, self.btn_world):
            row.addWidget(b)

        self._world_on = True
        self.canvas.set_world_extent(WORLD_EXTENT)
        self._sync_world_button()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        lay.addWidget(self.label)
        lay.addWidget(self.cursor_label)
        lay.addLayout(row)

    # ── actions ───────────────────────────────────────────────────────────

    def _add_tileset(self) -> None:
        """A 3x3 grid stepped by one field of view less the overlap."""
        step = TILE * PIXEL_SIZE * (1 - OVERLAP)
        seed = 0
        for row in range(3):
            for col in range(3):
                x = (col - 1) * step
                y = (row - 1) * step
                self.canvas.add_image(
                    _tile(seed),
                    centre=(x, y),
                    pixel_size=PIXEL_SIZE,
                    key=f"r{row}c{col}",
                )
                seed += 1
        self._refresh_markers()
        self._describe()

    def _refresh_markers(self) -> None:
        """Place the stage markers, converting metres to canvas coordinates.

        Deferred until an image exists, because the reference pixel size — and so the
        conversion — is set by the first image added.
        """
        if self.canvas.reference_pixel_size is None:
            return
        points = [self.canvas.metres_to_canvas(x, y) for x, y in STAGE_MARKERS]
        labels = [f"({x * 1e6:+.1f}, {y * 1e6:+.1f}) um" for x, y in STAGE_MARKERS]
        self.markers.set_points(points, labels=labels)

    def _add_coarse(self) -> None:
        """The same-sized array at 4x the pixel size, so it must cover 4x the ground.

        This is the property that matters: a canvas that resampled everything onto one
        raster, or that ignored pixel size, would draw this the same size as a grid tile.
        It is what a decimated live preview or a binned overview looks like next to a
        full-resolution tile.

        Only the pixel size differs from a grid tile — same array shape — so the
        comparison reads directly as "4x the pixel size, 4x the footprint". Placed clear
        of the grid, since images draw in the order added and an overlapping one would
        simply hide what is beneath.
        """
        self._n_coarse += 1
        grid_half = TILE * PIXEL_SIZE * (1 - OVERLAP) + TILE * PIXEL_SIZE / 2
        coarse_half = TILE * PIXEL_SIZE * 4 / 2
        pitch = 2 * coarse_half + 5e-6
        # Wrap into a block rather than a single column: marching in one direction gives
        # a long thin footprint, and with aspect="equal" that squeezes the whole scene
        # into a sliver of the window.
        col, row = (self._n_coarse - 1) % 4, (self._n_coarse - 1) // 4
        x = grid_half + coarse_half + 5e-6 + col * pitch
        y = row * pitch - coarse_half
        self.canvas.add_image(
            _tile(self._n_coarse),
            centre=(x, y),
            pixel_size=PIXEL_SIZE * 4,
            key=f"coarse-{self._n_coarse}",
        )
        self._describe()

    def _toggle_world(self) -> None:
        """Compare a declared working area against framing that hugs the content."""
        self._world_on = not self._world_on
        self.canvas.set_world_extent(WORLD_EXTENT if self._world_on else None)
        self._sync_world_button()
        self._describe()

    def _sync_world_button(self) -> None:
        self.btn_world.setText(
            f"Working area: {WORLD_EXTENT * 1e6:.0f} um"
            if self._world_on
            else "Working area: off (fit to content)"
        )

    def _on_cursor_moved(self, event) -> None:
        """Report the cursor in canvas-frame metres — the demo's real-space readout."""
        if event is None or event.xdata is None or event.ydata is None:
            self.cursor_label.setText("cursor:  —")
            return
        x, y = self.canvas.canvas_to_metres(event.xdata, event.ydata)
        self.cursor_label.setText(
            f"cursor:  x={x * 1e6:+8.2f} um   y={y * 1e6:+8.2f} um"
        )

    def _reload(self) -> None:
        self.canvas.clear_images()
        self._n_coarse = 0
        self._add_tileset()

    def _clear(self) -> None:
        self.canvas.clear_images()
        self._describe()

    def _describe(self) -> None:
        rect = self.canvas._content_rect()
        ref = self.canvas.reference_pixel_size
        if rect.is_empty or not ref:
            self.label.setText("no images placed")
            return
        self.label.setText(
            f"{len(self.canvas.placed_keys)} images   "
            f"reference={ref * 1e9:.0f} nm/px   "
            f"span={rect.width * ref * 1e6:.1f} x {rect.height * ref * 1e6:.1f} um   "
            f"centre=({rect.cx * ref * 1e6:+.1f}, {rect.cy * ref * 1e6:+.1f}) um"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RealSpaceCanvasDemo()
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
