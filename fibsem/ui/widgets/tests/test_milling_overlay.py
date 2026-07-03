"""Standalone demo for MillingPatternOverlay.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_overlay.py

Shows a few milling stages drawn on a blank FIB canvas — one colour per stage,
each with a crosshair at its point-of-interest. "Toggle patterns" clears / re-shows
them to demonstrate that the overlay draws nothing when there is no milling.
"""
import sys

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import (
    CirclePattern,
    LinePattern,
    RectanglePattern,
    TrenchPattern,
)
from fibsem.structures import FibsemImage, Point
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.overlays.milling_overlay import MillingPatternOverlay


def _stage(name: str, pattern, x_um: float, y_um: float) -> FibsemMillingStage:
    pattern.point = Point(x=x_um * 1e-6, y=y_um * 1e-6)
    return FibsemMillingStage(name=name, pattern=pattern)


def build_stages():
    """A spread of pattern types / positions to exercise the overlay."""
    return [
        _stage("Trench", TrenchPattern(), 0, 0),
        _stage("Rect top", RectanglePattern(width=12e-6, height=5e-6), 0, 16),
        _stage("Rotated", RectanglePattern(width=10e-6, height=3e-6, rotation=25), -22, -2),
        _stage("Circle", CirclePattern(radius=4e-6), 20, 10),
        _stage("Line", LinePattern(start_x=-8e-6, end_x=8e-6, start_y=-16e-6, end_y=-16e-6), 0, 0),
    ]


class MillingOverlayTest(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Milling overlay — test")
        self.resize(900, 800)

        self.image = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
        self.canvas = FibsemImageCanvas()
        self.canvas.set_image(self.image)

        self.overlay = MillingPatternOverlay()
        self.canvas.add_overlay(self.overlay)

        stages = build_stages()
        self.fg = stages[:3]          # foreground (coloured); one is "selected"
        self.bg = stages[3:]          # background (drawn black)
        self.selected = 0
        self._shown = True
        self._render()

        self.btn = QPushButton("Toggle patterns (clear / show)")
        self.btn.clicked.connect(self._toggle)
        self.btn_sel = QPushButton("Cycle selected stage")
        self.btn_sel.clicked.connect(self._cycle_selected)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        lay.addWidget(self.btn)
        lay.addWidget(self.btn_sel)

    def _render(self) -> None:
        self.overlay.set_stages(
            self.fg, self.image, background_stages=self.bg, selected_index=self.selected
        )

    def _toggle(self) -> None:
        self._shown = not self._shown
        if self._shown:
            self._render()
        else:
            self.overlay.clear()

    def _cycle_selected(self) -> None:
        self.selected = (self.selected + 1) % len(self.fg)
        if self._shown:
            self._render()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MillingOverlayTest()
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
