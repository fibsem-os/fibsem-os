"""Standalone demo for AlignmentAreaOverlay.

Run directly:
    python fibsem/ui/widgets/tests/test_alignment_overlay.py

Drag/resize the dashed lime alignment rectangle on the FIB image; the label shows
the normalized FibsemRectangle (and whether it's valid). "Toggle editable"
switches between editable (corner handles + drag/resize) and read-only (static).
"""
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from fibsem.structures import FibsemImage, FibsemRectangle
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.overlays.alignment_overlay import AlignmentAreaOverlay


class AlignmentOverlayTest(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Alignment overlay — test")
        self.resize(900, 800)

        self.image = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
        self.canvas = FibsemImageCanvas()
        self.canvas.set_image(self.image)

        self.overlay = AlignmentAreaOverlay(editable=True)
        self.canvas.add_overlay(self.overlay)
        self.overlay.set_area(FibsemRectangle(0.25, 0.25, 0.5, 0.5))
        self.overlay.alignment_area_changed.connect(self._show_area)

        self.label = QLabel()
        self.label.setStyleSheet("color: #d1d2d4; padding: 6px; font-family: monospace;")
        self._show_area(self.overlay.get_area())

        self._editable = True
        self.btn = QPushButton("Toggle editable (now: editable)")
        self.btn.clicked.connect(self._toggle_editable)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        lay.addWidget(self.label)
        lay.addWidget(self.btn)

    def _show_area(self, a: FibsemRectangle) -> None:
        self.label.setText(
            f"alignment area:  left={a.left:.3f}  top={a.top:.3f}  "
            f"width={a.width:.3f}  height={a.height:.3f}  valid={a.is_valid_reduced_area}"
        )

    def _toggle_editable(self) -> None:
        self._editable = not self._editable
        self.overlay.set_editable(self._editable)
        self.btn.setText(
            f"Toggle editable (now: {'editable' if self._editable else 'read-only'})"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AlignmentOverlayTest()
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
