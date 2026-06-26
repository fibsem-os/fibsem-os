"""Standalone demo for the quad-view microscope display.

Run directly:
    python fibsem/ui/widgets/tests/test_quad_view.py

Shows a QuadViewWidget driven through a MicroscopeViewController: SEM / FIB / FM
get blank images, the 4th cell is the inert "No Data" placeholder. Each image
cell carries the full FibsemImageCanvas toolbar (reset / scalebar / crosshair /
contrast) and supports zoom / pan.
"""
import sys

from PyQt5.QtWidgets import QApplication

from fibsem.structures import BeamType, FibsemImage
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.quad_view import MicroscopeViewController


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    controller = MicroscopeViewController()
    controller.set_image(
        BeamType.ELECTRON, FibsemImage.generate_blank_image(hfw=80e-6, random=True)
    )
    controller.set_image(
        BeamType.ION, FibsemImage.generate_blank_image(hfw=80e-6, random=True)
    )
    controller.set_fm_image(
        FibsemImage.generate_blank_image(hfw=150e-6, random=True)
    )

    win = controller.widget
    win.setWindowTitle("Quad view — test")
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
