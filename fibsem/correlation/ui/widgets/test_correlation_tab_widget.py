"""Test script for CorrelationTabWidget.

Usage
-----
    python fibsem/correlation/ui/widgets/test_correlation_tab_widget.py
"""
import os
from pprint import pprint
import sys

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow

from fibsem.correlation.ui.widgets.correlation_tab_widget import CorrelationTabWidget
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.correlation.structures import CorrelationResult, CorrelationInputData

_DEV_PATH = "/home/patrick/github/fibsem/fibsem/applications/test-data"
_FIB_IMAGE = "ref_ReferenceImage-Spot-Burn-Fiducial-10-36-30_res_02_ib.tif"
_FM_IMAGE  = "zstack-Feature-1-Active-002.ome.tiff"


class TestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CorrelationTabWidget — test")
        self.resize(1400, 900)

        fib_image = fm_image = None
        try:
            fib_image = FibsemImage.load(os.path.join(_DEV_PATH, _FIB_IMAGE))
            print("FIB image loaded.")
        except Exception as exc:
            print(f"Could not load FIB image: {exc}")
        try:
            fm_image = FluorescenceImage.load(os.path.join(_DEV_PATH, _FM_IMAGE))
            print("FM image loaded.")
        except Exception as exc:
            print(f"Could not load FM image: {exc}")

        self._widget = CorrelationTabWidget(fib_image=fib_image, fm_image=fm_image)
        self.setCentralWidget(self._widget)


        self._widget.continue_pressed_signal.connect(self._print_continue)

    def _print_continue(self, result: CorrelationResult):
        # print("Continue pressed! Result:", result)

        print(f"POI: {len(result.poi)}")
        pprint(result.poi[0])

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(NAPARI_STYLE)

    win = TestWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
