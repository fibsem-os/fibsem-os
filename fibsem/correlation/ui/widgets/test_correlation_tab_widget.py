"""Manual launcher for CorrelationTabWidget, pre-loaded from a project directory.

Quickstart: loads the FIB + FM images and the saved correlation result (or
coordinate data) from a project directory — defaults to the worktree's ``tmp/``.

Usage
-----
    # quickstart from tmp/ (FIB + FM images + correlation_result.json)
    python -m fibsem.correlation.ui.widgets.test_correlation_tab_widget

    # or point at any correlation project directory
    python -m fibsem.correlation.ui.widgets.test_correlation_tab_widget /path/to/project
"""
import sys
from pprint import pprint

from PyQt5.QtWidgets import QApplication, QMainWindow

from fibsem.correlation.structures import CorrelationResult
from fibsem.correlation.ui.widgets.correlation_tab_widget import (
    CorrelationTabWidget,
    load_project,
)
from fibsem.ui.stylesheets import NAPARI_STYLE

_DEFAULT_PROJECT = "tmp"


class _LauncherWindow(QMainWindow):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.setWindowTitle("CorrelationTabWidget — test")
        self.resize(1400, 900)

        self._widget = CorrelationTabWidget()
        load_project(self._widget, directory)
        self.setCentralWidget(self._widget)

        self._widget.continue_pressed_signal.connect(self._print_continue)

    def _print_continue(self, result: CorrelationResult) -> None:
        print(f"POI: {len(result.poi)}")
        if result.poi:
            pprint(result.poi[0])


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PROJECT

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(NAPARI_STYLE)

    win = _LauncherWindow(directory)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
