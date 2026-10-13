"""FibsemOverviewWidget on the Demo microscope.

Run with:
    python demos/widgets/demo_overview_widget.py

Starts the simulated Demo microscope and opens the overview on its own, outside the
app. The QApplication is deliberately left unstyled: in the app this widget is hosted
in a tab, so only the widget gets the stylesheet here, as it does there. Styling the
whole application would make the demo look better than the real thing, and hide a
defect that only shows in the tab.
"""

import sys

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.ui import stylesheets
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget


def main() -> int:
    app = QApplication(sys.argv)
    microscope, _ = utils.setup_session(manufacturer="Demo")

    widget = FibsemOverviewWidget(microscope)
    widget.setStyleSheet(stylesheets.NAPARI_STYLE)
    widget.resize(1400, 900)
    widget.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
