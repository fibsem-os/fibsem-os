"""FibsemDualBeamWidget on the Demo microscope.

Run with:
    python demos/widgets/demo_dual_beam_widget.py

Starts the simulated Demo microscope. Switch between the SEM and FIB tabs and change
any setting: the change goes to the simulator, and the widget's signals print it.
"Print Active Settings" prints the beam and detector settings the active tab holds.
"""

import sys
from pprint import pprint

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from fibsem import utils
from fibsem.ui.widgets.dual_beam_widget import FibsemDualBeamWidget


def main() -> int:
    app = QApplication(sys.argv)

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    window = QWidget()
    layout = QVBoxLayout(window)

    widget = FibsemDualBeamWidget(microscope=microscope)
    widget.populate_combos()
    widget.sync_from_microscope()
    layout.addWidget(widget)

    widget.beam_settings_changed.connect(lambda s: print(f"beam_settings_changed: {s}"))
    widget.detector_settings_changed.connect(
        lambda s: print(f"detector_settings_changed: {s}")
    )

    def _print(checked=False):
        print(f"--- {widget.beam_type.name} beam ---")
        pprint(widget.get_beam_settings().to_dict())
        pprint(widget.get_detector_settings().to_dict())

    button = QPushButton("Print Active Settings")
    button.clicked.connect(_print)
    layout.addWidget(button)

    window.setWindowTitle("FibsemDualBeamWidget")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
