"""Quick test script for LinePlotWidget with random data generation."""
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from fibsem.ui.fm.widgets.line_plot_widget import LinePlotWidget


def main():
    app = QApplication(sys.argv)

    widget = LinePlotWidget(max_length=5000)
    widget.setWindowTitle("LinePlotWidget Test")
    widget.resize(600, 400)
    widget.show()

    # Generate random data with a slow drift
    step = [0]
    value = [50.0]

    def add_point():
        # Random walk with slight upward drift
        value[0] += np.random.randn() * 2 + 0.05
        widget.append_value(value[0], "Test Plot")
        step[0] += 1

    # Add a new point every 50ms
    timer = QTimer()
    timer.timeout.connect(add_point)
    timer.start(50)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
