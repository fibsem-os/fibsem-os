"""Standalone test script for FibsemPatternSettingsWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_pattern_settings_widget.py
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.milling.patterning import get_pattern
from fibsem.ui.widgets.pattern_settings_widget import FibsemPatternSettingsWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    pattern = get_pattern("Rectangle")

    win = QWidget()
    win.setWindowTitle("FibsemPatternSettingsWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.resize(500, 600)

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    # Controls
    controls = QWidget()
    ctrl_layout = QFormLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)
    adv_check = QCheckBox("Show advanced")
    ctrl_layout.addRow("Advanced:", adv_check)
    root.addWidget(controls)

    # Widget under test
    widget = FibsemPatternSettingsWidget(microscope=microscope, pattern=pattern)
    root.addWidget(widget)

    # Status
    status = QLabel("Change any control to see pattern_changed.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_get = QPushButton("Print get_pattern()")
    btn_layout.addWidget(btn_get)
    root.addWidget(btn_row)

    root.addStretch()

    # Connections
    def on_pattern_changed(p) -> None:
        status.setText(f"pattern_changed: {p.name}")
        print("pattern_changed:", p.to_dict())

    def on_print() -> None:
        p = widget.get_pattern()
        print(p.to_dict())

    adv_check.toggled.connect(widget.set_advanced_visible)
    widget.pattern_changed.connect(on_pattern_changed)
    btn_get.clicked.connect(on_print)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
