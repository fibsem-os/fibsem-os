"""Standalone test script for FibsemMillingStagesWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_stages_widget.py
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.milling.base import FibsemMillingStage, get_strategy
from fibsem.milling.patterning import get_pattern
from fibsem.structures import FibsemMillingSettings
from fibsem.ui.widgets.milling_stages_widget import FibsemMillingStagesWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    rough = FibsemMillingStage(name="Rough Cut")
    rough.pattern = get_pattern("Trench")
    rough.milling = FibsemMillingSettings(milling_current=7.6e-9, rate=3.0e-1)
    rough.strategy = get_strategy("Standard")

    polish = FibsemMillingStage(name="Polish")
    polish.pattern = get_pattern("Rectangle")
    polish.milling = FibsemMillingSettings(milling_current=300e-12, rate=5.0e-2)
    polish.strategy = get_strategy("Overtilt")

    fiducial = FibsemMillingStage(name="Fiducial")
    fiducial.pattern = get_pattern("Fiducial")
    fiducial.milling = FibsemMillingSettings(milling_current=1e-9)
    fiducial.strategy = get_strategy("Standard")

    stages = [rough, polish, fiducial]

    win = QWidget()
    win.setWindowTitle("FibsemMillingStagesWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.resize(700, 700)

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    # --- controls row ---
    controls = QWidget()
    ctrl_layout = QFormLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)

    mfr_combo = QComboBox()
    mfr_combo.addItems(["Demo (none)", "ThermoFisher", "Tescan"])
    ctrl_layout.addRow("Manufacturer:", mfr_combo)

    adv_check = QCheckBox("Show advanced")
    ctrl_layout.addRow("Advanced:", adv_check)

    root.addWidget(controls)

    # --- the widget under test ---
    widget = FibsemMillingStagesWidget(microscope=microscope, stages=stages)
    root.addWidget(widget)

    # --- status + buttons ---
    status = QLabel("Select a stage and change settings to see stages_changed.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_get = QPushButton("Print get_stages()")
    btn_enabled = QPushButton("Print get_enabled_stages()")
    btn_layout.addWidget(btn_get)
    btn_layout.addWidget(btn_enabled)
    root.addWidget(btn_row)

    # --- connections ---
    def on_mfr_changed(idx: int) -> None:
        mfr_map = {0: "Demo", 1: "ThermoFisher", 2: "Tescan"}
        widget.set_manufacturer(mfr_map[idx])

    def on_stages_changed(changed_stages) -> None:
        names = [s.name for s in changed_stages]
        status.setText(f"stages_changed: {names}")
        print("stages_changed:", [(s.name, s.milling.milling_current) for s in changed_stages])

    def on_print_all() -> None:
        for s in widget.get_stages():
            print(s.name, s.milling.to_dict())

    def on_print_enabled() -> None:
        for s in widget.get_enabled_stages():
            print(s.name, s.milling.to_dict())

    mfr_combo.currentIndexChanged.connect(on_mfr_changed)
    adv_check.toggled.connect(widget.set_advanced_visible)
    widget.stages_changed.connect(on_stages_changed)
    btn_get.clicked.connect(on_print_all)
    btn_enabled.clicked.connect(on_print_enabled)

    # start with ThermoFisher
    mfr_combo.setCurrentIndex(1)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
