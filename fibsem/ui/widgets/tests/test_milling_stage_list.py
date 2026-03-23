"""Standalone test script for MillingStageListWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_stage_list.py
"""
import sys

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning import get_pattern
from fibsem.structures import FibsemMillingSettings
from fibsem.ui.widgets.milling_stage_list_widget import MillingStageListWidget


def _make_stage(name: str, pattern_name: str, depth: float, current: float) -> FibsemMillingStage:
    pattern = get_pattern(pattern_name, {"depth": depth, "width": 10e-6, "height": 5e-6})
    milling = FibsemMillingSettings(milling_current=current)
    return FibsemMillingStage(name=name, milling=milling, pattern=pattern)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    stages = [
        _make_stage("Rough Cut",    "Rectangle", depth=10e-6, current=2e-9),
        _make_stage("Regular Cut",  "Rectangle", depth=5e-6,  current=300e-12),
        _make_stage("Polish",       "Rectangle", depth=1e-6,  current=50e-12),
        _make_stage("Fiducial",     "Circle",    depth=2e-6,  current=100e-12),
    ]

    # --- main window ---
    win = QWidget()
    win.setWindowTitle("MillingStageListWidget — test")
    # win.setMinimumWidth(800)
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    list_widget = MillingStageListWidget()
    list_widget.set_stages(stages)
    root.addWidget(list_widget)

    # status label shows selected stage
    status = QLabel("Click a row to select a stage")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    def on_selected(stage: FibsemMillingStage) -> None:
        status.setText(f"Selected: {stage.name}  |  {stage.pattern.name}  |  {stage.strategy.name}")

    def on_removed(stage: FibsemMillingStage) -> None:
        status.setText(f"Removed: {stage.name}")

    def on_enabled(enabled_stages) -> None:
        names = [s.name for s in enabled_stages]
        print("Enabled stages:", names)

    def on_order(ordered_stages) -> None:
        names = [s.name for s in ordered_stages]
        print("Order changed:", names)

    list_widget.stage_selected.connect(on_selected)
    list_widget.stage_removed.connect(on_removed)
    list_widget.enabled_changed.connect(on_enabled)
    list_widget.order_changed.connect(on_order)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
