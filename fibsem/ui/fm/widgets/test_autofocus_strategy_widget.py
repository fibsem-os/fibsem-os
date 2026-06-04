"""Standalone test script for AutoFocusStrategyWidget.

Run directly:
    python fibsem/ui/fm/widgets/test_autofocus_strategy_widget.py
"""
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fibsem.fm.strategy import CoarseFineAutoFocusConfig
from fibsem.fm.strategy.base import AutoFocusStrategyConfig
from fibsem.ui.fm.widgets.autofocus_strategy_widget import AutoFocusStrategyWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QWidget()
    win.setWindowTitle("AutoFocusStrategyWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.setMinimumWidth(340)

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    # Start with a CoarseFine config to show a non-default initial state
    initial_config = CoarseFineAutoFocusConfig(coarse_range=80e-6, fine_range=15e-6)
    widget = AutoFocusStrategyWidget(config=initial_config)
    root.addWidget(widget)

    status = QLabel(f"Config: {initial_config.name}")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    def on_config_changed(config: AutoFocusStrategyConfig) -> None:
        d = config.to_dict()
        print(f"config_changed: {d}")
        status.setText(f"Config: {config.name} — {d}")

    widget.config_changed.connect(on_config_changed)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
