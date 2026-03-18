#!/usr/bin/env python
"""Standalone test/runner for MicroscopeConfigWidget — no napari required.

Usage:
    python scripts/test_microscope_config_widget.py
    python scripts/test_microscope_config_widget.py --config /path/to/config.yaml
"""

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from fibsem import config as cfg
from fibsem.ui.widgets.microscope_config_widget import MicroscopeConfigWidget


def main():
    parser = argparse.ArgumentParser(description="MicroscopeConfigWidget test runner")
    parser.add_argument(
        "--config",
        default=cfg.DEFAULT_CONFIGURATION_PATH,
        help="Path to microscope configuration YAML file",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = MicroscopeConfigWidget(path=args.config)
    w.setWindowTitle("Microscope Configuration")
    w.resize(640, 720)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
