"""FMOverviewWidget on the simulated microscope, at the FM.

Run with:
    python demos/fm/demo_fm_overview_widget.py
    python demos/fm/demo_fm_overview_widget.py --output ~/fm-overviews   # and save them
    python demos/fm/demo_fm_overview_widget.py --config fibsem/config/sim-iflm-configuration.yaml

Starts the Demo microscope from a simulator configuration and sends the stage to the
FM the way the app does, with `move_to_device("FM")`: on the default configuration, a
compustage, that turns the stage into its own FM orientation; on the offset-mount
simulator it travels there. The configuration decides the geometry, which is what the
tile projection depends on.

Overviews are shown but not saved unless `--output` is given, so a run against the
simulator does not leave tiles wherever it was started from.
"""

import argparse
import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow

import fibsem.config as cfg
from fibsem import utils
from fibsem.imaging.overview_session import FLUORESCENCE, OverviewSession
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

SIM_COMPUSTAGE_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=SIM_COMPUSTAGE_CONFIG,
        help="simulator configuration (default: the compustage simulator); "
        "fibsem/config/sim-iflm-configuration.yaml simulates an offset FM mount",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="write acquired overviews under DIR, one subdirectory per run",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheets.NAPARI_STYLE)

    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=args.config)
    if microscope.fm is None:
        print(f"{args.config} fits no fluorescence microscope")
        return 1
    microscope.move_to_device("FM")

    window = QMainWindow()
    tilt = np.rad2deg(microscope.get_stage_position().t)
    window.setWindowTitle(f"FM Overview · t = {tilt:.0f}°")
    widget = FMOverviewWidget(OverviewSession(microscope, FLUORESCENCE))
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        widget.set_save_directory(args.output)
        print(f"Overviews will be saved under {args.output}")
    window.setCentralWidget(widget)
    window.resize(1500, 900)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
