"""The coincidence milling viewer on the simulated Arctis.

    python demos/autolamella/demo_coincidence_viewer.py
    python demos/autolamella/demo_coincidence_viewer.py <experiment-dir>

Starts the Demo microscope from the simulator configuration, which has a
fluorescence microscope. Without an experiment folder, the viewer opens on an
empty experiment in a temporary directory.
"""

import os
import sys
import tempfile

from PyQt5.QtWidgets import QApplication

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
    open_coincidence_viewer_window,
)

SIM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def main() -> int:
    app = QApplication(sys.argv)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=SIM_CONFIG)
    if len(sys.argv) > 1:
        experiment = Experiment.load(os.path.join(sys.argv[1], "experiment.yaml"))
    else:
        experiment = Experiment(path=tempfile.mkdtemp(), name="coincidence-demo")
    viewer = open_coincidence_viewer_window(microscope, experiment)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
