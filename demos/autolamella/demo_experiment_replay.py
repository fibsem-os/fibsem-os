"""ExperimentReplayWidget on a recorded experiment.

Needs an experiment folder: one with an ``events.jsonl``, or at least a
``logfile.log``. Pass it on the command line, or choose it when asked.

    python demos/autolamella/demo_experiment_replay.py <experiment-dir>

Read-only: the replay talks to no microscope and writes nothing to the experiment.
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog

from fibsem.applications.autolamella.tools.replay_viewer import open_replay_window


def main() -> int:
    app = QApplication(sys.argv)
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else QFileDialog.getExistingDirectory(None, "Choose an experiment to replay")
    )
    if not path:
        return 0
    widget = open_replay_window(Path(path))
    if widget is None:
        print(f"Nothing to replay in {path}: no logfile.log")
        return 1
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
