"""Standalone viewer for LamellaTaskImageWidget.

Opens the review panel for one lamella of a real experiment, without launching the
whole application. Defaults to the most recent experiment on this machine that has
any completed tasks.

    python fibsem/applications/autolamella/ui/tests/test_lamella_task_image_widget.py
    python fibsem/applications/autolamella/ui/tests/test_lamella_task_image_widget.py <experiment.yaml>
    python fibsem/applications/autolamella/ui/tests/test_lamella_task_image_widget.py --lamella 02-awake-stork

Also reports, per task, whether its images were found through the paths the run
recorded or through the filename convention -- the two routes the panel supports.
"""

import argparse
import glob
import os
import sys
from typing import List, Optional

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.task_outputs import (
    final_reference_images,
    fluorescence_images,
)
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    LamellaTaskImageWidget,
)
from fibsem.ui.stylesheets import NAPARI_STYLE

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "applications", "autolamella", "log",
)


def _experiments_with_history() -> List[str]:
    """Experiment files under the default log directory, newest first."""
    paths = glob.glob(os.path.join(LOG_DIR, "*", "experiment.yaml"))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def _resolve(path: Optional[str]) -> str:
    """Resolve a directory or file argument to an experiment.yaml, or find one."""
    if path is not None:
        if os.path.isdir(path):
            return os.path.join(path, "experiment.yaml")
        return path

    for candidate in _experiments_with_history():
        experiment = Experiment.load(candidate)
        if any(p.task_history for p in experiment.positions):
            return candidate
    raise SystemExit(
        f"No experiment with completed tasks found under {LOG_DIR}.\n"
        "Pass one explicitly: ... test_lamella_task_image_widget.py <experiment.yaml>"
    )


def _pick_lamella(experiment: Experiment, name: Optional[str]) -> Lamella:
    if name is not None:
        for position in experiment.positions:
            if position.name == name:
                return position
        raise SystemExit(
            f"No lamella named {name!r}. Available: "
            f"{[p.name for p in experiment.positions]}"
        )

    for position in experiment.positions:
        if position.task_history:
            return position
    raise SystemExit("No lamella with completed tasks in this experiment.")


def _report(lamella: Lamella) -> None:
    """Print what the panel will show, and which route found it."""
    print(f"\n{lamella.name}  ({lamella.path})")
    seen = {t.name: t for t in lamella.task_history}
    for task in seen.values():
        recorded = any(task.outputs.get(role) for role in ("final_sem", "final_fib"))
        reference = final_reference_images(lamella, task)
        fluorescence = fluorescence_images(lamella, task)
        route = "recorded" if recorded else "convention"
        print(
            f"  {task.name:<28} {route:<11} "
            f"{len(reference)} reference, {len(fluorescence)} fluorescence"
        )
        for path in reference:
            print(f"      {os.path.basename(path)}")
        for path in fluorescence:
            print(f"      {os.path.basename(path)}  [z-stack]")
        if not reference and not fluorescence:
            print("      (nothing recorded — the panel shows an empty-state row)")
    if not seen:
        print("  no completed tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", help="experiment.yaml, or its directory")
    parser.add_argument("--lamella", help="lamella name (default: first with task history)")
    args = parser.parse_args()

    path = _resolve(args.experiment)
    print(f"experiment: {path}")
    experiment = Experiment.load(path)
    lamella = _pick_lamella(experiment, args.lamella)
    _report(lamella)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(NAPARI_STYLE)

    window = QWidget()
    window.setWindowTitle(f"Task Images - {lamella.name}")
    window.setMinimumSize(600, 800)

    layout = QVBoxLayout(window)
    widget = LamellaTaskImageWidget()
    layout.addWidget(widget)
    widget.set_lamella(lamella)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
