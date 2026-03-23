"""Standalone test for LamellaTaskImageWidget."""

import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from fibsem.applications.autolamella.structures import Experiment
from fibsem.ui.widgets.lamella_task_image_widget import LamellaTaskImageWidget


def main():
    app = QApplication(sys.argv)

    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2026-03-11-12-33/experiment.yaml"

    exp = Experiment.load(PATH)

    # Find first lamella with task history
    lamella = None
    for p in exp.positions:
        if len(p.task_history) > 0:
            lamella = p
            break

    if lamella is None:
        print("No lamella with completed tasks found.")
        return

    print(f"Showing images for: {lamella.name}")
    print(f"  Completed tasks: {[t.name for t in lamella.task_history]}")

    window = QWidget()
    window.setWindowTitle(f"Task Images - {lamella.name}")
    window.setMinimumSize(600, 800)
    window.setStyleSheet("background: #2b2d31;")

    layout = QVBoxLayout(window)
    widget = LamellaTaskImageWidget()
    layout.addWidget(widget)

    widget.set_lamella(lamella)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
