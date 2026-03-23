"""Test script for LamellaNameListWidget with real experiment data."""
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from fibsem.applications.autolamella.structures import Experiment
from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

EXPERIMENT_PATH = Path(
    "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/"
    "AutoLamella-2026-03-11-17-14/experiment.yaml"
)


def main():
    app = QApplication(sys.argv)

    experiment = Experiment.load(EXPERIMENT_PATH)

    window = QWidget()
    window.setWindowTitle("LamellaNameListWidget Test")
    window.resize(600, 400)
    layout = QVBoxLayout(window)

    widget = LamellaNameListWidget()

    # Enable toolbuttons
    widget.enable_add_button(True)
    widget.enable_defect_button(True)
    widget.enable_actions_button(True)
    widget.enable_move_to_action(True)
    widget.enable_edit_action(True)
    widget.enable_update_action(True)
    widget.enable_remove_button(True)

    widget.set_lamella(experiment.positions)

    # Connect signals
    widget.lamella_selected.connect(
        lambda lam: print(f"Selected: {lam.name if lam else None} (index={widget.selected_index})")
    )
    widget.add_requested.connect(lambda: print("Add requested"))
    widget.move_to_requested.connect(lambda lam: print(f"Move to: {lam.name}"))
    widget.edit_requested.connect(lambda lam: print(f"Edit: {lam.name}"))
    widget.update_requested.connect(lambda lam: print(f"Update: {lam.name}"))
    widget.remove_requested.connect(lambda lam: print(f"Remove: {lam.name}"))
    widget.defect_changed.connect(lambda lam: print(f"Defect changed: {lam.name} -> {lam.defect.state}"))

    layout.addWidget(widget)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
