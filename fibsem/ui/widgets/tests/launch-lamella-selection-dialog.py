"""Launch the LamellaSelectionDialog standalone — place lamella positions on a
grid overview without booting the whole AutoLamella UI.

Sets up a Demo microscope, acquires an overview image (so the canvas + the
stage<->pixel reprojection have real metadata), and a throwaway experiment with
one grid pre-seeded with a couple of lamellae. A tiny stand-in "host" commits
Accept by adding lamellae to the experiment and printing the result.

Try it:
- scroll to zoom, left-drag to pan, left-click a marker (or a list row) to select;
- right-click → Add lamella position here / Move selected position here;
- Accept → the new/moved positions are committed and printed to the console.

Run (PYTHONPATH so the worktree's fibsem is imported, not an installed copy):
    PYTHONPATH=$PWD python fibsem/ui/widgets/tests/launch-lamella-selection-dialog.py
"""

import logging
import sys
import tempfile
from copy import deepcopy

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.ui.widgets.lamella_selection_dialog import LamellaSelectionDialog


class _Host:
    """Minimal stand-in for AutoLamellaUI's lamella-creation plumbing."""

    def __init__(self, experiment):
        self.experiment = experiment

        class _Sig:
            def emit(self_inner):
                print("[host] lamella_added_signal emitted")
        self.lamella_added_signal = _Sig()

    def add_new_lamella(self, stage_position, name, microscope_state, grid_id, notify):
        state = deepcopy(microscope_state)
        state.stage_position = deepcopy(stage_position)
        self.experiment.add_new_lamella(
            state, self.experiment.task_protocol.task_config, name=name, grid_id=grid_id)
        print(f"[host] added lamella '{name}' on grid {grid_id} (notify={notify})")

    def update_lamella_combobox(self):
        pass

    def update_ui(self):
        pass


def _setup():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    image = microscope.acquire_image(
        ImageSettings(resolution=(1536, 1024), hfw=900e-6, beam_type=BeamType.ELECTRON)
    )

    experiment = Experiment.create(path=tempfile.mkdtemp(), name="overview-demo")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    grid = GridRecord(name="Grid-01")
    experiment.add_grid(grid)

    # pre-seed a couple of existing lamellae offset from the overview centre
    base = image.metadata.microscope_state.stage_position
    for dx, dy in ((150e-6, 80e-6), (-120e-6, -60e-6)):
        sp = deepcopy(base)
        sp.x += dx
        sp.y += dy
        experiment.add_new_lamella(
            deepcopy(image.metadata.microscope_state),
            experiment.task_protocol.task_config,
            grid_id=grid._id,
        )
        experiment.positions[-1].milling_pose.stage_position = sp

    return microscope, experiment, grid, image


def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # match the in-app look (the dialog normally lives inside a napari viewer)
    try:
        from napari.qt import get_stylesheet
        app.setStyleSheet(get_stylesheet("dark"))
    except Exception:
        pass

    microscope, experiment, grid, image = _setup()
    host = _Host(experiment)

    dialog = LamellaSelectionDialog(
        experiment=experiment, grid_record=grid, image=image,
        microscope=microscope, host=host)

    def _on_accept():
        print("\n[accepted] grid now has these lamellae:")
        for lam in experiment.get_lamellae_for_grid(grid):
            print(f"  - {lam.name}  @ {lam.stage_position.pretty_string}")

    dialog.accepted_positions.connect(_on_accept)
    dialog.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
