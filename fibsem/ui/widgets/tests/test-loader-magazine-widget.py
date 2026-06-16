"""Manual demo launcher for the Magazine + Holder sub-section (Phase 4 grid UI).

Shows LoaderMagazineWidget (left) and SampleHolderWidget (right) wired to a
demo CompuStage microscope (which has a loader + a single working slot).

Try it:
- Type a grid name (and description) directly in a magazine row — the slot
  loads automatically (no need to tick the box first).
- Click "Run Inventory" to rescan the magazine.
- Click the Load (tray) button on a named magazine row -> the grid is
  exchanged into the holder working slot (right), via GridTaskManager.ensure_loaded.

Run:  python fibsem/ui/widgets/tests/test-loader-magazine-widget.py
"""

import sys

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.applications.autolamella.structures import GridRecord
from fibsem.applications.autolamella.workflows.tasks.grid_manager import (
    GridExchangeError,
    GridTaskManager,
)
from fibsem.microscopes._stage import SampleGrid, _create_sample_stage
from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget
from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget


def _make_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # loader + 1 working slot
    # pre-load a couple of magazine slots so there's something to see
    loader = microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="grid-aspen"))
    loader.assign_grid("Magazine-02", SampleGrid(name="grid-birch"))
    return microscope


class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Magazine + Holder — Test")
        self.setStyleSheet("background: #262930; color: #d6d6d6;")
        self.resize(900, 560)

        self.microscope = _make_microscope()
        self._manager = GridTaskManager(self.microscope, experiment=None)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Grids — Magazine + Holder")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 6px;")
        root.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(12)

        self.magazine = LoaderMagazineWidget(microscope=self.microscope)
        self.magazine.set_microscope(self.microscope)
        columns.addWidget(self.magazine, 1)

        self.holder = SampleHolderWidget(microscope=self.microscope)
        # don't let this demo write over the shared sample-holder.yaml
        self.holder._auto_save = lambda *a, **k: None
        self.holder.set_holder(self.microscope._stage.holder)
        columns.addWidget(self.holder, 1)

        root.addLayout(columns)

        self.log_label = QLabel("Type a grid name in a row to load it, then click the tray button to load it into the beam.")
        self.log_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px 6px;")
        root.addWidget(self.log_label)

        btn_row = QHBoxLayout()
        btn_inv = QPushButton("Run Inventory")
        btn_inv.clicked.connect(self.magazine._on_run_inventory)
        btn_row.addWidget(btn_inv)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # wire the magazine signals to a log + the Load -> beam exchange.
        # the magazine slot and the holder working slot can reference the same
        # SampleGrid (once loaded), so refresh the holder view when either edits.
        self.magazine.magazine_changed.connect(self._on_magazine_changed)
        self.magazine.presence_toggled.connect(
            lambda name, loaded: self._log(f"{name}: {'loaded' if loaded else 'empty'}")
        )
        self.magazine.load_requested.connect(self._on_load_requested)
        self.magazine.unload_requested.connect(self._on_unload_requested)

    def _log(self, msg: str) -> None:
        self.log_label.setText(msg)

    def _on_magazine_changed(self) -> None:
        # keep the holder view in sync — it may show the same (now-edited) grid
        self.holder.refresh()
        self._log("magazine changed")

    def _on_load_requested(self, grid_name: str) -> None:
        try:
            self._manager.ensure_loaded(GridRecord(name=grid_name))
            self.holder.refresh()
            self.magazine.refresh_rows()  # update status dots (green = in beam)
            self._log(f"Loaded '{grid_name}' into the working slot.")
        except GridExchangeError as e:
            self._log(f"Exchange failed: {e}")

    def _on_unload_requested(self) -> None:
        loader = self.microscope._stage.loader
        unloaded = [s.loaded_grid.name for s in loader.loaded_slots]
        for slot in list(loader.loaded_slots):
            loader.unload_grid(slot.name)
        self.holder.refresh()
        self.magazine.refresh_rows()  # update status dots (white = available)
        self._log(
            f"Unloaded {unloaded} from the working slot." if unloaded
            else "Nothing loaded in the working slot."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
