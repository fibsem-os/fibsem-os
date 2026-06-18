"""Launch the full AutoLamella app pre-connected to the compustage sim, so the
Sample tab (holder + loader magazine) is populated without clicking through the
Connection tab.

The arctis sim configuration has is_compustage: true, which gives the stage a
SampleGridLoader (12-slot magazine) plus a 1-slot working holder — exactly what
the Sample tab needs to show both widgets.

Open the "Microscope" tab → "Sample" sub-tab to see:
- the loader magazine (top): name/describe grids inline, click a status dot to
  mark a slot available, then the login button to exchange it into the beam;
- the sample holder (bottom): the working slot, updates when a grid is loaded.

For the holder-only view (non-compustage), connect with the default config
instead — the magazine is simply absent and the holder fills the tab.

Run (PYTHONPATH so the worktree's fibsem is imported, not an installed copy):
    PYTHONPATH=$PWD python fibsem/ui/widgets/tests/launch-sample-tab.py
"""

import sys

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
    AutoLamellaSingleWindowUI,
)
from fibsem.microscopes._stage import SampleGrid

ARCTIS_SIM_CONFIG = "fibsem/config/sim-arctis-configuration.yaml"


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AutoLamellaSingleWindowUI()
    window.show()

    # connect to the compustage sim directly (skip the Connection tab clicks)
    system_widget = window.autolamella_ui.system_widget
    system_widget.microscope, system_widget.settings = utils.setup_session(
        config_path=ARCTIS_SIM_CONFIG,
    )

    # pre-load a couple of magazine slots so there's something to see
    loader = system_widget.microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="grid-aspen"))
    loader.assign_grid("Magazine-02", SampleGrid(name="grid-birch"))

    # emits connected_signal -> builds the microscope tabs, including "Sample"
    system_widget.update_ui()

    # create a throwaway experiment so the Grids tab's "Add from Loader" has
    # somewhere to import into (experiment.grids).
    import tempfile
    from fibsem.applications.autolamella.structures import Experiment

    exp = Experiment.create(path=tempfile.mkdtemp(), name="grids-demo")
    window.autolamella_ui.experiment = exp
    window.autolamella_ui.experiment_update_signal.emit()

    # jump straight to the Sample sub-tab
    inner = window.autolamella_ui.tabWidget
    if window.autolamella_ui.sample_tab is not None:
        inner.setCurrentIndex(inner.indexOf(window.autolamella_ui.sample_tab))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
