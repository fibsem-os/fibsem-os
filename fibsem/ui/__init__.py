# These imports are eager: anything under `fibsem.ui` pulls in every widget named here.
# Keep that in mind before adding one -- `FibsemMinimapWidget` used to sit in this list
# and dragged AutoLamella into every module that did `from fibsem.ui import stylesheets`.
from fibsem.ui.FibsemCryoDepositionWidget import FibsemCryoDepositionWidget
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

try:
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI

    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    import logging

    logging.debug("Could not import FibsemEmbeddedDetectionWidget")
