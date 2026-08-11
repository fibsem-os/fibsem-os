# FibsemMinimapWidget is deliberately NOT re-exported here. It depends on AutoLamella,
# and these imports are eager, so re-exporting it dragged AutoLamella into every module
# that did `from fibsem.ui import stylesheets`. Import it from its own module instead:
#     from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.FibsemCryoDepositionWidget import FibsemCryoDepositionWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

try:
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    import logging
    logging.debug("Could not import FibsemEmbeddedDetectionWidget")

