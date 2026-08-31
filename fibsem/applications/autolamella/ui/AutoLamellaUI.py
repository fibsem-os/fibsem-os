import sys
import time
import warnings

from fibsem import conversions

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTabWidget,
    QWidget,
)

from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.applications.autolamella.ui.qt_responder import QtResponder
from fibsem.applications.autolamella.ui.selected_lamella_widget import (
    SelectedLamellaWidget,
)
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    MicroscopeSettings,
)
from fibsem.ui import (
    DETECTION_AVAILABLE,
    FibsemCryoDepositionWidget,
    FibsemImageSettingsWidget,
    FibsemMovementWidget,
    FibsemSpotBurnWidget,
    FibsemSystemSetupWidget,
    MillingTaskViewerWidget,
    notification_service,
    stylesheets,
)
from fibsem.ui import utils as fui
from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.fm.widgets import FMImageViewerWidget
from fibsem.ui.qt.threading import FunctionWorker

if (
    DETECTION_AVAILABLE
):  # ml dependencies are option, so we need to check if they are available
    from fibsem.ui.FibsemEmbeddedDetectionWidget import (
        FibsemEmbeddedDetectionUI as FibsemEmbeddedDetectionWidget,
    )

from psygnal import EmissionInfo
from superqt import ensure_main_thread

# Paired with the disabled completion_summary hook in setup_hooks; FunctionHook and
# HookEvent come back from fibsem.hooks below at the same time.
# from fibsem.applications.autolamella.tools.artifacts import write_completion_summary
import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.hook_defaults import build_hook_manager
from fibsem.applications.autolamella.poses import (
    build_lamella_poses,
    sync_fluorescence_pose,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.autolamella_create_experiment_widget import (
    create_experiment_dialog,
)
from fibsem.applications.autolamella.ui.autolamella_load_experiment_widget import (
    load_experiment_dialog,
)
from fibsem.applications.autolamella.ui.autolamella_load_task_protocol_widget import (
    load_task_protocol_dialog,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import HookManager
from fibsem.ui.fm.widgets import MinimapPlotWidget
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.workflow_summary_dialog import WorkflowSummaryDialog

if TYPE_CHECKING:
    import pandas as pd

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )
    from fibsem.applications.autolamella.workflows.tasks.status import (
        WorkflowStatusEvent,
    )

# Suppress a specific upstream Napari/NumPy warning from shapes miter computation. This
# module no longer imports napari, but the minimap still loads it, and a filter is matched
# by module *name* — so this keeps working and is still worth having.
warnings.filterwarnings(
    "ignore",
    message=r"'where' used without 'out', expect unit?ialized memory in output\. If this is intentional, use out=None\.",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._shapes_utils",
)

REPORTING_AVAILABLE: bool = False
try:
    from fibsem.applications.autolamella.ui.autolamella_generate_report_widget import (
        generate_report_dialog,
    )
    from fibsem.applications.autolamella.ui.autolamella_overview_image_widget import (
        create_overview_image_widget,
    )

    REPORTING_AVAILABLE = True
except ImportError as e:
    logging.debug(
        f"Could not import generate_report from fibsem.applications.autolamella.tools.reporting: {e}"
    )

AUTOLAMELLA_CHECKPOINTS = []
try:
    from fibsem.segmentation.utils import list_available_checkpoints_v2

    AUTOLAMELLA_CHECKPOINTS = list_available_checkpoints_v2()
except ImportError as e:
    logging.debug(
        f"Could not import list_available_checkpoints from fibsem.segmentation.utils: {e}"
    )
except Exception as e:
    logging.warning(f"Could not retreive checkpoints from huggingface: {e}")


# instructions
# What to do next, shown in the window's status bar. Each names something visible
# on screen at the moment it is shown -- three of these used to name menus that do
# not exist ("Connection ->", "Experiment ->" are not menus; the File entries are
# "New Experiment" and "Load Experiment"), which is worse than saying nothing.
INSTRUCTIONS = {
    "NOT_CONNECTED": "Connect to the microscope to begin.",
    "NO_EXPERIMENT": "Create or load an experiment to begin.",
    "NO_PROTOCOL": "Load a protocol for this experiment (File \u2192 Load Protocol).",
    "NO_LAMELLA": (
        "Add a lamella position with + in the lamella list, "
        "or mark one on the Overview tab."
    ),
    "AUTOLAMELLA_READY": "Ready to run. Choose lamella and tasks in the Workflow tab.",
}


class AutoLamellaUI(QMainWindow):
    workflow_update_signal = pyqtSignal(dict)
    # The fire-and-forget third of workflow_update_signal's traffic, moving to its
    # own channel for the same reason queue_changed_signal has one: that signal's
    # handler clears WAITING_FOR_UI_UPDATE on every emission, so on the shared
    # channel merely saying something releases whoever is blocked on that flag.
    # Carries a WorkflowStatusEvent.
    workflow_status_signal = pyqtSignal(object)
    # Kept separate from workflow_update_signal on purpose: that one drives
    # handle_workflow_update, which reconfigures the interaction UI on every
    # emission. A queue edit is not a step in the task lifecycle and must not
    # disturb any of that.
    queue_changed_signal = pyqtSignal(dict)
    step_update_signal = pyqtSignal(str)  # emits human-readable step label
    _workflow_finished_signal = pyqtSignal(bool)
    experiment_update_signal = pyqtSignal()
    _hook_toast_signal = pyqtSignal(
        str, str
    )  # (message, notification_type) — thread-safe bridge for NotificationHook

    def __init__(
        self,
        parent_ui: "AutoLamellaSingleWindowUI",
    ) -> None:
        super().__init__()

        self._setup_ui()
        self.parent_widget = parent_ui

        # The Qt side of the workflow's Responder seam: workflow code is handed
        # this one-method object, never the window itself.
        self.ui_responder = QtResponder(self)

        self._protocol_lock = threading.RLock()

        self.experiment: Optional[Experiment] = None
        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        # Read here rather than taken from parent_ui: this widget is constructed
        # with parent_ui=None in tests, and the tab it gates is built in __init__.
        self._connection_chip_enabled = (
            fibsem_cfg.load_user_preferences().features.connection_chip
        )
        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.spot_burn_widget: Optional[FibsemSpotBurnWidget] = None
        self.fm_control_widget: Optional[FMControlWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskViewerWidget] = None
        self.det_widget: Optional["FibsemEmbeddedDetectionWidget"] = None

        # minimap plot widget — a floating tool window, shown on demand (was a
        # napari dock; relocated here so it no longer needs a viewer).
        self.minimap_plot_widget = MinimapPlotWidget(self)
        self.minimap_plot_widget.setWindowFlags(Qt.Tool)
        self.minimap_plot_widget.setWindowTitle("Minimap Plot")
        self.minimap_plot_widget.hide()

        # add widgets to tabs.
        #
        # The Connection tab is a gate in a bar that otherwise means "the instrument
        # needs you here now" -- it is the only tab that can never be a workflow
        # step, and it sits at position 0, the default landing spot, for something
        # done once a session. With the connection dialog on it has somewhere else
        # to be reached from, so it goes (FIB-775).
        #
        # The widget itself stays either way: it owns the connection, and everything
        # in the application follows its signals. Only the tab is conditional.
        if not self._connection_chip_enabled:
            self.tabWidget.insertTab(0, self.system_widget, "Connection")

        # Display state, not a handshake: a question is up and waiting for a
        # click. QtResponder is the only setter; the attention button, border
        # and timeline pause read it. The cross-thread flag-poll it used to be
        # -- and USER_RESPONSE and WAITING_FOR_UI_UPDATE alongside it -- is
        # gone: every workflow interaction is a typed request on its own future
        # (workflows/interaction.py).
        self.WAITING_FOR_USER_INTERACTION: bool = False
        # A run is active but nothing is executing -- today only during a
        # scheduled-start wait. Set from the worker thread, read by the border.
        self.WORKFLOW_PENDING: bool = False
        self._workflow_stop_event: threading.Event = threading.Event()
        self._task_worker_thread: Optional[FunctionWorker] = None
        self._task_manager: Optional[TaskManager] = None
        self._last_run_summary: Optional["pd.DataFrame"] = None

        # setup connections
        self.setup_connections()

    def _setup_ui(self):
        """Create all UI widgets inline (replaces generated setupUi from .ui file)."""
        self.resize(788, 1234)
        self.setAutoFillBackground(True)

        # Central widget
        self.centralwidget = QWidget(self)
        self.gridLayout = QGridLayout(self.centralwidget)

        # --- Tab widget (row 0, colspan 2) ---
        self.tabWidget = QTabWidget(self.centralwidget)

        # Experiment tab
        self.tab = QWidget()
        self.grid_layout_experiment = QGridLayout(self.tab)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_add_button(True)
        self.lamella_list.enable_defect_button(True)
        self.lamella_list.enable_actions_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_update_action(True)
        self.lamella_list.enable_remove_button(True)

        self.selected_lamella_widget = SelectedLamellaWidget()

        self.grid_layout_experiment.addWidget(self.lamella_list, 0, 0, 1, 2)
        self.grid_layout_experiment.addWidget(self.selected_lamella_widget, 1, 0, 1, 2)

        self.grid_layout_experiment.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 2, 0, 1, 2
        )

        # Add Experiment tab to tabWidget
        self.tabWidget.addTab(self.tab, "Experiment")

        self.label_workflow_information = QLabel("Workflow Information")

        # The question a running workflow is asking; hidden when it is not asking
        # one. Idle guidance lives in the window's status bar.
        self.label_instructions = QLabel("Instructions")

        self.pushButton_yes = QPushButton("Yes")
        self.pushButton_no = QPushButton("No")

        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.label_workflow_information, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.label_instructions, 3, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_yes, 4, 0)
        self.gridLayout.addWidget(self.pushButton_no, 4, 1)

        self.setCentralWidget(self.centralwidget)
        self.tabWidget.setCurrentIndex(0)

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        with self._protocol_lock:
            return (
                self.experiment.task_protocol if self.experiment is not None else None
            )

    @property
    def is_workflow_running(self) -> bool:
        return (
            self._task_worker_thread is not None and self._task_worker_thread.is_alive()
        )

    def _script_runner_is_busy(self) -> bool:
        """Whether a user script is currently driving the microscope (FIB-340).

        Reached through the parent window, which owns the Scripts menu. Absent in
        the tests and in any host that never built one, so this stays optional
        rather than asserting the attribute exists.
        """
        controller = getattr(self.parent_widget, "script_menu_controller", None)
        return bool(controller is not None and controller.runner.is_running)

    def setup_connections(self):

        # lamella controls
        self.lamella_list.add_requested.connect(
            lambda: self.add_new_lamella(stage_position=None)  # type: ignore
        )
        self.lamella_list.remove_requested.connect(self._on_lamella_remove_requested)
        self.lamella_list.move_to_requested.connect(self._on_lamella_move_to_requested)
        self.lamella_list.update_requested.connect(self._on_lamella_update_requested)
        # `defect_changed` is deliberately not wired here. The window this widget sits
        # in connects the same signal to its own handler, which saves *and* redraws the
        # rows, the cards and the name list -- so a second handler here only bought a
        # second full `experiment.save()` for one toggled icon, which is 0.8 s of frozen
        # GUI at 100 lamella (FIB-682). Every other host of this list keeps its own
        # handler, because nothing else is listening for them (FIB-564).
        self.lamella_list.lamella_selected.connect(self.update_lamella_ui)

        # system widget
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        # workflow interaction
        self.pushButton_yes.clicked.connect(self.push_interaction_button)
        self.pushButton_no.clicked.connect(self.push_interaction_button)

        # signals
        self.workflow_update_signal.connect(self.handle_workflow_update)
        self.workflow_status_signal.connect(self.handle_workflow_status)
        self._workflow_finished_signal.connect(self._workflow_finished)  # type: ignore

        # workflow info
        self.set_current_workflow_message(msg=None, show=False)
        self.label_instructions.setWordWrap(True)
        self.label_workflow_information.setWordWrap(True)

        # refresh ui
        self.update_ui()

        self.selected_lamella_widget.objective_position_changed.connect(
            self.update_lamella_objective_position
        )
        self.selected_lamella_widget.use_current_objective_requested.connect(
            self._use_current_objective_position
        )
        self.selected_lamella_widget.apply_objective_to_all_requested.connect(
            self._apply_objective_position_to_all
        )
        self.selected_lamella_widget.move_objective_requested.connect(
            self._move_objective_to_lamella_position
        )
        self.selected_lamella_widget.pose_update_requested.connect(
            self._set_current_position_as_pose
        )
        self.selected_lamella_widget.pose_move_to_requested.connect(
            self._move_to_lamella_pose
        )

    ##########

    @ensure_main_thread
    def _on_experiment_updated(self, evt: EmissionInfo) -> None:
        """Handle when positions are updated from the minimap."""
        if self.experiment is None:
            return

        if evt.signal.name not in ["inserted", "removed", "changed"]:
            # TODO: update the ui with new state
            # logging.info(f"Unhandled event: {evt.signal.name}: {evt.path}, {evt.args}")
            return

        self.update_lamella_combobox()
        self.update_ui()

    @property
    def minimap_widget(self) -> Optional[FibsemMinimapWidget]:
        if self.parent_widget is None:
            return None
        return self.parent_widget.minimap_widget

    @ensure_main_thread
    def _on_stage_position_updated(self, stage_position: FibsemStagePosition) -> None:
        """Callback for when the stage position is updated."""
        if self.minimap_widget is not None and self.minimap_widget.is_acquiring:
            return
        if self.movement_widget is not None:
            # pass the position from the signal; re-querying the microscope here races
            # with the worker thread driving the move (see TescanMicroscope socket lock)
            self.movement_widget.update_ui(stage_position=stage_position)

        self._update_minimap_data(stage_position=stage_position)

    def _disconnect_experiment_events(self) -> None:
        """Disconnect existing experiment and microscope event subscribers.

        This prevents duplicate event connections when creating/loading multiple experiments.
        """
        # Disconnect experiment events
        if self.experiment is not None:
            try:
                self.experiment.events.disconnect(self._on_experiment_updated)  # type: ignore
                logging.info("Disconnected previous experiment event subscribers.")
            except Exception as e:
                logging.debug(f"Could not disconnect experiment events: {e}")

        # Disconnect microscope stage position events
        if self.microscope is not None:
            try:
                self.microscope.stage_position_changed.disconnect(
                    self._on_stage_position_updated
                )
                logging.info(
                    "Disconnected previous microscope stage position event subscribers."
                )
            except Exception as e:
                logging.debug(f"Could not disconnect microscope stage events: {e}")

    def _setup_experiment_connections(self) -> None:
        """Setup connections and metadata for the loaded/created experiment.

        This handles:
        - Updating settings image path
        - Connecting event subscribers
        - Registering metadata
        - Updating UI components
        """
        if self.experiment is None:
            logging.warning("Cannot setup experiment connections: experiment is None")
            return

        # Update settings path
        if self.settings is not None:
            self.settings.image.path = self.experiment.path

        # Connect position updates
        self.experiment.events.connect(self._on_experiment_updated)  # type: ignore
        if self.microscope is not None:
            self.microscope.stage_position_changed.connect(
                self._on_stage_position_updated
            )

        # Register metadata
        if self.microscope is not None:
            self.experiment.register_metadata(self.microscope)

        # Update UI
        self.update_lamella_combobox()
        self.update_ui()

        # set the experiment tab as active
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab))

    def create_experiment(self) -> None:
        """Create a new experiment using the experiment creation dialog."""
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        # Open the experiment creation dialog
        experiment = create_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            notification_service.show_toast("Experiment creation cancelled.", "info")
            return

        self._adopt_experiment(experiment)

    def load_experiment(self) -> None:
        """Load an existing experiment using the experiment loading dialog."""
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        # Open the experiment loading dialog
        experiment = load_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            notification_service.show_toast("Experiment loading cancelled.", "info")
            return

        self._adopt_experiment(experiment)

    def quickstart(self, load_experiment: bool = False) -> None:
        """Connect to the microscope without waiting to be clicked (``--quickstart``).

        The developer shortcut past the two clicks that start every session: the same
        calls the Connection tab and the load dialog make, with the default microscope
        configuration and the most recent experiment assumed.

        Every failure here is reported and swallowed rather than raised. This runs
        unattended on the way up, and an unreachable microscope or an experiment that
        has moved should still leave the ordinary window -- the one where the user can
        pick a different configuration -- rather than a half-started application.
        """
        if self.system_widget.microscope is None:
            try:
                self.system_widget.connect_to_microscope()
            except Exception as e:
                logging.warning(f"Quickstart: unable to connect to the microscope: {e}")
                notification_service.show_toast(
                    f"Quickstart: could not connect to the microscope: {e}", "error"
                )
                return

        # connect_to_microscope reports its own failures and returns without a
        # microscope (an unselectable configuration, say). Nothing below works
        # without one, so stop here rather than reporting a second failure.
        if self.system_widget.microscope is None:
            return

        if load_experiment:
            self.quickload_experiment()

    def quickload_experiment(self) -> None:
        """Reopen the most recent experiment, skipping the load dialog (``--quickload``).

        Requires a connected microscope, as the dialog path does: the tabs that adopt
        an experiment are built at connection time.
        """
        experiment_path = fibsem_cfg.get_last_experiment_file()

        if experiment_path is None:
            msg = "Quickload: no recent experiment to load."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        try:
            experiment = Experiment.load(Path(experiment_path))
        except Exception as e:
            logging.warning(f"Quickload: unable to load {experiment_path}: {e}")
            notification_service.show_toast(
                f"Quickload: could not load experiment: {e}", "error"
            )
            return

        # The same bar the load dialog sets. An experiment with no protocol has
        # nothing to run, and adopting one here would only look loaded.
        if experiment.task_protocol is None:
            msg = f"Quickload: {experiment.name} has no protocol; not loaded."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        self._adopt_experiment(experiment)
        logging.info(
            f"Quickload: loaded experiment {experiment.name} from {experiment_path}"
        )

    def _adopt_experiment(self, experiment: Experiment) -> None:
        """Make ``experiment`` the current one, and point logging at its logfile.

        The single place the app takes ownership of an experiment, whether it was
        just created or loaded from disk. Logging is configured here rather than in
        Experiment.load/create because those are also called to *read* an
        experiment -- the load dialog calls load() on every single click to preview
        a recent entry, which previously repointed the app's root logger at each one
        in turn, and closed the previous handler, even when the dialog was then
        cancelled. See FIB-421.
        """
        # Disconnect existing event subscribers if there's an existing experiment
        self._disconnect_experiment_events()

        # Assign the experiment
        self.experiment = experiment

        experiment.configure_logging()
        logging.info(f"Logging to experiment {experiment.name} at {experiment.path}")

        # Setup experiment connections and update UI
        self._setup_experiment_connections()

        self.experiment_update_signal.emit()

    ##################################################################

    # TODO: create a dialog to get the user to connect to microscope and create load experiment before continuing
    # then remove the system widget entirely... you will always be connected once you start
    def connect_to_microscope(self):
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        if self.experiment is not None:
            self.settings.image.path = self.experiment.path
        self.update_microscope_ui()
        self.update_ui()
        if self.experiment is not None:
            self._disconnect_experiment_events()
            self._setup_experiment_connections()

    def disconnect_from_microscope(self):
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()

    def update_microscope_ui(self):
        """Update the ui based on the current state of the microscope."""

        if self.microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,  # type: ignore
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                parent=self,
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.milling_task_config_widget = MillingTaskViewerWidget(
                microscope=self.microscope,
                image_widget=self.image_widget,
                parent=self,
            )
            self.tabWidget.addTab(self.milling_task_config_widget, "Milling")

            if self.microscope.fm is not None:
                self.fm_control_widget = FMControlWidget(
                    microscope=self.microscope, parent=self
                )
                if self.settings is not None and self.settings.fm is not None:
                    self.fm_control_widget._apply_fluorescence_configuration(
                        self.settings.fm
                    )
                self.tabWidget.addTab(self.fm_control_widget, "Fluorescence")

            # add the detection widget if ml dependencies are available
            if DETECTION_AVAILABLE:
                self.det_widget = FibsemEmbeddedDetectionWidget(parent=self)
                self.tabWidget.addTab(self.det_widget, "Detection")
                self.tabWidget.setTabVisible(
                    self.tabWidget.indexOf(self.det_widget), False
                )

            # spot burn widget (optional)
            self.spot_burn_widget = FibsemSpotBurnWidget(parent=self)
            self.tabWidget.addTab(self.spot_burn_widget, "Spot Burn")
            self.tabWidget.setTabVisible(
                self.tabWidget.indexOf(self.spot_burn_widget), False
            )

            try:
                from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

                if isinstance(self.microscope, OdemisThermoMicroscope):
                    logging.info(
                        "OdemisThermoMicroscope detected, enabling Odemis specific features."
                    )

            except Exception as e:
                logging.debug(f"OdemisThermoMicroscope not available: {e}")

            self.image_widget.acquisition_progress_signal.connect(
                self.handle_acquisition_update
            )
        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.fm_control_widget is not None:
                # deleteLater fires neither closeEvent nor close_widget, so tear
                # down the FM widget's external signal connections explicitly
                self.fm_control_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.fm_control_widget))
                self.fm_control_widget.deleteLater()
                self.fm_control_widget = None
            if self.det_widget is not None:
                self.det_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.det_widget))
                self.det_widget.deleteLater()
                self.det_widget = None
            if self.spot_burn_widget is not None:
                self.spot_burn_widget.disconnect_signals()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.spot_burn_widget))
                self.spot_burn_widget.deleteLater()
                self.spot_burn_widget = None
            if self.milling_task_config_widget is not None:
                self.tabWidget.removeTab(
                    self.tabWidget.indexOf(self.milling_task_config_widget)
                )
                self.milling_task_config_widget.deleteLater()
                self.milling_task_config_widget = None
            if self.movement_widget is not None:
                self.movement_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.movement_widget))
                self.movement_widget.deleteLater()
                self.movement_widget = None
            if self.image_widget is not None:
                self.image_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.image_widget))
                self.image_widget.acquisition_progress_signal.disconnect(
                    self.handle_acquisition_update
                )
                self.image_widget.deleteLater()
                self.image_widget = None

    def import_fm_configuration(self) -> None:
        """Load a fluorescence microscope configuration via the control widget."""
        if self.fm_control_widget is None:
            msg = "Fluorescence control not available. Connect to an FM-enabled microscope first."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        try:
            self.fm_control_widget.import_fm_configuration()
        except Exception:
            logging.exception("Failed to load FM configuration from AutoLamella UI.")
            notification_service.show_toast(
                "Failed to load FM configuration. Check logs for details.", "error"
            )

    def export_fm_configuration(self) -> None:
        """Save the current fluorescence microscope configuration via the control widget."""
        if self.fm_control_widget is None:
            msg = "Fluorescence control not available. Connect to an FM-enabled microscope first."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        try:
            self.fm_control_widget.export_fm_configuration()
        except Exception:
            logging.exception("Failed to save FM configuration from AutoLamella UI.")
            notification_service.show_toast(
                "Failed to save FM configuration. Check logs for details.", "error"
            )

    #### REPORT GENERATION
    def action_generate_report(self) -> None:
        """Generate a pdf report of the experiment."""
        if self.experiment is None:
            return

        generate_report_dialog(self.experiment, parent=self)
        return

    def action_generate_overview_plot(self) -> None:
        """Generate an plot with the lamella position on an overview image."""
        if self.experiment is None:
            return

        if not REPORTING_AVAILABLE:
            notification_service.show_toast(
                "Reporting tools are not available.", "warning"
            )
            return

        dialog = create_overview_image_widget(experiment=self.experiment, parent=self)
        dialog.exec_()

        return

    #### PROTOCOL EDITOR

    def open_information_dialog(self) -> None:
        # No connection guard: checking which version you are running is exactly
        # what you want to do *before* connecting. The dialog drops the
        # microscope section when there is nothing to report.
        fui.open_information_dialog(self.microscope, self, application="AutoLamella")

    def _open_experiment_directory(self) -> None:
        """Open the experiment directory in the system file explorer."""
        if self.experiment is None or self.experiment.path is None:
            notification_service.show_toast(
                "Please load an experiment first... [No Experiment Loaded]", "warning"
            )
            return

        experiment_path = os.fspath(self.experiment.path)
        if not os.path.isdir(experiment_path):
            notification_service.show_toast(
                f"Experiment directory not found: {experiment_path}", "error"
            )
            return

        if not fui.open_path_in_file_explorer(experiment_path):
            notification_service.show_toast(
                "Failed to open experiment directory.", "error"
            )

    def export_targeting_ml_data(self) -> None:
        """Export the experiment's lamella targets as targeting ML training data.

        One sample per lamella: the final FIB reference image from its Select Milling
        Position task, labelled with the operator's point of interest. Writes to
        <experiment>/targeting-export. See
        fibsem.applications.autolamella.tools.ml_export for the layout.
        """
        from fibsem.applications.autolamella.tools import ml_export

        if self.experiment is None:
            notification_service.show_toast(
                "Please load an experiment first... [No Experiment Loaded]", "warning"
            )
            return

        # no directory prompt: the destination is a subfolder of the experiment that
        # does not exist yet, which an existing-directory dialog cannot select. The
        # folder is opened afterwards so the location is still discoverable.
        output_path = ml_export.default_output_path(self.experiment)

        try:
            summary = ml_export.export_experiment(self.experiment, output_path)
        except Exception as e:
            logging.error(f"Failed to export targeting ML data: {e}", exc_info=True)
            notification_service.show_toast(f"Export failed: {e}", "error")
            return

        if summary.n_samples == 0:
            reason = summary.skipped[0] if summary.skipped else "nothing to export"
            notification_service.show_toast(f"Nothing exported: {reason}", "warning")
            return

        message = f"Exported {summary.n_samples} lamella target(s)."
        if summary.skipped:
            message += f" Skipped {len(summary.skipped)}."
        notification_service.show_toast(message, "success")
        fui.open_path_in_file_explorer(output_path)

    #### FLUORESCENCE IMAGE VIEWER

    def _open_fm_image_viewer(self):
        """Open the FM Image Viewer as a standalone window."""
        if self.experiment is None:
            notification_service.show_toast(
                "Please load an experiment first... [No Experiment Loaded]", "warning"
            )
            return

        experiment_path = str(self.experiment.path) if self.experiment.path else None
        # Parented to None so it gets its own taskbar entry and native minimise, like the
        # coincidence viewer. That means nothing else owns it, so the reference here is
        # what keeps it alive — drop it and Python collects the window mid-session.
        self._fm_image_viewer_window = FMImageViewerWidget(
            start_directory=experiment_path
        )
        self._fm_image_viewer_window.resize(1180, 700)
        self._fm_image_viewer_window.show()
        self._fm_image_viewer_window.activateWindow()

    def _open_coincidence_milling_viewer(self):
        """Open FluorescenceCoincidenceViewerWidget as a standalone dialog."""
        if self.microscope is None or self.experiment is None:
            notification_service.show_toast(
                "Please connect a microscope and load an experiment first.", "warning"
            )
            return
        if self.microscope.fm is None:
            notification_service.show_toast(
                "Coincidence milling requires a fluorescence microscope.", "warning"
            )
            return
        from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
            open_coincidence_viewer_window,
        )

        # seed the viewer's FM tab from the live main-UI FM configuration
        fm_config = None
        if self.fm_control_widget is not None:
            try:
                fm_config = self.fm_control_widget._build_fluorescence_configuration()
            except Exception as e:
                logging.warning(f"Could not read current FM configuration: {e}")

        self._coincidence_viewer_window = open_coincidence_viewer_window(
            microscope=self.microscope,
            experiment=self.experiment,
            parent=self,
            fm_config=fm_config,
        )

    #### MINIMAP

    def _update_minimap_data(
        self,
        stage_position: Optional[FibsemStagePosition] = None,
        selected_name: Optional[str] = None,
    ) -> None:
        if self.microscope is None:
            return
        if self.experiment is None:
            return

        if self.minimap_plot_widget is None:
            return

        if not self.minimap_plot_widget.isVisible():
            return

        try:
            image: Optional[FibsemImage] = None
            if self.minimap_plot_widget.image is None:
                ms = self.microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
                image = FibsemImage.generate_blank_image(
                    resolution=(2048, 2048), hfw=4000e-6
                )
                image.metadata.microscope_state = ms  # type: ignore
                image.metadata.system_info = self.microscope.system.info  # type: ignore
                image.metadata.hardware_geometry = self.microscope.hardware_geometry()  # type: ignore
                self.minimap_plot_widget.image = image

            beam_type = self.minimap_plot_widget.image.metadata.beam_type  # type: ignore
            fov = self.microscope.get_field_of_view(beam_type=beam_type)

            # Set the data (delay redraw until all data updated...)
            if selected_name is not None:
                self.minimap_plot_widget.selected_name = selected_name
            self.minimap_plot_widget.lamella_positions = (
                self.experiment.get_milling_positions()
            )
            if self.minimap_plot_widget.grid_positions is None:
                self.minimap_plot_widget.grid_positions = [
                    s.position for s in self.microscope._stage.holder.slots.values()
                ]
            self.minimap_plot_widget.fov_width = fov
            if stage_position is not None:
                self.minimap_plot_widget.set_current_position(stage_position)
            else:
                self.minimap_plot_widget.update_minimap()
            if image is not None:
                self.minimap_plot_widget.reset_zoom()
        except Exception as e:
            logging.warning(f"Failed to update minimap data: {e}")

    #### TASK WORKFLOW

    def _start_run_workflow_thread(
        self, selected_tasks: List[str], selected_lamella: List[str]
    ) -> None:
        """Start the workflow thread with the selected tasks and lamella, and update the UI accordingly."""

        # A user script may be driving the microscope right now. Scripts are already
        # blocked while a workflow runs; this is the other direction, so the two
        # cannot end up moving the stage at the same time (FIB-340).
        if self._script_runner_is_busy():
            msg = "A microscope script is running. Stop it before starting a workflow."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        # clear milling task config
        self.milling_task_config_widget.clear()  # type: ignore

        # Start acquisition thread
        self._task_worker_thread = FunctionWorker(
            self._run_tasks_worker, selected_tasks, selected_lamella
        )
        self._task_worker_thread.start()

    def _run_tasks_worker(
        self, task_names: List[str], lamella_names: Optional[List[str]] = None
    ) -> None:
        """Worker thread for task worker."""
        try:
            self._workflow_stop_event.clear()
            if self.microscope is None or self.experiment is None:
                logging.error("No microscope or experiment loaded.")
                return

            # turn beams on if required
            if not self.microscope.is_on(BeamType.ELECTRON):
                self.microscope.turn_on(BeamType.ELECTRON)
            if not self.microscope.is_on(BeamType.ION):
                self.microscope.turn_on(BeamType.ION)

            logging.info(f"Starting tasks: {task_names}, for lamella: {lamella_names}")
            self._task_manager = TaskManager(
                microscope=self.microscope,
                experiment=self.experiment,
                parent_ui=self,
                hook_manager=self.setup_hooks(),
            )
            # Honor a stop requested before the manager existed (e.g. clicked
            # during beam-on, while _task_manager was still None).
            if self._workflow_stop_event.is_set():
                self._task_manager.stop()
            self._task_manager.run(
                task_names=task_names, required_lamella=lamella_names
            )
        except Exception as e:
            logging.error(f"Error during running tasks: {e}")

        finally:
            cancelled = self._task_manager is not None and self._task_manager.is_stopped
            # capture the per-run summary before the manager is torn down
            if self._task_manager is not None:
                try:
                    self._last_run_summary = (
                        self._task_manager.build_run_summary_dataframe()
                    )
                except Exception as e:
                    logging.warning(f"Failed to build workflow run summary: {e}")
                    self._last_run_summary = None
            self._task_manager = None
            self._task_worker_thread = None
            self._workflow_finished_signal.emit(cancelled)  # type: ignore

    def stop_task_workflow(self):
        if not self.is_workflow_running:
            return
        self._stop_workflow_thread()

    def setup_hooks(self) -> HookManager:
        """Build the HookManager for one workflow run.

        Called per run from the workflow worker, not once at startup, so the hook set is
        rebuilt each time — which is what will let a user's saved configuration take
        effect on the next run without a restart, and means the config dialog can never
        be editing a manager a running workflow is holding. See FIB-497.

        The hook set comes from the user's saved configuration when there is one, and
        from hook_defaults.default_hooks() when there is not. Re-read here rather than
        cached at startup, so editing user-preferences.yaml by hand takes effect on the
        next run of a workflow instead of needing a restart.

        Code-registered hooks go after: a configuration load replaces the *defaults*,
        and must never be able to remove a hook that only exists in Python.
        """
        preferences = fibsem_cfg.load_user_preferences()
        manager = build_hook_manager(preferences.hooks)

        # Deliberately not registered yet. The trigger is proven end to end and the
        # writer is tested, but completion-summary.json is a placeholder for the real
        # artifacts (the PDF and the overview PNG), and turning it on here would start
        # writing a throwaway file into every user's lamella directories. Re-enable when
        # the artifact is the one people actually want -- see FIB-461. Three imports at
        # the top of this file come back with it: write_completion_summary, and
        # FunctionHook + HookEvent from fibsem.hooks.
        #
        # Per lamella, not per experiment: a lamella is what gets delivered, and an
        # experiment with one abandoned lamella never reaches completion at all, so
        # hanging the artifact off the experiment would mean it was almost never
        # written.
        #
        # A FunctionHook rather than a template-driven one because this needs the
        # experiment itself, which the context deliberately does not carry -- the path
        # is derivable from the record, and in-process hooks can reach the record. A
        # webhook could not, which is the distinction.
        #
        # HookManager.fire contains what this raises, so a summary that cannot be
        # written is logged and the run carries on. FIB-461 asks for exactly that, and
        # it comes for free rather than needing a second guard here.
        #
        # manager.register(
        #     FunctionHook(
        #         name="completion_summary",
        #         events=[HookEvent.ITEM_COMPLETED],
        #         callback=lambda ctx: write_completion_summary(self.experiment, ctx),
        #     )
        # )
        # Signals are thread-safe to emit; hooks fire on the task worker thread.
        manager.set_notifier(self._hook_toast_signal.emit)
        return manager

    #### UI UPDATES

    def update_ui(self):
        """Update the ui based on the current state of the application."""

        if self.is_workflow_running:
            self.selected_lamella_widget.setEnabled(False)

            return

        # state flags
        is_experiment_loaded = bool(self.experiment is not None)
        is_microscope_connected = bool(self.microscope is not None)
        is_protocol_loaded = (
            bool(self.settings is not None) and self.protocol is not None
        )
        has_lamella = bool(self.experiment.positions) if is_experiment_loaded else False
        is_experiment_ready = is_experiment_loaded and is_protocol_loaded

        # force order: connect -> experiment -> protocol
        self.tabWidget.setTabVisible(
            self.tabWidget.indexOf(self.tab), is_microscope_connected
        )
        if self.det_widget is not None:
            idx = self.tabWidget.indexOf(self.det_widget)
            self.tabWidget.setTabVisible(idx, False)  # hide detection tab for now

        if is_experiment_loaded and self.experiment is not None:
            self.lamella_list.setEnabled(has_lamella)

        # buttons
        self.lamella_list.setEnabled(is_experiment_ready)
        self.selected_lamella_widget.setEnabled(is_experiment_ready)

        # clear the panel when no lamella is selected; populated by update_lamella_ui otherwise
        if not has_lamella:
            self.selected_lamella_widget.set_lamella(None)

        # disable lamella controls while workflow is running
        self.selected_lamella_widget.setEnabled(not self.is_workflow_running)

        # Current Lamella Status
        if has_lamella and self.experiment is not None:
            self.update_lamella_ui()

        if self.is_workflow_running:
            return

        # Nothing to say while idle. The same guidance is in the window's status
        # bar, which is where it belongs -- shown in both places it read as two
        # different messages that happened to agree. This label is for the question
        # a running workflow is asking, which the Yes/No buttons below it answer.
        self.set_instructions_msg("")

    def _on_workflow_config_changed(self, wcfg: AutoLamellaWorkflowConfig):
        if self.experiment is None or self.experiment.task_protocol is None:
            return
        self.experiment.task_protocol.workflow_config = wcfg
        self.experiment.save()
        self.experiment.save_protocol()

        self.update_ui()

    def _on_workflow_options_changed(self, options: AutoLamellaWorkflowOptions):
        if self.experiment is None or self.experiment.task_protocol is None:
            return
        self.experiment.task_protocol.options = options
        self.experiment.save()
        self.experiment.save_protocol()

    def update_lamella_combobox(self, latest: bool = False):
        if self.experiment is None:
            return
        if self.is_workflow_running:
            return

        # detail lamella list
        preferred = (
            self.experiment.positions[-1].name
            if latest and self.experiment.positions
            else ""
        )
        self.lamella_list.set_lamella(
            self.experiment.positions, preferred_name=preferred
        )

    def update_lamella_ui(self, _lamella=None):
        # set the info for the current selected lamella
        if self.experiment is None or self.experiment.positions == []:
            return

        if self.protocol is None:
            return

        if self.is_workflow_running:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            return

        lamella: Lamella = self.experiment.positions[idx]
        logging.info(f"Updating Lamella UI for {lamella.status_info}")

        # refresh objective position + pose display for the selected lamella
        self.selected_lamella_widget.set_lamella(lamella)

        self._update_minimap_data(selected_name=lamella.name)

    def set_spot_burn_widget_active(self, active: bool = True) -> None:
        """Set the spot burn widget active (sets the tab visible, activate point layer)."""
        if self.spot_burn_widget is None:
            return

        idx = self.tabWidget.indexOf(self.spot_burn_widget)
        self.tabWidget.setTabVisible(idx, active)
        if active:
            self.tabWidget.setCurrentIndex(idx)
            self.spot_burn_widget.set_active()
        else:
            self.spot_burn_widget.set_inactive()

    ##### LAMELLA CONTROLS

    def move_to_lamella_position(self):
        """Move the stage to the position of the selected lamella."""
        if self.experiment is None or self.experiment.positions == []:
            return
        if self.movement_widget is None:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            return
        lamella: Lamella = self.experiment.positions[idx]
        stage_position = lamella.milling_pose.stage_position

        # confirmation dialog
        ret = QMessageBox.question(
            self,
            "Move to Lamella Position",
            f"Move to position of Lamella {lamella.name}?\n{stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        logging.info(f"Moving to position of {lamella.name}.")
        self.movement_widget.move_to_position(stage_position)

    def _add_lamella_from_odemis(self):
        if self.experiment is None:
            return

        filename = fui.open_existing_directory_dialog(
            msg="Select Odemis Project Directory",
            path=str(self.experiment.path),
            parent=self,
        )
        if filename == "":
            return

        from fibsem.applications.autolamella.compat.odemis import (
            _add_features_from_odemis,
        )

        stage_positions = _add_features_from_odemis(filename)

        for pos in stage_positions:
            self.add_new_lamella(pos)

    def add_new_lamella(
        self,
        stage_position: Optional[FibsemStagePosition] = None,
        name: Optional[str] = None,
        objective_position: Optional[float] = None,
        marked_at: Optional[str] = None,
    ) -> Lamella:
        """Add a lamella to the experiment.

        Args:
            stage_position: Where the lamella is, in any orientation -- which one is
                read off the position itself, so a position picked on the fluorescence
                side is taken as the fluorescence pose rather than as somewhere to mill.
                If None, the current stage position is used.
            name: The name of the lamella. If None, a default name will be generated.
            objective_position: The objective position of the lamella. If None, the 'focused' objective position is used.
            marked_at: The orientation *stage_position* is in, for a caller that knows.
                Left alone it is read off the position, which is right on a compustage
                and cannot be on an offset mount -- see `build_lamella_poses`.
        Returns:
            lamella: The created lamella.
        """
        if self.experiment is None:
            raise ValueError("No experiment loaded. Please load an experiment first.")
        if self.protocol is None:
            raise ValueError("No protocol loaded. Please load a protocol first.")
        if self.microscope is None:
            raise ValueError(
                "No microscope connected. Please connect a microscope first."
            )

        poses = build_lamella_poses(
            microscope=self.microscope,
            position=stage_position,
            objective_position=objective_position,
            marked_at=marked_at,
        )

        # create the lamella, with both poses already on it -- see
        # `Experiment.add_new_lamella`. Assigning the fluorescence pose after the append
        # left every listener that redraws on `inserted` deciding the new lamella had
        # none, which is how each newly marked lamella went missing from the FM overview.
        self.experiment.add_new_lamella(
            microscope_state=poses.milling,
            task_config=self.experiment.task_protocol.task_config,
            name=name,
            fluorescence_pose=poses.fluorescence,
        )
        lamella = self.experiment.positions[-1]

        # derive the milling angle from the milling-pose stage tilt
        lamella.update_milling_angle(self.microscope)

        self.experiment.save()
        self.update_lamella_combobox(latest=True)
        self.update_ui()

        return lamella

    def _on_lamella_move_to_requested(self, lamella):
        """Handle move-to request from the list row's actions menu."""
        self.lamella_list.select(lamella.name)
        self.move_to_lamella_position()

    def _on_lamella_update_requested(self, lamella):
        """Handle update-position request from the list row's actions menu."""
        self.lamella_list.select(lamella.name)
        self.update_lamella_position_ui()

    def _on_lamella_remove_requested(self, lamella):
        """Handle removal of a lamella via the list row's remove button.

        Confirmation is already handled by the row widget.
        """
        if self.experiment is None:
            return
        try:
            self.experiment.positions.remove(lamella)
        except ValueError:
            return
        self.experiment.save()
        logging.debug("Lamella removed from experiment")
        self.update_lamella_combobox(latest=True)
        self.update_ui()

    def delete_lamella_ui(self):
        """Handle the removal of a lamella from the experiment (legacy path)."""

        idx = self.lamella_list.selected_index
        if idx == -1:
            logging.warning("No lamella is selected, cannot remove.")
            return

        if self.experiment is None or self.experiment.positions == []:
            logging.warning("No lamella in the experiment, cannot remove.")
            return

        pos = self.experiment.positions[idx]
        ret = fui.message_box_ui(
            title="Remove Lamella",
            text=f"Are you sure you want to remove Lamella {pos.name}?",
            parent=self,
        )
        if ret is False:
            logging.debug("User cancelled lamella removal.")
            return

        # TODO: also remove data from disk

        # remove the lamella
        self.experiment.positions.pop(idx)
        self.experiment.save()

        logging.debug("Lamella removed from experiment")
        self.update_lamella_combobox(latest=True)
        self.update_ui()

    def update_lamella_position_ui(self):
        """Update the stage position of the selected lamella to the current stage position."""

        if self.microscope is None:
            return
        if self.protocol is None:
            return
        if self.experiment is None or self.experiment.positions == []:
            return

        # toggle between saving position and marking as ready
        idx = self.lamella_list.selected_index
        if idx == -1:
            logging.warning("No lamella is selected, cannot save.")
            return

        lamella: Lamella = self.experiment.positions[idx]
        current_position = self.microscope.get_stage_position()

        # message box to confirm
        ret = QMessageBox.question(
            self,
            "Save Position Confirmation",
            f"Save new position for Lamella {lamella.name} position?\n\n"
            f"New Stage Position: {current_position.pretty}\n"
            f"Existing Stage Position: {lamella.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        lamella.milling_pose = deepcopy(self.microscope.get_microscope_state())

        # keep the milling angle consistent with the updated milling pose
        lamella.update_milling_angle(self.microscope)
        # ...and the fluorescence pose, which describes the same piece of sample from
        # the other side. Left behind, it would go on naming where this lamella used to
        # be -- and nothing about a stale pose looks wrong.
        sync_fluorescence_pose(self.microscope, lamella)

        self.update_lamella_combobox()
        self.update_ui()
        self.experiment.save()
        self.experiment.positions.events.changed.emit()

    def _set_current_position_as_pose(self, pose_name: str):
        """Set the current stage position as the given pose for the current lamella."""

        if self.microscope is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        if self.experiment is None or self.experiment.positions == []:
            notification_service.show_toast("No lamella available.", "warning")
            return
        idx = self.lamella_list.selected_index
        if idx == -1:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        lamella: Lamella = self.experiment.positions[idx]
        if pose_name == "":
            notification_service.show_toast("No pose selected.", "warning")
            return
        state = self.microscope.get_microscope_state()

        if state is None or state.stage_position is None:
            notification_service.show_toast(
                "Failed to get microscope state.", "warning"
            )
            return

        # confirmation dialog
        ret = QMessageBox.question(
            self,
            "Set Pose Confirmation",
            f"Set current position as pose '{pose_name}' for {lamella.name}?\n{state.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        # preserve the configured objective (focus) position of the existing pose:
        # get_microscope_state() does not capture the objective position, so replacing
        # the pose outright would wipe the fluorescence pose's focus setting.
        existing_pose = lamella.poses.get(pose_name)
        if existing_pose is not None and existing_pose.objective_position is not None:
            state.objective_position = existing_pose.objective_position

        lamella.poses[pose_name] = state

        # Replacing the milling pose moves the lamella, so what is derived from it has to
        # follow: the milling angle, and the fluorescence pose, which describes the same
        # piece of sample from the other side. Left behind, that pose would go on naming
        # where this lamella used to be -- and nothing about a stale pose looks wrong.
        if pose_name == "MILLING":
            lamella.update_milling_angle(self.microscope)
            if sync_fluorescence_pose(self.microscope, lamella):
                self.selected_lamella_widget.refresh_pose(
                    "FLUORESCENCE", lamella.fluorescence_pose.stage_position.pretty
                )

        self.experiment.save()
        self.selected_lamella_widget.refresh_pose(
            pose_name, state.stage_position.pretty
        )
        # The FM overview canvas draws these positions itself rather than reading them
        # back, so a pose that moved here is one it only hears about by being told.
        self.experiment.positions.events.changed.emit()
        notification_service.show_toast(
            f"Set current position as pose '{pose_name}' for {lamella.name}.", "info"
        )

    def _move_to_lamella_pose(self, pose_name: str):
        """Move the stage to the given pose for the current lamella."""

        if self.microscope is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        if self.experiment is None or self.experiment.positions == []:
            notification_service.show_toast("No lamella available.", "warning")
            return
        if self.movement_widget is None:
            notification_service.show_toast("No movement widget available", "warning")
            return
        idx = self.lamella_list.selected_index
        if idx == -1:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        lamella: Lamella = self.experiment.positions[idx]
        if pose_name == "":
            notification_service.show_toast("No pose selected.", "warning")
            return
        if pose_name not in lamella.poses:
            notification_service.show_toast(
                f"Pose '{pose_name}' not found for {lamella.name}.", "warning"
            )
            return
        pose = lamella.poses[pose_name]
        if pose.stage_position is None:
            notification_service.show_toast(
                f"Pose '{pose_name}' has no stage position.", "warning"
            )
            return

        # confirmation dialog
        ret = QMessageBox.question(
            self,
            "Move to Pose Confirmation",
            f"Move to pose '{pose_name}' for {lamella.name}?\n{pose.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        logging.info(f"Moving to pose '{pose_name}' for {lamella.name}.")
        self.movement_widget.move_to_position(pose.stage_position)
        notification_service.show_toast(
            f"Moved to pose '{pose_name}' for {lamella.name}.", "info"
        )

    def _use_current_objective_position(self):
        """Read the current FM objective position and apply it to the selected lamella."""
        if self.microscope is None or self.microscope.fm is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self.get_selected_lamella()
        if lamella is None or lamella.fluorescence_pose is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        obj = self.microscope.fm.objective
        if obj.state == "Inserted":
            value_m = obj.position
        else:
            value_m = obj.focus_position
        if value_m is None:
            notification_service.show_toast(
                "Objective position unavailable.", "warning"
            )
            return
        lamella.fluorescence_pose.objective_position = value_m
        self.experiment.save()
        # full refresh so the objective value shows and "Apply to All" re-enables
        self.selected_lamella_widget.set_lamella(lamella)
        notification_service.show_toast(
            f"Set objective position to {value_m * METRE_TO_MICRON:.1f} µm for {lamella.name}.",
            "info",
        )

    def _move_objective_to_lamella_position(self):
        """Move the FM objective to the selected lamella's stored objective position.

        Independent of the stage move-to: this only drives the objective.
        """
        if self.microscope is None or self.microscope.fm is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self.get_selected_lamella()
        if lamella is None or lamella.fluorescence_pose is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        objective_position = lamella.fluorescence_pose.objective_position
        if objective_position is None:
            notification_service.show_toast(
                f"{lamella.name} has no stored objective position.", "warning"
            )
            return
        obj = self.microscope.fm.objective
        if obj.state != "Inserted":
            notification_service.show_toast(
                "Insert the objective before moving to a stored position.", "warning"
            )
            return

        # confirmation dialog
        ret = QMessageBox.question(
            self,
            "Move Objective",
            f"Move objective to {objective_position * METRE_TO_MICRON:.1f} µm "
            f"for {lamella.name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        try:
            logging.info(
                f"Moving objective to {objective_position * METRE_TO_MICRON:.1f} µm "
                f"for {lamella.name}."
            )
            obj.move_absolute(objective_position)
            notification_service.show_toast(
                f"Moved objective to {objective_position * METRE_TO_MICRON:.1f} µm "
                f"for {lamella.name}.",
                "info",
            )
        except Exception as e:
            logging.error(f"Failed to move objective: {e}", exc_info=e)
            notification_service.show_toast(f"Failed to move objective: {e}", "warning")

    def update_lamella_objective_position(self, value: float):
        """Update the objective position of the current lamella."""

        # get current lamella
        idx = self.lamella_list.selected_index
        if idx == -1 or self.experiment is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return

        lamella = self.experiment.positions[idx]
        if lamella.fluorescence_pose is None:
            return
        # convert from µm to m
        lamella.fluorescence_pose.objective_position = value * MICRON_TO_METRE
        self.experiment.save()

    def _apply_objective_position_to_all(self):
        """Copy the current spinbox objective position to all lamella that have a fluorescence pose."""
        if self.experiment is None:
            return
        value_um = self.selected_lamella_widget.objective_value_um()
        value_m = value_um * MICRON_TO_METRE
        count = 0
        for lamella in self.experiment.positions:
            if lamella.fluorescence_pose is not None:
                lamella.fluorescence_pose.objective_position = value_m
                count += 1
        if count:
            self.experiment.save()
            notification_service.show_toast(
                f"Applied objective position ({value_um:.1f} µm) to {count} lamella.",
                "info",
            )

    def get_selected_lamella(self) -> Optional[Lamella]:
        """Get the currently selected lamella from the combobox.

        Returns:
            The selected lamella, or None if no experiment, no positions, or invalid selection.
        """
        if self.experiment is None:
            return None

        if not self.experiment.positions:
            return None

        idx = self.lamella_list.selected_index
        if idx == -1 or idx >= len(self.experiment.positions):
            return None

        return self.experiment.positions[idx]

    #### PROTOCOL
    def load_protocol(self):
        """Load a protocol into the current experiment using the protocol loading dialog."""
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        if self.experiment is None:
            notification_service.show_toast(
                "Please load an experiment first.", "warning"
            )
            return

        # Open the protocol loading dialog
        protocol = load_task_protocol_dialog(experiment=self.experiment, parent=self)

        if protocol is None:
            notification_service.show_toast("Protocol loading cancelled.", "info")
            return

        # assign protocol to experiment
        self.experiment.task_protocol = protocol
        self.experiment.save_protocol()

        notification_service.show_toast(
            f"Protocol '{protocol.name}' loaded successfully with {len(protocol.task_config)} tasks.",
            "info",
        )

        # Update UI
        self.update_ui()
        self.experiment_update_signal.emit()

    def export_protocol_ui(self):
        """Export the current protocol to file."""

        if self.experiment is None or self.experiment.task_protocol is None:
            notification_service.show_toast("No protocol loaded.", "info")
            return

        protocol_path = fui.open_save_file_dialog(
            msg="Select a protocol file",
            path=str(cfg.TASK_PROTOCOL_PATH),
            _filter="*.yaml",
            parent=self,
        )

        if protocol_path == "":
            notification_service.show_toast("No path selected", "info")
            return

        self.experiment.task_protocol.save(protocol_path)
        notification_service.show_toast(
            f"Saved Protocol to {os.path.basename(protocol_path)}", "info"
        )

    #########
    def cryo_deposition(self):
        if self.microscope is None:
            return
        cryo_deposition_widget = FibsemCryoDepositionWidget(self.microscope)
        cryo_deposition_widget.exec_()

    def set_instructions_msg(
        self,
        msg: str = "",
        pos: Optional[str] = None,
        neg: Optional[str] = None,
    ) -> None:
        """Set the instructions message, and user interaction buttons.
        Args:
            msg: The message to display.
            pos: The positive button text.
            neg: The negative button text.
        """
        self.label_instructions.setText(msg)
        # An empty prompt is not a blank line: with no message there is no question,
        # so the label goes away rather than reserving space for one.
        self.label_instructions.setVisible(bool(msg))
        self.pushButton_yes.setText(pos)
        self.pushButton_no.setText(neg)

        # enable buttons
        self.pushButton_yes.setEnabled(pos is not None)
        self.pushButton_yes.setVisible(pos is not None)
        self.pushButton_no.setEnabled(neg is not None)
        self.pushButton_no.setVisible(neg is not None)

        if pos in ("Run Milling", "Run Spot Burn"):
            self.pushButton_yes.setStyleSheet(
                stylesheets.SUPERVISION_STATUS_AUTOMATED_STYLESHEET
            )
        else:
            self.pushButton_yes.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_no.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

    def set_current_workflow_message(
        self, msg: Optional[str] = None, show: bool = True
    ):
        """Set the current workflow information message"""
        if msg is not None:
            self.label_workflow_information.setText(msg)
        self.label_workflow_information.setVisible(show)

    def push_interaction_button(self):
        """Handle the user interaction with the workflow."""
        self.pushButton_yes.setEnabled(False)
        self.pushButton_no.setEnabled(False)

        clicked_yes = bool(self.sender() == self.pushButton_yes)
        # The pending question owns this click; with every interaction converted
        # to the Responder there is no other path. A click with nothing pending
        # (a stray double-click after the answer landed) means nothing.
        self.ui_responder.answer_confirm(clicked_yes)

    def handle_acquisition_update(self, ddict: dict) -> None:
        if ddict.get("finished", False):
            self.update_lamella_ui()

    def stop_current_operations(self) -> None:
        """Interrupt whatever the microscope is doing right now.

        Shared by Stop Workflow and Stop Task, which have to bring the hardware to
        a halt identically and differ only in what the queue does afterwards.

        A task raising is not enough on its own: milling runs on its own thread
        with its own stop event, owned by the milling widget, and the task merely
        waits for it. Unwinding the task without this stops the waiting, not the
        mill.
        """
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.milling_widget.stop_milling()
        if self.spot_burn_widget is not None:
            self.spot_burn_widget.cancel_spot_burn()

    def _stop_workflow_thread(self):
        if self._task_manager is not None:
            self._task_manager.stop()
        else:
            self._workflow_stop_event.set()
        self.stop_current_operations()

    def _workflow_finished(self):
        """Handle the completion of the workflow."""
        logging.info("Workflow finished.")
        # Before the early returns: whatever question the run left behind must
        # come down even if the widgets below are gone. Covers the abort race
        # where a finished mill re-parks the prompt in the gap before the
        # aborting waiter cancels its future — by the time this runs, the
        # workflow thread has exited, so anything still parked belongs to nobody.
        self.ui_responder.abandon()
        if self.image_widget is None:
            return
        if self.microscope is None:
            return
        if self.experiment is None or self.protocol is None:
            return

        self._workflow_stop_event.clear()
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab))

        self.WAITING_FOR_USER_INTERACTION = False
        self.WORKFLOW_PENDING = False

        # clear milling task config
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()
            self.milling_task_config_widget.milling_widget.pushButton_run_milling.setVisible(
                True
            )

        # restore the spot burn widget: an aborted workflow skips the clear_spot_burn
        # message that normally resets it, which would leave the Burn button hidden
        # (no-op after a normal completion, where clear_spot_burn already ran)
        if self.spot_burn_widget is not None:
            self.spot_burn_widget.set_workflow_mode(False)
            self.spot_burn_widget.clear_points_layer()

        # clear detection layers
        if self.det_widget is not None:
            self.det_widget.clear_layers()

        # clear the image settings save settings etc
        self.image_widget.checkBox_image_save_image.setChecked(False)
        self.image_widget.lineEdit_image_path.setText(str(self.experiment.path))
        self.image_widget.lineEdit_image_label.setText("default-image")
        self.update_ui()

        # optionally turn off the beams when finished
        if self.protocol.options.turn_beams_off:
            self.microscope.turn_off(BeamType.ELECTRON)
            self.microscope.turn_off(BeamType.ION)

        self.set_current_workflow_message(msg=None, show=False)

        # show the post-workflow summary of tasks run this session
        self._show_workflow_summary()

    def _show_workflow_summary(self) -> None:
        """Show a modal summary dialog of the tasks run in the last workflow."""
        summary = self._last_run_summary
        self._last_run_summary = None
        if summary is None or summary.empty:
            return
        try:
            dialog = WorkflowSummaryDialog(summary, parent=self)
            dialog.exec_()
        except Exception as e:
            logging.warning(f"Failed to show workflow summary dialog: {e}")

    def handle_workflow_status(self, event: "WorkflowStatusEvent") -> None:
        """Show a fire-and-forget status update. GUI thread, via workflow_status_signal.

        The display half of what handle_workflow_update does for a status payload,
        and nothing else. Two deliberate absences: no widget-existence guards (these
        two labels exist from construction, so there is nothing to raise about in a
        queued slot), and no touching of the WAITING_* flags — a status update on its
        own channel can never release a blocked waiter, which is the point of the
        channel.
        """
        self.set_instructions_msg(event.message)
        self.set_current_workflow_message(event.workflow_info)

    def handle_workflow_update(self, info: dict) -> None:
        """Update the UI with the given information, ready for user interaction"""

        if self.image_widget is None:
            raise ValueError(
                "No image widget available. Please create an image widget first."
            )

        if self.milling_task_config_widget is None:
            raise ValueError(
                "No milling task config widget available. Please create a milling task config widget first."
            )

        # Images no longer arrive here: set_images_ui sends a SetImages request
        # through the Responder seam (QtResponder._set_images).

        # Detections, the alignment area, POI selection and the milling question
        # no longer arrive here: they are questions over the Responder seam
        # (QtResponder._confirm_detection, _edit_alignment_area, _pick_poi).

        # Milling config, spot-burn settings and fluorescence channels no longer
        # arrive here: their instructions go through the Responder seam
        # (QtResponder). The spot-burn question converted last (RunSpotBurn), so
        # no variant payloads remain — only status text below.

        # Instruction message. Read with `.get`, not indexed: this signal has no
        # declared contract, and 12 of its 13 emit sites pass an opaque variable, so
        # what a payload carries is not knowable without running it. Every emitter in
        # this repository sends `msg` today -- but a raise here does not degrade a
        # label, it aborts the process. PyQt5 calls `qFatal()` on any exception that
        # escapes a slot invoked from C++, which a queued signal from the workflow
        # thread is, and the abort takes the run with it and reaches no logfile
        # (FIB-329, FIB-402). It has happened: a queue edit put a payload without
        # `msg` on this signal and killed the app on every queue action.
        #
        # The default is `""` rather than a placeholder because an empty message
        # already means something here: `set_instructions_msg` hides the label, which
        # is what one emit site sends `{"msg": ""}` deliberately to do. A payload
        # carrying no message is an update about something other than the prompt.
        self.set_instructions_msg(
            info.get("msg", ""), info.get("pos", None), info.get("neg", None)
        )
        self.set_current_workflow_message(info.get("workflow_info", None))
