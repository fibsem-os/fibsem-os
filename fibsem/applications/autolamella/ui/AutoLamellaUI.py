import sys
import time
import warnings

from fibsem import conversions
from fibsem.structures import InsertableDeviceState
from fibsem.ui.widgets.custom_widgets import scrollable

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass
import functools
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

from fibsem.acting import OPERATOR
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.applications.autolamella.ui.poi_marker import PoiMarker
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
    FibsemImageSettingsWidget,
    FibsemMovementWidget,
    FibsemSpotBurnWidget,
    FibsemSystemSetupWidget,
    MillingTaskViewerWidget,
    notification_service,
    stylesheets,
)
from fibsem.ui import utils as fui
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.fm.widgets import FMImageViewerWidget

if (
    DETECTION_AVAILABLE
):  # ml dependencies are option, so we need to check if they are available
    from fibsem.ui.FibsemEmbeddedDetectionWidget import (
        FibsemEmbeddedDetectionUI as FibsemEmbeddedDetectionWidget,
    )

from psygnal import EmissionInfo
from superqt import ensure_main_thread

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.experiment_setup import load_for_adoption
from fibsem.applications.autolamella.poses import (
    add_lamella,
    current_objective_position,
    remove_lamella,
    set_lamella_pose,
    set_objective_position,
    set_objective_position_on_all,
    stored_objective_position,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
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
from fibsem.applications.autolamella.ui.autolamella_overview_image_widget import (
    create_overview_image_widget,
)
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.workflow_summary_dialog import (
    WorkflowSummaryDialog,
    run_counts_text,
)
from fibsem.workflows.session import AppSession

if TYPE_CHECKING:
    from concurrent.futures import Future

    import pandas as pd

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )
    from fibsem.workflows.status import (
        Hold,
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

# The PDF report needs the `reporting` extra (reportlab). The overview plot does not --
# it is matplotlib and Qt -- so it is imported with the other dialogs above rather than
# here, where a missing reportlab used to take it down too.
REPORTING_AVAILABLE: bool = False
try:
    from fibsem.applications.autolamella.ui.autolamella_generate_report_widget import (
        generate_report_dialog,
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


def _complete(call, outcome) -> None:
    """Complete ``outcome`` with ``call()``'s result, or its exception."""
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - the requester owns the failure
        outcome.set_exception(exc)
        return
    outcome.set_result(result)


class AutoLamellaUI(QMainWindow):
    # Everything the workflow says without needing an answer, as a
    # WorkflowStatusEvent. Questions and instructions do not travel on a signal
    # at all: they are typed requests to the QtResponder, each on its own
    # future (fibsem/workflows/interaction.py) — the dict workflow_update_signal that
    # once carried all three kinds of traffic is gone.
    workflow_status_signal = pyqtSignal(object)
    # Its own signal: a queue edit is not a step in the task lifecycle and must
    # not disturb the interaction UI.
    queue_changed_signal = pyqtSignal(dict)
    step_update_signal = pyqtSignal(str)  # emits human-readable step label
    _workflow_finished_signal = pyqtSignal(bool)
    # A stop was asked of the running workflow, by anyone: the operator's Stop,
    # the agent's workflow_stop (from the App Server's thread), closing the
    # window. The window shows "stopping" until the run has unwound.
    workflow_stopping_signal = pyqtSignal()
    experiment_update_signal = pyqtSignal()
    _hook_toast_signal = pyqtSignal(
        str, str
    )  # (message, notification_type) — thread-safe bridge for NotificationHook
    # A remote (agent) request, marshalled to the GUI thread the same way agent
    # answers are: the work, bound to its arguments and its Future, and marked
    # as whoever asked (FIB-1276) -- the mark stays on the requesting thread.
    _agent_call = pyqtSignal(object)
    # The agent's suggested answer to the question up (a dict, see
    # show_suggestion), or None when it goes.
    suggestion_changed = pyqtSignal(object)
    # The agent added or renamed grid records (App Server); the grid views redraw.
    grids_changed = pyqtSignal()

    @property
    def microscope(self) -> Optional[FibsemMicroscope]:
        return self.session.microscope

    @microscope.setter
    def microscope(self, microscope: Optional[FibsemMicroscope]) -> None:
        self.session.microscope = microscope

    @property
    def experiment(self) -> Optional[Experiment]:
        return self.session.experiment

    @experiment.setter
    def experiment(self, experiment: Optional[Experiment]) -> None:
        self.session.experiment = experiment

    def __init__(
        self,
        parent_ui: "AutoLamellaSingleWindowUI",
    ) -> None:
        super().__init__()
        # The microscope, the open experiment and the connection's event
        # stream are the session's; this window is one client of it.
        self.session = AppSession()

        self._setup_ui()
        self.parent_widget = parent_ui

        # The agent's suggestion the responder last showed (see show_suggestion).
        self.suggestion_shown: Optional[Dict[str, object]] = None
        # The point-of-interest marker on the FIB view: one for whoever asks
        # for the point, the question on the record or a PickPOI prompt.
        self.poi_marker = PoiMarker(self)
        # The Qt side of the workflow's Responder seam: workflow code is handed
        # this one-method object, never the window itself.
        self.ui_responder = QtResponder(self)
        self.session.responder = self.ui_responder
        # The operator answers the run's questions here, for the session's life.
        self.session.attach_answerer(OPERATOR)
        # The timeline renders the responder's question-lifecycle feed; the
        # widget itself is built in _setup_ui, before the responder exists.
        self.ui_responder.add_question_observer(self.question_timeline.record)
        # A bound method, not a lambda: the slot must run on this window's thread.
        self._agent_call.connect(self._run_agent_call)
        # Edits to the experiment run on this thread: its widgets listen to it.
        self.session.executor = self._run_on_gui_thread

        self._protocol_lock = threading.RLock()

        # the microscope whose moves the stage readout follows
        self._stage_followed: Optional[FibsemMicroscope] = None
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
        self.sample_widget: Optional[FibsemSampleWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskViewerWidget] = None
        self.milling_tab: Optional[QWidget] = None  # the scroll area around it
        self.det_widget: Optional["FibsemEmbeddedDetectionWidget"] = None

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

        # Display state, not a handshake: the run is held -- a question is up
        # for a click, or the run is parked on decisions -- and by whom. Set by
        # whoever takes the hold (the responder, the task manager, the main
        # window handing an agent's question over); the attention button,
        # border, status bar and timeline pause read it. The cross-thread
        # flag-poll it used to be -- USER_RESPONSE, WAITING_FOR_UI_UPDATE -- is
        # gone: every workflow interaction is a typed request on its own future
        # (fibsem/workflows/interaction.py).
        self.hold: Optional[Hold] = None
        # A state question a task asked (``AutoLamellaTask.ask`` with the
        # ``state`` kind), shown on the prompt bar: (item_id, task_name,
        # proposal_id). Continue decides it; the decision takes it down.
        self._state_question: Optional[Tuple[str, str, str]] = None
        # A pass of a loop a task asked (``milling``, ``spot_burn``), shown on
        # the prompt bar over its widget: (item_id, task_name, proposal). A
        # button decides it with the widget's value; the decision takes it down.
        self._loop_question: Optional[Tuple[str, str, Any]] = None
        # A value a task asked for that is edited on a widget of its own (a
        # ``detection`` on the Detection tab, an ``alignment_area`` or a
        # ``point_of_interest`` on the FIB view): (item_id, task_name,
        # proposal). Continue decides it with the widget's value.
        self._widget_question: Optional[Tuple[str, str, Any]] = None
        self._poi_frames_followed: Optional[object] = None
        # A run is active but nothing is executing -- today only during a
        # scheduled-start wait. Set from the worker thread, read by the border.
        self.WORKFLOW_PENDING: bool = False
        # The summary already toasted (by identity): the run-end toast is
        # once-per-run, while the runner's last_summary itself must survive as
        # the record remote readers see.
        self._shown_run_summary: Optional["pd.DataFrame"] = None

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

        # Who answered what, under the buttons: fed by the responder's
        # question-lifecycle feed, hidden until the first answer.
        from fibsem.applications.autolamella.ui.question_timeline_widget import (
            QuestionTimelineWidget,
        )

        self.question_timeline = QuestionTimelineWidget(self.centralwidget)

        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.label_workflow_information, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.label_instructions, 3, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_yes, 4, 0)
        self.gridLayout.addWidget(self.pushButton_no, 4, 1)
        self.gridLayout.addWidget(self.question_timeline, 5, 0, 1, 2)

        self.setCentralWidget(self.centralwidget)
        self.tabWidget.setCurrentIndex(0)

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        with self._protocol_lock:
            return (
                self.experiment.task_protocol if self.experiment is not None else None
            )

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
        self.workflow_status_signal.connect(self.handle_workflow_status)
        self._workflow_finished_signal.connect(self._workflow_finished)  # type: ignore
        # Every run starts and ends through the session's runner, whoever
        # started it: the window hears the run's status, and redraws.
        runner = self.session.runner
        runner.starting.connect(self._on_run_starting)
        runner.finished.connect(self._on_run_finished)
        runner.notice.connect(self._on_run_notice)
        runner.queue_changed.connect(self._on_run_queue_changed)
        runner.status.connect(self._on_run_status)
        runner.held.connect(self._on_run_held)
        runner.pending.connect(self._on_run_pending)
        runner.step.connect(self._on_run_step)

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
    def _overview_is_acquiring(self) -> bool:
        """Whether either overview modality is mid-tileset.

        Asked of the Overview tab rather than of a widget held here: the tab owns both
        modalities and rebuilds its widgets on every reconnection, so anything holding
        one of them directly would be answering for a widget that had been replaced.

        This used to ask the napari minimap, which means it stopped being true the
        moment the Overview tab took over acquisition -- a run from there did not
        suppress anything.
        """
        tab = getattr(self.parent_widget, "overview_tab", None)
        return tab is not None and tab.is_acquiring

    @ensure_main_thread
    def _on_stage_position_updated(self, stage_position: FibsemStagePosition) -> None:
        """Callback for when the stage position is updated."""
        if self._overview_is_acquiring:
            # Mid-tileset the stage moves once per tile, and refreshing the readout on
            # each is churn nobody reads.
            return
        if self.movement_widget is not None:
            # pass the position from the signal; re-querying the microscope here races
            # with the worker thread driving the move (see TescanMicroscope socket lock)
            self.movement_widget.update_ui(stage_position=stage_position)

    def _follow_stage(self, microscope: Optional[FibsemMicroscope]) -> None:
        """Keep the stage readout on ``microscope``'s moves, and off any other's.

        The readout follows the microscope, not the experiment: a move with no
        experiment open still moves the stage (FIB-1421)."""
        followed = self._stage_followed
        if followed is microscope:
            return
        if followed is not None and followed.stage is not None:
            try:
                followed.stage.position.changed.disconnect(
                    self._on_stage_position_updated
                )
            except Exception as e:
                logging.debug(f"Could not disconnect microscope stage events: {e}")
        if microscope is not None and microscope.stage is not None:
            microscope.stage.position.changed.connect(self._on_stage_position_updated)
        self._stage_followed = microscope

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

        self.adopt_experiment(experiment)

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

        self.adopt_experiment(experiment)

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

        experiment, why = load_for_adoption(experiment_path)
        if experiment is None:
            msg = f"Quickload: {why}; not loaded."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        self.adopt_experiment(experiment)
        logging.info(
            f"Quickload: loaded experiment {experiment.name} from {experiment_path}"
        )

    def adopt_experiment(self, experiment: Experiment) -> None:
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

        self.session.adopt(experiment)

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
        self._follow_stage(self.microscope)
        if self.experiment is not None:
            self.settings.image.path = self.experiment.path
        self.update_microscope_ui()
        self.update_ui()
        if self.experiment is not None:
            self._disconnect_experiment_events()
            self._setup_experiment_connections()
        self._start_event_recorder()
        self._start_app_server()

    def _start_event_recorder(self) -> None:
        """Start recording events for this microscope connection, whatever else runs.

        This is the slot for the system widget's ``connected_signal``, which fires
        whenever that widget refreshes while connected -- not once per connection
        -- so it can run again for a microscope that already has a stream, and
        that microscope keeps it. A different microscope (the connection dialog can
        hand one back) gets a new stream, and the session stops a running App
        Server so it restarts on the new buffer.
        """
        self.session.start_recording()

    def closeEvent(self, event) -> None:
        """Closed on its own, as tests and scripts use it, the window closes its
        recorder, whose writer thread would otherwise run until the process
        ends. Embedded, it is never closed: the main window's closeEvent does
        this instead."""
        self.session.stop_recording()
        super().closeEvent(event)

    def _start_app_server(self) -> None:
        """Host the App Server over this session, if the preference asks for it.

        The agent starts at Watch; never raises — an optional observer
        must not be able to take the session down (see fibsem/server/hosting.py).
        """
        preference = fibsem_cfg.load_user_preferences().app_server
        if preference.enabled:
            from fibsem.applications.autolamella.server.context import (
                AutoLamellaAppContext,
            )

            if self.session.serve_microscope is None:
                # the operator's choice in Tools → App Server is per session,
                # like the agent level; it starts at the preference's default
                self.session.serve_microscope = preference.microscope
            self.session.start_app_server(
                lambda buffer: AutoLamellaAppContext(
                    self.session, window=self, event_buffer=buffer
                )
            )

    def sync_app_server_with_preference(self) -> None:
        """Start or stop the embedded server to match the saved preference.

        Called when Tools → App Server saves the switch, so it acts now rather
        than at the next connect. Starting needs a connected microscope (the
        server wraps it); without one this is a no-op and the connect path
        picks the preference up as before.
        """
        host = self.session.app_server
        preference = fibsem_cfg.load_user_preferences().app_server
        if preference.enabled:
            if self.microscope is not None and (host is None or not host.running):
                self._start_app_server()
        elif host is not None and host.running:
            host.stop()

    def set_app_server_microscope(self, serve: bool) -> None:
        """Serve the microscope, or stop serving it (FIB-1359), from Tools →
        App Server, which also saves the choice for the next launch.

        The services are fixed when the server starts, so a change restarts a
        running server. The agent's token and level are the session's and
        carry over, so a connected agent carries on.
        """
        serve = bool(serve)
        if serve == self.session.serve_microscope:
            return
        logging.info(
            "App Server: the microscope %s by the operator",
            "served" if serve else "no longer served",
        )
        if self.session.set_serve_microscope(serve) and self.microscope is not None:
            self._start_app_server()

    def disconnect_from_microscope(self):
        self.session.stop_app_server()
        self.session.stop_recording()
        self._follow_stage(None)
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()

    def front_tab(self, widget: QWidget) -> None:
        """Bring forward the tab that holds *widget*.

        The widget may be the tab itself or sit inside a wrapper (the Milling tab is
        a scroll area around its editor). ``setCurrentWidget`` on a widget that is
        not a direct tab is a silent no-op, which is how the responder stopped
        fronting the Milling tab once the wrapper arrived.
        """
        candidate = widget
        while candidate is not None and self.tabWidget.indexOf(candidate) == -1:
            candidate = candidate.parentWidget()
        if candidate is not None:
            # A hidden tab cannot be fronted: Detection is hidden until it is
            # asked for, so bringing it forward has to un-hide it first or the
            # call is a second silent no-op.
            self.tabWidget.setTabVisible(self.tabWidget.indexOf(candidate), True)
            self.tabWidget.setCurrentWidget(candidate)

    def front_question(self) -> None:
        """Bring forward the tab the question now up is asked on, if it has one.

        A question on its own tab is gone from view the moment the operator
        looks elsewhere, and the attention button is how they get back to it --
        so the button cannot just go to the Microscope tab and call it done.
        Does nothing when no question is up, or when its prompt is the shared
        one the Microscope tab already shows.
        """
        widget = self.question_host()
        if widget is not None:
            self.front_tab(widget)

    def question_host(self):
        """The tab the question now up is answered on, when it has one of its
        own (a mill's pass, a detection); None when it is on the Microscope
        tab, or nothing is up."""
        return (
            self.loop_question_host()
            or self.widget_question_host()
            or self.ui_responder.question_host()
        )

    def update_microscope_ui(self):
        """Update the ui based on the current state of the microscope."""

        if self.microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,  # type: ignore
                parent=self,
                images=self.session.images,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                parent=self,
            )

            # The defaults' "Read from Acquire Tab" reads this tab.
            self.system_widget.set_current_imaging(
                self.image_widget._get_image_settings_from_ui
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.milling_task_config_widget = MillingTaskViewerWidget(
                microscope=self.microscope,
                image_widget=self.image_widget,
                parent=self,
            )
            # The milling widget no longer scrolls itself; its tab does. The tab is
            # kept by name because indexOf / setCurrentWidget want the tab, not the
            # widget inside it -- see front_tab.
            self.milling_tab = scrollable(self.milling_task_config_widget)
            self.tabWidget.addTab(self.milling_tab, "Milling")

            # The hardware view of the grids: the holder, and the magazine when
            # there is one. Slot moves go through the Movement widget, the same
            # route as a saved position, so the readout and post-move images follow.
            self.sample_widget = FibsemSampleWidget(
                microscope=self.microscope, parent=self
            )
            self.sample_widget.move_to_requested.connect(
                self.movement_widget.move_to_position
            )
            self.tabWidget.addTab(self.sample_widget, "Sample")

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

            self.image_widget.acquisition_progress_signal.connect(
                self.handle_acquisition_update
            )
        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.sample_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.sample_widget))
                self.sample_widget.deleteLater()
                self.sample_widget = None
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
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.milling_tab))
                self.milling_tab.deleteLater()  # owns the widget
                self.milling_tab = None
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
                self.system_widget.set_current_imaging(None)

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

        if not REPORTING_AVAILABLE:
            notification_service.show_toast(
                "Reporting tools are not available. "
                'Install the reporting extra: pip install "fibsem[reporting]"',
                "warning",
            )
            return

        generate_report_dialog(self.experiment, parent=self)
        return

    def action_generate_overview_plot(self) -> None:
        """Generate an plot with the lamella position on an overview image."""
        if self.experiment is None:
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

    def _fm_image_viewer_start_directory(self) -> str:
        """Where the viewer's Load dialog opens: the open experiment, else the folder
        of the most recent one, else the log directory. The viewer reads files, so it
        needs no experiment (FIB-942); this only picks a sensible first folder."""
        if self.experiment is not None and self.experiment.path:
            return str(self.experiment.path)
        recent = fibsem_cfg.load_user_preferences().experiment.recent_experiments
        for path in recent:
            parent = os.path.dirname(path)
            if os.path.isdir(parent):
                return parent
        return fibsem_cfg.LOG_PATH

    def _open_fm_image_viewer(self):
        """Open the FM Image Viewer as a standalone window."""
        # Parented to None so it gets its own taskbar entry and native minimise, like the
        # coincidence viewer. That means nothing else owns it, so the reference here is
        # what keeps it alive — drop it and Python collects the window mid-session.
        self._fm_image_viewer_window = FMImageViewerWidget(
            start_directory=self._fm_image_viewer_start_directory()
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

    #### TASK WORKFLOW

    def _on_run_starting(self, request) -> None:
        """Where the run was started, before its thread starts."""
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()

    def _on_run_finished(self, cancelled: bool) -> None:
        """The run's thread: over to the GUI thread."""
        self._workflow_finished_signal.emit(cancelled)  # type: ignore

    def _on_run_notice(self, message: str, notification_type: str) -> None:
        """A hook's notice, from the run's thread: signals are thread-safe."""
        self._hook_toast_signal.emit(message, notification_type)

    def _on_run_queue_changed(self, info: dict) -> None:
        """The running queue was edited, from any thread: signals are
        thread-safe."""
        self.queue_changed_signal.emit(info)

    def _on_run_status(self, event: "WorkflowStatusEvent") -> None:
        """A status line or a task's report, from the run's thread."""
        self.workflow_status_signal.emit(event)

    def _on_run_held(self, hold: Optional["Hold"]) -> None:
        """Who holds the run, from the run's thread. Kept here for the chrome,
        which the status line that follows redraws."""
        self.hold = hold

    def _on_run_pending(self, pending: bool) -> None:
        """Whether the run waits between tasks, from the run's thread."""
        self.WORKFLOW_PENDING = pending

    def _on_run_step(self, step: str) -> None:
        """The step a task is on, from the run's thread."""
        self.step_update_signal.emit(step)

    def _run_on_gui_thread(self, call) -> "Future":
        """The session's executor: ``call()`` on the GUI thread as whoever
        called this; any thread. Resolves to its result, or its exception."""
        from concurrent.futures import Future as _Future

        from fibsem.acting import carried

        outcome: "_Future" = _Future()
        self._agent_call.emit(carried(functools.partial(_complete, call, outcome)))
        return outcome

    def _run_agent_call(self, call) -> None:
        """GUI thread. Run a call :meth:`_run_on_gui_thread` sent."""
        call()

    def after_agent_edit(self, select_latest: bool = False) -> None:
        """GUI thread. Redraw after an agent's edit to the experiment (adding a
        lamella, setting a pose), as after the Lamella tab's own buttons,
        selecting the newest lamella when ``select_latest``."""
        try:
            self.update_lamella_combobox(latest=select_latest)
            self.update_ui()
            lamella = self.get_selected_lamella()
            if lamella is not None:
                self.selected_lamella_widget.set_lamella(lamella)
            if self.experiment is not None:
                # The FM overview draws positions itself; it hears of a moved pose
                # only by being told.
                self.experiment.positions.events.changed.emit()
        except Exception:
            logging.exception("redraw after an agent's experiment edit failed")

    def before_config_edit(self) -> None:
        """GUI thread. Before an agent's config edit: flush any edit still
        sitting in the editors (the FIB-683 one-writer rule), so an operator
        edit that changes the same config makes the agent's version stale."""
        parent = self.parent_widget
        if parent is not None:
            try:
                parent.lamella_widget.flush_pending_save()
            except Exception:
                logging.exception("flush before agent config edit failed; continuing")

    def after_config_edit(
        self, level: str, item_name: str, task_name: str, result: dict
    ) -> None:
        """GUI thread. After an agent's config edit applied: rebuild the
        editor showing it and toast the change."""
        if level == "apply_protocol":
            self._show_agent_protocol_apply(item_name, result.get("task_names") or [])
            return
        changes = [(c["path"], c["old"], c["new"]) for c in result.get("changes") or []]
        if changes:
            self._show_agent_config_patch(level, item_name, task_name, changes)

    def _show_agent_protocol_apply(self, item_name: str, names) -> None:
        try:
            notification_service.show(
                f"Agent applied protocol to {item_name} — {', '.join(names)}",
                "info",
            )
        except Exception:
            logging.exception("toast after protocol apply failed")
        parent = self.parent_widget
        if parent is not None:
            try:
                parent.lamella_widget.refresh_if_showing(item_name)
            except Exception:
                logging.exception("editor refresh after protocol apply failed")

    def _show_agent_config_patch(
        self, level: str, item_name: str, task_name: str, changes
    ) -> None:
        """Rebuild the editor showing this config and toast the change.

        GUI thread, after a successful apply. The rebuild is what stops a
        stale open form writing old values back on the operator's next edit;
        the toast is what stops the form appearing to change by itself.
        Chrome only — a failure here must never fail the applied patch.
        """
        try:
            leaf = changes[0][0].rsplit(".", 1)[-1]
            summary = f"{leaf}: {changes[0][1]!r} → {changes[0][2]!r}"
            if len(changes) > 1:
                summary += f" (+{len(changes) - 1} more)"
            if level == "item":
                target = f"{task_name} for {item_name}"
            elif level == "item_fields":
                target = item_name
            else:
                target = task_name
            # show(), not show_toast(): agent changes are workflow events —
            # they persist in the notification bell, not just flash for 3 s
            # while the operator is looking at the canvas (found live: the
            # toasts fired and left no trace).
            notification_service.show(f"Agent edited {target} — {summary}", "info")
        except Exception:
            logging.exception("toast after agent config patch failed")
        parent = self.parent_widget
        if parent is None:
            return
        try:
            if level in ("item", "item_fields"):
                parent.lamella_widget.refresh_if_showing(item_name)
            else:
                parent.task_widget.refresh_if_showing_task(task_name)
        except Exception:
            logging.exception("editor refresh after agent config patch failed")

    def stop_task_workflow(self, actor=None, via: str = "") -> None:
        """Stop the run and bring the hardware to a halt; ``actor`` and ``via``
        say who stopped it and from where, for the record (``runner.stop``)."""
        if not self.session.runner.is_running:
            return
        self.session.runner.stop(actor=actor, via=via)
        self.stop_current_operations()
        self.workflow_stopping_signal.emit()

    def stop_current_task(self, actor=None, via: str = "", **details) -> None:
        """Abandon the task now running and carry on with the rest of the queue.

        Stop Task's two halves, as Stop Workflow has them: tell the queue this
        task is being abandoned, and bring the hardware to a halt. Without the
        second, the task stops waiting for the mill but the mill keeps going.
        """
        self.session.runner.stop_task(actor=actor, via=via, **details)
        self.stop_current_operations()

    #### UI UPDATES

    def update_ui(self):
        """Update the ui based on the current state of the application."""

        if self.session.runner.is_running:
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
        self.selected_lamella_widget.setEnabled(not self.session.runner.is_running)

        # Current Lamella Status
        if has_lamella and self.experiment is not None:
            self.update_lamella_ui()

        if self.session.runner.is_running:
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

    def update_lamella_combobox(self, latest: bool = False):
        if self.experiment is None:
            return
        if self.session.runner.is_running:
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

        if self.session.runner.is_running:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            return

        lamella: Lamella = self.experiment.positions[idx]
        logging.debug(f"Updating Lamella UI for {lamella.status_info}")

        # refresh objective position + pose display for the selected lamella
        self.selected_lamella_widget.set_lamella(lamella)

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

    def add_new_lamella(
        self,
        stage_position: Optional[FibsemStagePosition] = None,
        name: Optional[str] = None,
        objective_position: Optional[float] = None,
        marked_at: Optional[str] = None,
        grid_id: Optional[str] = None,
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
            grid_id: The grid this lamella is on, for a caller that knows -- one
                marked on a grid's overview belongs to that grid whether or not it
                is on the stage. Left alone it is resolved from the stage.
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

        lamella = add_lamella(
            self.experiment,
            self.microscope,
            stage_position=stage_position,
            name=name,
            objective_position=objective_position,
            marked_at=marked_at,
            grid_id=grid_id,
        )
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
        if not remove_lamella(self.experiment, lamella):
            return
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

        # As Set pose does: the stored objective position is kept, and the
        # milling angle and fluorescence pose follow the new milling pose.
        set_lamella_pose(
            self.microscope,
            lamella,
            "MILLING",
            deepcopy(self.microscope.get_microscope_state()),
        )

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

        if set_lamella_pose(self.microscope, lamella, pose_name, state):
            self.selected_lamella_widget.refresh_pose(
                "FLUORESCENCE", lamella.fluorescence_pose
            )

        self.experiment.save()
        self.selected_lamella_widget.refresh_pose(pose_name, state)
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
        value_m = current_objective_position(self.microscope.fm)
        if value_m is None:
            notification_service.show_toast(
                "Objective position unavailable.", "warning"
            )
            return
        set_objective_position(self.experiment, lamella, value_m)
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
        try:
            objective_position = stored_objective_position(self.microscope.fm, lamella)
        except ValueError as e:
            notification_service.show_toast(str(e), "warning")
            return
        obj = self.microscope.fm.objective

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
        set_objective_position(self.experiment, lamella, value * MICRON_TO_METRE)

    def _apply_objective_position_to_all(self):
        """Copy the current spinbox objective position to all lamella that have a fluorescence pose."""
        if self.experiment is None:
            return
        value_um = self.selected_lamella_widget.objective_value_um()
        count = set_objective_position_on_all(
            self.experiment, value_um * MICRON_TO_METRE
        )
        if count:
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

        self.adopt_protocol(protocol)
        notification_service.show_toast(
            f"Protocol '{protocol.name}' loaded successfully with {len(protocol.task_config)} tasks.",
            "info",
        )

    def adopt_protocol(self, protocol) -> None:
        """Make *protocol* the open experiment's, save it beside the experiment
        and redraw: the Load Protocol dialog's OK, and an agent's load. GUI
        thread."""
        self.experiment.task_protocol = protocol
        self.experiment.save_protocol()
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

    def show_suggestion(
        self,
        label: Optional[str],
        reason: str = "",
        point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """The agent's suggested answer, or None to take it down. Called by the
        responder, which owns when it stands. Not on the prompt: the agent card
        shows it (FIB-1407), so this says so through `suggestion_changed`.
        *point* is a suggested point of interest, in the FIB image's pixels."""
        if label is None:
            self.suggestion_shown = None
        else:
            self.suggestion_shown = {
                "question": self.label_instructions.text(),
                "answer": label,
                "reason": reason,
            }
            if point is not None:
                self.suggestion_shown["point"] = point
        self.suggestion_changed.emit(self.suggestion_shown)

    def take_suggestion(self) -> bool:
        """The operator takes the agent's suggestion: their answer, suggested by
        the agent. False when it had gone."""
        return self.ui_responder.take_suggestion()

    def dismiss_suggestion(self, reason: str = "") -> bool:
        """The operator declines the agent's suggestion, with why if they
        like; the question stays up. False when it had gone."""
        return self.ui_responder.dismiss_suggestion(reason)

    def push_interaction_button(self):
        """Handle the user interaction with the workflow."""
        self.pushButton_yes.setEnabled(False)
        self.pushButton_no.setEnabled(False)

        clicked_yes = bool(self.sender() == self.pushButton_yes)
        # The pending question owns this click. Otherwise a state question on
        # the record may: its Continue is a decision, not an answer. A click
        # with neither (a stray double-click after the answer landed) means
        # nothing.
        if self.ui_responder.answer_confirm(clicked_yes):
            return
        if self._loop_question is not None:
            self._decide_loop_question(clicked_yes)
            return
        if self._widget_question is not None:
            if clicked_yes:
                self._decide_widget_question()
            return
        self._decide_state_question(clicked_yes)

    # -- a state question, on the prompt bar ----------------------------------

    def show_state_question(self, item_id: str, task_name: str, proposal) -> None:
        """The prompt bar is the renderer for the ``state`` kind: the task's
        message and its button, Continue unless the task names another
        (``positive`` in the provenance). Continue records the operator's
        confirmation through ``Experiment.decide``, the one write path; the
        task then reads the instrument for the position as confirmed. A task
        that can do without it names a second button (``negative``: the
        tilt's Skip), which rejects it: "don't", and the task carries on."""
        from fibsem.workflows.status import (
            WorkflowStatusEvent,
        )

        self._state_question = (item_id, task_name, proposal.id)
        message = str(
            proposal.provenance.get("message") or "Press Continue when ready."
        )
        self.set_instructions_msg(
            message,
            str(proposal.provenance.get("positive") or "Continue"),
            proposal.provenance.get("negative") or None,
        )
        self.workflow_status_signal.emit(WorkflowStatusEvent())

    def clear_state_question(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the question up,
        the prompt comes down (answered here, by an agent, or withdrawn)."""
        shown = self._state_question
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.workflows.status import (
            WorkflowStatusEvent,
        )

        self._state_question = None
        self.set_instructions_msg("")
        self.workflow_status_signal.emit(WorkflowStatusEvent(message=""))

    def _decide_state_question(self, clicked_yes: bool = True) -> None:
        shown = self._state_question
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal_id = shown
        from fibsem.workflows.proposals import Decision, DecisionOutcome

        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed
                if clicked_yes
                else DecisionOutcome.Rejected,
                actor=experiment.actor(),
                via="workflow",
                proposal_id=proposal_id,
                reason="" if clicked_yes else f"{self.pushButton_no.text()} pressed",
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(
                f"{task_name}: could not confirm the state: {result.reason}"
            )
            self.pushButton_yes.setEnabled(True)
            self.pushButton_no.setEnabled(bool(self.pushButton_no.text()))

    # -- a loop's pass, on the prompt bar over its widget ----------------------

    def _loop_widget(self, kind: str):
        """The widget a pass of ``kind`` is decided in, or None without one."""
        from fibsem.workflows.proposals import MILLING, SPOT_BURN

        if kind == MILLING:
            return self.milling_task_config_widget
        if kind == SPOT_BURN:
            return self.spot_burn_widget
        return None

    def show_loop_question(self, item_id: str, task_name: str, proposal) -> None:
        """The prompt bar over the milling editor (or the spot-burn widget) is
        the renderer for a loop's pass: the task's message and the pass's two
        buttons. The task has already put what the pass would run in the
        widget; the operator edits it there, and a button records the decision
        through ``Experiment.decide`` with the widget's value. Without the
        widget (no spot-burn hardware) the buttons decide the pass as
        proposed."""
        from fibsem.applications.autolamella.ui.loop_question import loop_prompt
        from fibsem.workflows.proposals import SPOT_BURN
        from fibsem.workflows.status import WorkflowStatusEvent

        widget = self._loop_widget(proposal.kind)
        if widget is None:
            pass
        elif proposal.kind == SPOT_BURN:
            widget.set_settings(proposal.values["spot_burn"])
            self.set_spot_burn_widget_active(True)
            widget.set_workflow_mode(True)
        else:
            widget.setEnabled(True)
            self.front_tab(widget)
            # The task owns the mill while it asks: the editor's own Run stands down.
            widget.milling_widget.pushButton_run_milling.setVisible(False)
        self._loop_question = (item_id, task_name, proposal)
        prompt = loop_prompt(proposal)
        self.set_instructions_msg(prompt.message, prompt.positive, prompt.negative)
        self.workflow_status_signal.emit(WorkflowStatusEvent())

    def loop_question_host(self):
        """The widget the pass now up is decided in, for the attention button
        to go back to; None when no pass is up."""
        if self._loop_question is None:
            return None
        return self._loop_widget(self._loop_question[2].kind)

    def clear_loop_question(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the pass up, the
        prompt comes down (decided here, by an agent, or withdrawn). The widget keeps its value: the task asks again, or clears
        it when the loop ends."""
        shown = self._loop_question
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.workflows.status import WorkflowStatusEvent

        self._loop_question = None
        self.set_instructions_msg("")
        self.workflow_status_signal.emit(WorkflowStatusEvent(message=""))

    def _decide_loop_question(self, clicked_yes: bool) -> None:
        shown = self._loop_question
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal = shown
        from fibsem.applications.autolamella.ui.loop_question import loop_prompt
        from fibsem.workflows.proposals import (
            ACTION_EDIT,
            ACTION_RUN,
            SPOT_BURN,
            Decision,
            DecisionOutcome,
        )

        prompt = loop_prompt(proposal)
        action = prompt.yes if clicked_yes else prompt.no
        if action is None:
            return
        values = {}
        widget = self._loop_widget(proposal.kind)
        if action in (ACTION_RUN, ACTION_EDIT) and widget is not None:
            # What runs is what the widget holds: the operator's edits. Skip
            # and accept carry nothing: the task goes on with what it had.
            if proposal.kind == SPOT_BURN:
                values = {"spot_burn": deepcopy(widget.get_settings())}
            else:
                values = {"milling_config": deepcopy(widget.get_config())}
        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor=experiment.actor(),
                values=values,
                via="workflow",
                proposal_id=proposal.id,
                action=action,
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(f"{task_name}: could not decide the pass: {result.reason}")
            self.pushButton_yes.setEnabled(True)
            self.pushButton_no.setEnabled(prompt.negative is not None)

    # -- a value edited on its own widget, on the prompt bar ------------------

    def show_widget_question(self, item_id: str, task_name: str, proposal) -> None:
        """The widget a value is edited on is the renderer for its kind: a
        detection's features on the Detection tab (the task loaded them there
        first), an alignment area as the rectangle on the FIB view, a point of
        interest as the marker on it. The prompt bar carries the task's
        message and Continue, which records the decision through
        ``Experiment.decide`` with the widget's value. Without the widget the
        Continue confirms the value as proposed."""
        from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST
        from fibsem.workflows.proposals import ALIGNMENT_AREA, DETECTION
        from fibsem.workflows.status import WorkflowStatusEvent

        message = str(proposal.provenance.get("message") or "")
        if proposal.kind == DETECTION and self._detection_shown():
            self.front_tab(self.det_widget)
            message = "Confirm Feature Detection. Press Continue to proceed."
        elif proposal.kind == ALIGNMENT_AREA and self.image_widget is not None:
            self.tabWidget.setCurrentWidget(self.tab)
            self.image_widget.toggle_alignment_area(
                proposal.values.get("alignment_area")
            )
            message = f"{message} Press Continue when done.".strip()
        elif proposal.kind == POINT_OF_INTEREST and self.poi_marker.show(
            proposal.values.get("poi"),
            fallback=self._reference_image(item_id, proposal),
        ):
            self.tabWidget.setCurrentWidget(self.tab)
            self._follow_poi_frames()
        self._widget_question = (item_id, task_name, proposal)
        self.set_instructions_msg(message or "Press Continue when ready.", "Continue")
        self.workflow_status_signal.emit(WorkflowStatusEvent())

    def _reference_image(self, item_id: str, proposal):
        """The image the proposal's values sit on, for a view showing none."""
        from fibsem.applications.autolamella.review_images import load_reference_image

        item = self.experiment.get_item_by_id(item_id)
        return load_reference_image(self.experiment, item, proposal)

    def _detection_shown(self) -> bool:
        """The Detection tab holds the features the task sent before asking;
        a window that started listening mid-question has none to correct."""
        return self.det_widget is not None and self.det_widget.det is not None

    def widget_question_host(self):
        """The tab the value now up is edited on, when it has one of its own
        (a detection), for the attention button to go back to."""
        from fibsem.workflows.proposals import DETECTION

        shown = self._widget_question
        if shown is None or shown[2].kind != DETECTION:
            return None
        return self.det_widget

    def _follow_poi_frames(self) -> None:
        """Keep the marker on the frame the FIB view shows, once per widget."""
        signal = getattr(self.image_widget, "viewer_update_signal", None)
        if signal is None or self._poi_frames_followed is self.image_widget:
            return
        signal.connect(self._sync_poi_question)
        self._poi_frames_followed = self.image_widget

    def _sync_poi_question(self, *_args) -> None:
        from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST

        shown = self._widget_question
        if shown is not None and shown[2].kind == POINT_OF_INTEREST:
            self.poi_marker.sync()

    def clear_widget_question(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the value up, the
        prompt and the widget's editing come down (decided here, by an agent,
        or withdrawn)."""
        shown = self._widget_question
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST
        from fibsem.workflows.proposals import ALIGNMENT_AREA, DETECTION
        from fibsem.workflows.status import WorkflowStatusEvent

        self._widget_question = None
        kind = shown[2].kind
        if kind == DETECTION and self.det_widget is not None:
            self.det_widget.clear_layers()
        elif kind == ALIGNMENT_AREA and self.image_widget is not None:
            self.image_widget.clear_alignment_area()
        elif kind == POINT_OF_INTEREST:
            self.poi_marker.remove()
        self.set_instructions_msg("")
        self.workflow_status_signal.emit(WorkflowStatusEvent(message=""))

    def _widget_value(self, proposal) -> Dict[str, Any]:
        """What the widget holds for ``proposal``'s kind; empty (as proposed)
        when the widget was not there to edit it on."""
        from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST
        from fibsem.detection.review import detection_values
        from fibsem.workflows.proposals import ALIGNMENT_AREA, DETECTION

        if proposal.kind == DETECTION and self._detection_shown():
            return detection_values(self.det_widget._get_detected_features())
        if proposal.kind == ALIGNMENT_AREA and self.image_widget is not None:
            area = self.image_widget.get_alignment_area()
            return {"alignment_area": deepcopy(area)} if area is not None else {}
        if proposal.kind == POINT_OF_INTEREST and self.poi_marker.shown:
            poi = self.poi_marker.read()
            return {"poi": poi} if poi is not None else {}
        return {}

    def _decide_widget_question(self) -> None:
        shown = self._widget_question
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal = shown
        from fibsem.workflows.proposals import Decision, DecisionOutcome

        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor=experiment.actor(),
                values=self._widget_value(proposal),
                via="workflow",
                proposal_id=proposal.id,
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(f"{task_name}: could not decide it: {result.reason}")
            self.pushButton_yes.setEnabled(True)

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

        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab))

        self.hold = None
        self.WORKFLOW_PENDING = False

        # clear milling task config
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()
            self.milling_task_config_widget.milling_widget.pushButton_run_milling.setVisible(
                True
            )

        # restore the spot burn widget: an aborted workflow may skip the ClearSpotBurn
        # that normally resets it, which would leave the Burn button hidden
        # (no-op after a normal completion, where ClearSpotBurn already ran)
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
        """Toast the end of the last workflow, and open its summary unless the
        workflow's "Show summary after completion" is off.

        The toast always comes, with a Show summary link, and stays in the
        notification bell, so any run's summary is a click away after it
        fades. The dialog opens without blocking the window (FIB-1376): as a
        modal it held the window at the end of every run, which an agent
        starting many short runs, or a run ending overnight, left standing
        over the app. That is also when to turn it off.

        Show-once, but never consume: the runner's ``last_summary`` is also
        the record the App Server's ``run_summary`` endpoint reads after the
        manager is gone. The record stays until the next run's capture
        overwrites it; only the toast is once-only.
        """
        runner = self.session.runner
        summary = runner.last_summary
        if summary is None or summary is self._shown_run_summary:
            return
        self._shown_run_summary = summary
        if summary.empty:
            return
        note = runner.last_note
        failed = (
            "task_status" in summary.columns
            and (summary["task_status"] == "Failed").any()
        )
        notification_service.show(
            f"Run ended: {run_counts_text(summary)}",
            "warning" if note or failed else "success",
            action=(
                "Show summary",
                lambda: self._open_workflow_summary(summary, note),
            ),
        )
        protocol = self.protocol
        if protocol is None or protocol.options.show_run_summary:
            self._open_workflow_summary(summary, note)

    def _open_workflow_summary(self, summary: "pd.DataFrame", note: str) -> None:
        """Open a run's summary dialog, without blocking the window."""
        try:
            dialog = WorkflowSummaryDialog(summary, note=note, parent=self)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.show()
        except Exception as e:
            logging.warning(f"Failed to show workflow summary dialog: {e}")

    def handle_workflow_status(self, event: "WorkflowStatusEvent") -> None:
        """Show a fire-and-forget status update. GUI thread, via workflow_status_signal.

        Two deliberate absences: no widget-existence guards (these two labels
        exist from construction, so there is nothing to raise about in a queued
        slot), and no touching of the waiting display state — a status update on
        its own channel can never release a blocked waiter, which is the point
        of the channel. A ``message`` of None says nothing about the prompt and
        leaves it standing — the responder pings this signal for chrome
        refreshes while its question is up.
        """
        if event.message is not None:
            self.set_instructions_msg(event.message)
        self.set_current_workflow_message(event.workflow_info)
