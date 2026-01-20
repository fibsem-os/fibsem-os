import sys


try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass
import glob
import logging
import os
import subprocess
import threading
from copy import deepcopy
from typing import List, Optional
import napari
import napari.utils.notifications
import fibsem
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeSettings,
)
from fibsem.ui import (
    DETECTION_AVAILABLE,
    FibsemCryoDepositionWidget,
    FibsemImageSettingsWidget,
    FibsemMinimapWidget,
    FibsemMovementWidget,
    FibsemSystemSetupWidget,
    FibsemSpotBurnWidget,
    stylesheets,
)
from fibsem.ui import utils as fui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QDialog, QVBoxLayout, QDialogButtonBox, QMessageBox, QAction
if DETECTION_AVAILABLE: # ml dependencies are option, so we need to check if they are available
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI as FibsemEmbeddedDetectionWidget

from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget
from fibsem.ui.widgets.autolamella_create_experiment_widget import create_experiment_dialog
from fibsem.ui.widgets.autolamella_load_experiment_widget import load_experiment_dialog
from fibsem.ui.widgets.autolamella_load_task_protocol_widget import load_task_protocol_dialog
from fibsem.ui.widgets.autolamella_task_config_editor import show_protocol_editor, AutoLamellaProtocolEditorTabWidget
from fibsem.ui.widgets.autolamella_task_history_widget import AutoLamellaWorkflowDisplayWidget
from fibsem.ui.widgets.autolamella_defect_state_widget import AutoLamellaDefectStateWidget
from fibsem.ui.fm.widgets import MinimapPlotWidget
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
    Experiment,
    Lamella,
    DefectState,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import run_tasks
from fibsem.applications.autolamella.ui.qt import AutoLamellaUI as AutoLamellaMainUI
from psygnal import EmissionInfo
from superqt import ensure_main_thread

REPORTING_AVAILABLE: bool = False
try:
    from fibsem.ui.widgets.autolamella_generate_report_widget import generate_report_dialog
    from fibsem.ui.widgets.autolamella_experiment_task_summary_widget import create_experiment_task_summary_widget
    from fibsem.ui.widgets.autolamella_overview_image_widget import create_overview_image_widget
    REPORTING_AVAILABLE = True
except ImportError as e:
    logging.debug(f"Could not import generate_report from fibsem.applications.autolamella.tools.reporting: {e}")

AUTOLAMELLA_CHECKPOINTS = []
try:
    from fibsem.segmentation.utils import list_available_checkpoints_v2
    AUTOLAMELLA_CHECKPOINTS = list_available_checkpoints_v2()
except ImportError as e:
    logging.debug(f"Could not import list_available_checkpoints from fibsem.segmentation.utils: {e}")
except Exception as e:
    logging.warning(f"Could not retreive checkpoints from huggingface: {e}")


# instructions
INSTRUCTIONS = {
    "NOT_CONNECTED": "Please connect to the microscope (Connection -> Connect to Microscope).",
    "NO_EXPERIMENT": "Please create or load an experiment (File -> Create / Load Experiment)",
    "NO_PROTOCOL": "Please load a protocol (File -> Load Protocol).",
    "NO_LAMELLA": "Please Add Lamella Positions (Experiment -> Add Lamella).",
    "TRENCH_READY": "Trench Positions Selected. Ready to Run Waffle Trench.",
    "UNDERCUT_READY": "Undercut Positions Selected. Ready to Run Waffle Undercut.",
    "LAMELLA_READY": "Lamella Positions Selected. Ready to Run Setup AutoLamella.",
    "AUTOLAMELLA_READY": "Lamella Positions Selected. Ready to Run AutoLamella.",
}


class AutoLamellaUI(AutoLamellaMainUI.Ui_MainWindow, QMainWindow):
    workflow_update_signal = pyqtSignal(dict)
    detection_confirmed_signal = pyqtSignal(bool)
    _workflow_finished_signal = pyqtSignal()

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()

        self.setupUi(self)

        self._protocol_lock = threading.RLock()

        self.label_title.setText(f"AutoLamella v{fibsem.__version__}")
        self.viewer = viewer
        self.viewer.title = f"AutoLamella v{fibsem.__version__}"

        self.experiment: Optional[Experiment] = None
        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.minimap_widget: Optional[FibsemMinimapWidget] = None
        self.spot_burn_widget: Optional[FibsemSpotBurnWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskConfigWidget] = None
        self.det_widget: Optional['FibsemEmbeddedDetectionWidget'] = None
        self.protocol_editor_widget: Optional[AutoLamellaProtocolEditorTabWidget] = None

        # minimap plot widget
        self.minimap_plot_widget = MinimapPlotWidget(self)
        self.minimap_plot_dock = self.viewer.window.add_dock_widget(self.minimap_plot_widget,
                                                                name="Minimap Plot",
                                                                area='left',
                                                                add_vertical_stretch=False,
                                                                tabify=True)

        # add widgets to tabs
        self.tabWidget.insertTab(0, self.system_widget, "Connection")

        self.WAITING_FOR_USER_INTERACTION: bool = False
        self.USER_RESPONSE: bool = False
        self.WAITING_FOR_UI_UPDATE: bool = False
        self._workflow_stop_event: threading.Event = threading.Event()
        self._task_worker_thread: Optional[threading.Thread] = None

        # setup connections
        self.setup_connections()

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        with self._protocol_lock:
            return self.experiment.task_protocol if self.experiment is not None else None

    @property
    def is_workflow_running(self) -> bool:
        return self._task_worker_thread is not None and self._task_worker_thread.is_alive()

    def setup_connections(self):

        # lamella controls
        self.pushButton_add_lamella.clicked.connect(
            lambda: self.add_new_lamella(stage_position=None) # type: ignore
        )
        self.pushButton_remove_lamella.clicked.connect(self.delete_lamella_ui)
        self.pushButton_go_to_lamella.clicked.connect(self.move_to_lamella_position)
        self.comboBox_current_lamella.currentIndexChanged.connect(self.update_lamella_ui)
        self.pushButton_save_position.clicked.connect(self.save_lamella_ui)

        # workflow button group
        self.pushButton_run_setup_autolamella.setVisible(False)

        # system widget
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        # file menu
        self.actionNew_Experiment.triggered.connect(self.create_experiment)
        self.actionLoad_Experiment.triggered.connect(self.load_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.export_protocol_ui)
        # tool menu
        self.actionCryo_Deposition.triggered.connect(self.cryo_deposition)
        self.actionCryo_Deposition.setEnabled(False) # TMP: disable until tested
        self.actionCryo_Deposition.setToolTip("Cryo Deposition is currently disabled via the UI.")
        self.actionOpen_Minimap.triggered.connect(self.open_minimap_widget)
        self.actionGenerate_Report.triggered.connect(self.action_generate_report)
        self.actionGenerate_Overview_Plot.triggered.connect(self.action_generate_overview_plot)

        # self.actionSelectLandingPositions = QAction(text="Select Landing Positions", parent=self)
        # self.actionSelectLandingPositions.triggered.connect(
        #     lambda: self._run_workflow(workflow="select-landing-positions")
        # )
        # self.menuDevelopment.addAction(self.actionSelectLandingPositions)
        # self.actionSelectLandingPositions.setVisible(False)  # TMP: disable until tested

        # task config editor
        self.action_open_protocol_editor = QAction(  # type: ignore
            "Open Protocol Editor",
            parent=self,
            triggered=self._open_protocol_editor,
        )

        self.action_open_experiment_directory = QAction(  # type: ignore
            "Open Experiment Directory",
            parent=self,
            triggered=self._open_experiment_directory,
        )

        # add to menu
        if os.name == "posix":
            self.menuBar().setNativeMenuBar(False) # required for macOS
        self.menuDevelopment.addSeparator()
        self.menuDevelopment.addAction(self.action_open_experiment_directory)

        # reporting
        self.action_open_experiment_workflow_summary = QAction(  # type: ignore
            "Open Workflow Summary",
            parent=self,
            triggered=self._open_experiment_workflow_summary,
        )

        self.menuTools.addSeparator()
        self.menuTools.addAction(self.action_open_protocol_editor)
        self.menuTools.addAction(self.action_open_experiment_workflow_summary)
        self.action_open_experiment_workflow_summary.setVisible(REPORTING_AVAILABLE)

        # development
        self.menuDevelopment.setVisible(False)
        self.actionAdd_Lamella_from_Odemis.setVisible(False)    # TMP: disable until tested
        self.actionSpot_Burn.setVisible(False)                  # TMP: disable until tested
        self.actionRun_Spot_Burn_Workflow.setVisible(False)     # TMP: disable until tested
        self.actionAdd_Lamella_from_Odemis.triggered.connect(self._add_lamella_from_odemis)
        # help menu
        self.actionInformation.triggered.connect(
            lambda: fui.open_information_dialog(self.microscope, self)
        )

        # workflow interaction
        self.pushButton_run_setup_autolamella.clicked.connect(self.run_task_workflow)
        self.pushButton_stop_workflow.setVisible(False)
        self.pushButton_stop_workflow.clicked.connect(self._stop_workflow_thread)
        self.pushButton_yes.clicked.connect(self.push_interaction_button)
        self.pushButton_no.clicked.connect(self.push_interaction_button)

        # signals
        self.detection_confirmed_signal.connect(self.handle_confirmed_detection_signal)
        self.workflow_update_signal.connect(self.handle_workflow_update)
        self._workflow_finished_signal.connect(self._workflow_finished)  # type: ignore

        self.pushButton_stop_workflow.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_add_lamella.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_lamella.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_go_to_lamella.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        # labels and placeholders
        self.lineEdit_experiment_name.setPlaceholderText("No Experiment Loaded")
        self.lineEdit_protocol_name.setPlaceholderText("No Protocol Loaded")
        self.lineEdit_protocol_name.setReadOnly(True)
        self.lineEdit_experiment_name.setReadOnly(True)

        self.task_history_widget = AutoLamellaWorkflowDisplayWidget(experiment=None, parent=self)
        self.scrollArea_lamella_info.setWidget(self.task_history_widget)


        # workflow info
        self.set_current_workflow_message(msg=None, show=False)
        self.label_instructions.setWordWrap(True)
        self.label_workflow_information.setWordWrap(True)

        # refresh ui
        self.update_ui()

        self.label_lamella_objective_position.setText("Objective Position")
        self.doubleSpinBox_lamella_objective_position.setSuffix(" mm")
        self.doubleSpinBox_lamella_objective_position.setDecimals(3)
        self.doubleSpinBox_lamella_objective_position.setSingleStep(0.001)
        self.doubleSpinBox_lamella_objective_position.setRange(-20.0, 20.0)
        self.doubleSpinBox_lamella_objective_position.valueChanged.connect(self.update_lamella_objective_position)
        self.doubleSpinBox_lamella_objective_position.setKeyboardTracking(False)

        self.comboBox_lamella_pose.currentIndexChanged.connect(self._on_lamella_pose_combobox_changed)
        self.pushButton_lamella_set_pose.clicked.connect(self._set_current_position_as_pose)
        self.pushButton_lamella_set_pose.setToolTip("Set the current stage position as the lamella pose position.")

        self.pushButton_lamella_move_to_pose.clicked.connect(self._move_to_lamella_pose)
        self.pushButton_lamella_move_to_pose.setToolTip("Move the stage to the lamella pose position.")
        self.pushButton_lamella_set_pose.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_lamella_move_to_pose.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.label_lamella_pose_position.setWordWrap(True)

        rows = self.gridLayout_7.rowCount()
        self.lamella_defect_widget = AutoLamellaDefectStateWidget(parent=self)
        self.gridLayout_7.addWidget(self.lamella_defect_widget, rows, 0, 1, 2)
        self.lamella_defect_widget.defect_state_changed.connect(self._on_defect_state_changed)

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

        logging.info('-'*80)
        logging.info(f"event: {evt.signal.name} path: {evt.path}, {len(self.experiment.positions)} Positions")
        logging.info('-'*80)

        self.update_lamella_combobox()
        self.update_ui()

    @ensure_main_thread
    def _on_stage_position_updated(self, stage_position: FibsemStagePosition) -> None:
        """Callback for when the stage position is updated."""
        if self.minimap_widget is not None and self.minimap_widget.is_acquiring:
            return
        if self.movement_widget is not None:
            self.movement_widget.update_ui()

        self._update_minimap_data(stage_position=stage_position)
        self._update_lamella_display()

    def _update_lamella_display(self, selected_name: Optional[str] = None) -> None:
        """Update the lamella display in the live fib view."""

        if not cfg.FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED:
            return

        if self.experiment is None:
            return

        if self.image_widget is None:
            return

        if self.image_widget.ib_image is None or self.image_widget.ib_layer is None:
            return

        from fibsem.imaging.tiled import reproject_stage_positions_onto_image2
        from fibsem.ui.napari.utilities import (
            NapariShapeOverlay,
            create_crosshair_shape,
        )
        from fibsem.ui.FibsemMinimapWidget import CROSSHAIR_CONFIG

        if selected_name is None:
            selected_name = self.comboBox_current_lamella.currentText()

        try:
            # image.metadata.image_settings.beam_type = BeamType.ION # WHYYY
            points = reproject_stage_positions_onto_image2(image=self.image_widget.ib_image, 
                                                           positions=self.experiment.get_milling_positions(), 
                                                           bound=True)

            # NOTE: this displayes the previous lamella position when using workflow becasue when the stage position moves, the image is still the previous one
            # so the reprojected position is wrong until a new image is acquired. but this doesn't trigger a new render until the next stage move. so its always 'one behind'
            # should disable until we can fix this properly

            layer_scale = None
            overlays: List[NapariShapeOverlay] = []
            for pt in points:
                saved_lines = create_crosshair_shape(pt, CROSSHAIR_CONFIG["crosshair_size"], layer_scale)

                # Use lime for selected position, cyan for others
                color = CROSSHAIR_CONFIG["colors"]["saved_unselected"]
                if pt.name == selected_name:
                    color = CROSSHAIR_CONFIG["colors"]["saved_selected"]

                # Show position name on crosshair if saved position FOV is disabled
                # label = saved_pos.name if not self.show_saved_positions_fov else ""

                for line, txt in zip(saved_lines, [pt.name, ""]):
                    overlays.append(NapariShapeOverlay(
                        shape=line,
                        color=color,
                        label=txt,
                        shape_type="line"
                    ))

            crosshair_overlays = overlays

            # TODO: use a common function for this?
            # QUERY: also do it for the SEM?
            # Collect all crosshair overlays
            layer_name = CROSSHAIR_CONFIG["layer_name"] 

            if len(crosshair_overlays) == 0:
                logging.info("No crosshair overlays to display.")
                # Remove existing layer if no overlays to display
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
                return

            # Extract data for napari
            crosshair_lines = [overlay.shape for overlay in crosshair_overlays]
            colors = [overlay.color for overlay in crosshair_overlays]
            labels = [overlay.label for overlay in crosshair_overlays]

            # Prepare text properties for labels
            text_properties = {
                "string": labels,
                **CROSSHAIR_CONFIG["text_properties"]
            }
            # text_properties["color"] = colors # displays the text the same color as the line

            # Update or create the napari layer
            if layer_name in self.viewer.layers:
                # Update existing layer
                layer = self.viewer.layers[layer_name]
                layer.data = crosshair_lines
                # Note: edge_color and text updates may not work with all napari versions
                try:
                    layer.edge_color = colors
                    layer.edge_width = CROSSHAIR_CONFIG["line_style"]["edge_width"]
                    layer.face_color = CROSSHAIR_CONFIG["line_style"]["face_color"]
                    layer.text = text_properties
                    layer.visible = True
                    layer.translate = self.image_widget.ib_layer.translate
                except AttributeError:
                    logging.warning("Could not update layer properties directly")
            else:
                # Create new layer
                self.viewer.add_shapes(
                    data=crosshair_lines,
                    name=layer_name,
                    shape_type="line",
                    edge_color=colors,
                    edge_width=CROSSHAIR_CONFIG["line_style"]["edge_width"],
                    face_color=CROSSHAIR_CONFIG["line_style"]["face_color"],
                    scale=layer_scale,
                    text=text_properties,
                    translate=self.image_widget.ib_layer.translate,
                )

        except Exception as e:
            logging.warning(f"Could not update lamella display: {e}")
        finally:
            self.image_widget.restore_active_layer_for_movement()

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
                self.microscope.stage_position_changed.disconnect(self._on_stage_position_updated)
                logging.info("Disconnected previous microscope stage position event subscribers.")
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
            self.microscope.stage_position_changed.connect(self._on_stage_position_updated)

        # Register metadata
        if self.microscope is not None:
            utils._register_metadata(
                microscope=self.microscope,
                application_software="autolamella",
                application_software_version=fibsem.__version__,
                experiment_name=self.experiment.name,
                experiment_method="null",
            )

        # Update UI
        self.update_lamella_combobox()
        self.update_ui()

        # set the experiment tab as active
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab))

    def create_experiment(self) -> None:
        """Create a new experiment using the experiment creation dialog."""
        if self.microscope is None:
            napari.utils.notifications.show_warning("Please connect to microscope first.")
            return

        # Open the experiment creation dialog
        experiment = create_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            napari.utils.notifications.show_info("Experiment creation cancelled.")
            return

        # Disconnect existing event subscribers if there's an existing experiment
        self._disconnect_experiment_events()

        # Assign the experiment
        self.experiment = experiment
        napari.utils.notifications.show_info(f"Experiment '{self.experiment.name}' created successfully.")

        # Setup experiment connections and update UI
        self._setup_experiment_connections()

    def load_experiment(self) -> None:
        """Load an existing experiment using the experiment loading dialog."""
        if self.microscope is None:
            napari.utils.notifications.show_warning("Please connect to microscope first.")
            return

        # Open the experiment loading dialog
        experiment = load_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            napari.utils.notifications.show_info("Experiment loading cancelled.")
            return

        # Disconnect existing event subscribers if there's an existing experiment
        self._disconnect_experiment_events()

        # Assign the experiment
        self.experiment = experiment
        napari.utils.notifications.show_info(
            f"Experiment '{self.experiment.name}' loaded with {len(self.experiment.positions)} lamella."
        )

        # Setup experiment connections and update UI
        self._setup_experiment_connections()

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
                image_settings=self.settings.image, # type: ignore
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                parent=self,
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            # TODO: replace this with MillingTaskWidget to support multi-task configuration
            self.milling_task_config_widget = MillingTaskConfigWidget(microscope=self.microscope, parent=self)
            self.tabWidget.addTab(self.milling_task_config_widget, "Milling")

            # add the detection widget if ml dependencies are available
            if DETECTION_AVAILABLE:
                self.det_widget = FibsemEmbeddedDetectionWidget(parent=self)
                self.tabWidget.addTab(self.det_widget, "Detection")
                self.tabWidget.setTabVisible(self.tabWidget.indexOf(self.det_widget), False)

            # spot burn widget (optional)
            self.spot_burn_widget = FibsemSpotBurnWidget(parent=self)
            self.tabWidget.addTab(self.spot_burn_widget, "Spot Burn")
            self.tabWidget.setTabVisible(self.tabWidget.indexOf(self.spot_burn_widget), False)

            try:
                from fibsem.microscopes.odemis_microscope import OdemisMicroscope
                if isinstance(self.microscope, OdemisMicroscope):
                    logging.info("OdemisMicroscope detected, enabling Odemis specific features.")
                    self.actionAdd_Lamella_from_Odemis.setVisible(True)
            except Exception as e:
                logging.debug(f"OdemisMicroscope not available: {e}")

            self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)
        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.det_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.det_widget))
                self.det_widget.deleteLater()
                self.det_widget = None
            if self.spot_burn_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.spot_burn_widget))
                self.spot_burn_widget.deleteLater()
                self.spot_burn_widget = None
            if self.milling_task_config_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.milling_task_config_widget))
                self.milling_task_config_widget.deleteLater()
                self.milling_task_config_widget = None
            if self.movement_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.movement_widget))
                self.movement_widget.deleteLater()
                self.movement_widget = None
            if self.image_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.image_widget))
                self.image_widget.clear_viewer()
                self.image_widget.acquisition_progress_signal.disconnect(self.handle_acquisition_update)
                self.image_widget.deleteLater()
                self.image_widget = None

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
            napari.utils.notifications.show_warning("Reporting tools are not available.")
            return

        dialog = create_overview_image_widget(experiment=self.experiment, parent=self)
        dialog.exec_()

        return

    def _open_experiment_workflow_summary(self):
        """Open the experiment task workflow summary dialog."""

        if self.experiment is None:
            napari.utils.notifications.show_warning("Please load an experiment first.")
            return

        if not REPORTING_AVAILABLE:
            napari.utils.notifications.show_warning("Reporting tools are not available.")
            return

        dialog = create_experiment_task_summary_widget(experiment=self.experiment, parent=self)
        dialog.exec_()

#### PROTOCOL EDITOR

    def _open_protocol_editor(self):
        """Open the protocol editor dialog."""

        if self.protocol_editor_widget is not None:
            napari.utils.notifications.show_info("Protocol editor is already open.")
            self.protocol_editor_widget.viewer.window.activate()
            return
        show_protocol_editor(parent=self)

    def _open_experiment_directory(self) -> None:
        """Open the experiment directory in the system file explorer."""
        if self.experiment is None or self.experiment.path is None:
            napari.utils.notifications.show_warning(
                "Please load an experiment first... [No Experiment Loaded]"
            )
            return

        experiment_path = os.fspath(self.experiment.path)
        if not os.path.isdir(experiment_path):
            napari.utils.notifications.show_error(
                f"Experiment directory not found: {experiment_path}"
            )
            return

        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", experiment_path])
            elif os.name == "nt":
                os.startfile(experiment_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", experiment_path])
        except Exception:
            logging.exception("Failed to open experiment directory.")
            napari.utils.notifications.show_error("Failed to open experiment directory.")

#### MINIMAP

    def open_minimap_widget(self):
        if self.microscope is None:
            napari.utils.notifications.show_warning(
                "Please connect to a microscope first... [No Microscope Connected]"
            )
            return

        if self.movement_widget is None:
            napari.utils.notifications.show_warning(
                "Please connect to a microscope first... [No Movement Widget]"
            )
            return

        if self.experiment is None:
            napari.utils.notifications.show_warning(
                "Please load an experiment first... [No Experiment Loaded]"
            )
            return

        self.viewer_minimap = napari.Viewer(ndisplay=2, title="AutoLamella Minimap")
        self.minimap_widget = FibsemMinimapWidget(
            viewer=self.viewer_minimap,
            parent=self
        )
        self.viewer_minimap.window.add_dock_widget(
            widget=self.minimap_widget,
            area="right",
            add_vertical_stretch=True,
            name="AutoLamella Minimap",
        )

        self.viewer_minimap.window.activate()

    def _update_minimap_data(self,
                             stage_position: Optional[FibsemStagePosition] = None, 
                             selected_name: Optional[str] = None) -> None:
        if self.microscope is None:
            return
        if self.experiment is None:
            return

        if not cfg.FEATURE_MINIMAP_PLOT_WIDGET_ENABLED:
            return

        try:

            image: Optional[FibsemImage] = None
            if self.minimap_plot_widget.image is None:
                ms = self.microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
                image = FibsemImage.generate_blank_image(resolution=(2048, 2048), hfw=4000e-6)
                image.metadata.microscope_state = ms                # type: ignore
                image.metadata.system = self.microscope.system      # type: ignore
                self.minimap_plot_widget.image = image


            beam_type = self.minimap_plot_widget.image.metadata.beam_type # type: ignore
            fov = self.microscope.get_field_of_view(beam_type=beam_type)


            # Set the data (delay redraw until all data updated...)
            if selected_name is not None:
                self.minimap_plot_widget.selected_name = selected_name
            self.minimap_plot_widget.lamella_positions = self.experiment.get_milling_positions()
            if self.minimap_plot_widget.grid_positions is None and cfg.FEATURE_DISPLAY_GRID_CENTER_MARKER:
                self.minimap_plot_widget.grid_positions = [g.position for g in self.microscope._stage.holder.grids.values()]
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

    def run_task_workflow(self):

        if self.microscope is None or self.experiment is None or self.protocol is None or self.experiment.task_protocol is None:
            napari.utils.notifications.show_warning(
                "Please connect to a microscope and load an experiment first... [No Microscope or Experiment]"
            )
            return
        
        if self.experiment.positions == []:
            napari.utils.notifications.show_warning(
                "Please add at least one lamella position to the experiment first... [No Lamella Positions]"
            )
            return

        if self.milling_task_config_widget is None:
            napari.utils.notifications.show_warning(
                "Milling task configuration not available... [No Milling Configuration]"
            )
            return

        from fibsem.applications.autolamella.ui.autolamella_task_selection_dialog import open_task_selection_dialog

        lamella_names = [p.name for p in self.experiment.positions]
        tasks_names = [t.name for t in self.experiment.task_protocol.workflow_config.tasks]
        workflow_accepted, selected_tasks, selected_lamella = open_task_selection_dialog(lamella_names=lamella_names, 
                                                                                         task_names=tasks_names, 
                                                                                         parent_ui=self)
        if not workflow_accepted:
            logging.info("User cancelled workflow selection.")
            return
        if not selected_tasks or not selected_lamella:
            logging.info("No tasks selected or no lamella selected.")
            return
        logging.info(f"Selected tasks: {selected_tasks}, for lamella: {selected_lamella}")

        self.pushButton_stop_workflow.setVisible(True)
        self.pushButton_run_setup_autolamella.setEnabled(False)

        # clear milling task config
        self.milling_task_config_widget.clear()

        # TODO: disable other widgets??

        # Start acquisition thread
        self._task_worker_thread = threading.Thread(
            target=self._run_tasks_worker,
            args=(selected_tasks, selected_lamella),
            daemon=True
        )
        self._task_worker_thread.start()

    def _run_tasks_worker(self, task_names: List[str], lamella_names: Optional[List[str]] = None) -> None:
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
            napari.utils.notifications.show_info(f"Starting {len(task_names)} tasks...")
            run_tasks(microscope=self.microscope,
                      experiment=self.experiment,
                      task_names=task_names,
                      required_lamella=lamella_names,
                      parent_ui=self)
        except Exception as e:
            logging.error(f"Error during running tasks: {e}")

        finally:
            self._task_worker_thread = None
            self._workflow_finished_signal.emit()  # type: ignore
            napari.utils.notifications.show_info("Tasks completed.")

    def stop_task_workflow(self):
        if not self.is_workflow_running:
            return
        self._stop_workflow_thread()

#### UI UPDATES

    def update_ui(self):
        """Update the ui based on the current state of the application."""

        # state flags
        is_experiment_loaded = bool(self.experiment is not None)
        is_microscope_connected = bool(self.microscope is not None)
        is_protocol_loaded = bool(self.settings is not None) and self.protocol is not None
        has_lamella = bool(self.experiment.positions) if is_experiment_loaded else False
        is_experiment_ready = is_experiment_loaded and is_protocol_loaded

        self.action_open_experiment_directory.setEnabled(is_experiment_loaded)

        # force order: connect -> experiment -> protocol
        self.tabWidget.setTabVisible(self.tabWidget.indexOf(self.tab), is_microscope_connected)
        self.actionNew_Experiment.setVisible(is_microscope_connected)
        self.actionLoad_Experiment.setVisible(is_microscope_connected)
        self.actionInformation.setVisible(is_microscope_connected)
        if self.det_widget is not None:
            idx = self.tabWidget.indexOf(self.det_widget)
            self.tabWidget.setTabVisible(idx, False)  # hide detection tab for now

        # workflow
        # self.update_protocol_ui()

        # setup experiment -> connect to microscope -> select lamella -> run autolamella

        # experiment loaded
        # file menu
        self.actionLoad_Protocol.setVisible(is_experiment_loaded)
        self.actionSave_Protocol.setVisible(is_protocol_loaded)
        self.actionUpdateMilling_Protocol.setVisible(is_protocol_loaded)
        # tool menu
        self.actionCryo_Deposition.setVisible(True)
        self.actionOpen_Minimap.setVisible(is_protocol_loaded)
        self.actionLoad_Minimap_Image.setVisible(is_protocol_loaded)
        self.actionLoad_Positions.setVisible(is_protocol_loaded)
        # help menu
        self.actionGenerate_Report.setVisible(is_experiment_ready and REPORTING_AVAILABLE)
        self.actionGenerate_Overview_Plot.setVisible(is_experiment_ready and REPORTING_AVAILABLE)

        # labels
        self.lineEdit_experiment_name.setToolTip("No Experiment Loaded")
        if is_experiment_loaded and self.experiment is not None:
            self.lineEdit_experiment_name.setText(f"{self.experiment.name}")
            self.lineEdit_experiment_name.setToolTip(f"Experiment Directory: {self.experiment.path}")
            self.comboBox_current_lamella.setVisible(has_lamella)

            self.task_history_widget.set_experiment(self.experiment)
            self.groupBox_lamella.setMinimumHeight(min(200, 150 + 10 * len(self.experiment.positions)))

        if self.protocol is not None:
            self.lineEdit_protocol_name.setText(f"{self.protocol.name}")

        # buttons
        self.pushButton_add_lamella.setEnabled(is_experiment_ready)
        self.pushButton_remove_lamella.setEnabled(has_lamella)
        self.pushButton_save_position.setEnabled(has_lamella)
        self.pushButton_go_to_lamella.setEnabled(has_lamella)

        # set visible if protocol loaded
        self.pushButton_add_lamella.setVisible(is_experiment_ready)
        self.pushButton_remove_lamella.setVisible(is_experiment_ready)
        self.pushButton_save_position.setVisible(is_experiment_ready)
        self.pushButton_go_to_lamella.setVisible(is_experiment_ready)
        self.label_current_lamella_header.setVisible(is_experiment_ready)
        self.comboBox_current_lamella.setVisible(is_experiment_ready)
        self.groupBox_setup.setVisible(is_experiment_ready)
        self.groupBox_lamella.setVisible(has_lamella)
        self.groupBox_selected_lamella.setVisible(has_lamella)
        
        # workflow buttons
        self.label_run_autolamella_info.setVisible(has_lamella)
        self.pushButton_run_setup_autolamella.setVisible(is_experiment_ready)
        self.pushButton_run_setup_autolamella.setEnabled(has_lamella)
        self.pushButton_run_setup_autolamella.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

        # disable lamella controls while workflow is running
        self.groupBox_setup.setEnabled(not self.is_workflow_running)
        self.groupBox_lamella.setEnabled(not self.is_workflow_running)
        self.groupBox_selected_lamella.setEnabled(not self.is_workflow_running)

        # Current Lamella Status
        if has_lamella and self.experiment is not None:
            self.update_lamella_ui()
            estimated_time = self.experiment.estimate_remaining_time()
            txt = f"Estimated time remaining: {utils.format_duration(estimated_time)}"
            self.label_run_autolamella_info.setText(txt)
            self.pushButton_run_setup_autolamella.setToolTip("Run the AutoLamella workflow on the selected lamella positions.")
        else:
            self.pushButton_run_setup_autolamella.setToolTip("Please add at least one lamella position to run the AutoLamella workflow.")

        if self.is_workflow_running:
            self.pushButton_run_setup_autolamella.setEnabled(False)
            return

        if not is_microscope_connected:
            self.set_instructions_msg(INSTRUCTIONS["NOT_CONNECTED"])
        elif not is_experiment_loaded:
            self.set_instructions_msg(INSTRUCTIONS["NO_EXPERIMENT"])
        elif not is_protocol_loaded:
            self.set_instructions_msg(INSTRUCTIONS["NO_PROTOCOL"])
        elif not has_lamella:
            self.set_instructions_msg(INSTRUCTIONS["NO_LAMELLA"])
        elif has_lamella:
            self.set_instructions_msg(INSTRUCTIONS["AUTOLAMELLA_READY"])

    def update_protocol_ui(self):
        """Update the protocol editor ui based on the current experiment protocol."""
        if self.protocol_editor_widget is not None and self.experiment is not None:
            self.protocol_editor_widget.workflow_widget.set_experiment(self.experiment) 
            self.protocol_editor_widget.task_widget._initialise_widgets()

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
        if self.experiment is None or self.experiment.positions == []:
            return

        # detail combobox
        idx = self.comboBox_current_lamella.currentIndex()
        self.comboBox_current_lamella.blockSignals(True)
        self.comboBox_current_lamella.clear()
        self.comboBox_current_lamella.addItems(
            [lamella.name for lamella in self.experiment.positions]
        )
        if idx != -1 and self.experiment.positions and idx < len(self.experiment.positions):
            self.comboBox_current_lamella.setCurrentIndex(idx)
        if latest and self.experiment.positions:
            self.comboBox_current_lamella.setCurrentIndex(
                len(self.experiment.positions) - 1
            )
        self.comboBox_current_lamella.blockSignals(False)

    def update_lamella_ui(self):
        # set the info for the current selected lamella
        if self.experiment is None or self.experiment.positions == []:
            return

        if self.protocol is None:
            return

        if self.is_workflow_running:
            return
        
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return

        lamella: Lamella = self.experiment.positions[idx]
        logging.info(f"Updating Lamella UI for {lamella.status_info}")

        # buttons
        self.pushButton_save_position.setText("Update Position")
        self.pushButton_save_position.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_save_position.setEnabled(True)

        # set objective position (show as mm)
        obj_controls_enabled = lamella.objective_position is not None
        if obj_controls_enabled:
            self.doubleSpinBox_lamella_objective_position.blockSignals(True)
            self.doubleSpinBox_lamella_objective_position.setValue(lamella.objective_position * 1e3)
            self.doubleSpinBox_lamella_objective_position.blockSignals(False)
        self.label_lamella_objective_position.setVisible(obj_controls_enabled)
        self.doubleSpinBox_lamella_objective_position.setVisible(obj_controls_enabled)

        # set lamella pose display
        if lamella.poses:

            self.comboBox_lamella_pose.blockSignals(True)
            current_text = self.comboBox_lamella_pose.currentText()
            self.comboBox_lamella_pose.clear()
            self.comboBox_lamella_pose.addItems(list(lamella.poses.keys()))

            # restore
            if current_text in lamella.poses:
                self.comboBox_lamella_pose.setCurrentText(current_text)
            else:
                self.comboBox_lamella_pose.setCurrentIndex(0)

            current_pose = self.comboBox_lamella_pose.currentText()
            pose = lamella.poses.get(current_pose, None)

            txt = "Pose: Unknown"
            if pose is not None and pose.stage_position is not None:
                txt = f"Pose: {pose.stage_position.pretty}"
            self.label_lamella_pose_position.setText(txt)
            self.comboBox_lamella_pose.blockSignals(False)

        # hide pose controls if no poses
        enable_pose_controls = bool(lamella.poses) and cfg.FEATURE_POSE_CONTROLS_ENABLED
        self.label_lamella_pose.setVisible(enable_pose_controls)
        self.comboBox_lamella_pose.setVisible(enable_pose_controls)
        self.label_lamella_pose_position.setVisible(enable_pose_controls)
        self.pushButton_lamella_move_to_pose.setVisible(enable_pose_controls)
        self.pushButton_lamella_set_pose.setVisible(enable_pose_controls)

        # defect state
        self.lamella_defect_widget.set_defect_state(lamella.defect)

        self._update_minimap_data(selected_name=lamella.name)
        self._update_lamella_display(selected_name=lamella.name)

    def _on_defect_state_changed(self, defect: 'DefectState'):
        if self.experiment is None:
            return
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return
        lamella = self.experiment.positions[idx]
        lamella.defect = defect
        self.experiment.save()
        self.update_ui()

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

        idx = self.comboBox_current_lamella.currentIndex()
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

        from fibsem.applications.autolamella.compat.odemis import _add_features_from_odemis
        stage_positions = _add_features_from_odemis(filename)

        for pos in stage_positions:
            self.add_new_lamella(pos)

    def add_new_lamella(self,
                        stage_position: Optional[FibsemStagePosition] = None,
                        name: Optional[str] = None,
                        objective_position: Optional[float] = None) -> Lamella:
        """Add a lamella to the experiment.
        Args:
            stage_position: The stage position of the lamella. If None, the current stage position is used.
            name: The name of the lamella. If None, a default name will be generated.
            objective_position: The objective position of the lamella. If None, the 'focused' objective position is used.
        Returns:
            lamella: The created lamella.
        """
        if self.experiment is None:
            raise ValueError("No experiment loaded. Please load an experiment first.")
        if self.protocol is None:
            raise ValueError("No protocol loaded. Please load a protocol first.")
        if self.microscope is None:
            raise ValueError("No microscope connected. Please connect a microscope first.")

        # get microscope state
        microscope_state = self.microscope.get_microscope_state()  
        if stage_position is not None: 
            microscope_state.stage_position = deepcopy(stage_position)

        # create the lamella
        self.experiment.add_new_lamella(microscope_state=microscope_state, 
                                        task_config=self.experiment.task_protocol.task_config, 
                                        name=name)
        lamella = self.experiment.positions[-1]

        # if the objective position is not provided, use the 'focus' position from the microscope
        if self.microscope.fm is not None:
            lamella.fluorescence_pose = deepcopy(microscope_state)
            if objective_position is None:
                objective_position = self.microscope.fm.objective.focus_position
            lamella.objective_position = objective_position

        self.experiment.save()
        self.update_lamella_combobox(latest=True)
        self.update_ui()

        return lamella

    def delete_lamella_ui(self):
        """Handle the removal of a lamella from the experiment."""

        idx = self.comboBox_current_lamella.currentIndex()
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

    def fail_lamella_ui(self):
        """Toggle the defect state of the selected lamella."""
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return

        if self.experiment is None or self.experiment.positions == []:
            return

        # get the current state
        lamella = self.experiment.positions[idx]
        is_failure = lamella.defect.has_defect

        # if marking as failure, get user reason for failure
        if not is_failure:
            msg, ret = fui.open_text_input_dialog(
                msg="Enter defect reason:",
                title=f"Mark Lamella {lamella.name} as defect?",
                default="",
                parent=self,
            )

            if ret is False:
                logging.debug(f"User cancelled marking lamella {lamella.name} as failure.")
                return

            lamella.defect.set_defect(msg)
        else:
            lamella.defect.clear()
        self.experiment.save()
        self.update_ui()

    def save_lamella_ui(self):
        """Update the stage position of the selected lamella to the current stage position."""

        if self.microscope is None:
            return
        if self.protocol is None:
            return
        if self.experiment is None or self.experiment.positions == []:
            return

        # toggle between saving position and marking as ready
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            logging.warning("No lamella is selected, cannot save.")
            return

        lamella: Lamella = self.experiment.positions[idx]
        current_position = self.microscope.get_stage_position()

        # message box to confirm
        ret = QMessageBox.question(
            self,
            "Save Position Confirmation",
            f"Save current position as Lamella {lamella.name} position?\n\n"
            f"Current Stage Position: {current_position.pretty}\n"
            f"Saved Stage Position: {lamella.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        lamella.milling_pose = deepcopy(self.microscope.get_microscope_state())

        self.update_lamella_combobox()
        self.update_ui()
        self.experiment.save()
        self.experiment.positions.events.changed.emit()

    def _on_lamella_pose_combobox_changed(self):
        """Update the pose position label when the pose combobox is changed."""

        lamella: Lamella
        if self.experiment is None or self.experiment.positions == []:
            return
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return
        lamella = self.experiment.positions[idx]
        pose_name = self.comboBox_lamella_pose.currentText()
        if pose_name not in lamella.poses:
            return
        pose = lamella.poses[pose_name]
        if pose.stage_position is None:
            return
        self.label_lamella_pose_position.setText(f"Pose: {pose.stage_position.pretty}")

    def _set_current_position_as_pose(self):
        """Set the current stage position as the selected pose for the current lamella."""

        if self.microscope is None:
            napari.utils.notifications.show_warning("No microscope connected.")
            return
        if self.experiment is None or self.experiment.positions == []:
            napari.utils.notifications.show_warning("No lamella available.")
            return
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            napari.utils.notifications.show_warning("No lamella selected.")
            return
        lamella: Lamella = self.experiment.positions[idx]
        pose_name = self.comboBox_lamella_pose.currentText()
        if pose_name == "":
            napari.utils.notifications.show_warning("No pose selected.")
            return
        state = self.microscope.get_microscope_state()

        if state is None or state.stage_position is None:
            napari.utils.notifications.show_warning("Failed to get microscope state.")
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

        lamella.poses[pose_name] = state
        self.experiment.save()
        self.label_lamella_pose_position.setText(f"{state.stage_position.pretty}")
        napari.utils.notifications.show_info(f"Set current position as pose '{pose_name}' for {lamella.name}.")

    def _move_to_lamella_pose(self):
        """Move the stage to the selected pose for the current lamella."""

        if self.microscope is None:
            napari.utils.notifications.show_warning("No microscope connected.")
            return
        if self.experiment is None or self.experiment.positions == []:
            napari.utils.notifications.show_warning("No lamella available.")
            return
        if self.movement_widget is None:
            napari.utils.notifications.show_warning("No movement widget available")
            return
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            napari.utils.notifications.show_warning("No lamella selected.")
            return
        lamella: Lamella = self.experiment.positions[idx]
        pose_name = self.comboBox_lamella_pose.currentText()
        if pose_name == "":
            napari.utils.notifications.show_warning("No pose selected.")
            return
        if pose_name not in lamella.poses:
            napari.utils.notifications.show_warning(f"Pose '{pose_name}' not found for {lamella.name}.")
            return
        pose = lamella.poses[pose_name]
        if pose.stage_position is None:
            napari.utils.notifications.show_warning(f"Pose '{pose_name}' has no stage position.")
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
        napari.utils.notifications.show_info(f"Moved to pose '{pose_name}' for {lamella.name}.")

    def update_lamella_objective_position(self, value: float):
        """Update the objective position of the current lamella."""

        # get current lamella
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1 or self.experiment is None:
            napari.utils.notifications.show_warning("No lamella selected.")
            return

        lamella = self.experiment.positions[idx]
        # convert from mm to m
        lamella.objective_position = value * 1e-3
        self.experiment.save()

    def get_selected_lamella(self) -> Optional[Lamella]:
        """Get the currently selected lamella from the combobox.

        Returns:
            The selected lamella, or None if no experiment, no positions, or invalid selection.
        """
        if self.experiment is None:
            return None

        if not self.experiment.positions:
            return None

        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1 or idx >= len(self.experiment.positions):
            return None

        return self.experiment.positions[idx]

#### PROTOCOL
    def load_protocol(self):
        """Load a protocol into the current experiment using the protocol loading dialog."""
        if self.microscope is None:
            napari.utils.notifications.show_warning("Please connect to microscope first.")
            return

        if self.experiment is None:
            napari.utils.notifications.show_warning("Please load an experiment first.")
            return

        # Open the protocol loading dialog
        protocol = load_task_protocol_dialog(experiment=self.experiment, parent=self)

        if protocol is None:
            napari.utils.notifications.show_info("Protocol loading cancelled.")
            return

        # assign protocol to experiment
        self.experiment.task_protocol = protocol

        napari.utils.notifications.show_info(
            f"Protocol '{protocol.name}' loaded successfully with {len(protocol.task_config)} tasks."
        )

        # Update UI
        self.update_ui()
        self.update_protocol_ui()

    def export_protocol_ui(self):
        """Export the current protocol to file."""

        if self.experiment is None or self.experiment.task_protocol is None:
            napari.utils.notifications.show_info("No protocol loaded.")
            return

        protocol_path = fui.open_save_file_dialog(
            msg="Select a protocol file",
            path=str(cfg.TASK_PROTOCOL_PATH),
            _filter="*.yaml",
            parent=self,
        )

        if protocol_path == "":
            napari.utils.notifications.show_info("No path selected")
            return

        self.experiment.task_protocol.save(protocol_path)
        napari.utils.notifications.show_info(
            f"Saved Protocol to {os.path.basename(protocol_path)}"
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
        self.pushButton_yes.setText(pos)
        self.pushButton_no.setText(neg)

        # enable buttons
        self.pushButton_yes.setEnabled(pos is not None)
        self.pushButton_yes.setVisible(pos is not None)
        self.pushButton_no.setEnabled(neg is not None)
        self.pushButton_no.setVisible(neg is not None)

        if pos == "Run Milling":
            self.pushButton_yes.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        else:
            self.pushButton_yes.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

    def set_current_workflow_message(self, msg: Optional[str] = None, show: bool = True):
        """Set the current workflow information message"""
        if msg is not None:
            self.label_workflow_information.setText(msg)
        self.label_workflow_information.setVisible(show)

    def push_interaction_button(self):
        """Handle the user interaction with the workflow."""

        # positve / negative response
        self.USER_RESPONSE = bool(self.sender() == self.pushButton_yes)
        self.WAITING_FOR_USER_INTERACTION = False

    def handle_acquisition_update(self, ddict: dict) -> None:
        if ddict.get("finished", False):
            self.update_lamella_ui()

    def handle_confirmed_detection_signal(self):
        # TODO: this seem very redundant if we just use the signal directly
        if self.det_widget is not None:
            self.det_widget.confirm_button_clicked()

    def _stop_workflow_thread(self):
        self._workflow_stop_event.set()
        self.milling_task_config_widget.milling_widget.stop_milling() # stop milling if running
        napari.utils.notifications.show_error("Abort requested by user.")

    def _workflow_finished(self):
        """Handle the completion of the workflow."""
        logging.info("Workflow finished.")
        if self.image_widget is None:
            return
        if self.microscope is None:
            return
        if self.experiment is None or self.protocol is None:
            return

        self._workflow_stop_event.clear()
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab))
        self.pushButton_stop_workflow.setVisible(False)

        # clear milling task config
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()

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

        # set electron image as active layer
        self.image_widget.restore_active_layer_for_movement()

        self.set_current_workflow_message(msg=None, show=False)

    def handle_workflow_update(self, info: dict) -> None:
        """Update the UI with the given information, ready for user interaction"""

        if self.image_widget is None:
            raise ValueError("No image widget available. Please create an image widget first.")

        if self.milling_task_config_widget is None:
            raise ValueError("No milling task config widget available. Please create a milling task config widget first.")

        # update the image viewer
        sem_image = info.get("sem_image", None)
        if sem_image is not None:
            self.image_widget.eb_image = sem_image
            self.image_widget._on_acquire(sem_image)
            self.image_widget._update_ui_from_images(beam_type=BeamType.ELECTRON, set_ui_from_image=True)

        fib_image = info.get("fib_image", None)
        if fib_image is not None:
            self.image_widget.ib_image = fib_image
            self.image_widget._on_acquire(fib_image)
            self.image_widget._update_ui_from_images(beam_type=BeamType.ION, set_ui_from_image=True)

        # what?
        enable_milling = info.get("milling_enabled", None)
        if enable_milling is not None:
            self.tabWidget.setCurrentWidget(self.milling_task_config_widget)
            self.milling_task_config_widget.milling_widget.pushButton_run_milling.setVisible(False)

        # update milling stages
        detections = info.get("det", None)
        if self.det_widget is not None and detections is not None:
            self.det_widget.set_detected_features(detections)
            det_idx = self.tabWidget.indexOf(self.det_widget)
            if det_idx != -1:
                self.tabWidget.setTabVisible(det_idx, True)
                self.tabWidget.setCurrentIndex(det_idx)

        # update the alignment area
        alignment_area = info.get("alignment_area", None)
        if isinstance(alignment_area, FibsemRectangle):
            self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.image_widget))
            self.image_widget.toggle_alignment_area(alignment_area)
        if alignment_area == "clear":
            self.image_widget.clear_alignment_area()

        # spot_burn
        spot_burn = info.get("spot_burn", None)
        if spot_burn:
            self.set_spot_burn_widget_active(True)

        milling_config = info.get("milling_config", None)
        if milling_config is not None:
            self.milling_task_config_widget.update_from_settings(milling_config)
            self.milling_task_config_widget.setEnabled(True)
            self.tabWidget.setCurrentWidget(self.milling_task_config_widget)
        if info.get("clear_milling_config", False):
            self.milling_task_config_widget.clear()
            self.milling_task_config_widget.setEnabled(False)
            self.tabWidget.setCurrentWidget(self.milling_task_config_widget)

        # no specific interaction, just update the ui
        if (detections is None and
            enable_milling is None and
            alignment_area is None and
            spot_burn is None):
            self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab)) # NOTE: this is what causes the view to reset to this tab for milling

        # instruction message
        self.set_instructions_msg(info["msg"], info.get("pos", None), info.get("neg", None))
        self.set_current_workflow_message(info.get("workflow_info", None))

        self.WAITING_FOR_UI_UPDATE = False

        # status_msg = info.get("status", None)
        # if status_msg is not None:
        #     logging.info(f"Workflow Status Update: {status_msg}")
        #     task_name = status_msg.get("task_name", "Unknown Task")
        #     lamella_name = status_msg.get("lamella_name", "Unknown Lamella")
        #     # step_name = status_msg.get("step_name", "Unknown Step")
        #     # step_status = status_msg.get("step_status", "Unknown Status")
        #     current_lamella_index = status_msg.get("current_lamella_index", None)
        #     current_task_index = status_msg.get("current_task_index", None)
        #     total_lamellae = status_msg.get("total_lamellas", None)
        #     total_tasks = status_msg.get("total_tasks", None)
        #     txt = f"Workflow Running: {task_name} on {lamella_name}"
        #     if (current_lamella_index is not None and total_lamellae is not None):
        #         txt += f" | Lamella {current_lamella_index + 1}/{total_lamellae}"
        #     if (current_task_index is not None and total_tasks is not None):
        #         txt += f" | Task {current_task_index + 1}/{total_tasks}"
        #     logging.info(txt)

def main():
    autolamella_ui = AutoLamellaUI(viewer=napari.Viewer())
    autolamella_ui.viewer.window.add_dock_widget(
        widget=autolamella_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"AutoLamella v{fibsem.__version__}",
    )
    napari.run()

if __name__ == "__main__":
    main()
