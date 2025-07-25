import sys

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass
import logging
import os
import threading
from collections import Counter
from copy import deepcopy
from datetime import datetime
from typing import List, Optional

import napari
import napari.utils.notifications
import fibsem
from fibsem import utils
from fibsem.constants import DEGREE_SYMBOL
from fibsem.imaging.spot import run_spot_burn
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import get_milling_stages, get_protocol_from_stages
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeSettings,
    Point,
)
from fibsem.ui import (
    DETECTION_AVAILABLE,
    FibsemCryoDepositionWidget,
    FibsemImageSettingsWidget,
    FibsemMillingWidget,
    FibsemMinimapWidget,
    FibsemMovementWidget,
    FibsemSystemSetupWidget,
    FibsemSpotBurnWidget,
    stylesheets,
)
from fibsem.ui import utils as fui
from fibsem.ui.napari.patterns import remove_all_napari_shapes_layers
from napari.qt.threading import thread_worker
from PyQt5.QtCore import pyqtSignal
from qtpy import QtWidgets

if DETECTION_AVAILABLE: # ml dependencies are option, so we need to check if they are available
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI

from fibsem.applications import autolamella
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.protocol.validation import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    NOTCH_KEY,
    TRENCH_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaMethod,
    AutoLamellaProtocol,
    AutoLamellaStage,
    Experiment,
    Lamella,
    LamellaState,
    create_new_lamella,
    get_autolamella_method,
)
from fibsem.applications.autolamella.ui.qt import AutoLamellaUI as AutoLamellaMainUI
from fibsem.applications.autolamella.ui.AutoLamellaWorkflowDialog import (
    display_lamella_info,
    display_selected_lamella_info,
    open_workflow_dialog,
)
from fibsem.applications.autolamella.ui.tooltips import TOOLTIPS
from fibsem.applications.autolamella.ui.utils import setup_experiment_ui_v2

REPORTING_AVAILABLE: bool = False
try:
    from fibsem.applications.autolamella.tools.reporting import generate_report, save_final_overview_image
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

CONFIGURATION = {
    "TABS_ID": {
        0: "Connection",
        1: "Experiment",
        2: "Protocol",
        3: "Image",
        4: "Movement",
        5: "Milling",
        6: "Detection",
        7: "Manipulator",
    },
    "SHOW_INDIVIDUAL_STAGES": True,
}

# invert the dictionary
CONFIGURATION["TABS"] = {v: k for k, v in CONFIGURATION["TABS_ID"].items()}

# instructions
INSTRUCTIONS = {
    "NOT_CONNECTED": "Please connect to the microscope (System -> Connect to Microscope).",
    "NO_EXPERIMENT": "Please create or load an experiment (File -> Create / Load Experiment)",
    "NO_PROTOCOL": "Please load a protocol (File -> Load Protocol).",
    "NO_LAMELLA": "Please Add Lamella Positions (Experiment -> Add Lamella).",
    "TRENCH_READY": "Trench Positions Selected. Ready to Run Waffle Trench.",
    "UNDERCUT_READY": "Undercut Positions Selected. Ready to Run Waffle Undercut.",
    "LAMELLA_READY": "Lamella Positions Selected. Ready to Run Setup AutoLamella.",
    "AUTOLAMELLA_READY": "Lamella Positions Selected. Ready to Run AutoLamella.",
}

def _find_matching_position(position: FibsemStagePosition, experiment: Experiment) -> int:
    """Find the matching position in the experiment."""
    lamella_names = [lamella.name for lamella in experiment.positions]
    if position.name not in lamella_names:
        raise ValueError(f"Position {position.name} not found in experiment positions.")
    idx = lamella_names.index(position.name)
    return idx

# preparation stages
PREPARTION_WORKFLOW_STAGES = [
    AutoLamellaStage.Created,
    AutoLamellaStage.PositionReady,
]


# TODO: when autolamella is closed, also close the minimap...
class AutoLamellaUI(AutoLamellaMainUI.Ui_MainWindow, QtWidgets.QMainWindow):
    workflow_update_signal = pyqtSignal(dict)
    detection_confirmed_signal = pyqtSignal(bool)
    update_experiment_signal = pyqtSignal(Experiment)
    run_milling_signal = pyqtSignal()
    sync_positions_to_minimap_signal = pyqtSignal(list)
    lamella_created_signal = pyqtSignal(Lamella)

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()

        self.setupUi(self)

        self._protocol_lock = threading.RLock()

        self.label_title.setText(f"AutoLamella v{fibsem.__version__}")
        self.viewer = viewer
        self.viewer.title = f"AutoLamella v{fibsem.__version__}"


        self.IS_MICROSCOPE_UI_LOADED: bool = False
        self.is_protocol_loaded: bool = False
        self.UPDATING_PROTOCOL_UI: bool = False

        self.experiment: Optional[Experiment] = None
        self.worker = None
        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None
        self.protocol: Optional[AutoLamellaProtocol] = None

        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.tabWidget.insertTab(
            CONFIGURATION["TABS"]["Connection"], self.system_widget, "Connection"
        )

        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.milling_widget: Optional[FibsemMillingWidget] = None
        self.minimap_widget: Optional[FibsemMinimapWidget] = None
        self.spot_burn_widget: Optional[FibsemSpotBurnWidget] = None

        self.WAITING_FOR_USER_INTERACTION: bool = False
        self.USER_RESPONSE: bool = False
        self.WAITING_FOR_UI_UPDATE: bool = False
        self.is_milling: bool = False
        self.WORKFLOW_IS_RUNNING: bool = False
        self.STOP_WORKFLOW: bool = False

        # setup connections
        self.setup_connections()

    def setup_connections(self):

        # lamella controls
        self.pushButton_add_lamella.clicked.connect(
            lambda: self.add_new_lamella(stage_position=None)
        )
        self.pushButton_remove_lamella.clicked.connect(self.remove_lamella_ui)
        self.pushButton_go_to_lamella.clicked.connect(self.move_to_lamella_position)
        self.comboBox_current_lamella.currentIndexChanged.connect(
            self.update_lamella_ui
        )
        self.pushButton_save_position.clicked.connect(self.save_lamella_ui)
        self.pushButton_fail_lamella.clicked.connect(self.fail_lamella_ui)
        self.pushButton_revert_stage.clicked.connect(self.revert_stage)

        # workflow controls
        self.pushButton_run_waffle_trench.clicked.connect(
            lambda: self._run_workflow(workflow="trench")
        )
        self.pushButton_run_waffle_undercut.clicked.connect(
            lambda: self._run_workflow(workflow="undercut")
        )
        self.pushButton_run_setup_autolamella.clicked.connect(
            lambda: self._run_workflow(workflow="autolamella")
        )
        self.pushButton_setup_autoliftout.clicked.connect(
            lambda: self._run_workflow(workflow="setup-liftout")
        )
        self.pushButton_run_autoliftout.clicked.connect(
            lambda: self._run_workflow(workflow="autoliftout")
        )
        self.pushButton_run_serial_liftout_landing.clicked.connect(
            lambda: self._run_workflow(workflow="serial-liftout-landing")
        )

        # workflow button group
        self.workflow_buttons = [
            self.pushButton_run_waffle_trench,
            self.pushButton_run_waffle_undercut,
            self.pushButton_run_setup_autolamella,
            self.pushButton_setup_autoliftout,
            self.pushButton_run_autoliftout,
            self.pushButton_run_serial_liftout_landing,
        ]
        for button in self.workflow_buttons:
            button.setVisible(False)

        self.pushButton_update_protocol.clicked.connect(self.export_protocol_ui)

        # system widget
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        # file menu
        self.actionNew_Experiment.triggered.connect(self.setup_experiment)
        self.actionLoad_Experiment.triggered.connect(self.setup_experiment)
        self.actionLoad_Protocol.triggered.connect(self.load_protocol)
        self.actionSave_Protocol.triggered.connect(self.export_protocol_ui)
        self.actionUpdateMilling_Protocol.triggered.connect(self.update_experiment_milling_protocol)
        # tool menu
        self.actionCryo_Deposition.triggered.connect(self.cryo_deposition)
        self.actionCryo_Deposition.setEnabled(False) # TMP: disable until tested
        self.actionCryo_Deposition.setToolTip("Cryo Deposition is currently disabled via the UI.")
        self.actionOpen_Minimap.triggered.connect(self.open_minimap_widget)
        self.actionGenerate_Report.triggered.connect(self.action_generate_report)
        self.actionGenerate_Overview_Plot.triggered.connect(self.action_generate_overview_plot)

        # add protocol editor
        self.actionOpenProtocolEditor = QtWidgets.QAction("Open Protocol Editor", self,
            triggered=self.open_protocol_editor,
        )
        self.menuDevelopment.addAction(self.actionOpenProtocolEditor)

        self.actionSelectLandingPositions = QtWidgets.QAction("Select Landing Positions", self)
        self.actionSelectLandingPositions.triggered.connect(
            lambda: self._run_workflow(workflow="select-landing-positions")
        )
        self.menuDevelopment.addAction(self.actionSelectLandingPositions)

        self.actionOpenProtocolEditor.setVisible(True)  # TMP: disable until tested

        # development
        self.menuDevelopment.setVisible(False)
        self.actionAdd_Lamella_from_Odemis.setVisible(False)    # TMP: disable until tested
        self.actionSpot_Burn.setVisible(False)                  # TMP: disable until tested
        self.actionRun_Spot_Burn_Workflow.setVisible(False)     # TMP: disable until tested
        self.actionAdd_Lamella_from_Odemis.triggered.connect(self._add_lamella_from_odemis)
        self.actionSpot_Burn.triggered.connect(lambda: self.set_spot_burn_widget_active(True))
        self.actionRun_Spot_Burn_Workflow.triggered.connect(self.run_spot_burns_workflow)
        # help menu
        self.actionInformation.triggered.connect(
            lambda: fui.open_information_dialog(self.microscope, self)
        )

        # workflow interaction
        self.pushButton_stop_workflow.setVisible(False)
        self.pushButton_stop_workflow.clicked.connect(self._stop_workflow_thread)
        self.pushButton_yes.clicked.connect(self.push_interaction_button)
        self.pushButton_no.clicked.connect(self.push_interaction_button)

        # signals
        self.detection_confirmed_signal.connect(self.handle_confirmed_detection_signal)
        self.update_experiment_signal.connect(self.hande_update_experiment_signal)
        self.workflow_update_signal.connect(self.handle_workflow_update)
        self.run_milling_signal.connect(self.run_milling)

        self.pushButton_stop_workflow.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_add_lamella.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_lamella.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_go_to_lamella.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        # comboboxes
        self.comboBox_method.addItems([m.name for m in AutoLamellaMethod]) # TODO: restrict available methods
        self.comboBox_method.currentIndexChanged.connect(self.change_protocol_method)
        self.comboBox_ml_checkpoint.addItems(AUTOLAMELLA_CHECKPOINTS)

        self.comboBox_options_liftout_joining_method.addItems(cfg.LIFTOUT_JOIN_METHODS)
        self.comboBox_options_landing_joining_method.addItems(cfg.LIFTOUT_LANDING_JOIN_METHODS)

        self.AVAILABLE_POSITIONS = utils._get_positions()
        self.comboBox_options_trench_start_position.addItems(self.AVAILABLE_POSITIONS)
        self.comboBox_options_landing_start_position.addItems(self.AVAILABLE_POSITIONS)

        # workflow info
        self.set_current_workflow_message(msg=None, show=False)

        # tooltips # TODO: migrate to config/yaml
        self._add_tooltips()

        # refresh ui
        self.update_ui()

#### SPOT_BURN
    def run_spot_burns_workflow(self):
        """Run the spot burn workflow for all positions in the experiment."""

        self._spot_worker = self._spot_burn_workflow(self.microscope, self.protocol, self.experiment, self)
        self._spot_worker.finished.connect(self._spot_burn_workflow_finished)
        self._spot_worker.errored.connect(self._spot_burn_workflow_errored)
        self._spot_worker.start()

    @thread_worker
    def _spot_burn_workflow(
        self,
        microscope: FibsemMicroscope,
        protocol: AutoLamellaProtocol,
        experiment: Experiment,
        parent_ui: "AutoLamellaUI",
    ):
        """Thread worker to run the spot burn workflow."""
        from fibsem.applications.autolamella.workflows.runners import run_spot_burn_workflow
        run_spot_burn_workflow(microscope=microscope,
                               protocol=protocol,
                               experiment=experiment,
                               parent_ui=parent_ui)

    def _spot_burn_workflow_finished(self):
        logging.info("Spot burn workflow finished.")
        self.set_spot_burn_widget_active(active=False)

    def _spot_burn_workflow_errored(self):
        logging.error("Spot burn workflow failed.")
        self.set_spot_burn_widget_active(active=False)

    def run_spot_burns(self):
        # DEPRECATED
        """Run the spot burning tool"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Spot Burn Parameters")
        dialog.setGeometry(100, 100, 300, 200)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(QtWidgets.QLabel("Spot Burn Parameters"))
        layout.addWidget(QtWidgets.QLabel("Exposure Time (s)"))
        exposure_time_input = QtWidgets.QDoubleSpinBox(dialog)
        exposure_time_input.setRange(0, 10000)
        exposure_time_input.setValue(10)
        exposure_time_input.setSuffix(" s")

        layout.addWidget(exposure_time_input)
        layout.addWidget(QtWidgets.QLabel("Milling Current (pA)"))
        milling_current_input = QtWidgets.QDoubleSpinBox(dialog)
        milling_current_input.setRange(0, 500)
        milling_current_input.setValue(60)
        milling_current_input.setSuffix(" pA")
        layout.addWidget(milling_current_input)

        # add standard buttons
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, dialog)
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setText("Run Spot Burn")
        button_box.setContentsMargins(0, 0, 0, 0)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        ret = dialog.exec_()

        if ret == QtWidgets.QDialog.Rejected:
            logging.debug("Spot burn cancelled by user.")
            return

        exposure_time = exposure_time_input.value()
        milling_current = milling_current_input.value() * 1e-12
        logging.info(f"Spot burn parameters: {exposure_time} ms, {milling_current} pA")
        
        # get the points layer
        if "Points" not in self.viewer.layers:
            napari.utils.notifications.show_warning("No points layer found. Requires 'Points' layer.")
            return

        pt_layer = self.viewer.layers["Points"]

        # check if there is a points layer, and that it has points in it
        if pt_layer is None:
            napari.utils.notifications.show_warning("No points layer found.")
            return
    
        if len(pt_layer.data) == 0:
            napari.utils.notifications.show_warning("No points selected.")
            return


        # get the fib image parameters
        layer_translated = pt_layer.data - self.image_widget.ib_layer.translate
        image_shape = self.image_widget.ib_layer.data.shape

        # convert to relative image coordinates (0-1)
        coordinates = [Point(x=pt[1]/image_shape[1], y=pt[0] / image_shape[0]) for pt in layer_translated]

        # TODO: create the points layer, set add mode
        self.spot_worker = self._spot_burn_worker(coordinates=coordinates, 
                                                  exposure_time=exposure_time, 
                                                  milling_current=milling_current)
        self.spot_worker.finished.connect(self._spot_burn_finished)
        self.spot_worker.errored.connect(self._spot_burn_errored)
        self.spot_worker.start() # TODO: display a progress bar / indicator?

    @thread_worker
    def _spot_burn_worker(self, coordinates: List[Point], exposure_time: float, milling_current: float):
        """Run the spot burn worker."""
        run_spot_burn(microscope=self.microscope,
                    coordinates=coordinates, 
                    exposure_time=exposure_time, 
                    milling_current=milling_current)

    def _spot_burn_finished(self):
        napari.utils.notifications.show_info("Spot burn finished.")

    def _spot_burn_errored(self):
        napari.utils.notifications.show_error("Spot burn failed.")

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

##########
    def _add_tooltips(self) -> None:

        # protocol tooltips
        self.checkBox_align_at_milling_current.setToolTip(TOOLTIPS["protocol"]["alignment_at_milling_current"])
        self.checkBox_align_use_fiducial.setToolTip(TOOLTIPS["protocol"]["use_fiducial"])
        self.checkBox_take_final_reference_images.setToolTip(TOOLTIPS["protocol"]["take_final_reference_images"])
        # self.checkBox_take_final_high_quality_reference.setToolTip(TOOLTIPS["protocol"]["take_final_high_quality_reference"])
        self.checkBox_turn_beams_off.setToolTip(TOOLTIPS["protocol"]["turn_beams_off"])
        self.checkBox_use_microexpansion.setToolTip(TOOLTIPS["protocol"]["use_microexpansion"])
        self.checkBox_use_notch.setToolTip(TOOLTIPS["protocol"]["use_notch"])
        self.beamshift_attempts.setToolTip(TOOLTIPS["protocol"]["alignment_attempts"])
        self.doubleSpinBox_lamella_tilt_angle.setToolTip(TOOLTIPS["protocol"]["milling_angle"])
        self.doubleSpinBox_undercut_tilt.setToolTip(TOOLTIPS["protocol"]["undercut_tilt_angle"])
        self.comboBox_ml_checkpoint.setToolTip(TOOLTIPS["protocol"]["checkpoint"])

        # supervision
        self.checkBox_setup.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_supervise_mill_rough.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_supervise_mill_polishing.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_trench.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_undercut.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_supervise_liftout.setToolTip(TOOLTIPS["protocol"]["supervision"])
        self.checkBox_supervise_landing.setToolTip(TOOLTIPS["protocol"]["supervision"])

    def change_protocol_method(self):
        """Change the protocol method and refresh the UI."""
        if self.protocol is None or self.experiment is None:
            return
        method = get_autolamella_method(self.comboBox_method.currentText())
        self.protocol.method = method
        self.experiment.method = method
        self.update_protocol_ui()

    def update_protocol_ui(self):
        if not self.is_protocol_loaded or self.UPDATING_PROTOCOL_UI or self.protocol is None:
            return

        self.UPDATING_PROTOCOL_UI = True

        # TODO: auto update protocol when editing, don't require button press
        with self._protocol_lock:

            protocol: AutoLamellaProtocol = self.protocol
            method = protocol.method

            # protocol name and method
            self.lineEdit_name.setText(protocol.name)
            self.comboBox_method.setCurrentText(method.name)

            # options
            self.checkBox_turn_beams_off.setChecked(protocol.options.turn_beams_off)
            self.checkBox_align_use_fiducial.setChecked(protocol.options.use_fiducial)

            self.beamshift_attempts.setValue(protocol.options.alignment_attempts)
            self.checkBox_align_at_milling_current.setChecked(protocol.options.alignment_at_milling_current)

            self.checkBox_take_final_reference_images.setChecked(protocol.options.take_final_reference_images)
            self.checkBox_take_final_high_quality_reference.setChecked(
                protocol.tmp.get("high_quality_image", {}).get("enabled", False)
            )

            # lamella
            self.doubleSpinBox_lamella_tilt_angle.setValue(protocol.options.milling_angle)
            self.checkBox_use_microexpansion.setChecked(protocol.options.use_microexpansion)
            self.checkBox_use_notch.setChecked(protocol.options.use_notch)

            # supervision
            supervision = protocol.supervision
            if method is not AutoLamellaMethod.TRENCH: ## TODO: SIMPLIFY
                self.checkBox_setup.setChecked(supervision[AutoLamellaStage.SetupLamella])
                self.checkBox_supervise_mill_rough.setChecked(supervision[AutoLamellaStage.MillRough])
                self.checkBox_supervise_mill_polishing.setChecked(supervision[AutoLamellaStage.MillPolishing])

            # TRENCH METHOD ONLY (waffle, serial-liftout)
            if method.is_trench:
                # supervision
                self.checkBox_trench.setChecked(supervision.get(AutoLamellaStage.MillTrench, False))
                self.checkBox_undercut.setChecked(supervision.get(AutoLamellaStage.MillUndercut, False))

                # machine learning
                self.comboBox_ml_checkpoint.setCurrentText(protocol.options.checkpoint)

                # undercut
                self.doubleSpinBox_undercut_tilt.setValue(protocol.options.undercut_tilt_angle)

            self.checkBox_trench.setVisible(method.is_trench)
            self.checkBox_undercut.setVisible(method.is_trench)

            self.label_ml_header.setVisible(method.is_trench)
            self.label_ml_checkpoint.setVisible(method.is_trench)
            self.comboBox_ml_checkpoint.setVisible(method.is_trench)

            self.doubleSpinBox_undercut_tilt.setVisible(method.is_trench)
            self.label_protocol_undercut_tilt_angle.setVisible(method.is_trench)

            # autoliftout components
            is_liftout_method = method.is_liftout
            is_classic_liftout_method = method == AutoLamellaMethod.LIFTOUT
            self.label_options_trench_start_position.setVisible(is_liftout_method)
            self.label_options_liftout_joining_method.setVisible(
                is_liftout_method and is_classic_liftout_method
            )
            self.label_options_landing_start_position.setVisible(is_liftout_method)
            self.label_options_landing_joining_method.setVisible(
                is_liftout_method and is_classic_liftout_method
            )
            self.comboBox_options_trench_start_position.setVisible(is_liftout_method)
            self.comboBox_options_liftout_joining_method.setVisible(
                is_liftout_method and is_classic_liftout_method
            )
            self.comboBox_options_landing_start_position.setVisible(is_liftout_method)
            self.comboBox_options_landing_joining_method.setVisible(
                is_liftout_method and is_classic_liftout_method
            )
            self.checkBox_supervise_liftout.setVisible(is_liftout_method)
            self.checkBox_supervise_landing.setVisible(is_liftout_method)

            # disable some options for serial liftout
            self.checkBox_use_microexpansion.setVisible(not is_liftout_method)
            self.checkBox_use_notch.setVisible(not is_liftout_method)

            if is_liftout_method:
                self.comboBox_options_liftout_joining_method.setCurrentText(
                    protocol.tmp.get("liftout_joining_method", "None")
                )
                self.comboBox_options_landing_joining_method.setCurrentText(
                    protocol.tmp.get("landing_joining_method", "Weld")
                )

                self.comboBox_options_trench_start_position.setCurrentText(
                    protocol.tmp.get("trench_start_position", self.AVAILABLE_POSITIONS[0])
                )
                self.comboBox_options_landing_start_position.setCurrentText(
                    protocol.tmp.get("landing_start_position", self.AVAILABLE_POSITIONS[0])
                )

                # supervision
                self.checkBox_supervise_liftout.setChecked(supervision[AutoLamellaStage.LiftoutLamella])
                self.checkBox_supervise_landing.setChecked(supervision[AutoLamellaStage.LandLamella])

        self.UPDATING_PROTOCOL_UI = False

    def export_protocol_ui(self):
        
        if self.protocol is None:
            return         
        if not self.is_protocol_loaded:
            return
        
        with self._protocol_lock:
            self.protocol.name = self.lineEdit_name.text()
            self.protocol.method = get_autolamella_method(self.comboBox_method.currentText().lower())

            # options
            self.protocol.options.turn_beams_off = self.checkBox_turn_beams_off.isChecked()
            self.protocol.options.use_fiducial = (self.checkBox_align_use_fiducial.isChecked())
            self.protocol.options.alignment_attempts = int(self.beamshift_attempts.value())
            self.protocol.options.alignment_at_milling_current = self.checkBox_align_at_milling_current.isChecked()
            self.protocol.options.take_final_reference_images = self.checkBox_take_final_reference_images.isChecked()
            self.protocol.tmp.get("high_quality_image", {})["enabled"] = self.checkBox_take_final_high_quality_reference.isChecked()

            self.protocol.options.milling_angle = self.doubleSpinBox_lamella_tilt_angle.value()
            self.protocol.options.use_microexpansion = self.checkBox_use_microexpansion.isChecked()
            self.protocol.options.use_notch = self.checkBox_use_notch.isChecked()

            # supervision
            self.protocol.supervision[AutoLamellaStage.SetupLamella] = self.checkBox_setup.isChecked()
            self.protocol.supervision[AutoLamellaStage.MillRough] = self.checkBox_supervise_mill_rough.isChecked()
            self.protocol.supervision[AutoLamellaStage.MillPolishing] = self.checkBox_supervise_mill_polishing.isChecked()

            method = self.protocol.method

            if method.is_trench:
                # supervision
                self.protocol.supervision[AutoLamellaStage.MillTrench] = self.checkBox_trench.isChecked()
                self.protocol.supervision[AutoLamellaStage.MillUndercut] = self.checkBox_undercut.isChecked()

                # machine learning
                self.protocol.options.checkpoint = self.comboBox_ml_checkpoint.currentText()

                # undercut
                self.protocol.options.undercut_tilt_angle = self.doubleSpinBox_undercut_tilt.value()

            if method.is_liftout:
                # supervision
                self.protocol.supervision[AutoLamellaStage.LiftoutLamella] = self.checkBox_supervise_liftout.isChecked()
                self.protocol.supervision[AutoLamellaStage.LandLamella] = self.checkBox_supervise_landing.isChecked()

                # joining methods
                self.protocol.tmp["liftout_joining_method"] = (
                    self.comboBox_options_liftout_joining_method.currentText()
                )
                self.protocol.tmp["landing_joining_method"] = (
                    self.comboBox_options_landing_joining_method.currentText()
                )

                # start positions
                self.protocol.tmp["trench_start_position"] = (
                    self.comboBox_options_trench_start_position.currentText()
                )
                self.protocol.tmp["landing_start_position"] = (
                    self.comboBox_options_landing_start_position.currentText()
                )

            if self.sender() == self.actionSave_Protocol:
                path = fui.open_save_file_dialog(
                    msg="Export protocol",
                    path=cfg.PROTOCOL_PATH,
                    _filter="*yaml",
                    parent=self,
                )
                self.protocol.save(path)

                logging.info("Protocol saved to file")
            elif self.sender() == self.pushButton_update_protocol:
                logging.info("Protocol updated")

            # auto save copy to experiment folder
            if self.experiment:
                self.protocol.save(os.path.join(self.experiment.path, "protocol.yaml"))

        napari.utils.notifications.show_info("Protocol updated.")
        if self.WORKFLOW_IS_RUNNING:
            return # don't update the ui
        self.update_ui()

    def update_experiment_milling_protocol(self):
        """Update the milling protocol based on the currently selected lamella."""

        if self.experiment is None:
            napari.utils.notifications.show_warning("No experiment loaded.")
            return
        if self.protocol is None:
            napari.utils.notifications.show_warning("No protocol loaded.")
            return

        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            napari.utils.notifications.show_warning("No lamella selected.")
            return
        lamella: Lamella = self.experiment.positions[idx]

        # message box confirmation
        ret = fui.message_box_ui(
            title="Update Protocol?",
            text=f"Update base milling protocol using {lamella.name}?",
            parent=self,
        )
        if ret is False:
            logging.debug("User cancelled protocol update.")
            return

        # convert protocol to milling workflow (stages)
        milling_protocol = deepcopy(lamella.protocol)
        milling_workflow = {k: get_milling_stages(k, milling_protocol) for k in milling_protocol}

        # reset the pattern points to 0,0
        for k, stages in milling_workflow.items():
            if k == FIDUCIAL_KEY:
                stages[0].imaging.path = None
                continue
            for stage in stages:
                stage.pattern.point = Point(x=0, y=0)
                stage.imaging.path = None

        # update the protocol with the new stages
        with self._protocol_lock:
            self.protocol.milling = deepcopy(milling_workflow)

        # save protocol to file
        self.export_protocol_ui()

        napari.utils.notifications.show_info(f"Protocol updated using {lamella.name}.")

    def get_protocol(self) -> AutoLamellaProtocol:
        """Thread-safe getter for the protocol."""
        with self._protocol_lock:
            return self.protocol

    def setup_experiment(self):
        new_experiment = bool(self.sender() is self.actionNew_Experiment)
        experiment = setup_experiment_ui_v2(self, new_experiment=new_experiment)

        if experiment is None:
            napari.utils.notifications.show_info("Experiment not loaded.")
            return

        self.experiment = experiment
        napari.utils.notifications.show_info(
            f"Experiment {self.experiment.name} loaded."
        )
        if self.settings is not None:
            self.settings.image.path = self.experiment.path

        # register metadata
        from fibsem.applications import autolamella  # NB: microscope needs to be connected beforehand

        utils._register_metadata(
            microscope=self.microscope,
            application_software="autolamella",
            application_software_version=fibsem.__version__,
            experiment_name=self.experiment.name,
            experiment_method="null",
        )  # TODO: add method to experiment

        # automatically re-load protocol if available
        if not new_experiment and self.settings is not None:
            # try to load protocol from file
            PROTOCOL_PATH = os.path.join(self.experiment.path, "protocol.yaml")
            if os.path.exists(PROTOCOL_PATH):
                self.protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)
                self.protocol.configuration = deepcopy(self.settings)
                self.experiment.method = self.protocol.method
                self.is_protocol_loaded = True
                self.update_protocol_ui()

        self.update_lamella_combobox()
        self.update_ui()

    ##################################################################

    # TODO: move this to system wideget??
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

        if self.microscope is not None and not self.IS_MICROSCOPE_UI_LOADED:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,
                viewer=self.viewer,
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                viewer=self.viewer,
                parent=self,
            )
            self.milling_widget = FibsemMillingWidget(
                microscope=self.microscope,
                viewer=self.viewer,
                parent=self,
            )

            # add widgets to tabs
            self.tabWidget.insertTab(
                CONFIGURATION["TABS"]["Image"], self.image_widget, "Image"
            )
            self.tabWidget.insertTab(
                CONFIGURATION["TABS"]["Movement"], self.movement_widget, "Movement"
            )
            self.tabWidget.insertTab(
                CONFIGURATION["TABS"]["Milling"], self.milling_widget, "Milling"
            )

            # add the detection widget if ml dependencies are available
            self.det_widget = None
            if DETECTION_AVAILABLE:
                self.det_widget = FibsemEmbeddedDetectionUI(parent=self)
                self.tabWidget.insertTab(
                    CONFIGURATION["TABS"]["Detection"], self.det_widget, "Detection"
                )
                self.tabWidget.setTabVisible(CONFIGURATION["TABS"]["Detection"], False)

            # spot burn widget (optional)
            self.spot_burn_widget = FibsemSpotBurnWidget(parent=self)
            self.tabWidget.insertTab(-1, self.spot_burn_widget, "Spot Burn")
            self.tabWidget.setTabVisible(self.tabWidget.indexOf(self.spot_burn_widget), False)

            try: 
                from fibsem.microscopes.odemis_microscope import OdemisMicroscope
                if isinstance(self.microscope, OdemisMicroscope):
                    logging.info("OdemisMicroscope detected, enabling Odemis specific features.")
                    self.actionAdd_Lamella_from_Odemis.setVisible(True)
                    self.actionRun_Spot_Burn_Workflow.setVisible(True)
                else:
                    logging.debug("OdemisMicroscope not detected, disabling Odemis specific features.")
            except Exception as e:
                logging.debug(f"OdemisMicroscope not available: {e}")

            self.IS_MICROSCOPE_UI_LOADED = True
            self.milling_widget.milling_position_changed.connect(
                self._update_milling_position
            )
            self.milling_widget.milling_progress_signal.connect(self.handle_milling_update)
            self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)
        else:
            if self.image_widget is None:
                return

            # remove tabs
            self.tabWidget.removeTab(CONFIGURATION["TABS"]["Detection"])
            self.tabWidget.removeTab(CONFIGURATION["TABS"]["Milling"])
            self.tabWidget.removeTab(CONFIGURATION["TABS"]["Movement"])
            self.tabWidget.removeTab(CONFIGURATION["TABS"]["Image"])
            self.tabWidget.removeTab(CONFIGURATION["TABS"]["Protocol"])

            self.image_widget.clear_viewer()
            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            if self.det_widget is not None:
                self.det_widget.deleteLater()

            self.IS_MICROSCOPE_UI_LOADED = False


#### REPORT GENERATION
    def action_generate_report(self) -> None:

        # get user to select the output filename
        filename = fui.open_save_file_dialog(
            msg="Save Report",
            path=os.path.join(self.experiment.path, f"{self.experiment.name}.pdf"),
            _filter="*.pdf",
            parent=self,
        )
        if filename == "":
            return

        # threaded report generation
        self.report_worker = self.report_gen_worker(deepcopy(self.experiment), filename)
        self.report_worker.finished.connect(self._report_gen_finished)
        self.report_worker.errored.connect(self._report_gen_error)
        self.report_worker.start() # TODO: display a progress bar / indicator?

    def _report_gen_error(self):
        napari.utils.notifications.show_error("Report generation failed.")

    def _report_gen_finished(self):
        napari.utils.notifications.show_info("Report generated successfully.")

    @thread_worker
    def report_gen_worker(self, experiment: Experiment, filename: str) -> None:
        # generate the report
        generate_report(experiment=experiment,
                        output_filename=filename, 
                        encoding="cp1252" if os.name == "nt" else "utf-8")
        return

    def action_generate_overview_plot(self) -> None:

        # get overview image
        image_filename = fui.open_existing_file_dialog(
            msg="Select Overview Image",
            path=self.experiment.path,
            _filter='Image Files (*.tif *.tiff)',
            parent=self,
        )

        if image_filename == "":
            return
        image = FibsemImage.load(image_filename)
        
        # get user to select the output filename
        filename = fui.open_save_file_dialog(
            msg="Save Report",
            path=os.path.join(self.experiment.path, "final-overview-image.png"),
            _filter="*.png",
            parent=self,
        )
        if filename == "":
            return

        # threaded overview generation
        self.overview_worker = self.overview_gen_worker(deepcopy(self.experiment), image, filename)
        self.overview_worker.finished.connect(self._overview_gen_finished)
        self.overview_worker.errored.connect(self._overview_gen_error)
        self.overview_worker.start()

    def _overview_gen_error(self):
        napari.utils.notifications.show_error("Overview generation failed.")

    def _overview_gen_finished(self):
        napari.utils.notifications.show_info("Overview generated successfully.")

    @thread_worker
    def overview_gen_worker(self, experiment: Experiment, image: FibsemImage, filename: str) -> None:

        # generate the overview plot
        save_final_overview_image(exp=experiment, 
                                  image=image, 
                                  output_path=filename)


#### PROTOCOL EDITOR
    def open_protocol_editor(self):
        from fibsem.ui.FibsemMillingStageEditorWidget import show_protocol_editor

        self.protocol_viewer = napari.Viewer(title="AutoLamella Protocol Editor")
        self.protocol_editor_widget = show_protocol_editor(
            self.protocol_viewer,
            microscope=self.microscope,
            experiment=self.experiment,
            protocol=self.protocol,
            parent=self,
        )

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
        
        # update image path
        self.settings.image.path = self.experiment.path
        self.image_widget.image_settings.path = self.experiment.path

        self.viewer_minimap = napari.Viewer(ndisplay=2)
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

        self.sync_experiment_positions_to_minimap()

        self.minimap_widget.stage_position_added_signal.connect(
            self.add_position_from_minimap
        )
        self.minimap_widget.stage_position_updated_signal.connect(
            self.update_position_from_minimap
        )
        self.minimap_widget.stage_position_removed_signal.connect(
            self.remove_position_from_minimap
        )
        napari.run(max_loop_level=2)
        self.viewer_minimap.window.activate()

    def add_position_from_minimap(self, position: FibsemStagePosition):
        """Add the position to the experiment when added in the minimap."""
        lamella = self.add_new_lamella(position)
        self.lamella_created_signal.emit(lamella)

    def update_position_from_minimap(self, position: FibsemStagePosition):
        """Update the position in the experiment when updated in the minimap."""
        idx = _find_matching_position(position, self.experiment)
        if idx == -1:
            logging.warning(f"Position {position.name} not found in experiment.")
            return

        self.experiment.positions[idx].state.microscope_state.stage_position = position
        self.experiment.save()
        self.update_ui() # TODO: convert to signals

    def remove_position_from_minimap(self, position: FibsemStagePosition):
        """Remove the corresponding position from the experiment when removed from the minimap."""
        idx = _find_matching_position(position, self.experiment)
        
        # check if not found
        if idx == -1:
            logging.warning(f"Position {position.name} not found in experiment.")
            return

        self.experiment.positions.pop(idx)
        self.experiment.save()
        self.update_lamella_combobox()
        self.update_ui() # TODO: convert to signals

    def sync_experiment_positions_to_minimap(self):
        """Sync the current experiment positions to the minimap."""

        # TODO: replace this manual syncing with a shared data model
        if self.minimap_widget is None:
            logging.debug("Minimap widget not loaded.")
            return

        if self.experiment is None:
            logging.warning("No experiment loaded")
            return
        
        # get the updated positions
        positions = []
        for i, p in enumerate(self.experiment.positions):
            position = deepcopy(p.state.microscope_state.stage_position)
            position.name = p.name
            positions.append(position)

        self.sync_positions_to_minimap_signal.emit(positions)


##### LAMELLA CONTROLS

    def update_ui(self):
        """Update the ui based on the current state of the application."""

        # state flags
        is_experiment_loaded = bool(self.experiment is not None)
        is_microscope_connected = bool(self.microscope is not None)
        is_protocol_loaded = bool(self.settings is not None) and self.is_protocol_loaded
        has_lamella = bool(self.experiment.positions) if is_experiment_loaded else False
        is_experiment_ready = is_experiment_loaded and is_protocol_loaded

        # force order: connect -> experiment -> protocol
        self.tabWidget.setTabVisible(
            CONFIGURATION["TABS"]["Experiment"], is_microscope_connected
        )
        self.tabWidget.setTabVisible(
            CONFIGURATION["TABS"]["Protocol"], is_protocol_loaded
        )
        self.actionNew_Experiment.setVisible(is_microscope_connected)
        self.actionLoad_Experiment.setVisible(is_microscope_connected)
        self.actionInformation.setVisible(is_microscope_connected)

        # workflow

        # setup experiment -> connect to microscope -> select lamella -> run autolamella
        self.pushButton_fail_lamella.setVisible(has_lamella)
        self.pushButton_revert_stage.setVisible(has_lamella)
        self.comboBox_lamella_history.setVisible(has_lamella)
        self.pushButton_lamella_landing_selected.setVisible(has_lamella)

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
        if is_experiment_loaded:
            self.label_experiment_name.setText(f"Experiment: {self.experiment.name}")
            self.comboBox_current_lamella.setVisible(has_lamella)

            # display lamella info as grid
            if is_protocol_loaded:
                display_lamella_info(grid_layout=self.gridLayout_lamella_info, 
                                     positions=self.experiment.positions, 
                                     method=self.protocol.method)
                # set minimum hiehgt
                self.groupBox_lamella.setMinimumHeight(min(150, 
                                                           100 + 10*len(self.experiment.positions)))

        if is_protocol_loaded:
            method = self.protocol.method
            self.label_protocol_name.setText(
                f"Protocol: {self.protocol.name} ({method.name.title()} Method)"
            )

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

        # disable lamella controls while workflow is running
        self.groupBox_setup.setEnabled(not self.WORKFLOW_IS_RUNNING)
        self.groupBox_lamella.setEnabled(not self.WORKFLOW_IS_RUNNING)
        self.groupBox_selected_lamella.setEnabled(not self.WORKFLOW_IS_RUNNING)

        # workflow buttons
        if is_experiment_ready:

            method = self.protocol.method
            is_trench_method = method.is_trench
            is_liftout_method = method.is_liftout
            is_serial_liftout_method = method == AutoLamellaMethod.SERIAL_LIFTOUT


            # check the status of each lamella
            workflow_state_counter = Counter([p.state.stage.name for p in self.experiment.positions])
            ready_for_trench = workflow_state_counter[AutoLamellaStage.PositionReady.name] > 0
            ready_for_undercut = workflow_state_counter[AutoLamellaStage.MillTrench.name] > 0
            undercut_finished = workflow_state_counter[AutoLamellaStage.MillUndercut.name] > 0
            ready_for_landing = workflow_state_counter[AutoLamellaStage.LiftoutLamella.name] > 0
            has_landed = workflow_state_counter[AutoLamellaStage.LandLamella.name] > 0
            ready_for_setup_lamella = workflow_state_counter[AutoLamellaStage.PositionReady.name] > 0
            ready_for_rough = (workflow_state_counter[AutoLamellaStage.SetupLamella.name] > 0)
            ready_for_setup_polishing = workflow_state_counter[AutoLamellaStage.MillRough.name] > 0
            ready_for_polish = workflow_state_counter[AutoLamellaStage.SetupPolishing.name] > 0
            ready_for_autolamella = (ready_for_rough or 
                                        ready_for_setup_polishing or
                                        ready_for_polish or
                                        has_landed or
                                        undercut_finished)

            # flags to show buttons
            show_undercut = is_trench_method and method != "autolamella-trench"

            # flags to enable workflows
            enable_trench = is_trench_method and ready_for_trench
            enable_undercut = is_trench_method and ready_for_undercut
            enable_liftout = is_liftout_method and (
                ready_for_trench 
                or ready_for_undercut 
                or undercut_finished
            )
            enable_landing = is_liftout_method and ready_for_landing
            enable_lamella = ready_for_setup_lamella
            enable_autolamella = ready_for_autolamella

            # if any of the stages are ready, enable the autolamella button
            enable_full_autolamella = (
                enable_lamella
                or enable_autolamella
                or enable_trench
                or enable_undercut
            )

            # trench
            self.pushButton_run_waffle_trench.setVisible(is_trench_method)
            self.pushButton_run_waffle_trench.setEnabled(enable_trench)
            self.pushButton_run_waffle_undercut.setVisible(show_undercut)
            self.pushButton_run_waffle_undercut.setEnabled(enable_undercut)
            # liftout
            self.pushButton_setup_autoliftout.setVisible(False) # disable liftout
            self.pushButton_run_autoliftout.setVisible(is_liftout_method)
            self.pushButton_run_serial_liftout_landing.setVisible(is_serial_liftout_method)
            self.pushButton_run_autoliftout.setEnabled(enable_liftout)
            self.pushButton_run_serial_liftout_landing.setEnabled(enable_landing)
            # autolamella
            self.pushButton_run_setup_autolamella.setVisible(True)
            self.pushButton_run_setup_autolamella.setEnabled(enable_full_autolamella)
            self.label_run_autolamella_info.setVisible(enable_full_autolamella)

            self.pushButton_run_waffle_trench.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if enable_trench
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )
            self.pushButton_run_waffle_undercut.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if enable_undercut
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )
            self.pushButton_run_setup_autolamella.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if enable_full_autolamella
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )
            # liftout
            self.pushButton_setup_autoliftout.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if is_liftout_method
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )
            self.pushButton_run_autoliftout.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if enable_liftout
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )
            self.pushButton_run_serial_liftout_landing.setStyleSheet(
                stylesheets.GREEN_PUSHBUTTON_STYLE
                if enable_landing
                else stylesheets.DISABLED_PUSHBUTTON_STYLE
            )

            # global button visibility configuration
            show_individual_workflows = CONFIGURATION["SHOW_INDIVIDUAL_STAGES"]
            self.pushButton_run_waffle_trench.setVisible(show_individual_workflows and is_trench_method)
            self.pushButton_run_waffle_undercut.setVisible(show_individual_workflows and show_undercut)

            # tab visibity / enabled
            # self.tabWidget.setTabVisible(CONFIGURATION["TABS"]["Detection"], self.WORKFLOW_IS_RUNNING and not _ON_GRID_METHOD)
            # self.tabWidget.setTabEnabled(CONFIGURATION["TABS"]["Connection"], not self.WORKFLOW_IS_RUNNING)
            # self.tabWidget.setTabEnabled(CONFIGURATION["TABS"]["Experiment"], not self.WORKFLOW_IS_RUNNING)
            # self.tabWidget.setTabEnabled(CONFIGURATION["TABS"]["Protocol"], not self.WORKFLOW_IS_RUNNING)

            if self.WORKFLOW_IS_RUNNING:
                self.pushButton_run_waffle_trench.setEnabled(False)
                self.pushButton_run_waffle_undercut.setEnabled(False)
                self.pushButton_run_setup_autolamella.setEnabled(False)
                self.pushButton_run_waffle_trench.setStyleSheet(
                    stylesheets.DISABLED_PUSHBUTTON_STYLE
                )
                self.pushButton_run_waffle_undercut.setStyleSheet(
                    stylesheets.DISABLED_PUSHBUTTON_STYLE
                )
                self.pushButton_run_setup_autolamella.setStyleSheet(
                    stylesheets.DISABLED_PUSHBUTTON_STYLE
                )

        # Current Lamella Status
        if has_lamella:
            self.update_lamella_ui()
            estimated_time = self.experiment.estimate_remaining_time()
            txt = f"Estimated time remaining: {utils.format_duration(estimated_time)}"
            self.label_run_autolamella_info.setText(txt)

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

    def update_lamella_combobox(self, latest: bool = False):
        # detail combobox
        idx = self.comboBox_current_lamella.currentIndex()
        self.comboBox_current_lamella.currentIndexChanged.disconnect()
        self.comboBox_current_lamella.clear()
        self.comboBox_current_lamella.addItems(
            [lamella.info for lamella in self.experiment.positions]
        )
        if idx != -1 and self.experiment.positions:
            self.comboBox_current_lamella.setCurrentIndex(idx)
        if latest and self.experiment.positions:
            self.comboBox_current_lamella.setCurrentIndex(
                len(self.experiment.positions) - 1
            )
        self.comboBox_current_lamella.currentIndexChanged.connect(
            self.update_lamella_ui
        )

    def update_lamella_ui(self):
        # set the info for the current selected lamella
        if self.experiment is None:
            return

        if self.experiment.positions == []:
            return

        if self.WORKFLOW_IS_RUNNING:
            return
        
        if self.milling_widget is None:
            return

        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            # clear milling patterns
            self.milling_widget.set_milling_stages([])
            return

        lamella: Lamella = self.experiment.positions[idx]
        logging.info(f"Updating Lamella UI for {lamella.info}")

        # buttons
        if self.is_protocol_loaded:

            method = self.protocol.method
            if lamella.workflow is AutoLamellaStage.Created:
                self.pushButton_save_position.setText("Save Position")
                self.pushButton_save_position.setStyleSheet(
                    stylesheets.ORANGE_PUSHBUTTON_STYLE
                )
                self.pushButton_save_position.setEnabled(True)
                self.milling_widget.CAN_MOVE_PATTERN = True
                self.milling_widget.setEnabled(True)
            elif lamella.workflow is AutoLamellaStage.PositionReady:
                self.pushButton_save_position.setText("Position Ready")
                self.pushButton_save_position.setStyleSheet(
                    stylesheets.GREEN_PUSHBUTTON_STYLE
                )
                self.pushButton_save_position.setEnabled(True)
                self.milling_widget.CAN_MOVE_PATTERN = False
                self.milling_widget.setEnabled(False)
            else:
                self.pushButton_save_position.setText("")
                self.pushButton_save_position.setStyleSheet(
                    stylesheets.DISABLED_PUSHBUTTON_STYLE
                )
                self.pushButton_save_position.setEnabled(False)
                self.milling_widget.CAN_MOVE_PATTERN = True
                self.milling_widget.setEnabled(True)

            # landing grid selected
            self.pushButton_lamella_landing_selected.setVisible(method.is_liftout)
            if method.is_liftout: 
                # TODO: refactor how this is handled, only select it during workflow...
                self.pushButton_lamella_landing_selected.setVisible(True)
                if method is AutoLamellaMethod.SERIAL_LIFTOUT:
                    if self.experiment.landing_positions:
                        n = len(self.experiment.landing_positions)
                        self.pushButton_lamella_landing_selected.setEnabled(True)
                        self.pushButton_lamella_landing_selected.setText(f"{n} Landing Positions Selected")
                        self.pushButton_lamella_landing_selected.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
                    else:
                        self.pushButton_lamella_landing_selected.setText("No Landing Positions")
                        self.pushButton_lamella_landing_selected.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
                        self.pushButton_lamella_landing_selected.setToolTip("Run Development/Select Landing Positions to select Landing Positions")
                elif lamella.landing_selected:
                    self.pushButton_lamella_landing_selected.setText("Landing Position Selected")
                    self.pushButton_lamella_landing_selected.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
                    self.pushButton_lamella_landing_selected.setEnabled(True)
                else:
                    self.pushButton_lamella_landing_selected.setEnabled(False)
                    self.pushButton_lamella_landing_selected.setText("No Landing Position")
                    self.pushButton_lamella_landing_selected.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
                    self.pushButton_lamella_landing_selected.setToolTip("Run Setup Liftout to select a Landing Position")

        # update the milling widget
        if self.WORKFLOW_IS_RUNNING:
            self.milling_widget.CAN_MOVE_PATTERN = True
            self.milling_widget.setEnabled(True)

        # clear milling patterns, and set new ones
        self.milling_widget.clear_all_milling_stages()
        if lamella.workflow in PREPARTION_WORKFLOW_STAGES:
            if not self.is_protocol_loaded:
                raise ValueError("I SHOULDNT BE ABLE TO GET HERE")
            
            method = self.protocol.method

            if method.is_trench:
                stages = get_milling_stages(TRENCH_KEY, lamella.protocol)

            else:
                mill_rough_stages = get_milling_stages(MILL_ROUGH_KEY, lamella.protocol)
                mill_polishing_stages = get_milling_stages(MILL_POLISHING_KEY, lamella.protocol)
                stages = mill_rough_stages + mill_polishing_stages

                if self.protocol.options.use_notch:
                    stages.extend(get_milling_stages(NOTCH_KEY, lamella.protocol))

                # microexpansion
                if self.protocol.options.use_microexpansion:
                    stages.extend(get_milling_stages(MICROEXPANSION_KEY, lamella.protocol))

                # fiducial
                if self.protocol.options.use_fiducial:
                    stages.extend(get_milling_stages(FIDUCIAL_KEY, lamella.protocol))

            self.milling_widget.set_milling_stages(stages)

            # TODO: migrate to this structure instead of the above
            # milling_workflows: Dict[str, List[FibsemMillingStage]]: 
            #     # MILL_ROUGH_KEY: [FibsemMillingStage, FibsemMillingStage]
            #     # MILL_POLISHING_KEY: [FibsemMillingStage, FibsemMillingStage]
            #     # NOTCH_KEY: [FibsemMillingStage]
            #     # MICROEXPANSION_KEY: [FibsemMilling
            #     # FIDUCIAL_KEY: [FibsemMillingStage]

            # milling widget?
            # add / remove workflow?
            # when used independently, only single workflow?
            # when used in autolamella, multiple workflows?
        
        # defect button
        msg = "Mark as Active" if lamella.is_failure else "Mark As Defect"
        self.pushButton_fail_lamella.setText(msg)

        # time travel controls
        has_history = bool(lamella.history)
        self.comboBox_lamella_history.setVisible(has_history)
        self.pushButton_revert_stage.setVisible(has_history)
        if has_history:
            self.comboBox_lamella_history.clear()
            self.comboBox_lamella_history.addItems([state.completed for state in lamella.history])

        display_selected_lamella_info(grid_layout=self.gridLayout_selected_lamella_history, 
                                      pos=lamella, method=self.protocol.method)

    def _update_milling_position(self):
        # triggered when milling position is moved
        if self.experiment is None:
            return

        if self.experiment.positions == []:
            return

        idx = self.comboBox_current_lamella.currentIndex()
        lamella: Lamella = self.experiment.positions[idx]

        if lamella.workflow not in PREPARTION_WORKFLOW_STAGES:
            return

        logging.info(f"Updating Lamella Pattern for {lamella.info}")

        # update the trench point
        self._update_milling_protocol(idx=idx, method=self.protocol.method, stage=lamella.workflow)

        self.experiment.save()

    def load_protocol(self):
        """Load a protocol from file."""

        if self.settings is None:
            napari.utils.notifications.show_info(
                "Please connect to the microscope first."
            )
            return

        PROTOCOL_PATH = fui.open_existing_file_dialog(
            msg="Select a protocol file", path=cfg.PROTOCOL_PATH, parent=self
        )

        if PROTOCOL_PATH == "":
            napari.utils.notifications.show_info("No path selected")
            return

        # validate the protocol and up-convert it
        self.protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)
        self.protocol.configuration = deepcopy(self.settings)

        # TODO:
        # replace self.settings with self.protocol
        # use self.protocol.configuration to store settings
        # deprecate self.settings

        self.is_protocol_loaded = True
        self.update_protocol_ui()
        napari.utils.notifications.show_info(
            f"Loaded Protocol from {os.path.basename(PROTOCOL_PATH)}"
        )

        # save a copy of the protocol to the experiment.path
        if self.experiment:
            self.protocol.save(os.path.join(self.experiment.path, "protocol.yaml"))
            self.experiment.method = self.protocol.method

        milling_angle = self.protocol.options.milling_angle
        _ = fui.message_box_ui(
            title="Protocol Redefined",
            text="The protocol option 'milling_angle' now represents the angle between the FIB and sample stage. "
            "Previously, this referred to the stage tilt angle. Please update your protocol accordingly." \
            f"\n\nThe current milling angle in the protocol is: {milling_angle} {DEGREE_SYMBOL}",
            buttons=QtWidgets.QMessageBox.Ok,
            parent=self,
        )

        self.update_ui()

    def cryo_deposition(self):
        cryo_deposition_widget = FibsemCryoDepositionWidget(
            self.microscope, self.settings
        )
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
        logging.debug("Sender: {}".format(self.sender().objectName()))

        # positve / negative response
        self.USER_RESPONSE = bool(self.sender() == self.pushButton_yes)
        self.WAITING_FOR_USER_INTERACTION = False

    def move_to_lamella_position(self):
        """Move the stage to the position of the selected lamella."""

        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return
        lamella: Lamella = self.experiment.positions[idx]
        stage_position = lamella.state.microscope_state.stage_position

        logging.info(f"Moving to position of {lamella.name}.")
        self.movement_widget.move_to_position(stage_position)

    def _add_lamella_from_odemis(self):

        filename = fui.path = fui.open_existing_directory_dialog(
            msg="Select Odemis Project Directory",
            path=self.experiment.path,
            parent=self,
        )
        if filename == "":
            return

        from fibsem.applications.autolamella.compat.odemis import _add_features_from_odemis
        stage_positions = _add_features_from_odemis(filename)

        for pos in stage_positions:
            self.add_new_lamella(pos)

    def add_new_lamella(self, stage_position: Optional[FibsemStagePosition] = None) -> Lamella:
        """Add a lamella to the experiment.
        Args:
            pos: The stage position of the lamella. If None, the current stage position is used.
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
        mprotocol = self.protocol.milling # TODO: migrate to milling_workflow
        tmp_protocol = deepcopy({k: get_protocol_from_stages(v) for k, v in mprotocol.items()})
        state = LamellaState(stage=AutoLamellaStage.Created, microscope_state=microscope_state)
        lamella = create_new_lamella(experiment_path=self.experiment.path, 
                                     number=len(self.experiment.positions) + 1, 
                                     state=state, 
                                     protocol=tmp_protocol)

        from fibsem.applications.autolamella.workflows.core import log_status_message
        log_status_message(lamella, "STARTED")
        self.experiment.positions.append(deepcopy(lamella))
        
        # if created from minimap, automatically mark as ready
        if stage_position is not None:

            # TODO: this is excessive, we need a cleaner way to do this

            from fibsem.applications.autolamella.workflows.core import (
                end_of_stage_update,
                start_of_stage_update,
            )
            self.experiment = end_of_stage_update(
                microscope=self.microscope,
                experiment=self.experiment,
                lamella=lamella,
                parent_ui=self,
                save_state=False,
                update_ui=False
            )

            # start ready stage
            self.experiment.positions[-1] = start_of_stage_update(
                microscope=self.microscope,
                lamella=lamella,
                next_stage=AutoLamellaStage.PositionReady,
                parent_ui=self,
                restore_state=False,
                update_ui=False
            )

            # end the stage
            self.experiment = end_of_stage_update(
                microscope=self.microscope,
                experiment=self.experiment,
                lamella=lamella,
                parent_ui=self,
                save_state=False,
                update_ui=False,
            )

        self.experiment.save()
        self.update_lamella_combobox(latest=True)
        self.update_ui()

        return lamella

    def remove_lamella_ui(self):
        """Handle the removal of a lamella from the experiment."""

        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            logging.warning("No lamella is selected, cannot remove.")
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
        self.milling_widget.clear_all_milling_stages()
        self.update_lamella_combobox(latest=True)
        self.update_ui()

        self.sync_experiment_positions_to_minimap()  # update the minimap

        # clear milling widget if no lamella
        if self.experiment.positions == []:
            self.milling_widget.clear_all_milling_stages()
            remove_all_napari_shapes_layers(
                self.viewer, layer_type=napari.layers.points.points.Points
            )

    def fail_lamella_ui(self):
        idx = self.comboBox_current_lamella.currentIndex()
        if idx == -1:
            return

        # get the current state
        is_failure = self.experiment.positions[idx].is_failure
        name = self.experiment.positions[idx].name

        # if marking as failure, get user reason for failure
        if not is_failure:
            msg, ret = fui.open_text_input_dialog(
                msg="Enter defect reason:",
                title=f"Mark Lamella {name} as defect?",
                default="",
                parent=self,
            )

            if ret is False:
                logging.debug(f"User cancelled marking lamella {name} as failure.")
                return

            self.experiment.positions[idx].is_failure = True
            self.experiment.positions[idx].failure_note = msg
            self.experiment.positions[idx].failure_timestamp = datetime.timestamp(
                datetime.now()
            )
        else:
            self.experiment.positions[idx].is_failure = False
            self.experiment.positions[idx].failure_note = ""
            self.experiment.positions[idx].failure_timestamp = None

        self.experiment.save()
        self.update_ui()

    def revert_stage(self):
        idx = self.comboBox_current_lamella.currentIndex()
        hidx = self.comboBox_lamella_history.currentIndex()

        self.experiment.positions[idx].state = deepcopy(
            self.experiment.positions[idx].history[hidx]
        )
        self.experiment.positions[idx].state.start_timestamp = datetime.timestamp(
            datetime.now()
        )
        from fibsem.applications.autolamella.workflows.core import log_status_message

        log_status_message(self.experiment.positions[idx], "STARTED")
        # TODO: use start of stage update to restore the state properly
        # TODO: save the experiment to disk? not sure if that's correct to do?

        self.update_ui()

    def save_lamella_ui(self):
        # triggered when save button is pressed

        if self.experiment.positions == []:
            return

        idx = self.comboBox_current_lamella.currentIndex()
        # TOGGLE BETWEEN READY AND SETUP

        method = self.protocol.method

        lamella: Lamella = self.experiment.positions[idx]
        from fibsem.applications.autolamella.workflows.core import (
            end_of_stage_update,
            start_of_stage_update,
        )

        if lamella.workflow not in PREPARTION_WORKFLOW_STAGES:
            return

        # we need to be at the lamella position to save, check we are...
        # usually this is required when selected positions from minimap.
        # TODO: change how we do this, so that this is not required
        current_position = self.microscope.get_stage_position()

        if (
            not lamella.state.microscope_state.stage_position.is_close(
                current_position, 1e-6
            )
            and lamella.workflow is AutoLamellaStage.Created
        ):
            ret = fui.message_box_ui(
                title="Far away from lamella position",
                text="The current position is far away from the initial lamella position. Move to initial lamella position? (Press No to save at current position)",
            )

            # TODO: handle the case when user exits dialog box
            # TODO: only do this check if position created by minimap?

            if ret is True:
                # move to lamella position
                self.movement_widget.move_to_position(
                    lamella.state.microscope_state.stage_position
                )

        # end of stage update
        self.experiment = end_of_stage_update(
            microscope=self.microscope,
            experiment=self.experiment,
            lamella=lamella,
            parent_ui=self,
            save_state=True,
            update_ui=False
        )

        if lamella.workflow is AutoLamellaStage.Created:
            # start of stage update
            self.experiment.positions[idx] = start_of_stage_update(
                microscope=self.microscope,
                lamella=lamella,
                next_stage=AutoLamellaStage.PositionReady,
                parent_ui=self,
                restore_state=False,
                update_ui=False
            )

            # update the protocol / point
            self._update_milling_protocol(idx, method, AutoLamellaStage.PositionReady)

            # get current ib image, save as reference
            fname = os.path.join(
                self.experiment.positions[idx].path,
                f"ref_{self.experiment.positions[idx].status}",
            )
            self.image_widget.ib_image.save(fname)
            self.milling_widget.CAN_MOVE_PATTERN = False
            self.milling_widget.setEnabled(False)

            self.experiment = end_of_stage_update(
                microscope=self.microscope,
                experiment=self.experiment,
                lamella=lamella,
                parent_ui=self,
                save_state=True,
                update_ui=False
            )

        elif lamella.workflow is AutoLamellaStage.PositionReady:
            self.experiment.positions[idx] = start_of_stage_update(
                self.microscope,
                lamella,
                AutoLamellaStage.Created,
                parent_ui=self,
                restore_state=False,
                update_ui=False
            )

            self.milling_widget.CAN_MOVE_PATTERN = True
            self.milling_widget.setEnabled(True)

        self.experiment.positions[idx].state.microscope_state.stage_position.name = lamella.name

        self.sync_experiment_positions_to_minimap()
        self.update_lamella_combobox()
        self.update_ui()
        self.experiment.save()

    def _update_milling_protocol(
        self, idx: int, method: AutoLamellaMethod, stage: AutoLamellaStage
    ):

        if stage not in PREPARTION_WORKFLOW_STAGES:
            return
        
        stages = deepcopy(self.milling_widget.get_milling_stages())
        
        # TRENCH SETUP
        if method.is_trench:
            self.experiment.positions[idx].protocol[TRENCH_KEY] = get_protocol_from_stages(stages)
            return 

        # ON-GRID SETUP
        # TODO: allow the user to select the number of rough and polishing stages?, and re-assign?
        # TODO: use lmaella.protocol
        n_mill_rough = len(self.protocol.milling[MILL_ROUGH_KEY])
        n_mill_polishing = len(self.protocol.milling[MILL_POLISHING_KEY])

        # rough milling
        self.experiment.positions[idx].protocol[MILL_ROUGH_KEY] = get_protocol_from_stages(stages[:n_mill_rough])

        # polishing
        self.experiment.positions[idx].protocol[MILL_POLISHING_KEY] = get_protocol_from_stages(stages[n_mill_rough:n_mill_rough+n_mill_polishing])
        
        # total number of stages in lamella
        n_lamella = n_mill_rough + n_mill_polishing

        # stress relief features
        use_notch = self.protocol.options.use_notch
        use_microexpansion = self.protocol.options.use_microexpansion

        if use_notch:
            self.experiment.positions[idx].protocol[NOTCH_KEY] = get_protocol_from_stages(stages[n_lamella])

        if use_microexpansion:
            self.experiment.positions[idx].protocol[MICROEXPANSION_KEY] = get_protocol_from_stages(stages[n_lamella + use_notch])

        # fiducial (optional)
        if self.protocol.options.use_fiducial:
            self.experiment.positions[idx].protocol[FIDUCIAL_KEY] = get_protocol_from_stages(stages[-1])

    def run_milling(self):
        self.is_milling = True
        self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Milling"])
        self.milling_widget.run_milling()

    def handle_milling_update(self, ddict: dict) -> None:

        is_finished = ddict.get("finished", False)
        if is_finished:
            self.is_milling = False

    def handle_acquisition_update(self, ddict: dict) -> None:
        if ddict.get("finished", False):
            self.update_lamella_ui()

    def handle_confirmed_detection_signal(self):
        # TODO: this seem very redundant if we just use the signal directly
        if self.det_widget is not None:
            self.det_widget.confirm_button_clicked()

    def _stop_workflow_thread(self):
        self.STOP_WORKFLOW = True
        napari.utils.notifications.show_error("Abort requested by user.")

    def _run_workflow(self, workflow: str) -> None:
        """Run the specified workflow."""
        
        accepted, stc, supervision = open_workflow_dialog(
            experiment=deepcopy(self.experiment),
            protocol=self.protocol,
            parent=self,
        )
        if not accepted:
            return

        logging.info(f"Accepted: {accepted}, STC: {stc}, Supervision: {supervision}")

        self.protocol.supervision = supervision
        self.update_protocol_ui()
        
        try:
            self.milling_widget.milling_position_changed.disconnect()
        except Exception:
            pass

        self.STOP_WORKFLOW = False
        self.WORKFLOW_IS_RUNNING = True
        self.milling_widget.CAN_MOVE_PATTERN = True
        self.milling_widget.setEnabled(True)
        self.milling_widget.clear_all_milling_stages()
        self.WAITING_FOR_USER_INTERACTION = False
        remove_all_napari_shapes_layers(
            self.viewer, layer_type=napari.layers.points.points.Points
        )

        # disable ui buttons
        self.pushButton_run_waffle_trench.setEnabled(False)
        self.pushButton_run_waffle_undercut.setEnabled(False)
        self.pushButton_run_setup_autolamella.setEnabled(False)
        self.pushButton_run_waffle_trench.setStyleSheet(
            stylesheets.DISABLED_PUSHBUTTON_STYLE
        )
        self.pushButton_run_waffle_undercut.setStyleSheet(
            stylesheets.DISABLED_PUSHBUTTON_STYLE
        )
        self.pushButton_run_setup_autolamella.setStyleSheet(
            stylesheets.DISABLED_PUSHBUTTON_STYLE
        )
        self.pushButton_stop_workflow.setVisible(True)
        self.set_instructions_msg(f"Running {workflow.title()} workflow...")

        # turn on the beams, if not already on
        if not self.microscope.get("on", BeamType.ELECTRON):
            self.microscope.turn_on(BeamType.ELECTRON)
        if not self.microscope.get("on", BeamType.ION):
            self.microscope.turn_on(BeamType.ION)

        try:
            self.worker = self._threaded_worker(
                microscope=self.microscope,
                settings=self.settings,
                protocol=self.protocol,
                experiment=deepcopy(self.experiment),
                method = self.protocol.method,
                workflow=workflow,
                stc=stc,
            )
        except Exception as e:
            logging.error(f"An error occurred while running workflow: {e}")
        self.worker.finished.connect(self._workflow_finished)
        self.worker.errored.connect(self._workflow_aborted)
        self.worker.start()

    def _workflow_aborted(self):
        """Handle the abortion of the workflow."""
        logging.info("Workflow aborted.")
        if self.is_milling:
            self.milling_widget.stop_milling()

        for lamella in self.experiment.positions:
            if lamella.workflow is not lamella.history[-1].stage:
                lamella.state = deepcopy(lamella.history[-1])
                logging.info("restoring state for {}".format(lamella.info))

        self._workflow_finished()

    def _workflow_finished(self):
        """Handle the completion of the workflow."""
        logging.info("Workflow finished.")
        self.WORKFLOW_IS_RUNNING = False
        self.STOP_WORKFLOW = False
        self.milling_widget.milling_position_changed.connect(
            self._update_milling_position
        )
        self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Experiment"])
        self.pushButton_stop_workflow.setVisible(False)

        # clear the image settings save settings etc
        self.image_widget.checkBox_image_save_image.setChecked(False)
        self.image_widget.lineEdit_image_path.setText(self.experiment.path)
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

        # update the image viewer
        sem_image = info.get("sem_image", None)
        if sem_image is not None:
            self.image_widget.eb_image = sem_image
            self.image_widget.update_viewer(arr=sem_image.data, 
                                            beam_type=BeamType.ELECTRON, 
                                            set_ui_from_image=True)
        
        fib_image = info.get("fib_image", None)
        if fib_image is not None:
            self.image_widget.ib_image = fib_image
            self.image_widget.update_viewer(arr=fib_image.data, 
                                            beam_type=BeamType.ION, 
                                            set_ui_from_image=True)

        # what?
        enable_milling = info.get("milling_enabled", None)
        if enable_milling is not None:
            self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Milling"])

        # update milling stages
        milling_stages = info.get("stages", None)
        if isinstance(milling_stages, list):
            self.milling_widget.set_milling_stages(milling_stages)
        if milling_stages == "clear":
            self.milling_widget.clear_all_milling_stages()
        detections = info.get("det", None)
        if self.det_widget is not None and detections is not None:
            self.det_widget.set_detected_features(detections)
            self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Detection"])
            self.tabWidget.setTabVisible(CONFIGURATION["TABS"]["Detection"], True)

        # update the alignment area
        alignment_area = info.get("alignment_area", None)
        if isinstance(alignment_area, FibsemRectangle):
            self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Image"])
            self.image_widget.toggle_alignment_area(alignment_area)
        if alignment_area == "clear":
            self.image_widget.clear_alignment_area()
        
        # spot_burn
        spot_burn = info.get("spot_burn", None)
        if spot_burn:
            self.set_spot_burn_widget_active(True)
        
        # no specific interaction, just update the ui
        if detections is None and enable_milling is None and alignment_area is None and spot_burn is None:
            self.tabWidget.setCurrentIndex(CONFIGURATION["TABS"]["Experiment"])

        # ui interaction
        self.milling_widget.pushButton_run_milling.setEnabled(False)
        self.milling_widget.pushButton_run_milling.setVisible(False) # TODO: re-enable??

        # instruction message
        self.set_instructions_msg(info["msg"], info.get("pos", None), info.get("neg", None))
        self.set_current_workflow_message(info.get("workflow_info", None))

        self.WAITING_FOR_UI_UPDATE = False

    def hande_update_experiment_signal(self, experiment: Experiment):
        """Callback for updating the experiment object."""
        self.experiment = experiment
        self.update_lamella_combobox()
        self.update_ui()

    @thread_worker
    def _threaded_worker(
        self,
        microscope: FibsemMicroscope,
        settings: MicroscopeSettings,
        protocol: AutoLamellaProtocol,
        experiment: Experiment,
        method: AutoLamellaMethod,
        workflow: str = "trench",
        stc: List[AutoLamellaStage] = None
    ):
        logging.info(f"Started {workflow.title()} Workflow...")

        if stc is None:
            stc = method.workflow

        from fibsem.applications.autolamella.workflows import runners as wfl  # avoiding circular import
        if method in [AutoLamellaMethod.LIFTOUT, AutoLamellaMethod.SERIAL_LIFTOUT]:
            from fibsem.applications.autolamella.workflows import autoliftout

        if workflow == "trench":
            wfl.run_trench_milling(microscope, protocol, experiment, parent_ui=self)

        if workflow == "undercut":
            wfl.run_undercut_milling(microscope, protocol, experiment, parent_ui=self)

        if workflow == "autolamella":
            if method in wfl.METHOD_WORKFLOWS_FN:
                self.experiment = wfl.METHOD_WORKFLOWS_FN[method](microscope=microscope, 
                                                protocol=protocol, 
                                                experiment=experiment, 
                                                parent_ui=self, 
                                                stages_to_complete=stc)
    
            if method is AutoLamellaMethod.SERIAL_LIFTOUT:
                from fibsem.applications.autolamella.workflows.runners import run_autolamella
                run_autolamella(microscope=microscope, 
                                protocol=protocol, 
                                experiment=experiment, 
                                parent_ui=self, 
                                stages_to_complete=stc) #TODO: consolidate

        # liftout workflows
        if workflow == "setup-liftout":
            self.experiment = autoliftout.run_setup_autoliftout(
                microscope=microscope,
                settings=settings,
                experiment=experiment,
                parent_ui=self,
            )
        if workflow == "autoliftout":
            if method is AutoLamellaMethod.LIFTOUT:
                settings.image.autogamma = True
                self.experiment = autoliftout.run_autoliftout_workflow(
                    microscope=microscope,
                    settings=settings,
                    experiment=experiment,
                    parent_ui=self,
                )
            if method is AutoLamellaMethod.SERIAL_LIFTOUT:
                from fibsem.applications.autolamella.workflows import serial as serial_workflow

                self.experiment = serial_workflow.run_serial_liftout_workflow(
                    microscope=microscope,
                    protocol=protocol,
                    experiment=experiment,
                    parent_ui=self,
                )
        if workflow == "serial-liftout-landing":
            from fibsem.applications.autolamella.workflows import serial as serial_workflow

            self.experiment = serial_workflow.run_serial_liftout_landing(
                microscope=microscope,
                protocol=protocol,
                experiment=experiment,
                parent_ui=self,
            )
        if workflow == "select-landing-positions":
            from fibsem.applications.autolamella.workflows import serial as serial_workflow

            self.experiment = serial_workflow.select_landing_positions(
                microscope=microscope,
                protocol=protocol,
                experiment=experiment,
                parent_ui=self,
            )

        self.update_experiment_signal.emit(self.experiment)

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
