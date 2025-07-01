
import logging
import sys
from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Tuple

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QGridLayout,
)
from fibsem.utils import format_duration
from autolamella.structures import (
    AutoLamellaMethod,
    AutoLamellaProtocol,
    AutoLamellaStage,
    Experiment,
    Lamella,
    get_completed_stages,
)
from autolamella.ui.qt import AutoLamellaWorkflowDialog as AutoLamellaWorkflowDialogUI


def display_selected_lamella_info(grid_layout: QGridLayout, pos: Lamella, method: AutoLamellaMethod) -> None:
    """Display the history of a selected lamella."""
    
    # clear the existing layout
    for i in reversed(range(grid_layout.count())):
        for i in reversed(range(grid_layout.count())):
            grid_layout.itemAt(i).widget().setParent(None)

    # filter out the states that are not in the method (setups, finishes, etc.)

    workflow_states = get_completed_stages(pos, method)

    # if there are no states, return
    if len(workflow_states) == 0:
        return

    # add headers
    hworkflow = QLabel("Workflow")
    hcompleted = QLabel("Completed At")
    hduration = QLabel("Duration")
    hworkflow.setStyleSheet("font-weight: bold")
    hcompleted.setStyleSheet("font-weight: bold")
    hduration.setStyleSheet("font-weight: bold")
    grid_layout.addWidget(hworkflow, 0, 0)
    grid_layout.addWidget(hcompleted, 0, 1)
    grid_layout.addWidget(hduration, 0, 2)

    for i, wf in enumerate(workflow_states, 1):
        state = pos.states[wf]
        grid_layout.addWidget(QLabel(wf.name), i, 0)
        grid_layout.addWidget(QLabel(state.completed_at), i, 1)
        grid_layout.addWidget(QLabel(state.duration_str), i, 2)
    return 

def display_lamella_info(grid_layout: QGridLayout, 
                         positions: List[Lamella], 
                         method: AutoLamellaMethod) -> None:
    """Create a grid layout of lamella information."""
    
    # clear the existing layout
    for i in reversed(range(grid_layout.count())):
        for i in reversed(range(grid_layout.count())):
            grid_layout.itemAt(i).widget().setParent(None)

    # add headers
    name_header = QLabel("Name")
    name_header.setStyleSheet("font-weight: bold")
    status_header = QLabel("Status")
    status_header.setStyleSheet("font-weight: bold")
    last_header = QLabel("Last Completed")
    last_header.setStyleSheet("font-weight: bold")
    next_header = QLabel("Starting From")
    next_header.setStyleSheet("font-weight: bold")
    grid_layout.addWidget(name_header, 0, 0)
    grid_layout.addWidget(status_header, 0, 1)
    grid_layout.addWidget(last_header, 0, 2)
    grid_layout.addWidget(next_header, 0, 3)

    pos: Lamella
    for i, pos in enumerate(positions, 1):

        next_workflow = method.get_next(pos.workflow)
        is_finished = pos.workflow is AutoLamellaStage.Finished or next_workflow is None
        is_creation = pos.workflow is AutoLamellaStage.Created
        is_failure = pos.is_failure
        
        # get the name of the lamella
        name_label = QLabel(f"Lamella {pos.name}")

        # get the status of the lamella
        status_label = QLabel()
        status_msg = "Active"
        status_label.setStyleSheet("color: cyan")
        if is_finished:
            status_msg = "Finished"
            status_label.setStyleSheet("color: limegreen")
        if is_creation:
            status_msg = "Created"
            status_label.setStyleSheet("color: orange")
        if is_failure:
            if len(pos.failure_note) > 5:
                note = f"{pos.failure_note[:3]}..."
            else:
                note = pos.failure_note
            status_msg = f"Defect ({note})"
            status_label.setStyleSheet("color: red")
            status_label.setToolTip(pos.failure_note)
        status_label.setText(status_msg)

        # get the last completed workflow stage
        last_label = QLabel(pos.last_completed)
        if is_finished:
            # special case for finished lamella
            prev = method.get_previous(pos.workflow)
            state = pos.states.get(prev, None)
            if state is not None:
                last_label.setText(state.completed)

        # get the next workflow stage
        next_label = QLabel()
        if not is_creation and not is_finished and not is_failure:
            next_label.setText(next_workflow.name)
        
        # control_widget = QComboBox()
        # control_widget.addItems([s.name for s in START_STATES])
        # print(f"Next stage: {NEXT_WORKFLOW_STAGE[pos.stage].name}")
        # control_widget.setCurrentText(NEXT_WORKFLOW_STAGE[pos.workflow].name)

        grid_layout.addWidget(name_label, i, 0)
        grid_layout.addWidget(status_label, i, 1)
        grid_layout.addWidget(last_label, i, 2)
        grid_layout.addWidget(next_label, i, 3)
        # self.gridLayout_lamella.addWidget(control_widget, i, 2)

    
class AutoLamellaWorkflowDialog(QDialog, AutoLamellaWorkflowDialogUI.Ui_Dialog):

    def __init__(self, 
                 experiment: Experiment, 
                 protocol: AutoLamellaProtocol, parent=None):
        super().__init__(parent=parent)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        self.setupUi(self)
        self.setWindowTitle("AutoLamella Workflow")
        self.parent = parent

        self.experiment = experiment
        self.protocol = protocol
        self.method = self.protocol.method

        self.workflow_widgets: Dict[AutoLamellaStage, Tuple[QCheckBox, QCheckBox]] = {}

        self.stages_to_complete: List[AutoLamellaStage] = None
        self.supervision: Dict[AutoLamellaStage, bool] = None

        self.setup_connections()
        self._update_estimated_duration()

    def setup_connections(self):

        self.label_method.setText(f"Method: {self.method.name}")

        self.buttonBox.button(QDialogButtonBox.Ok).setText("Start")

        # connect buttonBox events
        self.buttonBox.accepted.connect(self.on_start)
        self.buttonBox.rejected.connect(self.on_exit)

        # display lamella information
        display_lamella_info(grid_layout=self.gridLayout_lamella, 
                             positions=self.experiment.positions, 
                             method=self.method)

        # display workflow settings
        for i, state in enumerate(self.method.workflow):
            label = QLabel(state.name)
            enable_checkbox = QCheckBox()
            enable_checkbox.setChecked(True)
            enable_checkbox.setText("Enable")

            supervised_checkbox = QCheckBox()
            supervised_checkbox.setChecked(self.protocol.supervision[state])
            supervised_checkbox.setText("Supervised")

            self.gridLayout_workflow.addWidget(label, i, 0)
            self.gridLayout_workflow.addWidget(enable_checkbox, i, 1)
            self.gridLayout_workflow.addWidget(supervised_checkbox, i, 2)

            self.workflow_widgets[state] = (enable_checkbox, supervised_checkbox)

    def get_workflow_settings(self) -> Dict[AutoLamellaStage, Tuple[bool, bool]]:
        return {state: (enable.isChecked(), supervised.isChecked()) for state, (enable, supervised) in self.workflow_widgets.items()}

    def _update_estimated_duration(self):
        """Update the estimated duration of the workflow."""
        estimated_time = self.experiment.estimate_remaining_time()
        txt = f"Estimated time remaining: {format_duration(estimated_time)}"
        self.label_information.setText(txt)

    def on_start(self):
        logging.info("Starting AutoLamella Workflow")
        
        wf = self.get_workflow_settings()
        self.stages_to_complete = [k for k, v in wf.items() if v[0]]
        self.supervision: Dict[AutoLamellaStage, bool] = {k: v[1] for k, v in wf.items()}

        self.accept()

    def on_exit(self):
        print("Exit")
        self.reject()


def open_workflow_dialog(
    experiment: Experiment, 
    protocol: AutoLamellaProtocol, 
    parent=None,
) -> Tuple[bool, List[AutoLamellaStage], Dict[AutoLamellaStage, bool]]:
    """Open the AutoLamella Workflow Dialog. 
    Allows the user to select the workflow stages and supervision settings.
    Args:
        experiment: Experiment
        protocol: AutoLamellaProtocol
    Returns:
        accepted: Start the workflow
        stages_to_complete: List of stages to complete
        supervision: Supervision settings for each stage
    """
    dialog = AutoLamellaWorkflowDialog(experiment=experiment, 
                                       protocol=protocol, 
                                       parent=parent)
    ret = dialog.exec_()

    stages_to_complete = dialog.stages_to_complete
    supervision = dialog.supervision
    accepted = bool(ret == QDialog.Accepted)

    return accepted, stages_to_complete, supervision


def main():
    app = QApplication(sys.argv)

    accepted, stc, supervision = open_workflow_dialog(
        experiment=exp, protocol=protocol
    )
    
    print(f"Accepted: {accepted}")
    print(f"Stages to complete: {stc}")
    print(f"Supervision: {supervision}")

if __name__ == "__main__":
    main()


# ReviewWidget
# Select Lamella ComboBox
# Display Figure