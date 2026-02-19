

import os
from datetime import datetime
from typing import List, Optional

from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel, QWidget

from fibsem.applications.autolamella.structures import Experiment


class AutoLamellaWorkflowDisplayWidget(QWidget):
    def __init__(self, experiment: Optional[Experiment] = None, parent=None):
        super().__init__(parent)

        self.experiment = experiment
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        if self.experiment is not None:
            self.update_ui()

    def set_experiment(self, experiment: Experiment):
        self.experiment = experiment
        self.update_ui()

    def update_ui(self):
        # clear the existing layout
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        # add headers
        name_header = QLabel("Name")
        name_header.setStyleSheet("font-weight: bold")
        status_header = QLabel("Status")
        status_header.setStyleSheet("font-weight: bold")
        last_header = QLabel("Last Completed")
        last_header.setStyleSheet("font-weight: bold")
        self.grid_layout.addWidget(name_header, 0, 0)
        self.grid_layout.addWidget(status_header, 0, 1)
        self.grid_layout.addWidget(last_header, 0, 2)

        if self.experiment is None:
            return
        if self.experiment.task_protocol is None:
            return

        workflow_config = self.experiment.task_protocol.workflow_config

        for i, pos in enumerate(self.experiment.positions, 1):
            name_label = QLabel(f"Lamella {pos.name}")
            is_finished = workflow_config.is_completed(pos)
            remaining_tasks = workflow_config.get_remaining_tasks(pos)

            status_label = QLabel()
            status_msg = "Active"
            status_label.setStyleSheet("color: cyan")
            if not is_finished and len(remaining_tasks) > 0:
                status_label.setToolTip(f"<strong>Remaining Tasks:</strong><br>{'<br>'.join(remaining_tasks)}")
            if is_finished:
                status_msg = "Finished"
                status_label.setStyleSheet("color: limegreen")
            if pos.defect.has_defect or pos.defect.requires_rework:
                desc = pos.defect.description
                note = desc
                if len(desc) > 5:
                    note = f"{desc[:5]}..."
                if note != "":
                    note = f"({note})"
                # req_rework = pos.defect.requires_rework
                if pos.defect.has_defect:
                    status_msg = f"Defect {note}"
                    status_label.setStyleSheet("color: red")
                else:
                    status_msg = f"Rework {note}"
                    status_label.setStyleSheet("color: orange")
                status_label.setToolTip(f"{pos.defect.description} - {datetime.fromtimestamp(pos.defect.updated_at).strftime('%Y-%m-%d %H:%M:%S')}")
            status_label.setText(status_msg)

            last_completed_txt = ""
            if pos.last_completed_task is not None:
                last_completed_txt = pos.last_completed_task.completed
            last_label = QLabel(last_completed_txt)
            last_label.setWordWrap(True)

            # add a tooltip with the completed tasks, separated by new lines
            completed_tasks = workflow_config.get_completed_tasks(pos, with_timestamps=True)
            if len(completed_tasks) > 0:
                last_label.setToolTip("<strong>Completed Tasks:</strong><br>" + "<br>".join(completed_tasks))

            self.grid_layout.addWidget(name_label, i, 0)
            self.grid_layout.addWidget(status_label, i, 1)
            self.grid_layout.addWidget(last_label, i, 2)

        # add stretch to the layout
        self.grid_layout.setRowStretch(self.grid_layout.rowCount(), 1)

def main():
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-09-18-15-31"
    exp = Experiment.load(os.path.join(PATH, "experiment.yaml"))
    print(exp)

    ACQUIRE_REFERENCE_TASK = "Acquire Reference Image (FIB)"
    SPOT_BURN_TASK = "Spot Burn Fiducial"
    FLUORESCENCE_TASK = "Acquire Fluorescence Image"
    MILL_FIDUCIAL_TASK = "Mill Fiducial"
    MILL_ROUGH_TASK = "Rough Milling"
    from fibsem.applications.autolamella.protocol.constants import (
        FIDUCIAL_KEY,
        MICROEXPANSION_KEY,
        MILL_ROUGH_KEY,
    )
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskConfig,
        AutoLamellaTaskDescription,
        AutoLamellaTaskProtocol,
        AutoLamellaWorkflowConfig,
        ImageSettings,
    )
    from fibsem.applications.autolamella.workflows.tasks.tasks import (
        STRESS_RELIEF_KEY,
        AcquireFluorescenceImageConfig,
        AcquireReferenceImageConfig,
        MillRoughTaskConfig,
        MillFiducialTaskConfig,
        SpotBurnFiducialTaskConfig,
    )
    protocol = AutoLamellaTaskProtocol(
        name="Test Protocol",
        description="A test protocol",
        task_config={
            ACQUIRE_REFERENCE_TASK: AcquireReferenceImageConfig(
                            task_name=ACQUIRE_REFERENCE_TASK,
                            beams="FIB",
                            imaging=ImageSettings(hfw=150e-6,
                            resolution=(1536, 1024),
                            dwell_time=1e-6,
                            autocontrast=True,
                            )),
        SPOT_BURN_TASK: SpotBurnFiducialTaskConfig(
                    task_name=SPOT_BURN_TASK,
                    milling_current=60e-12,
                    exposure_time=10,
                    orientation=None),
    MILL_FIDUCIAL_TASK: MillFiducialTaskConfig(
        task_name=MILL_FIDUCIAL_TASK,
        alignment_expansion=100,
        use_fiducial=True,
    ),
    MILL_ROUGH_TASK: MillRoughTaskConfig(
        task_name=MILL_ROUGH_TASK,
    )
    },
    workflow_config=AutoLamellaWorkflowConfig(tasks=[
        AutoLamellaTaskDescription(name=ACQUIRE_REFERENCE_TASK, supervise=False, required=False),
        AutoLamellaTaskDescription(name=SPOT_BURN_TASK, supervise=False, required=False),
        AutoLamellaTaskDescription(name=FLUORESCENCE_TASK, supervise=False, required=False),
        AutoLamellaTaskDescription(name=MILL_FIDUCIAL_TASK, supervise=True, required=True),
        AutoLamellaTaskDescription(name=MILL_ROUGH_TASK, supervise=True, required=True, requires=[MILL_FIDUCIAL_TASK])
        ])
        )

    exp.task_protocol = protocol


    df_exp = exp.experiment_summary_dataframe()
    df_task_history = exp.task_history_dataframe()

    # display(df_exp)
    # display(df_task_history)

    for p in exp.positions:
        print(f"Lamella {p.name}, {p.last_completed_task.completed}, {p.milling_angle:.01f} degrees")
        print(f"Completed Tasks: {exp.task_protocol.workflow_config.get_completed_tasks(p)}")
        print(f"Workflow Complete: {exp.task_protocol.workflow_config.is_completed(p)}")


    exp.positions[1].defect.has_defect = True
    exp.positions[1].defect.description = "Milling stopped due to low current."
    exp.positions[1].defect.updated_at = datetime.now().timestamp()
    widget = AutoLamellaWorkflowDisplayWidget(experiment=exp)
    widget.show()
    return widget


if __name__ == "__main__":
    import sys

    import napari
    viewer = napari.Viewer()
    w = main()
    viewer.window.add_dock_widget(w, area='right', add_vertical_stretch=False)
    napari.run()


    # sys.exit(app.exec_())
