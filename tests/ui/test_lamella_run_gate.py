"""Run and Add-to-queue refuse a lamella whose grid is off the stage (FIB-71).

The window reads the inventory, names the lamellae and the grid, and starts
nothing; there is no override. Once the grid is loaded the same click goes on
to the confirmation as before.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from fibsem.structures import FibsemStagePosition


@pytest.fixture
def main_ui(qapp, monkeypatch):
    import fibsem.config as fibsem_config

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    monkeypatch.setattr(
        ui.system_widget, "load_configuration", lambda configuration_name=None: config
    )
    ui.system_widget.connect_to_microscope()
    yield window
    ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def test_run_is_refused_by_name_until_the_grid_is_loaded(
    main_ui, tmp_path, monkeypatch
):
    ui = main_ui.autolamella_ui
    stage = ui.microscope._stage
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name="Trench", supervise=False, required=False
                )
            ]
        ),
        task_config={"Trench": AcquireReferenceImageConfig(task_name="Trench")},
    )
    ui.experiment = exp
    exp.sync_grids_from_inventory(stage)
    grid = exp.get_grid_by_name("Grid-02")
    ui.add_new_lamella(
        stage_position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0),
        name="on-two",
        grid_id=grid.id,
    )
    main_ui._on_experiment_update()
    main_ui.lamella_workflow_widget.lamella_list.set_all_selected(True)
    main_ui.lamella_workflow_widget.workflow.set_all_selected(True)
    assert main_ui.lamella_workflow_widget.get_selected_lamella()
    assert main_ui.lamella_workflow_widget.get_selected_tasks()

    warnings, confirmations, starts = [], [], []
    monkeypatch.setattr(
        module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2])
    )
    monkeypatch.setattr(
        module,
        "confirm_run_workflow_dialog",
        lambda *a, **k: (confirmations.append(a), False)[1],
    )
    monkeypatch.setattr(
        ui, "_start_run_workflow_thread", lambda *a, **k: starts.append(a)
    )

    main_ui._on_run_workflow_clicked()
    assert warnings == [
        "Cannot run: on-two is on Grid-02, which is not on the stage. "
        "Load Grid-02 from the Grids tab first, or leave that lamella out."
    ]
    assert confirmations == [] and starts == []

    stage.ensure_loaded("Grid-02")
    main_ui._on_run_workflow_clicked()
    assert len(warnings) == 1  # no new refusal
    assert len(confirmations) == 1  # went on to the confirmation, which declined
    assert starts == []
