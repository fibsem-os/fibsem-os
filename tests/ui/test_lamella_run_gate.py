"""Run and Add-to-queue refuse a lamella whose grid is off the stage (FIB-71).

The window reads the inventory, names the lamellae and the grid, and starts
nothing; there is no override. Once the grid is loaded the same click goes on
to the confirmation as before.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.ui.workflow import tab as workflow_tab
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.runner import ITEMS
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from tests.protocols import planned, task_protocol
from tests.run_state import hold_runs, started_with

main_ui = main_window_fixture(connect=True, config="sim-arctis-configuration.yaml")


def test_run_is_refused_by_name_until_the_grid_is_loaded(
    main_ui, tmp_path, monkeypatch
):
    ui = main_ui
    stage = ui.microscope.sample_handling
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = task_protocol(
        planned(name="Trench", required=False),
        task_config={"Trench": AcquireReferenceImageConfig(task_name="Trench")},
    )
    ui.experiment = exp
    exp.sync_grids_from_inventory(stage)
    grid = exp.get_grid_by_name("Grid-02")
    ui.panel.experiment_tab.add_new_lamella(
        stage_position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0),
        name="on-two",
        grid_id=grid.id,
    )
    main_ui._on_experiment_update()
    main_ui.workflow_tab.lamella_workflow_widget.lamella_list.set_all_selected(True)
    main_ui.workflow_tab.lamella_workflow_widget.workflow.set_all_selected(True)
    assert main_ui.workflow_tab.lamella_workflow_widget.get_selected_lamella()
    assert main_ui.workflow_tab.lamella_workflow_widget.get_selected_tasks()

    warnings, confirmations = [], []
    monkeypatch.setattr(
        workflow_tab.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2])
    )
    monkeypatch.setattr(
        workflow_tab,
        "confirm_run_workflow_dialog",
        lambda *a, **k: (confirmations.append(a), False)[1],
    )
    held = hold_runs(monkeypatch, ITEMS)

    main_ui.workflow_tab.run()
    assert warnings == [
        "Cannot run: on-two is on Grid-02, which is not on the stage. "
        "Load Grid-02 first, or leave that lamella out."
    ]
    assert confirmations == [] and started_with(held) == []

    stage.ensure_loaded("Grid-02")
    main_ui.workflow_tab.run()
    assert len(warnings) == 1  # no new refusal
    assert len(confirmations) == 1  # went on to the confirmation, which declined
    assert started_with(held) == []
