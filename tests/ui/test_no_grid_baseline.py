"""The base workflow with no grids at all is what it was before the grid work.

A plain Demo session on a fixed holder with nothing calibrated: a lamella made
there is linked to no grid, and every grid-aware surface added by the
lamella↔grid association stack stays out of the way -- no grid name on any row,
nothing withheld in the row menus, no filter offered, every lamella marked on
the Overview canvas, and Run reaching the confirmation with no refusal.
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
def main_ui(qapp):
    """The default Demo configuration: not a compustage, no loader, a fixed
    holder with no slot calibrated. Nothing about grids is set up."""
    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    stage = ui.microscope._stage
    assert stage.loader is None, "the default Demo grew a loader"
    yield window
    ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def test_nothing_grid_aware_gets_in_the_way(main_ui, tmp_path, monkeypatch):
    ui = main_ui.autolamella_ui
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
    assert exp.grids == []

    for i in range(2):
        ui.add_new_lamella(
            stage_position=FibsemStagePosition(x=i * 1e-4, y=0, z=0, r=0, t=0)
        )
    main_ui._on_experiment_update()
    assert [p.grid_id for p in exp.positions] == [None, None]

    # The rows: no grid named, nothing withheld, no filter on offer.
    for card in main_ui.lamella_card_container._cards.values():
        assert not card._grid_label.isVisibleTo(card)
        assert card._action_move.isEnabled() and card._action_update.isEnabled()
        assert card._action_move.text() == "Move to Position"
    workflow_list = main_ui.lamella_list_widget
    for i in range(2):
        row = workflow_list._row(i)
        assert not row.grid_label.isVisibleTo(row)
    assert not workflow_list.grid_filter.isVisibleTo(workflow_list)
    for row in ui.lamella_list._rows():
        assert not row.grid_label.isVisibleTo(row)
        assert row.action_move_to.isEnabled() and row.action_update.isEnabled()

    # The Overview canvas marks every lamella.
    beam = main_ui.beam_overview_tab
    if beam.overview is not None:
        beam.refresh_positions()
        assert len(beam.overview._positions) == 2

    # Run: no refusal, straight to the confirmation.
    workflow_list.set_all_selected(True)
    main_ui.lamella_workflow_widget.workflow.set_all_selected(True)
    assert len(main_ui.lamella_workflow_widget.get_selected_lamella()) == 2
    warnings, confirmations = [], []
    monkeypatch.setattr(
        module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2])
    )
    monkeypatch.setattr(
        module,
        "confirm_run_workflow_dialog",
        lambda *a, **k: (confirmations.append(a), False)[1],
    )
    main_ui._on_run_workflow_clicked()
    assert warnings == []
    assert len(confirmations) == 1
