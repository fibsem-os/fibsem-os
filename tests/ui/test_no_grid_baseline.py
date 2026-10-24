"""The base workflow with no grids at all is what it was before the grid work.

A plain Demo session on a fixed holder with nothing calibrated: a lamella made
there is linked to no grid, and every grid-aware surface added by the
lamella↔grid association stack stays out of the way -- no grid name on any row,
nothing withheld in the row menus, no filter offered, every lamella marked on
the Overview canvas, and Run reaching the confirmation with no refusal.
"""

import os

from tests.fixtures.main_window import close_window

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.ui import window as module
from fibsem.applications.autolamella.ui.workflow import tab as workflow_tab
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from tests.protocols import planned, task_protocol


@pytest.fixture
def main_ui(qapp):
    """The default Demo configuration: not a compustage, no loader, a fixed
    holder with no slot calibrated. Nothing about grids is set up."""
    window = module.AutoLamellaWindow()
    ui = window
    ui.system_widget.connect_to_microscope()
    stage = ui.microscope.sample_handling
    assert stage.magazine is None, "the default Demo grew a loader"
    yield window
    ui.microscope.disconnect()
    close_window(window)


def test_nothing_grid_aware_gets_in_the_way(main_ui, tmp_path, monkeypatch):
    ui = main_ui
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = task_protocol(
        planned(name="Trench", required=False),
        task_config={"Trench": AcquireReferenceImageConfig(task_name="Trench")},
    )
    ui.experiment = exp
    assert exp.grids == []

    for i in range(2):
        ui.panel.experiment_tab.add_new_lamella(
            stage_position=FibsemStagePosition(x=i * 1e-4, y=0, z=0, r=0, t=0)
        )
    main_ui._on_experiment_update()
    assert [p.grid_id for p in exp.positions] == [None, None]

    # The rows: no grid named, nothing withheld, no filter on offer.
    for card in main_ui.lamella_tab.cards._cards.values():
        assert not card._grid_label.isVisibleTo(card)
        assert card._action_move.isEnabled() and card._action_update.isEnabled()
        assert card._action_move.text() == "Move to Position"
    workflow_list = main_ui.workflow_tab.lamella_workflow_widget.lamella_list
    for i in range(2):
        row = workflow_list._row(i)
        assert not row.grid_label.isVisibleTo(row)
    assert not workflow_list.grid_filter.isVisibleTo(workflow_list)
    for row in ui.panel.experiment_tab.lamella_list._rows():
        assert not row.grid_label.isVisibleTo(row)
        assert row.action_move_to.isEnabled() and row.action_update.isEnabled()

    # The Overview canvas marks every lamella.
    beam = main_ui.overview_tab.beam_tab
    if beam.overview is not None:
        beam.refresh_positions()
        assert len(beam.overview.session.positions) == 2

    # Run: no refusal, straight to the confirmation.
    workflow_list.set_all_selected(True)
    main_ui.workflow_tab.lamella_workflow_widget.workflow.set_all_selected(True)
    assert len(main_ui.workflow_tab.lamella_workflow_widget.get_selected_lamella()) == 2
    warnings, confirmations = [], []
    monkeypatch.setattr(
        workflow_tab.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2])
    )
    monkeypatch.setattr(
        workflow_tab,
        "confirm_run_workflow_dialog",
        lambda *a, **k: (confirmations.append(a), False)[1],
    )
    main_ui.workflow_tab.run()
    assert warnings == []
    assert len(confirmations) == 1
