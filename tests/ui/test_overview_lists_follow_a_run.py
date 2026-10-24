"""The lamella lists beside both Overview pages follow a running workflow (FIB-995).

A task's status report refreshed the Experiment list by name and left the two
Overview pages' lists -- which carry the same Status column -- stale for the
whole run.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.status import TaskStatus, WorkflowStatusUpdate

main_ui = main_window_fixture(connect=True, config="sim-arctis-configuration.yaml")


def _statuses(tab):
    return [row.status_label.text() for row in tab.lamella_list._rows()]


def test_a_status_report_reaches_both_overview_lists(main_ui, tmp_path):
    ui = main_ui
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui._on_experiment_update()
    for i, name in enumerate(("one", "two")):
        ui.panel.experiment_tab.add_new_lamella(
            stage_position=FibsemStagePosition(x=i * 1e-4, y=0, z=0, r=0, t=0),
            name=name,
        )
    beam, fm = main_ui.overview_tab.beam_tab, main_ui.overview_tab.fm_tab
    assert beam.lamella_list._list.count() == 2 == fm.lamella_list._list.count()
    assert _statuses(beam) == ["", ""] == _statuses(fm)

    # What the workflow writes, then reports, for a running task.
    one = exp.positions[0]
    one.task_state.name = "Trench"
    one.task_state.status = TaskStatus.InProgress
    main_ui._apply_status_report(
        WorkflowStatusUpdate(
            task_name="Trench",
            item_name="one",
            status=TaskStatus.InProgress,
            queue_position=1,
            queue_total=2,
        )
    )
    assert _statuses(beam) == ["Trench", ""]
    assert _statuses(fm) == ["Trench", ""]
    assert [
        r.status_label.text() for r in ui.panel.experiment_tab.lamella_list._rows()
    ] == ["Trench", ""]
