"""The workflow row's attention chip: one control for when a person is
involved in a task, toggling Automated ↔ Supervised.

There is no Agent step (FIB-1035): a supervised question goes to the operator
and any connected agent alike."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.workflow import task_list as module
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import RunPlan, TaskPlan
from tests.protocols import planned, task_protocol


@pytest.fixture
def row(qapp):
    widget = module.WorkflowTaskRowWidget(*planned(name="Mill Fiducial"))
    yield widget
    widget.deleteLater()
    qapp.processEvents()


A = Attention


def test_the_chip_shows_the_configs_attention(row):
    assert row.btn_attention.text() == "Automated"
    row.config.attention = A.supervised
    row.refresh()
    assert row.btn_attention.text() == "Supervised"
    assert row.btn_attention.toolTip().startswith("Supervised")
    row.config.attention = A.automated
    row.refresh()
    assert row.btn_attention.text() == "Automated"


def test_each_click_asks_for_a_toggle_and_changes_nothing(row):
    """The host sets supervision through the function the agent uses too."""
    seen = []
    row.attention_toggled.connect(seen.append)
    for _ in range(3):
        row.btn_attention.click()
    assert seen == [row.task] * 3
    assert row.config.attention is A.automated


def test_a_requirement_reads_the_same_whatever_its_mode(qapp):
    """Nothing waits on a decision once a task ends, so a requirement on a
    supervised task is said like any other."""
    protocol = task_protocol(
        planned(name="Setup", attention=Attention.supervised),
        planned(name="Rough", attention=Attention.supervised),
    )
    widget = module.WorkflowTaskList()
    widget.set_protocol(protocol)
    widget.set_config(protocol.workflow_config)
    rows = [widget._row(i) for i in range(2)]
    assert rows[1].requires_label.text() == ""
    protocol.task_config["Rough"].requires = ["Setup"]
    widget.refresh_all()
    assert rows[1].requires_label.text() == "after Setup"
    assert rows[1].toolTip().startswith("Requires: Setup")
    protocol.task_config["Setup"].attention = Attention.automated
    widget.refresh_all()
    assert rows[1].requires_label.text() == "after Setup"
    widget.deleteLater()
    qapp.processEvents()


def test_a_schedule_shares_the_dependency_column(qapp):
    """The clock button only duplicated the pencil; the row now says when in
    the same column as what it waits for, so a scheduled row is not a wider
    row, and says nothing at all when there is no schedule."""
    from datetime import datetime

    task, config = planned(name="Rough", requires=["Fiducial"])
    row = module.WorkflowTaskRowWidget(task, config)
    assert row.requires_label.text() == "after Fiducial"
    assert not hasattr(row, "btn_schedule")
    task.scheduled_at = datetime(2026, 9, 15, 21, 30)
    row.refresh()
    assert row.requires_label.text() == "after Fiducial · at 15 Sep 21:30"
    assert "Scheduled:" in row.toolTip() and "Requires:" in row.toolTip()
    config.requires = []
    row.refresh()
    assert row.requires_label.text() == "at 15 Sep 21:30"
    row.deleteLater()
    qapp.processEvents()


def test_remove_lives_in_the_edit_dialog_not_on_the_row(qapp, monkeypatch):
    """A trash can on every row was the one thing there nobody pressed. The
    edit dialog offers Remove task…, confirms, closes, and the list drops the
    row and announces it on the same signal the hosts already listen to."""
    from PyQt5.QtWidgets import QMessageBox

    from fibsem.applications.autolamella.ui.workflow.lamella_run import (
        LamellaWorkflowWidget,
    )

    setup = TaskPlan(name="Setup")
    rough = TaskPlan(name="Rough")
    widget = LamellaWorkflowWidget()
    widget.workflow.set_config(RunPlan(tasks=[setup, rough]))
    row = widget.workflow._row(0)
    assert not hasattr(row, "btn_remove")
    removed = []
    widget.task_remove_requested.connect(removed.append)

    # No is no
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    widget._on_task_edit_requested(setup)
    dialog = widget._editor_dialog
    assert dialog.editor._remove_btn.isVisibleTo(dialog)
    dialog.editor._remove_btn.click()
    assert removed == [] and [t.name for t in widget.workflow.get_tasks()] == [
        "Setup",
        "Rough",
    ]
    dialog.reject()

    # Yes removes, closes the dialog, and announces the task
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    widget._on_task_edit_requested(setup)
    dialog.editor._remove_btn.click()
    qapp.processEvents()
    assert removed == [setup]
    assert [t.name for t in widget.workflow.get_tasks()] == ["Rough"]
    assert not dialog.isVisible()

    # a host that never removes hides the button
    widget.workflow.enable_remove_button(False)
    widget._on_task_edit_requested(rough)
    assert not dialog.editor._remove_btn.isVisibleTo(dialog)
    dialog.reject()
    widget.deleteLater()
    qapp.processEvents()


def test_the_chip_says_what_the_task_needs_from_you(qapp):
    """With the protocol known, the chip's tooltip and the row's say whether
    the operator must be there while the task runs and what waits afterwards,
    derived from the task's type."""
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )
    from fibsem.applications.autolamella.workflows.tasks.undercut import (
        MillUndercutTaskConfig,
    )

    protocol = task_protocol(
        planned(
            name="Undercut",
            attention=Attention.supervised,
            config=MillUndercutTaskConfig(),
        ),
        planned(
            name="Rough",
            attention=Attention.automated,
            requires=["Undercut"],
            config=MillRoughTaskConfig(),
        ),
    )
    widget = module.WorkflowTaskList()
    widget.set_protocol(protocol)
    widget.set_config(protocol.workflow_config)
    rows = [widget._row(i) for i in range(2)]
    tip = rows[0].btn_attention.toolTip()
    assert tip.startswith("Supervised — Needs you at the microscope while it runs: ")
    assert "detection and milling" in tip
    assert "waits for your decision" not in tip, "nothing waits once it ends"
    assert "Needs you at the microscope" in rows[0].toolTip()
    assert rows[1].btn_attention.toolTip().startswith("Automated — Nobody is asked.")
    # Without a protocol nothing is claimed.
    widget.set_protocol(None)
    assert "Needs you" not in rows[0].btn_attention.toolTip()
    widget.deleteLater()
    qapp.processEvents()
