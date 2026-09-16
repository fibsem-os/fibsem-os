"""The workflow row's attention chip: one control for when a person is
involved in a task, cycling Automated → Supervised → (Agent) → (Review).

The Agent step exists only while the agent-server preference is on and the
Review step only while interactive review is on; with neither this is the old
two-state toggle. A stored ``supervisor: agent`` displays as plain Supervised
with the feature off, and a stored ``review: true`` as Automated -- the state
the task will actually run in, not the one in the file."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.ui import workflow_config_widget as module


@pytest.fixture
def row(qapp, monkeypatch):
    monkeypatch.setattr(module, "_review_available", lambda: False)
    task = AutoLamellaTaskDescription(name="Mill Fiducial", required=True)
    widget = module.WorkflowTaskRowWidget(task)
    yield widget
    widget.deleteLater()
    qapp.processEvents()


A = Attention


def _fields(task):
    return (task.attention, task.supervisor)


def test_without_either_feature_the_chip_is_two_state(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    assert row.btn_attention.text() == "Automated"
    row._on_attention_clicked()
    assert _fields(row.task) == (A.supervised, "human")
    assert row.btn_attention.text() == "Supervised"
    row._on_attention_clicked()
    assert _fields(row.task) == (A.automated, "human")
    assert row.btn_attention.text() == "Automated"


def test_with_the_agent_feature_the_cycle_gains_the_agent_step(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    row._on_attention_clicked()
    assert _fields(row.task) == (A.supervised, "human")
    assert row.btn_attention.toolTip().startswith("Supervised")
    row._on_attention_clicked()
    assert _fields(row.task) == (A.supervised, "agent")
    assert row.btn_attention.text() == "Agent"
    row._on_attention_clicked()
    # Leaving the agent state resets the designation: nothing hidden survives.
    assert _fields(row.task) == (A.automated, "human")
    assert row.btn_attention.text() == "Automated"


def test_with_interactive_review_the_cycle_ends_in_review(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    monkeypatch.setattr(module, "_review_available", lambda: True)
    row._on_attention_clicked()
    assert _fields(row.task) == (A.supervised, "human")
    row._on_attention_clicked()
    assert _fields(row.task) == (A.review, "human"), "review clears supervise"
    assert row.btn_attention.text() == "Review"
    row._on_attention_clicked()
    assert _fields(row.task) == (A.automated, "human")


def test_a_designated_task_displays_as_supervised_when_the_feature_is_off(
    row, monkeypatch
):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    row.task.attention = Attention.supervised
    row.task.supervisor = "agent"
    row.refresh()
    assert row.btn_attention.text() == "Supervised"
    # And the click matches what is displayed: Supervised → Automated.
    row._on_attention_clicked()
    assert _fields(row.task) == (A.automated, "human")


def test_a_gated_task_displays_as_automated_when_interactive_review_is_off(
    row, monkeypatch
):
    row.task.attention = Attention.review
    row.refresh()
    assert row.btn_attention.text() == "Automated"
    assert "interactive review is off" in row.btn_attention.toolTip()
    # the click writes what is displayed, then moves on: nothing hidden survives
    row._on_attention_clicked()
    assert _fields(row.task) == (A.supervised, "human")


def test_each_click_announces_only_what_changed(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    monkeypatch.setattr(module, "_review_available", lambda: True)
    seen = []
    row.attention_changed.connect(lambda t: seen.append(_fields(t)))
    for _ in range(4):
        row._on_attention_clicked()
    assert seen == [
        (A.supervised, "human"),  # Automated -> Supervised
        (A.supervised, "agent"),  # Supervised -> Agent
        (A.review, "human"),  # Agent -> Review: the designation resets
        (A.automated, "human"),  # Review -> Automated
    ]


def test_a_review_task_nothing_requires_says_so(qapp, monkeypatch):
    monkeypatch.setattr(module, "_review_available", lambda: True)
    setup = AutoLamellaTaskDescription(
        name="Setup", required=True, attention=Attention.review
    )
    rough = AutoLamellaTaskDescription(
        name="Rough", required=True, attention=Attention.review
    )
    widget = module.WorkflowConfigWidget()
    widget.set_config(AutoLamellaWorkflowConfig(tasks=[setup, rough]))
    rows = [widget._row(i) for i in range(2)]
    assert rows[0].requires_label.text() == "nothing waits on this"
    assert rows[1].requires_label.text() == "nothing waits on this"
    rough.requires = ["Setup"]
    widget.refresh_all()
    assert rows[0].requires_label.text() == ""
    assert "nothing waits" not in rows[0].btn_attention.toolTip()
    assert rows[1].requires_label.text() == "after review of Setup", (
        "the wait is said where it is felt"
    )
    assert rows[1].toolTip().startswith("Requires: review of Setup")
    assert module.stylesheets.WARN_COLOR in rows[1].requires_label.styleSheet()
    assert "nothing waits" in rows[1].btn_attention.toolTip()
    # Setup back to Automated: Rough's row stops saying it waits for a review
    rows[0]._on_attention_clicked()
    assert rows[1].requires_label.text() == "after Setup"
    widget.deleteLater()
    qapp.processEvents()


def test_the_requires_phrase_leads_with_the_review():
    """Task names are long and the column elides the tail, so the marker goes
    first where every requirement is reviewed."""
    assert module._requires_phrase(["Setup"], {"Setup"}) == "review of Setup"
    assert module._requires_phrase(["Setup", "Rough"], {"Setup", "Rough"}) == (
        "review of Setup, Rough"
    )
    assert module._requires_phrase(["Setup", "Fiducial"], {"Setup"}) == (
        "Setup (reviewed), Fiducial"
    )
    assert module._requires_phrase(["Setup"], set()) == "Setup"


def test_a_schedule_shares_the_dependency_column(qapp, monkeypatch):
    """The clock button only duplicated the pencil; the row now says when in
    the same column as what it waits for, so a scheduled row is not a wider
    row, and says nothing at all when there is no schedule."""
    from datetime import datetime

    monkeypatch.setattr(module, "_review_available", lambda: False)
    task = AutoLamellaTaskDescription(
        name="Rough", required=True, requires=["Fiducial"]
    )
    row = module.WorkflowTaskRowWidget(task)
    assert row.requires_label.text() == "after Fiducial"
    assert not hasattr(row, "btn_schedule")
    task.scheduled_at = datetime(2026, 9, 15, 21, 30)
    row.refresh()
    assert row.requires_label.text() == "after Fiducial · at 15 Sep 21:30"
    assert "Scheduled:" in row.toolTip() and "Requires:" in row.toolTip()
    task.requires = []
    row.refresh()
    assert row.requires_label.text() == "at 15 Sep 21:30"
    row.set_schedule_visible(False)
    assert row.requires_label.text() == ""
    row.deleteLater()
    qapp.processEvents()


def test_remove_lives_in_the_edit_dialog_not_on_the_row(qapp, monkeypatch):
    """A trash can on every row was the one thing there nobody pressed. The
    edit dialog offers Remove task…, confirms, closes, and the list drops the
    row and announces it on the same signal the hosts already listen to."""
    from PyQt5.QtWidgets import QMessageBox

    from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
        LamellaWorkflowWidget,
    )

    monkeypatch.setattr(module, "_review_available", lambda: False)
    setup = AutoLamellaTaskDescription(name="Setup", required=True)
    rough = AutoLamellaTaskDescription(name="Rough", required=True)
    widget = LamellaWorkflowWidget()
    widget.workflow.set_config(AutoLamellaWorkflowConfig(tasks=[setup, rough]))
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
