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
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.ui import workflow_config_widget as module


@pytest.fixture
def row(qapp, monkeypatch):
    monkeypatch.setattr(module, "_review_available", lambda: False)
    task = AutoLamellaTaskDescription(
        name="Mill Fiducial", supervise=False, required=True
    )
    widget = module.WorkflowTaskRowWidget(task)
    yield widget
    widget.deleteLater()
    qapp.processEvents()


def _fields(task):
    return (task.supervise, task.supervisor, task.review)


def test_without_either_feature_the_chip_is_two_state(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    assert row.btn_attention.text() == "Automated"
    row._on_attention_clicked()
    assert _fields(row.task) == (True, "human", False)
    assert row.btn_attention.text() == "Supervised"
    row._on_attention_clicked()
    assert _fields(row.task) == (False, "human", False)
    assert row.btn_attention.text() == "Automated"


def test_with_the_agent_feature_the_cycle_gains_the_agent_step(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    row._on_attention_clicked()
    assert _fields(row.task) == (True, "human", False)
    assert row.btn_attention.toolTip().startswith("Supervised")
    row._on_attention_clicked()
    assert _fields(row.task) == (True, "agent", False)
    assert row.btn_attention.text() == "Agent"
    row._on_attention_clicked()
    # Leaving the agent state resets the designation: nothing hidden survives.
    assert _fields(row.task) == (False, "human", False)
    assert row.btn_attention.text() == "Automated"


def test_with_interactive_review_the_cycle_ends_in_review(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    monkeypatch.setattr(module, "_review_available", lambda: True)
    row._on_attention_clicked()
    assert _fields(row.task) == (True, "human", False)
    row._on_attention_clicked()
    assert _fields(row.task) == (False, "human", True), "review clears supervise"
    assert row.btn_attention.text() == "Review"
    row._on_attention_clicked()
    assert _fields(row.task) == (False, "human", False)


def test_a_designated_task_displays_as_supervised_when_the_feature_is_off(
    row, monkeypatch
):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    row.task.supervise = True
    row.task.supervisor = "agent"
    row.refresh()
    assert row.btn_attention.text() == "Supervised"
    # And the click matches what is displayed: Supervised → Automated.
    row._on_attention_clicked()
    assert _fields(row.task) == (False, "human", False)


def test_a_gated_task_displays_as_automated_when_interactive_review_is_off(
    row, monkeypatch
):
    row.task.review = True
    row.refresh()
    assert row.btn_attention.text() == "Automated"
    assert "interactive review is off" in row.btn_attention.toolTip()
    # the click writes what is displayed, then moves on: nothing hidden survives
    row._on_attention_clicked()
    assert _fields(row.task) == (True, "human", False)


def test_both_fields_on_reads_as_supervised(row, monkeypatch):
    """A combination the chip never writes; if you are at the microscope for
    it, you answer there. The next click writes it back cleanly."""
    monkeypatch.setattr(module, "_review_available", lambda: True)
    row.task.supervise = True
    row.task.review = True
    row.refresh()
    assert row.btn_attention.text() == "Supervised"
    row._on_attention_clicked()
    assert _fields(row.task) == (False, "human", True)


def test_each_click_announces_only_what_changed(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    monkeypatch.setattr(module, "_review_available", lambda: True)
    seen = []
    row.supervised_changed.connect(lambda t: seen.append(("supervise", t.supervisor)))
    row.review_changed.connect(lambda t: seen.append(("review", t.review)))
    for _ in range(4):
        row._on_attention_clicked()
    assert seen == [
        ("supervise", "human"),  # Automated -> Supervised
        ("supervise", "agent"),  # Supervised -> Agent
        ("supervise", "human"),  # Agent -> Review: supervise off ...
        ("review", True),  # ... and review on
        ("review", False),  # Review -> Automated
    ]


def test_a_review_task_nothing_requires_says_so(qapp, monkeypatch):
    monkeypatch.setattr(module, "_review_available", lambda: True)
    setup = AutoLamellaTaskDescription(
        name="Setup", supervise=False, required=True, review=True
    )
    rough = AutoLamellaTaskDescription(
        name="Rough", supervise=False, required=True, review=True
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
    assert rows[1].requires_label.text() == "after Setup", "its own dependency stays"
    assert module.stylesheets.WARN_COLOR in rows[1].requires_label.styleSheet()
    assert "nothing waits" in rows[1].btn_attention.toolTip()
    widget.deleteLater()
    qapp.processEvents()
