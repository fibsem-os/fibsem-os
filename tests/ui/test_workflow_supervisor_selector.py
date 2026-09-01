"""The workflow tab's supervision selector: a cycle, gated on the feature.

With the agent-server preference off the control is exactly the old two-state
toggle, and a stored ``supervisor: agent`` displays as plain Supervised — the
same hard-gate rule as the window chrome. With it on, a third step joins the
cycle: Automated → Supervised → Agent → Automated."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription
from fibsem.applications.autolamella.ui import workflow_config_widget as module


@pytest.fixture
def row(qapp):
    task = AutoLamellaTaskDescription(
        name="Mill Fiducial", supervise=False, required=True
    )
    widget = module.WorkflowTaskRowWidget(task)
    yield widget
    widget.deleteLater()
    qapp.processEvents()


def test_without_the_feature_the_toggle_is_two_state(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    row._on_supervise_clicked()
    assert (row.task.supervise, row.task.supervisor) == (True, "human")
    row._on_supervise_clicked()
    assert (row.task.supervise, row.task.supervisor) == (False, "human")


def test_with_the_feature_the_cycle_gains_the_agent_step(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    row._on_supervise_clicked()
    assert (row.task.supervise, row.task.supervisor) == (True, "human")
    assert row.btn_supervise.toolTip() == "Supervised"
    row._on_supervise_clicked()
    assert (row.task.supervise, row.task.supervisor) == (True, "agent")
    assert row.btn_supervise.toolTip().startswith("Agent")
    row._on_supervise_clicked()
    # Leaving the agent state resets the designation: nothing hidden survives.
    assert (row.task.supervise, row.task.supervisor) == (False, "human")
    assert row.btn_supervise.toolTip() == "Automated"


def test_a_designated_task_displays_as_supervised_when_the_feature_is_off(
    row, monkeypatch
):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: False)
    row.task.supervise = True
    row.task.supervisor = "agent"
    row.refresh()
    assert row.btn_supervise.toolTip() == "Supervised"
    # And the click matches what is displayed: Supervised → Automated.
    row._on_supervise_clicked()
    assert (row.task.supervise, row.task.supervisor) == (False, "human")


def test_each_click_announces_the_change(row, monkeypatch):
    monkeypatch.setattr(module, "_agent_supervision_available", lambda: True)
    seen = []
    row.supervised_changed.connect(lambda task: seen.append(task.supervisor))
    row._on_supervise_clicked()
    row._on_supervise_clicked()
    row._on_supervise_clicked()
    assert seen == ["human", "agent", "human"]
