"""AgentContext against the real window.

The unit tests (tests/autolamella/test_agent_context.py) drive the facade over
real domain objects in a plain holder; this file proves the production host —
an actual AutoLamellaUI — satisfies the same contract, in the states the facade
will actually meet it: freshly constructed (nothing connected, nothing loaded)
and with an experiment adopted."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.structures import MicroscopeState


@pytest.fixture
def ui(qapp):
    window = AutoLamellaUI(parent_ui=None)
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_a_fresh_window_satisfies_the_host_contract(ui):
    ctx = AgentContext(ui)
    status = ctx.status()
    json.dumps(status)
    assert status["microscope_connected"] is False
    assert status["experiment"] is None
    assert status["workflow"]["running"] is False
    assert ctx.queue()["available"] is False
    assert ctx.run_summary()["available"] is False


def test_an_adopted_experiment_is_visible_through_the_facade(ui, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="host-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())

    ctx = AgentContext(ui)  # built BEFORE the experiment exists — call-time wins
    ui.experiment = exp

    status = ctx.status()
    assert status["experiment"]["name"] == "host-exp"
    assert status["experiment"]["num_items"] == 1
    assert ctx.task_outputs(exp.positions[0].name)["available"] is True


def test_run_summary_survives_the_post_run_dialog(ui, monkeypatch):
    # The live-run regression: _show_workflow_summary used to consume
    # _last_run_summary to make the dialog once-only, so a remote read after
    # the run always found nothing. The dialog stays once-only; the record
    # stays readable.
    import pandas as pd

    from fibsem.applications.autolamella.ui import AutoLamellaUI as ui_module

    shown = []

    class _Dialog:
        def __init__(self, summary, parent=None):
            shown.append(summary)

        def exec_(self):
            return 0

    monkeypatch.setattr(ui_module, "WorkflowSummaryDialog", _Dialog)
    ui._last_run_summary = pd.DataFrame(
        [
            {
                "lamella_name": "01",
                "task_name": "Rough Milling",
                "task_status": "Finished",
            }
        ]
    )

    ui._show_workflow_summary()
    ui._show_workflow_summary()  # a second finished-signal must not re-show
    assert len(shown) == 1

    payload = AgentContext(ui).run_summary()
    assert payload["available"] is True
    assert payload["items"][0]["task_name"] == "Rough Milling"
    json.dumps(payload)

    # The next run's capture is a new object: the dialog shows again.
    ui._last_run_summary = pd.DataFrame([{"task_name": "Polishing"}])
    ui._show_workflow_summary()
    assert len(shown) == 2
