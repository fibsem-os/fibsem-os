"""The workflow edits the window and the agent share: the config's rules for
supervision and the schedule, and ``workflow_edits`` saving and recording."""

from datetime import datetime, timedelta, timezone

import pytest

from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflow_edits import (
    WORKFLOW_CHANGED,
    set_supervision,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.session import AppSession
from fibsem.workflows.task_config import SCHEDULE_DAYS_AHEAD
from tests.protocols import planned, task_protocol


def _protocol() -> AutoLamellaTaskProtocol:
    return task_protocol(*(planned(name=n) for n in ("Rough", "Polish")))


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    experiment = Experiment(path=tmp_path, name="edits")
    (tmp_path / "edits").mkdir()
    experiment.task_protocol = _protocol()
    return experiment


def test_set_attention_refuses_an_unknown_task():
    protocol = _protocol()
    assert "No task named 'Mill'" in protocol.set_attention(
        "Mill", Attention.supervised
    )
    assert protocol.set_attention("Polish", Attention.supervised) == ""
    assert protocol.get_attention("Polish") is Attention.supervised


def test_a_schedule_past_the_window_is_refused_and_changes_nothing():
    config = _protocol().workflow_config
    late = datetime.now() + timedelta(days=SCHEDULE_DAYS_AHEAD, hours=1)
    why = config.set_schedule("Polish", late)
    assert f"up to {SCHEDULE_DAYS_AHEAD} days" in why
    assert config.get_scheduled_at("Polish") is None


@pytest.mark.parametrize(
    "when",
    [
        datetime.now() + timedelta(hours=3),
        datetime.now(timezone.utc) + timedelta(days=1),
        datetime.now() - timedelta(minutes=5),
        None,
    ],
    ids=["naive", "aware", "past", "cleared"],
)
def test_a_schedule_inside_the_window_is_set(when):
    config = _protocol().workflow_config
    assert config.set_schedule("Polish", when) == ""
    assert config.get_scheduled_at("Polish") == when


def test_set_supervision_saves_and_records(experiment):
    recorded = []

    result = set_supervision(
        experiment, "Polish", True, lambda kind, payload: recorded.append(kind)
    )

    assert result["applied"] and result["saved"]
    assert experiment.task_protocol.get_attention("Polish") is Attention.supervised
    assert recorded == [WORKFLOW_CHANGED]
    reloaded = AutoLamellaTaskProtocol.load(f"{experiment.path}/protocol.yaml")
    assert reloaded.get_attention("Polish") is Attention.supervised


def test_the_agent_and_the_status_bar_chip_set_supervision_alike(
    experiment, monkeypatch
):
    """The agent's set_supervision is the shared function; the window's chip
    calls it too (checked by reading the window, which needs napari)."""
    from pathlib import Path

    import fibsem

    calls = []
    import fibsem.applications.autolamella.workflow_edits as edits

    real = edits.set_supervision
    monkeypatch.setattr(
        edits, "set_supervision", lambda *a: calls.append(a[1:3]) or real(*a)
    )
    session = AppSession()
    session.experiment = experiment
    assert AutoLamellaAppContext(session).set_supervision("Rough", True)["applied"]
    assert calls == [("Rough", True)]

    window = (
        Path(fibsem.__path__[0]) / "applications/autolamella/ui/window.py"
    ).read_text(encoding="utf-8")
    start = window.index("    def _on_supervised_status_clicked(")
    body = window[start : window.index("\n    def ", start + 1)]
    assert "set_supervision(" in body
    assert "self.session.record_event" in body


def test_the_agent_schedule_is_refused_past_the_window(experiment):
    session = AppSession()
    session.experiment = experiment
    late = (datetime.now() + timedelta(days=SCHEDULE_DAYS_AHEAD + 1)).isoformat()

    refused = AutoLamellaAppContext(session).set_task_schedule("Polish", late)

    assert refused["applied"] is False
    assert "days from now" in refused["error"]
    assert experiment.task_protocol.workflow_config.get_scheduled_at("Polish") is None
