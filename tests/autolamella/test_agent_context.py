"""Tests for AgentContext: the read-side facade the embedded server will use.

Real domain objects throughout — a real Experiment with real lamellae, the real
TaskQueue mechanics, the Demo microscope — held by a plain host object standing
where AutoLamellaUI stands in production (the real-window pairing lives in
tests/ui/test_agent_context_host.py)."""

import json
import os

import pytest
from psygnal.containers import EventedDict

from fibsem import utils
from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import MicroscopeState


class Host:
    """The four attributes the facade resolves through; all start empty."""

    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path / "exp", name="agent-context-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


def _assert_json_safe(payload):
    json.dumps(payload)  # raises on anything a wire client couldn't carry


def test_every_method_tolerates_an_empty_host():
    ctx = AgentContext(Host())
    payloads = [
        ctx.status(),
        ctx.queue(),
        ctx.experiment_summary(),
        ctx.task_history(),
        ctx.run_summary(),
        ctx.protocol(),
        ctx.task_outputs("anything"),
        ctx.stage_position(),
    ]
    for payload in payloads:
        _assert_json_safe(payload)
    assert ctx.status()["experiment"] is None
    assert ctx.queue() == {"available": False, "items": [], "version": None}


def test_resolution_is_call_time_not_construction_time(experiment):
    # The app rebinds `experiment` on load; a facade built earlier must see the
    # new binding, not the one from construction (the ScriptContext hazard).
    host = Host()
    ctx = AgentContext(host)
    assert ctx.status()["experiment"] is None
    host.experiment = experiment
    assert ctx.status()["experiment"]["name"] == "agent-context-exp"
    host.experiment = None
    assert ctx.status()["experiment"] is None


def test_status_and_summaries_reflect_a_real_experiment(experiment):
    host = Host()
    host.experiment = experiment
    ctx = AgentContext(host)

    status = ctx.status()
    assert status["experiment"]["num_items"] == 2
    assert status["workflow"]["running"] is False

    summary = ctx.experiment_summary()
    assert summary["available"] is True
    assert len(summary["items"]) == 2
    _assert_json_safe(summary)

    protocol = ctx.protocol()
    assert protocol["available"] is True
    _assert_json_safe(protocol)


def test_task_outputs_for_a_real_item_and_a_missing_one(experiment):
    host = Host()
    host.experiment = experiment
    ctx = AgentContext(host)

    name = experiment.positions[0].name
    payload = ctx.task_outputs(name)
    assert payload["available"] is True
    assert payload["item_name"] == name
    assert payload["final_reference_images"] == []  # nothing has run
    _assert_json_safe(payload)

    # With history: the live-run 500 — the field is `name`, and an empty
    # history hid the read of a field that doesn't exist.
    from fibsem.applications.autolamella.structures import AutoLamellaTaskState

    experiment.positions[0].task_history.append(
        AutoLamellaTaskState(name="Rough Milling")
    )
    completed = ctx.task_outputs(name)
    assert completed["completed_tasks"] == ["Rough Milling"]
    _assert_json_safe(completed)

    missing = ctx.task_outputs("no-such-item")
    assert missing["available"] is False
    assert "no-such-item" in missing["error"]


def test_queue_snapshot_uses_item_name_vocabulary(experiment, microscope):
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    manager = TaskManager(microscope, experiment, parent_ui=None)
    names = [p.name for p in experiment.positions]
    manager.queue.build_from_matrix(["Task A", "Task B"], names)

    host = Host()
    host.experiment = experiment
    host._task_manager = manager
    ctx = AgentContext(host)

    queue = ctx.queue()
    assert queue["available"] is True
    assert len(queue["items"]) == 4  # 2 tasks x 2 items
    first = queue["items"][0]
    assert set(first) == {"id", "item_name", "task_name", "status"}
    assert "lamella_name" not in first  # the deprecated alias never crosses the wire
    assert first["status"] == AutoLamellaTaskStatus.NotStarted.name
    assert queue["version"] == manager.queue.version
    _assert_json_safe(queue)

    status = ctx.status()
    assert status["workflow"]["queue_total"] == 4
    assert status["workflow"]["current_task"] is None  # nothing running


def test_stage_position_reads_the_cache_only(microscope):
    host = Host()
    host.microscope = microscope
    ctx = AgentContext(host)

    microscope.get_stage_position()  # populate the cache, as the app would
    cached = ctx.stage_position()
    assert cached["available"] is True
    _assert_json_safe(cached)

    # The facade must not have issued a hardware read itself: poking the cache
    # attribute directly proves which side it reads from.
    microscope._stage_position = None
    assert ctx.stage_position() == {"available": False, "position": None}


def test_recent_experiments_is_json_safe():
    ctx = AgentContext(Host())
    _assert_json_safe(ctx.recent_experiments())
