"""Tests for AutoLamellaAppContext: the read-side facade the embedded server will use.

Real domain objects throughout — a real Experiment with real lamellae, the real
TaskQueue mechanics, the Demo microscope — held by a plain host object standing
where AutoLamellaUI stands in production (the real-window pairing lives in
tests/ui/test_agent_context_host.py)."""

import json
import os

import pytest
from psygnal.containers import EventedDict

from fibsem import utils
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures import MicroscopeState
from fibsem.workflows.status import TaskStatus


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
    ctx = AutoLamellaAppContext(Host())
    payloads = [
        ctx.status(),
        ctx.queue(),
        ctx.experiment_summary(),
        ctx.task_history(),
        ctx.run_summary(),
        ctx.protocol(),
        ctx.task_outputs("anything"),
        ctx.item_detail("anything"),
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
    ctx = AutoLamellaAppContext(host)
    assert ctx.status()["experiment"] is None
    host.experiment = experiment
    assert ctx.status()["experiment"]["name"] == "agent-context-exp"
    host.experiment = None
    assert ctx.status()["experiment"] is None


def test_status_and_summaries_reflect_a_real_experiment(experiment):
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

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
    ctx = AutoLamellaAppContext(host)

    name = experiment.positions[0].name
    payload = ctx.task_outputs(name)
    assert payload["available"] is True
    assert payload["item_name"] == name
    assert payload["final_reference_images"] == []  # nothing has run
    _assert_json_safe(payload)

    # With history: the live-run 500 — the field is `name`, and an empty
    # history hid the read of a field that doesn't exist.
    from fibsem.applications.autolamella.structures import AutoLamellaTaskState
    from fibsem.workflows.status import TaskStatus

    experiment.positions[0].task_history.append(
        AutoLamellaTaskState(name="Rough Milling", status=TaskStatus.Completed)
    )
    completed = ctx.task_outputs(name)
    assert completed["completed_tasks"] == ["Rough Milling"]
    _assert_json_safe(completed)

    missing = ctx.task_outputs("no-such-item")
    assert missing["available"] is False
    assert "no-such-item" in missing["error"]


def test_output_image_renders_recorded_outputs_only(experiment):
    import numpy as np
    import tifffile

    from fibsem.applications.autolamella.structures import AutoLamellaTaskState

    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    lamella = experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    fname = "ref_Rough Milling_final_res_01_ib.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, fname),
        (np.random.rand(48, 64) * 65535).astype(np.uint16),
    )
    # A z-stack fluorescence output: projected, not rejected.
    zname = "fm_stack.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, zname),
        (np.random.rand(5, 48, 64) * 65535).astype(np.uint16),
        photometric="minisblack",
    )
    lamella.task_history.append(
        AutoLamellaTaskState(
            name="Rough Milling",
            outputs={"final_fib": [fname], "fluorescence": [zname]},
        )
    )

    served = ctx.output_image(lamella.name, fname)
    assert served["jpeg"][:2] == b"\xff\xd8"
    assert (served["width"], served["height"]) == (64, 48)

    stack = ctx.output_image(lamella.name, zname)
    assert stack["jpeg"][:2] == b"\xff\xd8"

    unknown = ctx.output_image(lamella.name, "not-listed.tif")
    assert unknown["jpeg"] is None
    assert sorted(unknown["filenames"]) == [zname, fname]

    missing = ctx.output_image("no-such-item", fname)
    assert missing["available"] is False

    # task_outputs keeps the per-run grouping: which task produced which file.
    payload = ctx.task_outputs(lamella.name)
    assert payload["tasks"] == [
        {
            "name": "Rough Milling",
            "status": "NotStarted",
            "files": {"final_fib": [fname], "fluorescence": [zname]},
        }
    ]


def test_task_config_documents_carry_the_full_config_and_a_version(experiment):
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    config = MillRoughTaskConfig(task_name="Rough Milling")
    experiment.task_protocol.task_config["Rough Milling"] = config
    lamella = experiment.positions[0]
    lamella.task_config["Rough Milling"] = config

    doc = ctx.protocol_task_config("Rough Milling")
    assert doc["available"] is True
    assert doc["level"] == "protocol"
    assert doc["config"]["task_type"] == config.to_dict()["task_type"]
    assert len(doc["version"]) == 16
    _assert_json_safe(doc)

    item_doc = ctx.item_task_config(lamella.name, "Rough Milling")
    assert item_doc["level"] == "item"
    assert item_doc["item_name"] == lamella.name
    # Same content hashes to the same version: the token names the state.
    assert item_doc["version"] == doc["version"]
    _assert_json_safe(item_doc)

    # A content change changes the version — the write side's stale check.
    config.sync_polishing_position = not config.sync_polishing_position
    assert ctx.protocol_task_config("Rough Milling")["version"] != doc["version"]

    unknown = ctx.protocol_task_config("No Such Task")
    assert unknown["available"] is True
    assert "task_names" in unknown
    missing_item = ctx.item_task_config("no-such-item", "Rough Milling")
    assert missing_item["available"] is False
    no_config = ctx.item_task_config(lamella.name, "No Such Task")
    assert no_config["task_names"] == ["Rough Milling"]


def test_item_detail_serves_the_durable_item_facts(experiment):
    """One read for what a supervisor judges an item by — the exemplar-mode
    seam: after the operator answers, the accepted POI/alignment area are
    readable here rather than raced out of the live prompt mirror."""
    from fibsem.structures import FibsemRectangle, MicroscopeState, Point

    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    lamella = experiment.positions[0]
    lamella.poi = Point(1.5e-6, -2.5e-6)
    lamella.alignment_area = FibsemRectangle(left=0.6, top=0.3, width=0.3, height=0.4)
    lamella.milling_angle = 15.0
    lamella.milling_pose = MicroscopeState()

    detail = ctx.item_detail(lamella.name)
    assert detail["available"] is True
    assert detail["item_name"] == lamella.name
    assert detail["poi"] == {"x": 1.5e-6, "y": -2.5e-6}
    assert detail["alignment_area"]["left"] == 0.6
    assert detail["milling_angle"] == 15.0
    assert detail["is_failure"] is False
    assert "MILLING" in detail["poses"]
    _assert_json_safe(detail)

    missing = ctx.item_detail("no-such-item")
    assert missing["available"] is False
    assert "no-such-item" in missing["error"]


def test_stop_workflow_takes_the_stop_path_only_while_running():
    calls = []

    class StoppableHost(Host):
        def stop_task_workflow(self):
            calls.append("stop")

    host = StoppableHost()
    ctx = AutoLamellaAppContext(host)
    idle = ctx.stop_workflow()
    assert idle == {
        "available": True,
        "stopped": False,
        "reason": "no workflow is running",
    }
    assert calls == []

    host.is_workflow_running = True
    assert ctx.stop_workflow() == {"available": True, "stopped": True}
    assert calls == ["stop"]

    # A host without the stop path at all reads as unavailable, not an error.
    assert AutoLamellaAppContext(Host()).stop_workflow()["available"] is False


def test_set_supervision_flips_the_live_protocol(experiment):
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )

    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name="Rough Milling", required=True)
    )
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    applied = ctx.set_supervision("Rough Milling", True)
    assert applied["applied"] is True
    # The same read the workflow's decision points use sees the new value.
    assert (
        experiment.task_protocol.get_attention("Rough Milling") is Attention.supervised
    )

    unknown = ctx.set_supervision("No Such Task", True)
    assert unknown["applied"] is False
    assert unknown["task_names"] == ["Rough Milling"]
    _assert_json_safe(unknown)

    assert (
        AutoLamellaAppContext(Host()).set_supervision("x", True)["available"] is False
    )


def test_set_workflow_options_turns_the_run_summary_off_and_saves(experiment):
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    assert ctx.protocol()["options"]["show_run_summary"] is True  # the default

    applied = ctx.set_workflow_options(show_run_summary=False)
    assert applied == {
        "available": True,
        "applied": True,
        "saved": True,
        "show_run_summary": False,
    }
    assert experiment.task_protocol.options.show_run_summary is False
    assert ctx.protocol()["options"]["show_run_summary"] is False
    _assert_json_safe(applied)

    assert AutoLamellaAppContext(Host()).set_workflow_options(
        show_run_summary=False
    ) == {
        "available": False,
        "applied": False,
    }


def test_supervision_sets_attention_and_names_no_supervisor(experiment):
    # A supervised question goes to the operator and the agent alike, and the
    # first answer wins (FIB-1035): no task says whose its questions are.
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )

    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Rough Milling", attention=Attention.automated, required=True
        )
    )
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    assert "supervisor" not in ctx.protocol()["tasks"][-1]
    applied = ctx.set_supervision("Rough Milling", True)
    assert applied == {
        "available": True,
        "applied": True,
        "saved": True,
        "task_name": "Rough Milling",
        "supervise": True,
    }
    task = experiment.task_protocol.workflow_config.tasks[-1]
    assert task.attention is Attention.supervised


def test_supervision_is_saved_with_the_protocol(experiment):
    # As the Workflow tab's chip saves it: a reload keeps the agent's change.
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )

    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Rough Milling", attention=Attention.automated, required=True
        )
    )
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    assert ctx.set_supervision("Rough Milling", True)["saved"] is True
    protocol = AutoLamellaTaskProtocol.load(
        os.path.join(experiment.path, "protocol.yaml")
    )
    assert protocol.get_attention("Rough Milling") is Attention.supervised


def test_requeue_task_reruns_a_completed_pair(experiment, microscope):
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name="Rough Milling", required=True)
    )
    host = Host()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    # Before a run there is no queue: a structured refusal, not an error.
    idle = ctx.requeue_task(experiment.positions[0].name, "Rough Milling")
    assert idle["queued"] is False
    assert "no workflow is running" in idle["reason"]

    manager = TaskManager(microscope, experiment, parent_ui=None)
    manager.queue.build_from_matrix(
        ["Rough Milling"], [p.name for p in experiment.positions]
    )
    host._task_manager = manager
    name = experiment.positions[0].name

    # A pair still waiting runs once: asking again is refused, not queued twice.
    waiting = ctx.requeue_task(name, "Rough Milling")
    assert waiting["queued"] is False
    assert waiting["error_type"] == "already_queued"

    first = next(i for i in manager.queue.items if i.item_name == name)
    manager.queue.mark_done(first, TaskStatus.Completed)
    before = len(manager.queue.items)
    queued = ctx.requeue_task(name, "Rough Milling")
    assert queued["queued"] is True
    assert queued["queue_version"] == manager.queue.version
    assert len(manager.queue.items) == before + 1
    _assert_json_safe(queued)
    again = ctx.requeue_task(name, "Rough Milling")
    assert again["error_type"] == "already_queued"
    assert len(manager.queue.items) == before + 1

    # Refusals stay structured: unknown item, unknown task.
    assert ctx.requeue_task("no-such-item", "Rough Milling")["queued"] is False
    bad_task = ctx.requeue_task(experiment.positions[0].name, "No Such Task")
    assert bad_task["queued"] is False
    assert bad_task["task_names"] == ["Rough Milling"]


def test_queue_snapshot_uses_item_name_vocabulary(experiment, microscope):
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    manager = TaskManager(microscope, experiment, parent_ui=None)
    names = [p.name for p in experiment.positions]
    manager.queue.build_from_matrix(["Task A", "Task B"], names)

    host = Host()
    host.experiment = experiment
    host._task_manager = manager
    ctx = AutoLamellaAppContext(host)

    queue = ctx.queue()
    assert queue["available"] is True
    assert len(queue["items"]) == 4  # 2 tasks x 2 items
    first = queue["items"][0]
    assert set(first) == {"id", "item_name", "task_name", "status"}
    assert "lamella_name" not in first  # the deprecated alias never crosses the wire
    assert first["status"] == TaskStatus.NotStarted.name
    assert queue["version"] == manager.queue.version
    _assert_json_safe(queue)

    status = ctx.status()
    assert status["workflow"]["queue_total"] == 4
    assert status["workflow"]["current_task"] is None  # nothing running


def test_stage_position_reads_the_cache_only(microscope):
    host = Host()
    host.microscope = microscope
    ctx = AutoLamellaAppContext(host)

    microscope.get_stage_position()  # populate the cache, as the app would
    cached = ctx.stage_position()
    assert cached["available"] is True
    _assert_json_safe(cached)

    # The facade must not have issued a hardware read itself: forgetting the stage
    # device's cached value directly proves which side it reads from.
    from fibsem.devices.core import _UNSET

    microscope.stage.position._cached = _UNSET
    assert ctx.stage_position() == {"available": False, "position": None}


def test_recent_experiments_is_json_safe():
    ctx = AutoLamellaAppContext(Host())
    _assert_json_safe(ctx.recent_experiments())


def test_edit_queue_removes_or_moves_a_waiting_pair(experiment, microscope):
    from fibsem.applications.autolamella.server.events import EventBuffer
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    host = Host()
    host.experiment = experiment
    buffer = EventBuffer()
    ctx = AutoLamellaAppContext(host, event_buffer=buffer)
    first, second = (p.name for p in experiment.positions[:2])

    # No run, no queue: a structured refusal.
    assert ctx.edit_queue(first, "Rough Milling", "remove")["applied"] is False

    manager = TaskManager(microscope, experiment, parent_ui=None)
    manager.queue.build_from_matrix(["Rough Milling"], [first, second])
    host._task_manager = manager

    running = manager.queue.next()
    assert running.item_name == first
    refused = ctx.edit_queue(first, "Rough Milling", "remove")
    assert refused["error_type"] == "not_pending"
    assert "running" in refused["error"]

    removed = ctx.edit_queue(second, "Rough Milling", "remove")
    assert removed["applied"] is True and removed["changed"] is True
    assert removed["queue_version"] == manager.queue.version
    _assert_json_safe(removed)
    assert manager.queue.pending == []
    again = ctx.edit_queue(second, "Rough Milling", "remove")
    assert again["error_type"] == "not_pending"
    assert "removed" in again["error"]

    (event,) = [
        e for e in buffer.events_since(0)["events"] if e["kind"] == "queue_changed"
    ]
    assert event["payload"]["action"] == "remove"
    assert event["payload"]["item_name"] == second

    assert ctx.edit_queue("no-such", "Rough Milling", "remove")["error_type"] == (
        "not_found"
    )
    bad = ctx.edit_queue(second, "Rough Milling", "shuffle")
    assert bad["error_type"] == "invalid"
    assert "remove" in bad["action_names"]


def test_item_version_changes_when_a_pose_moves(experiment):
    from fibsem.applications.autolamella.server.context import item_fields_version

    lamella = experiment.positions[0]
    before = item_fields_version(lamella)
    lamella.milling_pose.stage_position.x = 10e-6
    assert item_fields_version(lamella) != before
