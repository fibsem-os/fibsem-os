"""Tests for AutoLamellaAppContext: the read-side facade the embedded server will use.

Real domain objects throughout — a real Experiment with real lamellae, the real
TaskQueue mechanics, the Demo microscope — held by a session with no window
(the real-window pairing lives in tests/ui/test_agent_context_host.py)."""

import json
import os

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.codec import encode
from fibsem.server.responses import Refusal
from fibsem.structures import MicroscopeState
from fibsem.workflows.attention import Attention
from fibsem.workflows.status import TaskStatus
from tests.protocols import add_planned, planned
from tests.run_state import stub_session


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path / "exp", name="agent-context-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _plain(value):
    """A context read as the wire carries it; a refusal as its error detail."""
    if isinstance(value, Refusal):
        return value.detail
    return value if isinstance(value, dict) else encode(value)


def _refused(value, error_type):
    """*value* is a refusal of *error_type*; its detail."""
    assert isinstance(value, Refusal), value
    assert value.error_type == error_type, value
    return value.detail


def _assert_json_safe(payload):
    json.dumps(payload)  # raises on anything a wire client couldn't carry


def test_every_method_tolerates_an_empty_host():
    ctx = AutoLamellaAppContext(stub_session())
    payloads = [
        _plain(ctx.status()),
        _plain(ctx.queue()),
        _plain(ctx.experiment_summary()),
        _plain(ctx.task_history()),
        _plain(ctx.run_summary()),
        _plain(ctx.protocol()),
        _plain(ctx.task_outputs("anything")),
        _plain(ctx.item_detail("anything")),
    ]
    for payload in payloads:
        _assert_json_safe(payload)
    assert _plain(ctx.status())["experiment"] is None
    assert _plain(ctx.queue()) == {"items": [], "version": None}


def test_resolution_is_call_time_not_construction_time(experiment):
    # The app rebinds `experiment` on load; a facade built earlier must see the
    # new binding, not the one from construction (the ScriptContext hazard).
    host = stub_session()
    ctx = AutoLamellaAppContext(host)
    assert _plain(ctx.status())["experiment"] is None
    host.experiment = experiment
    assert _plain(ctx.status())["experiment"]["name"] == "agent-context-exp"
    host.experiment = None
    assert _plain(ctx.status())["experiment"] is None


def test_status_and_summaries_reflect_a_real_experiment(experiment):
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    status = _plain(ctx.status())
    assert status["experiment"]["num_items"] == 2
    assert status["workflow"]["running"] is False

    summary = _plain(ctx.experiment_summary())
    assert len(summary["items"]) == 2
    _assert_json_safe(summary)

    protocol = _plain(ctx.protocol())
    assert protocol["tasks"] is not None
    _assert_json_safe(protocol)


def test_task_outputs_for_a_real_item_and_a_missing_one(experiment):
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    name = experiment.positions[0].name
    payload = _plain(ctx.task_outputs(name))
    assert payload["item_name"] == name
    assert payload["final_reference_images"] == []  # nothing has run
    _assert_json_safe(payload)

    # With history: the live-run 500 — the field is `name`, and an empty
    # history hid the read of a field that doesn't exist.
    from fibsem.workflows.status import TaskState, TaskStatus

    experiment.positions[0].task_history.append(
        TaskState(name="Rough Milling", status=TaskStatus.Completed)
    )
    completed = _plain(ctx.task_outputs(name))
    assert completed["completed_tasks"] == ["Rough Milling"]
    _assert_json_safe(completed)

    missing = _refused(ctx.task_outputs("no-such-item"), "not_found")
    assert "no-such-item" in missing["message"]


def test_output_image_renders_recorded_outputs_only(experiment):
    import numpy as np
    import tifffile

    from fibsem.workflows.status import TaskState

    host = stub_session()
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
        TaskState(
            name="Rough Milling",
            outputs={"final_fib": [fname], "fluorescence": [zname]},
        )
    )

    served = ctx.output_image(lamella.name, fname)
    assert served["jpeg"][:2] == b"\xff\xd8"
    assert (served["width"], served["height"]) == (64, 48)

    stack = ctx.output_image(lamella.name, zname)
    assert stack["jpeg"][:2] == b"\xff\xd8"

    unknown = _refused(ctx.output_image(lamella.name, "not-listed.tif"), "not_found")
    assert sorted(unknown["filenames"]) == [zname, fname]

    _refused(ctx.output_image("no-such-item", fname), "not_found")

    # task_outputs keeps the per-run grouping: which task produced which file.
    payload = _plain(ctx.task_outputs(lamella.name))
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

    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    config = MillRoughTaskConfig(task_name="Rough Milling")
    experiment.task_protocol.task_config["Rough Milling"] = config
    lamella = experiment.positions[0]
    lamella.task_config["Rough Milling"] = config

    doc = ctx.protocol_task_config("Rough Milling")
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

    unknown = _refused(ctx.protocol_task_config("No Such Task"), "not_found")
    assert "task_names" in unknown
    _refused(ctx.item_task_config("no-such-item", "Rough Milling"), "not_found")
    no_config = _refused(
        ctx.item_task_config(lamella.name, "No Such Task"), "not_found"
    )
    assert no_config["task_names"] == ["Rough Milling"]


def test_item_detail_serves_the_durable_item_facts(experiment):
    """One read for what a supervisor judges an item by — the exemplar-mode
    seam: after the operator answers, the accepted POI/alignment area are
    readable here rather than raced out of the live prompt mirror."""
    from fibsem.structures import FibsemRectangle, MicroscopeState, Point

    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    lamella = experiment.positions[0]
    lamella.poi = Point(1.5e-6, -2.5e-6)
    lamella.alignment_area = FibsemRectangle(left=0.6, top=0.3, width=0.3, height=0.4)
    lamella.milling_angle = 15.0
    lamella.milling_pose = MicroscopeState()

    detail = _plain(ctx.item_detail(lamella.name))
    assert detail["item_name"] == lamella.name
    assert detail["poi"] == {"x": 1.5e-6, "y": -2.5e-6, "name": None}
    assert detail["alignment_area"]["left"] == 0.6
    assert detail["milling_angle"] == 15.0
    assert detail["is_failure"] is False
    assert "MILLING" in detail["poses"]
    _assert_json_safe(detail)

    missing = _refused(ctx.item_detail("no-such-item"), "not_found")
    assert "no-such-item" in missing["message"]


def test_stop_workflow_takes_the_stop_path_only_while_running():
    host = stub_session()
    ctx = AutoLamellaAppContext(host)
    idle = _plain(ctx.stop_workflow())
    assert idle == {"stopped": False, "reason": "no workflow is running"}
    assert host.runner.stops == []

    host.runner.is_running = True
    assert _plain(ctx.stop_workflow()) == {"stopped": True}
    assert host.runner.stops == ["run"]


def test_a_stop_tells_the_run_who_stopped_it_and_why():
    """The stopped task's record says the agent stopped it and why (FIB-1517)."""
    from fibsem.acting import ActorKind, acting

    manager = type(
        "Manager", (), {"stop": lambda self, message="": setattr(self, "said", message)}
    )()
    host = stub_session(manager=manager, is_running=True)
    ctx = AutoLamellaAppContext(host)
    with acting(ActorKind.agent):
        ctx.stop_workflow("The trench drifted.")
    assert manager.said == "Stopped by the agent: The trench drifted."


def test_set_supervision_flips_the_live_protocol(experiment):
    add_planned(experiment.task_protocol, planned(name="Rough Milling", required=True))
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    applied = ctx.set_supervision("Rough Milling", True)
    assert applied["applied"] is True
    # The same read the workflow's decision points use sees the new value.
    assert (
        experiment.task_protocol.get_attention("Rough Milling") is Attention.supervised
    )

    unknown = _refused(ctx.set_supervision("No Such Task", True), "not_found")
    assert unknown["task_names"] == ["Rough Milling"]
    _assert_json_safe(unknown)

    _refused(
        AutoLamellaAppContext(stub_session()).set_supervision("x", True),
        "no_experiment",
    )


def test_set_workflow_options_turns_the_run_summary_off_and_saves(experiment):
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    assert ctx.protocol()["options"]["show_run_summary"] is True  # the default

    applied = ctx.set_workflow_options(show_run_summary=False)
    assert applied == {"applied": True, "saved": True, "show_run_summary": False}
    assert experiment.task_protocol.options.show_run_summary is False
    assert ctx.protocol()["options"]["show_run_summary"] is False
    _assert_json_safe(applied)

    _refused(
        AutoLamellaAppContext(stub_session()).set_workflow_options(
            show_run_summary=False
        ),
        "no_experiment",
    )


def test_supervision_sets_attention_and_names_no_supervisor(experiment):
    # A supervised question goes to the operator and the agent alike, and the
    # first answer wins (FIB-1035): no task says whose its questions are.
    add_planned(
        experiment.task_protocol,
        planned(name="Rough Milling", attention=Attention.automated, required=True),
    )
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    assert "supervisor" not in ctx.protocol()["tasks"][-1]
    applied = ctx.set_supervision("Rough Milling", True)
    assert applied == {
        "applied": True,
        "saved": True,
        "task_name": "Rough Milling",
        "supervise": True,
    }
    protocol = experiment.task_protocol
    assert protocol.get_attention("Rough Milling") is Attention.supervised


def test_supervision_is_saved_with_the_protocol(experiment):
    # As the Workflow tab's chip saves it: a reload keeps the agent's change.
    add_planned(
        experiment.task_protocol,
        planned(name="Rough Milling", attention=Attention.automated, required=True),
    )
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    assert ctx.set_supervision("Rough Milling", True)["saved"] is True
    protocol = AutoLamellaTaskProtocol.load(
        os.path.join(experiment.path, "protocol.yaml")
    )
    assert protocol.get_attention("Rough Milling") is Attention.supervised


def test_requeue_task_reruns_a_completed_pair(experiment, microscope):
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    add_planned(experiment.task_protocol, planned(name="Rough Milling", required=True))
    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)

    # Before a run there is no queue: a structured refusal, not an error.
    idle = _refused(
        ctx.requeue_task(experiment.positions[0].name, "Rough Milling"), "not_running"
    )
    assert "no workflow is running" in idle["message"]

    manager = TaskManager(microscope, experiment)
    manager.queue.build_from_matrix(
        ["Rough Milling"], [p.name for p in experiment.positions]
    )
    host.runner.manager = manager
    name = experiment.positions[0].name

    # A pair still waiting runs once: asking again is refused, not queued twice.
    _refused(ctx.requeue_task(name, "Rough Milling"), "already_queued")

    first = next(i for i in manager.queue.items if i.item_name == name)
    manager.queue.mark_done(first, TaskStatus.Completed)
    before = len(manager.queue.items)
    queued = _plain(ctx.requeue_task(name, "Rough Milling"))
    assert queued["queued"] is True
    assert queued["queue_version"] == manager.queue.version
    assert len(manager.queue.items) == before + 1
    _assert_json_safe(queued)
    _refused(ctx.requeue_task(name, "Rough Milling"), "already_queued")
    assert len(manager.queue.items) == before + 1

    # Refusals stay structured: unknown item, unknown task.
    _refused(ctx.requeue_task("no-such-item", "Rough Milling"), "not_found")
    bad_task = _refused(
        ctx.requeue_task(experiment.positions[0].name, "No Such Task"), "not_found"
    )
    assert bad_task["task_names"] == ["Rough Milling"]


def test_queue_snapshot_uses_item_name_vocabulary(experiment, microscope):
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    manager = TaskManager(microscope, experiment)
    names = [p.name for p in experiment.positions]
    manager.queue.build_from_matrix(["Task A", "Task B"], names)

    host = stub_session()
    host.experiment = experiment
    host.runner.manager = manager
    ctx = AutoLamellaAppContext(host)

    queue = _plain(ctx.queue())
    assert len(queue["items"]) == 4  # 2 tasks x 2 items
    first = queue["items"][0]
    assert set(first) == {"id", "item_name", "task_name", "status"}
    assert "lamella_name" not in first  # the deprecated alias never crosses the wire
    assert first["status"] == TaskStatus.NotStarted.name
    assert queue["version"] == manager.queue.version
    _assert_json_safe(queue)

    status = _plain(ctx.status())
    assert status["workflow"]["queue_total"] == 4
    assert status["workflow"]["current_task"] is None  # nothing running


def test_recent_experiments_is_json_safe():
    ctx = AutoLamellaAppContext(stub_session())
    _assert_json_safe(ctx.recent_experiments())


def test_edit_queue_removes_or_moves_a_waiting_pair(experiment, microscope):
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    host = stub_session()
    host.experiment = experiment
    # The session's stream, which the runner records queue edits to.
    buffer = host.event_buffer
    ctx = AutoLamellaAppContext(host)
    first, second = (p.name for p in experiment.positions[:2])

    # No run, no queue: a structured refusal.
    _refused(ctx.edit_queue(first, "Rough Milling", "remove"), "not_running")

    manager = TaskManager(microscope, experiment)
    manager.queue.build_from_matrix(["Rough Milling"], [first, second])
    host.runner.manager = manager

    running = manager.queue.next()
    assert running.item_name == first
    refused = _refused(ctx.edit_queue(first, "Rough Milling", "remove"), "not_pending")
    assert "running" in refused["message"]

    removed = _plain(ctx.edit_queue(second, "Rough Milling", "remove"))
    assert removed["applied"] is True and removed["changed"] is True
    assert removed["queue_version"] == manager.queue.version
    _assert_json_safe(removed)
    assert manager.queue.pending == []
    again = _refused(ctx.edit_queue(second, "Rough Milling", "remove"), "not_pending")
    assert "removed" in again["message"]

    (event,) = [
        e for e in buffer.events_since(0)["events"] if e["kind"] == "queue_changed"
    ]
    assert event["payload"]["action"] == "remove"
    assert event["payload"]["item_name"] == second

    _refused(ctx.edit_queue("no-such", "Rough Milling", "remove"), "not_found")
    bad = _refused(ctx.edit_queue(second, "Rough Milling", "shuffle"), "invalid")
    assert "remove" in bad["action_names"]


def test_edit_queue_keeps_the_grid_load_step(experiment, microscope):
    """FIB-1454: the server refuses what the timeline refuses."""
    from fibsem.workflows.grid.manager import (
        LOAD_ENTRY_NAME,
        GridTaskManager,
    )

    host = stub_session()
    host.experiment = experiment
    ctx = AutoLamellaAppContext(host)
    manager = GridTaskManager(microscope, experiment)
    manager.queue.build_from_pairs(
        [("grid-1", LOAD_ENTRY_NAME), ("grid-1", "Overview")]
    )
    host.runner.manager = manager

    _refused(ctx.edit_queue("grid-1", LOAD_ENTRY_NAME, "remove"), "fixed")
    assert [i.task_name for i in manager.queue.pending] == [LOAD_ENTRY_NAME, "Overview"]


def test_item_version_changes_when_a_pose_moves(experiment):
    from fibsem.applications.autolamella.server.context import item_fields_version

    lamella = experiment.positions[0]
    before = item_fields_version(lamella)
    lamella.milling_pose.stage_position.x = 10e-6
    assert item_fields_version(lamella) != before
