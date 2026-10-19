"""Runs live in the services (`fibsem.devices.runs`): a mill a workflow task or
the operator starts is followed the way an agent's is, through ``GET
.../milling/run`` and ``service_wait_for_run``, under the id its ``milling``
events carry."""

import asyncio
import json
import os
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import config as fcfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.runs import current_run  # noqa: E402
from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
MILLING = "/microscope/services/milling"


@pytest.fixture(scope="module")
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


@pytest.fixture
def client(microscope, tmp_path, monkeypatch):
    monkeypatch.setattr(fcfg, "DATA_CC_PATH", str(tmp_path / "data"))
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


@pytest.fixture
def events(microscope):
    seen = []

    def record(kind, payload):
        seen.append((kind, payload))

    microscope.session.record_signal.connect(record)
    yield seen
    microscope.session.record_signal.disconnect(record)


def _config(*names, name="Trench"):
    stages = []
    for stage_name in names:
        stage = FibsemMillingStage(name=stage_name)
        stage.alignment.enabled = False
        stages.append(stage)
    config = FibsemMillingTaskConfig.from_stages(stages, name=name)
    config.alignment.enabled = False
    config.acquisition.acquire_sem = config.acquisition.acquire_fib = False
    return config


def _read(client, **params):
    resp = client.get(f"{MILLING}/run", params={"wait": 10, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()["run"]


def test_a_task_s_mill_is_a_run_the_route_reads(client, microscope, events):
    """As a workflow task mills: run_milling_task, not the App Server."""
    task = run_milling_task(microscope, _config("Rough", "Polish"))

    run = _read(client, run_id=task.task_id)
    assert run["run_id"] == task.task_id
    assert run["service"] == "milling" and run["command"] == "run_task"
    assert run["state"] == "done" and run["ended_at"]
    assert run["result"]["status"] == "completed"
    assert [s["status"] for s in run["result"]["stages"]] == ["completed"] * 2
    # Its milling events carry the run's id, and it says when it starts and ends.
    milling = [p for kind, p in events if kind == "milling"]
    assert [p["milling_run_id"] for p in milling] == [task.task_id]
    commands = [(kind, p["run_id"]) for kind, p in events if kind.startswith("command")]
    assert commands == [
        ("command_started", task.task_id),
        ("command_ended", task.task_id),
    ]


def test_with_no_run_id_the_route_reads_the_mill_going_on(client, microscope):
    """No run id: the run going on, waited for, else the last one."""
    go = threading.Event()
    started = threading.Event()
    strategy_run = FibsemMillingStage().strategy.run

    def held(microscope, stage, **kwargs):
        started.set()
        assert go.wait(10)
        return strategy_run(microscope=microscope, stage=stage, **kwargs)

    config = _config("Rough")
    config.stages[0].strategy.run = held
    worker = threading.Thread(target=run_milling_task, args=(microscope, config))
    worker.start()
    try:
        assert started.wait(10)
        running = _read(client, wait=0)
        assert running["state"] == "running" and running["command"] == "run_task"
        go.set()
        ended = _read(client)
        assert ended["run_id"] == running["run_id"] and ended["state"] == "done"
    finally:
        go.set()
        worker.join(10)


def test_an_agent_s_run_task_is_one_run_under_one_id(client, events):
    resp = client.post(
        f"{MILLING}/commands/run_task",
        json={"kwargs": {"config": _config("Rough").to_dict()}},
    )
    assert resp.status_code == 200, resp.text
    started = resp.json()["result"]
    assert started["state"] == "running"

    run = _read(client, run_id=started["run_id"])
    assert run["state"] == "done" and run["result"]["status"] == "completed"
    # The task took the run's id: its events and its run are one.
    milling = [p for kind, p in events if kind == "milling"]
    assert [p["milling_run_id"] for p in milling] == [started["run_id"]]
    assert [kind for kind, _ in events if kind.startswith("command")] == [
        "command_started",
        "command_ended",
    ]


def test_run_ids_are_unique_and_the_last_run_stays_readable(client, microscope):
    first = run_milling_task(microscope, _config("Rough")).task_id
    second = run_milling_task(microscope, _config("Rough")).task_id
    assert first != second
    # Asked for the earlier run, the route answers at once with the later one.
    assert _read(client, run_id=first)["run_id"] == second
    # The ended run is still there to read, as often as it is read.
    assert _read(client)["run_id"] == second
    assert _read(client, wait=0)["state"] == "done"


def test_a_long_command_inside_a_run_is_part_of_it(microscope):
    """A spot burn's mill, a task's stage: no run of their own."""
    milling = microscope.milling
    before = milling.runs.latest
    with microscope.spot_burn.runs.tracking("run") as run:
        assert current_run() is run
        with milling.runs.tracking("run") as inner:
            assert inner is None
    assert milling.runs.latest is before
    assert microscope.spot_burn.runs.latest is run and run.state == "done"


def test_a_failed_task_s_run_says_why(client, microscope):
    def fails(microscope, stage, **kwargs):
        raise RuntimeError("the beam would not blank")

    config = _config("Rough")
    config.stages[0].strategy.run = fails
    task = run_milling_task(microscope, config)

    run = _read(client, run_id=task.task_id)
    assert run["state"] == "failed"
    assert run["error"]["error_type"] == "RuntimeError"
    assert "would not blank" in run["error"]["message"]
    assert run["result"]["status"] == "failed"


def test_service_wait_for_run_waits_for_a_task_s_mill(client, microscope):
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    task = run_milling_task(microscope, _config("Rough"))
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(
        sidecar.call_tool("service_wait_for_run", {"service": "milling", "wait": 5})
    )
    contents = result[0] if isinstance(result, tuple) else result
    contents = getattr(contents, "content", contents)
    run = json.loads("".join(getattr(c, "text", "") for c in contents))["run"]
    assert run["run_id"] == task.task_id
    assert run["state"] == "done"
