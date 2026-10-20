"""A volume session over the App Server: the /app core and the Items part, each
position edit made by the experiment's own method."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.acting import AGENT  # noqa: E402
from fibsem.applications.volume.experiment import VolumeExperiment  # noqa: E402
from fibsem.applications.volume.server import VolumeAppContext  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import FibsemStagePosition, Point  # noqa: E402
from fibsem.workflows.attention import Attention  # noqa: E402
from fibsem.workflows.events import EventBuffer  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402

TOKEN = "volume-token"


@pytest.fixture
def session(microscope, experiment):
    session = AppSession()
    session.microscope = microscope
    session.experiment = experiment
    yield session
    session.runner.stop()
    session.runner.join(30)


@pytest.fixture
def agent(session, microscope):
    app = build_server(
        microscope,
        app_context=VolumeAppContext(session, event_buffer=EventBuffer()),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def _ok(answer):
    assert answer.status_code == 200, answer.text
    return answer.json()


def _milling_action(proposal) -> str:
    """Run a milling pass, then accept it: what an operator does."""
    if proposal["kind"] != "milling":
        return ""
    allowed = proposal["provenance"].get("actions") or ["run"]
    return "accept" if "accept" in allowed else "run"


def _add(agent, name, **body):
    return _ok(agent.post("/app/experiment/items", json={"name": name, **body}))


def test_volume_serves_the_core_and_items(agent):
    capabilities = _ok(agent.get("/capabilities"))
    assert capabilities["app_name"] == "Volume"
    assert capabilities["parts"] == ["items"]
    assert _ok(agent.get("/app/workflow/status"))["experiment"]["name"] == "vol"
    tasks = _ok(agent.get("/app/experiment/protocol"))["tasks"]
    assert [t["name"] for t in tasks][:2] == [
        "Setup FIB Milling Position",
        "Setup SEM Acquisition",
    ]
    assert tasks[0]["supervise"] is True


def test_a_position_is_added_at_a_stage_position_and_saved(agent, experiment):
    added = _add(agent, "aspen", stage_position=FibsemStagePosition(x=1e-4).to_dict())

    assert added["item"]["item_name"] == "aspen"
    saved = VolumeExperiment.load(experiment.path).get_position_by_name("aspen")
    assert saved.pose("MILL").stage_position.x == pytest.approx(1e-4)
    assert (
        agent.post("/app/experiment/items", json={"name": "aspen"}).status_code == 422
    )
    assert _add(agent, "")["item"]["item_name"] == "position-02"


def test_a_position_patch_goes_through_the_experiment_methods(agent, experiment):
    version = _add(agent, "aspen")["item"]["version"]
    path = "/app/experiment/items/aspen"

    patched = _ok(
        agent.post(
            path,
            json={
                "patch": {
                    "iterations": 4,
                    "poi": {"x": 2e-6, "y": 3e-6},
                    "name": "birch",
                },
                "version": version,
            },
        )
    )

    assert patched["fields"] == ["iterations", "poi", "name"]
    birch = VolumeExperiment.load(experiment.path).get_position_by_name("birch")
    assert birch.iterations == 4 and birch.poi.x == pytest.approx(2e-6)
    stale = agent.post(
        "/app/experiment/items/birch",
        json={"patch": {"iterations": 5}, "version": version},
    )
    assert stale.status_code == 409
    current = _ok(agent.get("/app/experiment/items/birch"))["version"]
    outside = agent.post(
        "/app/experiment/items/birch",
        json={
            "patch": {
                "alignment_area": {"left": 0.8, "top": 0.8, "width": 0.5, "height": 0.5}
            },
            "version": current,
        },
    )
    assert outside.status_code == 422
    assert "inside the frame" in outside.json()["detail"]["message"]


def test_a_removed_position_leaves_the_experiment(agent, experiment):
    _add(agent, "aspen")

    _ok(agent.delete("/app/experiment/items/aspen"))

    assert experiment.positions == []
    assert agent.delete("/app/experiment/items/aspen").status_code == 404


def test_what_volume_does_not_serve_yet_says_so(agent):
    answer = _ok(
        agent.post(
            "/app/experiment/protocol/supervision",
            json={"task_name": "Volume FIB Milling", "supervise": True},
        )
    )
    assert answer["applied"] is False
    assert "Volume does not" in answer["error"]


def test_an_agent_runs_a_position_from_set_up_to_its_last_slice(
    agent, session, experiment, microscope, milled
):
    for config in experiment.task_protocol.task_config.values():
        config.attention = Attention.automated
    experiment.add_position(
        "aspen", microscope.get_microscope_state(), Point(0, 1e-6), iterations=2
    )
    tasks = [t["name"] for t in _ok(agent.get("/app/experiment/protocol"))["tasks"]]

    assert _ok(agent.post("/app/workflow/start", json={"task_names": tasks}))["started"]
    deadline = time.monotonic() + 120
    while session.runner.is_running and time.monotonic() < deadline:
        time.sleep(0.2)

    (row,) = _ok(agent.get("/app/experiment"))["items"]
    assert row["slices_cut"] == 2 and not row["failed"]
    history = _ok(agent.get("/app/experiment/task_history"))["items"]
    assert [r["iteration"] for r in history] == [None, None, 0, 0, 1, 1]


def test_the_agent_decides_a_supervised_set_up(
    agent, session, experiment, microscope, milled
):
    setup = "Setup FIB Milling Position"
    position = experiment.add_position(
        "aspen", microscope.get_microscope_state(), Point(0, 1e-6), iterations=1
    )
    # the agent listening at Drive, as a hosted App Server counts it
    session.attach_decider(AGENT)
    assert _ok(agent.post("/app/workflow/start", json={"task_names": [setup]}))[
        "started"
    ]

    decided = []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        workflow = _ok(agent.get("/app/workflow/status"))["workflow"]
        if not workflow["running"] and not workflow["asking"]:
            break
        if not workflow["asking"]:
            time.sleep(0.1)
            continue
        (shown,) = _ok(agent.get("/app/workflow/proposals"))["proposals"]
        assert shown["item_name"] == "aspen"
        _ok(
            agent.post(
                "/app/workflow/decide",
                json={
                    "item_id": position.id,
                    "task_name": setup,
                    "proposal_id": shown["proposal_id"],
                    "outcome": "Confirmed",
                    "action": _milling_action(shown),
                },
            )
        )
        decided.append(shown["kind"])

    assert session.runner.join(30)
    assert set(decided) == {"state", "milling", "point_of_interest", "alignment_area"}
    assert all(str(p.current.actor).startswith("agent") for p in position.proposals)
