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
from fibsem.applications.volume.tasks import VolumeFIBMillingTaskConfig  # noqa: E402
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
    assert tasks[0]["attention"] == "supervised"


def test_a_position_is_added_at_a_stage_position_and_saved(agent, experiment):
    added = _add(agent, "aspen", stage_position=FibsemStagePosition(x=1e-4).to_dict())

    assert added["item"]["item_name"] == "aspen"
    saved = VolumeExperiment.load(experiment.path).get_position_by_name("aspen")
    assert saved.pose("MILLING").stage_position.x == pytest.approx(1e-4)
    assert (
        agent.post("/app/experiment/items", json={"name": "aspen"}).status_code == 422
    )
    assert _add(agent, "")["item"]["item_name"].startswith("02-")


def test_a_position_patch_goes_through_the_experiment_in_one_save(agent, experiment):
    version = _add(agent, "aspen")["item"]["version"]
    path = "/app/experiment/items/aspen"

    patched = _ok(
        agent.post(
            path,
            json={
                "patch": {
                    "iterations": 4,
                    "poi": {"x": 2e-6, "y": 3e-6},
                    "description": "near the crack",
                },
                "version": version,
            },
        )
    )

    assert patched["fields"] == ["iterations", "poi", "description"]
    aspen = VolumeExperiment.load(experiment.path).get_position_by_name("aspen")
    assert aspen.iterations == 4 and aspen.poi.x == pytest.approx(2e-6)
    assert aspen.description == "near the crack"
    stale = agent.post(path, json={"patch": {"iterations": 5}, "version": version})
    assert stale.status_code == 409
    current = _ok(agent.get(path))["version"]
    outside = agent.post(
        path,
        json={
            "patch": {
                "iterations": 9,
                "alignment_area": {
                    "left": 0.8,
                    "top": 0.8,
                    "width": 0.5,
                    "height": 0.5,
                },
            },
            "version": current,
        },
    )
    assert outside.status_code == 422
    assert "inside the frame" in outside.json()["detail"]["message"]
    assert experiment.get_position_by_name("aspen").iterations == 4
    renamed = agent.post(path, json={"patch": {"name": "birch"}, "version": current})
    assert renamed.status_code == 422
    assert experiment.get_position_by_name("aspen") is not None


def test_a_removed_position_leaves_the_experiment(agent, experiment):
    _add(agent, "aspen")

    _ok(agent.delete("/app/experiment/items/aspen"))

    assert experiment.positions == []
    assert agent.delete("/app/experiment/items/aspen").status_code == 404


MILL, IMAGE = "Volume FIB Milling", "Volume SEM Acquisition"


def _saved_protocol(experiment):
    return VolumeExperiment.load(experiment.path).task_protocol


def test_supervision_schedule_and_requirements_are_saved_with_the_protocol(
    agent, experiment
):
    _ok(
        agent.post(
            "/app/experiment/protocol/supervision",
            json={"task_name": MILL, "supervise": True},
        )
    )
    _ok(
        agent.post(
            "/app/workflow/schedule",
            json={"task_name": MILL, "scheduled_at": "2026-10-21T06:00:00"},
        )
    )
    _ok(
        agent.post(
            f"/app/experiment/protocol/tasks/{IMAGE}/requirements",
            json={"requires": [], "required": False},
        )
    )

    saved = _saved_protocol(experiment)
    assert saved.get_attention(MILL) is Attention.supervised
    assert saved.workflow_config.get(MILL).scheduled_at is not None
    assert saved.requirements(IMAGE) == []
    assert saved.workflow_config.get(IMAGE).required is False
    missing = agent.post(
        "/app/experiment/protocol/supervision",
        json={"task_name": "Nope", "supervise": True},
    )
    assert _ok(missing)["applied"] is False


def test_tasks_are_reordered_and_removed(agent, experiment):
    order = [t["name"] for t in _ok(agent.get("/app/experiment/protocol"))["tasks"]]
    swapped = [order[1], order[0], *order[2:]]

    _ok(agent.post("/app/experiment/protocol/tasks/reorder", json={"order": swapped}))
    _ok(agent.delete(f"/app/experiment/protocol/tasks/{IMAGE}"))

    saved = _saved_protocol(experiment)
    assert saved.ordered_task_names == swapped[:3]
    assert (
        agent.post(
            "/app/experiment/protocol/tasks/reorder", json={"order": swapped[:2]}
        ).status_code
        == 422
    )
    assert agent.delete(f"/app/experiment/protocol/tasks/{IMAGE}").status_code == 404


def test_a_task_config_patch_names_the_version_it_read(agent, experiment):
    path = f"/app/experiment/protocol/task_config/{IMAGE}"
    read = _ok(agent.get(path))
    assert read["config"]["autofocus_every"] == 0

    patched = _ok(
        agent.post(
            path, json={"patch": {"autofocus_every": 5}, "version": read["version"]}
        )
    )

    assert patched["changes"] == [{"path": "autofocus_every", "old": 0, "new": 5}]
    assert _saved_protocol(experiment).task_config[IMAGE].autofocus_every == 5
    stale = agent.post(
        path, json={"patch": {"autofocus_every": 6}, "version": read["version"]}
    )
    assert stale.status_code == 409
    supervision = agent.post(
        path, json={"patch": {"attention": "automated"}, "version": patched["version"]}
    )
    assert supervision.status_code == 422


def test_a_task_is_added_from_the_volume_registry(agent, experiment):
    task_type = VolumeFIBMillingTaskConfig.task_type
    body = {"task_type": task_type, "task_name": " Mill again "}

    added = _ok(agent.post("/app/experiment/protocol/tasks", json=body))

    assert added["task_names"][-1] == "Mill again"
    saved = _saved_protocol(experiment)
    assert saved.task_config["Mill again"].task_type == task_type
    assert saved.get_attention("Mill again") is Attention.supervised
    again = agent.post("/app/experiment/protocol/tasks", json=body)
    assert again.status_code == 422
    unknown = agent.post(
        "/app/experiment/protocol/tasks",
        json={"task_type": "NOPE", "task_name": "x"},
    )
    assert unknown.status_code == 404
    assert task_type in unknown.json()["detail"]["task_type_names"]


def test_turning_the_beams_off_is_saved_with_the_protocol(agent, experiment):
    _ok(agent.post("/app/workflow/options", json={"turn_beams_off": True}))

    assert VolumeExperiment.load(experiment.path).turns_beams_off


def test_an_agent_runs_a_position_from_set_up_to_its_last_slice(
    agent, session, experiment, microscope, milled
):
    for task in experiment.task_protocol.workflow_config.tasks:
        task.attention = Attention.automated
    experiment.add_position(
        "aspen", microscope.get_microscope_state(), Point(0, 1e-6), iterations=2
    )
    tasks = [t["name"] for t in _ok(agent.get("/app/experiment/protocol"))["tasks"]]

    assert _ok(agent.post("/app/workflow/start", json={"task_names": tasks}))["started"]
    deadline = time.monotonic() + 120
    while session.runner.is_running and time.monotonic() < deadline:
        time.sleep(0.2)

    (row,) = _ok(agent.get("/app/experiment"))["items"]
    assert row["slices_cut"] == 2 and not row["is_failure"]
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


def test_volume_answers_decode_as_its_own_classes(agent):
    from tests.server.test_app_response_models import _Checked

    call = _Checked(agent)
    added = call("POST", "/app/experiment/items", json={"name": "aspen"}).item
    assert added.iterations is not None  # a PositionDetail
    base = "/app/experiment/items/{item_name}"
    call(
        "POST",
        "/app/experiment/items/aspen",
        base,
        json={"patch": {"iterations": 2}, "version": added.version},
    )
    assert call("GET", "/app/experiment").items[0].item_name == "aspen"
    task = call("GET", "/app/experiment/protocol").tasks[0]
    assert task.attention == "supervised"
    call(
        "GET",
        f"/app/experiment/items/aspen/task_config/{task.name}",
        base + "/task_config/{task_name}",
    )
    call("DELETE", "/app/experiment/items/aspen", base)


def test_a_task_never_goes_before_or_requires_a_later_task(agent, experiment):
    order = [t["name"] for t in _ok(agent.get("/app/experiment/protocol"))["tasks"]]
    mill, image = order.index(MILL), order.index(IMAGE)
    image_first = list(order)
    image_first[mill], image_first[image] = IMAGE, MILL

    refused = agent.post(
        "/app/experiment/protocol/tasks/reorder", json={"order": image_first}
    )
    later = agent.post(
        f"/app/experiment/protocol/tasks/{MILL}/requirements",
        json={"requires": [IMAGE]},
    )

    assert refused.status_code == 422
    assert "comes after it" in refused.json()["detail"]["message"]
    assert later.status_code == 422
    assert "can only require" in later.json()["detail"]["message"]
    assert _saved_protocol(experiment).ordered_task_names == order


def test_a_per_position_config_edit_is_refused_as_invalid_not_missing(agent):
    _add(agent, "aspen")
    refused = agent.post(
        f"/app/experiment/items/aspen/task_config/{MILL}",
        json={"patch": {"slice_thickness": 1e-7}, "version": "v"},
    )
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["error_type"] == "invalid"
    assert "positions share the protocol's" in detail["message"]
