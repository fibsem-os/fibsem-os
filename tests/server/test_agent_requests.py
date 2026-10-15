"""The agent's requests (FIB-1283): at Assist the agent asks, the operator approves
in the app, and the approved actions run through the App Server's own routes."""

import gc
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.acting import current_actor  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.control import WORKFLOW  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.server.catalog import CATALOG  # noqa: E402
from fibsem.server.requests import RequestRefused, route_call  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

AGENT = {"Authorization": "Bearer agent"}
NUDGE = {"tool": "move_stage_relative", "arguments": {"dx": 1e-6}}
SUPERVISE = {
    "tool": "set_task_supervision",
    "arguments": {"task_name": "MILL_ROUGH", "supervise": True},
}
STOP_MILL = {"tool": "stop_milling", "arguments": {"reason": "the mill is off target"}}
MOVE_RELATIVE = "/microscope/services/stage_movement/commands/move_relative"


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


class RecordingContext(AutoLamellaAppContext):
    """Records each event with who was marked as acting when it was emitted."""

    def __init__(self, host):
        super().__init__(host)
        self.recorded = []

    def record_event(self, kind, payload):
        self.recorded.append((kind, payload, current_actor()))


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def auth():
    auth = AuthConfig(token="agent", dashboard_token="dashboard")
    auth.set_level(AgentLevel.ASSIST)
    return auth


def _context(tmp_path, microscope):
    host = Host()
    exp = Experiment(path=tmp_path / "exp", name="requests-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return RecordingContext(host)


@pytest.fixture
def context(tmp_path, microscope):
    return _context(tmp_path, microscope)


@pytest.fixture
def app(microscope, context, auth):
    return build_server(microscope, app_context=context, auth=auth)


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _file(client, *actions, reason="checking the trenches"):
    return client.post(
        "/app/agent/requests",
        headers=AGENT,
        json={"actions": list(actions), "reason": reason},
    )


def _detail(response):
    return response.json()["detail"]


def test_only_assist_files_requests(client, auth):
    for level in (AgentLevel.WATCH, AgentLevel.DRIVE):
        auth.set_level(level)
        refused = _file(client, NUDGE)
        assert refused.status_code == 403
        assert _detail(refused)["error_type"] == "level"
    auth.set_level(AgentLevel.OFF)
    assert _detail(_file(client, NUDGE))["error_type"] == "agent_off"


def test_assist_files_a_request_and_reads_it_back(client):
    filed = _file(client, NUDGE, SUPERVISE)
    assert filed.status_code == 202, filed.text
    body = filed.json()
    assert body["status"] == "pending"
    assert [a["path"] for a in body["actions"]] == [
        MOVE_RELATIVE,
        "/app/agent/supervision",
    ]
    read = client.get(f"/app/agent/requests/{body['request_id']}", headers=AGENT)
    assert read.json() == body
    listed = client.get("/app/agent/requests", headers=AGENT).json()["requests"]
    assert [r["request_id"] for r in listed] == [body["request_id"]]


def test_one_request_waits_at_a_time(client):
    first = _file(client, NUDGE).json()
    second = _file(client, SUPERVISE)
    assert second.status_code == 409
    assert _detail(second)["error_type"] == "request_pending"
    assert _detail(second)["request_id"] == first["request_id"]


@pytest.mark.parametrize(
    "action, reason, error_type",
    [
        ({"tool": "no_such_tool"}, "why", "unknown_tool"),
        ({"tool": "get_stage_position"}, "why", "not_an_action"),
        (
            {"tool": "update_item_task_config", "arguments": {"patch": {}}},
            "why",
            "invalid_arguments",
        ),
        (NUDGE, "", "missing_field"),
    ],
)
def test_a_malformed_request_is_refused(client, action, reason, error_type):
    refused = _file(client, action, reason=reason)
    assert refused.status_code == 422
    assert _detail(refused)["error_type"] == error_type


def test_hardware_needs_the_devices_service(microscope, context, auth):
    app = build_server(
        microscope, app_context=context, auth=auth, serve_microscope=False
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        refused = _file(client, NUDGE)
    assert refused.status_code == 422
    assert _detail(refused)["error_type"] == "not_available"


def test_approve_runs_each_action_through_its_route(client, app, auth, microscope):
    before = microscope.get_stage_position()
    filed = _file(client, NUDGE, SUPERVISE).json()
    heard = auth.last_seen_monotonic

    store = app.state.requests
    store.approve(filed["request_id"])
    done = store.wait(filed["request_id"], timeout=10)

    assert done.status == "applied"
    assert [r["ok"] for r in done.results] == [True, True]
    assert microscope.get_stage_position().x == pytest.approx(before.x + 1e-6)
    # The app ran these, not the agent: it was not heard from.
    assert auth.last_seen_monotonic == heard
    # Drive was lent for those calls only.
    direct = client.post(MOVE_RELATIVE, headers=AGENT, json={})
    assert direct.status_code == 403


def test_every_step_is_on_the_record_with_who_acted(client, app, context):
    filed = _file(client, NUDGE).json()
    store = app.state.requests
    store.approve(filed["request_id"])
    store.wait(filed["request_id"], timeout=10)

    steps = [(kind, actor) for kind, _, actor in context.recorded if "request" in kind]
    assert steps == [
        ("request_raised", "agent"),
        ("request_decided", None),
        ("request_applied", "agent"),
        ("request_finished", "agent"),
    ]
    raised = context.recorded[0][1]
    assert raised["reason"] == "checking the trenches"
    assert raised["actions"] == [NUDGE]
    decided = context.recorded[1][1]
    assert decided["outcome"] == "approved"
    assert decided["decided_by"] == "operator"


def test_a_refused_action_stops_the_batch_with_the_routes_error(client, app):
    bad = {
        "tool": "call_service_command",
        "arguments": {"service": "no_such_service", "command": "run"},
    }
    filed = _file(client, NUDGE, bad, NUDGE).json()
    store = app.state.requests
    store.approve(filed["request_id"])
    done = store.wait(filed["request_id"], timeout=10)

    assert done.status == "failed"
    assert [r["ok"] for r in done.results] == [True, False]
    assert done.results[1]["status_code"] == 404


def test_a_run_in_control_refuses_an_approved_move(client, app, microscope):
    filed = _file(client, NUDGE).json()
    store = app.state.requests
    with microscope.control.take(WORKFLOW, "a lamella run"):
        store.approve(filed["request_id"])
        done = store.wait(filed["request_id"], timeout=10)
    assert done.status == "failed"
    assert done.results[0]["status_code"] == 409
    assert done.results[0]["error"]["error_type"] == "microscope_busy"
    assert done.results[0]["error"]["in_control"]["by"] == WORKFLOW


def test_reject_runs_nothing_and_keeps_the_operators_words(client, app, microscope):
    before = microscope.get_stage_position()
    filed = _file(client, NUDGE).json()
    store = app.state.requests
    store.reject(filed["request_id"], reason="leave it, I'll centre it")

    read = client.get(f"/app/agent/requests/{filed['request_id']}", headers=AGENT)
    assert read.json()["status"] == "rejected"
    assert read.json()["reject_reason"] == "leave it, I'll centre it"
    assert microscope.get_stage_position().x == pytest.approx(before.x)
    # Decided once: a second decision is refused.
    with pytest.raises(RequestRefused):
        store.approve(filed["request_id"])
    # And the next request can be filed.
    assert _file(client, NUDGE).status_code == 202


def test_a_request_made_at_assist_is_not_run_after_the_level_drops(client, app, auth):
    filed = _file(client, NUDGE).json()
    auth.set_level(AgentLevel.WATCH)
    store = app.state.requests
    with pytest.raises(RequestRefused) as refused:
        store.approve(filed["request_id"])
    # The level change already cancelled it: there is nothing left to approve.
    assert refused.value.detail["error_type"] == "not_pending"
    assert store.get(filed["request_id"]).status == "cancelled"


def test_assist_asks_to_stop_as_it_asks_for_the_rest(client, app):
    """A stop that is not the agent's own needs Drive (Patrick, 2026-10-14): at
    Assist it is a request like any action, and runs once approved."""
    why = {"reason": "the mill is off target"}
    for path, body in (
        ("/app/workflow/stop", why),
        ("/microscope/services/milling/commands/stop", why),
        (MOVE_RELATIVE, {}),
        ("/app/workflow/prompt/answer", {}),
    ):
        refused = client.post(path, headers=AGENT, json=body)
        assert refused.status_code == 403, path
        assert _detail(refused)["error_type"] == "ask_first"
        assert "request_actions" in _detail(refused)["message"]
    filed = _file(client, STOP_MILL)
    assert filed.status_code == 202, filed.text
    store = app.state.requests
    store.approve(filed.json()["request_id"])
    done = store.wait(filed.json()["request_id"], timeout=10)
    assert done.status == "applied", done.results


@pytest.mark.parametrize("stop_agent", [False, True])
def test_a_level_change_or_stop_agent_cancels_the_waiting_request(
    client, app, auth, context, stop_agent
):
    filed = _file(client, NUDGE).json()
    if stop_agent:
        auth.stop_agent()
    else:
        auth.set_level(AgentLevel.DRIVE)
    store = app.state.requests
    assert store.get(filed["request_id"]).status == "cancelled"
    assert store.pending() is None
    why = [p["why"] for kind, p, _ in context.recorded if kind == "request_cancelled"]
    assert why == [
        "the operator stopped the agent"
        if stop_agent
        else "the agent level changed to Drive"
    ]


def test_setting_the_same_level_again_cancels_nothing(client, app, auth):
    filed = _file(client, NUDGE).json()
    auth.set_level(AgentLevel.ASSIST)
    assert app.state.requests.get(filed["request_id"]).status == "pending"


def test_a_level_change_mid_batch_runs_no_more_of_it(client, app, auth, microscope):
    before = microscope.get_stage_position()
    filed = _file(client, NUDGE, NUDGE).json()
    store = app.state.requests
    # Leave Assist and come back while the first action runs: the rest of the
    # batch was approved at a level that has since changed, so it does not run.
    store.subscribe(
        lambda request: (
            request.results
            and len(request.results) == 1
            and (auth.set_level(AgentLevel.WATCH), auth.set_level(AgentLevel.ASSIST))
        )
    )
    store.approve(filed["request_id"])
    done = store.wait(filed["request_id"], timeout=10)
    assert done.status == "cancelled"
    assert len(done.results) == 1
    assert microscope.get_stage_position().x == pytest.approx(before.x + 1e-6)


def test_a_level_listener_does_not_keep_its_owner_alive(auth):
    class Owner:
        def heard(self, level, stopped):
            raise AssertionError("a dropped owner is not called")

    owner = Owner()
    auth.on_level_change(owner.heard)
    del owner
    gc.collect()
    auth.set_level(AgentLevel.WATCH)
    assert auth._listeners == []


def test_cancel_clears_the_waiting_request(client, app):
    filed = _file(client, NUDGE).json()
    store = app.state.requests
    assert store.cancel("Stop agent").id == filed["request_id"]
    assert store.pending() is None
    assert store.cancel("again") is None


def test_an_expiring_request_lapses_as_it_says(app):
    store = app.state.requests
    request = store.create([NUDGE], "countdown", expires_at=100.0, on_expiry="reject")
    assert store.expire_due(now=99.0) is None
    assert store.expire_due(now=100.0).status == "rejected"
    assert request.decided_by == "expiry"


# The arguments a named tool cannot be called without.
_AXES = {axis: 0.0 for axis in "xyzrt"}
_SAMPLE_ARGUMENTS = {
    "stable_move": {"dx": 1e-6, "dy": 0.0},
    "vertical_move": {"dy": 1e-6},
    "move_to_orientation": {"orientation": "SEM"},
    "move_stage_absolute": _AXES,
    "safe_move_stage_absolute": _AXES,
    "move_to_milling_angle": {"milling_angle_deg": 18.0},
    "call_command": {"device": "stage", "command": "home"},
    "call_service_command": {"service": "milling", "command": "run"},
}


def test_every_acting_tool_resolves_to_its_route():
    for spec in CATALOG:
        if spec.level != "drive":
            continue
        arguments = {name: "x" for name in spec.params if "{" + name + "}" in spec.path}
        arguments.update(_SAMPLE_ARGUMENTS.get(spec.name, {}))
        method, path, _ = route_call(spec, arguments)
        assert method == spec.method
        assert "{" not in path, spec.name


def test_each_action_is_described_by_the_server_with_old_and_new(client, context):
    item = context._host.experiment.positions[0]
    item.description = "first pass"
    edit = {
        "tool": "update_item_detail",
        "arguments": {
            "item_name": item.name,
            "patch": {"description": "trench floor high"},
            "version": "v",
        },
    }
    filed = _file(client, NUDGE, edit, SUPERVISE).json()
    lines = [(a["what"], a["change"]) for a in filed["actions"]]
    assert lines == [
        ("Stage", "Move by x 1 µm"),
        (item.name, "description: first pass → trench floor high"),
        ("MILL_ROUGH", "Supervised: ask at each decision"),
    ]


def test_an_unknown_tool_in_the_describer_falls_back_to_its_name():
    from fibsem.server.request_text import describe_action

    assert describe_action("new_tool", {}, "/x") == {
        "what": "new_tool",
        "change": "Run",
    }


@pytest.mark.parametrize(
    "action",
    [
        {"tool": "call_command", "arguments": {"device": "stage", "command": "home"}},
        {
            "tool": "set_parameter",
            "arguments": {"device": "electron", "parameter": "on", "value": False},
        },
    ],
)
def test_what_the_agent_may_never_touch_is_refused_at_filing(client, app, action):
    """The operator is never asked about an action that cannot run: a blocked
    command or write is refused when it is filed, as it is when called."""
    refused = _file(client, NUDGE, action)
    assert refused.status_code == 403
    assert _detail(refused)["error_type"] == "not_allowed"
    assert _detail(refused)["message"].startswith("Action 2: ")
    assert app.state.requests.pending() is None


def test_the_agent_withdraws_its_waiting_request_with_a_reason(client, app, context):
    filed = _file(client, NUDGE).json()
    url = f"/app/agent/requests/{filed['request_id']}/withdraw"
    no_reason = client.post(url, headers=AGENT, json={"reason": " "})
    assert no_reason.status_code == 422
    withdrawn = client.post(
        url, headers=AGENT, json={"reason": "the operator moved it"}
    )
    assert withdrawn.status_code == 200, withdrawn.text
    why = "withdrawn by the agent: the operator moved it"
    assert withdrawn.json()["status"] == "cancelled"
    assert withdrawn.json()["cancel_reason"] == why
    cancelled = [p for kind, p, _ in context.recorded if kind == "request_cancelled"]
    assert cancelled == [{"request_id": filed["request_id"], "why": why}]
    # A new one can be filed; a decided one can't be withdrawn.
    again = _file(client, NUDGE).json()
    app.state.requests.reject(again["request_id"])
    late = client.post(
        f"/app/agent/requests/{again['request_id']}/withdraw",
        headers=AGENT,
        json={"reason": "too late"},
    )
    assert late.status_code == 409
    assert _detail(late)["error_type"] == "not_pending"
    missing = client.post(
        "/app/agent/requests/nope/withdraw", headers=AGENT, json={"reason": "x"}
    )
    assert missing.status_code == 404


def test_the_sidecar_shows_an_approved_image_as_an_image(client, app):
    """get_request answers an applied image action's preview as an image the
    agent can look at, not ~200k characters of base64 in the JSON."""
    pytest.importorskip("mcp")
    import asyncio

    from fibsem.mcp.sidecar import build_sidecar

    image = {"tool": "acquire_image_preview", "arguments": {"beam_type": "ION"}}
    request_id = _file(client, image).json()["request_id"]
    store = app.state.requests
    store.approve(request_id)
    assert store.wait(request_id, timeout=10).status == "applied"

    client.headers.update(AGENT)
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(sidecar.call_tool("get_request", {"request_id": request_id}))
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    assert [getattr(c, "type", None) for c in contents] == ["text", "image"]
    assert "image_b64_jpeg" not in contents[0].text
    answer = json.loads(contents[0].text)
    preview = answer["results"][0]["result"]["result"]["preview"]
    assert preview["image_index"] == 0
