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

import fibsem.applications.autolamella  # noqa: E402
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
NUDGE = {"tool": "stage_move_relative", "arguments": {"dx": 1e-6}}
SUPERVISE = {
    "tool": "protocol_set_supervision",
    "arguments": {"task_name": "MILL_ROUGH", "supervise": True},
}
STOP_MILL = {"tool": "milling_stop", "arguments": {"reason": "the mill is off target"}}
MOVE_RELATIVE = "/microscope/services/stage_movement/commands/move_relative"
_PROTOCOL = os.path.join(
    os.path.dirname(fibsem.applications.autolamella.__file__),
    "protocol",
    "task-protocol.yaml",
)


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
        ({"tool": "stage_get_position"}, "why", "not_an_action"),
        (
            {"tool": "item_update_task_config", "arguments": {"patch": {}}},
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


def test_a_request_carries_the_image_its_evidence_is_in(client, tmp_path):
    # FIB-1411: the agent's own image, shown under Why. Inside the experiment.
    folder = tmp_path / "exp" / "requests-exp"
    folder.mkdir(parents=True, exist_ok=True)
    inside = folder / "ref_trench.tif"
    inside.write_bytes(b"tif")
    outside = tmp_path / "elsewhere.tif"
    outside.write_bytes(b"tif")

    def file(image):
        return client.post(
            "/app/agent/requests",
            headers=AGENT,
            json={"actions": [NUDGE], "reason": "the trench", "image": image},
        )

    assert _detail(file(str(outside)))["error_type"] == "not_allowed"
    missing = file(str(folder / "none.tif"))
    assert missing.status_code == 422 and _detail(missing)["error_type"] == "invalid"
    filed = file(str(inside))
    assert filed.status_code == 202, filed.text
    assert filed.json()["image"] == str(inside)


@pytest.mark.parametrize(
    "tool, arguments, unknown, missing",
    [
        # FIB-1425: both were filed, and read "None" on the card.
        (
            "item_move_to_pose",
            {"item_name": "01", "pose_name": "MILLING"},
            ["pose_name"],
            ["pose"],
        ),
        ("protocol_load", {"path": "/data/protocol.yaml"}, ["path"], ["protocol"]),
        ("protocol_load", {"protocol": None}, [], ["protocol"]),
    ],
)
def test_arguments_the_tool_does_not_take_are_refused_at_filing(
    client, app, tool, arguments, unknown, missing
):
    refused = _file(client, {"tool": tool, "arguments": arguments})
    assert refused.status_code == 422
    detail = _detail(refused)
    assert detail["error_type"] == "invalid_arguments"
    assert (detail["tool"], detail["unknown"], detail["missing"]) == (
        tool,
        unknown,
        missing,
    )
    assert detail["message"].startswith(f"Action 1: {tool} ")
    for name in unknown + missing:
        assert name in detail["message"]
    assert app.state.requests.pending() is None


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
    # With the card's words, for the app's Recent list (sim session 2026-10-16 F7).
    (action,) = raised["actions"]
    assert action.pop("change").startswith("Move 1 µm in x, from x ")
    assert action == {**NUDGE, "what": "Stage"}
    decided = context.recorded[1][1]
    assert decided["outcome"] == "approved"
    assert decided["decided_by"] == "operator"


def test_a_refused_action_stops_the_batch_with_the_routes_error(client, app):
    bad = {
        "tool": "service_call_command",
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


def test_assist_asks_to_stop_as_it_asks_for_the_rest(
    client, app, microscope, monkeypatch
):
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
        assert "request_create" in _detail(refused)["message"]
    filed = _file(client, STOP_MILL)
    assert filed.status_code == 202, filed.text
    # Something to stop (FIB-1398).
    monkeypatch.setattr(microscope.milling, "_running", True)
    store = app.state.requests
    store.approve(filed.json()["request_id"])
    done = store.wait(filed.json()["request_id"], timeout=10)
    assert done.status == "applied", done.results


def test_a_request_whose_mill_was_stopped_says_so(
    client, app, context, microscope, monkeypatch
):
    """A mill stopped part way answers ``cancelled``, not a failure: the request
    is applied, and its result and end say the run was cut short (FIB-1398)."""
    monkeypatch.setattr(microscope.milling, "run", lambda stop_event=None: "cancelled")
    run = {
        "tool": "service_call_command",
        "arguments": {"service": "milling", "command": "run"},
    }
    filed = _file(client, run).json()
    store = app.state.requests
    store.approve(filed["request_id"])
    done = store.wait(filed["request_id"], timeout=10)
    assert done.status == "applied", done.results
    assert done.results[0]["cancelled"] is True
    finished = [p for kind, p, _ in context.recorded if kind == "request_finished"]
    assert finished[-1]["cancelled"] == 1


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
    "stage_stable_move": {"dx": 1e-6, "dy": 0.0},
    "stage_vertical_move": {"dy": 1e-6},
    "stage_move_to_orientation": {"orientation": "SEM"},
    "stage_move_absolute": _AXES,
    "stage_move_to_milling_angle": {"milling_angle_deg": 18.0},
    "alignment_align_to_reference": {"reference_path": "experiment/reference.tif"},
    "beam_acquire_overview": {
        "settings": {"nrows": 2, "ncols": 2},
        "path": "experiment",
    },
    "beam_acquire_focus_stack": {"image_settings": {"hfw": 1e-4}, "path": "experiment"},
    "fm_acquire_image": {"channels": [{"name": "GFP"}], "path": "experiment"},
    "milling_run_stage": {"stage": {"name": "Rough"}},
    "milling_run_task": {"config": {"name": "Mill"}, "path": "experiment"},
    "fm_acquire_overview": {
        "channels": [{"name": "GFP"}],
        "parameters": {"rows": 1, "cols": 2},
        "path": "experiment",
    },
    "grid_load": {"grid_name": "G2"},
    "grid_move_to": {"grid_name": "G2"},
    "grid_move_to_slot": {"slot_name": "Shuttle"},
    "device_call_command": {"device": "stage", "command": "home"},
    "service_call_command": {"service": "milling", "command": "run"},
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
        "tool": "item_update",
        "arguments": {
            "item_name": item.name,
            "patch": {"description": "trench floor high"},
            "version": "v",
        },
    }
    filed = _file(client, NUDGE, edit, SUPERVISE).json()
    lines = [(a["what"], a["change"]) for a in filed["actions"]]
    # The nudge says from where on the axis (sim session 2026-10-16).
    assert lines[0][1].startswith("Move 1 µm in x, from x ")
    lines[0] = ("Stage", "Move 1 µm in x")
    assert lines == [
        ("Stage", "Move 1 µm in x"),
        (item.name, "description: first pass → trench floor high"),
        ("MILL_ROUGH", "Supervised: ask at each decision"),
    ]


def test_a_config_edit_reads_as_its_form_names_it_in_display_units(client, context):
    """FIB-1403: the setting under its milling stage, old -> new in the form's
    units, and the same as a table; never the dotted path."""
    item = context._host.experiment.positions[0]
    protocol = AutoLamellaTaskProtocol.load(_PROTOCOL)
    rough = next(
        c for c in protocol.task_config.values() if c.task_type == "MILL_ROUGH"
    )
    item.task_config["Rough Milling"] = rough
    stage = "milling.mill_rough.stages.0"
    edit = {
        "tool": "item_update_task_config",
        "arguments": {
            "item_name": item.name,
            "task_name": "Rough Milling",
            "patch": {
                f"{stage}.pattern.depth": 2.5e-6,
                f"{stage}.milling.milling_current": 6e-11,
            },
            "version": "v",
        },
    }
    action = _file(client, edit).json()["actions"][0]
    assert action["what"] == f"{item.name} · Rough Milling"
    assert action["change"] == (
        "Rough Mill 01 · trench depth 2 µm → 2.5 µm\n"
        "Rough Mill 01 · milling current 740 pA → 60 pA"
    )
    assert action["details"] == {
        "columns": ["Setting", "Now", "New"],
        "rows": [
            ["Rough Mill 01 · trench depth", "2 µm", "2.5 µm"],
            ["Rough Mill 01 · milling current", "740 pA", "60 pA"],
        ],
    }


def test_a_mill_reads_as_its_patterns_and_its_run_as_how_long():
    from fibsem.server.request_text import describe_actions
    from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

    def rectangle(x):
        pattern = FibsemRectangleSettings(
            width=2e-6, height=1e-6, depth=0.2e-6, centre_x=x, centre_y=4e-6
        )
        return {**pattern.to_dict(), "type": "Rectangle"}

    settings = FibsemMillingSettings(milling_current=2e-11).to_dict()
    prepare, run = describe_actions(
        [
            (
                "service_call_command",
                {
                    "service": "milling",
                    "command": "prepare",
                    "kwargs": {
                        "settings": settings,
                        "patterns": [rectangle(-3e-6), rectangle(0.0)],
                        "name": "agent test mill",
                    },
                },
                "/x",
            ),
            ("service_call_command", {"service": "milling", "command": "run"}, "/x"),
        ]
    )
    assert prepare["what"] == "Milling"
    assert prepare["change"] == (
        "Prepare 'agent test mill': 2 rectangles, 2 × 1 µm, 0.2 µm deep, 20 pA at 30 kV"
    )
    assert prepare["details"]["rows"] == [
        ["Rectangle 1", "2 × 1 µm", "0.2 µm", "(-3.0, +4.0) µm"],
        ["Rectangle 2", "2 × 1 µm", "0.2 µm", "(+0.0, +4.0) µm"],
    ]
    assert run["what"] == "Milling"
    assert run["change"].startswith("Run, about ")


@pytest.mark.parametrize(
    "tool, arguments, line",
    [
        (
            "stage_stable_move",
            {"dx": 33e-6, "dy": -27e-6, "beam_type": "ION"},
            ("Stage", "Stable move in the ion beam: 33 µm right, 27 µm down"),
        ),
        (
            "milling_stop",
            {"reason": "off target"},
            ("Milling", "Stop the mill now running"),
        ),
        (
            "stage_move_to_orientation",
            {"orientation": "fib"},
            ("Stage", "Move to the FIB orientation"),
        ),
        (
            "service_call_command",
            {"service": "milling", "command": "run"},
            ("Milling", "Run"),
        ),
        (
            "device_call_command",
            {"device": "ion_beam", "command": "autofocus"},
            ("Ion beam", "Autofocus"),
        ),
    ],
)
def test_an_action_reads_as_one_plain_line(tool, arguments, line):
    from fibsem.server.request_text import describe_action

    described = describe_action(tool, arguments, "/x")
    assert (described["what"], described["change"]) == line


def test_a_long_argument_is_cut_in_the_line_and_whole_in_the_details():
    from fibsem.server.request_text import describe_action

    path = "/data/" + "a" * 80 + ".tif"
    described = describe_action(
        "service_call_command",
        {"service": "imaging", "command": "focus_sweep", "kwargs": {"path": path}},
        "/x",
    )
    assert path not in described["change"]
    assert described["change"].startswith("Focus sweep: path /data/")
    assert described["details"]["rows"] == [["path", path]]


def test_an_unknown_tool_in_the_describer_falls_back_to_its_name():
    from fibsem.server.request_text import describe_action

    assert describe_action("new_tool", {}, "/x") == {
        "what": "New tool",
        "change": "Run",
    }
    described = describe_action("new_tool", {"depth": 2e-6, "names": ["a"]}, "/x")
    assert described["change"] == "Run"
    assert described["details"]["rows"] == [["depth", "2e-06"], ["names", "a"]]


def test_no_acting_tool_reads_as_its_name_or_as_json():
    """FIB-1412: every tool the agent can put in a request has a plain line:
    the subject as the header, never the tool's name, and no JSON."""
    from fibsem.server.request_text import describe_action

    for spec in CATALOG:
        if spec.level != "drive":
            continue
        arguments = {name: "x" for name in spec.params if "{" + name + "}" in spec.path}
        arguments.update(_SAMPLE_ARGUMENTS.get(spec.name, {}))
        described = describe_action(spec.name, arguments, spec.path)
        assert described["what"] != spec.name, spec.name
        assert described["what"] != spec.name.replace("_", " ").capitalize(), spec.name
        text = described["change"] + str(described.get("details", ""))
        assert "{" not in described["change"] and '"' not in text, spec.name


def test_no_acting_tool_reads_none_with_its_required_arguments():
    """FIB-1425: filing needs every required argument, so a line never shows a
    value left out as "None"."""
    from fibsem.server.request_text import describe_action

    for spec in CATALOG:
        if spec.level != "drive":
            continue
        arguments = {name: "x" for name in spec.params if name not in spec.optional}
        arguments.update(_SAMPLE_ARGUMENTS.get(spec.name, {}))
        described = describe_action(spec.name, arguments, spec.path)
        text = described["what"] + described["change"]
        assert "None" not in text, (spec.name, text)


def _reader(answers):
    return lambda path: answers.get(path)


def test_a_run_task_reads_by_its_stages_with_whose_estimate_it_is():
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.server.request_text import describe_action

    protocol = AutoLamellaTaskProtocol.load(_PROTOCOL)
    rough = next(
        c for c in protocol.task_config.values() if c.task_type == "MILL_ROUGH"
    )
    config = rough.milling["mill_rough"]
    assert isinstance(config, FibsemMillingTaskConfig)
    config.name = "agent test task"
    described = describe_action(
        "service_call_command",
        {
            "service": "milling",
            "command": "run_task",
            "kwargs": {"config": config.to_dict()},
        },
        "/x",
    )
    names = [s.name for s in config.stages if s.enabled]
    assert described["what"] == "Milling"
    assert described["change"].startswith(
        f"Run task 'agent test task': {len(names)} stage"
    )
    assert names[0] in described["change"]
    assert "estimated from the sputter rate" in described["change"]
    table = described["details"]
    assert table["columns"] == ["Stage", "Pattern", "Current", "Depth", "Position"]
    assert [row[0] for row in table["rows"]] == names
    assert table["rows"][0][2].endswith("A")


@pytest.mark.parametrize(
    "tool, arguments, answers, what, change",
    [
        (
            "stage_move_absolute",
            {"x": 1.23456e-3, "y": -9e-6, "z": 0.0, "r": 0.0, "t": 0.0},
            {
                "/microscope/devices/stage/position": {
                    "value": {"x": 1e-3, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0}
                }
            },
            "Stage",
            "Safe move to x 1.2346 mm, y −9 µm, z 0 µm, tilt 0°, rotation 0°\n"
            "from x 1 mm, y 0 µm, z 0 µm, tilt 0°, rotation 0°",
        ),
        (
            "stage_vertical_move",
            {"dy": 2e-6, "relaxation": 0.5},
            {},
            "Stage",
            "Vertical move, to centre a feature 2 µm up in the ion beam, "
            "50% of the way",
        ),
        (
            "beam_acquire_image",
            {"beam_type": "ION", "image_settings": {"hfw": 80e-6}},
            {},
            "Ion beam",
            "Take an image: 80 µm wide, 1536 × 1024, 1 µs dwell. "
            "These settings stay on the beam",
        ),
        (
            "device_set_parameter",
            {"device": "ion", "parameter": "current", "value": 6e-11},
            {
                "/microscope/devices/ion/current": {"value": 2e-11},
                "/microscope/devices/ion": {
                    "parameters": {"current": {"unit": "A", "display": None}}
                },
            },
            "Ion",
            "Set current 20 pA → 60 pA",
        ),
        (
            "item_update",
            {"item_name": "01-full-magpie", "patch": {"poi.x": 2e-6}, "version": "v"},
            {"/x": {"poi": {"x": 1e-6, "y": 0.0}}},
            "01-full-magpie",
            "Point of interest: (+1, 0) → (+2, 0) µm",
        ),
        (
            "fm_acquire_image",
            {
                "channels": [
                    {
                        "name": "GFP",
                        "excitation_wavelength": 488,
                        "power": 0.05,
                        "exposure_time": 0.1,
                    }
                ],
                "zparams": {"zmin": -10e-6, "zmax": 10e-6, "zstep": 1e-6},
            },
            {},
            "Fluorescence",
            "Take a z-stack, −10 µm to +10 µm in 1 µm steps (21 planes), "
            "in 1 channel (GFP), here",
        ),
        (
            "experiment_load",
            {"path": "C:\\data\\AutoLamella-2026-10-15\\experiment.yaml"},
            {
                "/app/workflow/status": {
                    "experiment": {"name": "AutoLamella-2026-10-14"}
                }
            },
            "Experiment",
            "Open 'AutoLamella-2026-10-15' in place of 'AutoLamella-2026-10-14'",
        ),
        (
            "experiment_load",
            {"path": "/data/AutoLamella-2026-10-15/experiment.yaml"},
            {
                "/app/experiment/recent": {
                    "items": [
                        {
                            "path": "/data/AutoLamella-2026-10-15/experiment.yaml",
                            "num_items": 5,
                            "created_at": "2026-10-15T09:12:00",
                        }
                    ]
                }
            },
            "Experiment",
            "Open 'AutoLamella-2026-10-15' (5 lamellae, created 15 Oct)",
        ),
        (
            "experiment_load",
            {"path": "/data/AutoLamella-2026-10-15/"},
            {},
            "Experiment",
            "Open 'AutoLamella-2026-10-15'",
        ),
        (
            "review_decide",
            {
                "item_id": "abc",
                "task_name": "Setup",
                "outcome": "Confirmed",
                "values": {"poi": {"x": 1.2e-6, "y": -3e-6}},
            },
            {
                "/app/workflow/reviews": {
                    "reviews": [{"item_id": "abc", "item_name": "02-saving-wolf"}]
                }
            },
            "02-saving-wolf · Setup",
            "Confirm the proposal with POI x 1.2 µm, y −3 µm",
        ),
        (
            "beam_acquire_overview",
            {
                "settings": {
                    "nrows": 3,
                    "ncols": 3,
                    "image_settings": {"hfw": 150e-6, "beam_type": "ELECTRON"},
                },
                "path": "experiment",
            },
            {},
            "Overview",
            "Take a 3 × 3 SEM overview, 150 µm tiles (about 420 × 280 µm), here",
        ),
        (
            "item_add",
            {"stage_position": {"x": 1.2e-6, "y": -9e-6, "z": 0.0, "r": 0.0, "t": 0.0}},
            {},
            "Lamella",
            "Add at x 1.2 µm, y −9 µm, z 0 µm, tilt 0°, rotation 0°",
        ),
        (
            "experiment_create",
            {"name": "assist-test-2026-10-15"},
            {"/app/workflow/status": {"experiment": {"name": "older"}}},
            "Experiment",
            "Create and open 'assist-test-2026-10-15', with the default protocol",
        ),
        (
            "workflow_stop",
            {"reason": "off target"},
            {},
            "Workflow",
            "Stop the run now running. A mill in progress stops too",
        ),
    ],
)
def test_each_kind_reads_in_display_units_with_old_and_new(
    tool, arguments, answers, what, change
):
    from fibsem.server.request_text import describe_action

    path = "/microscope/devices/ion/current" if tool == "device_set_parameter" else "/x"
    described = describe_action(tool, arguments, path, read=_reader(answers))
    assert (described["what"], described["change"]) == (what, change)


def test_a_new_experiment_says_what_it_replaces():
    from fibsem.server.request_text import describe_action

    described = describe_action(
        "experiment_create",
        {"name": "next", "protocol": "task-protocol.yaml"},
        "/x",
        read=_reader({"/app/workflow/status": {"experiment": {"name": "older"}}}),
    )
    assert ["Replaces", "older"] in described["details"]["rows"]
    assert described["change"].endswith("with protocol task-protocol.yaml")


@pytest.mark.parametrize(
    "action, call",
    [
        (
            {
                "tool": "device_call_command",
                "arguments": {"device": "stage", "command": "home"},
            },
            "device_call_command stage.home",
        ),
        (
            {
                "tool": "device_set_parameter",
                "arguments": {"device": "electron", "parameter": "on", "value": False},
            },
            "device_set_parameter electron.on",
        ),
    ],
)
def test_what_the_agent_may_never_touch_is_refused_at_filing(
    client, app, microscope, action, call
):
    """The operator is never asked about an action that cannot run: a blocked
    command or write is refused when it is filed, as it is when called. The
    record names the refused call, not the request route (FIB-1421)."""
    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload))

    microscope.session.record_signal.connect(record)
    try:
        refused = _file(client, NUDGE, action)
    finally:
        microscope.session.record_signal.disconnect(record)
    assert refused.status_code == 403
    assert _detail(refused)["error_type"] == "not_allowed"
    assert _detail(refused)["message"].startswith("Action 2: ")
    assert app.state.requests.pending() is None
    refusals = [payload for kind, payload in recorded if kind == "agent_refused"]
    assert [p["action"] for p in refusals] == [call]
    assert refusals[0]["route"] == "POST /app/agent/requests"


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
    assert cancelled == [
        {"request_id": filed["request_id"], "why": why, "withdrawn": True}
    ]
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

    image = {"tool": "beam_acquire_image", "arguments": {"beam_type": "ION"}}
    request_id = _file(client, image).json()["request_id"]
    store = app.state.requests
    store.approve(request_id)
    assert store.wait(request_id, timeout=10).status == "applied"

    client.headers.update(AGENT)
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(sidecar.call_tool("request_get", {"request_id": request_id}))
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    assert [getattr(c, "type", None) for c in contents] == ["text", "image"]
    assert "image_b64_jpeg" not in contents[0].text
    answer = json.loads(contents[0].text)
    preview = answer["results"][0]["result"]["result"]["preview"]
    assert preview["image_index"] == 0


def test_a_waiting_request_is_described_again_with_the_values_now(client, app, context):
    """FIB-1412: the old value is read when the operator looks, not only when
    the request was filed."""
    item = context._host.experiment.positions[0]
    item.description = "first pass"
    edit = {
        "tool": "item_update",
        "arguments": {
            "item_name": item.name,
            "patch": {"description": "trench floor high"},
            "version": "v",
        },
    }
    filed = _file(client, edit).json()
    store = app.state.requests
    changed = []
    store.subscribe(changed.append)
    item.description = "second pass"
    assert store.refresh(filed["request_id"])
    action = store.get(filed["request_id"]).to_dict()["actions"][0]
    assert action["change"] == "description: second pass → trench floor high"
    assert len(changed) == 1
    store.reject(filed["request_id"])
    assert not store.refresh(filed["request_id"])


def test_an_image_s_settings_read_in_display_units_now_and_new():
    """FIB-1428: no raw SI in the details; the beam's settings now beside what
    the image sets them to."""
    from fibsem.server.request_text import describe_action

    answers = {
        "/microscope/devices/ion/hfw": {"value": 3e-4},
        "/microscope/devices/ion/dwell_time": {"value": 2e-7},
    }
    described = describe_action(
        "beam_acquire_image",
        {"beam_type": "ION", "image_settings": {"hfw": 1.5e-4, "save": True}},
        "",
        read=_reader(answers),
    )
    assert described["details"] == {
        "columns": ["Setting", "Now", "New"],
        "rows": [
            ["Field of view", "300 µm", "150 µm"],
            ["Resolution", "", "1536 × 1024"],
            ["Dwell time", "200 ns", "1 µs"],
            ["Save", "", "yes"],
        ],
    }


def test_an_overview_s_details_are_labelled_and_short():
    from fibsem.server.request_text import describe_action

    described = describe_action(
        "beam_acquire_overview",
        {
            "settings": {
                "nrows": 2,
                "ncols": 3,
                "overlap": 0.1,
                "image_settings": {"hfw": 1e-4, "dwell_time": 1e-6},
            },
            "path": "/data/assist-test/overviews",
            "name": "grid-1",
        },
        "",
    )
    rows = dict(map(tuple, described["details"]["rows"]))
    assert rows["Rows"] == "2" and rows["Columns"] == "3"
    assert rows["Overlap"] == "10%"
    assert rows["Field of view"] == "100 µm" and rows["Dwell time"] == "1 µs"
    assert rows["Saved as"] == "grid-1 in overviews"


def test_an_item_edit_with_no_old_value_says_none():
    from fibsem.server.request_text import describe_action

    described = describe_action(
        "item_update",
        {"item_name": "01", "patch": {"poi": {"x": -1.8e-6, "y": 0.5e-6}}},
        "/x",
        read=_reader({"/x": {"poi": {"x": None, "y": None}}}),
    )
    assert described["change"] == "Point of interest: (none) → (−1.8, +0.5) µm"


def test_no_details_table_has_blank_headers():
    from fibsem.server.request_text import describe_action

    for tool, arguments in (
        ("item_add", {"stage_position": {"x": 0.0, "coordinate_system": "RAW"}}),
        ("experiment_create", {"name": "assist-test"}),
        ("experiment_load", {"path": "/data/assist-test"}),
    ):
        details = describe_action(tool, arguments, "").get("details")
        assert details is None or all(details["columns"]), tool


def test_fluorescence_channel_settings_read_in_the_fm_panel_s_units():
    """Sim session 2026-10-16 F5: "power 0.05, exposure time 0.1"."""
    from fibsem.server.request_text import describe_action

    channel = {
        "name": "GFP",
        "excitation_wavelength": 488,
        "emission_wavelength": 520,
        "power": 0.05,
        "exposure_time": 0.1,
    }
    line = describe_action("fm_acquire_image", {"channels": [channel]}, "")
    assert line["details"]["rows"] == [
        [
            "GFP",
            "excitation wavelength 488 nm, emission wavelength 520 nm, "
            "power 5%, exposure 100 ms",
        ]
    ]


def test_a_filed_request_and_a_beam_switched_off_read_as_words():
    """Sim session 2026-10-16: "Request create · Run", "Electron · Set on to off"."""
    from fibsem.server.request_text import describe_action

    filed = describe_action(
        "request_create",
        {"actions": [{"tool": "stage_move_relative", "arguments": {"dx": 1e-6}}]},
        "",
    )
    assert filed == {
        "what": "Request",
        "change": "Ask you to approve 1 action: Stage · Move 1 µm in x",
    }
    off = describe_action(
        "device_set_parameter",
        {"device": "electron", "parameter": "on", "value": False},
        "",
    )
    assert off == {"what": "Electron", "change": "Turn the electron beam off"}


def test_a_protocol_s_re_alignment_reads_as_the_milling_panel_says_it():
    """Sim session 2026-10-16: "Interval enabled off → on / Interval 30 → 60"."""
    from fibsem.server.request_text import describe_patch
    from fibsem.structures import MillingAlignment

    line = describe_patch(
        MillingAlignment(), {"interval_enabled": True, "interval": 60}
    )
    assert line["change"] == (
        "Re-align during milling: off → on\nRe-align every 30 s → 60 s"
    )
