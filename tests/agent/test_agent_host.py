"""The Agent Server: the agent's routes on their own, with no microscope or app.

The watch, the harness's checks and the Agent page's calls are made against it
as they are against the App Server, on a real socket where the watch runs."""

import base64
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.calls import EVIDENCE_HEADER
from fibsem.ai.agent.harness import _reachable
from fibsem.ai.agent.levels import AgentLevel
from fibsem.client.transport import Transport
from fibsem.server.agent.host import EVENTS_FILENAME, InstrumentLink, build_agent_host
from fibsem.server.auth import AuthConfig

AGENT = "agent-token"
OPERATOR = "operator-token"


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR)
    app = build_agent_host(auth, tmp_path / "host")
    yield app
    app.state.host.close()


def _as(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def agent(host):
    return TestClient(host, headers=_as(AGENT))


@pytest.fixture
def operator(host):
    return TestClient(host, headers=_as(OPERATOR))


def test_it_imports_no_microscope():
    code = (
        "import sys, fibsem.server.agent.host; "
        "print([m for m in sys.modules if m.startswith(('fibsem.microscope', "
        "'fibsem.server.server', 'fibsem.server.app_context'))])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_it_says_what_it_is(agent):
    capabilities = agent.get("/capabilities").json()
    assert capabilities["host"] == "agent"
    assert capabilities["instruments"] == []


def test_a_token_it_did_not_give_is_refused(host):
    assert TestClient(host).get("/app/events").status_code == 401
    assert TestClient(host, headers=_as("other")).get("/app/events").status_code == 401


def test_the_agent_can_t_reach_the_operator_s_side(agent):
    assert agent.put("/app/agent/level", json={"level": "drive"}).status_code == 403
    assert agent.post("/app/agent/stop").status_code == 403
    assert agent.post("/app/agent/messages", json={"text": "hi"}).status_code == 403


def test_the_session_and_its_plan_and_transcript(agent, operator):
    report = {"session_id": "s1", "model": "m", "goal": "read the papers"}
    assert agent.post("/app/agent/session", json=report).status_code == 200
    agent.post(
        "/app/agent/plan/section", json={"section": "Steps", "text": "- [ ] read"}
    )
    agent.post(
        "/app/agent/transcript",
        json={"session_id": "s1", "entries": [{"kind": "said", "text": "hello"}]},
    )

    session = operator.get("/app/agent/session").json()["session"]
    assert session["goal"] == "read the papers"
    assert session["checklist"]
    plan = operator.get("/app/agent/plan").json()
    assert plan["sections"] == [{"name": "Steps", "text": "- [ ] read"}]
    transcript = operator.get("/app/agent/transcript").json()
    assert [e["text"] for e in transcript["entries"]] == ["hello"]
    view = operator.get("/app/agent").json()
    assert view["level"] == "watch" and view["experiment"] is None


def test_operator_and_agent_messages_are_on_the_record(host, agent, operator):
    sent = operator.post("/app/agent/messages", json={"text": "look at 3"})
    assert sent.status_code == 201
    message_id = sent.json()["id"]
    answer = agent.post(
        "/app/experiment/messages", json={"text": "on it", "reply_to": message_id}
    )
    assert answer.status_code == 200

    events = agent.get("/app/events").json()["events"]
    to_agent, to_operator = [e for e in events if e["kind"] == "note"]
    assert to_agent["actor"] == "operator"
    assert to_agent["payload"] == {"text": "look at 3", "to": "agent", "id": message_id}
    assert to_operator["actor"] == "agent"
    assert to_operator["payload"]["reply_to"] == message_id


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 40, 160)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_evidence_waits_for_a_session(operator, agent):
    # Evidence is kept in the running session's folder: none before one.
    assert (
        agent.post("/app/experiment/evidence", json={"upload": _png()}).status_code
        == 409
    )
    marked = {"text": "this", "evidence": {"image": "a.png"}}
    assert operator.post("/app/agent/messages", json=marked).status_code == 409
    note = {"text": "x", "item_name": "01"}
    assert agent.post("/app/experiment/notes", json=note).status_code == 422


def test_pictures_go_both_ways_through_the_host(host, agent, operator):
    agent.post("/app/agent/session", json={"session_id": "s1", "model": "m"})
    kept = agent.post("/app/experiment/evidence", json={"upload": _png()})
    assert kept.status_code == 200
    evidence_id = kept.json()["evidence_id"]
    # The MCP server cites it on the call it judged for.
    cited = agent.post(
        "/app/experiment/messages",
        json={"text": "see this"},
        headers={EVIDENCE_HEADER: evidence_id},
    )
    assert cited.status_code == 200
    events = agent.get("/app/events").json()["events"]
    (note,) = [e for e in events if e["kind"] == "note"]
    assert note["payload"]["evidence"] == evidence_id
    assert [e["payload"]["evidence_id"] for e in events if e["kind"] == "evidence"] == [
        evidence_id
    ]

    # The Agent page shows it, and the operator marks it back.
    view = operator.get(f"/app/agent/evidence/{evidence_id}")
    assert view.status_code == 200
    assert view.json()["preview"]
    marked = {
        "text": "the left one",
        "evidence": {
            "marks": {"points": [{"x": 5, "y": 5}]},
            "answers": evidence_id,
            "verdict": "good",
        },
    }
    sent = operator.post("/app/agent/messages", json=marked)
    assert sent.status_code == 201, sent.text
    assert (
        operator.get(f"/app/agent/evidence/{sent.json()['evidence_id']}").status_code
        == 200
    )
    folder = host.state.host.experiment_path
    assert folder is not None and f"{evidence_id}.json" in json.dumps(
        sorted(p.name for p in Path(folder).rglob("*.json"))
    )


def test_the_level_gates_the_agent_and_stop_lowers_it(host, agent, operator):
    operator.put("/app/agent/level", json={"level": "off"})
    refused = agent.get("/app/events")
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "agent_off"

    operator.put("/app/agent/level", json={"level": "drive"})
    assert host.state.auth.level is AgentLevel.DRIVE
    stopped = operator.post("/app/agent/stop").json()
    assert stopped["level"] == "watch" and host.state.auth.stopped


def test_the_record_is_kept_on_disk(host, operator):
    operator.post("/app/agent/messages", json={"text": "kept"})
    host.state.host.close()
    path = host.state.host.folder / EVENTS_FILENAME
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["payload"]["text"] for r in lines if r["kind"] == "note"] == ["kept"]


def test_a_restarted_host_answers_after_an_id_from_its_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR)
    first = build_agent_host(auth, tmp_path / "host")
    operator = TestClient(first, headers=_as(OPERATOR))
    operator.post("/app/agent/messages", json={"text": "one"})
    operator.post("/app/agent/messages", json={"text": "two"})
    first.state.host.close()
    seen = first.state.host.buffer.events_since(0)["events"][0]["id"]

    again = build_agent_host(auth, tmp_path / "host")
    TestClient(again, headers=_as(OPERATOR)).post(
        "/app/agent/messages", json={"text": "three"}
    )
    answer = TestClient(again, headers=_as(AGENT)).get(f"/app/events?after={seen}")
    again.state.host.close()

    body = answer.json()
    assert body["resumed"] is True
    assert [e["payload"]["text"] for e in body["events"]] == ["two", "three"]


# --- the watch, on a real socket -------------------------------------------------


@pytest.fixture
def served(host):
    server = uvicorn.Server(
        uvicorn.Config(host, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "the host failed to start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


class _Seen(Exception):
    pass


def test_the_harness_reaches_it(served):
    client = Transport(served, AGENT, discover=False)
    try:
        assert _reachable(client)
    finally:
        client.close()


def test_the_watch_wakes_on_the_operator_s_message_level_and_stop(host, served):
    operator = TestClient(host, headers=_as(OPERATOR))
    client = Transport(served, AGENT, discover=False)
    requests = {"n": 0}
    real = client.request

    def request(method, path, **kwargs):
        # Once the watch holds its place, the operator acts.
        requests["n"] += 1
        if requests["n"] == 2:
            operator.post("/app/agent/messages", json={"text": "skip 03"})
            operator.put("/app/agent/level", json={"level": "assist"})
            operator.post("/app/agent/stop")
        return real(method, path, **kwargs)

    client.request = request
    lines = []

    def out(line):
        lines.append(str(line))
        if len(lines) == 3:
            raise _Seen

    try:
        watch_mod.watch(
            client, out=out, sleep_fn=lambda s: None, poll_s=0.3, report_reads=False
        )
    except _Seen:
        pass
    finally:
        client.close()

    assert lines[0].startswith("MESSAGE id=")
    assert "skip 03" in lines[0]
    assert "assist" in lines[1]
    assert lines[2].startswith("STOPPED by the operator: Stop the agent")


# --- an instrument the host follows ----------------------------------------------
# Another host stands in for the instrument's server: it answers the same
# events and Stop routes as the App Server.


def _serve(app):
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "the server failed to start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def _wait_for(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError("not seen in time")


@pytest.fixture
def linked(tmp_path, monkeypatch):
    """(the host, the instrument), the host following the instrument."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("fibsem.server.agent.host.RELAY_RETRY_S", 0.1)
    monkeypatch.setattr("fibsem.server.agent.host.RELAY_POLL_S", 1.0)
    instrument_auth = AuthConfig(token="its-agent", operator_token="its-operator")
    instrument = build_agent_host(instrument_auth, tmp_path / "instrument")
    server, thread, url = _serve(instrument)
    link = InstrumentLink(url, "its-operator", name="fib-1")
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR)
    host = build_agent_host(auth, tmp_path / "host", instrument=link)
    # The relay holds its place before the test acts.
    _wait_for(lambda: host.state.host.following)
    yield host, instrument
    host.state.host.close()
    server.should_exit = True
    thread.join(timeout=5)
    instrument.state.host.close()


def _kinds(host, kind):
    return [
        e for e in host.state.host.buffer.events_since(0)["events"] if e["kind"] == kind
    ]


def test_the_instrument_s_events_are_passed_on(linked):
    host, instrument = linked
    instrument.state.host.record("task_started", {"item_name": "01"}, actor="task")
    relayed = _wait_for(lambda: _kinds(host, "task_started"))
    assert relayed[0]["instrument"] == "fib-1"
    assert relayed[0]["actor"] == "task"
    assert relayed[0]["payload"] == {"item_name": "01"}


def test_stop_here_presses_the_instrument_s_stop(linked):
    host, instrument = linked
    instrument.state.auth.set_level(AgentLevel.DRIVE)
    TestClient(host, headers=_as(OPERATOR)).post("/app/agent/stop")

    assert instrument.state.auth.level is AgentLevel.WATCH
    assert instrument.state.auth.stopped
    # The agent hears the Stop once: the host's record of it, not the
    # instrument's coming back.
    _wait_for(lambda: _kinds(host, "action"))
    time.sleep(1.5)
    (stop,) = _kinds(host, "action")
    assert stop["actor"] == "operator"
    assert stop["payload"]["stops"] == "agent"
    assert "did not answer" not in stop["payload"]["reason"]


def test_stop_at_the_instrument_lowers_the_level_here(linked):
    host, instrument = linked
    host.state.auth.set_level(AgentLevel.DRIVE)
    TestClient(instrument, headers=_as("its-operator")).post("/app/agent/stop")

    _wait_for(lambda: _kinds(host, "action"))
    assert host.state.auth.level is AgentLevel.WATCH


def test_stop_holds_here_when_the_instrument_does_not_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    link = InstrumentLink("http://127.0.0.1:9", "x", name="fib-1")
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR, level=AgentLevel.DRIVE)
    host = build_agent_host(auth, tmp_path / "host", instrument=link)
    try:
        TestClient(host, headers=_as(OPERATOR)).post("/app/agent/stop")
        assert auth.level is AgentLevel.WATCH
        (stop,) = _kinds(host, "action")
        assert "fib-1 did not answer" in stop["payload"]["reason"]
    finally:
        host.state.host.close()


def test_the_agent_page_reads_the_host_s_record(host, operator):
    # The App Server's own operator routes, over the host.
    operator.post("/app/agent/messages", json={"text": "hello"})
    rows = operator.get("/app/agent/recent").json()["rows"]
    assert any("hello" in json.dumps(row) for row in rows)
    assert operator.get("/app/agent/connect").status_code == 200
    # Nothing is drawn: no request or suggestion waits here.
    assert operator.get("/app/agent/pictures/request").status_code == 404
    assert operator.get("/app/agent/customisation").status_code == 200


# --- run on its own: its own tokens, found and asked ----------------------------


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def on_its_own(tmp_path, monkeypatch):
    """fibsem-agent-host's main with no tokens set, its files in *tmp_path*."""
    from fibsem.client import pairing as client_pairing
    from fibsem.server import discovery
    from fibsem.server.agent import host as host_mod

    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    for name in (host_mod.HOST_TOKEN, host_mod.HOST_OPERATOR_TOKEN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(host_mod, "get_secret", lambda name: None)
    monkeypatch.setattr("fibsem.ai.agent.host_link.get_secret", lambda name: None)
    monkeypatch.setattr(host_mod, "load_operator_token", lambda: OPERATOR)
    monkeypatch.setattr(host_mod, "PAIRED_FILE", tmp_path / "paired.json")
    monkeypatch.setattr(discovery, "HOST_DISCOVERY_FILE", tmp_path / "agent-host.json")
    monkeypatch.setattr(discovery, "GATEWAY_FILE", tmp_path / "gateway.json")
    monkeypatch.setattr(client_pairing, "AGENT_DIR", tmp_path / "agent")
    return tmp_path / "agent-host.json"


def _serving(app, port):
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "the host failed to start"
        time.sleep(0.01)
    return server, thread


def test_on_its_own_it_sits_behind_the_gateway_which_allows_the_agent(
    on_its_own, tmp_path, monkeypatch
):
    """The host serves no pages and pairs nothing: the gateway passes calls to
    it, the agent asks there, the operator allows it there, and the host takes
    the token the gateway gave (one paired.json)."""
    import os

    from fibsem.ai.agent import host_link
    from fibsem.gateway.gateway import Gateway
    from fibsem.gateway.passthrough import PassThrough
    from fibsem.gateway.server import GatewayAuth, build_gateway_app
    from fibsem.server.agent import host as host_mod
    from tests.fixtures.app_server import serving

    port = _free_port()
    seen = {}

    def run(app, host, port, log_level):
        found = json.loads(on_its_own.read_text())
        seen["file"] = found
        server, thread = _serving(app, port)
        auth = GatewayAuth.from_disk(tmp_path / "op.json", tmp_path / "paired.json")
        gateway = Gateway(
            home=tmp_path / "home",
            app_registry=lambda: {},
            last=lambda: None,
            recent=lambda: [],
            new_folder=lambda: str(tmp_path),
            configurations=lambda: [],
            discovery=lambda: None,
        )
        passthrough = PassThrough(lambda: found["url"], is_connect=auth.is_connect)
        try:
            with serving(
                build_gateway_app(gateway, auth, passthrough=passthrough)
            ) as gw:
                (tmp_path / "gateway.json").write_text(
                    json.dumps(
                        {
                            "url": gw,
                            "pid": os.getpid(),
                            "connect_token": auth.connect_token,
                        }
                    )
                )
                # --host finds the gateway while a host runs behind it.
                assert host_link.host_url("") == gw
                # No pages and no pairing at the host itself.
                assert TestClient(app).get("/ui/agent").status_code == 404
                assert TestClient(app).post("/connect").status_code == 404
                operator = TestClient(
                    build_gateway_app(gateway, auth), headers=_as(auth.token)
                )
                got = {}
                asking = threading.Thread(
                    target=lambda: got.update(
                        token=host_link.host_token(gw, wait_s=20)
                    ),
                    daemon=True,
                )
                asking.start()
                deadline = time.monotonic() + 20
                while asking.is_alive() and time.monotonic() < deadline:
                    ask = operator.get("/gateway/connections").json()["asking"]
                    if ask:
                        operator.post(f"/gateway/connections/{ask['id']}/allow")
                    time.sleep(0.05)
                asking.join(timeout=5)
                token = got["token"]
                assert token not in (auth.connect_token, auth.token)
                # The host takes the gateway's token, directly and through it.
                assert (
                    TestClient(app, headers=_as(token)).get("/app/events").status_code
                    == 200
                )
                import httpx

                through = httpx.get(f"{gw}/app/events", headers=_as(token), timeout=10)
                assert through.status_code == 200
                # Kept: the next start (a resumed session's) doesn't ask again.
                assert host_link.host_token(gw, wait_s=0) == token
                seen["token"] = token
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    monkeypatch.setattr(uvicorn, "run", run)
    assert host_mod.main(["--folder", str(tmp_path / "host"), "--port", str(port)]) == 0
    assert "token" not in seen["file"] and "connect_token" not in seen["file"]
    assert seen["file"]["url"] == f"http://127.0.0.1:{port}"
    assert not on_its_own.exists()
    assert seen["token"]


def test_a_second_host_on_the_pc_is_refused(on_its_own, tmp_path, monkeypatch):
    import os

    from fibsem.server.agent import host as host_mod

    on_its_own.write_text(
        json.dumps(
            {"url": "http://127.0.0.1:1", "connect_token": "x", "pid": os.getpid()}
        )
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("served"))
    assert host_mod.main(["--folder", str(tmp_path / "host")]) == 1


def test_set_tokens_write_no_discovery_file(on_its_own, tmp_path, monkeypatch):
    from fibsem.server.agent import host as host_mod

    secrets = {host_mod.HOST_TOKEN: AGENT, host_mod.HOST_OPERATOR_TOKEN: OPERATOR}
    monkeypatch.setattr(host_mod, "get_secret", secrets.get)
    seen = {}

    def run(app, **kwargs):
        seen["file"] = on_its_own.exists()
        seen["auth"] = app.state.auth

    monkeypatch.setattr(uvicorn, "run", run)
    assert host_mod.main(["--folder", str(tmp_path / "host")]) == 0
    assert seen["file"] is False
    assert seen["auth"].token == AGENT and seen["auth"].pairing is None


def test_only_one_set_token_is_refused(on_its_own, tmp_path, monkeypatch):
    from fibsem.server.agent import host as host_mod

    monkeypatch.setattr(host_mod, "get_secret", {host_mod.HOST_TOKEN: AGENT}.get)
    assert host_mod.main(["--folder", str(tmp_path / "host")]) == 1


def test_the_agent_says_when_no_host_runs_on_the_pc(on_its_own, capsys):
    from fibsem.ai.agent import harness

    assert harness.main(["--host"]) == 1
    assert "no Agent Server is running" in capsys.readouterr().err


def test_a_data_folder_is_served_with_the_image_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    data = tmp_path / "data"
    (data / "run").mkdir(parents=True)
    ramp = (Image.new("L", (64, 32)), [(x * 4) for x in range(64)] * 32)
    ramp[0].putdata(ramp[1])
    ramp[0].save(data / "run" / "ramp.png")
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR)
    app = build_agent_host(auth, tmp_path / "host", data=data)
    try:
        agent = TestClient(app, headers=_as(AGENT))
        routers = agent.get("/capabilities").json()["routers"]
        assert routers["analysis"] and routers["files"] and not routers["app"]

        listing = agent.get("/app/experiment/files", params={"dir": "run"}).json()
        assert [e["name"] for e in listing["entries"]] == ["ramp.png"]
        crop = agent.get(
            "/app/experiment/files/run/ramp.png", params={"box": "0,0,16,16"}
        )
        assert crop.status_code == 200, crop.text
        histogram = agent.post(
            "/app/experiment/analysis/histogram", json={"path": "run/ramp.png"}
        )
        assert histogram.status_code == 200, histogram.text
        assert (data / "run" / "drawn" / "ramp-histogram.png").is_file()

        past = agent.get("/app/experiment/files", params={"experiment": "/elsewhere"})
        assert past.status_code == 404
        outside = agent.get("/app/experiment/files/../host/events.jsonl")
        assert outside.status_code in (403, 404)
        assert TestClient(app).get("/app/experiment/files").status_code == 401
    finally:
        app.state.host.close()


def test_without_a_data_folder_it_serves_no_image_tools(agent):
    routers = agent.get("/capabilities").json()["routers"]
    assert not routers["analysis"] and not routers["files"]
    assert agent.get("/app/experiment/files").status_code == 404


def test_the_playbooks_skills_and_subagents_are_served_with_no_app(agent, operator):
    # The Knowledge window and the gateway's Customise, with no app running.
    found = operator.get("/app/library/playbooks")
    assert found.status_code == 200, found.text
    assert "autolamella" in {item["name"] for item in found.json()["items"]}
    assert agent.get("/app/library/options").status_code == 200
    # Editing is the operator's.
    r = agent.put("/app/library/playbooks/autolamella", json={"text": "rules: []\n"})
    assert r.status_code in (401, 403)
