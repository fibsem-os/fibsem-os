"""The agent host: the agent's routes on their own, with no microscope or app.

The watch, the harness's checks and the Agent page's calls are made against it
as they are against the App Server, on a real socket where the watch runs."""

import json
import subprocess
import sys
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.harness import _reachable
from fibsem.ai.agent.levels import AgentLevel
from fibsem.client.transport import Transport
from fibsem.server.agent.host import EVENTS_FILENAME, build_agent_host
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


def test_a_message_here_is_text_only(operator, agent):
    marked = {"text": "this", "evidence": {"image": "a.png"}}
    assert operator.post("/app/agent/messages", json=marked).status_code == 422
    note = {"text": "x", "item_name": "01"}
    assert agent.post("/app/experiment/notes", json=note).status_code == 422


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
