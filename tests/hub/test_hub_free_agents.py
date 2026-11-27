"""An agent with no instrument on the hub PC: its start, its own Agent
Server under the Hub, the limit, its end and resume, and being taken back
after the Hub restarted."""

import json
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent.manager import RECORD, AgentSessionManager
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub import free_agents
from fibsem.hub import sessions as hub_sessions
from fibsem.hub.free_agents import AT, FreeAgents
from fibsem.hub.history import HubHistory
from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth

OPERATOR = {"Authorization": "Bearer operator-token"}
HUB_URL = "http://127.0.0.1:8020"

# Stands in for fibsem-agent: writes its session's record, says what it was
# started with (its arguments, its token), and runs until its stop file.
AGENT = """
import glob, json, os, sys, time
from fibsem.util.process import process_started
args = sys.argv[1:]
flag = "--session-id" if "--session-id" in args else "--resume"
session_id = args[args.index(flag) + 1]
folder = glob.glob(os.path.join(os.environ["FIBSEM_AGENT_SESSIONS"], "*-" + session_id))[0]
record = {"session_id": session_id, "goal": "Read the papers", "playbook": "plain",
          "runs": [{"pid": os.getpid(), "pid_started": process_started(os.getpid())}]}
with open(os.path.join(folder, "session.json"), "w") as out:
    json.dump(record, out)
with open(os.environ["SEEN"], "a") as out:
    out.write(json.dumps({"args": args, "token": os.environ.get("FIBSEM_HOST_TOKEN"),
                          "key": os.environ.get("ANTHROPIC_API_KEY")}) + "\\n")
while not os.path.exists(os.path.join(folder, "stop")):
    time.sleep(0.02)
record["runs"][-1].update(ended=time.time(), exit_code=0, reason="stopped")
with open(os.path.join(folder, "session.json"), "w") as out:
    json.dump(record, out)
"""


def _wait(until, what, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline, f"never: {what}"
        time.sleep(0.02)


@pytest.fixture
def made(tmp_path, monkeypatch):
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    seen = tmp_path / "seen.jsonl"
    monkeypatch.setenv("SEEN", str(seen))
    token_file = tmp_path / "operator-token.json"
    token_file.write_text('{"token": "operator-token"}')
    auth = hub_auth(token_file, tmp_path / "paired.json")
    history = HubHistory(tmp_path / "history.jsonl")
    most = [5]
    root = tmp_path / "free-sessions"

    def build():
        return FreeAgents(
            auth,
            lambda: HUB_URL,
            root=root,
            api_key=lambda: "key",
            history=history,
            limit=lambda: most[0],
            manager=AgentSessionManager(
                root=root,
                command=[sys.executable, str(script)],
                end_wait_s=10.0,
                env_for=lambda session_id: made.agents._env(session_id),
            ),
        )

    def runs():
        if not seen.is_file():
            return []
        return [json.loads(line) for line in seen.read_text().splitlines()]

    made = SimpleNamespace(
        auth=auth, history=history, most=most, root=root, build=build, runs=runs
    )
    made.agents = build()
    yield made
    made.agents.stop()


def _client(made):
    hub = Hub(file=made.root.parent / "hub.yaml", tokens=made.root.parent / "hub")
    return TestClient(build_hub_app(hub, made.auth, free=made.agents))


def _started(made, n=1):
    _wait(lambda: len(made.runs()) >= n, "the agent started")
    return made.runs()[n - 1]


def test_start_runs_a_plain_agent_against_its_own_agent_server(made):
    row = made.agents.start("Read the papers", "pat", 20)
    assert row["state"] == "running" and row["who"] == "pat"
    assert row["goal"] == "Read the papers" and row["budget_usd"] == 20
    run = _started(made)
    args = run["args"]
    assert args[args.index("--playbook") + 1] == "plain"
    assert args[args.index("--host") + 1] == f"{HUB_URL}{AT}/{row['session_id']}"
    assert args[args.index("--max-budget-usd") + 1] == "20"
    assert args[args.index("--goal") + 1] == "Read the papers"
    # No instrument: no server address, no level.
    assert "--url" not in args
    assert run["key"] == "key" and run["token"]
    said = [e["what"] for e in made.history.read(10)["events"]]
    assert "Started an agent with no instrument on this hub PC" in said


def test_the_agents_token_is_never_written_down_or_answered(made):
    row = made.agents.start(None, "pat")
    token = _started(made)["token"]
    assert token not in json.dumps(row) + json.dumps(made.agents.state())
    for path in made.root.rglob("*"):
        if path.is_file():
            assert token not in path.read_text(errors="replace"), path


def test_each_session_has_its_own_agent_server_under_the_hub(made):
    client = _client(made)
    one = made.agents.start("One", "pat")["session_id"]
    two = made.agents.start("Two", "pat")["session_id"]
    tokens = {
        run["args"][run["args"].index("--session-id") + 1]: run["token"]
        for run in (_started(made, 1), _started(made, 2))
    }
    mine = {"Authorization": f"Bearer {tokens[one]}"}
    answer = client.get(f"{AT}/{one}/capabilities", headers=mine)
    assert answer.status_code == 200
    assert answer.json()["host"] == "agent" and answer.json()["instrument"] is None
    # The Hub's operator reads it too; nobody else, and never another
    # session's agent.
    assert client.get(f"{AT}/{one}/capabilities", headers=OPERATOR).status_code == 200
    assert client.get(f"{AT}/{one}/capabilities").status_code == 401
    _, laptop = made.auth.pairing.tokens.add("laptop", role="operator")
    paired = {"Authorization": f"Bearer {laptop}"}
    assert client.get(f"{AT}/{one}/capabilities", headers=paired).status_code == 200
    assert client.get(f"{AT}/{two}/capabilities", headers=mine).status_code == 401
    # Its plan and transcript go in the session's own folder on this hub PC.
    files = made.agents.server(one).state.host.files.folder(one)
    assert files == made.agents._manager.folder(one) / "agent"
    missing = client.get(f"{AT}/no-such-session/capabilities", headers=OPERATOR)
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_type"] == "no_session"


def test_a_message_from_the_hubs_page_reaches_the_sessions_agent(made):
    client = _client(made)
    one = made.agents.start("One", "pat")["session_id"]
    two = made.agents.start("Two", "pat")["session_id"]
    agent = {"Authorization": f"Bearer {_started(made, 1)['token']}"}
    _started(made, 2)
    sent = client.post(
        f"{AT}/{one}/app/agent/messages",
        json={"text": "Start with 2024"},
        headers=OPERATOR,
    )
    assert sent.status_code == 201
    events = client.get(f"{AT}/{one}/app/events?since=0", headers=agent).json()
    assert "Start with 2024" in json.dumps(events)
    other = client.get(f"{AT}/{two}/app/events?since=0", headers=OPERATOR).json()
    assert "Start with 2024" not in json.dumps(other)
    # Its chat is kept in the session's own folder on this hub PC.
    folder = made.agents._manager.folder(one)
    _wait(
        lambda: "Start with 2024" in (folder / "host" / "events.jsonl").read_text(),
        "the record was written",
    )


def test_no_more_run_at_once_than_the_limit(made):
    made.most[0] = 2
    first = made.agents.start("One", "pat")
    made.agents.start("Two", "pat")
    with pytest.raises(GatewayRefused) as refused:
        made.agents.start("Three", "pat")
    assert refused.value.error_type == "too_many" and "limit is 2" in str(refused.value)
    _started(made, 2)
    made.agents.end(first["session_id"], "pat")
    assert made.agents.start("Three", "pat")["state"] == "running"


def test_no_api_key_refuses(made):
    made.agents._api_key = lambda: None
    with pytest.raises(GatewayRefused) as refused:
        made.agents.start("One", "pat")
    assert refused.value.error_type == "no_api_key"
    assert made.agents.state()["agents"] == []


def test_a_goal_too_long_is_refused(made):
    with pytest.raises(GatewayRefused) as refused:
        made.agents.start("x" * 5000, "pat")
    assert refused.value.error_type == "goal_too_long"


def test_end_stops_the_agent_and_its_page_still_reads(made):
    client = _client(made)
    row = made.agents.start("One", "pat")
    token = _started(made)["token"]
    session_id = row["session_id"]
    ended = made.agents.end(session_id, "pat")
    assert ended["state"] == "ended" and ended["why"] == "ended by pat"
    assert made.agents.running() == []
    # The token it had opens nothing now; the operator still reads the page.
    old = {"Authorization": f"Bearer {token}"}
    assert client.get(f"{AT}/{session_id}/capabilities", headers=old).status_code == 401
    assert (
        client.get(f"{AT}/{session_id}/capabilities", headers=OPERATOR).status_code
        == 200
    )
    with pytest.raises(GatewayRefused) as refused:
        made.agents.end(session_id, "pat")
    assert refused.value.error_type == "no_agent"


def test_an_agent_that_ends_by_itself_is_noticed(made):
    row = made.agents.start("One", "pat")
    _started(made)
    folder = made.agents._manager.folder(row["session_id"])
    (folder / "stop").touch()
    _wait(
        lambda: made.agents.poll() or made.agents.running() == [],
        "the agent's end was noticed",
    )
    assert made.agents.state()["agents"][0]["state"] == "ended"


def test_resume_carries_the_session_on_with_a_new_token(made):
    client = _client(made)
    session_id = made.agents.start("One", "pat")["session_id"]
    first = _started(made)["token"]
    client.post(
        f"{AT}/{session_id}/app/agent/messages",
        json={"text": "Before"},
        headers=OPERATOR,
    )
    made.agents.end(session_id, "pat")
    row = made.agents.resume(session_id, "sam", 5)
    assert row["state"] == "running" and row["resumed"] and row["who"] == "sam"
    run = _started(made, 2)
    assert run["args"][run["args"].index("--resume") + 1] == session_id
    assert run["token"] and run["token"] != first
    assert len(list(made.root.glob(f"*-{session_id}"))) == 1
    # What was said before is still on its page.
    chat = client.get(f"{AT}/{session_id}/app/agent/recent", headers=OPERATOR)
    assert "Before" in chat.text
    with pytest.raises(GatewayRefused) as refused:
        made.agents.resume(session_id, "sam")
    assert refused.value.error_type == "agent_running"
    with pytest.raises(GatewayRefused) as refused:
        made.agents.resume("no-such-session", "sam")
    assert refused.value.error_type == "no_session"


def test_a_start_carries_what_it_was_given_and_a_resume_is_checked_again(made):
    given = (["--files", "--read", "/data/papers"], {"folder": "/data/papers"})
    row = made.agents.start("Read the papers", "pat", given=given)
    session_id = row["session_id"]
    args = _started(made)["args"]
    assert args[args.index("--read") + 1] == "/data/papers" and "--files" in args
    listed = hub_sessions.hub_pc_sessions(made.root, 10, free=True)
    assert listed[0]["settings"] == {"folder": "/data/papers"}
    made.agents.end(session_id, "pat")
    asked = []

    def launch(settings):
        asked.append(settings)
        return ["--files", "--read", "/data/papers"], {"folder": "/data/papers"}

    made.agents.resume(session_id, "pat", launch=launch)
    # Checked from the session's record, as the settings are now.
    assert asked == [{"folder": "/data/papers"}]
    args = _started(made, 2)["args"]
    assert args[args.index("--read") + 1] == "/data/papers"
    made.agents.end(session_id, "pat")

    def refuses(settings):
        raise GatewayRefused(403, "folder_not_listed", "Not listed any more.")

    with pytest.raises(GatewayRefused) as refused:
        made.agents.resume(session_id, "pat", launch=refuses)
    assert refused.value.error_type == "folder_not_listed"
    assert len(made.runs()) == 2


def test_an_agent_left_by_an_earlier_hub_carries_on_with_a_new_token(made):
    session_id = made.agents.start("One", "pat")["session_id"]
    first = _started(made)["token"]
    # The Hub is gone without ending its agents; the next one takes over.
    old = made.agents
    old._held.clear()
    made.agents = made.build()
    assert made.agents.running() == [session_id]
    _wait(lambda: made.agents.poll() or len(made.runs()) >= 2, "it started again")
    run = made.runs()[1]
    assert run["args"][run["args"].index("--resume") + 1] == session_id
    assert "--carry-on" in run["args"]
    assert run["token"] and run["token"] != first
    assert made.agents.state()["agents"][0]["state"] == "running"
    # The earlier Hub's own handle on the process it started, which stopped.
    old._manager.agent(session_id).process.wait(10)
    old.stop()


def test_stopping_the_hub_ends_its_agents(made):
    made.agents.start("One", "pat")
    made.agents.start("Two", "pat")
    _started(made, 2)
    made.agents.stop()
    assert made.agents.running() == []
    assert made.agents.server(made.agents.state()["agents"][0]["session_id"]) is None
    for record in made.root.glob(f"*/{RECORD}"):
        assert "ended" in json.loads(record.read_text())["runs"][-1]


def test_the_routes_start_list_resume_and_end_as_the_operator(made):
    client = _client(made)
    body = {"goal": "Read the papers", "budget_usd": "20"}
    assert client.post("/hub/free-agents/start", json=body).status_code == 401
    started = client.post("/hub/free-agents/start", json=body, headers=OPERATOR)
    assert started.status_code == 200 and started.json()["budget_usd"] == 20
    token = _started(made)["token"]
    assert token not in started.text
    session_id = started.json()["session_id"]
    listed = client.get("/hub/free-agents", headers=OPERATOR).json()
    assert listed["limit"] == 5 and listed["at_this_pc"] is True
    assert [a["session_id"] for a in listed["agents"]] == [session_id]
    assert token not in json.dumps(listed)
    ended = client.post(f"/hub/free-agents/{session_id}/end", headers=OPERATOR)
    assert ended.json()["state"] == "ended"
    again = client.post(f"/hub/free-agents/{session_id}/end", headers=OPERATOR)
    assert again.status_code == 409
    resumed = client.post(
        "/hub/free-agents/resume", json={"session_id": session_id}, headers=OPERATOR
    )
    assert resumed.status_code == 200 and resumed.json()["resumed"] is True


def test_the_sessions_list_names_them_with_no_instrument(made):
    session_id = made.agents.start("Read the papers", "pat")["session_id"]
    _started(made)
    hub = Hub(file=made.root.parent / "hub.yaml", tokens=made.root.parent / "hub")
    listed = hub_sessions.list_sessions(
        hub, made.root.parent / "sessions", free_root=made.root
    )
    (row,) = listed["sessions"]
    assert row["session_id"] == session_id and row["instrument"] is None
    assert row["runs_on"] == "hub" and row["who"] == "pat"
    assert row["goal"] == "Read the papers"


def test_the_limit_is_changed_only_at_this_pc(made, monkeypatch, tmp_path):
    from fibsem import config
    from fibsem.server.pairing import OPERATOR as OPERATOR_ROLE

    kept = {"prefs": config.UserPreferences()}
    monkeypatch.setattr(config, "load_user_preferences", lambda: kept["prefs"])
    monkeypatch.setattr(
        config,
        "update_user_preferences",
        lambda change: change(kept["prefs"]),
    )
    made.agents._limit = free_agents.limit
    client = _client(made)
    assert client.get("/hub/free-agents", headers=OPERATOR).json()["limit"] == 5
    saved = client.put("/hub/free-agents/limit", json={"limit": 2}, headers=OPERATOR)
    assert saved.json() == {"limit": 2}
    assert client.get("/hub/free-agents", headers=OPERATOR).json()["limit"] == 2
    for bad in (-1, 21, "3", True, None):
        refused = client.put(
            "/hub/free-agents/limit", json={"limit": bad}, headers=OPERATOR
        )
        assert refused.status_code == 422, bad
    # A browser paired with the Hub reads it and may not change it.
    _, paired = made.auth.pairing.tokens.add("laptop", role=OPERATOR_ROLE)
    browser = {"Authorization": f"Bearer {paired}"}
    shown = client.get("/hub/free-agents", headers=browser).json()
    assert shown["limit"] == 2 and shown["at_this_pc"] is False
    refused = client.put("/hub/free-agents/limit", json={"limit": 9}, headers=browser)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "at_this_pc_only"


def test_the_shipped_plain_playbook_has_no_instrument_and_no_servers():
    from fibsem.ai.agent import harness, playbooks

    checkout = harness.find_checkout(None)
    plain = playbooks.load(free_agents.PLAYBOOK, checkout, harness.EFFORTS)
    assert plain.instrument == "none" and plain.mcp == [] and plain.skills == []
    assert harness.mcp_names(plain, {}, host=True) == []
    # Never offered for an instrument's agent: its row says it has none.
    rows = {row["name"]: row for row in harness._playbook_rows(checkout)}
    assert rows["plain"]["instrument"] == "none"
    assert rows["fibsem-agent"]["instrument"] == "fibsem"
