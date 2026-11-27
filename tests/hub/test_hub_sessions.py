"""The Hub's sessions list and history, resuming a session that ended on the
hub PC, and carrying an agent on after the Hub was restarted."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent.manager import MANAGED, RECORD, folder_name, write_json
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub import sessions as hub_sessions
from fibsem.hub.agents import HubAgents
from fibsem.hub.history import AGENT, HubHistory
from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth

URL = "https://scope-1:8010"
OPERATOR = {"Authorization": "Bearer operator-token"}


class Gateway:
    """An instrument's gateway, as the Hub asks it: the slot routes, its
    sessions, its app."""

    def __init__(self):
        self.calls = []
        self.down = False
        self.refuse = {}
        self.sessions = []

    def send(self, url, method, route, token, body, timeout):
        self.calls.append((route, dict(body or {})))
        if self.down:
            raise OSError("connection refused")
        name = route.split("?")[0].rsplit("/", 1)[-1]
        if name in self.refuse:
            reason = self.refuse[name]
            detail = {"error_type": reason, "message": f"Refused: {reason}."}
            return 409, {"detail": detail}
        if name == "sessions":
            return 200, {"sessions": self.sessions, "more": False}
        if name == "slot":
            token = "new-token" if body.get("rejoin") else "agent-token"
            return 200, {"token": token, "lease_s": 30, "grace_s": 120}
        return 200, {"lease_s": 30, "state": "running"}


class Manager:
    """The session manager, with real folders under *root*."""

    def __init__(self, root):
        self.root = root
        self.agents = {}
        self.resumed = []
        self.parked = []

    def folder(self, session_id):
        found = [p for p in self.root.glob(f"*{session_id}") if p.is_dir()]
        return found[0] if found and session_id else None

    def agent(self, name):
        return self.agents.get(name)

    def resume(self, name, session_id, args, env=None, about=None):
        self.resumed.append((name, session_id, args, env, about))
        self.agents[name] = SimpleNamespace(
            state="running", ended=None, session_id=session_id
        )

    def park(self, name):
        self.parked.append(name)
        self.agents[name].state = "waiting"
        return self.agents[name]

    def unpark(self, name):
        self.agents[name].state = "running"

    def end(self, name):
        if name in self.agents:
            self.agents[name].state = "ended"

    def poll(self):
        pass


def _session(root, session_id, instrument, started, spent=None, ended="ended by pat"):
    """A session folder as the harness and the manager leave it."""
    folder = root / folder_name(session_id, started)
    run = {"started": f"2026-11-2{int(started) % 10}T10:00:00+00:00"}
    if ended:
        run.update(ended=run["started"], reason=ended, exit_code=0, resumable=False)
    if spent is not None:
        run["spent_usd"] = spent
    write_json(
        folder / RECORD,
        {
            "session_id": session_id,
            "playbook": {"name": "fibsem-agent"},
            "goal": "Mill",
            "model": "a-model",
            "runs": [run],
        },
    )
    managed = {
        "microscope": instrument,
        "args": ["--playbook", "fibsem-agent", "--url", URL],
        "about": {"who": "pat", "level": "suggest", "budget_usd": 20},
    }
    if ended:
        managed["ended"] = ended
    write_json(folder / MANAGED, managed)
    return folder


@pytest.fixture
def made(tmp_path):
    gateway, now = Gateway(), [1000.0]
    root = tmp_path / "sessions"
    root.mkdir()
    hub = Hub(
        file=tmp_path / "hub.yaml",
        tokens=tmp_path / "hub",
        fetch=lambda *a: (200, {}),
        pair=lambda url, code, name: ("hub-token", "hub"),
        send=gateway.send,
    )
    hub.add("Scope 1", url=URL)
    hub.pair("Scope 1", "123456")
    history = HubHistory(tmp_path / "history.jsonl", clock=lambda: now[0])
    manager = Manager(root)

    def agents():
        return HubAgents(
            hub,
            root=root,
            api_key=lambda: "key",
            clock=lambda: now[0],
            manager=manager,
            history=history,
        )

    return SimpleNamespace(
        hub=hub,
        gateway=gateway,
        manager=manager,
        history=history,
        root=root,
        now=now,
        agents=agents,
        tmp=tmp_path,
    )


def _routes(made):
    return [route for route, _ in made.gateway.calls]


# -- resume ---------------------------------------------------------------------


def test_resume_takes_the_slot_for_the_same_session_and_starts_it_again(made):
    _session(made.root, "s-1", "Scope 1", 1)
    agents = made.agents()
    state = agents.resume("Scope 1", "s-1", "assist", "pat", budget_usd=5)
    assert state["session_id"] == "s-1" and state["level"] == "assist"
    route, body = made.gateway.calls[-1]
    assert route == "/gateway/agent/slot"
    assert body["session_id"] == "s-1" and body["level"] == "assist"
    assert body["playbook"] == "fibsem-agent" and "rejoin" not in body
    name, session_id, args, env, about = made.manager.resumed[0]
    assert args == [
        "--playbook",
        "fibsem-agent",
        "--url",
        URL,
        "--model",
        "a-model",
        "--max-budget-usd",
        "5",
    ]
    assert env == {"FIBSEM_SERVER_TOKEN": "agent-token", "ANTHROPIC_API_KEY": "key"}
    assert about["resumed"] is True and about["who"] == "pat"
    (line,) = made.history.read()["events"]
    assert line["kind"] == AGENT and line["instrument"] == "Scope 1"
    assert line["what"].startswith("Resumed a session") and line["who"] == "pat"


def test_resume_is_refused_for_no_session_another_instrument_or_a_running_agent(made):
    _session(made.root, "s-1", "Scope 2", 1)
    _session(made.root, "s-2", "Scope 1", 2)
    agents = made.agents()
    for session, reason in (("nope", "no_session"), ("s-1", "other_instrument")):
        with pytest.raises(GatewayRefused) as refused:
            agents.resume("Scope 1", session, "watch", "pat")
        assert refused.value.error_type == reason
    assert made.gateway.calls == []
    agents.resume("Scope 1", "s-2", "watch", "pat")
    with pytest.raises(GatewayRefused) as refused:
        agents.resume("Scope 1", "s-2", "watch", "pat")
    assert refused.value.error_type == "agent_running"


def test_a_gateway_that_refuses_the_slot_resumes_nothing(made):
    _session(made.root, "s-1", "Scope 1", 1)
    made.gateway.refuse["slot"] = "busy"
    with pytest.raises(GatewayRefused):
        made.agents().resume("Scope 1", "s-1", "watch", "pat")
    assert made.manager.resumed == []


# -- after the Hub was restarted ------------------------------------------------


def _left_running(made, session_id="s-1"):
    """An agent an earlier Hub started and left running."""
    folder = _session(made.root, session_id, "Scope 1", 1, ended=None)
    made.manager.agents["Scope 1"] = SimpleNamespace(
        state="running", ended=None, session_id=session_id, folder=folder
    )


def test_a_new_hub_takes_its_agent_back_and_renews_its_slot(made):
    _left_running(made)
    agents = made.agents()
    assert agents.state()["Scope 1"]["state"] == "away"
    agents.poll()
    assert _routes(made) == ["/gateway/agent/slot/renew"]
    assert made.gateway.calls[0][1] == {"session_id": "s-1"}
    state = agents.state()["Scope 1"]
    assert state["state"] == "running" and state["who"] == "pat"
    assert made.manager.parked == [], "it never stopped"
    assert "Took the agent back" in made.history.read()["events"][0]["what"]


def test_a_slot_freed_meanwhile_is_asked_for_again_as_the_same_paired_hub(made):
    _left_running(made)
    agents = made.agents()
    made.gateway.refuse["renew"] = "lost_slot"
    agents.poll()
    made.gateway.refuse.clear()
    agents.poll()
    route, body = made.gateway.calls[-1]
    assert route == "/gateway/agent/slot"
    # No token: that Hub kept it in memory only.
    assert body == {"session_id": "s-1", "rejoin": True, "token": None}
    # The agent is stopped and carries on with the new token.
    assert made.manager.parked == ["Scope 1"]
    assert made.manager.agents["Scope 1"].env["FIBSEM_SERVER_TOKEN"] == "new-token"
    assert agents.state()["Scope 1"]["state"] == "running"


def test_a_gateway_that_refuses_the_restarted_hub_ends_the_agent(made):
    _left_running(made)
    agents = made.agents()
    made.gateway.refuse["renew"] = "lost_slot"
    agents.poll()
    made.gateway.refuse["slot"] = "level_changed"
    agents.poll()
    state = agents.state()["Scope 1"]
    assert state["state"] == "ended" and "level_changed" in state["why"]


def test_an_agent_that_stopped_while_no_hub_ran_waits_for_a_person(made):
    _left_running(made)
    made.manager.agents["Scope 1"].state = "waiting"
    agents = made.agents()
    agents.poll()
    state = agents.state()["Scope 1"]
    assert state["state"] == "ended" and "resume it in Sessions" in state["why"]


def test_a_hub_stopped_on_purpose_leaves_nothing_to_take_back(made):
    _session(made.root, "s-1", "Scope 1", 1)
    agents = made.agents()
    agents.resume("Scope 1", "s-1", "watch", "pat")
    agents.stop()
    assert made.manager.agents["Scope 1"].state == "ended"
    again = made.agents()
    assert again.state() == {}
    again.poll()
    assert "/gateway/agent/slot/renew" not in _routes(made)


# -- the sessions list ------------------------------------------------------------


def test_the_list_has_this_pcs_sessions_and_each_instruments(made):
    _session(made.root, "s-1", "Scope 1", 1, spent=2.5)
    _session(made.root, "s-2", "Scope 1", 2, spent=1.25)
    made.gateway.sessions = [
        {
            "session_id": "g-1",
            "started": "2026-11-23T09:00:00+00:00",
            "spent_usd": 4.0,
            "status": "ended",
        }
    ]
    found = hub_sessions.list_sessions(made.hub, made.root)
    rows = {row["session_id"]: row for row in found["sessions"]}
    assert rows["s-1"]["runs_on"] == "hub" and rows["g-1"]["runs_on"] == "pc"
    assert rows["s-1"]["who"] == "pat" and rows["s-1"]["status"] == "ended"
    assert rows["s-1"]["last_level"] == "suggest"
    assert all(row["instrument"] == "Scope 1" for row in found["sessions"])
    assert [row["session_id"] for row in found["sessions"]] == ["g-1", "s-2", "s-1"]
    assert found["totals"]["spent_usd"] == 7.75
    assert found["totals"]["by_instrument"] == {"Scope 1": 7.75}
    assert found["totals"]["by_day"][0] == {"day": "2026-11-23", "spent_usd": 4.0}
    assert found["unread"] == [] and found["more"] is False


def test_an_instrument_that_does_not_answer_is_named_not_waited_for(made):
    _session(made.root, "s-1", "Scope 1", 1, spent=2.5)
    made.gateway.down = True
    found = hub_sessions.list_sessions(made.hub, made.root)
    assert [row["session_id"] for row in found["sessions"]] == ["s-1"]
    (unread,) = found["unread"]
    assert unread["instrument"] == "Scope 1" and "didn't answer" in unread["why"]


# -- the history and the routes -----------------------------------------------------


def test_the_history_is_read_newest_first_and_says_when_there_is_more(made):
    for n in range(5):
        made.now[0] += 1
        made.history.write(AGENT, f"line {n}", "pat", "Scope 1")
    found = made.history.read(3)
    assert [e["what"] for e in found["events"]] == ["line 4", "line 3", "line 2"]
    assert found["more"] is True and made.history.read(10)["more"] is False
    (made.tmp / "history.jsonl").write_text("not json\n")
    assert made.history.read() == {"events": [], "more": False}


@pytest.fixture
def page(made):
    token_file = made.tmp / "operator-token.json"
    token_file.write_text(json.dumps({"token": "operator-token"}))
    auth = hub_auth(token_file, made.tmp / "hub-paired.json")
    return TestClient(
        build_hub_app(made.hub, auth, agents=made.agents(), history=made.history)
    )


def test_the_routes_list_resume_and_keep_a_history(made, page):
    _session(made.root, "s-1", "Scope 1", 1, spent=2.5)
    assert page.get("/hub/sessions").status_code == 401
    assert page.get("/hub/history").status_code == 401
    listed = page.get("/hub/sessions", headers=OPERATOR).json()
    assert listed["sessions"][0]["session_id"] == "s-1"
    at = "/hub/instruments/Scope 1"
    body = {"session_id": "s-1", "level": "watch"}
    assert page.post(f"{at}/hub-agent/resume", json=body).status_code == 401
    resumed = page.post(f"{at}/hub-agent/resume", json=body, headers=OPERATOR)
    assert resumed.status_code == 200 and resumed.json()["session_id"] == "s-1"
    # Asked of the instrument through the Hub: done, and refused.
    assert page.post(f"{at}/gateway/stop", json={}, headers=OPERATOR).status_code == 200
    made.gateway.refuse["start"] = "no_grant"
    start = {"app": "autolamella", "window": False}
    refused = page.post(f"{at}/gateway/start", json=start, headers=OPERATOR)
    assert refused.status_code == 409
    page.post(f"{at}/hub-agent/end", headers=OPERATOR)
    page.post(
        "/hub/instruments", json={"name": "Scope 2", "url": URL}, headers=OPERATOR
    )
    page.delete("/hub/instruments/Scope 2", headers=OPERATOR)
    events = page.get("/hub/history", headers=OPERATOR).json()["events"]
    said = [(e["kind"], e["what"], e["ok"]) for e in reversed(events)]
    assert said == [
        ("agent", "Resumed a session on this hub PC (fibsem-agent, watch)", True),
        ("app", "Stopped the app", True),
        (
            "app",
            "Started autolamella (no window): refused. Refused: no_grant.",
            False,
        ),
        ("agent", "The agent on this hub PC ended: ended by operator", True),
        ("instrument", "Added the instrument", True),
        ("instrument", "Removed the instrument", True),
    ]
    assert {e["who"] for e in events} == {"operator"}
