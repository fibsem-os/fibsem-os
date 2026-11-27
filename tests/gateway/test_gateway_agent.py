"""The gateway's agent: Start's chain and its refusals, End, stopping the app
ending the agent, the level set again on a resume, and a restarted gateway
taking the agent back, with the stand-in ``fibsem-agent``
(tests/agent/stand_in.py) in real processes and a fake App Server."""

import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    WORK,
    SessionFolder,
    folders,
    read_json,
)
from fibsem.gateway.agent import AGENT_NAME, API_KEY, SESSIONS_SHOWN, GatewayAgent
from fibsem.gateway.gateway import OFF, RUNNING, GatewayRefused, GatewayState
from fibsem.server.pairing import AGENT, PairedTokens

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "agent" / "stand_in.py"
URL = "http://127.0.0.1:8001"
HOST_URL = "http://127.0.0.1:8030"
DEFAULT = {"playbook": {"name": "default"}, "mode": "none", "model": None}


class FakeGateway:
    def __init__(self, mode=RUNNING):
        self.mode = mode
        self.before_stop = []
        self.watchers = []
        self.stopped = []
        self.access = []

    def record_access(self, who, via, result):
        self.access.append((who, via, result))

    def state(self):
        return GatewayState(
            mode=self.mode, version="test", experiment=getattr(self, "experiment", None)
        )

    def stop(self, who, via, force=False, window=False):
        for hook in self.before_stop:
            hook()
        self.stopped.append((who, via, force))
        self.mode = OFF


class Host:
    """The Agent Server the gateway runs, as its discovery file says."""

    def __init__(self, up=True):
        self.up = up
        self.pid = 50

    def discovery(self):
        return {"url": HOST_URL, "pid": self.pid} if self.up else None


class App:
    """The App Server, as the gateway's agent reaches it."""

    def __init__(self):
        self.up = True
        self.pid = 1
        self.levels = []
        self.refuse = None
        self.last_heard_s = None

    def heard(self, url):
        return {"last_heard_s": self.last_heard_s}

    def discovery(self):
        return {"url": URL, "pid": self.pid} if self.up else None

    def level(self, url, level):
        if self.refuse:
            raise GatewayRefused(409, "level_refused", self.refuse)
        self.levels.append((url, level))


@pytest.fixture
def made(tmp_path, monkeypatch):
    monkeypatch.setenv("STAND_IN_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("STAND_IN_PLAN", "wait")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    built = []

    def make(
        gateway=None,
        app=None,
        configs=None,
        platform="linux",
        key="sk-test",
        stored=None,
        own_url=None,
        host=None,
    ):
        gateway = gateway or FakeGateway()
        app = app or App()
        host = host or Host(up=False)
        configs = configs if configs is not None else {"default": DEFAULT}

        def configuration(name):
            if name not in configs:
                raise GatewayRefused(422, "bad_playbook", f"No playbook {name!r}")
            return configs[name]

        agent = GatewayAgent(
            gateway,
            PairedTokens(tmp_path / "paired.json"),
            root=tmp_path / "sessions",
            command=[sys.executable, str(STAND_IN)],
            end_wait_s=5.0,
            discovery=app.discovery,
            level=app.level,
            reachable=lambda url: host.up if url == HOST_URL else app.up,
            last_heard=app.heard,
            playbooks=lambda: [{"name": name} for name in configs],
            configuration=configuration,
            environ={API_KEY: key} if key else {},
            stored=lambda name: stored if name == API_KEY else None,
            platform=platform,
            microscope="arctis-pc",
            own_url=own_url,
            host=host.discovery,
        )
        built.append(agent)
        return agent, gateway, app

    yield make
    for agent in built:
        for each in list(agent._manager._agents.values()):
            if each.process is not None and each.process.poll() is None:
                each.process.kill()
                each.process.wait(10)


def until(test, agent=None, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if agent is not None:
            agent._manager.poll()
        if test():
            return True
        time.sleep(0.05)
    return False


def _record(tmp_path):
    found = folders(tmp_path / "sessions")
    return read_json(found[0] / RECORD) if found else {}


def test_the_agent_reaches_the_app_through_the_gateways_port(made, tmp_path):
    agent, _, app = made(own_url="http://127.0.0.1:8010")
    agent.start("default", None, "assist", "operator", "page")
    # The level is set on the App Server itself; the agent goes through the gateway.
    assert app.levels == [(URL, "assist")]
    assert until(lambda: _record(tmp_path).get("args"))
    assert _record(tmp_path)["args"][-2:] == ["--url", "http://127.0.0.1:8010"]


def test_start_pairs_the_agent_sets_the_level_and_runs_the_playbook(made, tmp_path):
    agent, _, app = made()
    state = agent.start("default", "Mill 01 to 04", "assist", "operator", "page")
    assert app.levels == [(URL, "assist")]
    paired = PairedTokens(tmp_path / "paired.json").agent()
    assert paired is not None and paired.name == AGENT_NAME and paired.role == AGENT
    assert state.running["playbook"] == "default"
    assert state.running["level"] == "assist"
    assert state.running["goal"] == "Mill 01 to 04"
    assert until(lambda: _record(tmp_path).get("args"))
    record = _record(tmp_path)
    assert record["args"] == ["--playbook", "default", "--url", URL]
    assert record["goal"] == "Mill 01 to 04"
    managed = read_json(folders(tmp_path / "sessions")[0] / MANAGED)
    assert managed["about"]["who"] == "operator"
    written = json.dumps(record) + json.dumps(managed)
    assert "sk-test" not in written, "the API key is never written down"
    assert "FIBSEM_SERVER_TOKEN" not in written


def test_a_stored_key_is_used_and_goes_only_into_the_agents_environment(made, tmp_path):
    agent, _, _ = made(key=None, stored="sk-stored")
    assert agent.state().api_key is True
    agent.start("default", None, "watch", "operator", "page")
    assert agent._manager.agent("arctis-pc").env[API_KEY] == "sk-stored"
    assert until(lambda: _record(tmp_path).get("args"))
    managed = read_json(folders(tmp_path / "sessions")[0] / MANAGED)
    assert "sk-stored" not in json.dumps(_record(tmp_path)) + json.dumps(managed)
    assert "sk-stored" not in json.dumps(dataclasses.asdict(agent.state()))


def test_the_environments_key_wins_over_the_stored_one(made):
    agent, _, _ = made(key="sk-env", stored="sk-stored")
    agent.start("default", None, "watch", "operator", "page")
    assert agent._manager.agent("arctis-pc").env[API_KEY] == "sk-env"


def test_a_budget_override_is_passed_and_nothing_else_is(made, tmp_path):
    agent, _, _ = made()
    agent.start("default", None, "watch", "operator", "page", budget_usd=12.5)
    assert until(lambda: _record(tmp_path).get("args"))
    assert _record(tmp_path)["args"][-2:] == ["--max-budget-usd", "12.5"]
    assert "goal" not in _record(tmp_path)


@pytest.mark.parametrize(
    "setup, error",
    [
        (dict(gateway=FakeGateway(OFF)), "no_app"),
        (dict(key=None), "no_api_key"),
        (
            dict(
                configs={"coder": {**DEFAULT, "mode": "code"}},
                platform="win32",
            ),
            "code_mode_windows",
        ),
    ],
)
def test_start_is_refused_with_its_reason(made, tmp_path, setup, error):
    agent, _, app = made(**setup)
    playbook = next(iter(setup.get("configs", {"default": None})))
    with pytest.raises(GatewayRefused) as refused:
        agent.start(playbook, None, "watch", "operator", "page")
    assert refused.value.error_type == error
    assert app.levels == []
    assert PairedTokens(tmp_path / "paired.json").agent() is None
    assert agent.state().running is None


def test_start_needs_the_app_server_to_answer(made):
    app = App()
    app.up = False
    agent, _, _ = made(app=app)
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "watch", "operator", "page")
    assert refused.value.error_type == "no_app"


def test_start_waits_for_an_app_that_is_starting(made):
    gateway = FakeGateway()
    gateway.health = SimpleNamespace(
        state=lambda: SimpleNamespace(app=SimpleNamespace(status="starting"))
    )
    agent, _, app = made(gateway=gateway)
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "watch", "operator", "page")
    assert refused.value.error_type == "app_starting"
    assert app.levels == []


def test_start_needs_no_microscope(made):
    # The App Server answers with no microscope connected: an agent can work on
    # an experiment, or on devices served without a microscope.
    app = App()
    agent, _, _ = made(app=app)
    agent.start("default", None, "assist", "operator", "page")
    assert app.levels == [(URL, "assist")]


def test_a_code_playbook_runs_off_windows(made):
    agent, _, _ = made(configs={"coder": {**DEFAULT, "mode": "code"}})
    assert agent.start("coder", None, "watch", "operator", "page").running


def test_an_unknown_level_or_playbook_is_refused(made):
    agent, _, _ = made()
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "fly", "operator", "page")
    assert refused.value.error_type == "bad_level"
    with pytest.raises(GatewayRefused) as refused:
        agent.start("nightly", None, "watch", "operator", "page")
    assert refused.value.error_type == "bad_playbook"


def test_a_second_start_is_refused_while_one_runs(made):
    agent, _, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "watch", "operator", "page")
    assert refused.value.error_type == "agent_running"


def test_an_agent_paired_another_way_is_replaced_only_when_confirmed(made, tmp_path):
    tokens = PairedTokens(tmp_path / "paired.json")
    tokens.add("Claude Code on lab-mac", role=AGENT)
    agent, _, app = made()
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "watch", "operator", "page")
    assert refused.value.error_type == "agent_paired"
    assert "Claude Code on lab-mac" in str(refused.value)
    assert app.levels == [], "nothing is done before the confirmation"
    assert tokens.agent().name == "Claude Code on lab-mac"
    agent.start("default", None, "watch", "operator", "page", replace=True)
    assert tokens.agent().name == AGENT_NAME


def test_an_agent_from_another_computer_shows_when_it_was_last_heard(made, tmp_path):
    agent, _, app = made()
    assert agent.state().connected is None
    PairedTokens(tmp_path / "paired.json").add("agent-laptop", role=AGENT)
    app.last_heard_s = 4.0
    assert agent.state().connected == {
        "name": "agent-laptop",
        "last_heard_s": 4.0,
        "quiet": None,
    }
    app.last_heard_s = 300.0
    quiet = agent.state().connected["quiet"]
    assert "5 min" in quiet and "SSH tunnel" in quiet
    app.up = False
    assert agent.state().connected is None


def test_a_refused_level_unpairs_and_starts_nothing(made, tmp_path):
    app = App()
    app.refuse = "Open or create an experiment first."
    agent, _, _ = made(app=app)
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "assist", "operator", "page")
    assert refused.value.error_type == "level_refused"
    assert "experiment" in str(refused.value)
    assert PairedTokens(tmp_path / "paired.json").agent() is None
    assert folders(tmp_path / "sessions") == []


def test_end_stops_the_agent_and_unpairs_it(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    state = agent.end("operator", "page")
    assert state.running is None
    assert state.sessions[0]["ended"] == "ended by the operator"
    assert PairedTokens(tmp_path / "paired.json").agent() is None
    assert gateway.stopped == [], "End alone keeps the app"


def test_end_and_stop_the_app(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    # Started, so the stop is not taken for a stale one and cleared.
    assert until(lambda: _record(tmp_path).get("runs"))
    agent.end("operator", "page", stop_app=True)
    assert gateway.stopped == [("operator", "page", False)]
    assert agent.state().running is None


def test_stopping_the_app_leaves_the_agent_waiting_for_the_next_one(made, tmp_path):
    agent, gateway, app = made()
    agent.start("default", None, "watch", "operator", "page")
    # Started, so the stop is not taken for a stale one and cleared.
    assert until(lambda: _record(tmp_path).get("runs"))
    gateway.stop("hub-mac", "hub")
    assert agent.state().running["state"] == "waiting"
    # Still paired: it carries on as itself.
    assert PairedTokens(tmp_path / "paired.json").agent() is not None
    # Never on the app that was being stopped, however long it answers.
    for _ in range(3):
        agent._unpark()
        agent._manager.poll()
    assert agent.state().running["state"] == "waiting"
    assert len(_record(tmp_path)["runs"]) == 1
    app.pid = 2  # a new app
    agent._unpark()
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2, agent)
    agent.end("operator", "page")
    assert agent.state().running is None
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_the_level_is_set_again_when_the_agent_resumes(made, tmp_path, monkeypatch):
    monkeypatch.setenv("STAND_IN_PLAN", "1,wait")
    agent, _, app = made()
    agent.start("default", None, "drive", "operator", "page")
    app.pid = 2  # the app restarted, back at Watch
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2, agent)
    assert app.levels == [(URL, "drive"), (URL, "drive")]
    assert agent.state().running["state"] == "running"
    assert len(agent.state().running["restarts"]) == 1


def test_a_resume_leaves_the_level_of_an_app_that_kept_running(
    made, tmp_path, monkeypatch
):
    """The operator may have lowered it or pressed Stop agent: it stands."""
    monkeypatch.setenv("STAND_IN_PLAN", "1,wait")
    agent, _, app = made()
    agent.start("default", None, "drive", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2, agent)
    assert app.levels == [(URL, "drive")]


@pytest.mark.parametrize("last", ["assist", "drive"])
def test_a_resume_after_an_app_restart_comes_back_at_the_level_last_set(
    made, tmp_path, monkeypatch, last
):
    monkeypatch.setenv("STAND_IN_PLAN", "wait")
    agent, _, app = made()
    monkeypatch.setattr(
        "fibsem.gateway.agent.call_host", lambda url, method, path: {"level": last}
    )
    agent.start("default", None, "assist", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    agent._follow_level()  # the operator set *last*
    about = read_json(folders(tmp_path / "sessions")[0] / MANAGED)["about"]
    assert about["level_now"] == last

    app.pid = 2
    agent._again(about)
    assert app.levels == [(URL, "assist"), (URL, last)]


def test_an_app_restarted_under_a_running_agent_gets_the_level_back(
    made, tmp_path, monkeypatch
):
    agent, _, app = made()
    monkeypatch.setattr(
        "fibsem.gateway.agent.call_host", lambda url, method, path: {"level": "assist"}
    )
    agent.start("default", None, "drive", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    agent._follow_level()  # the operator lowered it to Assist
    app.pid = 2  # the app restarted; the agent waited for it
    agent._level_read = float("-inf")
    agent._follow_level()
    assert app.levels == [(URL, "drive"), (URL, "assist")]
    about = read_json(folders(tmp_path / "sessions")[0] / MANAGED)["about"]
    assert about["app_pid"] == 2 and about["level_now"] == "assist"


def test_a_resume_after_a_stop_agent_and_an_app_restart_stays_at_watch(made):
    agent, _, app = made()
    app.pid = 2
    agent._again({"level": "drive", "level_now": "watch", "app_pid": 1})
    assert app.levels == []


def test_end_sets_the_app_to_watch_but_the_record_keeps_the_level_set(
    made, tmp_path, monkeypatch
):
    agent, _, app = made()
    monkeypatch.setattr(
        "fibsem.gateway.agent.call_host", lambda url, method, path: {"level": "drive"}
    )
    agent.start("default", None, "drive", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    agent._after_end()
    agent._follow_level()
    agent.end("operator", "page")
    assert app.levels == [(URL, "drive"), (URL, "watch")]
    managed = read_json(folders(tmp_path / "sessions")[0] / MANAGED)
    assert managed["about"]["level_now"] == "drive", "Resume comes back at Drive"


def test_an_agent_that_ends_by_itself_sets_the_app_to_watch_once(
    made, tmp_path, monkeypatch
):
    monkeypatch.setenv("STAND_IN_PLAN", "0")
    agent, _, app = made()
    agent.start("default", None, "assist", "operator", "page")
    agent._after_end()  # seen running
    assert until(lambda: agent._manager.agent("arctis-pc").state == "ended", agent)
    agent._after_end()
    agent._after_end()
    assert app.levels == [(URL, "assist"), (URL, "watch")]


def test_an_old_ended_session_never_sets_the_level(made, tmp_path, monkeypatch):
    """Another agent may hold the slot: only an end this gateway saw."""
    monkeypatch.setenv("STAND_IN_PLAN", "0")
    first, _, app = made()
    first.start("default", None, "assist", "operator", "page")
    assert until(lambda: first._manager.agent("arctis-pc").state == "ended", first)
    again, _, _ = made(gateway=FakeGateway(), app=app)
    again._after_end()
    assert app.levels == [(URL, "assist")]


def test_resume_is_offered_at_the_level_last_set():
    from fibsem.gateway.agent import _row

    def row(about):
        return _row({"session_id": "s1", "manager": {"about": about}})

    assert row({"level": "drive", "level_now": "assist"})["last_level"] == "assist"
    assert row({"level": "drive"})["last_level"] == "drive"
    assert row({})["last_level"] is None


def test_a_watch_agent_is_resumed_without_a_level_call(made, tmp_path, monkeypatch):
    monkeypatch.setenv("STAND_IN_PLAN", "1,wait")
    agent, _, app = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2, agent)
    assert app.levels == [(URL, "watch")]


def test_a_restarted_gateway_takes_the_agent_back(made, tmp_path):
    first, gateway, app = made()
    first.start("default", "Mill 01", "assist", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    again, _, _ = made(gateway=FakeGateway(), app=app)
    state = again.state()
    assert state.running is not None
    assert state.running["goal"] == "Mill 01"
    assert state.running["level"] == "assist"
    again.end("operator", "page")
    assert until(lambda: "ended" in _record(tmp_path)["runs"][-1])
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_the_playbooks_and_a_configuration_come_from_fibsem_agent(made):
    agent, _, _ = made(configs={"default": DEFAULT, "overnight": DEFAULT})
    assert [row["name"] for row in agent.playbooks()] == ["default", "overnight"]
    assert agent.configuration("default") == DEFAULT


def test_the_state_says_whether_the_api_key_is_set(made):
    assert made()[0].state().api_key is True
    assert made(key=None)[0].state().api_key is False


def _by_hand(tmp_path, **about):
    """A session someone ran with fibsem-agent in a terminal, now ended."""
    session = SessionFolder.open(
        "by-hand",
        {"playbook": "default", "mode": "none", **about},
        tmp_path / "sessions",
    )
    session.end(0, "stopped")
    return session.folder


def test_history_pages_through_every_session_not_only_the_newest(made, tmp_path):
    for i in range(SESSIONS_SHOWN + 3):
        session = SessionFolder.open(
            f"s{i:02d}", {"playbook": "default", "mode": "none"}, tmp_path / "sessions"
        )
        session.end(0, "stopped")
    agent, _, _ = made()
    assert len(agent.state().sessions) == SESSIONS_SHOWN
    page = agent.sessions(3)
    assert len(page.sessions) == 3 and page.more
    every = agent.sessions()
    assert len(every.sessions) == SESSIONS_SHOWN + 3 and not every.more
    # The oldest can be resumed as any other.
    oldest = every.sessions[-1]["session_id"]
    assert (
        agent.resume(oldest, "watch", "operator", "page").running["session_id"]
        == oldest
    )


def test_resume_carries_an_ended_gateway_session_on(made, tmp_path):
    agent, _, app = made()
    agent.start("default", "Mill 01", "assist", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    session_id = agent.state().running["session_id"]
    agent.end("operator", "page")
    state = agent.resume(session_id, "drive", "operator", "page", budget_usd=4)
    assert app.levels[-1] == (URL, "drive")
    assert PairedTokens(tmp_path / "paired.json").agent().name == AGENT_NAME
    assert state.running["session_id"] == session_id
    assert state.running["resumed"] is True
    assert state.running["goal"] == "Mill 01"
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2)
    assert len(folders(tmp_path / "sessions")) == 1, "the same folder"
    assert _record(tmp_path)["args"] == [
        "--playbook",
        "default",
        "--url",
        URL,
        "--max-budget-usd",
        "4",
    ]


def test_a_session_that_kept_its_playbook_is_listed_and_resumed_by_name(made, tmp_path):
    """Once a session keeps its playbook, the record holds
    {name, hash, chosen_by, skills}: the page still gets the name."""
    folder = _by_hand(tmp_path)
    SessionFolder(folder).keep_playbook(
        {
            "playbook": {
                "name": "default",
                "hash": "3f9c",
                "chosen_by": "x",
                "skills": [],
            }
        }
    )
    agent, _, _ = made()
    row = agent.state().sessions[0]
    assert (row["playbook"], row["playbook_hash"]) == ("default", "3f9c")
    agent.resume("by-hand", "watch", "operator", "page")
    assert agent.state().running["playbook"] == "default"


def test_resume_takes_a_terminal_session_as_its_record_says(made, tmp_path):
    folder = _by_hand(tmp_path, model="claude-test", effort="low")
    (folder / WORK).mkdir()
    record = read_json(folder / RECORD)
    record.update(mode="files", cwd=str(folder / WORK))
    (folder / RECORD).write_text(json.dumps(record))
    agent, _, _ = made()
    assert agent.state().sessions[0]["cannot_resume"] is None
    agent.resume("by-hand", "watch", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2)
    assert _record(tmp_path)["args"] == [
        "--playbook",
        "default",
        "--url",
        URL,
        "--files",
        "--model",
        "claude-test",
        "--effort",
        "low",
    ]


def test_resume_names_a_working_folder_outside_the_session(made, tmp_path):
    work = tmp_path / "elsewhere"
    work.mkdir()
    _by_hand(tmp_path, mode="code", cwd=str(work))
    agent, _, _ = made()
    agent.resume("by-hand", "watch", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2)
    assert _record(tmp_path)["args"][-2:] == ["--code", str(work)]


@pytest.mark.parametrize(
    "about, platform, why",
    [
        ({"mode": "files", "cwd": "/no/such/folder"}, "linux", "folder is gone"),
        ({"mode": "code", "cwd": "/tmp"}, "win32", "no sandbox on Windows"),
    ],
)
def test_a_session_that_cant_resume_says_why(made, tmp_path, about, platform, why):
    _by_hand(tmp_path, **about)
    agent, _, app = made(platform=platform)
    assert why in agent.state().sessions[0]["cannot_resume"]
    with pytest.raises(GatewayRefused) as refused:
        agent.resume("by-hand", "watch", "operator", "page")
    assert refused.value.error_type == "cannot_resume"
    assert app.levels == []
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_resume_is_refused_with_starts_reasons(made, tmp_path):
    _by_hand(tmp_path)
    agent, _, _ = made()
    with pytest.raises(GatewayRefused) as refused:
        agent.resume("nowhere", "watch", "operator", "page")
    assert refused.value.error_type == "no_session"
    app = App()
    app.up = False
    agent, _, _ = made(app=app)
    with pytest.raises(GatewayRefused) as refused:
        agent.resume("by-hand", "watch", "operator", "page")
    assert refused.value.error_type == "no_app"
    agent, _, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    with pytest.raises(GatewayRefused) as refused:
        agent.resume("by-hand", "watch", "operator", "page")
    assert refused.value.error_type == "agent_running"


def test_a_session_from_before_run_resumes_in_its_checkout(made, tmp_path):
    checkout = tmp_path / "fibsem-os"
    checkout.mkdir()
    _by_hand(tmp_path, cwd=str(checkout))
    agent, _, _ = made()
    agent.resume("by-hand", "watch", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2)
    assert _record(tmp_path)["args"][-2:] == ["--checkout", str(checkout)]


def test_a_session_in_its_own_run_folder_needs_no_folder_named(made, tmp_path):
    folder = _by_hand(tmp_path)
    record = read_json(folder / RECORD)
    record["cwd"] = str(folder / "run")
    (folder / RECORD).write_text(json.dumps(record))
    agent, _, _ = made()
    assert agent.state().sessions[0]["cannot_resume"] is None
    agent.resume("by-hand", "watch", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2)
    assert _record(tmp_path)["args"] == ["--playbook", "default", "--url", URL]


def test_the_keyring_is_not_asked_on_every_refresh_of_the_page(tmp_path):
    from fibsem.gateway import agent as module

    asked = []
    agent = GatewayAgent(
        FakeGateway(),
        PairedTokens(tmp_path / "paired.json"),
        root=tmp_path / "sessions",
        environ={},
        stored=lambda name: asked.append(name) or "sk-stored",
    )
    assert agent.state().api_key and agent.state().api_key
    assert asked == [API_KEY]
    agent._key_seen = (time.monotonic() - module.KEY_KEPT_S - 1, False)
    assert agent.state().api_key and len(asked) == 2


def test_a_sessions_working_folder_is_looked_at_once_a_minute(tmp_path, monkeypatch):
    from fibsem.gateway import agent as module

    monkeypatch.setattr(module, "_folders_seen", {})
    folder = tmp_path / "work"
    folder.mkdir()
    record = {"session_id": "s1", "cwd": str(folder)}
    assert module._row(record)["cannot_resume"] is None
    folder.rmdir()
    assert module._row(record)["cannot_resume"] is None, "kept for a minute"
    # resuming always looks again
    assert module._why_not(record, "linux") == "Its working folder is gone."
    module._folders_seen[str(folder)] = (time.monotonic() - 61, True)
    assert module._row(record)["cannot_resume"] == "Its working folder is gone."


def test_the_shipped_playbooks_and_a_configuration_read_in_process():
    from fibsem.gateway.agent import read_configuration, read_playbooks

    names = [row["name"] for row in read_playbooks()]
    assert names
    config = read_configuration(names[0])
    assert config["playbook"]["name"] == names[0] and config["system_prompt"]
    with pytest.raises(GatewayRefused) as refused:
        read_configuration("no-such-playbook")
    assert refused.value.error_type == "bad_playbook"


# --- the one agent slot -------------------------------------------------------------


def test_the_slot_is_in_use_by_a_running_agent_or_one_heard_lately(made, tmp_path):
    agent, _, app = made()
    assert agent.in_use() is None
    PairedTokens(tmp_path / "paired.json").add("claude-code on this PC", role=AGENT)
    assert agent.in_use() is None, "never heard: not in use"
    app.last_heard_s = 4.0
    assert agent.in_use() == "claude-code on this PC"
    app.last_heard_s = 600.0
    assert agent.in_use() is None, "gone quiet: the slot is free to take"
    agent.start("default", None, "watch", "operator", "page", replace=True)
    assert agent.in_use().startswith("the agent started from the gateway at ")
    assert agent.own_running() == agent.in_use()


def _allowing(agent, gateway, tmp_path, always=False):
    """The gateway's routes over *agent*, and an ask from this PC waiting."""
    from fastapi.testclient import TestClient

    from fibsem.gateway.server import GatewayAuth, build_gateway_app

    token_file = tmp_path / "operator-token.json"
    token_file.write_text('{"token": "operator"}')
    auth = GatewayAuth.from_disk(token_file, tmp_path / "paired.json")
    auth.connections.always_allow = always
    client = TestClient(build_gateway_app(gateway, auth, agent=agent))
    client.headers["Authorization"] = "Bearer operator"
    return client, auth.connections


def test_allowing_an_agent_ends_the_one_the_gateway_started(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    client, connections = _allowing(agent, gateway, tmp_path)
    connections.ask("claude-code 2.1", "this PC", "lab", wait_s=0)
    card = client.get("/gateway/connections").json()["asking"]
    assert card["replaces"].startswith("the agent started from the gateway")
    assert "allowing ends it" in card["if_allowed"]
    allowed = client.post(f"/gateway/connections/{card['id']}/allow")
    assert allowed.json()["allowed"] is True
    # Ended as its End does, not left running without a token.
    assert agent.state().running is None
    assert agent.state().sessions[0]["ended"] == "ended by the operator"
    paired = PairedTokens(tmp_path / "paired.json").agent()
    assert paired.name == "claude-code on this PC"
    ((who, _, result),) = gateway.access
    assert who == "operator"
    assert result.startswith("Allowed claude-code on this PC, replacing the agent")


def test_an_allow_that_fails_leaves_the_gateways_agent_running(
    made, tmp_path, monkeypatch
):
    """The Allow before the end (L3): an ask that expired ends nothing."""
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    client, connections = _allowing(agent, gateway, tmp_path)
    connections.ask("claude-code 2.1", "this PC", "lab", wait_s=0)
    card = client.get("/gateway/connections").json()["asking"]

    def expired(ask_id):
        raise LookupError("That ask has expired.")

    monkeypatch.setattr(connections, "allow", expired)
    assert client.post(f"/gateway/connections/{card['id']}/allow").status_code == 404
    assert agent.state().running["state"] == "running"


def test_always_allow_never_replaces_an_agent_in_use(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    _, connections = _allowing(agent, gateway, tmp_path, always=True)
    answer = connections.ask(
        "claude-code 2.1", "this PC", "lab", wait_s=0, loopback=True
    )
    assert answer["state"] == "waiting", "the operator's card decides"
    assert agent.state().running is not None
    assert PairedTokens(tmp_path / "paired.json").agent().name == AGENT_NAME


def test_always_allow_takes_a_free_slot_and_says_so_on_the_history(made, tmp_path):
    agent, gateway, _ = made()
    _, connections = _allowing(agent, gateway, tmp_path, always=True)
    answer = connections.ask(
        "claude-code 2.1", "this PC", "lab", wait_s=0, loopback=True
    )
    assert answer["state"] == "allowed"
    assert gateway.access == [
        ("always allow", "setting", "Allowed claude-code on this PC")
    ]


# --- the slot record ------------------------------------------------------------


def test_the_slot_record_follows_the_gateways_agent(made, tmp_path, monkeypatch):
    agent, _, _ = made()
    monkeypatch.setattr(
        "fibsem.gateway.agent.call_host", lambda url, method, path: {"level": "assist"}
    )
    agent.start("default", None, "drive", "operator", "page")
    slot = agent.state().slot
    assert slot["holder"] == AGENT_NAME and slot["came_in"] == "start"
    assert slot["state"] == "running" and slot["level"] == "drive"
    assert slot["why"] == "started by operator (page)"
    assert until(lambda: _record(tmp_path).get("runs"))
    agent._follow_level()
    assert agent.slot.read()["level"] == "assist"
    agent.end("operator", "page")
    slot = agent.slot.read()
    assert slot["state"] == "ended" and slot["why"] == "ended by the operator"
    assert (tmp_path / "slot.json").is_file(), "written down, beside the sessions"


def test_an_allowed_agent_takes_the_slot(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    client, connections = _allowing(agent, gateway, tmp_path)
    connections.ask("claude-code 2.1", "this PC", "lab", wait_s=0)
    card = client.get("/gateway/connections").json()["asking"]
    client.post(f"/gateway/connections/{card['id']}/allow")
    agent._follow_slot()  # the old agent's end never overwrites the new holder
    slot = agent.slot.read()
    assert slot["holder"] == "claude-code on this PC"
    assert slot["came_in"] == "allow" and slot["state"] == "running"
    assert slot["why"].startswith("allowed by operator, replacing the agent")


def test_resume_is_offered_with_the_budget_it_last_ran_with():
    from fibsem.gateway.agent import _row

    def row(about, kept=None):
        record = {"session_id": "s1", "manager": {"about": about}}
        if kept is not None:
            record["budget"] = kept
        return _row(record)["last_budget_usd"]

    assert row({"budget_usd": 5}) == 5.0
    # One the operator changed while it ran wins over the start budget.
    assert row({"budget_usd": 5}, {"usd": 8, "spent_before_usd": 0.5}) == 8.0
    assert row({}) is None


def test_a_parked_agent_stopping_is_not_taken_for_an_end(made, tmp_path):
    # The manager's poll may see the stopped agent before park does.
    agent, _, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    managed = agent._manager.agent(agent.microscope)
    managed.parked = True
    (managed.folder / "stop").touch()
    assert until(lambda: not agent._manager._alive(managed))
    agent._manager.poll()
    assert managed.state == "waiting"
    assert "ended" not in read_json(managed.folder / MANAGED)
    agent.end("operator", "page")


def test_a_parked_agent_ends_when_the_next_app_opens_another_experiment(made, tmp_path):
    agent, gateway, app = made()
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    gateway.stop("operator", "page")
    assert agent.state().running["state"] == "waiting"
    gateway.experiment = "another"
    app.pid = 2
    agent._unpark()
    assert agent.state().running is None
    assert len(_record(tmp_path)["runs"]) == 1
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_start_with_no_app_waits_and_starts_at_its_level_when_the_app_answers(
    made, tmp_path
):
    app = App()
    app.up = False
    agent, _, _ = made(
        gateway=FakeGateway(OFF), app=app, own_url="http://127.0.0.1:8010"
    )
    state = agent.start("default", "Mill 01", "assist", "operator", "page")
    assert state.running["state"] == "waiting"
    assert app.levels == []
    agent._manager.poll()
    assert not _record(tmp_path).get("runs"), "nothing runs before the app"
    app.up = True
    assert until(lambda: _record(tmp_path).get("args"), agent)
    assert app.levels == [(URL, "assist")]
    assert _record(tmp_path)["args"][-2:] == ["--url", "http://127.0.0.1:8010"]
    assert agent.state().running["state"] == "running"
    agent.end("operator", "page")


def test_start_with_no_app_and_no_gateway_address_is_refused(made):
    agent, _, _ = made(gateway=FakeGateway(OFF))
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "watch", "operator", "page")
    assert refused.value.error_type == "no_app"


def test_a_start_waiting_for_its_app_ends_when_the_gateway_quits(made, tmp_path):
    app = App()
    app.up = False
    agent, _, _ = made(
        gateway=FakeGateway(OFF), app=app, own_url="http://127.0.0.1:8010"
    )
    agent.start("default", None, "assist", "operator", "page")
    folder = agent._manager.agent(agent.microscope).folder
    agent.left()
    assert agent.state().running is None
    assert PairedTokens(tmp_path / "paired.json").agent() is None
    managed = read_json(folder / MANAGED)
    assert managed["ended"] == "did not start (the gateway closed first)"


def test_start_refuses_a_goal_too_long_to_report(made):
    from fibsem.ai.agent.report import MAX_GOAL

    agent, _, app = made()
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", "g" * (MAX_GOAL + 1), "watch", "operator", "page")
    assert refused.value.error_type == "goal_too_long"
    assert app.levels == []


def test_history_says_each_sessions_status_and_experiment(made, tmp_path):
    ended = _by_hand(tmp_path)
    (ended / "agent").mkdir()
    (ended / "agent" / "experiment.json").write_text(
        json.dumps(
            {"session_id": "by-hand", "path": "/x/exp", "name": "exp", "id": "e-1"}
        )
    )
    # The gateway's agent, parked when its app was stopped: not ended.
    parked = SessionFolder.open(
        "parked", {"playbook": "default", "mode": "none"}, tmp_path / "sessions"
    )
    parked.end(0, "stopped")
    (parked.folder / MANAGED).write_text(json.dumps({"about": {"level": "assist"}}))
    agent, _, _ = made()
    rows = {row["session_id"]: row for row in agent.sessions().sessions}
    assert rows["by-hand"]["status"] == "ended"
    assert rows["by-hand"]["experiment"] == {
        "path": "/x/exp",
        "name": "exp",
        "id": "e-1",
    }
    assert rows["parked"]["status"] == "waiting"


def test_an_allowed_agent_leaves_the_slot_when_unpaired_or_quiet(made, tmp_path):
    agent, _, app = made()
    tokens = PairedTokens(tmp_path / "paired.json")
    paired, _ = tokens.add("claude-code on this PC", role=AGENT)
    agent.allowed("claude-code on this PC", "operator", None)
    agent._follow_slot()
    assert agent.slot.read()["state"] == "running", "just allowed: not yet heard"
    app.last_heard_s = 4.0
    agent._allowed_at = agent._allowed_read = float("-inf")
    agent._follow_slot()
    assert agent.slot.read()["state"] == "running"
    app.last_heard_s = 600.0
    agent._allowed_read = float("-inf")
    agent._follow_slot()
    held = agent.slot.read()
    assert held["state"] == "ended" and held["why"].startswith("not heard from")
    # Watch, as for a hub's agent, and only once.
    assert app.levels == [(URL, "watch")]
    agent._allowed_read = float("-inf")
    agent._follow_slot()
    assert app.levels == [(URL, "watch")]
    app.last_heard_s = 3.0  # back again: it carries on at Watch
    agent._allowed_read = float("-inf")
    agent._follow_slot()
    assert agent.slot.read()["state"] == "running"
    assert app.levels == [(URL, "watch")]
    tokens.remove(paired.id)
    agent._allowed_read = float("-inf")
    agent._follow_slot()
    held = agent.slot.read()
    assert (held["state"], held["why"]) == ("ended", "no longer paired")
    assert app.levels == [(URL, "watch"), (URL, "watch")]


def test_a_new_gateway_frees_a_slot_and_pairing_nothing_carries_on(made, tmp_path):
    agent, _, app = made()
    app.up = False
    agent.slot.take(AGENT_NAME, "start", "started by operator (page)", state="waiting")
    PairedTokens(tmp_path / "paired.json").add(AGENT_NAME, role=AGENT)
    again, _, _ = made()
    assert again.slot.read()["state"] == "ended"
    assert PairedTokens(tmp_path / "paired.json").agent() is None


# -- the Agent Server: the agent works with no app ----------------------------


def _levels_on(app, monkeypatch, level):
    monkeypatch.setattr(
        "fibsem.gateway.agent.call_host", lambda url, method, path: {"level": level}
    )


def test_start_with_no_app_runs_on_the_agent_host_and_moves_to_the_app(
    made, tmp_path, monkeypatch
):
    app, host = App(), Host()
    app.up = False
    gateway = FakeGateway(OFF)
    agent, _, _ = made(
        gateway=gateway, app=app, host=host, own_url="http://127.0.0.1:8010"
    )
    state = agent.start("default", "Plan the run", "assist", "operator", "page")
    # It runs at once, through the gateway, which passes its calls to the host.
    assert state.running["state"] == "running"
    assert state.agent_host
    assert app.levels == [(HOST_URL, "assist")]
    assert until(lambda: _record(tmp_path).get("args"), agent)
    assert _record(tmp_path)["args"][-2:] == ["--url", "http://127.0.0.1:8010"]
    # The app answers: it gets the level the agent runs at, and nothing restarts.
    _levels_on(app, monkeypatch, "assist")
    app.up, app.pid, gateway.mode = True, 2, RUNNING
    agent._follow_level()
    assert app.levels == [(HOST_URL, "assist"), (URL, "assist")]
    assert len(_record(tmp_path)["runs"]) == 1
    agent.end("operator", "page")


def test_stopping_the_app_moves_a_running_agent_to_the_agent_host(
    made, tmp_path, monkeypatch
):
    host = Host()
    agent, gateway, app = made(host=host, own_url="http://127.0.0.1:8010")
    agent.start("default", None, "drive", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    _levels_on(app, monkeypatch, "drive")
    gateway.stop("operator", "page")
    # Not parked: it carries on, and its calls go to the host once the app is gone.
    assert agent.state().running["state"] == "running"
    app.up = False
    agent._level_read = float("-inf")
    agent._follow_level()
    assert app.levels == [(URL, "drive"), (HOST_URL, "drive")]
    agent._unpark()
    # The same app and experiment again: it carries on there.
    app.up, app.pid, gateway.mode = True, 2, RUNNING
    agent._unpark()
    agent._level_read = float("-inf")
    agent._follow_level()
    assert agent.state().running["state"] == "running"
    assert app.levels[-1] == (URL, "drive")
    assert len(_record(tmp_path)["runs"]) == 1
    agent.end("operator", "page")


def test_an_agent_on_the_host_ends_when_the_next_app_opens_another_experiment(
    made, tmp_path
):
    agent, gateway, app = made(host=Host(), own_url="http://127.0.0.1:8010")
    agent.start("default", None, "watch", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    gateway.stop("operator", "page")
    app.up = False
    agent._unpark()
    assert agent.state().running["state"] == "running"
    gateway.experiment = "another"
    app.up, app.pid = True, 2
    agent._unpark()
    assert agent.state().running is None
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_resume_with_no_app_runs_on_the_agent_host(made, tmp_path):
    app, host = App(), Host()
    agent, gateway, _ = made(app=app, host=host, own_url="http://127.0.0.1:8010")
    agent.start("default", None, "assist", "operator", "page")
    assert until(lambda: _record(tmp_path).get("runs"))
    session_id = agent._manager.agent(agent.microscope).session_id
    agent.end("operator", "page")
    app.up, gateway.mode = False, OFF
    state = agent.resume(session_id, "assist", "operator", "page")
    assert state.running["state"] == "running"
    assert app.levels[-1] == (HOST_URL, "assist")
    agent.end("operator", "page")


def test_with_no_app_and_no_agent_host_start_still_waits(made):
    app = App()
    app.up = False
    agent, _, _ = made(
        gateway=FakeGateway(OFF), app=app, own_url="http://127.0.0.1:8010"
    )
    state = agent.start("default", None, "watch", "operator", "page")
    assert state.running["state"] == "waiting"
    assert not state.agent_host
    agent.end("operator", "page")
