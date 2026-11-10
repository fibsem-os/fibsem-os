"""The gateway's agent: Start's chain and its refusals, End, stopping the app
ending the agent, the level set again on a resume, and a restarted gateway
taking the agent back, with the stand-in ``fibsem-agent``
(tests/agent/stand_in.py) in real processes and a fake App Server."""

import dataclasses
import json
import sys
import time
from pathlib import Path

import pytest

from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    WORK,
    SessionFolder,
    folders,
    read_json,
)
from fibsem.gateway.agent import AGENT_NAME, API_KEY, GatewayAgent
from fibsem.gateway.gateway import OFF, RUNNING, GatewayRefused, GatewayState
from fibsem.server.pairing import AGENT, PairedTokens

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "agent" / "stand_in.py"
URL = "http://127.0.0.1:8001"
DEFAULT = {"playbook": {"name": "default"}, "mode": "none", "model": None}


class FakeGateway:
    def __init__(self, mode=RUNNING):
        self.mode = mode
        self.before_stop = []
        self.watchers = []
        self.stopped = []

    def state(self):
        return GatewayState(mode=self.mode, version="test")

    def stop(self, who, via, force=False, window=False):
        for hook in self.before_stop:
            hook()
        self.stopped.append((who, via, force))
        self.mode = OFF


class App:
    """The App Server, as the gateway's agent reaches it."""

    def __init__(self):
        self.up = True
        self.levels = []
        self.refuse = None

    def discovery(self):
        return {"url": URL, "pid": 1} if self.up else None

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
        stamp=lambda: (),
    ):
        gateway = gateway or FakeGateway()
        asked = []
        app = app or App()
        configs = configs if configs is not None else {"default": DEFAULT}

        def agent_json(flags):
            asked.append(list(flags))
            if flags[0] == "--list-playbooks":
                return [{"name": name} for name in configs]
            if flags[1] not in configs:
                raise GatewayRefused(422, "bad_playbook", f"No playbook {flags[1]!r}")
            return configs[flags[1]]

        agent = GatewayAgent(
            gateway,
            PairedTokens(tmp_path / "paired.json"),
            root=tmp_path / "sessions",
            command=[sys.executable, str(STAND_IN)],
            end_wait_s=5.0,
            discovery=app.discovery,
            level=app.level,
            reachable=lambda url: app.up,
            agent_json=agent_json,
            stamp=stamp,
            environ={API_KEY: key} if key else {},
            stored=lambda name: stored if name == API_KEY else None,
            platform=platform,
            microscope="arctis-pc",
        )
        agent.asked = asked
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
    agent.end("operator", "page", stop_app=True)
    assert gateway.stopped == [("operator", "page", False)]
    assert agent.state().running is None


def test_stopping_the_app_ends_the_agent_first(made, tmp_path):
    agent, gateway, _ = made()
    agent.start("default", None, "watch", "operator", "page")
    gateway.stop("hub-mac", "hub")
    assert agent.state().running is None
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_the_level_is_set_again_when_the_agent_resumes(made, tmp_path, monkeypatch):
    monkeypatch.setenv("STAND_IN_PLAN", "1,wait")
    agent, _, app = made()
    agent.start("default", None, "drive", "operator", "page")
    assert until(lambda: len(_record(tmp_path).get("runs", [])) == 2, agent)
    assert app.levels == [(URL, "drive"), (URL, "drive")]
    assert agent.state().running["state"] == "running"
    assert len(agent.state().running["restarts"]) == 1


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


def test_fibsem_agent_is_asked_again_only_when_a_playbook_file_changes(made):
    stamp = [("a.yaml", 1, 10)]
    agent, _, _ = made(stamp=lambda: tuple(stamp))
    agent.playbooks()
    agent.playbooks()
    agent.configuration("default")
    agent.configuration("default")
    assert len(agent.asked) == 2
    stamp[0] = ("a.yaml", 2, 10)
    agent.playbooks()
    assert len(agent.asked) == 3


def test_a_refused_playbook_is_asked_again(made):
    agent, _, _ = made()
    for _ in range(2):
        with pytest.raises(GatewayRefused):
            agent.configuration("missing")
    assert len(agent.asked) == 2


def test_the_knowledge_stamp_changes_with_a_lab_playbook(tmp_path, monkeypatch):
    from fibsem.gateway.agent import knowledge_stamp

    monkeypatch.setattr("fibsem.paths.state_root", lambda: tmp_path)
    folder = tmp_path / "knowledge" / "playbooks"
    folder.mkdir(parents=True)
    before = knowledge_stamp()
    (folder / "lab.yaml").write_text("name: lab\n")
    assert knowledge_stamp() != before


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
