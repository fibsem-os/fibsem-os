"""The gateway's agent: Start's chain and its refusals, End, stopping the app
ending the agent, the level set again on a resume, and a restarted gateway
taking the agent back, with the stand-in ``fibsem-agent``
(tests/agent/stand_in.py) in real processes and a fake App Server."""

import json
import sys
import time
from pathlib import Path

import pytest

from fibsem.ai.agent.manager import MANAGED, RECORD, folders, read_json
from fibsem.gateway.agent import AGENT_NAME, API_KEY, GatewayAgent
from fibsem.gateway.gateway import OFF, RUNNING, GatewayRefused, GatewayState
from fibsem.server.pairing import AGENT, PairedTokens

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "agent" / "stand_in.py"
URL = "http://127.0.0.1:8001"
DEFAULT = {"profile": {"name": "default"}, "mode": "none", "model": None}


class FakeGateway:
    def __init__(self, mode=RUNNING):
        self.mode = mode
        self.before_stop = []
        self.watchers = []
        self.stopped = []

    def state(self):
        return GatewayState(mode=self.mode, version="test")

    def stop(self, who, via, force=False):
        for hook in self.before_stop:
            hook()
        self.stopped.append((who, via, force))
        self.mode = OFF


class App:
    """The App Server, as the gateway's agent reaches it."""

    def __init__(self):
        self.up = True
        self.connected = True
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

    def make(gateway=None, app=None, configs=None, platform="linux", key="sk-test"):
        gateway = gateway or FakeGateway()
        app = app or App()
        configs = configs if configs is not None else {"default": DEFAULT}

        def agent_json(flags):
            if flags[0] == "--list-profiles":
                return [{"name": name} for name in configs]
            if flags[1] not in configs:
                raise GatewayRefused(422, "bad_profile", f"No profile {flags[1]!r}")
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
            connected=lambda url: app.connected,
            agent_json=agent_json,
            environ={API_KEY: key} if key else {},
            platform=platform,
            microscope="arctis-pc",
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


def test_start_pairs_the_agent_sets_the_level_and_runs_the_profile(made, tmp_path):
    agent, _, app = made()
    state = agent.start("default", "Mill 01 to 04", "assist", "operator", "page")
    assert app.levels == [(URL, "assist")]
    paired = PairedTokens(tmp_path / "paired.json").agent()
    assert paired is not None and paired.name == AGENT_NAME and paired.role == AGENT
    assert state.running["profile"] == "default"
    assert state.running["level"] == "assist"
    assert state.running["goal"] == "Mill 01 to 04"
    assert until(lambda: _record(tmp_path).get("args"))
    record = _record(tmp_path)
    assert record["args"] == ["--profile", "default", "--url", URL]
    assert record["goal"] == "Mill 01 to 04"
    managed = read_json(folders(tmp_path / "sessions")[0] / MANAGED)
    assert managed["about"]["who"] == "operator"
    written = json.dumps(record) + json.dumps(managed)
    assert "sk-test" not in written, "the API key is never written down"
    assert "FIBSEM_SERVER_TOKEN" not in written


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
    profile = next(iter(setup.get("configs", {"default": None})))
    with pytest.raises(GatewayRefused) as refused:
        agent.start(profile, None, "watch", "operator", "page")
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


def test_start_needs_a_microscope_connected(made, tmp_path):
    app = App()
    app.connected = False
    agent, _, _ = made(app=app)
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "assist", "operator", "page")
    assert refused.value.error_type == "not_connected"
    assert app.levels == []
    assert PairedTokens(tmp_path / "paired.json").agent() is None


def test_a_code_profile_runs_off_windows(made):
    agent, _, _ = made(configs={"coder": {**DEFAULT, "mode": "code"}})
    assert agent.start("coder", None, "watch", "operator", "page").running


def test_an_unknown_level_or_profile_is_refused(made):
    agent, _, _ = made()
    with pytest.raises(GatewayRefused) as refused:
        agent.start("default", None, "fly", "operator", "page")
    assert refused.value.error_type == "bad_level"
    with pytest.raises(GatewayRefused) as refused:
        agent.start("nightly", None, "watch", "operator", "page")
    assert refused.value.error_type == "bad_profile"


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


def test_the_profiles_and_a_configuration_come_from_fibsem_agent(made):
    agent, _, _ = made(configs={"default": DEFAULT, "overnight": DEFAULT})
    assert [row["name"] for row in agent.profiles()] == ["default", "overnight"]
    assert agent.configuration("default") == DEFAULT


def test_the_state_says_whether_the_api_key_is_set(made):
    assert made()[0].state().api_key is True
    assert made(key=None)[0].state().api_key is False
