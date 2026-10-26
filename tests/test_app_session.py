"""The running application without a window: microscope, experiment, events.

A window is one client of an ``AppSession``; these drive it with none.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.structures import Experiment
from fibsem.workflows.event_recording import EVENTS_FILENAME, read_events, recorder_for
from fibsem.workflows.session import EXPERIMENT, MICROSCOPE, AppSession
from tests.protocols import add_planned, planned


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment(path=tmp_path, name="session")
    os.makedirs(experiment.path, exist_ok=True)
    return experiment


def _recorded(session, experiment):
    """Mark the stream, close it, and read back what reached the experiment."""
    session.microscope.record_event("task_step", {"step": "MARK"})
    session.stop_recording()
    return [
        e
        for e in read_events(Path(experiment.path) / EVENTS_FILENAME)
        if e.get("payload", {}).get("step") == "MARK"
    ]


@pytest.fixture
def session():
    session = AppSession()
    yield session
    session.stop_recording()


def test_the_session_says_when_its_microscope_changes(session):
    heard = []
    session.changed.connect(heard.append)
    microscope = SimpleNamespace()
    session.microscope = microscope
    session.microscope = microscope
    session.microscope = None
    assert heard == [MICROSCOPE, MICROSCOPE]


def test_the_session_says_an_experiment_was_opened_once_it_is_set_up(
    session, experiment
):
    """A listener reads the new experiment and a fresh correlation."""
    old_correlation = session.correlation
    heard = []
    session.changed.connect(
        lambda what: heard.append(
            (what, session.experiment, session.correlation is old_correlation)
        )
    )
    session.adopt(experiment)
    assert heard == [(EXPERIMENT, experiment, False)]


def test_adopting_an_experiment_logs_to_its_logfile(session, experiment):
    session.adopt(experiment)

    logging.info("after adoption")
    for handler in logging.getLogger().handlers:
        handler.flush()
    logfile = Path(experiment.path) / "logfile.log"
    assert "after adoption" in logfile.read_text(encoding="utf-8")


def test_no_microscope_records_nothing(session):
    session.start_recording()

    assert session.recorder is None


def test_a_connection_is_recorded_to_the_open_experiment(
    session, microscope, experiment
):
    session.microscope = microscope
    session.adopt(experiment)

    session.start_recording()

    assert recorder_for(microscope) is session.recorder
    assert _recorded(session, experiment)


def test_asking_again_keeps_the_stream(session, microscope):
    session.microscope = microscope
    session.start_recording()
    first = session.recorder

    session.start_recording()

    assert session.recorder is first


def test_adopting_an_experiment_moves_the_stream(session, microscope, experiment):
    session.microscope = microscope
    session.start_recording()

    session.adopt(experiment)

    assert _recorded(session, experiment)


def test_stopping_closes_the_stream(session, microscope):
    session.microscope = microscope
    session.start_recording()

    session.stop_recording()

    assert session.recorder is None
    assert recorder_for(microscope) is None


class _FakeHost:
    """The App Server host's surface the session uses."""

    def __init__(self, session, app_context, serve_microscope=False):
        self.serve_microscope = serve_microscope
        self.running = False
        self.starts = 0
        # Seconds since the agent was last heard; None: never.
        self.seen = None

    def agent_seconds_since_seen(self):
        return self.seen

    def start(self):
        self.starts += 1
        self.running = True
        return True

    def stop(self):
        self.running = False


@pytest.fixture
def fake_hosting(monkeypatch):
    from fibsem.server import hosting

    monkeypatch.setattr(hosting, "AppServerHost", _FakeHost)


def test_the_app_server_is_the_sessions_and_starts_once(session, fake_hosting):
    assert session.start_app_server(lambda: None)
    host = session.app_server

    session.start_app_server(lambda: None)

    assert session.app_server is host and host.starts == 2 and host.running


def test_the_microscope_is_served_unless_the_operator_says_not(session, fake_hosting):
    session.start_app_server(lambda: None)
    host = session.app_server
    assert host.serve_microscope

    assert session.set_serve_microscope(False) is True

    assert not host.running and session.app_server is None
    session.start_app_server(lambda: None)
    assert not session.app_server.serve_microscope


def test_a_new_microscope_keeps_the_server(session, microscope, fake_hosting):
    """The server reads the session's microscope and stream at each call, so
    another microscope needs no restart."""
    session.microscope = microscope
    session.start_recording()
    session.start_app_server(lambda: None)
    first = session.event_buffer
    session.microscope = object()

    session.start_recording()

    assert session.app_server.running
    assert session.event_buffer is not first


def test_connect_and_disconnect_are_the_sessions(monkeypatch):
    """Connect sets the microscope and its settings and records its events,
    heard once on ``changed``; disconnect lets it go, and a reconnect gets a
    new stream."""
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    session = AppSession()
    heard, steps = [], []
    session.changed.connect(lambda what, *_: heard.append(what))
    idle = session.event_buffer
    assert session.connection["connected"] is False

    microscope = session.connect(
        progress=steps.append, manufacturer="Demo", ip_address="localhost"
    )

    assert session.microscope is microscope and session.settings is not None
    assert heard == [MICROSCOPE] and steps
    assert session.recorder is not None and session.event_buffer is not idle
    assert session.connection["connected"] is True
    assert session.connection["configuration"]["manufacturer"] == "Demo"
    with pytest.raises(ValueError, match="disconnect it first"):
        session.connect(manufacturer="Demo", ip_address="localhost")
    first = session.recorder

    assert session.disconnect() is None
    assert session.microscope is None and session.settings is None
    assert session.recorder is None and session.event_buffer is idle
    assert heard == [MICROSCOPE, MICROSCOPE]

    session.connect(manufacturer="Demo", ip_address="localhost")
    assert session.recorder is not first
    session.disconnect()


def test_disconnect_is_refused_while_a_run_is_on(session, microscope, monkeypatch):
    session.microscope = microscope
    monkeypatch.setattr(type(session.runner), "is_running", property(lambda _: True))

    with pytest.raises(RuntimeError, match="run is on"):
        session.disconnect()
    assert session.microscope is microscope


def test_a_failed_connect_says_why(monkeypatch):
    from fibsem import utils

    def fail(**_):
        raise ConnectionError("no route to 10.0.0.1")

    monkeypatch.setattr(utils, "setup_session", fail)
    session = AppSession()

    with pytest.raises(ConnectionError):
        session.connect("/nowhere.yaml")
    assert session.connection["error"] == "no route to 10.0.0.1"
    assert session.microscope is None and session.connecting is None


def test_an_edit_runs_and_resolves_to_its_result():
    session = AppSession()

    assert session.run_edit(lambda: 42).result(timeout=1) == 42


def test_a_failing_edit_resolves_to_its_exception():
    session = AppSession()

    def fail():
        raise ValueError("no such lamella")

    with pytest.raises(ValueError, match="no such lamella"):
        session.run_edit(fail).result(timeout=1)


def test_edits_run_where_the_client_says():
    ran = []

    def executor(call):
        ran.append(call)
        return AppSession().run_edit(call)

    session = AppSession(executor=executor)

    assert session.run_edit(lambda: "done").result(timeout=1) == "done"
    assert len(ran) == 1


def test_letting_the_agent_act_with_nothing_open_makes_a_plain_experiment(
    session, microscope
):
    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.server import AuthConfig
    from fibsem.workflows.experiment import FibsemExperiment

    session.agent_auth = AuthConfig.generate()
    session.agent_auth.set_level(AgentLevel.WATCH)
    session.microscope = microscope
    session.start_recording()

    assert session.agent.set_level(AgentLevel.WATCH) is None
    assert session.plain_experiment is None, "watching records nothing of its own"
    assert session.agent.set_level(AgentLevel.DRIVE) is None
    assert session.agent_auth.level is AgentLevel.DRIVE
    plain = session.plain_experiment
    assert isinstance(plain, FibsemExperiment)
    assert session.recording_experiment() is plain, "made once"
    assert _recorded(session, plain), "the stream records to it"


def test_the_level_is_refused_when_there_is_nowhere_to_record(
    session, monkeypatch, tmp_path
):
    import fibsem.workflows.session as module
    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.server import AuthConfig

    blocked = tmp_path / "a-file"
    blocked.write_text("")
    monkeypatch.setattr(module, "SESSIONS_PATH", str(blocked))
    session.agent_auth = AuthConfig.generate()
    session.agent_auth.set_level(AgentLevel.WATCH)

    refused = session.agent.set_level(AgentLevel.ASSIST)
    assert refused and "Open or create an experiment" in refused
    assert session.agent_auth.level is AgentLevel.WATCH


def test_an_open_experiment_is_what_the_agent_records_to(session, experiment):
    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.server import AuthConfig

    session.experiment = experiment
    session.agent_auth = AuthConfig.generate()
    assert session.agent.set_level(AgentLevel.ASSIST) is None
    assert session.plain_experiment is None
    assert session.recording_experiment() is experiment


def test_who_can_answer_is_what_is_attached_and_the_agent_at_drive(fake_hosting):
    """A window attaches the operator; the agent can answer while the App
    Server serves it at Drive and it is listening. Nobody, with neither: a
    supervised question still waits, for someone to come back."""
    from fibsem.acting import AGENT, OPERATOR
    from fibsem.ai.agent.levels import AGENT_LISTENING_S, AgentLevel

    session = AppSession()
    assert session.deciders() == ()

    detach = session.attach_decider(OPERATOR)
    assert session.deciders() == (OPERATOR,)
    detach()
    assert session.deciders() == ()

    session.start_app_server(lambda: None)
    session.agent_auth = SimpleNamespace(level=AgentLevel.ASSIST)
    assert session.deciders() == (), "at Assist the agent suggests, it does not answer"
    session.agent_auth.level = AgentLevel.DRIVE
    assert session.deciders() == (), "at Drive, but never heard: nobody listening"
    session.app_server.seen = 3.0
    assert session.deciders() == (AGENT,)
    session.app_server.seen = AGENT_LISTENING_S + 1
    assert session.deciders() == (), "gone quiet: it waits for the agent to return"
    session.app_server.seen = 3.0
    session.stop_app_server()
    assert session.deciders() == (), "no server, nobody to hear it"


# ── the agent, as the operator meets it (FIB-1461) ────────────────────────────


class _Auth:
    """The agent's token and level, as AuthConfig keeps them."""

    def __init__(self, level):
        self.level = level

    def set_level(self, level):
        self.level = level

    def stop_agent(self):
        self.level = AgentLevel.WATCH


class _Store:
    """The agent's request store's surface the session uses."""

    def __init__(self, *requests):
        self.requests = list(requests)
        self.listeners = []
        self.decided = []

    def recent(self):
        return list(self.requests)

    def subscribe(self, listener):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def approve(self, request_id):
        self.decided.append(("approve", request_id))

    def reject(self, request_id, reason=""):
        self.decided.append(("reject", request_id, reason))


def _request(request_id, status):
    return SimpleNamespace(id=request_id, status=status)


def _served(session, level, store=None, stopped=None):
    """An App Server serving the agent at ``level``, with ``store``."""
    session.agent_auth = _Auth(level)
    session.app_server = SimpleNamespace(
        running=True,
        auth=session.agent_auth,
        requests=store,
        stop_agent=lambda microscope: (session.agent_auth.stop_agent(), stopped)[1],
    )


def _kinds(session, kind):
    events = session.recorder.buffer.events_since(0)["events"]
    return [(e["payload"], e["actor"]) for e in events if e["kind"] == kind]


def test_a_level_set_by_the_operator_is_on_the_record(session, microscope):
    session.microscope = microscope
    session.start_recording()
    _served(session, AgentLevel.WATCH)

    assert session.agent.set_level(AgentLevel.WATCH) is None
    assert session.agent.level is AgentLevel.WATCH
    assert _kinds(session, "agent_level_set") == [
        ({"previous": "watch", "level": "watch"}, "operator")
    ]


def test_the_brake_drops_to_watch_and_says_what_it_stopped(session, microscope):
    session.microscope = microscope
    session.start_recording()
    _served(session, AgentLevel.DRIVE, stopped="start_milling")

    assert session.agent.stop() == "start_milling"
    assert session.agent.level is AgentLevel.WATCH
    assert _kinds(session, "action") == [
        (
            {
                "what": "Stop the agent",
                "ok": True,
                "reason": "Stopped by the operator",
                "stops": "agent",
                "previous": "drive",
                "level": "watch",
                "stopped": "start_milling",
            },
            "operator",
        )
    ]


def test_with_no_server_there_is_no_level_and_no_brake(session):
    assert session.agent.level is None
    assert session.agent.stop() is None
    assert session.agent.set_level(AgentLevel.WATCH) == "The App Server isn't running."


def test_the_current_request_is_the_open_one_as_it_changes(session):
    store = _Store(_request("old", "applied"), _request("now", "pending"))
    _served(session, AgentLevel.ASSIST, store)
    heard = []

    assert session.agent.follow(lambda: heard.append(True)) is store
    assert session.agent.current().id == "now"

    store.requests[1] = _request("now", "rejected")
    store.listeners[0](store.requests[1])
    assert heard == [True]
    assert session.agent.current() is None


def test_following_moves_to_the_store_the_server_has_now(session):
    first, second = _Store(), _Store()
    _served(session, AgentLevel.ASSIST, first)
    session.agent.follow(lambda: None)
    session.app_server.requests = second

    assert session.agent.follow(lambda: None) is second
    assert first.listeners == [] and len(second.listeners) == 1
    session.app_server.running = False
    assert session.agent.follow(lambda: None) is None
    assert second.listeners == []


def test_the_operator_decides_a_request(session):
    store = _Store(_request("r1", "pending"))
    _served(session, AgentLevel.ASSIST, store)

    session.agent.approve("r1")
    session.agent.reject("r1", reason="not that lamella")
    assert store.decided == [("approve", "r1"), ("reject", "r1", "not that lamella")]


def test_a_message_is_heard_only_while_something_records(session, microscope):
    assert session.agent.message("hold on") is None

    session.microscope = microscope
    session.start_recording()
    message_id = session.agent.message("hold on")
    assert _kinds(session, "note") == [
        ({"text": "hold on", "to": "agent", "id": message_id}, "operator")
    ]


# ── the run controller, heard with no window ──────────────────────────────────


def test_a_stop_is_said_whoever_asks(session):
    """The window draws a stop from the runner, the agent's as the operator's."""
    from tests.run_state import StubRunner

    session.runner = StubRunner(session)
    heard = []
    session.runner.stopping.connect(lambda: heard.append("stopping"))

    session.runner.stop()
    assert heard == [], "nothing was running"
    session.runner.is_running = True
    session.runner.stop()  # unnamed: the agent's
    assert heard == ["stopping"]


def test_an_operator_stop_is_recorded_by_the_runner(session, microscope):
    from tests.run_state import StubRunner

    session.microscope = microscope
    session.start_recording()
    session.runner = StubRunner(session, is_running=True)

    session.runner.stop(actor="operator", via="Stop")
    session.runner.stop_task(actor="operator", via="Stop Task", task="Trench (01)")
    session.runner.stop()  # unnamed: the agent's stop records itself
    # the operator's Stop buttons ask no reason (FIB-1495)
    why = {"reason": "Stopped by the operator"}
    run = {
        "what": "Stop the run",
        "ok": True,
        **why,
        "stops": "workflow",
        "via": "Stop",
    }
    task = {
        "what": "Stop the task",
        "ok": True,
        **why,
        "stops": "task",
        "via": "Stop Task",
    }
    assert _kinds(session, "action") == [
        (run, "operator"),
        ({**task, "task": "Trench (01)"}, "operator"),
    ]


def test_queue_changes_reach_the_runner_with_no_window(
    session, microscope, experiment, monkeypatch
):
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
    )
    from fibsem.workflows.runner import ITEMS, RunRequest
    from tests.run_state import hold_runs

    held = hold_runs(monkeypatch, ITEMS)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    add_planned(experiment.task_protocol, planned(name="Trench", required=True))
    session.microscope = microscope
    session.experiment = experiment
    heard = []
    session.runner.queue_changed.connect(heard.append)
    started = session.runner.start(
        experiment, RunRequest(kind=ITEMS, task_names=("Trench",))
    )
    try:
        assert started.started, started.reason
        held.managers[0].queue_changed.emit({"version": 3})
        assert heard == [{"version": 3}]
    finally:
        held.end(session.runner)
