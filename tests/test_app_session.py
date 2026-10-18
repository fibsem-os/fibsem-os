"""The running application without a window: microscope, experiment, events.

A window is one client of an ``AppSession``; these drive it with none.
"""

import os
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import Experiment
from fibsem.workflows.event_recording import EVENTS_FILENAME, read_events, recorder_for
from fibsem.workflows.session import AppSession


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    microscope.disconnect()


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
    assert session.start_app_server(lambda buffer: None)
    host = session.app_server

    session.start_app_server(lambda buffer: None)

    assert session.app_server is host and host.starts == 2 and host.running


def test_serving_the_microscope_drops_a_running_server(session, fake_hosting):
    session.start_app_server(lambda buffer: None)
    host = session.app_server

    assert session.set_serve_microscope(True) is True

    assert not host.running and session.app_server is None
    session.start_app_server(lambda buffer: None)
    assert session.app_server.serve_microscope


def test_a_new_microscope_stops_the_server_on_the_old_stream(
    session, microscope, fake_hosting
):
    session.microscope = microscope
    session.start_recording()
    session.start_app_server(lambda buffer: None)
    session.microscope = object()

    session.start_recording()

    assert not session.app_server.running


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
