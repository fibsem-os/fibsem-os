"""Moving a lamella's point of interest to a saved correlation over the App
Server: the correlation dialog's Continue, at Drive, through the item patch,
with no window."""

import pytest
from fastapi.testclient import TestClient
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.correlation.history import new_run_dir, save_run
from fibsem.correlation.structures import (
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
)
from fibsem.server import AuthConfig, build_server
from fibsem.structures.settings import MicroscopeState, Point
from fibsem.workflows.events import EventBuffer
from tests.run_state import stub_session

TOKEN = "test-token"
TARGET = Point(3e-6, -2e-6)


@pytest.fixture
def session(tmp_path, microscope):
    (tmp_path / "exp").mkdir()
    experiment = Experiment(path=tmp_path / "exp", name="correlation-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    session = stub_session()
    session.experiment = experiment
    session.microscope = microscope
    return session


@pytest.fixture
def lamella(session):
    lamella = session.experiment.positions[0]
    lamella.poi = Point(0.0, 0.0)
    return lamella


@pytest.fixture
def run_path(lamella):
    path = new_run_dir(str(lamella.path))
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(px_m=TARGET)], rms_error=0.5
    )
    save_run(path, CorrelationState(result=result))
    return path


@pytest.fixture
def events():
    return EventBuffer()


def _client(microscope, session, events, level):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(session, event_buffer=events),
        auth=AuthConfig.generate(level=level, token=TOKEN),
    )
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    return client


def _apply(client, name, run_path):
    return client.post(
        f"/app/experiment/items/{name}/correlation", json={"run_path": run_path}
    )


def test_the_poi_moves_to_the_runs_target_and_is_recorded(
    microscope, session, lamella, run_path, events
):
    with _client(microscope, session, events, "drive") as client:
        resp = _apply(client, lamella.name, run_path)

    assert resp.status_code == 200, resp.text
    assert resp.json()["run_path"] == run_path
    assert (lamella.poi.x, lamella.poi.y) == (TARGET.x, TARGET.y)
    recorded = [
        e for e in events.events_since(0)["events"] if e["kind"] == "correlation"
    ]
    assert len(recorded) == 1
    assert recorded[0]["payload"]["item"]["name"] == lamella.name
    assert recorded[0]["payload"]["poi"] == TARGET.to_dict()


def test_it_needs_drive(microscope, session, lamella, run_path, events):
    with _client(microscope, session, events, "watch") as client:
        assert _apply(client, lamella.name, run_path).status_code == 403
    assert (lamella.poi.x, lamella.poi.y) == (0.0, 0.0)


def test_a_run_outside_the_lamellas_correlation_folder_is_refused(
    microscope, session, lamella, events, tmp_path
):
    with _client(microscope, session, events, "drive") as client:
        resp = _apply(client, lamella.name, str(tmp_path))
    assert resp.status_code == 422, resp.text
    assert (lamella.poi.x, lamella.poi.y) == (0.0, 0.0)


def test_an_unknown_item_and_a_missing_run_are_refused(
    microscope, session, lamella, events
):
    with _client(microscope, session, events, "drive") as client:
        assert _apply(client, "nope", "x").status_code == 404
        missing = new_run_dir(str(lamella.path))
        assert _apply(client, lamella.name, missing).status_code == 404
        resp = client.post(f"/app/experiment/items/{lamella.name}/correlation", json={})
        assert resp.status_code == 422
    assert (lamella.poi.x, lamella.poi.y) == (0.0, 0.0)
