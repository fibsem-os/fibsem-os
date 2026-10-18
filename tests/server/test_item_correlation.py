"""Moving a lamella's point of interest to a saved correlation over the App
Server: the correlation dialog's Continue, at Drive, through the item patch."""

import os
from concurrent.futures import Future

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.server.context import (  # noqa: E402
    item_fields_version,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.correlation.history import new_run_dir, save_run  # noqa: E402
from fibsem.correlation.structures import (  # noqa: E402
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState, Point  # noqa: E402
from fibsem.workflows.events import EventBuffer  # noqa: E402
from tests.run_state import stub_session

TOKEN = "test-token"
TARGET = Point(3e-6, -2e-6)


class Window:
    """The GUI's item patch, recorded: the move itself is the item patch's,
    tested with the GUI."""

    def __init__(self, session):
        self.session = session
        self.patches = []

    def request_apply_item_patch(self, item_name, patch, version):
        self.patches.append((item_name, patch, version))
        outcome = Future()
        outcome.set_result({"applied": True, "item_name": item_name})
        return outcome


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def host(tmp_path, microscope):
    (tmp_path / "exp").mkdir()
    experiment = Experiment(path=tmp_path / "exp", name="correlation-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    session = stub_session()
    session.experiment = experiment
    session.microscope = microscope
    return Window(session)


@pytest.fixture
def lamella(host):
    return host.session.experiment.positions[0]


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


def _client(microscope, host, events, level):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(
            host.session, window=host, event_buffer=events
        ),
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
    microscope, host, lamella, run_path, events
):
    with _client(microscope, host, events, "drive") as client:
        resp = _apply(client, lamella.name, run_path)

    assert resp.status_code == 200, resp.text
    assert resp.json()["run_path"] == run_path
    assert host.patches == [
        (
            lamella.name,
            {"poi.x": TARGET.x, "poi.y": TARGET.y},
            item_fields_version(lamella),
        )
    ]
    recorded = [
        e for e in events.events_since(0)["events"] if e["kind"] == "correlation"
    ]
    assert len(recorded) == 1
    assert recorded[0]["payload"]["item"]["name"] == lamella.name
    assert recorded[0]["payload"]["poi"] == TARGET.to_dict()


def test_it_needs_drive(microscope, host, lamella, run_path, events):
    with _client(microscope, host, events, "watch") as client:
        assert _apply(client, lamella.name, run_path).status_code == 403
    assert host.patches == []


def test_a_run_outside_the_lamellas_correlation_folder_is_refused(
    microscope, host, lamella, events, tmp_path
):
    with _client(microscope, host, events, "drive") as client:
        resp = _apply(client, lamella.name, str(tmp_path))
    assert resp.status_code == 422, resp.text
    assert host.patches == []


def test_an_unknown_item_and_a_missing_run_are_refused(
    microscope, host, lamella, events
):
    with _client(microscope, host, events, "drive") as client:
        assert _apply(client, "nope", "x").status_code == 404
        missing = new_run_dir(str(lamella.path))
        assert _apply(client, lamella.name, missing).status_code == 404
        resp = client.post(f"/app/experiment/items/{lamella.name}/correlation", json={})
        assert resp.status_code == 422
    assert host.patches == []
