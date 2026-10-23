"""The experiment replay over the App Server: the timeline the web page lists,
and the images and overlays at one recorded action."""

import os
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from tests.autolamella._replay_experiment import (  # noqa: E402
    MILLING_STAGES,
    SPOTS,
    record_demo_experiment,
)
from tests.run_state import stub_session  # noqa: E402

TOKEN = "test-token"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        root = record_demo_experiment(
            tmp_path_factory.mktemp("replay") / "exp", monkeypatch, record_events=True
        )
    finally:
        monkeypatch.undo()
    return root


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def session(recorded, microscope):
    experiment = Experiment(path=recorded.parent, name=recorded.name)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    session = stub_session()
    session.experiment = experiment
    session.microscope = microscope
    return session


@pytest.fixture
def client(microscope, session):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(session),
        auth=AuthConfig.generate(level="watch", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def _timeline(client):
    resp = client.get("/app/experiment/replay")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _view(client, index, **params):
    resp = client.get(f"/app/experiment/replay/{index}", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_timeline_lists_every_recorded_action_in_time_order(client):
    timeline = _timeline(client)
    assert timeline["available"] and timeline["source"] == "events.jsonl"
    events = timeline["events"]
    assert [e["index"] for e in events] == list(range(len(events)))
    assert [e["time"] for e in events] == sorted(e["time"] for e in events)
    assert sum(timeline["counts"].values()) == len(events)
    assert timeline["start"] == events[0]["time"]


def test_a_milling_scene_draws_every_stage_over_the_fib_image(client):
    events = _timeline(client)["events"]
    index = next(
        e["index"]
        for e in events
        if e["kind"] == "milling" and MILLING_STAGES[0] in e["summary"]
    )
    fib = _view(client, index)["fib"]
    assert [s["name"] for s in fib["milling"]] == list(MILLING_STAGES)
    assert [s["current"] for s in fib["milling"]] == [True, False]
    for stage in fib["milling"]:
        assert stage["outlines"]
        for ring in stage["outlines"]:
            for x, y in ring:
                assert -0.5 < x < 1.5 and -0.5 < y < 1.5
    # The image is read through the experiment's files route.
    resp = client.get(f"/app/experiment/files/{quote(fib['path'])}?width=256")
    assert resp.status_code == 200, resp.text


def test_spot_burns_are_drawn_where_they_were_burnt(client):
    events = _timeline(client)["events"]
    spots = [
        e["index"]
        for e in events
        if e["kind"] == "milling" and "spot" in e["summary"].lower()
    ]
    assert spots
    fib = _view(client, spots[-1])["fib"]
    drawn = [v for spot in fib["spots"] for v in spot]
    assert drawn == pytest.approx([v for p in SPOTS for v in (p.x, p.y)])


def test_a_scene_past_the_end_is_not_found(client):
    count = len(_timeline(client)["events"])
    resp = client.get(f"/app/experiment/replay/{count}")
    assert resp.status_code == 404, resp.text


def test_without_an_experiment_there_is_nothing_to_replay(client, session):
    session.experiment = None
    assert _timeline(client)["available"] is False
    assert _view(client, 0)["available"] is False
