"""A server the gateway starts takes its secret from its environment, never
from disk, and while the gateway runs refuses every call without it, whoever
makes it: its /health and the stops excepted. The gateway hands a new one at
each start and adds it to every call it makes or passes on."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.gateway import child_secrets
from fibsem.server import gateway_secret
from fibsem.server.gateway_secret import ENV, HEADER, GatewayOnly

SECRET = "the-secret"
GATEWAY = {"url": "http://127.0.0.1:8010", "pid": 1}


def _app(secret=SECRET, from_gateway=True):
    app = FastAPI()
    app.state.auth = SimpleNamespace(gateway_secret=secret, from_gateway=from_gateway)
    app.add_middleware(GatewayOnly)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities():
        return {"level": "watch"}

    @app.post("/app/workflow/stop")
    def stop():
        return {"stopped": True}

    return TestClient(app)


@pytest.fixture
def gateway_runs(monkeypatch):
    running = {"file": GATEWAY}
    monkeypatch.setattr(gateway_secret, "read_gateway_file", lambda: running["file"])
    return running


def test_the_secret_is_taken_out_of_the_environment(monkeypatch):
    monkeypatch.setattr(gateway_secret, "_taken", [])
    monkeypatch.setenv(ENV, SECRET)
    assert gateway_secret.take() == SECRET
    import os

    assert ENV not in os.environ, "the programs it starts never see it"
    assert gateway_secret.take() == SECRET


def test_without_the_gateway_no_secret_is_taken(monkeypatch):
    monkeypatch.setattr(gateway_secret, "_taken", [])
    monkeypatch.delenv(ENV, raising=False)
    assert gateway_secret.take() is None


def test_a_call_without_the_secret_is_refused_whoever_makes_it(gateway_runs):
    client = _app()
    refused = client.get("/capabilities", headers={"Authorization": "Bearer op"})
    assert refused.status_code == 401
    assert refused.json()["detail"]["error_type"] == "through_gateway"
    assert "8010" in refused.json()["detail"]["message"]
    wrong = client.get("/capabilities", headers={HEADER: "not-it"})
    assert wrong.status_code == 401
    assert client.get("/capabilities", headers={HEADER: SECRET}).status_code == 200


def test_health_and_stops_are_never_refused(gateway_runs):
    client = _app()
    assert client.get("/health").status_code == 200
    assert client.post("/app/workflow/stop").status_code == 200


def test_with_no_gateway_running_the_server_still_answers(gateway_runs):
    gateway_runs["file"] = None
    assert _app().get("/capabilities").status_code == 200


def test_a_server_the_gateway_didnt_start_is_left_to_its_own_checks(gateway_runs):
    # Its own secret (in its discovery file) applies to agents' calls only,
    # in fibsem.server.auth.
    assert _app(from_gateway=False).get("/capabilities").status_code == 200
    assert _app(secret=None).get("/capabilities").status_code == 200


def test_the_gateway_hands_each_start_its_own_secret_and_sends_it(
    tmp_path, monkeypatch
):
    from fibsem.server import discovery

    app_file = tmp_path / "app-server.json"
    monkeypatch.setattr(discovery, "DISCOVERY_FILE", app_file)
    monkeypatch.setattr(discovery, "HOST_DISCOVERY_FILE", tmp_path / "host.json")
    first, second = child_secrets.make(), child_secrets.make()
    assert first != second
    assert child_secrets.environment(first, {"PATH": "x"}) == {"PATH": "x", ENV: first}
    found = {"url": "http://127.0.0.1:8001", "pid": 4242}
    monkeypatch.setattr(
        discovery,
        "read_discovery_file",
        lambda path=None: found if path in (None, app_file) else None,
    )
    assert child_secrets.headers("http://127.0.0.1:8001") == {}
    child_secrets.keep(4242, first)
    try:
        assert child_secrets.headers("http://127.0.0.1:8001/") == {HEADER: first}
        assert child_secrets.headers("http://127.0.0.1:9999") == {}
        # One the server wrote itself (opened by hand) is used when the gateway
        # handed none.
        assert child_secrets.for_server({"pid": 1, "gateway_secret": "own"}) == "own"
    finally:
        child_secrets.forget(4242)
    assert child_secrets.for_server(found) is None
