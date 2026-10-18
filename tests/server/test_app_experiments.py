"""Experiments and protocols for an agent at Drive: create, load, load a
protocol, kept to the default experiments folder, the recents and the protocol
folder."""

import os
import shutil

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import config as fibsem_cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella import config as cfg  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from tests.run_state import stub_session

TOKEN = "test-token"
SHIPPED_PROTOCOL = str(cfg.TASK_PROTOCOL_PATH)


class Host:
    experiment = None
    microscope = None

    def __init__(self):
        self.session = stub_session()


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Isolated preferences, experiments folder and protocol folder."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    monkeypatch.setattr(cfg, "LOG_PATH", str(experiments))
    protocols = tmp_path / "protocols"
    protocols.mkdir()
    shutil.copy(SHIPPED_PROTOCOL, protocols / "standard.yaml")
    shutil.copy(SHIPPED_PROTOCOL, protocols / "other.yaml")
    monkeypatch.setattr(cfg, "TASK_PROTOCOL_PATH", str(protocols / "standard.yaml"))
    return experiments


@pytest.fixture
def host():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    host = Host()
    host.microscope = microscope
    return host


@pytest.fixture
def driver(host, places):
    app = build_server(
        host.microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def test_the_protocols_are_the_protocol_folder(driver):
    body = driver.get("/app/experiment/protocols").json()
    names = [p["name"] for p in body["protocols"]]
    assert names == ["other.yaml", "standard.yaml"]
    assert [p["name"] for p in body["protocols"] if p["default"]] == ["standard.yaml"]


def test_an_experiment_is_created_in_the_default_folder_and_opened(
    driver, host, places
):
    resp = driver.post(
        "/app/experiment/create", json={"name": "session-1", "protocol": "other.yaml"}
    )
    assert resp.status_code == 200, resp.text
    assert host.experiment is not None and host.experiment.name == "session-1"
    assert os.path.isfile(places / "session-1" / "protocol.yaml")
    recent = driver.get("/app/experiment/recent").json()["items"]
    assert recent[0]["name"] == "session-1"

    again = driver.post("/app/experiment/create", json={"name": "session-1"})
    assert again.status_code == 422  # never overwritten
    escape = driver.post("/app/experiment/create", json={"name": "../elsewhere"})
    assert escape.status_code == 422
    unknown = driver.post(
        "/app/experiment/create", json={"name": "s2", "protocol": "nope.yaml"}
    )
    assert unknown.status_code == 404
    assert "standard.yaml" in unknown.json()["detail"]["protocol_names"]


def test_only_a_recent_experiment_is_loaded(driver, host):
    driver.post("/app/experiment/create", json={"name": "first"})
    driver.post("/app/experiment/create", json={"name": "second"})
    assert host.experiment.name == "second"
    recent = driver.get("/app/experiment/recent").json()["items"]
    first = next(e["path"] for e in recent if e["name"] == "first")

    loaded = driver.post("/app/experiment/load", json={"path": first})
    assert loaded.status_code == 200, loaded.text
    assert host.experiment.name == "first"

    refused = driver.post("/app/experiment/load", json={"path": "/etc/experiment.yaml"})
    assert refused.status_code == 404
    assert first in refused.json()["detail"]["recent_paths"]


def test_a_protocol_is_loaded_into_the_open_experiment(driver, host):
    driver.post("/app/experiment/create", json={"name": "with-protocol"})
    resp = driver.post("/app/experiment/protocol/load", json={"protocol": "other.yaml"})
    assert resp.status_code == 200, resp.text
    assert os.path.isfile(os.path.join(host.experiment.path, "protocol.yaml"))


def test_nothing_changes_experiment_while_a_run_is_on(driver, host):
    driver.post("/app/experiment/create", json={"name": "running"})
    host.session.runner.is_running = True
    assert driver.post("/app/experiment/create", json={"name": "x"}).status_code == 409
    assert (
        driver.post(
            "/app/experiment/protocol/load", json={"protocol": "other.yaml"}
        ).status_code
        == 409
    )


def test_experiment_changes_need_drive(host, places):
    app = build_server(
        host.microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        assert client.get("/app/experiment/protocols").status_code == 200
        assert (
            client.post("/app/experiment/create", json={"name": "x"}).status_code == 403
        )
