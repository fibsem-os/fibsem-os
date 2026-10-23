"""Experiments and protocols for an agent at Drive: create, load, load a
protocol, kept to the default experiments folder, the recents and the protocol
folder."""

import os
import shutil
from pathlib import Path

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
    monkeypatch.setattr(fibsem_cfg, "DEFAULT_EXPERIMENTS_PATH", str(experiments))
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
    host = stub_session()
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


def test_a_recent_experiment_is_loaded(driver, host):
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


def test_an_experiment_is_created_and_found_anywhere_under_the_folder(
    driver, host, places
):
    made = driver.post("/app/experiment/create", json={"name": "2026-10-20/grid-a"})
    assert made.status_code == 200, made.text
    assert host.experiment.name == "grid-a"
    assert os.path.isfile(places / "2026-10-20" / "grid-a" / "experiment.yaml")
    found = driver.get("/app/experiment/all").json()
    assert found["folder"] == str(places)
    assert [e["name"] for e in found["items"]] == ["grid-a"]
    assert found["items"][0]["relative_path"] == "2026-10-20/grid-a"

    for name in ("../out", "/tmp/abs", "a//b", "a/./b", "a/b/c/d"):
        refused = driver.post("/app/experiment/create", json={"name": name})
        assert refused.status_code == 422, name
    os.symlink(places.parent, places / "link")
    linked = driver.post("/app/experiment/create", json={"name": "link/out"})
    assert linked.status_code == 422


def test_the_listing_narrows_and_names_each_as_a_load_takes_it(driver, host):
    driver.post("/app/experiment/create", json={"name": "lab/first"})
    listed = driver.get("/app/experiment/all").json()["items"]
    assert [e["relative_path"] for e in listed] == ["lab/first"]
    assert {"operator", "instrument_serial", "instrument_model"} <= set(listed[0])
    assert driver.post("/app/experiment/load", json={"path": "lab/first"}).is_success

    assert (
        driver.get("/app/experiment/all", params={"since": "2999-01-01"}).json()[
            "items"
        ]
        == []
    )
    assert (
        len(
            driver.get("/app/experiment/all", params={"until": "2999-01-01"}).json()[
                "items"
            ]
        )
        == 1
    )
    assert (
        driver.get("/app/experiment/all", params={"operator": "nobody-here"}).json()[
            "items"
        ]
        == []
    )
    bad = driver.get("/app/experiment/all", params={"since": "last tuesday"})
    assert bad.status_code == 422


def test_an_experiment_under_the_folder_is_loaded(driver, host, places):
    driver.post("/app/experiment/create", json={"name": "lab/first"})
    driver.post("/app/experiment/create", json={"name": "second"})
    # off the recents list, but under the folder: by its folder, relative
    from fibsem.config import load_user_preferences, save_user_preferences

    prefs = load_user_preferences()
    prefs.experiment.recent_experiments = []
    save_user_preferences(prefs)
    loaded = driver.post("/app/experiment/load", json={"path": "lab/first"})
    assert loaded.status_code == 200, loaded.text
    assert host.experiment.name == "first"
    absolute = str(places / "second" / "experiment.yaml")
    assert driver.post("/app/experiment/load", json={"path": absolute}).is_success
    assert host.experiment.name == "second"
    for path in ("lab", "../elsewhere", "lab/first/protocol.yaml"):
        refused = driver.post("/app/experiment/load", json={"path": path})
        assert refused.status_code == 404, path


def test_the_session_names_its_own_experiments_folder(driver, host, tmp_path):
    own = tmp_path / "own"
    own.mkdir()
    host.experiments_folder = str(own)
    assert driver.post("/app/experiment/create", json={"name": "x"}).is_success
    assert os.path.isfile(own / "x" / "experiment.yaml")
    assert driver.get("/app/experiment/all").json()["folder"] == str(own)


def test_a_protocol_is_loaded_into_the_open_experiment(driver, host):
    driver.post("/app/experiment/create", json={"name": "with-protocol"})
    resp = driver.post("/app/experiment/protocol/load", json={"protocol": "other.yaml"})
    assert resp.status_code == 200, resp.text
    assert os.path.isfile(os.path.join(host.experiment.path, "protocol.yaml"))


def test_nothing_changes_experiment_while_a_run_is_on(driver, host):
    driver.post("/app/experiment/create", json={"name": "running"})
    host.runner.is_running = True
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


@pytest.fixture
def reads(host, places):
    """A client at Watch over a context whose record is kept in a list."""
    context = AutoLamellaAppContext(host)
    recorded = []
    context.record_event = lambda kind, payload, actor=None: recorded.append(
        (kind, payload)
    )
    app = build_server(
        host.microscope,
        app_context=context,
        auth=AuthConfig.generate(level="watch", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client, recorded


def _two_experiments(driver):
    driver.post("/app/experiment/create", json={"name": "lab/first"})
    driver.post("/app/experiment/create", json={"name": "second"})
    return next(
        e["path"]
        for e in driver.get("/app/experiment/all").json()["items"]
        if e["name"] == "first"
    )


def test_a_file_is_fetched_raw_and_the_fetch_is_recorded(driver, host, reads):
    _two_experiments(driver)
    client, recorded = reads
    (Path(host.experiment.path) / "data").mkdir()
    (Path(host.experiment.path) / "data" / "raw.bin").write_bytes(b"\x00\x01\xff")
    resp = client.get("/app/experiment/files/data/raw.bin", params={"raw": True})
    assert resp.status_code == 200, resp.text
    assert resp.content == b"\x00\x01\xff"
    assert resp.headers["X-Experiment"] == "second"
    assert resp.headers["X-Experiment-Path"] == "data/raw.bin"
    assert recorded == [
        (
            "experiment_file_read",
            {"path": "data/raw.bin", "raw": True},
        )
    ]
    # A preview read of the open experiment is not recorded.
    client.get("/app/experiment/files/experiment.yaml")
    assert len(recorded) == 1
    # The event log's read-only token sees previews, never the files themselves.
    dashboard = {"Authorization": f"Bearer {client.app.state.auth.dashboard_token}"}
    looked = client.get("/app/experiment/files/experiment.yaml", headers=dashboard)
    assert looked.status_code == 200
    fetched = client.get(
        "/app/experiment/files/data/raw.bin", params={"raw": True}, headers=dashboard
    )
    assert fetched.status_code == 403
    assert len(recorded) == 1
    folder = client.get("/app/experiment/files/data", params={"raw": True})
    assert folder.status_code == 422
    for path in ("..%2Felsewhere", "/etc/hostname"):
        refused = client.get(f"/app/experiment/files/{path}", params={"raw": True})
        assert refused.status_code in (403, 404), path


def test_a_past_experiment_is_read_by_its_path_and_every_read_recorded(
    driver, host, places, reads
):
    first = _two_experiments(driver)
    assert host.experiment.name == "second"
    client, recorded = reads
    relative = next(
        e["relative_path"]
        for e in client.get("/app/experiment/all").json()["items"]
        if e["name"] == "first"
    )
    assert relative == "lab/first"
    for name in (relative, first, os.path.dirname(first)):
        listed = client.get("/app/experiment/files", params={"experiment": name})
        assert listed.status_code == 200, (name, listed.text)
        assert "experiment.yaml" in [e["name"] for e in listed.json()["entries"]]
    text = client.get(
        "/app/experiment/files/experiment.yaml", params={"experiment": first}
    ).json()
    assert "first" in text["text"]
    raw = client.get(
        "/app/experiment/files/experiment.yaml",
        params={"experiment": first, "raw": True},
    )
    assert raw.headers["X-Experiment"] == "first"
    assert [kind for kind, _ in recorded] == ["experiment_file_read"] * 5
    assert recorded[-1][1] == {
        "path": "experiment.yaml",
        "raw": True,
        "experiment": first,
    }
    # Nothing that is not an experiment the app knows.
    (places / "loose").mkdir()
    for name in ("/etc", "..", "loose", str(places.parent)):
        refused = client.get("/app/experiment/files", params={"experiment": name})
        assert refused.status_code == 404, name
    # Its files stay inside it.
    escape = client.get(
        "/app/experiment/files/..%2F..%2Fsecond%2Fexperiment.yaml",
        params={"experiment": first},
    )
    assert escape.status_code == 403


def test_the_sidecar_fetches_a_file_into_the_working_folder(
    driver, host, reads, tmp_path, monkeypatch
):
    pytest.importorskip("mcp")
    import asyncio
    import json

    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    first = _two_experiments(driver)
    client, recorded = reads
    work = tmp_path / "work"
    monkeypatch.setenv("FIBSEM_AGENT_WORKDIR", str(work))
    sidecar = build_sidecar(client, client.get("/capabilities").json())

    def fetch(arguments):
        result = asyncio.run(sidecar.call_tool("experiment_fetch_file", arguments))
        if isinstance(result, tuple):
            result = result[0]
        return json.loads(list(getattr(result, "content", result))[0].text)

    answer = fetch({"path": "experiment.yaml", "experiment": first})
    saved = work / "fetched" / "first" / "experiment.yaml"
    assert answer["saved"] == str(saved.resolve())
    assert saved.read_bytes() == (Path(first)).read_bytes()
    absolute = os.path.join(host.experiment.path, "protocol.yaml")
    answer = fetch({"path": absolute})
    assert answer["path"] == "protocol.yaml"
    assert (work / "fetched" / "second" / "protocol.yaml").is_file()
