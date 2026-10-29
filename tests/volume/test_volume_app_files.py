"""Volume experiments and protocols over the App Server, kept to the
experiments folder, the recents and the volume protocol folder."""

import pytest
from fastapi.testclient import TestClient

from fibsem import config as fibsem_cfg
from fibsem.applications.volume import server as volume_server
from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.server import VolumeAppContext
from fibsem.server import AuthConfig, build_server
from fibsem.workflows.events import EventBuffer
from fibsem.workflows.session import AppSession

TOKEN = "volume-token"
MILL = "Volume FIB Milling"


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """Isolated preferences, experiments folder and protocol folder."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    monkeypatch.setattr(volume_server, "PROTOCOL_FOLDER", str(tmp_path / "protocols"))
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    return experiments, tmp_path / "protocols"


@pytest.fixture
def session(microscope, folders):
    session = AppSession()
    session.microscope = microscope
    session.experiments_folder = str(folders[0])
    yield session
    session.runner.stop()
    session.runner.join(30)


@pytest.fixture
def agent(session, microscope):
    app = build_server(
        microscope,
        app_context=VolumeAppContext(session, event_buffer=EventBuffer()),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def _ok(answer):
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_an_experiment_is_created_with_the_default_protocol_and_opened(
    agent, session, folders
):
    _ok(agent.post("/app/experiment/create", json={"name": "vol-1"}))

    assert isinstance(session.experiment, VolumeExperiment)
    assert session.experiment.name == "vol-1"
    assert (folders[0] / "vol-1" / "protocol.yaml").is_file()
    recent = _ok(agent.get("/app/experiment/recent"))["items"]
    assert recent[0]["name"] == "vol-1"
    again = agent.post("/app/experiment/create", json={"name": "vol-1"})
    assert again.status_code == 422


def test_a_saved_protocol_is_listed_and_creates_and_loads(agent, session, folders):
    _ok(agent.post("/app/experiment/create", json={"name": "vol-1"}))
    _ok(
        agent.post(
            "/app/experiment/protocol/supervision",
            json={"task_name": MILL, "supervise": True},
        )
    )

    saved = _ok(agent.post("/app/experiment/protocol/save", json={"name": "careful"}))

    assert saved["name"] == "careful.yaml"
    listed = _ok(agent.get("/app/experiment/protocols"))["protocols"]
    assert [p["name"] for p in listed] == ["careful.yaml"]
    _ok(
        agent.post(
            "/app/experiment/create",
            json={"name": "vol-2", "protocol": "careful.yaml"},
        )
    )
    assert session.experiment.task_protocol.get_attention(MILL).value == "supervised"
    _ok(agent.post("/app/experiment/load", json={"path": "vol-1"}))
    assert session.experiment.name == "vol-1"
    _ok(agent.post("/app/experiment/protocol/load", json={"protocol": "careful.yaml"}))
    unknown = agent.post("/app/experiment/protocol/load", json={"protocol": "nope"})
    assert unknown.status_code == 404
