"""``info.driver: remote``: a configuration whose microscope is another computer's
App Server. Connecting it builds a `RemoteMicroscope` with the token this
computer was paired with there, and this App Server serves it like a local one.
"""

import json
import os

import pytest
import yaml
from fastapi.testclient import TestClient

import fibsem.config as cfg
from fibsem.client import pairing
from fibsem.client.remote_microscope import RemoteMicroscope
from fibsem.client.transport import ServerUnreachable
from fibsem.drivers.registry import get_driver
from fibsem.server import AuthConfig, build_server
from fibsem.structures.settings import BeamType, ImageSettings
from fibsem.utils import configured_driver
from fibsem.workflows.session import AppSession
from tests.fixtures.app_server import serving

TOKEN = "support-pc-token"


@pytest.fixture
def support_pc(microscope):
    """The support PC: an App Server over the Demo microscope."""
    auth = AuthConfig.generate(level="drive", token=TOKEN)
    with serving(build_server(microscope, auth=auth)) as url:
        yield url


@pytest.fixture
def remote_configuration(support_pc, tmp_path):
    host, port = support_pc.rsplit("//", 1)[1].split(":")
    path = tmp_path / "remote.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "info": {
                    "name": "remote",
                    "driver": "remote",
                    "ip_address": host,
                    "port": int(port),
                },
            }
        )
    )
    return str(path)


@pytest.fixture
def paired(support_pc, tmp_path, monkeypatch):
    """This computer paired with the support PC (as `fibsem-cli pair` keeps it)."""
    monkeypatch.setattr(pairing, "REMOTE_DIR", tmp_path / "remote")
    path = pairing._file(support_pc)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"token": TOKEN}))


def test_a_configuration_names_its_driver(remote_configuration):
    assert configured_driver(remote_configuration) == "remote"
    # An older file names only the manufacturer: no driver of its own.
    shipped = os.path.join(cfg.SHIPPED_CONFIG_PATH, "tfs-hydra-configuration.yaml")
    assert configured_driver(shipped) is None
    example = os.path.join(
        cfg.SHIPPED_CONFIG_PATH, "remote-microscope-configuration.yaml"
    )
    assert configured_driver(example) == "remote"
    assert get_driver("remote").port == 8001


def test_the_session_connects_a_microscope_behind_another_app_server(
    remote_configuration, paired, microscope
):
    session = AppSession()
    try:
        remote = session.connect(remote_configuration)
        assert isinstance(remote, RemoteMicroscope)
        assert session.microscope is remote
        # The support PC's instrument, as it says it is.
        assert remote.system.info.model == microscope.system.info.model
        assert remote.get_stage_position() == microscope.get_stage_position()
        image = remote.acquire_image(
            ImageSettings(beam_type=BeamType.ELECTRON, resolution=(64, 48))
        )
        assert image.data.shape == (48, 64)

        # This computer's App Server serves it like a local microscope.
        app = build_server(session=session, auth=AuthConfig.generate(token="here"))
        client = TestClient(app)
        client.headers.update({"Authorization": "Bearer here"})
        assert set(client.get("/microscope/devices").json()) == set(microscope.devices)
        assert client.get("/microscope/state").status_code == 200
    finally:
        session.disconnect()


def test_without_a_pairing_it_says_how_to_pair(
    remote_configuration, tmp_path, monkeypatch
):
    monkeypatch.setattr(pairing, "REMOTE_DIR", tmp_path / "none")
    session = AppSession()
    with pytest.raises(ServerUnreachable, match="fibsem-cli pair"):
        session.connect(remote_configuration)
    assert session.microscope is None


def test_the_connection_names_the_instrument_behind_it(
    remote_configuration, paired, microscope
):
    session = AppSession()
    try:
        session.connect(remote_configuration)
        configuration = session.connection["configuration"]
        assert configuration["manufacturer"] == microscope.system.info.manufacturer
        assert configuration["model"] == microscope.system.info.model
        assert configuration["address"] == "127.0.0.1"
    finally:
        session.disconnect()


def test_the_connection_says_when_the_other_app_server_stops(
    microscope, tmp_path, monkeypatch
):
    import time

    auth = AuthConfig.generate(level="drive", token=TOKEN)
    session = AppSession()
    with serving(build_server(microscope, auth=auth)) as url:
        host, port = url.rsplit("//", 1)[1].split(":")
        monkeypatch.setattr(pairing, "REMOTE_DIR", tmp_path / "remote")
        token_file = pairing._file(url)
        token_file.parent.mkdir(parents=True)
        token_file.write_text(json.dumps({"token": TOKEN}))
        path = tmp_path / "remote.yaml"
        path.write_text(
            yaml.safe_dump(
                {"info": {"driver": "remote", "ip_address": host, "port": int(port)}}
            )
        )
        remote = session.connect(str(path))
        deadline = time.monotonic() + 10
        while not remote.reachable and time.monotonic() < deadline:
            time.sleep(0.05)
        assert session.connection["error"] is None
    try:
        deadline = time.monotonic() + 15
        while remote.reachable and time.monotonic() < deadline:
            time.sleep(0.05)
        state = session.connection
        assert state["connected"] is True
        assert "not answering" in state["error"]
    finally:
        session.disconnect()
