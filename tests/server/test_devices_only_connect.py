"""Connecting a configuration that lists only devices (``info.instrument: devices``).

The session builds the devices, local or remote, with no microscope; the App
Server serves them under ``/microscope/devices``; what needs a FIB/SEM says so;
disconnecting closes them.
"""

import socket
import time

import pytest
import yaml
from fastapi.testclient import TestClient

import fibsem.config as cfg
from fibsem import cli
from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
from fibsem.devices.entries import disconnect_devices
from fibsem.instrument import DeviceInstrument, configured_kind
from fibsem.server import AuthConfig, build_server
from fibsem.server.device_auth import DeviceAuth
from fibsem.server.devices import DeviceServer, configured_devices, served_fm_settings
from fibsem.server.pairing import PairedTokens, Pairing
from fibsem.workflows.session import AppSession

AGENT = "agent-token"
OPERATOR = "operator-token"
FM_PARTS = ["camera", "filter_set", "fm", "light_source", "objective"]


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def _write(path, devices):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "info": {"name": "cryo-fm", "instrument": "devices"},
                "hardware": {"devices": devices},
            }
        )
    )
    return str(path)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_fm(tmp_path, monkeypatch):
    """A devices configuration with the sim FM built on this PC, registered."""
    path = _write(
        tmp_path / "cryo-fm.yaml", served_fm_settings("fm")["hardware"]["devices"]
    )
    monkeypatch.setitem(cfg.USER_CONFIGURATIONS, "cryo-fm", {"path": path})
    return path


@pytest.fixture
def server():
    session = AppSession()
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token=AGENT)
    auth.operator_token = OPERATOR
    app = build_server(
        session=session,
        app_context=AutoLamellaAppContext(session),
        auth=auth,
        operator=session.agent,
    )
    with TestClient(app) as client:
        client.session = session
        yield client
    session.disconnect()


def _connected(client, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/microscope/connection", headers=_as(OPERATOR)).json()
        if state["run"] and state["run"]["state"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("the connect did not end")


def test_a_configuration_names_its_kind():
    assert configured_kind({"info": {"instrument": "devices"}}) == "devices"
    assert configured_kind({"info": {"manufacturer": "Demo"}}) == "fibsem"
    assert configured_kind({}) == "fibsem"


def test_the_session_connects_devices_with_no_microscope(local_fm):
    session = AppSession()
    instrument = session.connect(local_fm)
    try:
        assert isinstance(instrument, DeviceInstrument)
        assert session.instrument is instrument
        assert session.microscope is None
        assert sorted(instrument.devices) == FM_PARTS
        state = session.connection
        assert state["connected"] is True
        assert state["configuration"]["instrument"] == "devices"
        assert state["configuration"]["devices"] == FM_PARTS
        assert state["configuration"]["manufacturer"] is None
        assert session.recorder is not None
        with pytest.raises(ValueError, match="An instrument is connected"):
            session.connect(local_fm)
    finally:
        session.disconnect()
    assert session.instrument is None
    assert session.connection["connected"] is False
    assert session.recorder is None


def test_the_app_server_connects_and_serves_the_devices(server, local_fm):
    answer = server.post(
        "/microscope/connect", headers=_as(OPERATOR), json={"configuration": "cryo-fm"}
    )
    assert answer.status_code == 202, answer.text
    state = _connected(server)
    assert state["run"]["state"] == "done", state
    assert state["connected"] is True
    assert state["configuration"]["instrument"] == "devices"

    assert sorted(server.get("/microscope/devices", headers=_as(AGENT)).json()) == (
        FM_PARTS
    )
    capabilities = server.get("/capabilities", headers=_as(AGENT)).json()
    assert capabilities["instruments"] == ["devices"]
    power = server.get("/microscope/devices/light_source/power", headers=_as(AGENT))
    assert power.status_code == 200, power.text

    # What needs a FIB/SEM says so, and what is connected instead.
    state_route = server.get("/microscope/state", headers=_as(AGENT))
    assert state_route.status_code == 409
    assert "needs a FIB/SEM" in state_route.json()["detail"]["message"]
    started = server.session.runner.start(None, None)
    assert "needs a FIB/SEM" in started.reason

    again = server.post("/microscope/connect", headers=_as(OPERATOR), json={})
    assert again.status_code == 409

    assert (
        server.post("/microscope/disconnect", headers=_as(OPERATOR)).status_code == 200
    )
    assert server.session.instrument is None
    assert server.get("/microscope/devices", headers=_as(AGENT)).status_code == 409


@pytest.fixture
def fm_pc(tmp_path, monkeypatch):
    """The FM's own PC: a Device Server over the sim FM, with its pairing."""
    built, cleanups = configured_devices(served_fm_settings("fm"))
    auth = DeviceAuth("cryo-fm", Pairing(PairedTokens(tmp_path / "paired.json")))
    port = _free_port()
    monkeypatch.setattr(
        "fibsem.server.operator_token.DEVICE_TOKEN_FILE", tmp_path / "none.json"
    )
    server = DeviceServer(built.values(), port=port, auth=auth).start()
    yield built, auth, port
    server.stop()
    disconnect_devices(built, cleanups)


def test_a_remote_fm_connects_with_no_microscope(fm_pc, tmp_path):
    built, auth, port = fm_pc
    with pytest.raises(SystemExit) as paired:
        cli.main(["pair-device", f"127.0.0.1:{port}", auth.pairing.open()])
    assert paired.value.code == 0
    path = _write(
        tmp_path / "remote.yaml",
        [
            {
                "name": "fm",
                "type": "fm",
                "driver": "remote",
                "address": "127.0.0.1",
                "port": port,
            }
        ],
    )

    session = AppSession()
    instrument = session.connect(path)
    try:
        assert session.microscope is None
        assert "light_source" in instrument.devices
        instrument.devices["light_source"].power.set_value(0.3)
        assert built["light_source"].power.get_value() == pytest.approx(0.3)
    finally:
        session.disconnect()


def test_the_shipped_cryo_fm_configuration_lists_a_remote_fm():
    import os

    from fibsem.server.devices import read_server_settings
    from fibsem.structures.system import read_device_entries

    settings = read_server_settings(
        os.path.join(cfg.SHIPPED_CONFIG_PATH, "cryo-fm-remote-configuration.yaml")
    )
    assert configured_kind(settings) == "devices"
    entries = read_device_entries(settings)
    assert entries["fm"].driver == "remote"
