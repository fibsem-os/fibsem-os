"""Deployment B end to end: the FM served by a Device Server on its own PC, joined
into a host that drives the beams, as a METEOR runs with a Thermo support PC.

A Device Server on a thread, started from its file with the sim FM; the host pairs
by command, connects ``sim-iflm`` with a remote ``fm`` entry, travels to the FM and
back, and rides out the server going offline and coming back.
"""

import os
import socket
import time

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import fibsem.config as cfg  # noqa: E402
from fibsem import cli, utils  # noqa: E402
from fibsem.devices.entries import disconnect_devices  # noqa: E402
from fibsem.drivers.remote.devices import RemoteDeviceUnreachable  # noqa: E402
from fibsem.fm.remote import RemoteFluorescenceMicroscope  # noqa: E402
from fibsem.server.device_auth import DeviceAuth  # noqa: E402
from fibsem.server.devices import DeviceServer, configured_devices  # noqa: E402
from fibsem.server.pairing import PairedTokens, Pairing  # noqa: E402
from fibsem.structures.system import InsertableDeviceState  # noqa: E402

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(condition, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


@pytest.fixture
def meteor_pc(tmp_path, monkeypatch):
    """The METEOR PC: its Device Server's file, the devices it builds, and a server
    over them with its own token and pairing."""
    server_file = tmp_path / "device-server.yaml"
    server_file.write_text(
        yaml.safe_dump(
            {
                "info": {"manufacturer": "Demo"},
                "hardware": {"devices": [{"name": "fm", "type": "fm"}]},
            }
        )
    )
    built, cleanups = configured_devices(str(server_file))
    auth = DeviceAuth("meteor", Pairing(PairedTokens(tmp_path / "paired.json")))
    port = _free_port()
    servers = []

    def start():
        servers.append(DeviceServer(built.values(), port=port, auth=auth).start())
        return servers[-1]

    # The host is another computer: no Device Server token of its own to fall back on.
    monkeypatch.setattr(
        "fibsem.server.operator_token.DEVICE_TOKEN_FILE", tmp_path / "none.json"
    )
    start()
    yield built, auth, port, start, servers
    for server in servers:
        server.stop()
    disconnect_devices(built, cleanups)


def _host_config(tmp_path, port):
    """``sim-iflm`` with its FM named as remote: where the server is, and where the
    FM sits on the stage."""
    settings = utils.load_yaml(IFLM_CONFIG)
    utils.configuration_device(settings, "fm").update(
        driver="remote", address="127.0.0.1", port=port
    )
    path = tmp_path / "host-configuration.yaml"
    utils.save_yaml(path, settings)
    return str(path)


def test_deployment_b(meteor_pc, tmp_path):
    built, auth, port, start, servers = meteor_pc

    # Pair once, by command.
    with pytest.raises(SystemExit) as paired:
        cli.main(["pair-device", f"127.0.0.1:{port}", auth.pairing.open()])
    assert paired.value.code == 0

    microscope, _ = utils.setup_session(config_path=_host_config(tmp_path, port))
    fm = microscope.fm
    movement = microscope.stage_movement
    try:
        assert isinstance(fm, RemoteFluorescenceMicroscope)
        assert _wait_for(lambda: fm.online, timeout=10)
        far_objective = built["objective"]

        # To the FM and back: the far objective goes in at the FM, out to leave.
        microscope.move_to_orientation("FIB")
        movement.move_to_device("FM")
        assert movement.current_device() == "FM"
        assert far_objective.state.get_value() is InsertableDeviceState.INSERTED
        assert fm.acquire_image() is not None
        movement.move_to_device("FIBSEM")
        assert movement.current_device() == "FIBSEM"
        assert far_objective.state.get_value() is InsertableDeviceState.RETRACTED

        # The METEOR PC goes away: the FM's reads fail closed, the beams carry on.
        servers[-1].stop()
        assert _wait_for(lambda: not fm.client.connected, timeout=15)
        with pytest.raises(RemoteDeviceUnreachable):
            fm.objective.state
        microscope.get_stage_position()

        # It changes while away, then comes back: the host binds again and resyncs.
        built["light_source"].power.set_value(0.4)
        came_back = []
        fm.client.reconnected.connect(lambda: came_back.append(True))
        start()
        assert _wait_for(lambda: came_back and fm.client.connected, timeout=20)
        assert fm.devices["light_source"].power.cached == pytest.approx(0.4)
        assert fm.objective.state is InsertableDeviceState.RETRACTED
    finally:
        microscope.disconnect()
