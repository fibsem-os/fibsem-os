"""A `hardware.devices` entry with `driver: remote` builds a device served by a device
server at its `address` and `port` (``fibsem.drivers.remote.devices``), over
localhost.

The entry's name is the device's name on the server, and entries at one address
share one connection. A server that does not answer, or has no device of that name,
leaves the device out with a warning, or fails connect when the entry is required.
"""

import logging
import socket

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import yaml  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.entries import (  # noqa: E402
    DeviceBuildError,
    build_device_entries,
    resolve_device_entries,
)
from fibsem.drivers.registry import device_builder  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    RemoteBeam,
    RemoteCamera,
    RemoteObjective,
)
from fibsem.server.devices import (  # noqa: E402
    DeviceServer,
    demo_devices,
    demo_fm_devices,
)
from fibsem.structures import BeamType, DeviceEntry  # noqa: E402


@pytest.fixture
def server():
    server = DeviceServer([*demo_devices(), *demo_fm_devices()]).start()
    yield server
    server.stop()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build(*entries, microscope=None):
    configured = {e["name"]: DeviceEntry.from_dict(e) for e in entries}
    resolved = resolve_device_entries([], configured, "Demo")
    return build_device_entries(resolved, microscope)


def _close(built):
    for client in {device.client for device in built.values()}:
        client.close()


def test_the_remote_driver_builds_beams_and_the_fm_parts():
    for device_type in ("beam", "camera", "light_source", "filter_set", "objective"):
        assert device_builder("remote", device_type) is not None
    assert device_builder("Remote", "stage") is None


def test_remote_entries_build_the_devices_their_server_has(server):
    at = {"driver": "remote", "address": "127.0.0.1", "port": server.port}
    built = _build(
        {"name": "electron", **at},
        {"name": "camera", "type": "camera", **at},
        {"name": "objective", "type": "objective", **at},
    )
    try:
        assert list(built) == ["electron", "camera", "objective"]
        assert isinstance(built["electron"], RemoteBeam)
        assert built["electron"].beam_type is BeamType.ELECTRON
        assert isinstance(built["camera"], RemoteCamera)
        assert isinstance(built["objective"], RemoteObjective)
        # one connection for the one address
        assert len({device.client for device in built.values()}) == 1
        assert built["electron"].hfw.get_value() > 0
    finally:
        _close(built)


def test_a_device_the_server_does_not_have_is_left_out(server, caplog):
    at = {"driver": "remote", "address": "127.0.0.1", "port": server.port}
    with caplog.at_level(logging.WARNING):
        built = _build({"name": "ion", **at}, {"name": "cam2", "type": "camera", **at})
    try:
        assert list(built) == ["ion"]
        assert "has no device 'cam2'" in caplog.text
    finally:
        _close(built)


def test_a_server_that_does_not_answer_leaves_its_devices_out(caplog):
    at = {"driver": "remote", "address": "127.0.0.1", "port": _free_port()}
    with caplog.at_level(logging.WARNING):
        built = _build({"name": "electron", **at}, {"name": "ion", **at})
    assert built == {}
    assert "'electron' was not built" in caplog.text
    assert "'ion' was not built" in caplog.text


def test_a_required_device_on_a_server_that_does_not_answer_fails_connect():
    at = {"driver": "remote", "address": "127.0.0.1", "port": _free_port()}
    with pytest.raises(DeviceBuildError, match="'electron' was not built"):
        _build({"name": "electron", "required": True, **at})


def test_an_entry_without_an_address_is_left_out(caplog):
    with caplog.at_level(logging.WARNING):
        built = _build({"name": "electron", "driver": "remote"})
    assert built == {}
    assert "needs an `address` and a `port`" in caplog.text


def test_the_demo_takes_its_electron_beam_from_a_server(server, tmp_path):
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    (electron,) = [e for e in config["hardware"]["devices"] if e["name"] == "electron"]
    electron.update(driver="remote", address="127.0.0.1", port=server.port)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    try:
        assert isinstance(microscope.beams[BeamType.ELECTRON], RemoteBeam)
        assert not isinstance(microscope.beams[BeamType.ION], RemoteBeam)
    finally:
        microscope.beams[BeamType.ELECTRON].client.close()
