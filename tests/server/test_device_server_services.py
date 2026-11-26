"""A Device Server serves the services its devices' drivers build over them, with
the same routes as a device, and a client uses them as its own."""

from typing import Any, Mapping, Optional

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.beam import Beam
from fibsem.devices.core import Role, SafetyHalted, command, query
from fibsem.devices.entries import build_device_entries, resolve_device_entries
from fibsem.drivers.demo.microscope import DemoMicroscope
from fibsem.drivers.registry import REMOTE_SERVICES, ServiceBuilder
from fibsem.drivers.remote.devices import DeviceClient
from fibsem.drivers.remote.services import connect_remote_services
from fibsem.server import devices as server_devices
from fibsem.server.devices import DeviceServer, demo_devices, driver_services
from fibsem.services.core import Service
from fibsem.structures.settings import BeamType
from fibsem.structures.system import DeviceEntry


class LineScan(Service):
    """A stand-in for work that uses only this computer's devices."""

    beam = Role(Beam, doc="The beam it scans with.")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="line_scan", **kwargs)
        self.lines = 0

    @command
    def scan(self, lines: int = 1) -> int:
        self.lines += lines
        return self.lines

    @command(safety=True)
    def park(self) -> int:
        self.parked = True
        return self.lines

    @query
    def scanned(self) -> int:
        return self.lines


def build_line_scan(devices: Mapping[str, Any]) -> Optional[LineScan]:
    beam = devices.get("electron")
    if beam is None:
        return None
    service = LineScan()
    service.fill_roles(beam=beam)
    return service.connect()


BUILDERS = {"line_scan": ServiceBuilder(f"{__name__}:build_line_scan")}
DEMO_FILE = {
    "info": {"manufacturer": "Demo"},
    "hardware": {"devices": [{"name": "electron", "type": "beam"}]},
}


@pytest.fixture
def devices():
    return {device.name: device for device in demo_devices()}


@pytest.fixture
def client(devices):
    service = build_line_scan(devices)
    server = DeviceServer(devices.values(), services=[service]).start()
    client = DeviceClient("127.0.0.1", server.port, events=False)
    yield client, service
    client.close()
    server.stop()


def test_a_service_is_listed_like_a_device(client):
    client, _ = client
    listed = client.request("GET", "services", 5)
    assert list(listed) == ["line_scan"]
    assert "scan" in listed["line_scan"]["commands"]
    assert "scanned" in listed["line_scan"]["queries"]


def test_a_client_runs_it_on_the_server(client):
    client, service = client
    remote = connect_remote_services(client)["line_scan"]
    assert remote.scan(lines=3) == 3
    assert service.lines == 3
    assert remote.scanned() == 3


def test_a_server_with_none_lists_none(devices):
    server = DeviceServer(devices.values()).start()
    client = DeviceClient("127.0.0.1", server.port, events=False)
    try:
        assert client.request("GET", "services", 5) == {}
    finally:
        client.close()
        server.stop()


def test_a_drivers_services_are_built_over_the_files_devices(monkeypatch, devices):
    asked = []

    def builders(driver):
        asked.append(driver)
        return BUILDERS

    monkeypatch.setattr(server_devices, "service_builders", builders)
    services = driver_services(DEMO_FILE, devices)
    assert asked == ["Demo"]
    assert list(services) == ["line_scan"]
    assert services["line_scan"].beam is devices["electron"]


def test_a_service_whose_devices_are_missing_is_left_out(monkeypatch):
    monkeypatch.setattr(server_devices, "service_builders", lambda driver: BUILDERS)
    assert driver_services(DEMO_FILE, {}) == {}


def test_a_builder_that_fails_is_left_out_with_a_warning(monkeypatch, devices, caplog):
    broken = {"broken": ServiceBuilder(f"{__name__}:no_such_function")}
    monkeypatch.setattr(server_devices, "service_builders", lambda driver: broken)
    assert driver_services(DEMO_FILE, devices) == {}
    assert "service 'broken' was not built" in caplog.text


def test_the_built_in_drivers_have_none_yet(devices):
    assert driver_services(DEMO_FILE, devices) == {}


# -- the App Server's side: a remote entry brings its server's services along --


def _remote_entry(server):
    return DeviceEntry.from_dict(
        {
            "name": "electron",
            "driver": "remote",
            "address": "127.0.0.1",
            "port": server.port,
        }
    )


@pytest.fixture
def scanning_server(devices):
    service = build_line_scan(devices)
    server = DeviceServer(devices.values(), services=[service]).start()
    yield server, service
    server.stop()


class _Microscope:
    """What a build needs of the microscope it is for."""

    def __init__(self, own=()):
        self.taken, self.own = {}, set(own)

    def add_service(self, service):
        if service.name in self.own or service.name in self.taken:
            return False
        self.taken[service.name] = service
        return True


def _build_remote(server, microscope=None, shared=None):
    resolved = resolve_device_entries([], {"electron": _remote_entry(server)}, "Demo")
    return build_device_entries(resolved, microscope, shared=shared)


def _close(built):
    for client in {device.client for device in built.values()}:
        client.close()


def test_a_remote_entry_brings_its_servers_services(scanning_server):
    server, service = scanning_server
    microscope = _Microscope()
    built = _build_remote(server, microscope)
    try:
        assert list(microscope.taken) == ["line_scan"]
        assert microscope.taken["line_scan"].scan(lines=2) == 2
        assert service.lines == 2
        assert microscope.taken["line_scan"].parent is microscope
    finally:
        _close(built)


def test_a_name_the_microscope_has_already_is_left_out(scanning_server, caplog):
    server, _ = scanning_server
    microscope = _Microscope(own=["line_scan"])
    built = _build_remote(server, microscope)
    try:
        assert microscope.taken == {}
        assert "'line_scan' is left out" in caplog.text
    finally:
        _close(built)


def test_with_no_microscope_they_are_kept_for_the_instrument(scanning_server):
    server, _ = scanning_server
    shared = {}
    built = _build_remote(server, shared=shared)
    try:
        assert list(shared[REMOTE_SERVICES]) == ["line_scan"]
    finally:
        _close(built)


def test_a_device_instrument_serves_its_device_servers_services(scanning_server):
    server, service = scanning_server
    instrument = server_devices.device_instrument(
        {
            "hardware": {
                "devices": [
                    {
                        "name": "electron",
                        "type": "beam",
                        "driver": "remote",
                        "address": "127.0.0.1",
                        "port": server.port,
                    }
                ]
            }
        },
        allow_remote=True,
    )
    try:
        assert list(instrument.services) == ["line_scan"]
        assert instrument.services["line_scan"].scan() == 1
        assert service.lines == 1
    finally:
        instrument.disconnect()


def test_the_microscope_lists_them_after_its_own():
    microscope = DemoMicroscope.__new__(DemoMicroscope)
    assert microscope.add_service(LineScan())
    assert not microscope.add_service(LineScan())  # the name is taken
    milling = Service(name="milling")
    assert not microscope.add_service(milling)  # one of the microscope's own
    assert "line_scan" in microscope.services


def test_a_connected_microscope_serves_them_as_its_own(scanning_server, tmp_path):
    server, service = scanning_server
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    (electron,) = [e for e in config["hardware"]["devices"] if e["name"] == "electron"]
    electron.update(driver="remote", address="127.0.0.1", port=server.port)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    try:
        assert "milling" in microscope.services
        assert microscope.services["line_scan"].scan(lines=4) == 4
        assert service.lines == 4
    finally:
        microscope.beams[BeamType.ELECTRON].client.close()


# -- the App Server's Safety Halt reaches what it uses on a Device Server --


@pytest.fixture
def halted(scanning_server):
    """A microscope in a Safety Halt, with the server's beam and service built for it."""
    server, service = scanning_server
    microscope = DemoMicroscope.__new__(DemoMicroscope)
    built = _build_remote(server, microscope)
    microscope.resources.safety_halt.set("a test")
    yield microscope, built["electron"], service
    _close(built)


def test_a_remote_device_refuses_during_the_microscopes_halt(halted, devices):
    microscope, beam, _ = halted
    before = devices["electron"].working_distance.get_value()
    with pytest.raises(SafetyHalted):
        beam.working_distance.set_value(before + 1e-3)
    with pytest.raises(SafetyHalted):
        beam.turn_off()
    with pytest.raises(SafetyHalted):
        beam.call_command("turn_off")
    assert devices["electron"].working_distance.get_value() == before
    assert beam.working_distance.get_value() == before  # reads still work


def test_a_remote_device_keeps_its_own_locks(halted):
    microscope, beam, _ = halted
    assert beam.resources is not microscope.resources
    assert beam.safety_halt is microscope.resources.safety_halt


def test_a_device_servers_service_refuses_during_the_halt(halted):
    microscope, _, service = halted
    remote = microscope.services["line_scan"]
    with pytest.raises(SafetyHalted):
        remote.scan(lines=1)
    with pytest.raises(SafetyHalted):
        remote.call_command("scan", lines=1)
    assert service.lines == 0  # nothing was sent
    assert remote.scanned() == 0  # a query is a read


def test_a_safety_command_the_server_lists_still_runs(halted):
    microscope, _, service = halted
    microscope.services["line_scan"].park()
    assert service.parked


def test_they_run_again_once_the_halt_is_cleared(halted):
    microscope, beam, service = halted
    microscope.resources.safety_halt.clear()
    assert microscope.services["line_scan"].scan(lines=2) == 2
    beam.turn_off()
