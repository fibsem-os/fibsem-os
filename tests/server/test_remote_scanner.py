"""A scan generator on another computer, a plugin driver's device reached remotely, and
the devices a configuration file lists, served (``--config``).

``RemoteScanner`` scans on the device server and takes its limits from the served
device's facts. A served device names its driver and entry type, so the coordinator
makes it with the class that driver names for the type (``DriverEntry.remote_devices``)
when the driver is installed here too, and with the type's interface class otherwise,
saying what that loses.
"""

import logging

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.devices import Device, DeviceParameter  # noqa: E402
from fibsem.devices.entries import (  # noqa: E402
    ResolvedEntry,
    build_device_entries,
    disconnect_devices,
)
from fibsem.drivers import registry  # noqa: E402
from fibsem.drivers.demo.devices import DemoScanGenerator  # noqa: E402
from fibsem.drivers.registry import DriverEntry, register_driver  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    RemoteDevice,
    RemoteScanner,
    remote_class,
)
from fibsem.server.devices import (  # noqa: E402
    DeviceServer,
    configured_devices,
    describe_device,
)
from fibsem.structures import DeviceEntry, RangeLimit  # noqa: E402

DWELL = RangeLimit(min=100e-9, max=10e-6)


class LimitedScanner(DemoScanGenerator):
    def _dwell_time_limits(self):
        return DWELL

    def _resolution_limits(self):
        return (16, 16), (1024, 1024)


class Lockin(Device):
    """A device of a type fibsem has no interface for: a plugin's own."""

    frequency = DeviceParameter(float, unit="Hz")

    def read_frequency(self) -> float:
        return 1e3


class RemoteLockin(RemoteDevice, Lockin):
    pass


def _served(device, driver, entry_type):
    device.driver, device.entry_type = driver, entry_type
    return device.connect()


@pytest.fixture
def server():
    server = DeviceServer(
        [
            _served(LimitedScanner("scanner"), "Demo", "scan_generator"),
            _served(Lockin(name="lockin"), "TestScan", "lock_in"),
        ]
    ).start()
    yield server
    server.stop()


@pytest.fixture
def test_driver():
    saved = dict(registry.DRIVER_PLUGINS.registered)
    register_driver(
        DriverEntry(
            "TestScan",
            remote_devices={"lock_in": f"{__name__}:RemoteLockin"},
        )
    )
    yield
    registry.DRIVER_PLUGINS.registered.clear()
    registry.DRIVER_PLUGINS.registered.update(saved)


def _build(server, name, type_):
    entry = DeviceEntry.from_dict(
        {
            "name": name,
            "type": type_,
            "driver": "remote",
            "address": "127.0.0.1",
            "port": server.port,
        }
    )
    return build_device_entries([ResolvedEntry(entry, "remote")], None).get(name)


def test_a_served_device_names_its_driver_and_type():
    scanner = _served(LimitedScanner("scanner"), "Demo", "scan_generator")
    described = describe_device(scanner)
    assert (described["driver"], described["type"]) == ("Demo", "scan_generator")
    assert described["facts"]["dwell_time_limits"] == [DWELL.min, DWELL.max]
    assert described["facts"]["resolution_limits"] == [[16, 16], [1024, 1024]]


def test_a_remote_scanner_scans_on_the_server_within_its_limits(server):
    scanner = _build(server, "scanner", "scan_generator")
    try:
        assert type(scanner) is RemoteScanner
        assert scanner.dwell_time_limits == DWELL
        assert scanner.resolution_limits == ((16, 16), (1024, 1024))
        frame = scanner.acquire((32, 24), 1e-6)
        assert isinstance(frame, np.ndarray) and frame.shape == (24, 32)
        # refused here, before any request, as a local one would
        with pytest.raises(ValueError, match="dwell time"):
            scanner.acquire((32, 24), 1e-9)
        assert scanner.facts()["dwell_time_limits"] == [DWELL.min, DWELL.max]
    finally:
        scanner.client.close()


def test_the_drivers_remote_class_is_used_when_it_is_installed(server, test_driver):
    lockin = _build(server, "lockin", "lock_in")
    try:
        # a type fibsem has no interface for, reached with the plugin's class
        assert type(lockin) is RemoteLockin
        assert lockin.frequency.get_value() == 1e3
    finally:
        lockin.client.close()


def test_a_driver_not_installed_here_falls_back_to_the_interface(caplog):
    logging.getLogger().addHandler(caplog.handler)
    described = {
        "name": "scanner",
        "driver": "NotInstalledHere",
        "type": "scan_generator",
        "parameters": {"gain": {}},
    }
    assert remote_class("scan_generator", described) is RemoteScanner
    assert "driver 'NotInstalledHere' is not installed here" in caplog.text
    assert "Not available here: gain." in caplog.text


def test_a_plugin_type_with_no_remote_class_is_not_built(server, caplog):
    logging.getLogger().addHandler(caplog.handler)
    assert _build(server, "lockin", "lock_in") is None
    assert "there is no remote 'lock_in' device" in caplog.text


def test_an_entry_of_another_type_than_the_served_device_is_refused(server, caplog):
    logging.getLogger().addHandler(caplog.handler)
    assert _build(server, "scanner", "camera") is None
    assert "'scanner' is a scan_generator on the device server, not a camera" in (
        caplog.text
    )


# -- serving what a configuration file lists ------------------------------------


def _devices_file(tmp_path, *entries, **settings):
    path = tmp_path / "devices.yaml"
    path.write_text(
        yaml.safe_dump({**settings, "hardware": {"devices": list(entries)}})
    )
    return str(path)


def test_a_config_file_builds_each_device_with_its_own_driver(tmp_path):
    path = _devices_file(
        tmp_path, {"name": "unit", "type": "scan_generator", "driver": "Demo"}
    )
    built, cleanups = configured_devices(path)
    unit = built["unit"]
    assert isinstance(unit, DemoScanGenerator)
    assert (unit.driver, unit.entry_type) == ("Demo", "scan_generator")
    assert unit.parent is None  # no microscope on the unit's PC
    assert describe_device(unit)["type"] == "scan_generator"
    disconnect_devices(built, cleanups)


def test_a_config_file_entry_names_its_driver(tmp_path):
    path = _devices_file(tmp_path, {"name": "unit", "type": "scan_generator"})
    with pytest.raises(ValueError, match="no driver"):
        configured_devices(path)
    # unless the file names one for all of them
    path = _devices_file(
        tmp_path,
        {"name": "unit", "type": "scan_generator"},
        info={"manufacturer": "Demo"},
    )
    assert configured_devices(path)[0]["unit"].driver == "Demo"


def test_the_command_line_stops_on_a_file_it_cant_build(tmp_path):
    from fibsem.server.devices import main

    path = _devices_file(tmp_path, {"name": "unit", "type": "scan_generator"})
    with pytest.raises(SystemExit):
        main(["--config", path])
