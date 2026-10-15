"""The fake scanner, end to end through fibsemOS's plugin path.

Each test is one step a plugin device goes through: discovered from its entry point,
configured as a ``hardware.devices`` entry, built, bound to a beam's ``scanner`` role,
driven through its parameters and commands, served by the App Server and the device
server, shown in the application, and recorded. A step fibsemOS does not support yet
is a strict ``xfail`` naming the gap, so fixing the gap turns the test red until the
mark comes off.

Run from the repository root with the plugin installed::

    pip install --no-deps -e examples/plugins/fibsem-fake-scanner
    QT_QPA_PLATFORM=offscreen python -m pytest examples/plugins/fibsem-fake-scanner/tests -q
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import yaml

pytest.importorskip("fibsem_fake_scanner")

from fibsem_fake_scanner.devices import FakeScanner, build_fake_scanner  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.entries import (  # noqa: E402
    DeviceBuildError,
    ResolvedEntry,
    build_device_entries,
)
from fibsem.drivers.registry import (  # noqa: E402
    BuildContext,
    get_driver,
    get_driver_plugin_records,
)
from fibsem.structures import BeamType, DeviceEntry, ImageSettings  # noqa: E402

ENTRY = {
    "name": "fake_scanner",
    "type": "scan_generator",
    "driver": "FakeScan",
    "pattern": "checker",
}
SETTINGS = dict(resolution=(64, 48), dwell_time=1e-7, hfw=80e-6)


@pytest.fixture(autouse=True)
def _no_sim_delay(monkeypatch):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")


def _config(*entries, roles=None, **beam_roles):
    """The Demo's configuration with *entries* added, and the electron beam's
    ``scanner`` bound to the fake scanner unless *roles* says otherwise."""
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    devices = config["hardware"]["devices"]
    bindings = {"electron": {"scanner": "fake_scanner"}} if roles is None else roles
    for name, bound in bindings.items():
        beam = next((e for e in devices if e.get("name") == name), None)
        if beam is None:
            beam = {"name": name}
            devices.append(beam)
        beam["roles"] = bound
    devices.extend(entries or [ENTRY])
    return config


def _connect(tmp_path, config):
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    return microscope


@pytest.fixture
def microscope(tmp_path):
    microscope = _connect(tmp_path, _config())
    yield microscope
    microscope.disconnect()


def _standalone(name="fake_scanner"):
    """The scanner built on its own from its entry, as a device server on the unit's
    PC would: with no microscope, and knowing its driver and type."""
    entry = DeviceEntry.from_dict({**ENTRY, "name": name})
    return build_device_entries([ResolvedEntry(entry, "FakeScan")], None)[name]


# -- discovery ----------------------------------------------------------------------


def test_the_entry_point_is_found_and_listed():
    records = [r for r in get_driver_plugin_records() if r.name == "FakeScan"]
    assert len(records) == 1 and records[0].loaded
    assert records[0].distribution == "fibsem-fake-scanner"
    assert set(get_driver("fake scan").devices) == {"scan_generator"}


def test_a_device_only_driver_is_not_offered_as_a_manufacturer():
    from fibsem.drivers.registry import registered_manufacturers

    assert "FakeScan" not in registered_manufacturers()


@pytest.mark.xfail(
    strict=True,
    reason="GAP: the guided setup offers only microscope drivers (driver_setups "
    "filters on registered_manufacturers); a device-only driver can't offer itself "
    "or add its entry to hardware.devices",
)
def test_the_guided_setup_offers_the_scanner():
    from fibsem.drivers.registry import driver_setups

    assert "FakeScan" in [name for name, _ in driver_setups()]


# -- configuration and build --------------------------------------------------------


def test_an_entry_builds_the_scanner_and_binds_the_beam(microscope):
    scanner = microscope.devices["fake_scanner"]
    assert isinstance(scanner, FakeScanner)
    assert microscope.beams[BeamType.ELECTRON].scanner is scanner
    assert "scanner" not in microscope.beams[BeamType.ION].roles
    assert scanner.pattern.get_value() == "checker"


def test_the_entry_survives_a_save(microscope):
    saved = microscope.system.to_dict()["hardware"]["devices"]
    by_name = {e["name"]: e for e in saved}
    assert by_name["fake_scanner"] == ENTRY
    assert by_name["electron"]["roles"] == {"scanner": "fake_scanner"}


def test_an_unknown_key_fails_the_build(tmp_path):
    with pytest.raises(DeviceBuildError, match="unknown keys"):
        _connect(tmp_path, _config({**ENTRY, "adress": "10.0.0.2"}))


def test_a_missing_plugin_says_it_is_not_installed(tmp_path):
    with pytest.raises(DeviceBuildError, match="not installed|no driver"):
        _connect(tmp_path, _config({**ENTRY, "driver": "FakeScanner"}))


# -- parameters, commands, imaging -------------------------------------------------


def test_parameters_check_their_limits_and_choices(microscope):
    scanner = microscope.devices["fake_scanner"]
    scanner.gain.set_value(2.0)
    assert scanner.gain.cached == 1.0
    with pytest.raises(ValueError):
        scanner.input.set_value("adc9")
    scanner.input.set_value("adc2")
    assert scanner.gain.get_value() == 0.5  # gain is per input


def test_the_beam_images_through_the_scanner(microscope):
    scanner = microscope.devices["fake_scanner"]
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ELECTRON, **SETTINGS)
    )
    assert scanner._controller.frames == 1
    assert image.data.shape == (48, 64)
    assert set(np.unique(image.data)) <= {0, 255}  # the checker pattern


def test_live_view_runs_on_the_scanner(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    frames = []
    sem.live_frame.connect(lambda image: frames.append(image))
    sem.resolution.set_value((384, 256))
    sem.start_live()
    deadline = time.monotonic() + 5
    while not frames and time.monotonic() < deadline:
        time.sleep(0.02)
    sem.stop_live()
    assert frames and microscope.devices["fake_scanner"]._controller.frames >= 1


def test_the_beam_offers_only_what_the_scanner_can_scan(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    assert sem.dwell_time.limits.min >= 25e-9
    assert sem.dwell_time.limits.max <= 100e-6
    # a frame the unit can't scan is refused before it reaches the SDK
    with pytest.raises(ValueError, match="dwell time"):
        microscope.acquire_image(
            ImageSettings(
                beam_type=BeamType.ELECTRON, **{**SETTINGS, "dwell_time": 1e-9}
            )
        )


@pytest.mark.xfail(
    strict=True,
    reason="GAP: one scanner can be bound to both beams and both can run live on it "
    "at once; nothing says a scan generator drives one column at a time",
)
def test_one_scanner_cannot_serve_two_beams(tmp_path):
    roles = {
        "electron": {"scanner": "fake_scanner"},
        "ion": {"scanner": "fake_scanner"},
    }
    with pytest.raises(Exception):
        _connect(tmp_path, _config(roles=roles))


# -- recording ---------------------------------------------------------------------


def test_a_command_records_through_the_microscope_session(microscope):
    events = []
    microscope.session.record_signal.connect(lambda kind, payload: events.append(kind))
    microscope.devices["fake_scanner"].self_test()
    assert events == ["scanner_self_test"]


def test_the_image_says_which_scanner_made_it(microscope):
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ELECTRON, **SETTINGS)
    )
    record = image.metadata.roles["scanner"]
    assert record["name"] == "fake_scanner"
    assert record["driver"] == "FakeScan"
    assert record["values"]["input"] == "adc0"
    assert record["facts"]["serial"] == "FS-0001"


def test_disconnect_closes_the_scanners_connection(tmp_path):
    microscope = _connect(tmp_path, _config())
    controller = microscope.devices["fake_scanner"]._controller
    microscope.disconnect()
    assert not controller.open


# -- the application ---------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="GAP: the configuration window's Devices table is a fixed list "
    "(microscope_configuration_window.devices_rows); a plugin device is not in it",
)
def test_the_configuration_window_lists_the_scanner(microscope):
    pytest.importorskip("PyQt5")
    from fibsem.ui.widgets.microscope_configuration_window import devices_rows

    assert any("scanner" in row[0].lower() for row in devices_rows(microscope))


def test_the_plugins_dialog_lists_the_driver():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from fibsem.ui.widgets.plugins_dialog import PluginsDialog

    _app = QApplication.instance() or QApplication([])
    dialog = PluginsDialog()
    assert ("fibsem.drivers", "FakeScan") in [(r.group, r.name) for r in dialog.rows]


def test_the_form_builder_can_draw_every_parameter(microscope):
    """No panel in the application shows a plugin device's parameters (see gaps),
    but the shared form builder can draw each of them from its metadata."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from fibsem.ui.widgets.form_builder import build_control, parameter_field_metadata

    _app = QApplication.instance() or QApplication([])
    for name, parameter in microscope.devices["fake_scanner"].parameters.items():
        metadata = {"items": None, "type": None, **parameter_field_metadata(parameter)}
        control = build_control(
            metadata, parameter.get_value(), annotation=parameter.type
        )
        assert control is not None and control.editable, name


# -- servers -----------------------------------------------------------------------


def test_the_app_server_serves_the_scanner(microscope):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fibsem.server import AuthConfig, build_server

    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token="t"))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": "Bearer t"})
        assert "fake_scanner" in client.get("/microscope/devices").json()
        r = client.put("/microscope/devices/fake_scanner/gain", json={"value": 0.4})
        assert r.json() == {"value": 0.4}
        r = client.post(
            "/microscope/devices/fake_scanner/commands/self_test", json={"kwargs": {}}
        )
        assert r.json()["result"]["serial"] == "FS-0001"


def test_the_device_server_serves_a_standalone_scanner():
    requests = pytest.importorskip("requests")
    from fibsem.server.devices import DeviceServer

    server = DeviceServer([_standalone()]).start()
    try:
        base = f"http://127.0.0.1:{_wait(server)}"
        described = requests.get(base + "/devices").json()["fake_scanner"]
        assert described["facts"]["serial"] == "FS-0001"
        r = requests.post(
            base + "/devices/fake_scanner/commands/acquire",
            json={"kwargs": {"resolution": [16, 8], "dwell_time": 1e-7}},
        )
        assert np.load(io.BytesIO(r.content)).shape == (8, 16)
    finally:
        server.stop()


def test_a_hardware_fault_is_a_clean_refusal():
    requests = pytest.importorskip("requests")
    from fibsem.server.devices import DeviceServer

    scanner = _standalone()
    scanner._controller.close()  # the unit went away
    server = DeviceServer([scanner]).start()
    try:
        r = requests.post(
            f"http://127.0.0.1:{_wait(server)}/devices/fake_scanner/commands/acquire",
            json={"kwargs": {"resolution": [16, 8], "dwell_time": 1e-7}},
        )
        # the unit's refusal is a device fault: 502, naming the device
        assert r.status_code == 502
        assert r.json()["detail"]["device"] == "fake_scanner"
    finally:
        server.stop()


def _remote(server):
    return {
        "name": "fake_scanner",
        "type": "scan_generator",
        "driver": "remote",
        "address": "127.0.0.1",
        "port": _wait(server),
    }


def test_a_remote_entry_reaches_a_served_scanner(tmp_path):
    pytest.importorskip("websockets")
    from fibsem_fake_scanner.remote import RemoteFakeScanner

    from fibsem.server.devices import DeviceServer

    served = _standalone()
    server = DeviceServer([served]).start()
    try:
        microscope = _connect(tmp_path, _config(_remote(server)))
        sem = microscope.beams[BeamType.ELECTRON]
        scanner = sem.roles["scanner"]
        # the plugin is installed here too, so its remote class: its own parameters
        assert isinstance(scanner, RemoteFakeScanner)
        assert scanner.input.get_value() == "adc0"
        scanner.gain.set_value(0.25)
        assert served.gain.get_value() == 0.25
        assert scanner.self_test()["serial"] == "FS-0001"
        # its limits came with it, and the beam narrows to them
        assert sem.dwell_time.limits.min == served.dwell_time_limits.min
        image = microscope.acquire_image(image_settings=ImageSettings(**SETTINGS))
        assert image.data.shape == (48, 64)
        assert image.metadata.roles["scanner"]["facts"]["serial"] == "FS-0001"
        microscope.disconnect()
    finally:
        server.stop()


def test_without_the_plugin_here_it_is_a_plain_remote_scanner(tmp_path, caplog):
    pytest.importorskip("websockets")
    import logging

    from fibsem.drivers.remote.devices import RemoteScanner
    from fibsem.server.devices import DeviceServer

    served = _standalone()
    served.driver = "NotInstalledHere"  # as if the coordinator lacked the plugin
    server = DeviceServer([served]).start()
    logging.getLogger().addHandler(caplog.handler)
    try:
        microscope = _connect(tmp_path, _config(_remote(server)))
        scanner = microscope.beams[BeamType.ELECTRON].roles["scanner"]
        assert type(scanner) is RemoteScanner
        assert "input" not in scanner.parameters
        assert "Not available here: gain, input, offset, pattern." in caplog.text
        # scanning still works through the interface
        image = microscope.acquire_image(image_settings=ImageSettings(**SETTINGS))
        assert image.data.shape == (48, 64)
        microscope.disconnect()
    finally:
        server.stop()


def _wait(server, timeout=5.0):
    """The server's port, once it is listening."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return server.port
        except (AttributeError, IndexError, TypeError):
            time.sleep(0.05)
    raise TimeoutError("device server did not start")
