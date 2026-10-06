"""Serving a METEOR's FM from the computer that runs odemis (FIB-1095).

No odemis installation required: odemis is replaced by the stub modules in
tests/fm/_odemis_stubs.py, which return components by role as odemis does.
"""

import sys

import pytest

pytest.importorskip("fastapi")

from fibsem.server import devices as server_devices  # noqa: E402
from fibsem.server.devices import device_health, odemis_fm_devices  # noqa: E402
from tests.fm import _odemis_stubs as stubs  # noqa: E402


@pytest.fixture
def odemis_stubs():
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    stubs.install_odemis_stubs()
    stubs.use_components(stubs.default_components())

    yield stubs

    stubs.remove_odemis_stubs()
    for name in stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


class _Unanswered:
    """A VA whose backend has gone away."""

    @property
    def value(self):
        raise IOError("the odemis backend did not answer")


def test_the_odemis_fm_is_served_as_its_parts(odemis_stubs):
    devices = {d.name: d for d in odemis_fm_devices()}

    assert sorted(devices) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    assert device_health(devices["fm"]) == {"ok": True, "detail": None}


def test_health_reports_an_fm_that_stopped_answering(odemis_stubs):
    devices = {d.name: d for d in odemis_fm_devices()}
    devices["camera"]._camera.exposureTime = _Unanswered()

    health = device_health(devices["fm"])

    assert health["ok"] is False
    assert "did not answer" in health["detail"]


def test_an_odemis_backend_that_is_not_running_is_named(odemis_stubs):
    odemis_stubs.use_components({})  # no components: odemis is not running

    with pytest.raises(RuntimeError, match="odemis-start.*'odemis' group"):
        odemis_fm_devices()


def test_a_computer_without_odemis_is_named(odemis_stubs, monkeypatch):
    monkeypatch.setitem(sys.modules, "fibsem.fm.odemis", None)  # import fails

    with pytest.raises(RuntimeError, match="cannot be imported here"):
        odemis_fm_devices()


def test_the_command_line_stops_with_the_reason(odemis_stubs, caplog):
    odemis_stubs.use_components({})

    with pytest.raises(SystemExit) as stopped:
        server_devices.main(["--serve", "odemis-fm"])

    assert stopped.value.code == 1
    assert "odemis-start" in caplog.text


def test_a_remote_fm_shows_power_and_gain_in_the_hardwares_units(odemis_stubs):
    """Power and gain are fractions everywhere; the hardware's own full scale crosses
    the wire with their metadata, for a display to show beside them."""
    pytest.importorskip("websockets")
    from fibsem.devices.drivers.remote import DeviceClient
    from fibsem.fm.remote import RemoteFluorescenceMicroscope
    from fibsem.server.devices import DeviceServer

    components = stubs.default_components()
    components["ccd"].gain = stubs.FakeVA(4.0, range=(0.0, 16.0))
    stubs.use_components(components)
    server = DeviceServer(odemis_fm_devices()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    try:
        fm = RemoteFluorescenceMicroscope.connect(
            "127.0.0.1", server.port, client=client
        )
        assert fm.light_source.power_native_scale == (0.4, "W")
        assert fm.camera.gain_native_scale == (16.0, None)
        assert fm.camera.gain == 0.25
    finally:
        client.close()
        server.stop()


def test_a_remote_fm_brings_how_its_camera_is_mounted(odemis_stubs):
    """The METEOR's server states the mount; a client applies it to every frame
    before the user's own transform, without being told it separately."""
    pytest.importorskip("websockets")
    import numpy as np

    from fibsem.devices.drivers.remote import DeviceClient
    from fibsem.fm.remote import RemoteFluorescenceMicroscope
    from fibsem.fm.structures import CameraImageTransform
    from fibsem.server.devices import DeviceServer

    server = DeviceServer(odemis_fm_devices({"mount_transform": "flip-x"})).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    try:
        fm = RemoteFluorescenceMicroscope.connect(
            "127.0.0.1", server.port, client=client
        )
        assert fm.mount_transform is CameraImageTransform.FLIP_X
        frame = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(fm._apply_image_transform(frame), frame[:, ::-1])
    finally:
        client.close()
        server.stop()


def test_the_command_line_names_the_mount(odemis_stubs, monkeypatch):
    served = {}

    def odemis_fm_devices(config):
        served["config"] = config
        return []

    monkeypatch.setattr(server_devices, "odemis_fm_devices", odemis_fm_devices)
    monkeypatch.setattr(server_devices.uvicorn, "run", lambda *a, **k: None)

    server_devices.main(["--serve", "odemis-fm", "--mount-transform", "flip-y"])

    assert served["config"] == {"mount_transform": "flip-y"}
