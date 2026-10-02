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
    devices["fm"]._fm.camera._camera.exposureTime = _Unanswered()

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
