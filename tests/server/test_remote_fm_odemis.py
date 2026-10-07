"""The Odemis FM driver served over localhost, as a METEOR's Linux PC serves it.

Its band centres carry the float noise of a metre-to-nanometre conversion
(450.00000000000006), so this is where a value snapped on the client has to reach
the server, the hardware and back again unchanged (FIB-1094).

No odemis installation required: odemis is replaced by the stub modules in
tests/fm/_odemis_stubs.py.
"""

import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.drivers.remote.devices import DeviceClient  # noqa: E402
from fibsem.fm.remote import RemoteFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.server.devices import DeviceServer  # noqa: E402
from tests.fm import _odemis_stubs as stubs  # noqa: E402


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture(scope="module")
def odemis_fm_module():
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    stubs.install_odemis_stubs()
    import fibsem.drivers.odemis.devices as drivers
    import fibsem.fm.odemis as fm_odemis

    yield fm_odemis, drivers

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


@pytest.fixture
def served(odemis_fm_module):
    fm_odemis, drivers = odemis_fm_module
    stubs.use_components(stubs.default_components())
    devices = drivers.bind_odemis_fm()
    far = fm_odemis.DeviceOdemisFluorescenceMicroscope(devices)
    server = DeviceServer(devices.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    fm = RemoteFluorescenceMicroscope.connect("127.0.0.1", server.port, client=client)
    yield far, devices, fm
    client.close()
    server.stop()


def test_an_excitation_between_bands_selects_the_nearest_band(served):
    far, devices, fm = served
    band_450 = min(
        far.filter_set.available_excitation_wavelengths, key=lambda c: abs(c - 450)
    )

    fm.filter_set.excitation_wavelength = 488

    assert far.filter_set.excitation_wavelength == band_450
    assert fm.filter_set.excitation_wavelength == band_450
    # both caches hold what the hardware applied, float noise and all
    assert devices["filter_set"].excitation_wavelength.cached == band_450
    remote = fm.devices["filter_set"].excitation_wavelength
    assert wait_for(lambda: remote.cached == band_450)


def test_a_channel_set_up_from_the_client_matches_a_local_one(served):
    far, _, fm = served
    channel = ChannelSettings(
        excitation_wavelength=488,
        emission_wavelength=None,
        power=0.1,
        exposure_time=0.01,
    )

    fm.set_channel(channel)

    assert far.filter_set.excitation_wavelength == pytest.approx(450)
    assert far.filter_set.emission_wavelength is None
