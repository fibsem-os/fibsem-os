"""The FM widgets offer the exposure, power and binning the connected FM reports.

The FM here is the API over the simulator's devices, set to report a narrower camera
and light source than the widgets' old fixed ranges, as a real instrument would.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.devices.drivers.fm import bind_fm_devices  # noqa: E402
from fibsem.fm import microscope as fm_microscope  # noqa: E402
from fibsem.fm.api import DeviceFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.ui.fm.widgets import fm_limits  # noqa: E402

CHANNEL = ChannelSettings(name="green", excitation_wavelength=488, exposure_time=0.1)


@pytest.fixture
def fm(monkeypatch):
    """An FM whose camera takes 5 ms to 2 s at binning 1 or 2, and whose light
    source goes to 80%."""
    monkeypatch.setattr(
        fm_microscope.Camera,
        "exposure_time_limits",
        property(lambda self: (0.005, 2.0)),
    )
    monkeypatch.setattr(
        fm_microscope.Camera, "available_binnings", property(lambda self: (1, 2))
    )
    monkeypatch.setattr(
        fm_microscope.LightSource, "power_limits", property(lambda self: (0.0, 0.8))
    )
    return DeviceFluorescenceMicroscope(
        bind_fm_devices(fm_microscope.FluorescenceMicroscope())
    )


def _range(spin):
    return (spin.minimum(), spin.maximum())


def test_the_limits_come_from_the_fm(fm):
    assert fm_limits.exposure_range_ms(fm) == (5.0, 2000.0)
    assert fm_limits.power_range_percent(fm) == (0.0, 80.0)
    assert fm_limits.available_binnings(fm) == (1, 2)


def test_without_an_fm_the_widgets_keep_their_old_ranges():
    assert fm_limits.exposure_range_ms(None) == (1.0, 10000.0)
    assert fm_limits.power_range_percent(None) == (0.0, 100.0)
    assert fm_limits.available_binnings(None) == (1, 2, 4, 8)


def test_an_fm_that_cannot_answer_gets_the_old_ranges():
    class Unreachable:
        @property
        def camera(self):
            raise ConnectionError("the FM's computer is gone")

        light_source = camera

    assert fm_limits.exposure_range_ms(Unreachable()) == (1.0, 10000.0)
    assert fm_limits.power_range_percent(Unreachable()) == (0.0, 100.0)
    assert fm_limits.available_binnings(Unreachable()) == (1, 2, 4, 8)


def test_the_minimum_is_one_the_box_can_show():
    """A 1 µs camera minimum would show as 0.0 ms in a one-decimal box."""
    sim = fm_microscope.FluorescenceMicroscope()
    assert sim.camera.exposure_time_limits[0] == pytest.approx(1e-6)
    assert fm_limits.exposure_range_ms(sim) == (0.1, 60000.0)


def test_the_channel_settings_offer_the_fms_ranges(qapp, fm):
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm)

    assert _range(widget.exposure_spin) == (5.0, 2000.0)
    assert _range(widget.power_spin) == (0.0, 80.0)


def test_each_channel_row_offers_the_fms_ranges(qapp, fm):
    from fibsem.ui.fm.widgets.channel_list_widget import ChannelListWidget

    widget = ChannelListWidget(fm, [CHANNEL])
    row = widget._list.itemWidget(widget._list.item(0))

    assert _range(row.exposure_spin) == (5.0, 2000.0)
    assert _range(row.power_spin) == (0.0, 80.0)


def test_the_camera_offers_the_fms_binnings(qapp, fm):
    from fibsem.ui.fm.widgets.camera_widget import CameraWidget

    widget = CameraWidget(fm)
    combo = widget.combobox_binning

    assert [combo.itemData(i) for i in range(combo.count())] == [1, 2]
