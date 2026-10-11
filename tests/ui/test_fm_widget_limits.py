"""The FM widgets offer the exposure, power and binning the connected FM reports.

The FM here is the API over the Demo FM devices, set to report a narrower camera
and light source than the widgets' old fixed ranges, as a real instrument would.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.devices.core import ParameterMetadata  # noqa: E402
from fibsem.drivers.demo import devices as demo  # noqa: E402
from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope  # noqa: E402
from fibsem.fm.microscope import FluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.structures import RangeLimit  # noqa: E402
from fibsem.ui.fm.widgets import fm_limits  # noqa: E402

CHANNEL = ChannelSettings(name="green", excitation_wavelength=488, exposure_time=0.1)


@pytest.fixture
def fm(monkeypatch):
    """An FM whose camera takes 5 ms to 2 s at binning 1 or 2, and whose light
    source goes to 80%."""
    monkeypatch.setattr(
        demo.DemoCamera,
        "metadata_exposure_time",
        lambda self: ParameterMetadata(limits=RangeLimit(min=0.005, max=2.0)),
    )
    monkeypatch.setattr(
        demo.DemoCamera,
        "metadata_binning",
        lambda self: ParameterMetadata(choices=[1, 2]),
    )
    monkeypatch.setattr(
        demo.DemoLightSource,
        "metadata_power",
        lambda self: ParameterMetadata(limits=RangeLimit(min=0.0, max=0.8)),
    )
    return FluorescenceMicroscope(demo.bind_demo_fm())


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
    sim = DemoFluorescenceMicroscope()
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


# -- power and gain in the hardware's units ------------------------------------------


@pytest.fixture
def fm_with_units(monkeypatch):
    """An FM whose light reaches 0.4 W and whose camera gain goes to 16, as a METEOR's
    driver reports them."""
    fraction = RangeLimit(min=0.0, max=1.0)
    monkeypatch.setattr(
        demo.DemoLightSource,
        "metadata_power",
        lambda self: ParameterMetadata(
            limits=fraction, native_max=0.4, native_unit="W"
        ),
    )
    monkeypatch.setattr(
        demo.DemoCamera,
        "metadata_gain",
        lambda self: ParameterMetadata(limits=fraction, native_max=16.0),
        raising=False,
    )
    return FluorescenceMicroscope(demo.bind_demo_fm())


def _tooltip(spin, percent):
    spin.setValue(percent)
    return spin._native_units_tooltip.text(spin)


def test_the_units_read_as_a_person_would_say_them():
    assert fm_limits.in_native_units(50.0, (0.4, "W")) == "0.2 W"
    assert fm_limits.in_native_units(25.0, (16.0, None)) == "4 of 16"


def test_the_channel_settings_show_power_and_gain_in_hardware_units(
    qapp, fm_with_units
):
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm_with_units)

    assert _tooltip(widget.power_spin, 50.0) == "Light source power (%): 0.2 W"
    assert _tooltip(widget.gain_spin, 25.0) == "Detector gain (%): 4 of 16"


def test_each_channel_row_shows_power_and_gain_in_hardware_units(qapp, fm_with_units):
    from fibsem.ui.fm.widgets.channel_list_widget import ChannelListWidget

    widget = ChannelListWidget(fm_with_units, [CHANNEL])
    row = widget._list.itemWidget(widget._list.item(0))

    assert _tooltip(row.power_spin, 10.0) == "Light source power (%): 0.04 W"
    assert _tooltip(row.gain_spin, 50.0) == "Gain (%): 8 of 16"


def test_the_camera_shows_gain_in_hardware_units(qapp, fm_with_units):
    from fibsem.ui.fm.widgets.camera_widget import CameraWidget

    widget = CameraWidget(fm_with_units)

    assert _tooltip(widget.spinBox_gain, 100.0).endswith(": 16 of 16")


def test_without_hardware_units_the_tooltips_stay_as_they_were(qapp):
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    sim = FluorescenceMicroscope(demo.bind_demo_fm())
    widget = ChannelSettingsWidget(sim)

    assert widget.power_spin.toolTip() == "Light source power (%)"
    assert not hasattr(widget.power_spin, "_native_units_tooltip")


def test_hovering_shows_the_value_it_has_now(qapp, fm_with_units):
    """The box's value is often set with its signals blocked, so the tooltip is worked
    out when it opens, not when the value changes."""
    from PyQt5.QtCore import QEvent, QPoint
    from PyQt5.QtGui import QHelpEvent
    from PyQt5.QtWidgets import QToolTip

    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm_with_units)
    spin = widget.power_spin
    widget.show()
    spin.blockSignals(True)
    spin.setValue(30.0)
    spin.blockSignals(False)

    qapp.sendEvent(
        spin, QHelpEvent(QEvent.ToolTip, QPoint(5, 5), spin.mapToGlobal(QPoint(5, 5)))
    )

    assert QToolTip.text() == "Light source power (%): 0.12 W"
    QToolTip.hideText()


def test_the_boxes_show_values_as_the_devices_declare(fm):
    """Unit, step and decimals are the devices' display hints, not widget constants."""
    from fibsem.ui.fm.widgets.camera_widget import CameraWidget
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    channel = ChannelSettingsWidget(fm)
    camera = CameraWidget(fm)

    assert channel.exposure_spin.suffix() == " ms"
    assert channel.power_spin.suffix() == " %"
    assert (channel.gain_spin.suffix(), _range(channel.gain_spin)) == (
        " %",
        (0.0, 100.0),
    )
    assert camera.label_gain.text() == "Gain"
    assert (camera.spinBox_gain.decimals(), _range(camera.spinBox_gain)) == (
        1,
        (0.0, 100.0),
    )


def test_the_z_stack_boxes_show_the_z_parameters_fields():
    from fibsem.fm.structures import ZParameters
    from fibsem.ui.fm.widgets.z_parameters_widget import ZParametersWidget

    widget = ZParametersWidget(ZParameters(zmin=-5e-6, zmax=5e-6, zstep=0.5e-6))

    assert widget.label_zstep.text() == "Z Step"
    assert widget.doubleSpinBox_zmin.suffix() == " µm"
    assert _range(widget.doubleSpinBox_zstep) == (0.1, 10.0)
    assert widget.doubleSpinBox_zmin.value() == pytest.approx(-5.0)

    widget.doubleSpinBox_zstep.setValue(2.0)
    assert widget.z_parameters.zstep == pytest.approx(2e-6)
