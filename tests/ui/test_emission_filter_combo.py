"""The emission filter picker: shows each filter by name and band, and keeps today's
emission values, so channels and saved settings are unchanged."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.fm.microscope import FilterSet, FluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.ui.fm.widgets.emission_filter_combo import (  # noqa: E402
    EmissionFilterComboBox,
    emission_lookup_for,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _BandedFilters(FilterSet):
    """An Odemis-like filter set: bands known, one of them dual-band."""

    @property
    def available_emission_wavelengths(self):
        return (None, 425.0, 505.0)

    @property
    def emission_bands(self):
        return {425.0: ((425.0, 475.0),), 505.0: ((505.0, 535.0), (600.0, 650.0))}


class _FM:
    filter_set = _BandedFilters()


def _labels(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_items_show_the_filters_and_keep_their_values(qapp):
    combo = EmissionFilterComboBox(
        items=[None, 425.0, 505.0], lookup=emission_lookup_for(_FM())
    )
    assert _labels(combo) == ["Reflection", "425–475 nm", "505–535 / 600–650 nm"]
    combo.set_value(505.0)
    assert combo.value() == 505.0
    assert len(combo.emission_filter().bands) == 2
    tooltip = combo.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert tooltip == "425–475 nm: band 425 to 475 nm, centre 450 nm"
    assert not combo.itemIcon(1).isNull()


def test_the_simulator_multi_band_filter_shows_as_multi_band(qapp):
    fm = FluorescenceMicroscope()
    combo = EmissionFilterComboBox(
        items=list(fm.filter_set.available_emission_wavelengths),
        lookup=emission_lookup_for(fm),
    )
    assert _labels(combo) == ["Reflection", "Multi-band"]
    combo.set_value("Fluorescence")
    assert combo.value() == "Fluorescence"  # the stored value doesn't change


def test_a_filter_set_without_emission_filter_gets_plain_names(qapp):
    class Plain:
        class filter_set:
            available_emission_wavelengths = [450.0, 520.0]

    combo = EmissionFilterComboBox(
        items=[450.0, 520.0], lookup=emission_lookup_for(Plain())
    )
    assert _labels(combo) == ["450 nm", "520 nm"]


def test_the_settings_panel_keeps_the_channel_value(qapp):
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm=_FM())
    channel = ChannelSettings(name="GFP", emission_wavelength=505.0)
    widget.set_channel(channel)
    assert widget.emission_combo.currentText() == "505–535 / 600–650 nm"
    widget.emission_combo.set_value(425.0)
    assert channel.emission_wavelength == 425.0
