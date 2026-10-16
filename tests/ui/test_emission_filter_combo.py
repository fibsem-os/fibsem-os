"""The emission filter picker: shows each of the filter set's filters by name and
band, and finds a channel's filter among them."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import (  # noqa: E402
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
)
from fibsem.ui.fm.widgets.emission_filter_combo import (  # noqa: E402
    EmissionFilterComboBox,
)

BLUE = EmissionFilter("425–475 nm", low=425.0, high=475.0)
DUAL = EmissionFilter("505–535 / 600–650 nm", bands=[(505, 535), (600, 650)])
FILTERS = [REFLECTION, BLUE, DUAL]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _BandedFilters:
    """An Odemis-like filter set: bands known, one of them dual-band. The rest is the
    Demo FM's filter set."""

    def __init__(self):
        self._demo = DemoFluorescenceMicroscope().filter_set

    def __getattr__(self, name):
        return getattr(self._demo, name)

    @property
    def available_emission_filters(self):
        return tuple(FILTERS)


class _FM:
    filter_set = _BandedFilters()


def _labels(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_items_show_the_filters(qapp):
    combo = EmissionFilterComboBox(items=FILTERS)
    assert _labels(combo) == ["Reflection", "425–475 nm", "505–535 / 600–650 nm"]
    combo.set_value(DUAL)
    assert combo.value() == DUAL
    tooltip = combo.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert tooltip == "425–475 nm: band 425 to 475 nm, centre 450 nm"
    assert not combo.itemIcon(1).isNull()


def test_a_channel_saved_by_bottom_edge_finds_its_filter(qapp):
    combo = EmissionFilterComboBox(items=FILTERS)
    assert combo.index_of(EmissionFilter("505 nm", low=505.0)) == 2
    assert combo.index_of(REFLECTION) == 0


def test_the_simulator_multi_band_filter_shows_as_multi_band(qapp):
    fm = DemoFluorescenceMicroscope()
    combo = EmissionFilterComboBox(items=list(fm.filter_set.available_emission_filters))
    assert _labels(combo) == ["Reflection", "Multi-band"]


def test_the_settings_panel_sets_the_channel_filter(qapp):
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm=_FM())
    channel = ChannelSettings(
        name="GFP", emission_filter=EmissionFilter("505 nm", low=505.0)
    )
    widget.set_channel(channel)
    assert widget.emission_combo.currentText() == "505–535 / 600–650 nm"
    widget.emission_combo.set_value(BLUE)
    assert channel.emission_filter == BLUE


def test_hovering_the_box_shows_the_current_filter_in_full(qapp):
    combo = EmissionFilterComboBox(items=FILTERS)
    combo.set_value(DUAL)
    assert combo.toolTip() == "505–535 / 600–650 nm: 2 bands"
    combo.set_value(REFLECTION)
    assert combo.toolTip() == "Reflection: no emission filter"
