"""The spot burn widget's beam-conditions control is backend-dependent.

Preset-driven backends (TESCAN) get a Preset combo — the driver ignores
``milling_current`` there, so a current combo would be a lie — while every other
backend keeps the Beam Current combo. Values come from the cached
available-values getter: building the tab must not query the microscope, so the
tests seed the cache and stub nothing else.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python tests/test_spot_burn_widget_preset.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem.imaging.spot import SpotBurnSettings
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.ui.FibsemSpotBurnWidget import DEFAULT_BEAM_CURRENT, FibsemSpotBurnWidget

_app = QApplication.instance() or QApplication(sys.argv)

PRESETS = ["30 keV; 100 pA", "30 keV; 1 nA; my cool preset"]
CURRENTS = [20e-12, 60e-12, 1e-9]


def make_tescan_microscope() -> TescanMicroscope:
    """A TescanMicroscope without the SDK, with the preset cache pre-seeded.

    Seeding ``_available_values_cache`` (the store behind
    ``get_available_values_cached``) is the point: the widget must read the
    cache, never the SDK — an uncached call would hit ``connection`` and fail.
    """
    microscope = object.__new__(TescanMicroscope)
    microscope._available_values_cache = {"preset_ION": list(PRESETS)}
    return microscope


def make_demo_microscope() -> DemoMicroscope:
    """A DemoMicroscope (non-Tescan) with the current cache pre-seeded."""
    microscope = object.__new__(DemoMicroscope)
    microscope._available_values_cache = {"current_ION": list(CURRENTS)}
    return microscope


def make_widget(microscope) -> FibsemSpotBurnWidget:
    host = QWidget()
    host.microscope = microscope
    widget = FibsemSpotBurnWidget(parent=host)
    widget._host = host  # keep the parent alive for the widget's lifetime
    return widget


def test_tescan_gets_a_preset_combo():
    widget = make_widget(make_tescan_microscope())

    assert widget._use_presets
    assert widget.comboBox_beam_current is None
    items = [
        widget.comboBox_preset.itemText(i)
        for i in range(widget.comboBox_preset.count())
    ]
    assert items == PRESETS

    settings = widget.get_settings()
    assert settings.preset == PRESETS[0]
    # milling_current is carried but ignored by the preset-driven driver
    assert settings.milling_current == DEFAULT_BEAM_CURRENT


def test_non_tescan_keeps_the_current_combo():
    widget = make_widget(make_demo_microscope())

    assert not widget._use_presets
    assert widget.comboBox_preset is None

    settings = widget.get_settings()
    assert settings.preset is None
    assert settings.milling_current == 60e-12  # closest to the default


def test_off_list_preset_round_trips_losslessly():
    """A protocol preset not in the machine's list stays exactly selectable, so an
    untouched value is not rewritten on save (mirrors the off-grid current rule)."""
    widget = make_widget(make_tescan_microscope())

    widget.set_settings(SpotBurnSettings(preset="20 keV; 40 pA; custom"))

    assert widget.get_settings().preset == "20 keV; 40 pA; custom"


def test_listed_preset_selection_round_trips():
    widget = make_widget(make_tescan_microscope())

    widget.set_settings(SpotBurnSettings(preset=PRESETS[1]))

    assert widget.comboBox_preset.count() == len(PRESETS)  # no duplicate added
    assert widget.get_settings().preset == PRESETS[1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
