"""The beam-settings and milling-stage widgets read the beam device's parameters.

A control is hidden when the beam has no such parameter, shown read-only when the beam
reports it not settable, and a read-only combo lists only the value the beam reads. A
backend without beam devices keeps the manufacturer rules. Tescan runs over the fake
SDK (``tests/fixtures/tescan_sdk.py``), connected with and without its beam devices.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget
from fibsem.ui.widgets.milling_stages_widget import FibsemMillingStagesWidget
from tests.fixtures.tescan_sdk import connect

_app = QApplication.instance() or QApplication([])

E, I = BeamType.ELECTRON, BeamType.ION

CONTROLS = {
    "current": "beam_current_combo",
    "voltage": "beam_voltage_combo",
    "stigmation": "stigmation_row",
    "preset": "preset_combo",
    "working_distance": "working_distance_spinbox",
}


def _tescan(monkeypatch, devices=True):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    if not devices:
        monkeypatch.setattr(TescanMicroscope, "_build_beams", lambda self: None)
    microscope, _ = connect(monkeypatch, system)
    assert bool(microscope.beams) is devices
    return microscope


def _demo():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    return microscope


def _widget(microscope, beam_type, advanced=True):
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=beam_type)
    widget.populate_beam_combos()
    widget.set_advanced_visible(advanced)
    return widget


def _state(widget):
    """Each control: hidden, read-only or settable."""
    state = {}
    for name, attr in CONTROLS.items():
        control = getattr(widget, attr)
        if control.isHidden():
            state[name] = "hidden"
        else:
            state[name] = "settable" if control.isEnabled() else "read-only"
    return state


@pytest.mark.parametrize("beam_type", [E, I])
def test_demo_shows_every_control_but_preset(beam_type):
    assert _state(_widget(_demo(), beam_type)) == {
        "current": "settable",
        "voltage": "settable",
        "stigmation": "settable",
        "preset": "hidden",
        "working_distance": "settable",
    }


def test_tescan_electron_reads_its_beam(monkeypatch):
    # The SEM enumerates presets on the simulator, but has no preset parameter.
    assert _state(_widget(_tescan(monkeypatch), E)) == {
        "current": "read-only",
        "voltage": "settable",
        "stigmation": "read-only",
        "preset": "hidden",
        "working_distance": "settable",
    }


def test_tescan_ion_reads_its_beam(monkeypatch):
    assert _state(_widget(_tescan(monkeypatch), I)) == {
        "current": "read-only",
        "voltage": "read-only",
        "stigmation": "read-only",
        "preset": "settable",
        "working_distance": "hidden",
    }


def test_a_read_only_combo_lists_only_the_value_read(monkeypatch):
    microscope = _tescan(monkeypatch)
    widget = _widget(microscope, I)
    combo = widget.beam_current_combo
    assert combo.count() == 1
    assert combo.currentData() == microscope.get_beam_current(I)


def test_advanced_controls_hide_outside_advanced_mode(monkeypatch):
    state = _state(_widget(_tescan(monkeypatch), E, advanced=False))
    assert state["voltage"] == "hidden"
    assert state["stigmation"] == "hidden"
    assert state["current"] == "read-only"


@pytest.mark.parametrize("beam_type", [E, I])
def test_tescan_without_beam_devices_keeps_the_manufacturer_rules(
    monkeypatch, beam_type
):
    # An enabled Tescan column always has its device, and the old get/set branches
    # that read one without it are gone (FIB-1161), so the widget alone is shown none.
    monkeypatch.setattr(FibsemBeamSettingsWidget, "_beam_device", lambda self: None)
    assert _state(_widget(_tescan(monkeypatch), beam_type)) == {
        "current": "hidden",
        "voltage": "hidden",
        "stigmation": "hidden",
        "preset": "settable" if beam_type is I else "hidden",
        "working_distance": "settable" if beam_type is E else "read-only",
    }


@pytest.mark.parametrize(
    "make, show_preset",
    [
        (lambda mp: _demo(), False),
        (lambda mp: _tescan(mp), True),
        (lambda mp: _tescan(mp, devices=False), True),
    ],
    ids=["demo", "tescan", "tescan-without-beam-devices"],
)
def test_milling_stages_show_preset_when_the_ion_beam_has_presets(
    monkeypatch, make, show_preset
):
    widget = FibsemMillingStagesWidget(microscope=make(monkeypatch), stages=[])
    assert widget._list._show_preset is show_preset
