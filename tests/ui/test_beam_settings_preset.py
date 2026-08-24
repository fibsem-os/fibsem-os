"""SEM preset hiding and worker-thread preset activation (Tescan).

Two facts from the 2026-08 simulator session drive these tests. The simulator's SEM
*enumerates* presets, so the old count-based gate showed the combo -- but activating
one raises ``PresetNotFound``: enumerability is not settability, and the hardware
session had already established SEM presets cannot be set. And ``_on_preset_changed``
ran ``set_preset`` synchronously in the Qt slot, so that raise propagated out of a
slot (an abort, per the fail-fast policy) instead of surfacing as a toast.

The widget is the real ``FibsemBeamSettingsWidget``; only the microscope is faked.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from fibsem.structures import BeamType
from fibsem.ui import notification_service
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget

_app = QApplication.instance() or QApplication([])


def _pump(ms: int = 50) -> None:
    """Let queued cross-thread signals deliver."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


class FakeTescan:
    """The slice of the microscope the beam-settings widget touches, Tescan-flavoured."""

    manufacturer = "Tescan"

    def __init__(self, presets, current_preset, fail_activation=False):
        self.presets = presets
        self.current_preset = current_preset
        self.fail_activation = fail_activation
        self.activated = []
        self.activation_threads = []

    def get_beam_current(self, beam_type):
        return 1e-9

    def get_beam_voltage(self, beam_type):
        return 30000.0

    def get_available_values_cached(self, key, beam_type):
        if key == "preset":
            return list(self.presets)
        if key == "current":
            return [1e-9]
        if key == "voltage":
            return [30000.0]
        return []

    def get(self, key, beam_type=None):
        assert key == "preset"
        return self.current_preset

    def set_preset(self, preset, beam_type):
        self.activation_threads.append(threading.current_thread())
        if self.fail_activation:
            raise RuntimeError("PresetNotFound (5)")
        self.activated.append(preset)
        self.current_preset = preset


def make_widget(beam_type, presets, current_preset="A", fail_activation=False):
    microscope = FakeTescan(presets, current_preset, fail_activation)
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=beam_type)
    widget.populate_beam_combos()
    return widget, microscope


@pytest.fixture
def toasts(monkeypatch):
    events = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda msg, notification_type="info": events.append((msg, notification_type)),
    )
    return events


# ---------------------------------------------------------------------------
# visibility: the SEM never shows the preset control
# ---------------------------------------------------------------------------


def test_sem_preset_hidden_even_when_presets_enumerate():
    """The simulator case: SEM Enum() returns presets, the control must still hide."""
    widget, _ = make_widget(BeamType.ELECTRON, presets=["A", "B"])
    assert widget.preset_combo.isHidden()
    assert widget.preset_label.isHidden()


def test_fib_preset_visible_with_presets():
    widget, _ = make_widget(BeamType.ION, presets=["A", "B"])
    assert not widget.preset_combo.isHidden()


def test_fib_preset_hidden_without_presets():
    widget, _ = make_widget(BeamType.ION, presets=[])
    assert widget.preset_combo.isHidden()


# ---------------------------------------------------------------------------
# activation: off the GUI thread, failure toasts and reverts
# ---------------------------------------------------------------------------


def _wait_for(condition, timeout_ms=2000):
    for _ in range(timeout_ms // 25):
        if condition():
            return True
        _pump(25)
    return condition()


def test_preset_activation_runs_off_the_gui_thread(toasts):
    widget, microscope = make_widget(
        BeamType.ION, presets=["A", "B"], current_preset="A"
    )
    applied = []
    widget.settings_changed.connect(lambda s: applied.append(s))

    widget.preset_combo.setCurrentIndex(widget.preset_combo.findData("B"))

    assert _wait_for(lambda: microscope.activated == ["B"] and applied)
    assert microscope.activation_threads[0] is not threading.main_thread()
    assert widget.preset_combo.isEnabled()
    assert toasts == []


def test_preset_failure_toasts_and_reverts(toasts):
    widget, microscope = make_widget(
        BeamType.ION, presets=["A", "B"], current_preset="A", fail_activation=True
    )
    changed = []
    widget.settings_changed.connect(lambda s: changed.append(s))

    widget.preset_combo.setCurrentIndex(widget.preset_combo.findData("B"))

    assert _wait_for(lambda: bool(toasts))
    msg, level = toasts[0]
    assert "B" in msg and "PresetNotFound" in msg
    assert level == "error"
    # reverted to the preset the microscope still reports, control usable again,
    # and no settings_changed for a preset that never applied
    assert _wait_for(lambda: widget.preset_combo.currentData() == "A")
    assert widget.preset_combo.isEnabled()
    assert changed == []


def test_preset_failure_with_no_prior_preset_clears_selection(toasts):
    widget, microscope = make_widget(
        BeamType.ION, presets=["A", "B"], current_preset=None, fail_activation=True
    )

    widget.preset_combo.setCurrentIndex(widget.preset_combo.findData("B"))

    assert _wait_for(lambda: bool(toasts))
    assert _wait_for(lambda: widget.preset_combo.currentIndex() == -1)
