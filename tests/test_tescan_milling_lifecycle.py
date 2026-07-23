"""Tests for the Tescan DrawBeam milling lifecycle (FIB-292).

clear_patterns() is the single place that unloads a DrawBeam layer, and setup_milling
snapshots the preset that was active beforehand so finish_milling can put the column back
where the user left it instead of forcing a hardcoded one.

No hardware or Tescan SDK required: the microscope object is created without __init__, the
connection is stubbed, and the SDK names the driver imports are monkeypatched in.
"""

from typing import List, Optional

import pytest

from fibsem.microscopes import tescan as tescan_module
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings


class FakeDrawBeam:
    def __init__(self, unload_error: Optional[Exception] = None):
        self.calls: List[str] = []
        self._unload_error = unload_error

    def UnloadLayer(self):
        self.calls.append("UnloadLayer")
        if self._unload_error is not None:
            raise self._unload_error

    def Layer(self, name, settings):
        self.calls.append("Layer")
        return object()


class FakePreset:
    def __init__(self, activate_error: Optional[Exception] = None):
        self.activated: List[str] = []
        self._activate_error = activate_error

    def Activate(self, preset):
        self.activated.append(preset)
        if self._activate_error is not None:
            raise self._activate_error


class FakeFIB:
    def __init__(self, activate_error=None):
        self.Preset = FakePreset(activate_error)


class FakeConnection:
    def __init__(self, unload_error=None, activate_error=None):
        self.DrawBeam = FakeDrawBeam(unload_error)
        self.FIB = FakeFIB(activate_error)


def make_microscope(monkeypatch, current_preset="30 keV; 20 pA",
                    unload_error=None, activate_error=None):
    microscope = object.__new__(TescanMicroscope)
    microscope.connection = FakeConnection(unload_error, activate_error)
    microscope.milling_channel = BeamType.ION
    microscope._preset_before_milling = None
    microscope._prepare_beam = lambda beam_type: None

    state = {"preset": current_preset, "set_calls": []}
    microscope._test_state = state

    def fake_get(key, beam_type=None):
        return {"preset": state["preset"], "current": 1e-9}[key]

    def fake_set(key, value, beam_type=None):
        state["set_calls"].append((key, value))
        if key == "preset":
            state["preset"] = value

    microscope.get = fake_get
    microscope.set = fake_set

    monkeypatch.setattr(tescan_module, "IEtching", lambda **kwargs: kwargs, raising=False)
    return microscope


# --------------------------------------------------------------------------
# clear_patterns
# --------------------------------------------------------------------------

def test_clear_patterns_unloads_the_layer(monkeypatch):
    m = make_microscope(monkeypatch)
    m.clear_patterns()
    assert m.connection.DrawBeam.calls == ["UnloadLayer"]


def test_clear_patterns_swallows_the_no_layer_error(monkeypatch):
    """UnloadLayer raises when no layer is loaded; callers must not have to care."""
    m = make_microscope(monkeypatch, unload_error=RuntimeError("no layer loaded"))
    m.clear_patterns()  # must not raise
    assert m.connection.DrawBeam.calls == ["UnloadLayer"]


def test_clear_patterns_is_idempotent(monkeypatch):
    m = make_microscope(monkeypatch, unload_error=RuntimeError("no layer loaded"))
    for _ in range(3):
        m.clear_patterns()
    assert m.connection.DrawBeam.calls == ["UnloadLayer"] * 3


# --------------------------------------------------------------------------
# preset snapshot / restore
# --------------------------------------------------------------------------

def test_setup_milling_snapshots_the_active_preset(monkeypatch):
    m = make_microscope(monkeypatch, current_preset="30 keV; 20 pA")

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))

    assert m._preset_before_milling == "30 keV; 20 pA"
    assert ("preset", "30 keV; 2 nA") in m._test_state["set_calls"]


def test_setup_milling_twice_keeps_the_original_snapshot(monkeypatch):
    """The re-entry guard: without it the second call would snapshot the milling preset."""
    m = make_microscope(monkeypatch, current_preset="30 keV; 20 pA")

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    m.setup_milling(FibsemMillingSettings(preset="30 keV; 5 nA"))

    assert m._preset_before_milling == "30 keV; 20 pA"


def test_finish_milling_restores_the_snapshot(monkeypatch):
    m = make_microscope(monkeypatch, current_preset="30 keV; 20 pA")

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    m.finish_milling()

    assert m.connection.FIB.Preset.activated == ["30 keV; 20 pA"]


def test_finish_milling_clears_the_snapshot_for_the_next_cycle(monkeypatch):
    m = make_microscope(monkeypatch, current_preset="30 keV; 20 pA")

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    m.finish_milling()
    assert m._preset_before_milling is None

    # a second milling cycle snapshots whatever is active then
    m._test_state["preset"] = "30 keV; 100 pA"
    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    m.finish_milling()
    assert m.connection.FIB.Preset.activated == ["30 keV; 20 pA", "30 keV; 100 pA"]


def test_finish_milling_falls_back_when_no_snapshot_exists(monkeypatch):
    """get("preset") can be None if no image was acquired and no preset set this session."""
    m = make_microscope(monkeypatch, current_preset=None)

    m.finish_milling()

    assert m.connection.FIB.Preset.activated == [tescan_module.DEFAULT_IMAGING_PRESET]


def test_finish_milling_falls_back_when_snapshot_is_none(monkeypatch):
    m = make_microscope(monkeypatch, current_preset=None)

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    assert m._preset_before_milling is None  # nothing was active to snapshot
    m.finish_milling()

    assert m.connection.FIB.Preset.activated == [tescan_module.DEFAULT_IMAGING_PRESET]


# --------------------------------------------------------------------------
# cleanup independence
# --------------------------------------------------------------------------

def test_finish_milling_unloads_the_layer_even_if_the_preset_fails(monkeypatch):
    """The whole point of splitting the try blocks: a fragile preset restore must not
    skip the cheap layer unload."""
    m = make_microscope(
        monkeypatch,
        current_preset="30 keV; 20 pA",
        activate_error=RuntimeError("preset unavailable"),
    )

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))
    m.connection.DrawBeam.calls.clear()

    m.finish_milling()  # must not raise

    assert m.connection.DrawBeam.calls == ["UnloadLayer"]


def test_finish_milling_survives_both_steps_failing(monkeypatch):
    m = make_microscope(
        monkeypatch,
        current_preset="30 keV; 20 pA",
        unload_error=RuntimeError("no layer loaded"),
        activate_error=RuntimeError("preset unavailable"),
    )

    m.finish_milling()  # must not raise

    assert m.connection.DrawBeam.calls == ["UnloadLayer"]


def test_setup_milling_clears_a_stale_layer_first(monkeypatch):
    m = make_microscope(monkeypatch)

    m.setup_milling(FibsemMillingSettings(preset="30 keV; 2 nA"))

    assert m.connection.DrawBeam.calls[0] == "UnloadLayer"
    assert "Layer" in m.connection.DrawBeam.calls


def test_setup_milling_rejects_electron_channel(monkeypatch):
    m = make_microscope(monkeypatch)
    with pytest.raises(ValueError, match="FIB milling"):
        m.setup_milling(FibsemMillingSettings(milling_channel=BeamType.ELECTRON))
