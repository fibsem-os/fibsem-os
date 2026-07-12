"""Unit tests for the Autoscript sputter coater wrappers + microscope selector.

The wrappers and the ThermoMicroscope are built via object.__new__ to bypass the
hardware __init__, injecting a fake coater so the prepare/run/recover sequence,
current handling and platform selection can be checked without a live system.
"""

import types

import pytest

from fibsem.microscopes.autoscript import (
    AutoscriptArctisSputterCoater,
    AutoscriptSputterCoater,
)


class _FakeCurrent:
    def __init__(self):
        self.value = None


class _FakeCoater:
    def __init__(self, is_installed=True):
        self.is_installed = is_installed
        self.current = _FakeCurrent()
        self.calls = []

    def prepare(self): self.calls.append("prepare")
    def run(self, t): self.calls.append(("run", t))
    def recover(self): self.calls.append("recover")


def _wrapper(cls, coater):
    w = object.__new__(cls)
    w.parent = None
    w._coater = coater
    return w


# --- wrapper sequences ------------------------------------------------------

def test_standard_coater_wraps_run_in_prepare_recover():
    coater = _FakeCoater()
    _wrapper(AutoscriptSputterCoater, coater).run(10, current=0.01)
    assert coater.calls == ["prepare", ("run", 10), "recover"]
    assert coater.current.value == 0.01


def test_standard_coater_recovers_on_error():
    coater = _FakeCoater()
    coater.run = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        _wrapper(AutoscriptSputterCoater, coater).run(10)
    assert coater.calls[-1] == "recover"  # recover still runs (finally)


def test_arctis_coater_runs_without_prepare_recover():
    coater = _FakeCoater()
    _wrapper(AutoscriptArctisSputterCoater, coater).run(5, current=0.02)
    # Arctis run() is self-contained — no prepare/recover (unsupported there)
    assert coater.calls == [("run", 5)]
    assert coater.current.value == 0.02


def test_current_left_untouched_when_none():
    coater = _FakeCoater()
    _wrapper(AutoscriptSputterCoater, coater).run(3)
    assert coater.current.value is None  # not set when current is None


# --- microscope: _create_sputter_coater (built on connect) ------------------

def _thermo(model, sputter_coater=...):
    """Bare ThermoMicroscope with a fake connection/system (no hardware init)."""
    from fibsem.microscope import ThermoMicroscope
    specimen = types.SimpleNamespace()
    if sputter_coater is not ...:  # allow simulating an absent attribute
        specimen.sputter_coater = sputter_coater
    m = object.__new__(ThermoMicroscope)
    m.connection = types.SimpleNamespace(specimen=specimen)
    m.system = types.SimpleNamespace(info=types.SimpleNamespace(model=model))
    m._sputter_coater = None
    return m


def test_create_builds_standard_for_non_arctis():
    m = _thermo("Hydra", _FakeCoater())
    # exact type: Arctis subclasses the standard, so isinstance would match both
    assert type(m._create_sputter_coater()) is AutoscriptSputterCoater


def test_create_builds_arctis_for_arctis_model():
    m = _thermo("Arctis", _FakeCoater())
    assert type(m._create_sputter_coater()) is AutoscriptArctisSputterCoater


def test_create_returns_none_when_not_installed():
    m = _thermo("Hydra", _FakeCoater(is_installed=False))
    assert m._create_sputter_coater() is None


def test_create_returns_none_when_attribute_absent():
    m = _thermo("Hydra")  # specimen has no sputter_coater
    assert m._create_sputter_coater() is None


# --- microscope: run_sputter_coater delegates to the cached wrapper ---------

def test_run_delegates_to_cached_coater():
    coater = _FakeCoater()
    m = _thermo("Hydra", coater)
    m._sputter_coater = m._create_sputter_coater()  # as connect would
    m.run_sputter_coater(10, current=0.01)
    assert coater.calls == ["prepare", ("run", 10), "recover"]


def test_run_raises_when_no_coater():
    m = _thermo("Hydra")
    m._sputter_coater = None  # nothing built on connect
    with pytest.raises(NotImplementedError):
        m.run_sputter_coater(10)
