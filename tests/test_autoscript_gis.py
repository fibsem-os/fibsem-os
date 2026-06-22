"""Unit tests for AutoscriptGISPort.run_deposition (abort-aware deposition cycle).

The port is constructed via object.__new__ to bypass the hardware __init__
(which needs a live Autoscript connection), injecting a fake port + parent.
"""

import threading

import pytest

from fibsem.microscopes.autoscript import AutoscriptGISPort
from fibsem.structures import FibsemStagePosition


class _FakePort:
    def __init__(self):
        self.calls = []

    def insert(self): self.calls.append("insert")
    def open(self): self.calls.append("open")
    def close(self): self.calls.append("close")
    def retract(self): self.calls.append("retract")


class _FakeParent:
    """Stage sits below the GIS z-limit so the safety check passes."""
    def get_stage_position(self):
        return FibsemStagePosition(z=0.0)


def _gis_port(port):
    gp = object.__new__(AutoscriptGISPort)
    gp._port = port
    gp.parent = _FakeParent()
    return gp


def test_run_deposition_runs_full_cycle():
    port = _FakePort()
    progress = []
    # duration 0 → no wait, but the insert/open/close/retract cycle still runs
    _gis_port(port).run_deposition(0, on_progress=progress.append)
    assert port.calls == ["insert", "open", "close", "retract"]
    assert progress == []  # no wait iterations at duration 0


def test_run_deposition_stops_early_and_still_retracts():
    port = _FakePort()
    stop = threading.Event()
    stop.set()  # already stopped → wait breaks on first check, before any sleep
    _gis_port(port).run_deposition(9999, stop_event=stop)
    # close + retract always run, even on early stop (finally)
    assert port.calls == ["insert", "open", "close", "retract"]


def test_run_deposition_retracts_on_error():
    class _BoomPort(_FakePort):
        def open(self):
            self.calls.append("open")
            raise RuntimeError("boom")

    port = _BoomPort()
    with pytest.raises(RuntimeError):
        _gis_port(port).run_deposition(0)
    # open() raises after insertion, but close + retract still run (finally),
    # so the GIS is never left inserted on error
    assert port.calls == ["insert", "open", "close", "retract"]


# --- ThermoMicroscope.run_gis_deposition facade ----------------------------

class _FakeGISWrapper:
    """Stands in for the lazily-built gis_port wrapper."""
    def __init__(self):
        self.calls = []

    def _move_to_safe_gis_position(self): self.calls.append("safe")
    def _run_safety_check(self): self.calls.append("check")
    def run_deposition(self, duration, stop_event=None, on_progress=None):
        self.calls.append(("deposit", duration, stop_event))


def _thermo_with_gis(gis):
    import types
    from fibsem.microscope import ThermoMicroscope
    m = object.__new__(ThermoMicroscope)
    m._gis_port = gis
    m._state = []
    m.get_microscope_state = lambda *a, **k: "STATE"
    m.set_microscope_state = lambda s, *a, **k: m._state.append(s)
    return m


def test_facade_runs_safe_check_deposit_then_restores_state():
    gis = _FakeGISWrapper()
    m = _thermo_with_gis(gis)
    stop = threading.Event()
    m.run_gis_deposition(5, stop_event=stop)
    # safe height + safety check precede the deposit; stop event is threaded through
    assert gis.calls == ["safe", "check", ("deposit", 5, stop)]
    assert m._state == ["STATE"]  # initial state restored afterwards


def test_facade_restores_state_on_error():
    gis = _FakeGISWrapper()
    gis.run_deposition = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    m = _thermo_with_gis(gis)
    with pytest.raises(RuntimeError):
        m.run_gis_deposition(5)
    assert m._state == ["STATE"]  # restored even when deposition raises (finally)


def test_facade_raises_when_no_gis_port():
    m = _thermo_with_gis(None)
    with pytest.raises(NotImplementedError):
        m.run_gis_deposition(5)
