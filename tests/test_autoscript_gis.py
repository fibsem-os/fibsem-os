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
