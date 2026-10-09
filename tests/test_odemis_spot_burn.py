"""Odemis spot burns through the spot burn service, with the writes it always made.

Each case burns on an ``OdemisThermoMicroscope`` built as it is when created, over the
recording fake client of tests/test_odemis_devices.py, once through
``microscope.spot_burn`` and once with no service, so that the microscope's own spot
burn runs. The two must make the same odemis writes in the same order, log the same
and report the same progress. The exposures are whole seconds, which the
microscope's own burn counted down in.

What the service leaves out are reads: the microscope's own burn read the blanker
back after every blank and unblank, and the current after every write, through the
old methods' return values, which it never used.
"""

import threading

import pytest

from fibsem.imaging.spot import SpotBurnSettings
from fibsem.services.spot_burn import SpotBurn
from fibsem.structures import BeamType, Point
from tests.test_odemis_devices import make, odemis_cls, run  # noqa: F401


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)


def _settings(*points, exposure_time=2.0):
    return SpotBurnSettings(
        coordinates=list(points), exposure_time=exposure_time, milling_current=1e-9
    )


def _stopped():
    event = threading.Event()
    event.set()
    return event


CASES = {
    "two points": lambda m: m.run_spot_burn(
        _settings(Point(0.2, 0.3), Point(0.7, 0.4))
    ),
    "a point outside the image": lambda m: m.run_spot_burn(
        _settings(Point(1.5, 0.5), Point(0.5, 0.5))
    ),
    "stopped before the first point": lambda m: m.run_spot_burn(
        _settings(Point(0.5, 0.5)), stop_event=_stopped()
    ),
}


def _is_read(call):
    return call[0].startswith(("get_", "beam_is_")) or call[0].endswith("_info")


def _writes(calls):
    return [c for c in calls if not _is_read(c)]


def _burn(odemis_cls, call, service):
    microscope = make(odemis_cls)
    if not service:
        microscope.spot_burn = None
    reports = []
    microscope.spot_burn_progress_signal.connect(reports.append)
    ran = run(microscope, call)
    return ran, reports


def test_odemis_builds_its_spot_burn_service(odemis_cls):
    microscope = make(odemis_cls)
    assert type(microscope.spot_burn) is SpotBurn
    assert microscope.spot_burn.ion is microscope.beams[BeamType.ION]


def test_without_an_ion_beam_there_is_no_spot_burn(odemis_cls):
    assert make(odemis_cls, ion=False).spot_burn is None


@pytest.mark.parametrize("key", list(CASES))
def test_a_burn_makes_the_same_odemis_calls(odemis_cls, key):
    new, new_reports = _burn(odemis_cls, CASES[key], service=True)
    old, old_reports = _burn(odemis_cls, CASES[key], service=False)
    assert _writes(new["calls"]), key
    assert _writes(new["calls"]) == _writes(old["calls"])
    assert all(c in old["calls"] for c in new["calls"] if _is_read(c))
    assert new["result"] == old["result"]
    assert new["log"] == old["log"]
    assert new_reports == old_reports


def test_a_burn_parks_blanked_and_puts_the_current_back(odemis_cls):
    ran, _ = _burn(odemis_cls, CASES["two points"], service=True)
    point = [["blank_beam", ["ion"], {}], ["unblank_beam", ["ion"], {}]]
    writes = _writes(ran["calls"])
    assert writes[0] == ["set_beam_current", [1e-9, "ion"], {}]
    assert [w for w in writes if w[0] in ("blank_beam", "unblank_beam")] == point * 2
    assert [w[2] for w in writes if w[0] == "set_spot_scan_mode"] == [
        {"channel": "ion", "x": 0.2, "y": 0.3},
        {"channel": "ion", "x": 0.7, "y": 0.4},
    ]
    assert writes[-2:] == [
        ["set_full_frame_scan_mode", ["ion"], {}],
        ["set_beam_current", [3e-11, "ion"], {}],
    ]
