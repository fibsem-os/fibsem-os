"""Odemis spot burns through the spot burn service, with the writes it always made.

Each case burns on an ``OdemisThermoMicroscope`` built as it is when created, over the
recording fake client of tests/test_odemis_devices.py, through ``run_spot_burn`` and
``microscope.spot_burn``. The odemis writes and the log lines are pinned to what the
microscope's own burn made before the service replaced it (it was checked against the
service write for write until it was removed).
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


def _burn(odemis_cls, call):
    microscope = make(odemis_cls)
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


def _spot(x, y):
    return [
        ["blank_beam", ["ion"], {}],
        ["set_spot_scan_mode", [], {"channel": "ion", "x": x, "y": y}],
        ["unblank_beam", ["ion"], {}],
    ]


_BURN_AT = [["set_beam_current", [1e-9, "ion"], {}]]
_PUT_BACK = [
    ["set_full_frame_scan_mode", ["ion"], {}],
    ["set_beam_current", [3e-11, "ion"], {}],
]
_BURNING = "burning spot {}: Point(x={}, y={}, name=None), exposure time: 2.0, milling current: 1e-09"

# The odemis writes each case makes, in order, and what it logs: what the
# microscope's own burn made and logged before the service replaced it.
PINNED = {
    "two points": (
        _BURN_AT + _spot(0.2, 0.3) + _spot(0.7, 0.4) + _PUT_BACK,
        [
            ["INFO", _BURNING.format(1, 0.2, 0.3)],
            ["INFO", _BURNING.format(2, 0.7, 0.4)],
        ],
    ),
    "a point outside the image": (
        _BURN_AT + _spot(0.5, 0.5) + _PUT_BACK,
        [
            [
                "WARNING",
                "Skipping 1 spot burn coordinate(s) outside image bounds (0-1): "
                "[Point(x=1.5, y=0.5, name=None)]",
            ],
            ["INFO", _BURNING.format(1, 0.5, 0.5)],
        ],
    ),
    "stopped before the first point": (
        _BURN_AT + _PUT_BACK,
        [["INFO", "Spot burn cancelled before point 1/1."]],
    ),
}


@pytest.mark.parametrize("key", list(CASES))
def test_a_burn_makes_the_odemis_writes_it_always_made(odemis_cls, key):
    ran, reports = _burn(odemis_cls, CASES[key])
    writes, log = PINNED[key]
    assert _writes(ran["calls"]) == writes
    assert ran["log"] == log
    assert ran["result"] is None
    assert reports[-1].status.name == (
        "CANCELLED" if key.startswith("stopped") else "FINISHED"
    )


def test_a_burn_parks_blanked_and_puts_the_current_back(odemis_cls):
    ran, _ = _burn(odemis_cls, CASES["two points"])
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
