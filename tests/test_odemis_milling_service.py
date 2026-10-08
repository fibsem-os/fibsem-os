"""Odemis milling through the milling service, against the calls it always made.

Each case runs on an ``OdemisThermoMicroscope`` built as it is when created, over the
recording fake client of tests/test_odemis_devices.py, through ``microscope.milling``
(OdemisMilling). tests/fixtures/odemis_milling_calls.json holds what the microscope's
own milling code returned and called for the same cases before that code moved into
the service; the service must make the same odemis calls and return the same. What
the service changes on purpose, putting the beam back at the end, is asserted on its
own.
"""

import json
from pathlib import Path

import pytest

from fibsem.structures import (
    BeamType,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    MillingState,
)
from tests.test_odemis_devices import READS, make, odemis_cls, run  # noqa: F401

SETTINGS = FibsemMillingSettings(
    milling_current=1e-9,
    milling_voltage=30e3,
    hfw=80e-6,
    application_file="Si-new",
    patterning_mode="Parallel",
)
PATTERNS = [
    FibsemRectangleSettings(
        width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
    ),
    FibsemLineSettings(start_x=0, end_x=1e-6, start_y=0, end_y=0, depth=1e-6),
    FibsemCircleSettings(radius=2e-6, depth=1e-6, centre_x=0, centre_y=0),
]


@pytest.fixture(autouse=True)
def patterning(monkeypatch):
    monkeypatch.setitem(READS, "get_patterning_state", "Idle")
    monkeypatch.setitem(READS, "estimate_milling_time", 12.0)
    monkeypatch.setitem(READS, "create_rectangle", {"id": 1, "time": 1.0})
    monkeypatch.setitem(READS, "create_line", {"id": 2, "time": 1.0})
    monkeypatch.setitem(READS, "create_circle", {"id": 3, "time": 1.0})


with open(Path(__file__).parent / "fixtures" / "odemis_milling_calls.json") as f:
    OLD_CALLS = json.load(f)


def test_odemis_builds_its_milling_service(odemis_cls):
    from fibsem.drivers.odemis.services import OdemisMilling

    microscope = make(odemis_cls)
    assert isinstance(microscope.milling, OdemisMilling)
    assert microscope.milling.ion is microscope.beams[BeamType.ION]
    assert microscope.milling.electron is microscope.beams[BeamType.ELECTRON]


def test_without_an_ion_beam_there_is_no_milling(odemis_cls):
    microscope = make(odemis_cls, ion=False)
    assert microscope.milling is None
    with pytest.raises(ValueError, match="ION beam is not enabled"):
        microscope.setup_milling(SETTINGS)
    assert microscope.get_milling_state() is MillingState.IDLE


def _setup(m):
    m.setup_milling(SETTINGS)


def _draw(m):
    for pattern in PATTERNS:
        m.draw_pattern(pattern)


def _run(m):
    m.setup_milling(SETTINGS)
    _draw(m)
    return [
        m.estimate_milling_time(),
        m.get_milling_state(),
        m.start_milling(),
        m.pause_milling(),
        m.resume_milling(),
        m.stop_milling(),
        m.clear_patterns(),
    ]


# The service's first setup reads the beam it will put back; nothing else differs.
SAVES = [
    ["get_high_voltage", ["ion"], {}],
    ["get_beam_current", ["ion"], {}],
    ["get_field_of_view", ["ion"], {}],
]


@pytest.mark.parametrize("call", [_setup, _draw, _run], ids=lambda c: c.__name__)
def test_milling_makes_the_same_odemis_calls(odemis_cls, call):
    ran = run(make(odemis_cls), call)
    result = ran["result"]
    # a voltage write reads the current choices again (they follow the voltage)
    calls = [c for c in ran["calls"] if c[0] != "beam_current_info"]
    old = OLD_CALLS[call.__name__]
    assert result == old["result"]
    if call is not _draw:
        assert calls[: len(SAVES)] == SAVES
        calls = calls[len(SAVES) :]
    assert calls == old["calls"]


def test_the_state_is_read_on_the_milling_channel(odemis_cls):
    ran = run(make(odemis_cls), lambda m: m.get_milling_state())
    assert ran["result"] == repr(MillingState.IDLE)
    assert [c[0] for c in ran["calls"]] == [
        "set_active_view",
        "set_active_device",
        "get_patterning_state",
    ]


WRITES = ("set_high_voltage", "set_beam_current", "set_field_of_view")


def _finish(microscope, **overrides):
    """The odemis calls ``finish_milling`` makes after a setup, without the reads."""
    microscope.setup_milling(SETTINGS)
    calls = run(microscope, lambda m: m.finish_milling(**overrides))["calls"]
    return [
        c[:2]
        for c in calls
        if c[0] in (*WRITES, "clear_patterns", "set_patterning_mode")
    ]


def test_finish_milling_puts_the_beam_back(odemis_cls):
    """Setup saves what milling found; finish clears, puts it back, then Serial."""
    microscope = make(odemis_cls)
    assert _finish(microscope) == [
        ["clear_patterns", []],
        ["set_high_voltage", [30000.0, "ion"]],
        ["set_beam_current", [3e-11, "ion"]],
        ["set_field_of_view", [900e-6, "ion"]],
        ["set_patterning_mode", ["Serial"]],
    ]
    assert microscope.milling._saved is None


def test_an_imaging_current_and_voltage_given_win(odemis_cls):
    calls = _finish(make(odemis_cls), imaging_current=2e-11, imaging_voltage=16e3)
    # after what milling found is put back and the mode reset
    assert calls[-3:] == [
        ["set_patterning_mode", ["Serial"]],
        ["set_high_voltage", [16e3, "ion"]],
        ["set_beam_current", [2e-11, "ion"]],
    ]


def test_odemis_mills_with_the_settings_it_says(odemis_cls, monkeypatch):
    from tests.fixtures.milling_reads import fields_setup_reads

    files = ["Si", "Si-ccs"]
    monkeypatch.setitem(READS, "get_available_application_files", files)
    microscope = make(odemis_cls)
    supported = microscope.milling.supported_settings()
    assert list(supported["application_file"].choices) == files
    assert set(supported) == fields_setup_reads(microscope.milling, SETTINGS)
    directions = microscope.milling.supported_pattern_settings()["scan_direction"]
    assert directions.choices == (
        "TopToBottom",
        "BottomToTop",
        "LeftToRight",
        "RightToLeft",
    )
