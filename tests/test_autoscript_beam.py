"""The AutoScript beam driver makes the SDK calls Thermo's beam keys make today.

``AutoscriptBeam`` is ``ThermoMicroscope``'s beam branches moved onto the ``Beam``
device. Each case runs an old ``get``/``set`` on one microscope, and the same call on
another whose beam keys are routed to the drivers as ``ThermoMicroscope``'s connect
routes them, both over a fake AutoScript client that records every SDK call
and write. Each case requires the same result, the same calls in the same order, and
the same logged messages. A ``set`` case reads the key back after the write.

Cases: every moved key on both beams, with and without a plasma column; the hfw
clip; an unlisted plasma gas, which warns and is still set; and keys that have not
moved (detector keys, ``preset``), which both sides still answer with the old
branches. The fake SDK has to be in place before ``fibsem.microscopes.autoscript`` is
first imported, so the recording runs in its own interpreter
(``tests/fixtures/autoscript_beam_parity.py``). Nothing here has run on an instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_beam_parity.py"

MOVED = [
    "blanked",
    "current",
    "dwell_time",
    "hfw",
    "on",
    "resolution",
    "scan_rotation",
    "shift",
    "stigmation",
    "voltage",
    "working_distance",
]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_beam") / "recording.json"
    env = dict(os.environ, FIBSEM_SIM_NO_DELAY="1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(out.read_text())


def test_the_recording_covers_both_beams_and_makes_sdk_calls(recording):
    cases = recording["cases"]
    assert len(cases) > 100
    assert any(" ION " in c["key"] for c in cases)
    # writes and calls: the fake does not record plain reads, which show in the result
    assert sum(len(c["old"][1]) for c in cases) > 80
    assert sum(len(c["old"][2]) for c in cases) > 50


def test_no_old_call_raised(recording):
    raised = [
        c["key"]
        for c in recording["cases"]
        if isinstance(c["old"][0], str) and c["old"][0].startswith("EXC")
    ]
    assert raised == []


def test_routed_keys_make_the_same_sdk_calls_logs_and_results(recording):
    different = [c for c in recording["cases"] if c["old"] != c["new"]]
    assert different == [], json.dumps(different[:3], indent=1)


@pytest.mark.parametrize("plasma", [False, True])
@pytest.mark.parametrize("beam", ["ELECTRON", "ION"])
def test_the_moved_keys_are_the_ones_routed(recording, plasma, beam):
    facts = recording["facts"][f"plasma={plasma} {beam}"]
    gas = ["plasma_gas"] if plasma and beam == "ION" else []
    assert facts["routed"] == sorted(MOVED + gas)
    assert facts["parameters"] == sorted(MOVED + gas)
    # the scan commands and scanning_mode have not moved
    assert facts["commands"] == ["acquire", "blank", "unblank"]


def test_hfw_is_clipped_just_inside_the_maximum(recording):
    case = next(
        c
        for c in recording["cases"]
        if c["key"] == "plasma=False ELECTRON set hfw 10.0"
    )
    writes = [call for call in case["new"][1] if call[0] == "set"]
    assert writes == [
        [
            "set",
            "connection.beams.electron_beam.horizontal_field_width.value",
            2e-3 - 10e-6,
        ]
    ]


def test_an_unlisted_gas_warns_and_is_still_set(recording):
    case = next(
        c
        for c in recording["cases"]
        if c["key"] == "plasma=True ION set plasma_gas 'Unobtainium'"
    )
    _, calls, messages = case["new"]
    assert [
        "set",
        "connection.beams.ion_beam.source.plasma_gas.value",
        "Unobtainium",
    ] in calls
    assert messages[0][0] == "WARNING" and "Unobtainium not available" in messages[0][1]


@pytest.mark.parametrize("plasma", [False, True])
@pytest.mark.parametrize("beam", ["ELECTRON", "ION"])
def test_the_choices_are_get_available_values(recording, plasma, beam):
    facts = recording["facts"][f"plasma={plasma} {beam}"]
    for name in ("voltage", "current"):
        assert facts["choices"][name] == facts["old_choices"][name], name
    if plasma and beam == "ION":
        assert facts["choices"]["plasma_gas"] == facts["old_choices"]["plasma_gas"]
    else:
        assert facts["choices"]["plasma_gas"] is None  # no gas on this column


def test_new_api_refuses_a_voltage_off_the_list_before_any_sdk_call(recording):
    facts = recording["facts"]["voltage_off_the_list"]
    assert facts["refused"] is not None and "1234" in facts["refused"]
    assert facts["calls"] == []


def test_a_disabled_column_is_never_built_or_touched(recording):
    facts = recording["facts"]["ion_disabled"]
    assert facts["beams"] == ["ELECTRON"]
    assert not any("ion_beam" in call[1] for call in facts["calls"])


def test_connect_builds_the_beams_before_it_resets_their_shifts(recording):
    assert recording["facts"]["connect"] == {
        "beams": ["ELECTRON", "ION"],
        "routed": True,
    }
