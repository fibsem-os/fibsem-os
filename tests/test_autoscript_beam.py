"""The AutoScript beam driver makes the SDK calls Thermo's beam keys made before it.

``AutoscriptBeam`` is ``ThermoMicroscope``'s beam branches moved onto the ``Beam``
device. Each case runs a ``get``/``set`` on a microscope whose beam keys are routed to
the drivers as ``ThermoMicroscope``'s connect routes them, over a fake AutoScript
client that records every SDK call and write. It must give the result, the calls in
order and the logged messages the old branches gave, recorded over the same fake in
``tests/fixtures/autoscript_old_calls.json`` before they were deleted. A ``set`` case
reads the key back after the write.

Cases: every moved key on both beams, with and without a plasma column; the hfw
clip; an unlisted plasma gas, which warns and is still set; the detector keys, which
select the beam's channel first, and their refused values; the scan-mode methods,
through the scan commands on one side and the old keys on the other; the electron
beam's angular correction, whose tilt correction could only be set before: it reads
on the new API, and the old key's get still returns None; and ``preset``, which
Thermo does not have. The fake SDK has to be in place before
``fibsem.microscopes.autoscript`` is first imported, so the recording runs in its own interpreter
(``tests/fixtures/autoscript_beam_parity.py``). Nothing here has run on an instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.autoscript_recording import load

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_beam_parity.py"
PINS = Path(__file__).parent / "fixtures" / "available_values_pins.json"
RECORDED = Path(__file__).parent / "fixtures" / "autoscript_old_calls.json"

MOVED = [
    "blanked",
    "current",
    "detector_brightness",
    "detector_contrast",
    "detector_mode",
    "detector_type",
    "dwell_time",
    "hfw",
    "on",
    "resolution",
    "scan_rotation",
    "scanning_mode",
    "shift",
    "stigmation",
    "voltage",
    "working_distance",
]

# The electron beam's only.
ANGULAR_KEYS = ["angular_correction_angle", "angular_correction_tilt_correction"]


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
    recording = load(out.read_text())
    # the old code's side, recorded before it was deleted
    old = load(RECORDED.read_text())["beam"]
    assert sorted(c["key"] for c in recording["cases"]) == sorted(old)
    for case in recording["cases"]:
        case["old"] = old[case["key"]]
    return recording


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


def _quiet(old):
    """The old side with its "Unknown key" warnings dropped: a key the column's
    device does not have (``preset``, a plasma gas on a column without a source)
    was unknown to the old ``_set`` and warned; with no ``_set`` of Thermo's own
    it is unsupported, as on every backend, and logs at debug level."""
    result, calls, messages = old
    return [result, calls, [m for m in messages if not m[1].startswith("Unknown key")]]


_SELECT = ("connection.imaging.set_active_view", "connection.imaging.set_active_device")


def _selects_once(calls):
    """The calls with each run of repeated channel selections made once: after a
    detector type write the beam reads the type's modes again (the mode's choices
    depend on it), selecting its channel for the read, which adds a selection the
    old code did not make."""
    out = []
    i = 0
    while i < len(calls):
        two = calls[i : i + 2]
        if [c[1] for c in two] == list(_SELECT) and out[-2:] == two:
            i += 2  # the same selection again
            continue
        out.append(calls[i])
        i += 1
    return out


def _same(case):
    old = _quiet(case["old"])
    new = case["new"]
    if " set detector_type " in case["key"]:
        old = [old[0], _selects_once(old[1]), old[2]]
        new = [new[0], _selects_once(new[1]), new[2]]
    return old == new


def test_routed_keys_make_the_same_sdk_calls_logs_and_results(recording):
    different = [c for c in recording["cases"] if not _same(c)]
    assert different == [], json.dumps(different[:3], indent=1)


def test_only_the_unknown_key_warnings_and_mode_reads_are_new(recording):
    changed = [c["key"] for c in recording["cases"] if c["old"] != c["new"]]
    assert changed
    assert all(" set " in key for key in changed), changed
    for case in recording["cases"]:
        if " set detector_type " in case["key"]:
            assert len(case["new"][1]) > len(case["old"][1]), case["key"]


@pytest.mark.parametrize("plasma", [False, True])
@pytest.mark.parametrize("beam", ["ELECTRON", "ION"])
def test_the_moved_keys_are_the_ones_routed(recording, plasma, beam):
    facts = recording["facts"][f"plasma={plasma} {beam}"]
    gas = ["plasma_gas"] if plasma and beam == "ION" else []
    electron = beam == "ELECTRON"
    keys = ANGULAR_KEYS if electron else []
    parameters = ["angular_correction", "tilt_correction"] if electron else []
    assert facts["routed"] == sorted(MOVED + gas + keys)
    assert facts["parameters"] == sorted(MOVED + gas + parameters)
    assert facts["commands"] == [
        "acquire",
        "auto_focus",
        "autocontrast",
        "blank",
        "full_frame",
        "last_image",
        "reduced_area",
        "spot",
        "start_live",
        "stop_live",
        "unblank",
    ]


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
    """The choices are what the old get_available_values answered, as pinned in
    ``tests/fixtures/available_values_pins.json`` before it read them."""
    facts = recording["facts"][f"plasma={plasma} {beam}"]
    pins = json.loads(PINS.read_text())
    old = {
        name: pins[f"thermo plasma={plasma} {beam} {name}"]
        for name in (
            "voltage",
            "current",
            "detector_type",
            "detector_mode",
            "plasma_gas",
        )
    }
    # the modes are the detector type's, read again when the type changes
    for name in ("voltage", "current", "detector_type", "detector_mode"):
        assert facts["choices"][name] == old[name], name
    if plasma and beam == "ION":
        assert facts["choices"]["plasma_gas"] == old["plasma_gas"]
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


def test_the_scan_methods_make_the_old_scan_calls(recording):
    calls = {
        name: next(
            c["new"][1]
            for c in recording["cases"]
            if c["key"] == f"plasma=False ION scan {name}"
        )
        for name in ("spot", "reduced_area", "full_frame")
    }
    assert calls["spot"] == [
        [
            "call",
            "connection.beams.ion_beam.scanning.mode.set_spot",
            [],
            "{'x': 0.25, 'y': 0.75}",
        ]
    ]
    assert calls["reduced_area"][0][1].endswith("scanning.mode.set_reduced_area")
    assert calls["full_frame"][0][1].endswith("scanning.mode.set_full_frame")


def test_the_scan_mode_is_read_and_an_unknown_one_warns(recording):
    """New: nothing read the vendor's scan mode before. A mode with no ScanMode
    warns and reads None rather than failing the scan command that read it back."""
    facts = recording["facts"]["scanning_mode"]
    assert facts["full_frame"] == "full_frame"
    result, calls, messages = facts["line"]
    assert result is None and calls == []
    assert messages[0][0] == "WARNING" and "Line" in messages[0][1]


@pytest.mark.parametrize(
    "key",
    [
        "set detector_brightness 0.0",
        "set detector_contrast 1.5",
        "set detector_type 'Unobtainium'",
        "set detector_mode 'Unobtainium'",
    ],
)
def test_a_refused_detector_value_warns_and_is_not_written(recording, key):
    case = next(
        c for c in recording["cases"] if c["key"] == f"plasma=False ELECTRON {key}"
    )
    _, calls, messages = case["new"]
    assert not any(call[0] == "set" for call in calls)
    assert messages[0][0] == "WARNING"


def test_a_detector_read_selects_its_beams_channel_under_the_lock(recording):
    facts = recording["facts"]["detector_read"]
    assert facts["selected"] == [["ION", True]]
    assert facts["result"] == "ETD"


def test_the_angular_correction_is_set_as_before(recording):
    for value, call in ((True, "turn_on"), (False, "turn_off")):
        key = f"plasma=False ELECTRON set angular_correction_tilt_correction {value!r}"
        case = next(c for c in recording["cases"] if c["key"] == key)
        assert case["new"][1] == [
            [
                "call",
                f"connection.beams.electron_beam.angular_correction.tilt_correction.{call}",
                [],
                "{}",
            ]
        ]
    key = "plasma=False ELECTRON set angular_correction_angle 0.2"
    case = next(c for c in recording["cases"] if c["key"] == key)
    assert case["new"][0] == [None, 0.2]
    assert ["INFO", "Angular correction angle set to 0.2 radians."] in case["new"][2]


def test_the_tilt_correction_reads_on_the_new_api_only(recording):
    """The old key could only be set, and its get returned None. It still does; the
    device's parameter reads the vendor's state."""
    assert recording["facts"]["tilt_correction"] == {
        "before": False,
        "after": True,
        "key": None,
        "ion": None,
    }
