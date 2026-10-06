"""The AutoScript stage drivers make the SDK calls Thermo's stage code made before them.

``AutoscriptStage`` and ``AutoscriptCompustage`` are Thermo's stage code moved onto the
``Stage`` device. Each case runs a driver call over a fake AutoScript client that
records every SDK call and write, and requires the calls, arguments, order and result
the matching old ``ThermoMicroscope`` call gave, recorded over the same fake in
``tests/fixtures/autoscript_old_calls.json`` before it was deleted. It also runs the
old call on a microscope routed as connect routes it (stage keys and moves through
the device), which must return the same and make the same moves and writes.
Cases: the limits read at connect, position/homed/linked reads, home and link, and
absolute and relative moves over the orientations, partial poses, and the compustage
axis restrictions with and without an inserted objective.

The fake SDK has to be in place before ``fibsem.microscopes.autoscript`` is first
imported, so the recording runs in its own interpreter
(``tests/fixtures/autoscript_stage_parity.py``). Nothing here has run on an
instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_stage_parity.py"
RECORDED = Path(__file__).parent / "fixtures" / "autoscript_old_calls.json"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_stage") / "recording.json"
    env = dict(os.environ, FIBSEM_SIM_NO_DELAY="1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    recording = json.loads(out.read_text())
    # the old code's side, recorded before it was deleted
    old = json.loads(RECORDED.read_text())["stage"]
    assert sorted(c["key"] for c in recording["cases"]) == sorted(old)
    for case in recording["cases"]:
        case["old"] = old[case["key"]]
    return recording


def test_the_recording_covers_both_stages_and_makes_sdk_calls(recording):
    cases = recording["cases"]
    assert len(cases) > 60
    assert any(k["key"].startswith("compustage=True") for k in cases)
    assert sum(len(case["old"][1]) for case in cases) > 150
    moves = [c for c in cases if " absolute " in c["key"]]
    assert all(
        any(call[1].endswith("absolute_move") for call in c["old"][1]) for c in moves
    )


def test_no_old_call_raised(recording):
    raised = [
        c["key"]
        for c in recording["cases"]
        if isinstance(c["old"][0], str) and c["old"][0].startswith("EXC")
    ]
    assert raised == []


def test_driver_makes_the_same_sdk_calls_and_returns_the_same(recording):
    # home(): the old API reads homed back; the driver's _home alone does not
    different = [
        c
        for c in recording["cases"]
        if c["old"] != c["new"] and not c["key"].endswith(" home()")
    ]
    assert different == [], json.dumps(different[:3], indent=1)


def _actions(log):
    """The SDK calls and writes, without the position reads (a read selects the
    coordinate system first)."""
    return [
        call
        for call in log
        if call[0] != "get" and not call[1].endswith("set_default_coordinate_system")
    ]


def test_routed_thermo_makes_the_same_moves_and_returns_the_same(recording):
    """ThermoMicroscope as connect leaves it: stage keys and moves through the
    device. It returns the same and makes the same moves and writes; it reads the
    position back twice after a move (the device's read-back, then the old API's
    return), and reads homed after a home."""
    different = [
        c["key"]
        for c in recording["cases"]
        if c["routed"][0] != c["old"][0]
        or _actions(c["routed"][1]) != _actions(c["old"][1])
    ]
    assert different == []


def test_routed_thermo_unlinks_as_before(recording):
    """``stage_link`` stays with ``_set``, where a false value unlinks."""
    case = next(c for c in recording["cases"] if c["key"].endswith(" unlink"))
    assert any(
        call[:2] == ["call", "specimen.stage.unlink"] for call in case["routed"][1]
    )


def test_compustage_axis_restriction_is_exercised(recording):
    """An inserted objective drops z and tilt (the compustage's ``a``) from an
    absolute move, on both sides (FIB-640)."""
    dropped = [
        c
        for c in recording["cases"]
        if "fm=True absolute" in c["key"]
        and any(
            call[1].endswith("absolute_move")
            and call[2][0]["z"] is None
            and call[2][0]["a"] is None
            for call in c["new"][1]
        )
    ]
    assert dropped


def test_offset_stage_restores_working_distance_after_an_absolute_move(recording):
    case = next(
        c
        for c in recording["cases"]
        if c["key"].startswith("compustage=False") and " absolute " in c["key"]
    )
    writes = [call[1] for call in case["new"][1] if call[0] == "set"]
    assert writes == ["connection.beams.electron_beam.working_distance.value"]


def test_each_stage_gets_its_driver_class_and_axes(recording):
    stage, compustage = recording["facts"]["stage"], recording["facts"]["compustage"]
    assert stage["class"] == "AutoscriptStage"
    assert stage["axes"] == ["x", "y", "z", "r", "t"]
    assert stage["parameters"] == ["homed", "linked", "position"]
    assert stage["link_available"] is True
    assert compustage["class"] == "AutoscriptCompustage"
    assert compustage["axes"] == ["x", "y", "z", "t"]
    assert compustage["parameters"] == ["homed", "position"]
    assert compustage["link_available"] is False


@pytest.mark.parametrize("kind", ["stage", "compustage"])
def test_new_api_refuses_an_out_of_limit_move_before_any_sdk_call(recording, kind):
    facts = recording["facts"][kind]
    assert facts["refused"] is not None and "x=1" in facts["refused"]
    assert not any(call[1].endswith("_move") for call in facts["calls_after_refusal"])
    assert facts["moved"][0] == pytest.approx(5e-5)
