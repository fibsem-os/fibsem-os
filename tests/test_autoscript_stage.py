"""The AutoScript stage drivers make the SDK calls Thermo's stage code makes today.

``AutoscriptStage`` and ``AutoscriptCompustage`` are Thermo's stage code moved onto the
``Stage`` device. Each case runs an old ``ThermoMicroscope`` call on one microscope and
the matching driver call on another, both over a fake AutoScript client that records
every SDK call and write, and requires the same calls, arguments, order and result.
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
    return json.loads(out.read_text())


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
    different = [c for c in recording["cases"] if c["old"] != c["new"]]
    assert different == [], json.dumps(different[:3], indent=1)


def test_compustage_axis_restriction_is_exercised(recording):
    """An inserted objective drops z from an absolute move, on both sides."""
    dropped = [
        c
        for c in recording["cases"]
        if "fm=True absolute" in c["key"]
        and any(
            call[1].endswith("absolute_move") and call[2][0]["z"] is None
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
