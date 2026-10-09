"""Thermo's spot burn makes the same SDK calls through the spot burn service.

Each case burns on a ThermoMicroscope with the service built as connect builds it, and
again on the same microscope without one, so that the microscope's own spot burn runs,
over the fake AutoScript client of ``autoscript_beam_parity.py``, which records every
SDK call and write. The two must make the same calls in the same order, log the same
messages and report the same progress. The recording runs in its own interpreter
(``tests/fixtures/autoscript_spot_burn_parity.py``); the exposures are whole seconds,
which the microscope's own burn counted down in. Nothing here has run on an instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_spot_burn_parity.py"
KEYS = [
    "two points",
    "a point outside the image",
    "one point, one second",
    "stopped before the first point",
]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_spot_burn") / "recording.json"
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


def _case(recording, key):
    return next(c for c in recording["cases"] if c["key"] == key)


def test_thermo_burns_through_the_service(recording):
    facts = recording["facts"]
    assert facts["is_spot_burn"]
    assert facts["roles"] == ["ion"]
    assert facts["ion"] == "ion"
    assert facts["supported"] == ["coordinates", "exposure_time", "milling_current"]
    assert facts["no_ion"]


def test_every_case_burns_and_makes_sdk_calls(recording):
    assert [c["key"] for c in recording["cases"]] == KEYS
    for case in recording["cases"]:
        result, calls, _, reports = case["new"]
        assert result is None, case["key"]
        assert calls, case["key"]
        assert reports[-1]["status"] in ("finished", "cancelled"), case["key"]


@pytest.mark.parametrize("key", KEYS)
def test_the_service_makes_the_same_calls_logs_and_reports(recording, key):
    case = _case(recording, key)
    assert case["new"] == case["old"]


def test_a_burn_parks_blanked_and_puts_the_beam_back(recording):
    writes = [
        entry[1]
        for entry in _case(recording, "two points")["new"][1]
        if entry[0] in ("set", "call")
    ]
    beam = "connection.beams.ion_beam"
    point = [f"{beam}.blank", f"{beam}.scanning.mode.set_spot", f"{beam}.unblank"]
    assert writes == [
        f"{beam}.beam_current.value",
        *point,
        *point,
        f"{beam}.scanning.mode.set_full_frame",
        f"{beam}.beam_current.value",
    ]
