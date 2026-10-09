"""Thermo's milling methods make the same SDK calls through the milling service.

``AutoScriptMilling`` holds the code ``ThermoMicroscope`` milled with. Each case runs
a milling method on a microscope with the service built as connect builds it, over a
fake AutoScript client that records every SDK call and write, and requires the
result, the calls in their order and the logged messages that the microscope's own
code gave before it moved into the service (``fixtures/autoscript_milling_calls.json``).
``finish_milling`` is the change: it puts back what ``setup_milling`` found, field of
view included. The recording runs in its own interpreter
(``tests/fixtures/autoscript_milling_parity.py``). Nothing here has run on an
instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.demoted_messages import as_logged_now

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_milling_parity.py"
OLD_CALLS = Path(__file__).parent / "fixtures" / "autoscript_milling_calls.json"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_milling") / "recording.json"
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


def test_the_recording_makes_sdk_calls(recording):
    cases = recording["cases"]
    assert len(cases) == 16
    assert all(c["new"][1] for c in cases)
    raised = [c["key"] for c in cases if str(c["new"][0]).startswith("EXC")]
    assert raised == []


def test_the_service_makes_the_same_sdk_calls_logs_and_results(recording):
    old = {
        key: [result, calls, as_logged_now(messages)]
        for key, (result, calls, messages) in json.loads(OLD_CALLS.read_text()).items()
    }
    assert [c["key"] for c in recording["cases"]] == list(old)
    different = [
        {"key": c["key"], "old": old[c["key"]], "new": c["new"]}
        for c in recording["cases"]
        if c["new"] != old[c["key"]]
    ]
    assert different == [], json.dumps(different[:2], indent=1)


def test_the_old_methods_go_through_the_service(recording):
    facts = recording["facts"]
    assert facts["is_autoscript"]
    assert facts["roles"] == ["ion", "electron"]
    assert facts["used"] == [
        "_setup",
        "_clear",
        "_draw",
        "_start",
        "_estimate",
        "_clear",
    ]


def test_finish_milling_puts_back_what_setup_found(recording):
    facts = recording["facts"]
    assert facts["during"] != facts["before"]
    assert facts["after"] == facts["before"]
    writes = [entry[1] for entry in facts["finish"][1] if entry[0] in ("set", "call")]
    assert writes == [
        "connection.patterning.clear_patterns",
        "connection.beams.ion_beam.high_voltage.value",
        "connection.beams.ion_beam.beam_current.value",
        "connection.beams.ion_beam.horizontal_field_width.value",
        "connection.patterning.mode",
    ]
    # Serial is still what finishing leaves, as before
    assert ["set", "connection.patterning.mode", "Serial"] in facts["finish"][1]


def test_an_imaging_current_given_still_wins(recording):
    facts = recording["facts"]
    assert facts["given"] == [30e3, 1e-10, facts["before"][2]]


def test_without_an_ion_beam_there_is_no_service(recording):
    assert recording["facts"]["no_ion"]


def test_thermo_mills_with_the_settings_it_says(recording):
    facts = recording["facts"]
    assert facts["supported"] == [
        "application_file",
        "hfw",
        "milling_channel",
        "milling_current",
        "milling_voltage",
        "patterning_mode",
    ]
    assert facts["setup_reads"] == facts["supported"]
    assert facts["application_files"] == ["Si", "Si-ccs", "Si-multipass", "Al"]
    assert (
        "TopToBottom" in facts["scan_directions"]
    )  # the fallback a pattern draws with
