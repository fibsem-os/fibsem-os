"""Thermo's imaging methods make the same SDK calls through the beam commands.

``AutoscriptBeam``'s ``acquire``, ``last_image``, ``autocontrast`` and ``auto_focus``
are ``ThermoMicroscope``'s methods moved onto the beam. Each case runs an old method
on a microscope with no beam devices, and the same call on one whose beams are built
as connect builds them, both over a fake AutoScript client that records every SDK
call and write, and requires the same result, the same calls in the same order
(channel selection, scan area, hfw, grab, image read, autofunction) and the same
logged messages. The recording runs in its own interpreter
(``tests/fixtures/autoscript_imaging_parity.py``). Nothing here has run on an
instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_imaging_parity.py"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_imaging") / "recording.json"
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


def test_the_recording_makes_sdk_calls_on_both_beams(recording):
    cases = recording["cases"]
    assert len(cases) == 21
    assert any(c["key"].startswith("ION ") for c in cases)
    assert all(c["old"][1] for c in cases if c["key"] != "acquire nothing")


def test_the_old_methods_did_not_raise(recording):
    raised = [
        c["key"]
        for c in recording["cases"]
        if isinstance(c["old"][0], str) and c["old"][0].startswith("EXC")
    ]
    assert raised == ["acquire nothing"]


def test_the_beam_commands_make_the_same_sdk_calls_logs_and_results(recording):
    different = [c for c in recording["cases"] if c["old"] != c["new"]]
    assert different == [], json.dumps(different[:2], indent=1)


def test_the_old_methods_go_through_the_beams(recording):
    assert recording["facts"]["used"] == [
        "electron.acquire",
        "ion.acquire",
        "ion.last_image",
        "electron.autocontrast",
        "ion.auto_focus",
    ]


def test_each_vendor_call_holds_the_imaging_channel(recording):
    assert recording["facts"]["held"] == [
        ["grab_frame", True],
        ["grab_frame", True],
        ["get_image", True],
        ["run_auto_cb", True],
        ["run_auto_focus", True],
    ]


def test_an_autofunction_on_an_area_scans_it_then_the_full_frame(recording):
    case = next(c for c in recording["cases"] if c["key"] == "ION auto focus area")
    calls = [call[1].rsplit(".", 1)[-1] for call in case["new"][1]]
    assert calls == [
        "set_active_view",
        "set_active_device",
        "set_reduced_area",
        "run_auto_focus",
        "set_full_frame",
    ]


@pytest.mark.parametrize("beam", ["ELECTRON", "ION"])
def test_both_beams_have_the_imaging_commands(recording, beam):
    commands = recording["facts"]["commands"][beam]
    for name in ("acquire", "last_image", "autocontrast", "auto_focus"):
        assert name in commands


def test_a_beam_refuses_settings_for_the_other_beam(recording):
    assert recording["facts"]["other_beam"] == (
        "ion can't acquire an image for the ELECTRON beam"
    )


def test_a_disabled_column_has_no_beam(recording):
    assert recording["facts"]["ion_disabled"] == ["ELECTRON"]
