"""Thermo's imaging methods make the same SDK calls through the beam commands.

``AutoscriptBeam``'s ``acquire``, ``last_image``, ``autocontrast`` and ``auto_focus``
are ``ThermoMicroscope``'s methods moved onto the beam, and its ``start_live`` and
``stop_live`` its live view worker, whose frames the old signals forward. Each case
runs a method on a microscope whose beams are built as connect builds them, over a
fake AutoScript client that records every SDK call and write, and requires the
result, the calls in order (channel selection, scan area, hfw, grab, image read,
autofunction) and the logged messages the old methods gave, recorded over the same
fake in ``tests/fixtures/autoscript_old_calls.json`` before they were deleted. The
recording runs in its own interpreter
(``tests/fixtures/autoscript_imaging_parity.py``). Nothing here has run on an
instrument.
"""

import ast
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests.fixtures.autoscript_recording import load
from tests.fixtures.demoted_messages import as_logged_now

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_imaging_parity.py"
RECORDED = Path(__file__).parent / "fixtures" / "autoscript_old_calls.json"


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
    recording = load(out.read_text())
    # the old code's side, recorded before it was deleted
    old = load(RECORDED.read_text())["imaging"]
    assert sorted(c["key"] for c in recording["cases"]) == sorted(old)
    for case in recording["cases"]:
        result, calls, messages = old[case["key"]]
        case["old"] = [result, calls, as_logged_now(messages)]
    return recording


def test_the_recording_makes_sdk_calls_on_both_beams(recording):
    cases = recording["cases"]
    assert len(cases) == 23
    assert any(c["key"].startswith("ION ") for c in cases)
    assert all(c["old"][1] for c in cases if c["key"] != "acquire nothing")


def test_the_old_methods_did_not_raise(recording):
    raised = [
        c["key"]
        for c in recording["cases"]
        if isinstance(c["old"][0], str) and c["old"][0].startswith("EXC")
    ]
    assert raised == ["acquire nothing"]


# The one deliberate difference from the old methods (FIB-1190): they overwrote the
# state's timestamp with AutoScript's acquisition_datetime string, so a float field
# held a string (FIB-487). The state now keeps its own, and the vendor time is the
# image's acquisition_datetime. Compared separately, below.
TIMES = ("timestamp", "acquisition_datetime")


def _image(result):
    """A recorded image, which the recording keeps as its dict's repr."""
    if isinstance(result, str) and result.startswith("{'shape'"):
        return ast.literal_eval(result)
    return None


def _without_times(case):
    result, *rest = case
    image = _image(result)
    if image is not None:
        result = {k: v for k, v in image.items() if k not in TIMES}
    return [result, *rest]


def test_the_beam_commands_make_the_same_sdk_calls_logs_and_results(recording):
    different = [
        c
        for c in recording["cases"]
        if _without_times(c["old"]) != _without_times(c["new"])
    ]
    assert different == [], json.dumps(different[:2], indent=1)


def _images(recording, side):
    images = (_image(c[side][0]) for c in recording["cases"])
    return [image for image in images if image is not None]


def test_the_vendor_time_is_the_acquisition_time_and_the_state_keeps_its_own(
    recording,
):
    # The fake frame's AutoScript time, read as this machine's local time.
    vendor = datetime(2026, 10, 5, 6, 0, 0).astimezone().isoformat()
    new = _images(recording, "new")
    assert len(new) == 10  # acquire settings, reduced, current, both, last; per beam
    assert {image["timestamp"] for image in new} == {0.0}  # the state, as read
    assert {image["acquisition_datetime"] for image in new} == {vendor}
    # What the old methods wrote in the state instead.
    old = _images(recording, "old")
    assert {image["timestamp"] for image in old} == {"2026-10-05 06:00:00"}


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
    for name in (
        "acquire",
        "last_image",
        "autocontrast",
        "auto_focus",
        "start_live",
        "stop_live",
    ):
        assert name in commands


def test_a_beam_refuses_settings_for_the_other_beam(recording):
    assert recording["facts"]["other_beam"] == (
        "ion can't acquire an image for the ELECTRON beam"
    )


def test_a_disabled_column_has_no_beam(recording):
    assert recording["facts"]["ion_disabled"] == ["ELECTRON"]


@pytest.mark.parametrize("beam", ["ELECTRON", "ION"])
def test_live_view_through_the_beam_makes_the_old_calls_and_frames(recording, beam):
    case = next(c for c in recording["cases"] if c["key"] == f"{beam} live")
    assert case["old"] == case["new"]
    frames, acquiring = case["new"][0]
    assert frames == [
        [[4, 6], "uint16"],  # the fast path's frames
        [[4, 6], "uint16"],
        [[4, 6], "uint8"],  # then a grab per pass
        [[4, 6], "uint8"],
    ]
    assert acquiring is False


def test_start_acquisition_runs_the_beams_live_view_and_stop_stops_it(recording):
    assert recording["facts"]["live"] == {
        "live": True,
        "acquiring": True,
        "ion": False,  # already acquiring: the second start does nothing
        "stopped": [False, False],
    }
