"""The AutoScript FM drivers make the SDK calls Thermo's old FM class made.

``fibsem.drivers.autoscript.devices`` is the old ``ThermoFisherFluorescenceMicroscope``'s
parts moved onto the FM devices. Each case runs a device call over a fake AutoScript
client that records every SDK call, read and write with the view that was active when it
was made, and compares it with the pin of the matching old call: its result and SDK log,
recorded from the old class over the same fake before it was removed
(``tests/fixtures/autoscript_fm_old_pins.json``; for the FM API cases, the changes, the
settings read and the view hand-backs). Cases: every camera,
light source, filter set and objective parameter and command, acquiring a channel
(fluorescence, reflection, the current settings) and live view, each from the beam view
and from the FM view, with the objective in and out, and the filter on fluorescence and
on reflection.

What has to match:

- the result, or the exception, of every case;
- for the parts, every call in the same order, the channel's own
  ``get_active_view``/``set_active_view``/``set_active_device`` included. The one
  difference is a write the old setter checks first (exposure time, binning): the
  old setter hands the view back between the check and the write, and the device
  holds it across both, so only the channel calls between them differ;
- for an acquisition and live view, every change (each ``set`` and ``call``) in the
  same order, and the same settings read for the metadata. The reads come in another
  order and fewer times: the old metadata reads the binning nine times per image;
  live view's stop no longer switches the light off twice;
- every FM call made with the FM's view selected, and the view left where the old
  code left it.

The fake SDK has to be in place before ``fibsem.fm.autoscript`` is first imported, so
the recording runs in its own interpreter (``tests/fixtures/autoscript_fm_parity.py``).
Nothing here has run on an instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_fm_parity.py"
# What the old class did in each case, keyed as the cases are.
PINS = json.loads(
    (Path(__file__).parent / "fixtures" / "autoscript_fm_old_pins.json").read_text()
)

PARTS = ("camera", "light", "filter", "objective")
# The setters that check before writing, so hold the view across the check.
CHECKED_WRITES = ("camera set exposure_time", "camera set binning")
FM_VIEW = 3


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_fm") / "recording.json"
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


@pytest.fixture(scope="module")
def recording(recorded):
    """Each case with the old class's pinned result, log and view as ``old``."""
    assert {case["key"] for case in recorded["cases"]} == set(PINS["cases"])
    return [{**case, "old": PINS["cases"][case["key"]]} for case in recorded["cases"]]


@pytest.fixture(scope="module")
def api(recorded):
    """Each FM API case with what the old class's pin keeps of it as ``old``."""
    assert {case["key"] for case in recorded["api"]} == set(PINS["api"])
    return [{**case, "old": PINS["api"][case["key"]]} for case in recorded["api"]]


@pytest.fixture(scope="module")
def facts(recorded):
    return recorded["facts"]


def _name(case):
    return case["key"].split(" ", 3)[3]


def _without_channel(log):
    return [entry for entry in log if entry[0] != "chan"]


LIGHT_OFF = "detector.camera_settings.emission.stop"


def _changes(log):
    """Each set and call, in order. The old live view switches the light off twice
    when it stops, so a light-off straight after another counts once."""
    changes = [entry for entry in log if entry[0] in ("set", "call")]
    return [
        entry
        for i, entry in enumerate(changes)
        if not (i and entry[1] == LIGHT_OFF and changes[i - 1] == entry)
    ]


def _reads(log):
    return sorted({entry[1] for entry in log if entry[0] == "get"})


def test_the_recording_covers_every_part_and_both_views(recording):
    assert len(recording) > 300
    assert {case["group"] for case in recording} == {*PARTS, "acquire", "live"}
    assert {case["key"].split(" ")[0] for case in recording} == {"view=1", "view=3"}
    assert sum(len(case["old"][1]) for case in recording) > 3000
    grabs = [case for case in recording if case["group"] == "acquire"]
    assert all(
        any(entry[1] == "imaging.grab_frame" for entry in case["old"][1])
        for case in grabs
    )


def test_every_case_has_the_same_result(recording):
    differ = [
        (case["key"], case["old"][0], case["new"][0])
        for case in recording
        if case["old"][0] != case["new"][0]
    ]
    assert differ == []


def test_some_cases_raise_on_both_sides(recording):
    raised = {
        _name(case)
        for case in recording
        if str(case["old"][0]).startswith("EXC") and case["old"][0] == case["new"][0]
    }
    assert {
        "camera set exposure_time 100.0",
        "camera set binning 3",
        "camera set offset -1.0",
        "objective move_absolute 0.0095",
    } <= raised


def test_the_parts_make_the_same_calls(recording):
    differ = [
        case["key"]
        for case in recording
        if case["group"] in PARTS
        and not _name(case).startswith(CHECKED_WRITES)
        and case["old"][1] != case["new"][1]
    ]
    assert differ == []


def test_a_checked_write_differs_only_in_holding_the_view(recording):
    checked = [c for c in recording if _name(c).startswith(CHECKED_WRITES)]
    assert checked
    differ = [
        case["key"]
        for case in checked
        if _without_channel(case["old"][1]) != _without_channel(case["new"][1])
    ]
    assert differ == []


def test_acquisitions_make_the_same_changes_in_the_same_order(recording):
    acquisitions = [c for c in recording if c["group"] in ("acquire", "live")]
    differ = [
        case["key"]
        for case in acquisitions
        if _changes(case["old"][1]) != _changes(case["new"][1])
    ]
    assert differ == []


def test_acquisitions_read_the_same_settings(recording):
    acquisitions = [c for c in recording if c["group"] in ("acquire", "live")]
    differ = [
        (case["key"], _reads(case["old"][1]), _reads(case["new"][1]))
        for case in acquisitions
        if _reads(case["old"][1]) != _reads(case["new"][1])
    ]
    assert differ == []


def test_live_view_starts_and_stops_as_the_old_fast_acquisition(recording):
    live = next(c for c in recording if _name(c) == "live 3 frames")
    calls = [entry[1] for entry in _changes(live["new"][1]) if entry[0] == "call"]
    # The channel's power is set with the light on, then live view starts.
    assert calls == [
        "detector.camera_settings.emission.start",
        LIGHT_OFF,
        "detector.camera_settings.emission.start",
        "imaging.start_acquisition",
        "imaging.get_image",
        "imaging.get_image",
        "imaging.get_image",
        LIGHT_OFF,
        "imaging.stop_acquisition",
    ]
    # Each frame re-selects the FM first, as the old loop does: something else may
    # have taken the view since the last one.
    for side in ("old", "new"):
        log = live[side][1]
        frames = [i for i, entry in enumerate(log) if entry[1] == "imaging.get_image"]
        assert all(
            log[i - 2][1:3] == ["imaging.set_active_view", [FM_VIEW]] for i in frames
        ), side
    old_calls = [entry[1] for entry in live["old"][1] if entry[0] == "call"]
    assert old_calls.count(LIGHT_OFF) == 3  # the power write's, then twice on stop


def test_every_fm_call_is_made_on_the_fm_view(recording):
    off_view = [
        (case["key"], side, entry)
        for case in recording
        for side in ("old", "new")
        for entry in _without_channel(case[side][1])
        if entry[3] != FM_VIEW
    ]
    assert off_view == []


def test_the_view_is_left_where_the_old_code_left_it(recording):
    differ = [case["key"] for case in recording if case["old"][2] != case["new"][2]]
    assert differ == []
    # A part's read or write hands the beam view back; live view keeps the FM.
    beam_reads = [
        c
        for c in recording
        if c["key"].startswith("view=1") and c["group"] in PARTS + ("acquire",)
    ]
    assert all(case["new"][2] == 1 for case in beam_reads)


def test_the_drivers_are_the_fm_devices(facts):
    assert facts["devices"] == {
        "fm": "AutoscriptFM",
        "camera": "AutoscriptFMCamera",
        "light_source": "AutoscriptFMLightSource",
        "filter_set": "AutoscriptFMFilterSet",
        "objective": "AutoscriptFMObjective",
    }
    assert facts["parameters"]["camera"] == [
        "binning",
        "display_transform",
        "exposure_time",
        "gain",
        "mount_transform",
        "offset",
        "pixel_size",
        "resolution",
    ]
    assert facts["parameters"]["objective"] == [
        "limit_position",
        "magnification",
        "numerical_aperture",
        "position",
        "state",
    ]
    assert {"acquire_frame", "start_live", "stop_live"} <= set(facts["commands"]["fm"])


def test_the_fm_shares_the_microscope_lock_with_the_beams(facts):
    assert facts["shares_the_microscope_lock"]


def test_a_read_on_the_fm_view_does_not_wait_for_the_lock(facts):
    """Live view takes the lock every frame; the old scope skips it when the FM
    already has the view, which is what keeps the objective movable while live."""
    assert facts["reads_on_the_fm_view_without_the_lock"]
    assert facts["reads_on_the_beam_view_wait_for_the_lock"]


# -- the FM API over the devices, against the old Thermo class's pins ------------------

# Limits and choices are read once, when the devices connect, and cached, as for every
# device; the old class read them again on each use.
CACHED_AT_CONNECT = (
    "detector.brightness.limits",
    "detector.camera_settings.binning.available_values",
    "detector.camera_settings.exposure_time.limits",
    "detector.camera_settings.focus.limits",
)


def _restores(log):
    """How many times the view was handed back from the FM."""
    return sum(
        1
        for entry in log
        if entry[1] == "imaging.set_active_view" and entry[2] != [FM_VIEW]
    )


def test_the_api_cases_cover_acquisitions_and_live_view(api):
    names = {_name(case) for case in api}
    assert {
        "acquire_image fluorescence",
        "acquire_z_stack by channel",
        "acquire_z_stack by z level",
        "tileset",
        "live 3 frames",
        "objective moves",
        "set_channel fluorescence",
    } <= names
    assert len(api) > 150


def test_the_api_returns_the_same(api):
    differ = [
        (case["key"], case["old"]["result"], case["new"][0])
        for case in api
        if case["old"]["result"] != case["new"][0]
    ]
    assert differ == []


def test_live_view_through_the_api_stops(api):
    live = [case for case in api if _name(case).startswith("live")]
    assert live and all(case["new"][0] is True for case in live)


def test_the_api_makes_the_same_changes_in_the_same_order(api):
    differ = [
        case["key"]
        for case in api
        if case["old"]["changes"] != _changes(case["new"][1])
    ]
    assert differ == []


def test_the_api_reads_the_same_settings_but_limits_once(api):
    differ = []
    for case in api:
        old, new = set(case["old"]["reads"]), set(_reads(case["new"][1]))
        if not new <= old or not (old - new) <= set(CACHED_AT_CONNECT):
            differ.append((case["key"], sorted(old ^ new)))
    assert differ == []


def test_the_api_makes_every_fm_call_on_the_fm_view(api):
    off_view = [
        (case["key"], entry)
        for case in api
        for entry in _without_channel(case["new"][1])
        if entry[3] != FM_VIEW
    ]
    assert off_view == []


def test_the_api_leaves_the_view_where_it_did_and_never_hands_it_back_more(api):
    assert [c["key"] for c in api if c["old"]["view"] != c["new"][2]] == []
    more = [
        (case["key"], case["old"]["restores"], _restores(case["new"][1]))
        for case in api
        if _restores(case["new"][1]) > case["old"]["restores"]
    ]
    assert more == []
    # A tileset holds the FM's view for the whole run: one hand-back at the end.
    tilesets = [
        c for c in api if _name(c) == "tileset" and c["key"].startswith("view=1")
    ]
    assert tilesets and all(_restores(c["new"][1]) == 1 for c in tilesets)


def test_a_thermo_microscope_builds_its_fm_from_the_devices(facts):
    assert facts["thermo_microscope"] == {
        "fm": "DeviceThermoFisherFluorescenceMicroscope",
        "devices": ["camera", "filter_set", "fm", "light_source", "objective"],
        # Its own worker pulls live view, so nothing needs to stop it by itself.
        "live_timeout": None,
        "shares_the_microscope_lock": True,
        "parent": True,
        # As the configuration's fm entry states it.
        "mount_transform": "flip-y",
    }
