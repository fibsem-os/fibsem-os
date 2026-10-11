"""Thermo's spot burn through the spot burn service makes the SDK calls it always made.

Each case burns on a ThermoMicroscope with the service built as connect builds it,
over the fake AutoScript client of ``autoscript_beam_parity.py``, which records every
SDK call and write. The calls, log lines and progress are pinned to what the
microscope's own burn made before the service replaced it (it was checked against the
service call for call until it was removed). The recording runs in its own interpreter
(``tests/fixtures/autoscript_spot_burn_parity.py``). Nothing here has run on an
instrument.
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


BEAM = "connection.beams.ion_beam"
BURN_AT = [["set", f"{BEAM}.beam_current.value", 1e-09]]
PUT_BACK = [
    ["call", f"{BEAM}.scanning.mode.set_full_frame", [], "{}"],
    ["set", f"{BEAM}.beam_current.value", 2e-11],
]
BURNING = (
    "burning spot {}: Point(x={}, y={}, name=None), exposure time: {}, "
    "milling current: 1e-09"
)


def _spot(x, y):
    return [
        ["call", f"{BEAM}.blank", [], "{}"],
        ["call", f"{BEAM}.scanning.mode.set_spot", [], f"{{'x': {x}, 'y': {y}}}"],
        ["call", f"{BEAM}.unblank", [], "{}"],
    ]


# Per case: the SDK calls in order, the log lines, and the progress reported as
# (status, current point, seconds left on it).
PINNED = {
    "two points": (
        BURN_AT + _spot(0.2, 0.3) + _spot(0.7, 0.4) + PUT_BACK,
        [
            ["INFO", BURNING.format(1, 0.2, 0.3, 2.0)],
            ["INFO", BURNING.format(2, 0.7, 0.4, 2.0)],
        ],
        [
            ("burning", 0, 2.0),
            ("burning", 1, 1.0),
            ("burning", 1, 0.0),
            ("burning", 2, 1.0),
            ("burning", 2, 0.0),
            ("finished", 2, None),
        ],
    ),
    "a point outside the image": (
        BURN_AT + _spot(0.5, 0.5) + PUT_BACK,
        [
            [
                "WARNING",
                "Skipping 1 spot burn coordinate(s) outside image bounds (0-1): "
                "[Point(x=1.5, y=0.5, name=None)]",
            ],
            ["INFO", BURNING.format(1, 0.5, 0.5, 2.0)],
        ],
        [
            ("burning", 0, 2.0),
            ("burning", 1, 1.0),
            ("burning", 1, 0.0),
            ("finished", 1, None),
        ],
    ),
    "one point, one second": (
        BURN_AT + _spot(0.5, 0.5) + PUT_BACK,
        [["INFO", BURNING.format(1, 0.5, 0.5, 1.0)]],
        [("burning", 0, 1.0), ("burning", 1, 0.0), ("finished", 1, None)],
    ),
    "stopped before the first point": (
        BURN_AT + PUT_BACK,
        [["INFO", "Spot burn cancelled before point 1/1."]],
        [("burning", 0, 2.0), ("cancelled", 1, None)],
    ),
}


def test_every_case_is_recorded(recording):
    assert [c["key"] for c in recording["cases"]] == KEYS


@pytest.mark.parametrize("key", KEYS)
def test_a_burn_makes_the_calls_logs_and_reports_it_always_made(recording, key):
    result, calls, messages, reports = _case(recording, key)["burn"]
    pinned_calls, pinned_messages, pinned_reports = PINNED[key]
    assert result is None
    assert calls == pinned_calls
    assert messages == pinned_messages
    assert [
        (r["status"], r["current_point"], r["remaining_time"]) for r in reports
    ] == pinned_reports


def test_a_burn_parks_blanked_and_puts_the_beam_back(recording):
    writes = [
        entry[1]
        for entry in _case(recording, "two points")["burn"][1]
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
