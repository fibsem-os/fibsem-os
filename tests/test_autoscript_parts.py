"""Thermo's chamber and manipulator as devices make the SDK calls the old code makes.

``AutoscriptChamber`` and ``AutoscriptManipulator`` are ``ThermoMicroscope``'s chamber
branches and manipulator methods moved onto the devices. Each case runs an old call on
a microscope without the devices and on one that built them as connect does, both over
a fake AutoScript client that records every SDK call and write, and requires the same
result (or error), the same calls in the same order and the same logged messages.

Cases: the chamber keys, pump and vent; the manipulator keys, insert (each named
position, and an unknown one), retract, the raw moves, the corrected and offset moves,
which stay ``ThermoMicroscope``'s and move through the device, and the saved
positions. The recording runs in its own interpreter
(``tests/fixtures/autoscript_parts_parity.py``). Nothing here has run on an
instrument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fixtures" / "autoscript_parts_parity.py"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("autoscript_parts") / "recording.json"
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


def test_the_recording_makes_sdk_calls(recording):
    cases = recording["cases"]
    assert len(cases) > 20
    assert sum(len(c["old"][1]) for c in cases) > 20
    assert sum(len(c["old"][2]) for c in cases) > 15


def test_the_devices_make_the_same_sdk_calls_logs_and_results(recording):
    different = [c for c in recording["cases"] if c["old"] != c["new"]]
    assert different == [], json.dumps(different[:3], indent=1)


def test_the_old_errors_are_kept(recording):
    for key in ("insert BAD", "saved BAD"):
        assert _case(recording, key)["new"][0].startswith("EXC ValueError"), key


def test_the_calls_go_through_the_devices(recording):
    assert recording["facts"]["through_devices"] == [
        "chamber._pump",
        "chamber._vent",
        "manipulator._insert",
        "manipulator._move_relative",  # the corrected move
        "manipulator._move_absolute",  # the offset move
        "manipulator._retract",
    ]


def test_connect_builds_the_chamber_and_the_manipulator(recording):
    devices = recording["facts"]["devices"]
    assert devices["chamber"] == "AutoscriptChamber"
    assert devices["manipulator"] == "AutoscriptManipulator"
    assert devices["chamber_parameters"] == ["pressure", "state"]
    assert devices["manipulator_parameters"] == ["position", "state"]


def test_nothing_unfitted_is_built(recording):
    assert recording["facts"]["none_fitted"] == {
        "manipulator": True,
        "routed": False,
    }


def test_the_configuration_switches_parts_off(recording):
    """`hardware.devices` is an overlay on what the instrument has: an entry turns a
    fitted part off, so it is never built; a device no driver builds is left out with
    a warning."""
    facts = recording["facts"]["configured"]
    assert facts["devices"] == ["chamber"]
    assert facts["manipulator"] is True
    assert facts["routed"] is False


def test_a_required_device_that_cannot_be_built_fails_connect(recording):
    assert recording["facts"]["configured"]["required"] == (
        "Device 'laser' was not built: driver 'ThermoFisher' has no builder for a "
        "'laser' device."
    )


def test_connect_builds_the_parts_after_it_reads_what_is_fitted():
    """Whether a manipulator is fitted is read in
    ``_create_sample_stage``, so the parts are built after it."""
    import ast

    import fibsem

    source = (Path(fibsem.__file__).parent / "microscopes" / "autoscript.py").read_text(
        encoding="utf-8"
    )
    connect = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "connect_to_microscope"
    )
    order = [
        node.func.attr
        for node in ast.walk(connect)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in ("_build_beams", "_build_stage", "_create_sample_stage", "_build_parts")
    ]
    lines = {
        node.func.attr: node.lineno
        for node in ast.walk(connect)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert sorted(order) == sorted(
        ["_build_beams", "_build_stage", "_create_sample_stage", "_build_parts"]
    )
    assert lines["_create_sample_stage"] < lines["_build_parts"]
