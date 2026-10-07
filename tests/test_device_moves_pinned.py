"""Today's device moves, pinned on a compustage and on an offset mount.

The device moves are one path: a device is its pose plus its origin, on a compustage
as on an offset mount. This file pins what that path commands, so a change to it is a
change someone meant. The compustage cases were recorded first from its old route, and
re-pinned when that route was removed: the end states did not move, the flip now names
x, y and z at their current values as well as r and t, a move to a device the stage is
already at commands nothing, and `move_to_device("FIBSEM")` keeps a pose the beams image
from.

Each case records, on the Demo backend:

* **the commands**: every `move_stage_absolute` and `move_stage_relative` with the axes
  it set, and every objective insert and retract, in order. These are what a backend
  turns into SDK calls; `move_to_device` and the conversions are shared code above them,
  so the same commands here mean the same SDK calls on ThermoFisher;
* **where it ends**: the stage position, its orientation, which device it is at and the
  objective state, or the error a refused move raises;
* **the conversions**: `to_device`, `get_target_position` with an orientation only, the
  lamella conversions in `autolamella.poses`, and `get_current_device`, `is_at_device` and
  `get_device_imaging_state` at each start.

Mounts: `sim-arctis` (a compustage), `sim-iflm` (an FM 48.8 mm along x), and `sim-arctis`
with an objective that also images from SEM and MILLING. Starts: each orientation the
mount has, off centre, and again with a tilt a degree off nominal for the calls
that may carry a tilt across.

The expected values are in ``tests/fixtures/device_move_pins.json``, recorded from main
at the commit that added this file. A change that is meant to move one regenerates the
file with

    PYTHONPATH=. python tests/test_device_moves_pinned.py --write

and says in its pull request which values moved and why.
"""

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import _to_fluorescence, _to_milling
from fibsem.structures import FibsemStagePosition

PINS_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "device_move_pins.json")

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")

MOUNTS = ("arctis", "iflm", "arctis_beam_side")
DEVICES = ("FIBSEM", "FM")
ORIENTATIONS = ("SEM", "FIB", "MILLING", "FM")

# Off centre, so a half turn or a traverse shows in x and y.
START_XY = (100e-6, 50e-6)
AXES = ("x", "y", "z", "r", "t")


def _microscope(mount: str):
    config = IFLM_CONFIG if mount == "iflm" else ARCTIS_CONFIG
    microscope, _ = utils.setup_session(config_path=config, setup_logging=False)
    if mount == "arctis_beam_side":
        microscope.system.stage.devices["FM"].available_orientations = [
            "FM",
            "SEM",
            "MILLING",
        ]
    return microscope


def _starts(mount: str) -> List[str]:
    """Where a case starts: an orientation, `@FM` for at the offset FM, `+1` for a tilt
    a degree off nominal."""
    if mount == "iflm":
        poses = ["SEM", "FIB", "MILLING", "FIB@FM"]
    else:
        poses = ["SEM", "FIB", "MILLING", "FM"]
    return poses + [f"{pose}+1" for pose in poses]


def _start_position(microscope, start: str) -> FibsemStagePosition:
    off_nominal = start.endswith("+1")
    name = start[:-2] if off_nominal else start
    orientation, _, device = name.partition("@")
    pose = microscope.get_orientation(orientation)
    position = FibsemStagePosition(
        x=START_XY[0], y=START_XY[1], z=0.0, r=pose.r, t=pose.t
    )
    if device:
        position = position + microscope._device_translation("FIBSEM", device)
    if off_nominal:
        position.t += np.radians(1.0)
    return position


def _go(microscope, start: str) -> FibsemStagePosition:
    position = _start_position(microscope, start)
    microscope.move_stage_absolute(position)
    if microscope.fm is not None and "@FM" in start:
        microscope.fm.objective.insert()
    return microscope.get_stage_position()


def _point(position: FibsemStagePosition) -> Dict[str, Any]:
    return {axis: getattr(position, axis) for axis in AXES}


def _record_commands(microscope) -> List[Any]:
    """Log each raw stage move and objective move, in order, on this instance."""
    log: List[Any] = []

    def wrap(name, method):
        def recorded(position, *args, **kwargs):
            log.append([name, _point(position)])
            return method(position, *args, **kwargs)

        return recorded

    microscope.move_stage_absolute = wrap("absolute", microscope.move_stage_absolute)
    microscope.move_stage_relative = wrap("relative", microscope.move_stage_relative)

    if microscope.fm is not None:
        objective = microscope.fm.objective
        insert, retract = objective.insert, objective.retract

        def recorded_insert(*args, **kwargs):
            log.append(["insert"])
            return insert(*args, **kwargs)

        def recorded_retract(*args, **kwargs):
            log.append(["retract"])
            return retract(*args, **kwargs)

        objective.insert = recorded_insert
        objective.retract = recorded_retract
    return log


def _state(microscope) -> Dict[str, Any]:
    state = {
        "position": _point(microscope.get_stage_position()),
        "orientation": microscope.get_stage_orientation(),
        "device": microscope.get_current_device(),
    }
    if microscope.fm is not None:
        state["objective"] = microscope.fm.objective.state
    return state


def _attempt(fn) -> Any:
    try:
        result = fn()
    except Exception as e:  # a refusal is behaviour too
        return {"raises": type(e).__name__}
    if isinstance(result, FibsemStagePosition):
        return _point(result)
    if hasattr(result, "name"):
        return result.name
    return result


# -- the moves -------------------------------------------------------------------------


def _move_calls(start: str):
    """Every call from a nominal pose; from a tilt off nominal, only the calls that may
    carry it across (no orientation asked for)."""
    off_nominal = start.endswith("+1")
    for device in DEVICES:
        yield f"move_to_device({device},None)"
        yield f"move_to_microscope({device})"
        if off_nominal:
            continue
        for orientation in ORIENTATIONS:
            yield f"move_to_device({device},{orientation})"


def _call(microscope, call: str):
    name, _, args = call.partition("(")
    args = [None if a == "None" else a for a in args.rstrip(")").split(",")]
    return getattr(microscope, name)(*args)


def _move_case(mount: str, start: str, call: str) -> Dict[str, Any]:
    microscope = _microscope(mount)
    _go(microscope, start)
    log = _record_commands(microscope)
    result: Dict[str, Any] = {}
    try:
        _call(microscope, call)
    except Exception as e:  # a refusal is behaviour too
        result["raises"] = type(e).__name__
    result["commands"] = log
    result.update(_state(microscope))
    return result


# -- the conversions and questions -------------------------------------------------------


def _queries_case(mount: str, start: str) -> Dict[str, Any]:
    microscope = _microscope(mount)
    position = _go(microscope, start)
    result: Dict[str, Any] = {"orientation": microscope.get_stage_orientation(position)}
    result["current_device"] = _attempt(lambda: microscope.get_current_device(position))
    for device in DEVICES:
        result[f"is_at_device({device})"] = _attempt(
            lambda: microscope.is_at_device(device, position)
        )
        result[f"imaging_state({device})"] = _attempt(
            lambda: microscope.get_device_imaging_state(device, position)
        )
        for orientation in (None,) + ORIENTATIONS:
            result[f"to_device({device},{orientation})"] = _attempt(
                lambda: microscope.to_device(deepcopy(position), device, orientation)
            )
    for orientation in ORIENTATIONS:
        result[f"get_target_position({orientation})"] = _attempt(
            lambda: microscope.get_target_position(deepcopy(position), orientation)
        )
    result["lamella_to_fluorescence"] = _attempt(
        lambda: _to_fluorescence(microscope, deepcopy(position))
    )
    result["lamella_to_milling"] = _attempt(
        lambda: _to_milling(microscope, deepcopy(position))
    )
    return result


# -- the cases ---------------------------------------------------------------------------


def _case_ids():
    for mount in MOUNTS:
        for start in _starts(mount):
            yield f"{mount}|{start}|queries"
            for call in _move_calls(start):
                yield f"{mount}|{start}|{call}"


def run_case(case_id: str) -> Any:
    mount, start, what = case_id.split("|")
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        if what == "queries":
            return _queries_case(mount, start)
        return _move_case(mount, start, what)
    finally:
        logging.disable(previous)


def _load_pins() -> Dict[str, Any]:
    with open(PINS_PATH) as f:
        return json.load(f)


def _assert_same(actual, expected, where=""):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert sorted(actual) == sorted(expected), where
        for key in expected:
            _assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{where}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), where
    else:
        assert actual == expected, where


CASE_IDS = list(_case_ids())


def test_every_case_is_pinned():
    assert sorted(_load_pins()) == sorted(CASE_IDS)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_device_moves_are_unchanged(case_id):
    expected = _load_pins()[case_id]
    actual = json.loads(json.dumps(run_case(case_id)))
    _assert_same(actual, expected, case_id)


def _write_pins():
    pins = {case_id: run_case(case_id) for case_id in CASE_IDS}
    with open(PINS_PATH, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(pins)} cases to {PINS_PATH}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_pins()
