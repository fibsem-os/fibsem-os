"""Today's view-corrected stage moves, pinned before they move into a service.

The moves (``stable_move``, ``project_stable_move``, ``vertical_move``,
``move_to_orientation``, ``move_to_milling_angle``, ``safe_absolute_stage_movement``,
``fm_stable_move`` and ``project_fm_stable_move``) are shared code above the devices.
This file pins what they command, so moving that code keeps every command the same.
``move_to_device`` is pinned on its own in ``test_device_moves_pinned.py``.

Each case records:

* **the commands**, in order: every stage move the stage device's driver is asked for
  (absolute or relative, with the axes it set), every working distance written to a
  beam, and every objective insert and retract. Recorded at the driver, below the
  devices, so whatever calls the devices is pinned by what reaches the hardware;
* **the result**: what the call returned, or the error it raised, and where the
  stage ended;
* **the record**: the ``stage_moved`` event names the call made.

Backends: Demo on three mounts (the default rotation stage, ``sim-iflm`` with an FM
48.8 mm along x, and ``sim-arctis``, a compustage), and TESCAN over the recording fake
connection (``tests/fixtures/tescan_sdk.py``), where the commands are the SharkSEM
writes. ThermoFisher and Odemis run the same shared code over their own stage and beam
drivers, whose SDK calls are pinned per device (``test_autoscript_stage.py``,
``test_odemis_devices.py``), so the same device commands here mean the same SDK calls
there.

The expected values are in ``tests/fixtures/stage_movement_pins.json``, recorded from
main at the commit that added this file. A change that is meant to move one regenerates
the file with

    PYTHONPATH=. python tests/test_stage_movement_pins.py --write

and says in its pull request which values moved and why.
"""

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType, CameraImageTransform, FibsemStagePosition

PINS_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "stage_movement_pins.json"
)

MOUNTS = {
    "default": None,
    "iflm": os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml"),
    "arctis": os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
}
TESCAN_CONFIG = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")

# Off centre, so a rotation or a projection shows in x and y.
START = dict(x=100e-6, y=50e-6, z=1e-3)
BEAMS = (BeamType.ELECTRON, BeamType.ION)
DISPLACEMENTS = ((10e-6, 0.0), (0.0, 20e-6), (-15e-6, 7.5e-6))


def _plain(value: Any) -> Any:
    if isinstance(value, FibsemStagePosition):
        return {
            axis: None if getattr(value, axis) is None else float(getattr(value, axis))
            for axis in ("x", "y", "z", "r", "t")
        }
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating, int)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value if value is None or isinstance(value, str) else repr(value)


# -- Demo -----------------------------------------------------------------------------


class _Recorder:
    """Records the driver commands below the stage, beam and objective devices."""

    def __init__(self, microscope):
        self.commands: List[list] = []
        self.events: List[str] = []
        stage = microscope.devices["stage"]
        self._wrap(stage, "_move_absolute", lambda p: ["stage.absolute", _plain(p)])
        self._wrap(stage, "_move_relative", lambda p: ["stage.relative", _plain(p)])
        for name in ("electron", "ion"):
            beam = microscope.devices.get(name)
            param = None if beam is None else beam.parameters.get("working_distance")
            if param is not None:
                # The parameter holds its write, bound at connect.
                self._wrap(
                    param,
                    "_write",
                    lambda v, name=name: [f"{name}.working_distance", _plain(v)],
                )
        objective = microscope.devices.get("objective")
        if objective is not None:
            self._wrap(objective, "_insert", lambda: ["objective.insert"])
            self._wrap(objective, "_retract", lambda: ["objective.retract"])
        microscope.record_signal.connect(self._event)

    def _wrap(self, owner, name: str, describe: Callable) -> None:
        original = getattr(owner, name)

        def wrapper(*args):
            self.commands.append(describe(*args))
            return original(*args)

        setattr(owner, name, wrapper)

    def _event(self, kind: str, payload: Dict[str, Any]) -> None:
        if kind == "stage_moved":
            self.events.append(payload["move"])

    def clear(self) -> None:
        self.commands.clear()
        self.events.clear()


def _demo(mount: str):
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    path = MOUNTS[mount]
    if path is None:
        microscope, _ = utils.setup_session(
            manufacturer="Demo", ip_address="localhost", setup_logging=False
        )
    else:
        microscope, _ = utils.setup_session(config_path=path, setup_logging=False)
    return microscope, _Recorder(microscope)


def _orientations(microscope) -> List[str]:
    names = []
    for name in ("SEM", "FIB", "MILLING", "FM"):
        try:
            microscope.get_orientation(name)
        except ValueError:
            continue
        names.append(name)
    return names


def _place(microscope, orientation: str) -> None:
    """Put the stage at `orientation`, off centre, without going through the moves."""
    pose = microscope.get_orientation(orientation)
    stage = microscope.devices["stage"]
    position = FibsemStagePosition(r=pose.r, t=pose.t, **START)
    stage._move_absolute(position)
    microscope.get_stage_position()


def _run(microscope, recorder, call: Callable[[], Any]) -> Dict[str, Any]:
    recorder.clear()
    try:
        result = _plain(call())
        error = None
    except Exception as e:  # noqa: BLE001 - the refusal is part of the pin
        result = None
        error = f"{type(e).__name__}: {e}"
    return {
        "commands": deepcopy(recorder.commands),
        "events": list(recorder.events),
        "result": result,
        "error": error,
        "end": _plain(microscope.devices["stage"].position.get_value()),
    }


def _demo_cases(mount: str) -> Dict[str, Dict[str, Any]]:
    microscope, recorder = _demo(mount)
    stage = microscope.devices["stage"]
    cases: Dict[str, Dict[str, Any]] = {}

    def case(key: str, start: str, call: Callable[[], Any], setup=None) -> None:
        _place(microscope, start)
        if setup is not None:
            setup()
        cases[f"{mount} {key} @{start}"] = _run(microscope, recorder, call)

    for start in _orientations(microscope):
        for beam in BEAMS:
            for dx, dy in DISPLACEMENTS:
                for static_wd in (False, True):
                    case(
                        f"stable_move {beam.name} {dx} {dy} static_wd={static_wd}",
                        start,
                        lambda: microscope.stable_move(
                            dx=dx, dy=dy, beam_type=beam, static_wd=static_wd
                        ),
                    )
                case(
                    f"project_stable_move {beam.name} {dx} {dy}",
                    start,
                    lambda: microscope.project_stable_move(
                        dx=dx,
                        dy=dy,
                        beam_type=beam,
                        base_position=FibsemStagePosition(
                            x=1e-3, y=-2e-3, z=3e-3, r=0.5, t=0.1
                        ),
                    ),
                )

            # A scan rotation is undone before the projection.
            def rotated(beam=beam):
                microscope.set_scan_rotation(np.pi, beam)

            case(
                f"stable_move {beam.name} scan_rotation=180",
                start,
                lambda: microscope.stable_move(dx=10e-6, dy=20e-6, beam_type=beam),
                setup=rotated,
            )
            microscope.set_scan_rotation(0.0, beam)

            for dy, relaxation in ((5e-6, 1.0), (5e-6, 0.9), (250e-6, 1.0)):
                case(
                    f"vertical_move {beam.name} dy={dy} dx=2e-06 relaxation={relaxation}",
                    start,
                    lambda: microscope.vertical_move(
                        dy=dy, dx=2e-6, beam_type=beam, relaxation=relaxation
                    ),
                )

        # An unlinked stage leaves the working distance where it was.
        if "linked" in stage.parameters:

            def unlink():
                stage.sim_linked = False

            case(
                "stable_move ELECTRON unlinked",
                start,
                lambda: microscope.stable_move(
                    dx=10e-6, dy=20e-6, beam_type=BeamType.ELECTRON
                ),
                setup=unlink,
            )
            stage.sim_linked = True

        for target in _orientations(microscope):
            case(
                f"move_to_orientation {target}",
                start,
                lambda: microscope.move_to_orientation(target),
            )

        for angle in (10.0, 25.0):
            for rotation in (None, 0.0):
                case(
                    f"move_to_milling_angle {angle} rotation={rotation}",
                    start,
                    lambda: microscope.move_to_milling_angle(
                        np.radians(angle), rotation=rotation
                    ),
                )

        for name, target in (
            ("full", dict(x=1e-3, y=2e-3, z=3e-3, r=np.pi, t=np.radians(20))),
            ("small turn", dict(x=1e-3, y=2e-3, z=3e-3, r=np.radians(10), t=0.2)),
            ("xy only", dict(x=-1e-3, y=5e-4)),
        ):
            case(
                f"safe_absolute_stage_movement {name}",
                start,
                lambda: microscope.safe_absolute_stage_movement(
                    FibsemStagePosition(**target)
                ),
            )

    if microscope.fm is not None:
        fm = microscope.fm
        for start in _orientations(microscope):
            for transform in CameraImageTransform:
                for dx, dy in DISPLACEMENTS:
                    fm.set_image_transform(transform)
                    case(
                        f"fm_stable_move {transform.name} {dx} {dy}",
                        start,
                        lambda: microscope.fm_stable_move(dx=dx, dy=dy),
                    )
                    case(
                        f"project_fm_stable_move {transform.name} {dx} {dy}",
                        start,
                        lambda: microscope.project_fm_stable_move(
                            dx=dx,
                            dy=dy,
                            base_position=FibsemStagePosition(
                                x=1e-3, y=-2e-3, z=3e-3, r=0.5, t=0.1
                            ),
                        ),
                    )
            fm.set_image_transform(CameraImageTransform.NONE)
            for state in ("insert", "retract"):
                case(
                    f"fm_stable_move objective {state}",
                    start,
                    lambda: microscope.fm_stable_move(dx=10e-6, dy=20e-6),
                    setup=getattr(fm.objective, state),
                )
    else:
        case(
            "fm_stable_move without an FM",
            "SEM",
            lambda: microscope.fm_stable_move(dx=10e-6, dy=20e-6),
        )
    return cases


# -- TESCAN -----------------------------------------------------------------------------

TESCAN_START = [0.1, 0.05, 1.0]  # mm


def _tescan_cases() -> Dict[str, Dict[str, Any]]:
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(TESCAN_CONFIG).system
    system.stage.shuttle_pre_tilt = 35.0
    cases: Dict[str, Dict[str, Any]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        microscope, fake = connect(monkeypatch, system)
        events: List[str] = []
        microscope.record_signal.connect(
            lambda kind, payload: (
                events.append(payload["move"]) if kind == "stage_moved" else None
            )
        )

        def case(key: str, start: str, call: Callable[[], Any]) -> None:
            pose = microscope.get_orientation(start)
            microscope.devices["stage"]._move_absolute(
                FibsemStagePosition(
                    x=TESCAN_START[0] * 1e-3,
                    y=TESCAN_START[1] * 1e-3,
                    z=TESCAN_START[2] * 1e-3,
                    r=pose.r,
                    t=pose.t,
                )
            )
            microscope.get_stage_position()
            fake.log.clear()
            events.clear()
            try:
                result = _plain(call())
                error = None
            except Exception as e:  # noqa: BLE001 - the refusal is part of the pin
                result = None
                error = f"{type(e).__name__}: {e}"
            writes = [
                [path, args, kwargs]
                for path, args, kwargs in fake.log
                if path.split(".")[-1].startswith(("MoveTo", "Set"))
            ]
            cases[f"tescan {key} @{start}"] = {
                "commands": _plain(writes),
                "events": list(events),
                "result": result,
                "error": error,
            }

        for start in ("SEM", "FIB"):
            for beam in BEAMS:
                for dx, dy in DISPLACEMENTS:
                    case(
                        f"stable_move {beam.name} {dx} {dy}",
                        start,
                        lambda: microscope.stable_move(dx=dx, dy=dy, beam_type=beam),
                    )
                    case(
                        f"project_stable_move {beam.name} {dx} {dy}",
                        start,
                        lambda: microscope.project_stable_move(
                            dx=dx,
                            dy=dy,
                            beam_type=beam,
                            base_position=FibsemStagePosition(
                                x=1e-3, y=-2e-3, z=3e-3, r=0.5, t=0.1
                            ),
                        ),
                    )
                case(
                    f"vertical_move {beam.name}",
                    start,
                    lambda: microscope.vertical_move(dy=5e-6, dx=2e-6, beam_type=beam),
                )
            for target in ("SEM", "FIB"):
                case(
                    f"move_to_orientation {target}",
                    start,
                    lambda: microscope.move_to_orientation(target),
                )
            case(
                "move_to_milling_angle 15",
                start,
                lambda: microscope.move_to_milling_angle(np.radians(15.0)),
            )
            case(
                "safe_absolute_stage_movement full",
                start,
                lambda: microscope.safe_absolute_stage_movement(
                    FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=np.pi, t=0.3)
                ),
            )
    return cases


# -- the pins ---------------------------------------------------------------------------


def _record() -> Dict[str, Dict[str, Any]]:
    logging.disable(logging.CRITICAL)
    try:
        cases: Dict[str, Dict[str, Any]] = {}
        for mount in MOUNTS:
            cases.update(_demo_cases(mount))
        cases.update(_tescan_cases())
        return cases
    finally:
        logging.disable(logging.NOTSET)


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, float):
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-15))
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _close(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _close(actual[k], expected[k]) for k in expected
        )
    return actual == expected


@pytest.fixture(scope="module")
def recorded():
    return _record()


@pytest.fixture(scope="module")
def pinned():
    with open(PINS_PATH) as f:
        return json.load(f)


def test_every_pinned_case_is_recorded(recorded, pinned):
    assert sorted(recorded) == sorted(pinned)


def test_the_pins_command_moves(pinned):
    # The pins would hold trivially if nothing moved.
    moving = [c for c in pinned.values() if c["commands"]]
    assert len(moving) > len(pinned) // 2
    assert any(c["error"] for c in pinned.values())


def _pinned_keys() -> List[str]:
    try:
        with open(PINS_PATH) as f:
            return sorted(json.load(f))
    except (OSError, ValueError):
        return []


@pytest.mark.parametrize("key", _pinned_keys())
def test_a_move_commands_what_it_did(recorded, pinned, key):
    actual, expected = recorded[key], pinned[key]
    assert _close(actual, expected), (
        f"{key}\nexpected: {json.dumps(expected, indent=1)}\n"
        f"actual: {json.dumps(actual, indent=1)}"
    )


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("Run under pytest, or with --write to regenerate the pins.")
    with open(PINS_PATH, "w") as f:
        json.dump(_record(), f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {PINS_PATH}")
