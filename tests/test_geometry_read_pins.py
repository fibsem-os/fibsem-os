"""Today's geometry reads, pinned before they move off the microscope class.

The reads (``get_stage_orientation``, the orientation table, ``get_current_milling_angle``,
``is_close_to_milling_angle``, ``get_current_device``, ``is_at_device`` and
``get_target_position``) answer questions about stage positions and move nothing. They
are moving into ``fibsem.geometry`` and the stage movement service; this file pins
their answers so that move keeps every one the same.

Each case records the answer, or the error raised. Backends: Demo on three mounts (the
default rotation stage, ``sim-iflm`` with an FM 48.8 mm along x, and ``sim-arctis``, a
compustage), and TESCAN over the recording fake connection
(``tests/fixtures/tescan_sdk.py``). ThermoFisher and Odemis run the same shared code.

The expected values are in ``tests/fixtures/geometry_read_pins.json``, recorded from
main at the commit that added this file. A change that is meant to move one regenerates
the file with

    PYTHONPATH=. python tests/test_geometry_read_pins.py --write

and says in its pull request which values moved and why.
"""

import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition
from tests.fixtures.pins import close_to_pin, plain

PINS_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "geometry_read_pins.json"
)

MOUNTS = {
    "default": None,
    "iflm": os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml"),
    "arctis": os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
}
TESCAN_CONFIG = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")

NAMES = ("SEM", "FIB", "MILLING", "FM", "NONE")
# Off centre, so a rotation shows in x and y.
START = dict(x=100e-6, y=50e-6, z=1e-3)
# Off nominal: a real stage never sits exactly at a pose.
SLOP = ((0.0, 0.0), (np.radians(3), 0.0), (0.0, np.radians(4)), (np.radians(-4), -0.05))
# A sweep for the classifier: every 45 degrees of rotation, every 5 of tilt.
ROTATIONS = [np.radians(r) for r in range(0, 360, 45)]
TILTS = [np.radians(t) for t in range(-180, 91, 5)]


def _answer(call: Callable[[], Any]) -> Dict[str, Any]:
    try:
        return {"result": plain(call()), "error": None}
    except Exception as e:  # noqa: BLE001 - the refusal is part of the pin
        return {"result": None, "error": f"{type(e).__name__}: {e}"}


def _orientations(microscope) -> List[str]:
    return [name for name in NAMES if name in microscope.orientations]


def _cases(prefix: str, microscope, start: Dict[str, float]) -> Dict[str, Any]:
    """Every read, from every orientation, a little off nominal."""
    microscope.get_orientation("SEM")  # builds the table on first use
    cases: Dict[str, Any] = {}

    def case(key: str, call: Callable[[], Any]) -> None:
        cases[f"{prefix} {key}"] = _answer(call)

    case("orientations", lambda: dict(microscope.orientations))
    for r in ROTATIONS:
        for t in TILTS:
            case(
                f"orientation r={np.degrees(r):.0f} t={np.degrees(t):.0f}",
                lambda: microscope.get_stage_orientation(
                    FibsemStagePosition(r=r, t=t, **start)
                ),
            )
            case(
                f"milling_angle r={np.degrees(r):.0f} t={np.degrees(t):.0f}",
                lambda: microscope.get_current_milling_angle(
                    FibsemStagePosition(r=r, t=t, **start)
                ),
            )

    devices = sorted(microscope.system.stage.devices)
    for name in _orientations(microscope):
        pose = microscope.orientations[name]
        for dr, dt in SLOP:
            at = f"@{name}{np.degrees(dr):+.0f}r{np.degrees(dt):+.1f}t"
            position = FibsemStagePosition(r=pose.r + dr, t=pose.t + dt, **start)
            case(
                f"current_device {at}", lambda: microscope.get_current_device(position)
            )
            for device in devices:
                case(
                    f"is_at_device {device} {at}",
                    lambda: microscope.is_at_device(device, position),
                )
            for target in _orientations(microscope) + [None]:
                for device in devices + [None]:
                    case(
                        f"target {target} device={device} {at}",
                        lambda: microscope.get_target_position(
                            position, target_orientation=target, target_device=device
                        ),
                    )

    # The refusals: no orientation to convert from, an FM that is not a pose, a
    # position without a pose, an orientation nobody declared.
    nowhere = FibsemStagePosition(r=np.radians(90), t=np.radians(-60), **start)
    case("target SEM from NONE", lambda: microscope.get_target_position(nowhere, "SEM"))
    sem = microscope.orientations["SEM"]
    at_sem = FibsemStagePosition(r=sem.r, t=sem.t, **start)
    case("target FM @SEM", lambda: microscope.get_target_position(at_sem, "FM"))
    case(
        "orientation without r",
        lambda: microscope.get_stage_orientation(FibsemStagePosition(t=0.0, **start)),
    )
    case("get_orientation BOGUS", lambda: microscope.get_orientation("BOGUS"))

    # The stored milling angle reshapes the MILLING orientation.
    for angle in (10.0, 25.0):

        def milling(angle=angle):
            microscope.set_milling_angle(angle)
            return microscope.orientations["MILLING"]

        case(f"set_milling_angle {angle}", milling)
    return cases


def _demo_cases(mount: str) -> Dict[str, Any]:
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    path = MOUNTS[mount]
    if path is None:
        microscope, _ = utils.setup_session(
            manufacturer="Demo", ip_address="localhost", setup_logging=False
        )
    else:
        microscope, _ = utils.setup_session(config_path=path, setup_logging=False)
    cases = _cases(mount, microscope, START)

    # The live reads, at the stage itself.
    stage = microscope.devices["stage"]
    for name in _orientations(microscope):
        pose = microscope.orientations[name]
        stage._move_absolute(FibsemStagePosition(r=pose.r, t=pose.t, **START))
        cases[f"{mount} live orientation @{name}"] = _answer(
            microscope.get_stage_orientation
        )
        cases[f"{mount} live milling_angle @{name}"] = _answer(
            microscope.get_current_milling_angle
        )
        cases[f"{mount} live is_close 15 @{name}"] = _answer(
            lambda: microscope.is_close_to_milling_angle(15.0)
        )
    return cases


def _tescan_cases() -> Dict[str, Any]:
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(TESCAN_CONFIG).system
    system.stage.shuttle_pre_tilt = 35.0
    with pytest.MonkeyPatch.context() as monkeypatch:
        microscope, _ = connect(monkeypatch, system)
        return _cases("tescan", microscope, START)


# -- the pins ---------------------------------------------------------------------------


def _record() -> Dict[str, Any]:
    logging.disable(logging.CRITICAL)
    try:
        cases: Dict[str, Any] = {}
        for mount in MOUNTS:
            cases.update(_demo_cases(mount))
        cases.update(_tescan_cases())
        return cases
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def recorded():
    return _record()


@pytest.fixture(scope="module")
def pinned():
    with open(PINS_PATH) as f:
        return json.load(f)


def test_every_pinned_case_is_recorded(recorded, pinned):
    assert sorted(recorded) == sorted(pinned)


def test_the_pins_cover_every_answer(pinned):
    # The pins would hold trivially if every read refused.
    answers = [c for c in pinned.values() if c["error"] is None]
    assert len(answers) > len(pinned) // 2
    assert any(c["error"] for c in pinned.values())
    names = {c["result"] for c in answers if isinstance(c["result"], str)}
    assert {"SEM", "FIB", "MILLING", "FM", "NONE"} <= names


def test_every_read_answers_what_it_did(recorded, pinned):
    # One test over every case: a few thousand parametrized ones cost a minute.
    moved = [
        key for key in sorted(pinned) if not close_to_pin(recorded[key], pinned[key])
    ]
    assert not moved, (
        "\n".join(
            f"{key}\nexpected: {json.dumps(pinned[key])}\n"
            f"actual: {json.dumps(recorded[key])}"
            for key in moved[:10]
        )
        + f"\n({len(moved)} cases moved)"
    )


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("Run under pytest, or with --write to regenerate the pins.")
    with open(PINS_PATH, "w") as f:
        json.dump(_record(), f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {PINS_PATH}")
