"""The Tescan stage driver moves the stage as Tescan's stage code does today.

``TescanStage`` is ``TescanMicroscope``'s stage branches moved onto the ``Stage``
device, in Tescan's own frame. Each case runs an old call on a microscope connected
without the stage device, and the same call on one connected as the app connects it,
each over its own recording fake of the SDK (``tests/fixtures/tescan_sdk.py``). Each
case requires the same result, the same messages logged at info and above, and the
same SDK calls apart from position reads: the routed move reads the position back
once more, as the Thermo stage device does.

The frame pins (``tests/test_tescan_frame_pins.py``) run through the device too, so
every view-corrected move and projection there is a further check. Nothing here has
run on an instrument.
"""

import logging
import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.drivers.tescan import TescanStage, bind_tescan_stage
from fibsem.devices.stage import UNLIMITED
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemStagePosition
from tests.fixtures.tescan_sdk import connect

START = [1.2, -0.8, 5.0, 180.0, 30.0]  # mm and degrees


def _system(stage=True):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.stage.enabled = stage
    system.stage.shuttle_pre_tilt = 35.0
    return system


def _pair(monkeypatch):
    monkeypatch.setattr(TescanMicroscope, "_build_stage", lambda self: None)
    old, old_fake = connect(monkeypatch, _system())
    monkeypatch.undo()
    new, new_fake = connect(monkeypatch, _system())
    assert old.stage is None and isinstance(new.stage, TescanStage)
    for fake in (old_fake, new_fake):
        fake.Stage.position = list(START)
    return (old, old_fake), (new, new_fake)


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append([record.levelname, record.getMessage()])


def _run(microscope, fake, call):
    handler = _Messages()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    fake.log.clear()
    try:
        try:
            result = repr(call(microscope))
        except Exception as e:  # a refusal is behaviour too
            result = f"raises {type(e).__name__}: {e}"
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    sdk = [c for c in fake.log if c[0] != "Stage.GetPosition"]
    return {
        "result": result,
        "sdk": sdk,
        "log": handler.messages,
        "position": list(fake.Stage.position),
    }


P = FibsemStagePosition
CASES = {
    "get stage_position": lambda m: m.get("stage_position"),
    "get_stage_position": lambda m: m.get_stage_position(),
    "get stage_calibrated": lambda m: m.get("stage_calibrated"),
    "get stage_homed": lambda m: m.get("stage_homed"),
    "get stage_linked": lambda m: m.get("stage_linked"),
    "move absolute": lambda m: m.move_stage_absolute(
        P(x=1e-3, y=2e-3, z=4e-3, r=0.0, t=np.radians(20))
    ),
    "move absolute, some axes": lambda m: m.move_stage_absolute(
        P(r=np.radians(180), t=np.radians(-20))
    ),
    "move relative": lambda m: m.move_stage_relative(
        P(x=10e-6, y=-20e-6, z=5e-6, r=0.0, t=0.0)
    ),
    "stable_move SEM": lambda m: m.stable_move(20e-6, -15e-6, BeamType.ELECTRON),
    "stable_move FIB": lambda m: m.stable_move(20e-6, -15e-6, BeamType.ION),
    "vertical_move FIB": lambda m: m.vertical_move(10e-6, 0.0, BeamType.ION),
    "vertical_move SEM": lambda m: m.vertical_move(10e-6, 5e-6, BeamType.ELECTRON),
    "home": lambda m: m.home(),
    "set stage_home": lambda m: m.set("stage_home", True),
    "link_stage": lambda m: m.link_stage(),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_routed_stage_call_moves_logs_and_answers_the_same(monkeypatch, case):
    (old, old_fake), (new, new_fake) = _pair(monkeypatch)
    expected = _run(old, old_fake, CASES[case])
    actual = _run(new, new_fake, CASES[case])
    assert actual == expected
    assert new_fake.unlocked == []


def test_the_moves_go_through_the_device(monkeypatch):
    _, (new, new_fake) = _pair(monkeypatch)
    seen = []
    for name in ("_move_absolute", "_move_relative"):
        hook = getattr(TescanStage, name)
        monkeypatch.setattr(
            TescanStage,
            name,
            lambda self, p, n=name, h=hook: (seen.append(n), h(self, p))[1],
        )
    new.move_stage_absolute(P(x=0.0))
    new.move_stage_relative(P(x=1e-6))
    new.stable_move(1e-6, 1e-6, BeamType.ION)
    # a relative move is an absolute one to where the stage is plus the offset
    relative = ["_move_relative", "_move_absolute"]
    assert seen == ["_move_absolute"] + relative + relative


def test_the_stage_has_position_only_and_every_axis_unlimited(monkeypatch):
    _, (new, _) = _pair(monkeypatch)
    assert sorted(new.stage.parameters) == ["position"]
    assert list(new.stage.axes) == ["x", "y", "z", "r", "t"]
    assert all(new.stage.axes[a].limits == UNLIMITED for a in new.stage.axes)
    available = sorted(n for n, c in new.stage.commands.items() if c.available)
    assert available == ["move_absolute", "move_relative"]


def test_the_device_reads_tescan_units_as_metres_and_radians(monkeypatch):
    _, (new, new_fake) = _pair(monkeypatch)
    position = new.stage.position.get_value()
    assert (position.x, position.y, position.z) == pytest.approx(
        (1.2e-3, -0.8e-3, 5e-3)
    )
    assert (position.r, position.t) == pytest.approx(np.radians([180.0, 30.0]))


def test_a_disabled_stage_gets_no_device(monkeypatch):
    microscope, fake = connect(monkeypatch, _system(stage=False))
    assert microscope.stage is None and bind_tescan_stage(microscope) is None
    with pytest.raises(ValueError, match="Stage is not enabled"):
        microscope.get("stage_position")
