"""Stage moves and beam shifts on the experiment's record (FIB-1042).

Each test drives the real backend methods on Demo, with a real ``EventRecorder``
writing a real ``events.jsonl``, and reads back what it wrote -- except the TESCAN
one, which has no simulator and runs its real move methods on a stubbed connection,
and the last, which checks that no backend steps around the devices that record.

The devices record, through the session the microscope gives them
(`fibsem.devices.core.Session`): the stage's moves, the stage movement service's moves
(one record each, the outermost only) and the beam's ``shift_by``.
"""

import importlib
import inspect
import os
import pkgutil
import threading

import numpy as np
import pytest

import fibsem.config as cfg
import fibsem.drivers
from fibsem import utils
from fibsem.devices import stage as stage_module
from fibsem.devices.core import Session
from fibsem.devices.stage import recording_move
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import BeamType, FibsemStagePosition, Point
from fibsem.workflows.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
)
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope(scope="function")


@pytest.fixture
def recorded(microscope, tmp_path):
    """The events recorded while the test body runs, read back from disk."""
    microscope.get_stage_position()  # a position read, as the app has on connect
    recorder = EventRecorder(microscope, experiment_path=tmp_path)

    def read(kind):
        recorder.close()
        path = tmp_path / EVENTS_FILENAME
        if not path.exists():  # nothing recorded at all
            return []
        return [r["payload"] for r in read_events(path) if r["kind"] == kind]

    yield read
    recorder.close()


def test_a_stable_move_is_one_event_not_two(microscope, recorded):
    start = microscope.get_stage_position()
    end = microscope.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ELECTRON)

    (move,) = recorded("stage_moved")  # not also its relative move
    assert move["move"] == "stable_move"
    assert move["request"] == {
        "dx": 10e-6,
        "dy": 5e-6,
        "beam_type": "ELECTRON",
        "static_wd": False,
    }
    assert move["start"] == start.to_dict()
    assert move["end"] == end.to_dict()
    assert move["error"] is None
    assert move["duration"] >= 0


def test_a_vertical_move_from_the_sem_is_one_event_not_three(microscope, recorded):
    # a stable move, then a vertical move from the FIB, then its relative move
    microscope.vertical_move(dy=2e-6, beam_type=BeamType.ELECTRON)

    (move,) = recorded("stage_moved")
    assert move["move"] == "vertical_move"
    assert move["request"] == {
        "dy": 2e-6,
        "dx": 0.0,
        "beam_type": "ELECTRON",
        "relaxation": 1.0,
    }


def test_a_move_to_an_orientation_is_one_event_named_for_the_orientation(
    microscope, recorded
):
    start = microscope.get_stage_position()
    end = microscope.move_to_orientation("FIB")  # a safe move: rotate, then move

    (move,) = recorded("stage_moved")
    assert move["move"] == "move_to_orientation"
    assert move["request"] == {"orientation": "FIB"}
    assert move["start"] == start.to_dict()
    assert move["end"] == end.to_dict()
    assert move["end"]["r"] != move["start"]["r"]


def test_the_safe_move_ends_where_it_was_last_read(microscope, recorded):
    target = FibsemStagePosition(x=20e-6, y=0, z=0, r=0, t=np.radians(10))
    # it answers where it is, as every move does (FIB-1430)
    returned = microscope.safe_absolute_stage_movement(target)

    (move,) = recorded("stage_moved")
    assert move["move"] == "safe_absolute_stage_movement"
    assert move["request"]["stage_position"]["x"] == pytest.approx(20e-6)
    assert move["end"] == microscope.get_stage_position().to_dict()
    assert move["end"] == returned.to_dict()


def test_a_move_to_where_the_stage_already_is_ends_there(microscope, recorded):
    # The position it reads is unchanged: the end is the start, not unknown.
    here = microscope.get_stage_position()
    assert microscope.safe_absolute_stage_movement(here).to_dict() == here.to_dict()

    (move,) = recorded("stage_moved")
    assert move["end"] == move["start"] == here.to_dict()


def test_a_move_to_a_device_is_one_event_not_three():
    # From the SEM pose to an offset FM: to the beams, re-pose there, travel out.
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    microscope.move_to_orientation("SEM")
    events = []
    microscope.session.record_signal.connect(
        lambda kind, payload: events.append((kind, payload))
    )

    microscope.move_to_device("FM")  # the old name, for move_to_device

    moves = [payload for kind, payload in events if kind == "stage_moved"]
    assert [m["move"] for m in moves] == ["move_to_device"]
    assert moves[0]["request"] == {"device": "FM", "orientation": None}
    assert microscope.stage_movement.current_device() == "FM"


def test_moves_made_one_after_another_are_each_recorded(microscope, recorded):
    microscope.move_stage_relative(FibsemStagePosition(x=1e-6))
    microscope.move_stage_absolute(FibsemStagePosition(x=0, y=0))

    moves = recorded("stage_moved")
    assert [m["move"] for m in moves] == ["move_relative", "move_absolute"]
    assert moves[0]["request"] == {"position": FibsemStagePosition(x=1e-6).to_dict()}
    assert moves[1]["start"] == moves[0]["end"]


def test_a_move_made_on_the_stage_device_is_recorded(microscope, recorded):
    start = microscope.get_stage_position()
    end = microscope.stage.move_relative(FibsemStagePosition(x=1e-6))

    (move,) = recorded("stage_moved")
    assert move["move"] == "move_relative"
    assert move["start"] == start.to_dict()
    assert move["end"] == end.to_dict()


def test_a_move_on_the_movement_service_is_one_event(microscope, recorded):
    microscope.stage_movement.stable_move(dx=1e-6, dy=0, beam_type=BeamType.ION)

    assert [m["move"] for m in recorded("stage_moved")] == ["stable_move"]


def test_a_failed_move_is_recorded_with_why_and_the_caller_still_hears(
    microscope, recorded
):
    with pytest.raises(ValueError, match="SIDEWAYS"):
        microscope.move_to_orientation("SIDEWAYS")
    microscope.move_stage_relative(FibsemStagePosition(x=1e-6))  # still recorded after

    failed, after = recorded("stage_moved")
    assert failed["move"] == "move_to_orientation"
    assert failed["error"] == "ValueError: Orientation SIDEWAYS not supported."
    assert failed["end"] == failed["start"]  # it failed before moving
    assert after["move"] == "move_relative"


def test_a_move_on_another_thread_is_its_own_event(microscope, recorded):
    # One thread's move in progress must not make another thread's move look like
    # a step of it.
    inside, release = threading.Event(), threading.Event()

    def slow_move():
        with recording_move(microscope.session, microscope.stage, "slow_move"):
            inside.set()
            release.wait(5)
            microscope.stable_move(dx=1e-6, dy=0, beam_type=BeamType.ION)

    worker = threading.Thread(target=slow_move)
    worker.start()
    assert inside.wait(5)
    try:
        microscope.move_stage_relative(FibsemStagePosition(x=1e-6))
    finally:
        release.set()
        worker.join(5)

    moves = recorded("stage_moved")
    assert sorted(m["move"] for m in moves) == ["move_relative", "slow_move"]


def test_a_move_that_cannot_be_described_still_moves(microscope, recorded, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("cannot describe it")

    monkeypatch.setattr(stage_module, "plain", broken)
    end = microscope.move_stage_relative(FibsemStagePosition(x=5e-6))

    assert end.x == pytest.approx(microscope.get_stage_position().x)
    (move,) = recorded("stage_moved")  # recorded, without what it can't describe
    assert move["move"] == "move_relative"
    assert move["request"] is None and move["end"] is None
    assert move["error"] is None


def test_a_beam_shift_is_recorded_as_asked(microscope, recorded):
    before = microscope.get_beam_shift(BeamType.ION)
    after = microscope.beam_shift(1e-6, -2e-6, BeamType.ION)

    assert after == before + Point(1e-6, -2e-6)
    (shift,) = recorded("beam_shifted")
    assert shift == {
        "dx": 1e-6,
        "dy": -2e-6,
        "beam_type": "ION",
        "shift": after.to_dict(),
        "duration": shift["duration"],
        "error": None,
    }


def test_a_shift_made_on_the_beam_device_is_recorded(microscope, recorded):
    microscope.beams[BeamType.ELECTRON].shift_by(1e-6, 0)

    (shift,) = recorded("beam_shifted")
    assert shift["beam_type"] == "ELECTRON"
    assert shift["dx"] == 1e-6


def test_an_image_from_the_beam_device_is_stamped(microscope):
    microscope.user.name = "someone"
    image = microscope.beams[BeamType.ELECTRON].acquire()

    assert image.metadata.user.name == "someone"
    assert image.metadata.system_info is not None


def test_a_device_with_no_microscope_records_nothing():
    session = Session()
    assert session.record("stage_moved", {}) is None
    with session.recording("stage_moved", move="m") as entry:
        pass
    assert entry["error"] is None and entry["duration"] >= 0


def test_a_tescan_stable_move_is_one_event_not_three(monkeypatch):
    # On TESCAN a stable move is a relative move, and a relative move an absolute one.
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(
        os.path.join(cfg.SHIPPED_CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    microscope, fake = connect(monkeypatch, system)
    events = []
    microscope.session.record_signal.connect(
        lambda kind, payload: events.append((kind, payload))
    )

    microscope.stable_move(dx=1e-6, dy=1e-6, beam_type=BeamType.ELECTRON)

    assert (
        len(fake.calls("Stage.MoveTo")) == 1
    )  # it did move, through the absolute move
    assert [(kind, payload["move"]) for kind, payload in events] == [
        ("stage_moved", "stable_move")
    ]


# ── no backend steps around the devices ───────────────────────────────────────
#
# Nothing fails when a backend makes its own move or beam shift: its moves are just
# missing from events.jsonl. So a new backend is found here by itself, not by a list
# someone has to remember to extend.


def _backends():
    """Every concrete microscope class in a driver's ``microscope.py``."""
    for driver in pkgutil.iter_modules(fibsem.drivers.__path__):
        if not driver.ispkg:
            continue
        try:
            importlib.import_module(f"fibsem.drivers.{driver.name}.microscope")
        except ImportError:  # remote has none, or a vendor SDK this machine lacks
            continue
    found, pending = set(), [FibsemMicroscope]
    while pending:
        for cls in pending.pop().__subclasses__():
            pending.append(cls)
            if cls.__module__.startswith("fibsem.drivers."):
                found.add(cls)
    return sorted(
        (cls for cls in found if not inspect.isabstract(cls)),
        key=lambda cls: cls.__name__,
    )


BACKENDS = _backends()

# The base class's, which go through the stage device, the stage movement service
# and the beam device, which record.
_RECORDED = (
    "move_stage_absolute",
    "move_stage_relative",
    "stable_move",
    "vertical_move",
    "safe_absolute_stage_movement",
    "move_to_orientation",
    "move_to_milling_angle",
    "move_to_device",
    "beam_shift",
)


def test_every_backend_is_found():
    names = {cls.__name__ for cls in BACKENDS}
    assert {"ThermoMicroscope", "TescanMicroscope", "DemoMicroscope"} <= names


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_no_backend_makes_its_own_moves_or_beam_shifts(backend):
    for name in _RECORDED:
        assert getattr(backend, name) is getattr(FibsemMicroscope, name), name


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_every_backend_declares_its_moves_return_the_position(backend):
    # The contract `stage_moved` relies on for where a move ended, and the HTTP
    # server for its answer. Declared is not returned -- the drivers without a
    # simulator are held to it in test_stage_moves_return_position.py -- but a new
    # driver that declares otherwise, or nothing, fails here.
    for name in ("move_stage_absolute", "move_stage_relative"):
        returns = inspect.signature(getattr(backend, name)).return_annotation
        assert returns in (FibsemStagePosition, "FibsemStagePosition"), name
