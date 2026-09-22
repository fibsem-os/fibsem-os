"""Stage moves and beam shifts on the experiment's record (FIB-1042).

Each test drives the real backend methods on Demo, with a real ``EventRecorder``
writing a real ``events.jsonl``, and reads back what it wrote -- except the TESCAN
one, which has no simulator and runs its real move methods on a stubbed connection.
"""

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import microscope as microscope_module
from fibsem import utils
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
)
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemStagePosition


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    microscope.get_stage_position()  # a position read, as the app has on connect
    return microscope


@pytest.fixture
def recorded(microscope, tmp_path):
    """The events recorded while the test body runs, read back from disk."""
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


def test_a_move_that_returns_nothing_ends_where_it_was_last_read(microscope, recorded):
    target = FibsemStagePosition(x=20e-6, y=0, z=0, r=0, t=np.radians(10))
    assert microscope.safe_absolute_stage_movement(target) is None

    (move,) = recorded("stage_moved")
    assert move["move"] == "safe_absolute_stage_movement"
    assert move["request"]["stage_position"]["x"] == pytest.approx(20e-6)
    assert move["end"] == microscope.get_stage_position().to_dict()


def test_a_move_to_a_device_is_one_event_not_three():
    # From the SEM pose to an offset FM: to the beams, re-pose there, travel out.
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    microscope.move_to_orientation("SEM")
    events = []
    microscope.record_signal.connect(
        lambda kind, payload: events.append((kind, payload))
    )

    microscope.move_to_microscope("FM")  # the old name, for move_to_device

    moves = [payload for kind, payload in events if kind == "stage_moved"]
    assert [m["move"] for m in moves] == ["move_to_device"]
    assert moves[0]["request"] == {"device": "FM", "orientation": None}
    assert microscope.get_current_device() == "FM"


def test_moves_made_one_after_another_are_each_recorded(microscope, recorded):
    microscope.move_stage_relative(FibsemStagePosition(x=1e-6))
    microscope.move_stage_absolute(FibsemStagePosition(x=0, y=0))

    moves = recorded("stage_moved")
    assert [m["move"] for m in moves] == ["move_stage_relative", "move_stage_absolute"]
    assert moves[1]["start"] == moves[0]["end"]


def test_a_failed_move_is_recorded_with_why_and_the_caller_still_hears(
    microscope, recorded
):
    with pytest.raises(ValueError, match="SIDEWAYS"):
        microscope.move_to_orientation("SIDEWAYS")
    microscope.move_stage_relative(FibsemStagePosition(x=1e-6))  # still recorded after

    failed, after = recorded("stage_moved")
    assert failed["move"] == "move_to_orientation"
    assert failed["error"] == "ValueError: Orientation SIDEWAYS not supported."
    assert failed["end"] is None
    assert after["move"] == "move_stage_relative"


def test_a_move_on_another_thread_is_its_own_event(microscope, recorded):
    # One thread's move in progress must not make another thread's move look like
    # a step of it.
    inside, release = threading.Event(), threading.Event()
    stable_move = type(microscope).stable_move

    def slow_move(self, *args, **kwargs):
        inside.set()
        release.wait(5)
        return stable_move(self, *args, **kwargs)

    worker = threading.Thread(
        target=lambda: microscope_module._records_stage_move(slow_move)(
            microscope, dx=1e-6, dy=0, beam_type=BeamType.ION
        )
    )
    worker.start()
    assert inside.wait(5)
    try:
        microscope.move_stage_relative(FibsemStagePosition(x=1e-6))
    finally:
        release.set()
        worker.join(5)

    moves = recorded("stage_moved")
    assert sorted(m["move"] for m in moves) == ["move_stage_relative", "slow_move"]


def test_a_move_that_cannot_be_recorded_still_moves(microscope, recorded, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("cannot describe it")

    monkeypatch.setattr(microscope_module, "_call_arguments", broken)
    end = microscope.move_stage_relative(FibsemStagePosition(x=5e-6))

    assert end.x == pytest.approx(microscope.get_stage_position().x)
    assert recorded("stage_moved") == []


def test_a_beam_shift_is_recorded_as_asked(microscope, recorded):
    microscope.beam_shift(1e-6, -2e-6, BeamType.ION)

    (shift,) = recorded("beam_shifted")
    assert shift == {
        "dx": 1e-6,
        "dy": -2e-6,
        "beam_type": "ION",
        "shift": None,  # Demo's beam_shift returns nothing
        "error": None,
    }


def test_a_tescan_stable_move_is_one_event_not_three():
    # On TESCAN a stable move is a relative move, and a relative move an absolute one.
    microscope = object.__new__(TescanMicroscope)  # skip __init__ (requires the SDK)
    microscope._connection_lock = threading.RLock()
    microscope.system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    microscope.stage_is_compustage = False
    at = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW")
    microscope.get_stage_position = lambda: at
    microscope.get_scan_rotation = lambda beam_type: 0.0
    sent = []
    microscope.connection = SimpleNamespace(
        Stage=SimpleNamespace(MoveTo=lambda **axes: sent.append(axes))
    )
    events = []
    microscope.record_signal.connect(
        lambda kind, payload: events.append((kind, payload))
    )

    microscope.stable_move(dx=1e-6, dy=1e-6, beam_type=BeamType.ELECTRON)

    assert len(sent) == 1  # it did move, through the absolute move
    assert [(kind, payload["move"]) for kind, payload in events] == [
        ("stage_moved", "stable_move")
    ]
