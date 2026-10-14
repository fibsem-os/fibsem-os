"""The milling service's run loop (`Milling.run`): progress, pause, stop, failures.

The Demo runs it as it mills; a scripted service over the Demo's beams plays the
instrument where a case needs states the Demo doesn't produce, with a clock that the
loop's waits move on, so nothing sleeps.
"""

import threading
from typing import List

import pytest

from fibsem import utils
from fibsem.cancellation import OperationCancelledError
from fibsem.drivers.demo.services import DemoMilling
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.services import milling as milling_module
from fibsem.services.milling import Milling, progress_update
from fibsem.structures import (
    BeamType,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    MillingState,
)

RUNNING, PAUSED, IDLE = MillingState.RUNNING, MillingState.PAUSED, MillingState.IDLE


def _elapsed(progress: MillingProgress) -> float:
    return progress.estimated_time - progress.remaining_time


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    yield microscope
    microscope.disconnect()


def _draw(microscope, n=2):
    microscope.setup_milling(
        FibsemMillingSettings(milling_current=1e-9, milling_voltage=30e3)
    )
    microscope.draw_patterns(
        [
            FibsemRectangleSettings(
                width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
            )
        ]
        * n
    )


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Scripted(Milling):
    """Plays back one state per look, and records the steps a run takes."""

    def __init__(
        self, states, clock, estimate=10.0, polls=None, stop_timeout=30.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.stop_timeout = stop_timeout
        self.states = list(states)
        self.polls = polls
        self.clock = clock
        self.estimate_s = estimate
        self.steps: List[str] = []

    def read_state(self):
        return self.states[0] if self.states else IDLE

    def _poll(self):
        self.steps.append("poll")
        if self.polls is not None:
            return self.polls.pop(0)
        state = self.states.pop(0) if self.states else IDLE
        return progress_update(state=state)

    def _wait(self, seconds):
        self.clock.now += 1.0

    def _setup(self, settings, name):
        pass

    def _before_run(self):
        self.steps.append("before")

    def _estimate(self):
        return self.estimate_s

    def _start(self):
        self.steps.append("start")

    def _stop(self):
        self.steps.append("stop")
        self.states = [IDLE]

    def _after_run(self):
        self.steps.append("after")

    def _clear(self):
        self.steps.append("clear")


@pytest.fixture
def scripted(microscope, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(milling_module.time, "monotonic", clock)

    def make(states=(), **kwargs):
        service = Scripted(states, clock, parent=microscope, **kwargs)
        service.fill_roles(ion=microscope.beams[BeamType.ION])
        return service.connect()

    return make


def test_the_demo_mills_for_its_estimate_and_reports_as_it_goes(microscope):
    _draw(microscope, n=2)  # five simulated seconds a pattern
    reports = []
    microscope.milling.progress.changed.connect(reports.append)
    updates = []
    microscope.milling_progress_signal.connect(updates.append)

    microscope.run_milling()

    assert isinstance(microscope.milling, DemoMilling)
    assert microscope.get_milling_state() is IDLE
    assert microscope._connection.milling_system.patterns == []
    assert [_elapsed(r) for r in reports] == list(range(11))
    assert reports[-1] == MillingProgress(
        status=MillingProgressStatus.STAGE_UPDATE,
        milling_state=IDLE,
        start_time=reports[0].start_time,
        estimated_time=10,
        remaining_time=0,
        milling_run_id=reports[0].milling_run_id,  # a run outside a task's own
    )
    # the stage updates the milling widget and AutoLamella have always had
    assert updates == reports


def test_a_set_stop_event_stops_the_beam_and_cancels(microscope):
    _draw(microscope)
    stop = threading.Event()

    def stop_after_two(progress):
        if _elapsed(progress) >= 2:
            stop.set()

    microscope.milling.progress.changed.connect(stop_after_two)
    with pytest.raises(OperationCancelledError):
        microscope.run_milling(stop_event=stop)

    assert microscope.get_milling_state() is IDLE
    assert microscope._connection.milling_system.patterns == []
    assert _elapsed(microscope.milling.progress.cached) == 2


def test_a_stop_from_elsewhere_just_ends_the_run(scripted):
    service = scripted([RUNNING, RUNNING, IDLE])
    assert service.run() == "completed"  # no stop event: nothing to cancel
    assert service.steps == [
        "before",
        "start",
        "poll",
        "poll",
        "poll",
        "after",
        "clear",
    ]


def test_a_run_stopped_from_elsewhere_answers_cancelled(scripted):
    """A stopped mill must not answer like a finished one (FIB-1377)."""
    service = scripted([RUNNING, RUNNING, RUNNING, RUNNING])
    waits = service._wait

    def stop_on_the_first_wait(seconds):
        if "stop" not in service.steps:
            service.stop()  # as the operator's, or Stop agent's, from another thread
        waits(seconds)

    service._wait = stop_on_the_first_wait
    assert service.run() == "cancelled"
    assert service.steps[-2:] == ["after", "clear"]
    assert service.run() == "completed"  # the next run starts unstopped


def test_paused_time_does_not_count(scripted):
    service = scripted([RUNNING, PAUSED, PAUSED, RUNNING, RUNNING, IDLE])
    seen = []
    service.progress.changed.connect(seen.append)
    service.run()
    assert [(p.milling_state, _elapsed(p)) for p in seen] == [
        (RUNNING, 0),
        (PAUSED, 0),
        (RUNNING, 1),
        (RUNNING, 2),
        (IDLE, 2),
    ]
    assert seen[-1].remaining_time == 8


def test_a_mill_over_before_the_first_look_does_not_hang(scripted):
    """The old loop waited for the instrument to leave IDLE, with no end."""
    service = scripted([])
    service.run()
    assert service.steps.count("poll") == service.start_timeout + 1
    assert service.steps[-2:] == ["after", "clear"]


def test_idle_while_starting_is_not_the_end(scripted):
    service = scripted([IDLE, IDLE, RUNNING, IDLE])
    service.run()
    assert service.steps.count("poll") == 4


def test_a_failure_stops_the_beam_tidies_up_and_raises(scripted):
    service = scripted([RUNNING])

    def broken():
        service.steps.append("poll")
        raise RuntimeError("lost")

    service._poll = broken
    with pytest.raises(RuntimeError, match="lost"):
        service.run()
    assert service.steps == ["before", "start", "poll", "stop", "after", "clear"]


def test_the_instrument_s_times_win(scripted):
    # Tescan reports its total and (as elapsed) the time remaining; JEOL the time
    # remaining alone
    service = scripted(
        polls=[
            progress_update(RUNNING, total=12.0, remaining=9.0),
            progress_update(RUNNING, remaining=4.0),
            progress_update(IDLE),
        ]
    )
    seen = []
    service.progress.changed.connect(seen.append)
    service.run()
    assert [(p.milling_state, _elapsed(p), p.remaining_time) for p in seen] == [
        (RUNNING, 3.0, 9.0),
        (RUNNING, 8.0, 4.0),
        (IDLE, 8.0, 4.0),
    ]
    assert {p.estimated_time for p in seen} == {12.0}


def test_a_stop_event_the_beam_ignores_gives_up(scripted):
    service = scripted([RUNNING] * 100, stop_timeout=3)
    service._stop = lambda: service.steps.append("stop")  # never stops
    stop = threading.Event()
    stop.set()
    with pytest.raises(OperationCancelledError):
        service.run(stop_event=stop)
    assert service.steps.count("stop") == 1
    assert service.steps.count("poll") == 4


def _milling_events(microscope):
    events = []
    microscope.record_signal.connect(
        lambda kind, payload: events.append(payload) if kind == "milling" else None
    )
    return events


def test_a_mill_outside_a_task_is_on_the_record_with_its_name(microscope):
    """An agent's prepare and run left no milling event, and the name it was
    prepared with appeared nowhere (FIB-1377)."""
    events = _milling_events(microscope)
    updates = []
    microscope.milling.progress.changed.connect(updates.append)
    microscope.milling.prepare(
        FibsemMillingSettings(milling_current=1e-9, milling_voltage=30e3),
        [
            FibsemRectangleSettings(
                width=1e-6, height=1e-6, depth=1e-7, centre_x=0, centre_y=0
            )
        ],
        name="agent trench",
    )

    assert microscope.milling.run() == "completed"

    [event] = events
    assert event["milling_task_name"] == "agent trench"
    assert event["status"] == "completed"
    assert event["started_at"] and event["ended_at"]
    assert {u.milling_task_name for u in updates} == {"agent trench"}
    assert {u.milling_run_id for u in updates} == {event["milling_run_id"]}


def test_a_stopped_or_failed_mill_outside_a_task_records_how_it_ended(scripted):
    service = scripted([RUNNING, RUNNING, RUNNING])
    events = _milling_events(service.parent)
    service.setup(FibsemMillingSettings(), name="stopped")
    service._wait = lambda seconds: service.stop()
    assert service.run() == "cancelled"

    def broken():
        raise RuntimeError("lost")

    service._poll = broken
    with pytest.raises(RuntimeError):
        service.run()

    assert [(e["milling_task_name"], e["status"]) for e in events] == [
        ("stopped", "cancelled"),
        ("stopped", "failed"),
    ]
    assert events[1]["error"] == "RuntimeError: lost"


def test_a_mill_inside_a_task_or_a_spot_burn_is_not_recorded_twice(scripted):
    """The task, or the spot burn, records the whole."""
    service = scripted([])
    events = _milling_events(service.parent)
    service.report(
        MillingProgress(
            status=MillingProgressStatus.TASK_STARTED, milling_run_id="task-run"
        )
    )
    service.run()
    service.report(MillingProgress(status=MillingProgressStatus.TASK_FINISHED))
    with service.part_of():
        service.run()
    assert events == []
    service.run()
    assert len(events) == 1
