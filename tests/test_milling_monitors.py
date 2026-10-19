"""Milling monitors on a mill step (FIB-1481): fake monitors on the scripted
service from test_milling_run, whose clock moves on at each wait."""

import logging
import time
from dataclasses import dataclass

import pytest

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.monitors import (
    MONITOR_PLUGINS,
    MillingMonitor,
    MillingMonitorConfig,
    MillingMonitorResult,
    MillingPeriodicMonitor,
    MillingPeriodicMonitorConfig,
    MillingStreamMonitor,
)
from fibsem.services.milling import MillingStepFailed, MixedMonitorViews
from tests.test_milling_run import (  # noqa: F401
    IDLE,
    RUNNING,
    Scripted,
    microscope,
    scripted,
)


class Pausing(Scripted):
    """The scripted service, with a pause and a resume it records, and waits
    long enough in real time for a monitor's thread to act."""

    def _pause(self):
        self.steps.append("pause")

    def _resume(self):
        self.steps.append("resume")

    def _wait(self, seconds):
        time.sleep(0.02)
        super()._wait(seconds)


@pytest.fixture
def pausing(scripted, monkeypatch):
    """The scripted fixture, building a `Pausing` service."""
    import tests.test_milling_run as run_tests

    monkeypatch.setattr(run_tests, "Scripted", Pausing)
    return scripted


@pytest.fixture
def registered(monkeypatch):
    """Register monitors for one test."""

    def register(*monitors):
        for monitor in monitors:
            monkeypatch.setitem(MONITOR_PLUGINS.registered, monitor.name, monitor)

    return register


def _stage(*configs):
    stage = FibsemMillingStage(name="s")
    stage.monitors = list(configs)
    return stage


def _mill(service, stage):
    with service.stage_run(stage) as stage_run:
        status = service.run()
    (step,) = stage_run.steps
    return status, step


class Watch(MillingMonitor[MillingMonitorConfig]):
    name = "Watch"

    def on_start(self):
        self.seen = []
        self.step.readings.connect(self.seen.append)
        self.report({"mean": 1.0})

    def result(self):
        return MillingMonitorResult(facts={"Readings": len(self.seen)})


def test_a_monitor_watches_the_step_and_its_result_is_recorded(scripted, registered):
    registered(Watch)
    service = scripted([RUNNING, RUNNING])
    status, step = _mill(service, _stage(Watch.config_class().made_for(Watch)))

    assert status == "completed"
    assert step.end.kind == "finished"
    assert step.to_dict()["monitors"] == [{"name": "Watch", "facts": {"Readings": 1}}]


@dataclass
class EndsConfig(MillingPeriodicMonitorConfig):
    on_trigger: str = "end"


class Ends(MillingPeriodicMonitor[EndsConfig]):
    """Triggers at its first tick: ends the step, or suggests it."""

    name = "Ends"
    config_class = EndsConfig

    def on_tick(self):
        self.trigger("drop seen", {"mean": 0.2})


def test_a_monitor_ends_the_step_completed(pausing, registered):
    registered(Ends)
    service = pausing([RUNNING] * 20)
    status, step = _mill(service, _stage(EndsConfig(interval=2)))

    assert status == "completed"
    assert step.end.to_dict() == {
        "kind": "ended",
        "by": "monitor:Ends",
        "reason": "drop seen",
    }
    (notable,) = step.notable
    assert notable["reading"]["trigger"] == "drop seen"
    assert notable["monitor"] == "Ends"


def test_a_suggesting_monitor_reports_and_milling_goes_on(pausing, registered):
    registered(Ends)
    service = pausing([RUNNING] * 6)
    status, step = _mill(service, _stage(EndsConfig(interval=2, on_trigger="suggest")))

    assert step.end.kind == "finished", "the instrument finished it"
    assert [n["reading"]["on_trigger"] for n in step.notable] == ["suggest"] * len(
        step.notable
    )
    assert step.notable, "the drop was reported"


@dataclass
class LookConfig(MillingPeriodicMonitorConfig):
    hold: float = 0.0


class Look(MillingPeriodicMonitor[LookConfig]):
    """Pauses the mill at each tick, for ``hold`` seconds."""

    name = "Look"
    config_class = LookConfig
    polls_while_paused = []

    def on_tick(self):
        with self.paused("look"):
            polls = self.step.milling.steps.count("poll")
            time.sleep(self.config.hold)
            Look.polls_while_paused.append(
                self.step.milling.steps.count("poll") - polls
            )


def test_a_monitor_pauses_and_resumes_the_mill(pausing, registered, monkeypatch):
    registered(Look)
    monkeypatch.setattr(Look, "polls_while_paused", [])
    service = pausing([RUNNING] * 8)
    monitor_config = LookConfig(interval=2, hold=0.1)
    status, step = _mill(service, _stage(monitor_config))

    assert status == "completed"
    assert service.steps.count("pause") == service.steps.count("resume") >= 1
    assert step.pauses[0]["reason"] == "look" and step.pauses[0]["by"] == "monitor:Look"
    assert set(Look.polls_while_paused) == {0}, "no look at the instrument while paused"


def test_a_pause_longer_than_max_pause_fails_the_step(pausing, registered):
    registered(Look)
    service = pausing([RUNNING] * 50)
    with pytest.raises(MillingStepFailed, match="paused for longer than 0.05 s"):
        _mill(service, _stage(LookConfig(interval=1, hold=0.5, max_pause=0.05)))

    assert "resume" not in service.steps, "never resumed on a timer"
    assert "stop" in service.steps


class Broken(MillingMonitor[MillingMonitorConfig]):
    name = "Broken"

    def on_start(self):
        raise RuntimeError("no camera")


class BrokenRequired(Broken):
    name = "BrokenRequired"
    required = True


def test_a_failing_monitor_stops_and_the_step_goes_on(scripted, registered):
    registered(Broken)
    service = scripted([RUNNING, RUNNING])
    status, step = _mill(service, _stage(MillingMonitorConfig().made_for(Broken)))

    assert status == "completed"
    assert step.to_dict()["monitors"] == [
        {"name": "Broken", "facts": {}, "error": "RuntimeError: no camera"}
    ]
    assert step.notable[0]["reading"] == {"error": "RuntimeError: no camera"}


def test_a_failing_required_monitor_fails_the_step_before_it_mills(
    scripted, registered
):
    registered(BrokenRequired)
    service = scripted([RUNNING, RUNNING])
    stage = _stage(MillingMonitorConfig().made_for(BrokenRequired))
    with pytest.raises(MillingStepFailed, match="no camera"):
        _mill(service, stage)
    assert "start" not in service.steps


def test_a_stage_saves_its_monitors_and_drops_unknown_ones(registered, caplog):
    registered(Ends)
    stage = _stage(EndsConfig(interval=5, on_trigger="suggest"))
    saved = stage.to_dict()
    assert saved["monitors"] == [
        {
            "name": "Ends",
            "config": {"max_pause": 60.0, "interval": 5, "on_trigger": "suggest"},
        }
    ]
    assert FibsemMillingStage.from_dict(saved).monitors == stage.monitors

    saved["monitors"].append({"name": "NotInstalled", "config": {}})
    logging.getLogger().addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            loaded = FibsemMillingStage.from_dict(saved)
    finally:
        logging.getLogger().removeHandler(caplog.handler)
    assert [m.monitor_name for m in loaded.monitors] == ["Ends"]
    assert "NotInstalled" in caplog.text


def test_a_stage_without_monitors_saves_an_empty_list():
    saved = FibsemMillingStage(name="s").to_dict()
    assert saved["monitors"] == []
    assert FibsemMillingStage.from_dict(saved).monitors == []


# ── readings out (2b) ─────────────────────────────────────────────────────


def test_readings_are_kept_for_reading_and_following(scripted, registered):
    registered(Watch)
    service = scripted([RUNNING, RUNNING])
    notable = []
    service.notable_reading.connect(notable.append)
    _mill(service, _stage(MillingMonitorConfig().made_for(Watch)))

    snapshot = service.readings.get_value()
    assert snapshot["stage"] == "s" and snapshot["step"] == 0
    assert snapshot["monitors"]["Watch"]["latest"]["reading"] == {"mean": 1.0}

    followed = service.follow_readings(since=0)
    assert [r["monitor"] for r in followed["readings"]] == ["Watch", "step"]
    assert followed["readings"][-1]["reading"]["end"]["kind"] == "finished"
    assert followed["seq"] == followed["readings"][-1]["seq"]
    assert service.follow_readings(since=followed["seq"])["readings"] == []

    assert [n["monitor"] for n in notable] == ["step"], "only the end is notable"


def test_a_plain_stage_reports_no_readings(scripted):
    service = scripted([RUNNING, RUNNING])
    _mill(service, _stage())
    assert service.follow_readings(since=0)["readings"] == []


def test_a_follow_waits_for_the_next_reading():
    from fibsem.milling.readings import MillingReadings

    readings = MillingReadings()
    readings.start_step("s", 0)

    def later():
        time.sleep(0.05)
        readings.add(
            {
                "monitor": "m",
                "seq": readings.next_seq(),
                "milled_time": 0.0,
                "notable": False,
                "reading": {"x": 1},
            }
        )

    import threading

    threading.Thread(target=later).start()
    followed = readings.follow(since=0, timeout=5)
    assert [r["reading"] for r in followed["readings"]] == [{"x": 1}]


class Chatty(MillingMonitor[MillingMonitorConfig]):
    """Reports three readings at once, then a notable one."""

    name = "Chatty"

    def on_start(self):
        for n in (1, 2, 3):
            self.report({"n": n})
        self.report({"n": 4, "drop": True}, notable=True)


def test_readings_are_throttled_keeping_the_latest_and_notable_ones_go_at_once(
    scripted, registered
):
    registered(Chatty)
    service = scripted([RUNNING, RUNNING])
    _mill(service, _stage(MillingMonitorConfig().made_for(Chatty)))

    readings = service.follow_readings(since=0)["readings"]
    assert [r["reading"].get("n") for r in readings if r["monitor"] == "Chatty"] == [
        1,
        4,
        3,
    ], "the first, the notable one, then the latest held back"


# ── live monitors and one view per step (2b) ──────────────────────────────


class FmLive(MillingMonitor[MillingMonitorConfig]):
    name = "FmLive"
    view = "fm"


class SemLive(MillingMonitor[MillingMonitorConfig]):
    name = "SemLive"
    view = "sem"


def test_live_monitors_needing_different_views_are_refused_before_milling(
    scripted, registered
):
    registered(FmLive, SemLive)
    service = scripted([RUNNING, RUNNING])
    stage = _stage(
        MillingMonitorConfig().made_for(FmLive),
        MillingMonitorConfig().made_for(SemLive),
    )
    with pytest.raises(MixedMonitorViews, match="fm, sem"):
        _mill(service, stage)
    assert "start" not in service.steps


def test_a_live_monitor_s_step_does_not_look_at_the_instrument_until_milled(
    scripted, registered
):
    registered(FmLive)
    service = scripted([IDLE], estimate=3.0)
    status, step = _mill(service, _stage(MillingMonitorConfig().made_for(FmLive)))

    assert status == "completed"
    assert step.milled_time() == 3.0, "counted from the waits"
    assert service.steps.count("poll") == 1, "one look, once the time was milled"


@dataclass
class LiveLookConfig(MillingPeriodicMonitorConfig):
    pass


class LiveLook(MillingPeriodicMonitor[LiveLookConfig]):
    """A live FM monitor that also pauses once to look; counts the times it
    was asked to take its view back."""

    name = "LiveLook"
    config_class = LiveLookConfig
    view = "fm"
    restored = 0

    def on_tick(self):
        with self.paused("look"):
            pass

    def restore_view(self):
        LiveLook.restored += 1


def test_a_live_monitor_gets_its_view_back_after_a_pause_and_resume(
    pausing, registered, monkeypatch
):
    registered(LiveLook)
    monkeypatch.setattr(LiveLook, "restored", 0)
    service = pausing([IDLE], estimate=30.0)
    status, step = _mill(service, _stage(LiveLookConfig(interval=2)))

    assert status == "completed"
    pauses = service.steps.count("pause")
    assert pauses == service.steps.count("resume") >= 1
    assert LiveLook.restored == 1 + 2 * pauses, "after the start, each pause and resume"
    last_resume = len(service.steps) - 1 - service.steps[::-1].index("resume")
    assert (
        last_resume < service.steps.index("poll") and service.steps.count("poll") == 1
    )
    paused = [
        r["reading"]
        for r in service.follow_readings(since=0)["readings"]
        if "paused" in r["reading"]
    ]
    assert paused[0] == {"paused": "look", "by": "monitor:LiveLook"}
    assert step.notable == [], "the pauses are in the step's record already"


class Frames:
    from psygnal import Signal

    frame = Signal(object)


FRAMES = Frames()


class Bright(MillingStreamMonitor[MillingMonitorConfig]):
    """Reads each frame's brightness off a source, while the pattern mills."""

    name = "Bright"
    view = "fm"

    def source(self):
        return FRAMES.frame

    def on_start(self):
        super().on_start()
        self.seen = []
        FRAMES.frame.emit(0.5)

    def on_frame(self, frame):
        self.seen.append(frame)

    def result(self):
        return MillingMonitorResult(facts={"Frames": len(self.seen)})


def test_a_stream_monitor_reads_frames_until_the_step_ends(scripted, registered):
    registered(Bright)
    service = scripted([IDLE], estimate=2.0)
    _, step = _mill(service, _stage(MillingMonitorConfig().made_for(Bright)))

    assert step.to_dict()["monitors"] == [{"name": "Bright", "facts": {"Frames": 1}}]
    assert len(FRAMES.frame) == 0, "it let go of the source"
