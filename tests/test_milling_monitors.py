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
)
from fibsem.services.milling import MillingStepFailed
from tests.test_milling_run import RUNNING, Scripted, microscope, scripted  # noqa: F401


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
