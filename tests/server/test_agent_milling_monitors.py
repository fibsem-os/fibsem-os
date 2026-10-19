"""The agent and milling monitors (FIB-1484): it lists them, follows their
readings, and ends a stage on a monitor's suggestion, keeping what it milled.

The Demo is an Arctis here, its FM a made-up stream that drops."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from psygnal import Signal  # noqa: E402

from fibsem.ai.agent.mcp import watch as watch_mod  # noqa: E402
from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.monitors import FmIntensityMonitor  # noqa: E402
from fibsem.milling.monitors.fm_intensity import FmIntensityConfig  # noqa: E402
from fibsem.milling.patterning.patterns import RectanglePattern  # noqa: E402
from fibsem.milling.tasks import (  # noqa: E402
    FibsemMillingTaskConfig,
    patch_milling_config,
)
from tests.server.test_agent_gap_tools import _ok, driving  # noqa: E402, F401
from tests.server.test_grid_tools import _call, arctis  # noqa: E402, F401

RUN = "/microscope/services/milling/run"


class Frame:
    def __init__(self, level):
        self.data = np.full((16, 16), float(level))

    def save(self, filename):
        pass


class DroppingFM:
    """A frame every 0.05 s: bright for the first 20, then a third as bright."""

    acquisition_signal = Signal(object)

    def __init__(self):
        self._stop = threading.Event()

    def start_acquisition(self, channel_settings=None):
        self._stop.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        frames = 0
        while not self._stop.wait(0.05):
            frames += 1
            self.acquisition_signal.emit(Frame(1000 if frames <= 20 else 300))

    def stop_acquisition(self):
        self._stop.set()

    def set_active_channel(self):
        pass


@pytest.fixture
def milling_for_a_while(arctis, monkeypatch):  # noqa: F811
    """The Demo mills for a minute, looked at every 0.1 s, under a dropping FM."""
    import fibsem.drivers.demo.services as demo_services

    monkeypatch.setattr(demo_services, "_PATTERN_TIME", 60)
    monkeypatch.setattr(demo_services, "sim_sleep", time.sleep)
    arctis.milling.poll_interval = 0.1
    monkeypatch.setattr(arctis, "fm", DroppingFM())
    return arctis


def _stage(on_trigger="suggest"):
    stage = FibsemMillingStage(name="Rough")
    stage.pattern = RectanglePattern(width=10e-6, height=5e-6, depth=1e-6)
    config = FmIntensityConfig(
        warmup_duration=0.2,
        rolling_window=1,
        consecutive_triggers=2,
        on_trigger=on_trigger,
        save_interval=0,
    )
    stage.monitors = [config.made_for(FmIntensityMonitor)]
    return stage


def test_the_agent_lists_the_monitors_with_their_config(driving):  # noqa: F811
    options = _ok(_call(driving, "milling_list_options"))["result"]
    fm = options["monitors"]["FM Intensity"]["properties"]
    assert {"drop_fraction", "on_trigger", "warmup_duration"} <= set(fm)
    assert "interval" in options["monitors"]["SEM Peek"]["properties"]


def test_a_protocol_stage_s_monitor_config_is_patched_as_it_is_saved():
    config = FibsemMillingTaskConfig(stages=[_stage()])
    assert set(config.stages[0].to_dict()["monitors"][0]) == {"name", "config"}

    edited = patch_milling_config(
        config, {"stages.Rough.monitors.0.config.drop_fraction": 0.6}
    )

    assert edited.stages[0].monitors[0].drop_fraction == 0.6
    assert config.stages[0].monitors[0].drop_fraction == 0.4


def test_a_monitor_s_config_names_only_keys_it_has(driving):  # noqa: F811
    stage = _stage().to_dict()
    stage["monitors"][0]["config"]["drop_fractoin"] = 0.5

    refused = _call(driving, "milling_run_stage", stage=stage)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["parameter"] == "stage.monitors[0].config.drop_fractoin"
    assert "did you mean stage.monitors[0].config.drop_fraction" in detail["message"]


def test_the_agent_ends_a_stage_its_monitor_suggests_ending(
    driving,  # noqa: F811
    milling_for_a_while,
):
    run = _ok(_call(driving, "milling_run_stage", stage=_stage().to_dict()))
    run_id = run["result"]["run_id"]

    since, suggestion = 0, None
    deadline = time.monotonic() + 20
    while suggestion is None and time.monotonic() < deadline:
        followed = _ok(
            _call(driving, "milling_follow_readings", since=since, timeout=2)
        )["result"]
        since = followed["seq"]
        suggestion = next(
            (r for r in followed["readings"] if r["reading"].get("trigger")), None
        )
    assert suggestion is not None, "the monitor suggested the drop"
    assert suggestion["notable"] and suggestion["reading"]["on_trigger"] == "suggest"

    _ok(_call(driving, "milling_end_stage", reason="the fluorescence dropped"))

    ended = driving.get(RUN, params={"run_id": run_id, "wait": 10}).json()["run"]
    assert ended["state"] == "done", ended
    (stage,) = ended["result"]["stages"]
    assert stage["status"] == "completed"
    assert stage["end"] == {
        "kind": "ended",
        "by": "agent",
        "reason": "the fluorescence dropped",
    }
    (record,) = stage["steps"][0]["monitors"]
    assert record["facts"]["Drop seen"] is True


def test_ending_a_stage_needs_a_reason_and_is_refused_at_watch(driving):  # noqa: F811
    missing = _call(driving, "milling_end_stage")
    assert missing.status_code == 422
    assert missing.json()["detail"]["error_type"] == "missing_reason"

    driving.app.state.auth.level = type(driving.app.state.auth.level).WATCH
    refused = _call(driving, "milling_end_stage", reason="enough")
    assert refused.status_code == 403


def test_the_watch_prints_a_monitor_s_trigger_and_how_a_stage_ended():
    lines = []
    reading = {
        "monitor": "FM Intensity",
        "seq": 7,
        "milled_time": 31.0,
        "notable": True,
        "reading": {"trigger": "intensity drop", "on_trigger": "suggest"},
    }
    events = [
        {"seq": 1, "kind": "milling_reading", "payload": reading},
        {
            "seq": 2,
            "kind": "milling_reading",
            "payload": {**reading, "monitor": "step", "reading": {"paused": "x"}},
        },
        {
            "seq": 3,
            "kind": "milling",
            "payload": {
                "milling_run_id": "r1",
                "milling_task_name": "Rough",
                "status": "completed",
                "stages": [
                    {
                        "name": "Rough",
                        "status": "completed",
                        "end": {"kind": "ended", "by": "agent", "reason": "dropped"},
                    }
                ],
            },
        },
    ]

    watch_mod._print_events(events, 0, lines.append)

    assert lines == [
        "MILL READING monitor='FM Intensity' milled=31.0s trigger='intensity drop' "
        "on_trigger=suggest -- the mill goes on; milling_end_stage ends the stage "
        "if you agree",
        "MILL ENDED 'Rough' status=completed run=r1 stages: "
        "Rough=completed (ended by agent: dropped)",
    ]
