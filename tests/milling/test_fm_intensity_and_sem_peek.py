"""The built-in monitors (FIB-1482): FM Intensity, on the shared drop rule,
and SEM Peek."""

import os

import numpy as np
import pytest

from fibsem import config as fconfig
from fibsem import utils
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.monitors import FmIntensityMonitor, SemPeekMonitor, get_monitors
from fibsem.milling.monitors.fm_intensity import FmIntensityConfig
from fibsem.milling.monitors.intensity import IntensityDropRule
from fibsem.milling.monitors.sem_peek import SemPeekConfig
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.structures import FibsemRectangle
from tests.test_milling_run import (  # noqa: F401
    RUNNING,
    Scripted,
    microscope,
    scripted,
)


def test_the_drop_rule_waits_out_the_warmup_and_fires_once_per_drop():
    rule = IntensityDropRule(
        drop_fraction=0.5, rolling_window=1, warmup_duration=2, consecutive_triggers=2
    )
    values = [100, 100, 100, 100, 40, 40, 40, 40]
    fired = [rule.update(v, elapsed=t)["drop_detected"] for t, v in enumerate(values)]

    assert fired == [False, False, False, False, False, True, False, True]
    assert rule.peak == 100 and rule.drop_seen


def _stage(*configs):
    stage = FibsemMillingStage(name="s")
    stage.pattern = RectanglePattern(width=10e-6, height=5e-6, depth=1e-6)
    stage.monitors = list(configs)
    return stage


class Frame:
    def __init__(self, level):
        self.data = np.full((8, 8), float(level))

    def save(self, filename):
        open(filename, "w").close()


class FakeFM:
    """An FM whose frames the test feeds, one at each of the mill's waits."""

    from psygnal import Signal

    acquisition_signal = Signal(object)

    def __init__(self, levels):
        self.levels = list(levels)
        self.calls = []

    def start_acquisition(self, channel_settings=None):
        self.calls.append("start")

    def stop_acquisition(self):
        self.calls.append("stop")

    def set_active_channel(self):
        self.calls.append("view")

    def feed(self):
        if self.levels:
            self.acquisition_signal.emit(Frame(self.levels.pop(0)))


@pytest.fixture
def with_fm(scripted, monkeypatch):
    """The scripted service on a microscope with a fake FM fed at each wait."""
    import tests.test_milling_run as run_tests

    def make(levels, **kwargs):
        service = scripted([RUNNING] * 100, **kwargs)
        fm = FakeFM(levels)
        monkeypatch.setattr(service.parent, "fm", fm, raising=False)
        wait = service._wait

        def fed(seconds):
            fm.feed()
            wait(seconds)

        service._wait = fed
        return service, fm

    assert run_tests.Scripted is Scripted
    return make


def _mill(service, stage, tmp_path):
    stage.output_dir = str(tmp_path)
    with service.stage_run(stage) as stage_run:
        status = service.run()
    (step,) = stage_run.steps
    return status, step


def _fm(**kwargs):
    defaults = dict(warmup_duration=2, rolling_window=1, consecutive_triggers=2)
    return FmIntensityConfig(**{**defaults, **kwargs}).made_for(FmIntensityMonitor)


def test_fm_intensity_ends_the_step_on_a_drop(with_fm, tmp_path):
    service, fm = with_fm([100] * 5 + [30] * 5, estimate=50.0)
    status, step = _mill(service, _stage(_fm(on_trigger="end")), tmp_path)

    assert status == "completed"
    assert step.end.to_dict() == {
        "kind": "ended",
        "by": "monitor:FM Intensity",
        "reason": "intensity drop",
    }
    (record,) = step.to_dict()["monitors"]
    assert record["facts"]["Drop seen"] is True
    assert record["facts"]["Peak rolling mean"] == 100.0
    folder = record["files"]
    assert os.path.isfile(os.path.join(folder, "intensity.csv"))
    assert fm.calls[0] == "start" and fm.calls[-1] == "stop"
    assert "view" in fm.calls, "the FM view taken back after the mill started"


def test_fm_intensity_suggests_a_drop_and_milling_goes_on(with_fm, tmp_path):
    service, _ = with_fm([100] * 5 + [30] * 5, estimate=12.0)
    _, step = _mill(service, _stage(_fm()), tmp_path)

    assert step.end.kind == "finished"
    assert step.notable[0]["reading"]["on_trigger"] == "suggest"
    assert step.notable[0]["reading"]["drop_fraction"] == pytest.approx(0.3)


def test_fm_intensity_without_an_fm_fails_the_step_before_it_mills(
    scripted, monkeypatch, tmp_path
):
    from fibsem.services.milling import MillingStepFailed

    service = scripted([RUNNING] * 5)
    monkeypatch.setattr(service.parent, "fm", None, raising=False)
    with pytest.raises(MillingStepFailed, match="needs a fluorescence microscope"):
        _mill(service, _stage(_fm()), tmp_path)
    assert "start" not in service.steps


def test_both_are_built_in_and_their_configs_round_trip():
    assert {"FM Intensity", "SEM Peek"} <= set(get_monitors())
    stage = _stage(
        _fm(bbox=FibsemRectangle(0.25, 0.25, 0.5, 0.5)),
        SemPeekConfig(interval=10, hfw=50e-6).made_for(SemPeekMonitor),
    )
    loaded = FibsemMillingStage.from_dict(stage.to_dict())
    assert loaded.monitors == stage.monitors
    assert isinstance(loaded.monitors[0].bbox, FibsemRectangle)


@pytest.fixture
def milling_in_real_time(monkeypatch):
    """The Demo's mill takes its time (a second, looked at every 0.1 s), so a
    monitor's thread sees it; everything else stays simulated fast."""
    import time

    import fibsem.drivers.demo.services as demo_services

    monkeypatch.setattr(demo_services, "_PATTERN_TIME", 1)
    monkeypatch.setattr(demo_services, "sim_sleep", time.sleep)

    def setup(**kwargs):
        microscope, _ = utils.setup_session(
            manufacturer="Demo", setup_logging=False, **kwargs
        )
        microscope.milling.poll_interval = 0.1
        return microscope

    return setup


def test_sem_peek_pauses_images_and_resumes_on_the_demo(tmp_path, milling_in_real_time):
    demo = milling_in_real_time()
    peek = SemPeekConfig(interval=0.3, resolution=(96, 64))
    stage = _stage(peek.made_for(SemPeekMonitor))

    answer = demo.milling.run_stage(stage, path=str(tmp_path))

    (line,) = answer["stages"]
    (step,) = line["steps"]
    (record,) = step["monitors"]
    peeks = record["facts"]["Peeks"]
    assert peeks >= 1
    assert len(step["pauses"]) == peeks
    assert all(p["by"] == "monitor:SEM Peek" for p in step["pauses"])
    assert sorted(os.listdir(record["files"]))[0] == "sem-001.tif"
    assert record["figure"].endswith("sem-peeks.png")


def test_fm_intensity_reads_the_sim_arctis_fm(
    tmp_path, milling_in_real_time, monkeypatch
):
    import yaml

    import fibsem.drivers.demo.services as demo_services

    # The sample scene off: noise frames come fast, where a rendered scene
    # takes about a second a frame.
    with open(os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    config_path = tmp_path / "sim-arctis-configuration.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(demo_services, "_PATTERN_TIME", 2)
    arctis = milling_in_real_time(config_path=str(config_path))
    assert arctis.fm is not None
    stage = _stage(_fm(warmup_duration=0.5))
    stage.output_dir = str(tmp_path / "out")

    with arctis.milling.stage_run(stage) as stage_run:
        arctis.milling.draw(stage.pattern.define())
        assert arctis.milling.run() == "completed"

    (record,) = stage_run.steps[0].to_dict()["monitors"]
    assert "error" not in record
    assert record["facts"]["Frames"] >= 1
    assert os.path.isfile(os.path.join(record["files"], "intensity.csv"))
    assert not arctis.fm.is_acquiring, "the FM stopped with the step"
