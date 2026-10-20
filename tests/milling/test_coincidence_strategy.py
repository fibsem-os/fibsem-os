"""Serialization / registry behaviour for the coincidence milling strategy.

These lock in two fixes the coincidence viewer's config persistence relies on:
- the strategy config's ``bbox`` (FibsemRectangle) round-trips through to_dict/from_dict
- the strategy is resolvable by name for (de)serialisation while staying out of
  the generic strategy selectors (``selectable = False``)
"""

import os
import threading
import time

import numpy as np
import pytest
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.milling.base import FibsemMillingStage, get_strategy
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.steps import ENDED, MillingStepEnd
from fibsem.milling.strategy import get_strategy_names
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import FibsemMillingSettings, FibsemRectangle


def test_coincidence_config_bbox_roundtrip():
    """bbox must come back as a FibsemRectangle, not a plain dict."""
    config = CoincidenceMillingStrategyConfig(
        intensity_drop_fraction=0.6,
        supervised=False,
        bbox=FibsemRectangle(left=0.1, top=0.2, width=0.3, height=0.4),
    )
    restored = CoincidenceMillingStrategyConfig.from_dict(config.to_dict())

    assert isinstance(restored.bbox, FibsemRectangle)
    assert restored.bbox == config.bbox
    assert restored.intensity_drop_fraction == 0.6
    assert restored.supervised is False


def test_coincidence_config_none_bbox_roundtrip():
    """A None bbox must round-trip to None (not a dict / not crash)."""
    restored = CoincidenceMillingStrategyConfig.from_dict(
        CoincidenceMillingStrategyConfig().to_dict()
    )
    assert restored.bbox is None


def test_coincidence_strategy_resolvable_but_hidden():
    """Resolvable by get_strategy, but excluded from the generic selectors."""
    assert "CoincidenceMilling" not in get_strategy_names()
    assert CoincidenceMillingStrategy.selectable is False

    strategy = get_strategy("CoincidenceMilling")
    assert isinstance(strategy, CoincidenceMillingStrategy)


def test_coincidence_task_config_roundtrip_preserves_strategy():
    """A full task-config round-trip must keep the coincidence strategy + config,
    not silently fall back to the default strategy."""
    strategy = CoincidenceMillingStrategy()
    strategy.config.intensity_drop_fraction = 0.55
    strategy.config.supervised = False
    strategy.config.bbox = FibsemRectangle(left=0.1, top=0.2, width=0.3, height=0.4)

    config = FibsemMillingTaskConfig(
        name="Coincidence Milling Task",
        field_of_view=100e-6,
        stages=[
            FibsemMillingStage(
                name="Coincidence Milling Stage",
                milling=FibsemMillingSettings(hfw=100e-6, milling_current=0.1e-9),
                pattern=RectanglePattern(width=5e-6, height=8e-6, depth=2e-6),
                strategy=strategy,
            )
        ],
    )

    restored = FibsemMillingTaskConfig.from_dict(config.to_dict())
    restored_strategy = restored.stages[0].strategy

    assert isinstance(restored_strategy, CoincidenceMillingStrategyConfig)
    assert restored_strategy.intensity_drop_fraction == 0.55
    assert restored_strategy.supervised is False
    assert isinstance(restored_strategy.bbox, FibsemRectangle)
    assert restored_strategy.bbox == strategy.config.bbox


# ---------------------------------------------------------------------------
# Headless cancellation: a caller-supplied stop event ends the run
# ---------------------------------------------------------------------------

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)


@pytest.fixture
def fm_microscope(tmp_path):
    """A Demo microscope with a fluorescence module and the sample scene off.

    The shipped Arctis configuration renders the scene for every frame, which
    is far slower than a noise frame and irrelevant to control flow.
    """
    with open(SIM_ARCTIS_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    assert microscope.fm is not None
    yield microscope
    microscope.disconnect()


def _coincidence_config(path: str) -> FibsemMillingTaskConfig:
    strategy = CoincidenceMillingStrategy(
        config=CoincidenceMillingStrategyConfig(
            timeout=600,
            save_fm_images=False,
            acquire_fib_image=False,
            bbox=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
        )
    )
    config = FibsemMillingTaskConfig(
        name="Coincidence Cancel",
        field_of_view=80e-6,
        stages=[
            FibsemMillingStage(
                name="Coincidence 01",
                milling=FibsemMillingSettings(milling_current=60e-12),
                pattern=RectanglePattern(width=9e-6, height=20e-6, depth=4e-6),
                strategy=strategy,
            )
        ],
    )
    config.alignment.enabled = False
    config.acquisition.acquire_fib = False
    config.acquisition.acquire_sem = False
    config.acquisition.imaging.path = path
    return config


def test_headless_run_stops_on_the_callers_stop_event(fm_microscope, tmp_path):
    """No milling widget: a stop event set while the strategy is monitoring ends
    the run and it still finalises, saving its files."""
    config = _coincidence_config(str(tmp_path))
    stop_event = threading.Event()
    frames = []

    def _stop_on_a_frame(stats):
        frames.append(stats)
        stop_event.set()

    config.stages[0].get_strategy().intensity_stats_signal.connect(_stop_on_a_frame)

    milling = []

    def _milling(kind, payload):
        if kind == "milling":
            milling.append(payload)

    fm_microscope.session.record_signal.connect(_milling)
    try:
        started = time.time()
        task = run_milling_task(fm_microscope, config, stop_event=stop_event)
        elapsed = time.time() - started
    finally:
        fm_microscope.session.record_signal.disconnect(_milling)

    # the task keeps the config it was given; its stage's strategy is the one that ran
    strategy = task.config.stages[0].get_strategy()
    assert isinstance(strategy, CoincidenceMillingStrategy)
    assert strategy.microscope is fm_microscope
    assert task._stop_event is stop_event
    assert strategy.is_cancelled
    # stopped soon after the event, nowhere near the 600 s timeout
    assert elapsed < 30
    # finalised: FM acquisition stopped
    assert not fm_microscope.fm.is_acquiring
    # and the stage's record says why the mill ended, as the strategy saw it (FIB-1257)
    (run,) = milling
    said = run["stages"][0]["strategy"]
    assert said["name"] == CoincidenceMillingStrategy.name
    assert said["end_reason"] == "stopped by the operator"
    assert 0 < said["facts"]["Mill time (s)"] < 30
    assert said["facts"]["Intensity drop seen"] is False
    assert os.path.isfile(said["figure"])
    for name in ("intensities.npy", "intensity_stats.csv"):
        assert os.path.isfile(os.path.join(strategy.path, name))


def test_headless_run_never_starts_on_a_preset_stop_event(fm_microscope, tmp_path):
    """The milling task checks the same event before each stage, so a stop requested
    before the run begins never reaches the strategy."""
    config = _coincidence_config(str(tmp_path))
    stop_event = threading.Event()
    stop_event.set()

    task = run_milling_task(fm_microscope, config, stop_event=stop_event)

    strategy = task.config.stages[0].get_strategy()
    assert not hasattr(strategy, "microscope")  # never set up


def test_milling_task_cancels_by_the_stop_event_it_is_given(fm_microscope):
    """The milling widget hands its own event in, like any other caller."""
    from fibsem.milling.tasks import FibsemMillingTask

    explicit = threading.Event()
    config = FibsemMillingTaskConfig(name="x", stages=[])
    task = FibsemMillingTask(fm_microscope, config, stop_event=explicit)
    assert task._stop_event is explicit
    task = FibsemMillingTask(fm_microscope, config)
    assert task._stop_event is None


def _old_drop_stats(values, elapsed, settings):
    """Coincidence milling's drop detection as it was before the rule was
    shared (FIB-1482), reading the settings at every frame."""
    peak, below, warm, seen, out = 0.0, 0, False, [], []
    for i, (value, t) in enumerate(zip(values, elapsed)):
        drop_fraction, window, warmup, consecutive = settings(i)
        seen.append(value)
        rolling = float(np.mean(seen[-int(window) :]))
        if not warm:
            if t >= warmup:
                warm = True
            out.append((rolling, peak, 0.0, False, False))
            continue
        peak = max(peak, rolling)
        threshold = (1.0 - drop_fraction) * peak
        below = below + 1 if rolling < threshold else 0
        drop = below >= int(consecutive)
        if drop:
            below = 0
        out.append((rolling, peak, threshold, True, drop))
    return out


class _Step:
    """Just enough of a mill step for a monitor fed frames by hand: the milled
    time is whatever the test sets."""

    def __init__(self, fm):
        from types import SimpleNamespace

        self.microscope = SimpleNamespace(fm=fm)
        self.output_dir = None
        self.ending = threading.Event()
        self.pausing = threading.Event()
        self.time = 0.0
        self.ends = []

    def milled_time(self):
        return self.time

    def report(self, *args, **kwargs):
        pass

    def end_step(self, reason, by=None):
        self.ends.append((reason, by))
        self.ending.set()


class _Fm:
    def __init__(self):
        from psygnal import SignalInstance

        self.acquisition_signal = SignalInstance(signature=(object,))
        self.is_acquiring = False

    def start_acquisition(self):
        self.is_acquiring = True

    def stop_acquisition(self):
        self.is_acquiring = False

    @property
    def is_streaming(self):
        return self.is_acquiring

    def set_active_channel(self):
        pass


def _strategy(**config):
    strategy = CoincidenceMillingStrategy(
        config=CoincidenceMillingStrategyConfig(save_fm_images=False, **config)
    )
    strategy.camera_resolution = (8, 8)
    return strategy


def _monitoring(strategy):
    """The strategy's monitor, started on a step whose frames the test sends."""
    fm = _Fm()
    step = _Step(fm)
    monitor = strategy._make_monitor()
    monitor.start(step)
    return monitor, step, fm


def _frame(value, shape=(8, 8)):
    from types import SimpleNamespace

    return SimpleNamespace(data=np.full(shape, value, dtype=float))


def test_the_drop_rule_detects_as_the_old_strategy_did_with_live_edits():
    """The viewer's drop threshold changes the running strategy's config; the
    monitor takes it at the next frame, as the old strategy's own code did."""
    strategy = _strategy(rolling_window=3, warmup_duration=2.0, consecutive_triggers=2)
    monitor, step, fm = _monitoring(strategy)
    rng = np.random.default_rng(1)
    values = list(rng.normal(1000, 30, 20)) + list(rng.normal(500, 30, 20))
    values += list(rng.normal(1000, 30, 10)) + list(rng.normal(700, 30, 20))
    fractions = [0.4] * 45 + [0.2] * 25  # the operator lowers the threshold

    stats = []
    strategy.intensity_stats_signal.connect(stats.append)
    for i, value in enumerate(values):
        strategy.config.intensity_drop_fraction = fractions[i]
        step.time = float(i)
        fm.acquisition_signal.emit(_frame(value))
    monitor.stop()

    expected = _old_drop_stats(
        values,
        list(range(len(values))),
        lambda i: (fractions[i], 3, 2.0, 2),
    )
    got = [
        (
            s["rolling_mean"],
            s["peak_rolling_mean"],
            s["threshold_value"],
            s["warmup_complete"],
            s.get("drop_detected", False),
        )
        for s in stats
    ]
    assert [row[3:] for row in got] == [row[3:] for row in expected]
    assert [v for row in got for v in row[:3]] == pytest.approx(
        [v for row in expected for v in row[:3]]
    )
    drops = [i for i, row in enumerate(expected) if row[4]]
    assert drops and any(i >= 45 for i in drops), "the lowered threshold fires"
    assert strategy._drop_detected
    assert step.ends == [], "supervised: a drop is only suggested"
    assert len(strategy.intensities) == len(values)
    assert len(strategy._stats_records) == len(values)


def test_the_viewer_gets_the_stats_it_draws():
    strategy = _strategy(rolling_window=1, warmup_duration=1.0)
    _, step, fm = _monitoring(strategy)
    stats, crops = [], []
    strategy.intensity_stats_signal.connect(stats.append)
    strategy.cropped_image_signal.connect(crops.append)
    for t in (0.0, 2.0, 3.0):
        step.time = t
        fm.acquisition_signal.emit(_frame(100.0))

    warming, warm = stats[0], stats[-1]
    assert set(warming) == {
        "value",
        "rolling_mean",
        "peak_rolling_mean",
        "threshold_value",
        "warmup_complete",
    }
    for key in (
        "value",
        "peak_rolling_mean",
        "threshold_value",
        "warmup_complete",
        "drop_fraction",
        "threshold_fraction",
        "drop_detected",
    ):
        assert key in warm
    assert [c["arr"].shape for c in crops] == [(8, 8)] * 3
    assert strategy.first_fm_acq is not None and strategy.last_fm_acq is not None


def test_the_viewers_box_is_where_the_intensity_is_read():
    strategy = _strategy(rolling_window=1)
    _, step, fm = _monitoring(strategy)
    crops = []
    strategy.cropped_image_signal.connect(crops.append)
    strategy._on_bbox_update(FibsemRectangle(0.0, 0.0, 0.5, 0.5))
    data = np.zeros((8, 8))
    data[:4, :4] = 10.0
    fm.acquisition_signal.emit(type("F", (), {"data": data})())

    assert crops[-1]["arr"].shape == (4, 4)
    assert strategy.intensities == [10.0]


def _a_drop(strategy, step, fm, start=0):
    for i, value in enumerate([1000.0] * 5 + [100.0] * 3):
        step.time = float(start + i)
        fm.acquisition_signal.emit(_frame(value))


def test_unsupervised_a_drop_ends_the_mill():
    strategy = _strategy(
        supervised=False, rolling_window=1, warmup_duration=1.0, consecutive_triggers=1
    )
    monitor, step, fm = _monitoring(strategy)
    _a_drop(strategy, step, fm)

    assert step.ends == [("intensity drop", monitor.by)]
    step.end = MillingStepEnd(ENDED, by=monitor.by, reason="intensity drop")
    assert strategy._why_it_ended() == "drop"


def test_switching_to_unsupervised_after_a_drop_ends_the_mill():
    """The old loop ended an unsupervised run on a drop seen before the switch."""
    strategy = _strategy(rolling_window=1, warmup_duration=1.0, consecutive_triggers=1)
    monitor, step, fm = _monitoring(strategy)
    _a_drop(strategy, step, fm)
    assert step.ends == []

    strategy.config.supervised = False
    fm.acquisition_signal.emit(_frame(100.0))
    assert step.ends[0] == ("intensity drop", monitor.by)


def test_the_time_limit_ends_the_mill():
    strategy = _strategy(timeout=0.2)
    monitor, step, _ = _monitoring(strategy)
    assert step.ending.wait(5.0)
    monitor.stop()
    assert step.ends == [("reached the time limit", monitor.by)]
    step.end = MillingStepEnd(ENDED, by=monitor.by, reason="reached the time limit")
    assert strategy._why_it_ended() == "timeout"
