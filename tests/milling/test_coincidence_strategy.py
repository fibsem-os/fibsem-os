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
    fconfig.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
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
            timeout=600, save_fm_images=False, acquire_fib_image=False
        )
    )
    config = FibsemMillingTaskConfig(
        name="Coincidence Cancel",
        field_of_view=80e-6,
        stages=[
            FibsemMillingStage(
                name="Coincidence 01",
                milling=FibsemMillingSettings(milling_current=60e-12),
                pattern=RectanglePattern(width=9e-6, height=20e-6, depth=0.4e-6),
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
    """No milling widget: a stop event set while the strategy is monitoring must end
    the run on the next tick and still finalise. Before the strategy read the event
    a headless run could only end at its own timeout."""
    from fibsem.milling.progress import MillingProgressStatus

    config = _coincidence_config(str(tmp_path))
    stop_event = threading.Event()
    ticks = []

    def _stop_on_first_tick(progress):
        # the strategy's monitor loop emits STAGE_UPDATE once per tick
        if progress.status is MillingProgressStatus.STAGE_UPDATE:
            ticks.append(progress)
            stop_event.set()

    milling = []

    def _milling(kind, payload):
        if kind == "milling":
            milling.append(payload)

    fm_microscope.milling.progress.changed.connect(_stop_on_first_tick)
    fm_microscope.session.record_signal.connect(_milling)
    try:
        started = time.time()
        task = run_milling_task(fm_microscope, config, stop_event=stop_event)
        elapsed = time.time() - started
    finally:
        fm_microscope.milling.progress.changed.disconnect(_stop_on_first_tick)
        fm_microscope.session.record_signal.disconnect(_milling)

    # the task keeps the config it was given; its stage's strategy is the one that ran
    strategy = task.config.stages[0].get_strategy()
    assert isinstance(strategy, CoincidenceMillingStrategy)
    assert strategy.microscope is fm_microscope
    assert task._stop_event is stop_event
    assert strategy.is_cancelled
    # stopped on the tick after the event, nowhere near the 600 s timeout
    assert 1 <= len(ticks) <= 2
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


def test_the_shared_drop_rule_detects_as_the_strategy_did_with_live_edits(
    monkeypatch,
):
    """The viewer's drop threshold changes the running strategy's config; the
    shared rule takes it at the next frame, as the strategy's own code did."""
    from types import SimpleNamespace

    strategy = CoincidenceMillingStrategy(
        config=CoincidenceMillingStrategyConfig(
            save_fm_images=False,
            rolling_window=3,
            warmup_duration=2.0,
            consecutive_triggers=2,
        )
    )
    strategy.microscope = SimpleNamespace(fm=object())
    strategy.camera_resolution = (8, 8)
    strategy._acquisition_start_time = 1000.0
    rng = np.random.default_rng(1)
    values = list(rng.normal(1000, 30, 20)) + list(rng.normal(500, 30, 20))
    values += list(rng.normal(1000, 30, 10)) + list(rng.normal(700, 30, 20))
    fractions = [0.4] * 45 + [0.2] * 25  # the operator lowers the threshold

    stats = []
    strategy.intensity_stats_signal.connect(stats.append)
    for i, value in enumerate(values):
        strategy.config.intensity_drop_fraction = fractions[i]
        monkeypatch.setattr(time, "time", lambda now=1000.0 + i: now)
        strategy.on_fm_acquisition_signal(SimpleNamespace(data=np.full((8, 8), value)))

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
