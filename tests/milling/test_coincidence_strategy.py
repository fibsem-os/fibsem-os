"""Serialization / registry behaviour for the coincidence milling strategy.

These lock in two fixes the coincidence viewer's config persistence relies on:
- the strategy config's ``bbox`` (FibsemRectangle) round-trips through to_dict/from_dict
- the strategy is resolvable by name for (de)serialisation while staying out of
  the generic strategy selectors (``selectable = False``)
"""

import os
import threading
import time

import pytest
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.milling.base import FibsemMillingStage, get_strategy
from fibsem.milling.patterning.patterns2 import RectanglePattern
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

    assert isinstance(restored_strategy, CoincidenceMillingStrategy)
    assert restored_strategy.config.intensity_drop_fraction == 0.55
    assert restored_strategy.config.supervised is False
    assert isinstance(restored_strategy.config.bbox, FibsemRectangle)
    assert restored_strategy.config.bbox == strategy.config.bbox


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

    fm_microscope.milling_progress_signal.connect(_stop_on_first_tick)
    try:
        started = time.time()
        task = run_milling_task(fm_microscope, config, None, stop_event=stop_event)
        elapsed = time.time() - started
    finally:
        fm_microscope.milling_progress_signal.disconnect(_stop_on_first_tick)

    # the task runs its own copy of the config; that copy's strategy is the one that ran
    strategy = task.config.stages[0].strategy
    assert isinstance(strategy, CoincidenceMillingStrategy)
    assert strategy.microscope is fm_microscope
    assert task._stop_event is stop_event
    assert strategy.is_cancelled
    # stopped on the tick after the event, nowhere near the 600 s timeout
    assert 1 <= len(ticks) <= 2
    assert elapsed < 30
    # finalised: FM acquisition stopped
    assert not fm_microscope.fm.is_acquiring


def test_headless_run_never_starts_on_a_preset_stop_event(fm_microscope, tmp_path):
    """The milling task checks the same event before each stage, so a stop requested
    before the run begins never reaches the strategy."""
    config = _coincidence_config(str(tmp_path))
    stop_event = threading.Event()
    stop_event.set()

    task = run_milling_task(fm_microscope, config, None, stop_event=stop_event)

    strategy = task.config.stages[0].strategy
    assert not hasattr(strategy, "microscope")  # never set up


def test_milling_task_prefers_explicit_stop_event_over_widget(fm_microscope):
    """An explicit event wins; without one the widget's event is borrowed as before."""
    from fibsem.milling.tasks import FibsemMillingTask

    class _Widget:
        _milling_stop_event = threading.Event()

    explicit = threading.Event()
    config = FibsemMillingTaskConfig(name="x", stages=[])
    task = FibsemMillingTask(
        fm_microscope, config, parent_ui=_Widget(), stop_event=explicit
    )
    assert task._stop_event is explicit
    task = FibsemMillingTask(fm_microscope, config, parent_ui=_Widget())
    assert task._stop_event is _Widget._milling_stop_event
    task = FibsemMillingTask(fm_microscope, config, parent_ui=None)
    assert task._stop_event is None
