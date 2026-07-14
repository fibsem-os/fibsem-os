"""Serialization / registry behaviour for the coincidence milling strategy.

These lock in two fixes the coincidence viewer's config persistence relies on:
- the strategy config's ``bbox`` (FibsemRectangle) round-trips through to_dict/from_dict
- the strategy is resolvable by name for (de)serialisation while staying out of
  the generic strategy selectors (``selectable = False``)
"""

from fibsem.milling.base import FibsemMillingStage, get_strategy
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.strategy import get_strategy_names
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig
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
