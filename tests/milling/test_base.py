import pytest
from pydantic import ValidationError

from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
    get_strategy,
)
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.strategy import DEFAULT_STRATEGY
from fibsem.structures.settings import FibsemMillingSettings, MillingAlignment


def test_milling_stage():

    milling_settings = FibsemMillingSettings()
    pattern = RectanglePattern(width=10, height=5, depth=1)
    strategy = get_strategy("Standard")
    alignment = MillingAlignment(enabled=True)

    # Create a FibsemMillingStage instance
    milling_stage = FibsemMillingStage(
        name="Test Stage",
        num=1,
        milling=milling_settings,
        pattern=pattern,
        strategy=strategy,
        alignment=alignment,
    )

    # Check the attributes
    assert milling_stage.name == "Test Stage"
    assert milling_stage.num == 1
    assert isinstance(milling_stage.milling, FibsemMillingSettings)
    assert isinstance(milling_stage.pattern, RectanglePattern)
    assert isinstance(milling_stage.strategy, MillingStrategyConfig)
    assert isinstance(milling_stage.get_strategy(), MillingStrategy)
    assert isinstance(milling_stage.alignment, MillingAlignment)
    assert milling_stage.pattern.width == 10
    assert milling_stage.pattern.height == 5
    assert milling_stage.pattern.depth == 1
    assert milling_stage.strategy.strategy_name == DEFAULT_STRATEGY.name
    assert milling_stage.alignment.enabled is True

    # test to_dict method
    dict_repr = milling_stage.to_dict()
    assert dict_repr["name"] == "Test Stage"
    assert dict_repr["num"] == 1
    assert isinstance(dict_repr["milling"], dict)
    assert isinstance(dict_repr["pattern"], dict)
    assert isinstance(dict_repr["strategy"], dict)
    assert isinstance(dict_repr["alignment"], dict)

    # test from_dict method
    dict_repr = {
        "name": "Test Stage",
        "num": 1,
        "milling": milling_settings.to_dict(),
        "pattern": pattern.to_dict(),
        "strategy": strategy.to_dict(),
        "alignment": alignment.to_dict(),
    }
    new_milling_stage = FibsemMillingStage.from_dict(dict_repr)

    assert new_milling_stage.name == "Test Stage"
    assert new_milling_stage.num == 1
    assert isinstance(new_milling_stage.milling, FibsemMillingSettings)
    assert isinstance(new_milling_stage.pattern, RectanglePattern)
    assert isinstance(new_milling_stage.strategy, MillingStrategyConfig)
    assert isinstance(new_milling_stage.alignment, MillingAlignment)
    assert new_milling_stage.imaging.path is None


def test_get_strategy():
    # Test with default strategy
    strategy = get_strategy()
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == DEFAULT_STRATEGY.name

    # Test with a specific strategy name
    strategy_name = "Standard"
    strategy = get_strategy(name=strategy_name)
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == strategy_name

    # Test with a non-existent strategy name
    strategy = get_strategy(name="NonExistentStrategy")
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == DEFAULT_STRATEGY.name


def test_a_stage_holds_the_config_and_mills_with_one_strategy_made_from_it():
    stage = FibsemMillingStage()
    strategy = stage.get_strategy()

    assert strategy.config is stage.strategy
    assert stage.get_strategy() is strategy  # the same one until replaced
    stage.strategy = get_strategy("Overtilt").config
    assert stage.get_strategy() is not strategy
    assert stage.get_strategy().name == "Overtilt"


def test_a_stage_given_a_strategy_keeps_its_config_and_the_strategy():
    strategy = get_strategy("Overtilt")
    stage = FibsemMillingStage(strategy=strategy)

    assert stage.strategy is strategy.config
    assert stage.get_strategy() is strategy


def test_an_unknown_strategy_reads_as_standard_with_a_warning(caplog):
    data = FibsemMillingStage().to_dict()
    data["strategy"] = {"name": "NotInstalled", "config": {"passes": 3}}

    with caplog.at_level("WARNING"):
        stage = FibsemMillingStage.from_dict(data)

    assert stage.strategy.strategy_name == DEFAULT_STRATEGY.name
    assert "NotInstalled" in caplog.text


def test_a_strategy_sharing_a_config_class_keeps_its_name(monkeypatch):
    from fibsem.milling.strategy import STRATEGY_PLUGINS
    from fibsem.milling.strategy.standard import StandardMillingStrategy

    class _Shared(StandardMillingStrategy):
        name = "SharesStandardsConfig"

    monkeypatch.setitem(STRATEGY_PLUGINS.registered, _Shared.name, _Shared)
    stage = FibsemMillingStage(strategy=_Shared())
    assert stage.to_dict()["strategy"]["name"] == "SharesStandardsConfig"

    restored = FibsemMillingStage.from_dict(stage.to_dict())
    assert type(restored.get_strategy()) is _Shared
    assert restored.to_dict()["strategy"]["name"] == "SharesStandardsConfig"


@pytest.mark.parametrize("key", ["name", "milling", "pattern"])
def test_a_saved_stage_missing_a_key_it_cannot_mill_without_is_refused(key):
    saved = FibsemMillingStage(pattern=RectanglePattern(width=10, height=5)).to_dict()
    del saved[key]
    with pytest.raises(ValidationError, match=f"missing {key}"):
        FibsemMillingStage.from_dict(saved)
