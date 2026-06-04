import typing

from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig, get_autofocus_strategy, get_strategy_from_config, run_autofocus_strategy
from fibsem.fm.strategy.sweep import SweepAutoFocusStrategy, SweepAutoFocusConfig
from fibsem.fm.strategy.coarse_fine import CoarseFineAutoFocusStrategy, CoarseFineAutoFocusConfig
from fibsem.fm.strategy.iterative import IterativeAutoFocusStrategy, IterativeAutoFocusConfig

DEFAULT_STRATEGY = SweepAutoFocusStrategy
DEFAULT_STRATEGY_NAME = SweepAutoFocusConfig.name

# Maps config name → strategy class. Keyed on the config's ClassVar name so
# you can recover the strategy class from any config instance: AUTOFOCUS_STRATEGIES[config.name]
AUTOFOCUS_STRATEGIES: typing.Dict[str, typing.Type[AutoFocusStrategy]] = {
    SweepAutoFocusConfig.name: SweepAutoFocusStrategy,
    CoarseFineAutoFocusConfig.name: CoarseFineAutoFocusStrategy,
    IterativeAutoFocusConfig.name: IterativeAutoFocusStrategy,
}

# Maps config name → config class, for deserialising a dict without knowing the type up front.
AUTOFOCUS_STRATEGY_CONFIGS: typing.Dict[str, typing.Type[AutoFocusStrategyConfig]] = {
    SweepAutoFocusConfig.name: SweepAutoFocusConfig,
    CoarseFineAutoFocusConfig.name: CoarseFineAutoFocusConfig,
    IterativeAutoFocusConfig.name: IterativeAutoFocusConfig,
}


def get_strategy_names() -> typing.List[str]:
    return list(AUTOFOCUS_STRATEGIES.keys())


def load_autofocus_config(d: typing.Dict[str, typing.Any]) -> AutoFocusStrategyConfig:
    """Deserialise a config dict (which must contain a "name" key) to the correct config class."""
    name = d.get("name", DEFAULT_STRATEGY_NAME)
    config_cls = AUTOFOCUS_STRATEGY_CONFIGS.get(name, SweepAutoFocusConfig)
    return config_cls.from_dict(d)


__all__ = [
    "AutoFocusStrategy",
    "AutoFocusStrategyConfig",
    "get_autofocus_strategy",
    "get_strategy_from_config",
    "run_autofocus_strategy",
    "get_strategy_names",
    "load_autofocus_config",
    "SweepAutoFocusStrategy",
    "SweepAutoFocusConfig",
    "CoarseFineAutoFocusStrategy",
    "CoarseFineAutoFocusConfig",
    "IterativeAutoFocusStrategy",
    "IterativeAutoFocusConfig",
    "AUTOFOCUS_STRATEGIES",
    "AUTOFOCUS_STRATEGY_CONFIGS",
    "DEFAULT_STRATEGY",
    "DEFAULT_STRATEGY_NAME",
]
