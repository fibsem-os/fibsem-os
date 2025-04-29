from __future__ import annotations
import logging
import typing
from functools import cache

from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy

if typing.TYPE_CHECKING:
    from fibsem.milling.base import MillingStrategy

DEFAULT_STRATEGY = StandardMillingStrategy
DEFAULT_STRATEGY_NAME = DEFAULT_STRATEGY.name
BUILTIN_STRATEGIES: typing.Dict[str, MillingStrategy] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
}


@cache
def get_strategies() -> typing.AwaitableGeneratorDict[str, type[MillingStrategy]]:
    strategies = BUILTIN_STRATEGIES.copy()
    for strategy in _get_additional_strategies():
        strategies[strategy.name] = strategy
    return strategies


def get_strategy_names() -> typing.List[str]:
    return list(get_strategies().keys())


def _get_additional_strategies() -> typing.List[type[MillingStrategy]]:
    """Import new strategies and append them to the list here"""
    # Add adaptive polishing
    strategies: typing.List[MillingStrategy] = []
    try:
        from adaptive_polish.strategy import AdaptivePolishMillingStrategy

        strategies.append(AdaptivePolishMillingStrategy)
        logging.info("Added adaptive polishing strategy")
    except ImportError:
        logging.error("Adaptive polishing not found", exc_info=True)
    return strategies
