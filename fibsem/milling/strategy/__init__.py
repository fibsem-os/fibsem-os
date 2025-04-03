import logging
from typing import Dict
from fibsem.milling.base import MillingStrategy
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy

DEFAULT_STRATEGY = StandardMillingStrategy
DEFAULT_STRATEGY_NAME = DEFAULT_STRATEGY.name
strategies: Dict[str, MillingStrategy] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
}
MILLING_STRATEGY_NAMES = list(strategies.keys())
def register_strategy(strategy_cls: MillingStrategy):
    strategies[strategy_cls.name] = strategy_cls
    MILLING_STRATEGY_NAMES.append(strategy_cls.name)

# Add adaptive polishing
try:
    from adaptive_polish.strategy import AdaptivePolishMillingStrategy
    register_strategy(strategy_cls=AdaptivePolishMillingStrategy)
    logging.info("Added adaptive polishing strategy")
except ImportError:
    logging.info("Adaptive polishing not found")
