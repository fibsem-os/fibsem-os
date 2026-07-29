import typing

from fibsem.milling.base import MillingStrategy
from fibsem.plugins.loader import PluginRecord, load_entry_point_group, plugin_classes
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy # noqa: F401

DEFAULT_STRATEGY = StandardMillingStrategy
DEFAULT_STRATEGY_NAME = DEFAULT_STRATEGY.name
BUILTIN_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
    CoincidenceMillingStrategy.name: CoincidenceMillingStrategy,
}
REGISTERED_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {}
STRATEGY_ENTRY_POINT_GROUP = "fibsem.strategies"


def get_strategies() -> typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]]:
    # This order means that builtins > registered > plugins if there are any name clashes
    return {**_get_plugin_strategies(), **REGISTERED_STRATEGIES, **BUILTIN_STRATEGIES}


def get_strategy_names() -> typing.List[str]:
    return [name for name, cls in get_strategies().items() if getattr(cls, "selectable", True)]


def register_strategy(strategy_cls: typing.Type[MillingStrategy[typing.Any]]) -> None:
    global REGISTERED_STRATEGIES
    REGISTERED_STRATEGIES[strategy_cls.name] = strategy_cls


def get_strategy_plugin_records() -> typing.Tuple[PluginRecord, ...]:
    """Every ``fibsem.strategies`` entry point and what became of it.

    Loading happens once per process, on the first call. Includes the plugins
    that failed and the ones a built-in later shadows, neither of which
    survives into :func:`get_strategies` -- see ``fibsem.plugins.report``.

    To add a plugin strategy, add to your package's pyproject.toml:

    [project.entry-points.'fibsem.strategies']
    my_strategy = "my_package.strategies:MyCustomStrategy"
    """
    return load_entry_point_group(
        group=STRATEGY_ENTRY_POINT_GROUP,
        base_cls=MillingStrategy,
        name_of=lambda cls: cls.name,
        kind="strategy",
    )


def _get_plugin_strategies() -> typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]]:
    """Plugin strategies that loaded, as ``{name: class}``."""
    return plugin_classes(get_strategy_plugin_records())
