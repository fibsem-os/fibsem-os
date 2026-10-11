import typing

from fibsem.milling.base import MillingStrategy
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy  # noqa: F401
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.plugins.loader import PluginRecord, PluginRegistry, subclass_of

DEFAULT_STRATEGY = StandardMillingStrategy
DEFAULT_STRATEGY_NAME = DEFAULT_STRATEGY.name
BUILTIN_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
    CoincidenceMillingStrategy.name: CoincidenceMillingStrategy,
}
STRATEGY_ENTRY_POINT_GROUP = "fibsem.strategies"

# Built-ins, runtime registrations and the fibsem.strategies entry points. To add
# a plugin strategy, add to your package's pyproject.toml:
#
#     [project.entry-points.'fibsem.strategies']
#     my_strategy = "my_package.strategies:MyCustomStrategy"
STRATEGY_PLUGINS: PluginRegistry[typing.Type[MillingStrategy[typing.Any]]] = (
    PluginRegistry(
        group=STRATEGY_ENTRY_POINT_GROUP,
        kind="strategy",
        resolve=subclass_of(MillingStrategy, lambda cls: cls.name),
        builtins=BUILTIN_STRATEGIES,
    )
)
REGISTERED_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = (
    STRATEGY_PLUGINS.registered
)


def get_strategies() -> typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]]:
    """Every strategy by name: built-in, then registered, then plugin on a clash."""
    return STRATEGY_PLUGINS.all()


def get_strategy_names() -> typing.List[str]:
    return [
        name
        for name, cls in get_strategies().items()
        if getattr(cls, "selectable", True)
    ]


def register_strategy(strategy_cls: typing.Type[MillingStrategy[typing.Any]]) -> None:
    STRATEGY_PLUGINS.register(strategy_cls.name, strategy_cls)


def get_strategy_plugin_records() -> typing.Tuple[PluginRecord, ...]:
    """Every ``fibsem.strategies`` entry point and what became of it."""
    return STRATEGY_PLUGINS.plugin_records()
