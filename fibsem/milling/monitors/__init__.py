"""Milling monitors: what runs during a mill step (see ``base``).

Built-ins, runtime registrations and the ``fibsem.milling.monitors`` entry
points. To add a plugin monitor, add to your package's pyproject.toml:

    [project.entry-points.'fibsem.milling.monitors']
    my_monitor = "my_package.monitors:MyMonitor"
"""

import logging
import typing

from fibsem.milling.monitors.base import (
    ON_TRIGGER_END,
    ON_TRIGGER_SUGGEST,
    MillingMonitor,
    MillingMonitorConfig,
    MillingMonitorResult,
    MillingPeriodicMonitor,
    MillingPeriodicMonitorConfig,
)
from fibsem.plugins.loader import PluginRecord, PluginRegistry, subclass_of

__all__ = [
    "ON_TRIGGER_END",
    "ON_TRIGGER_SUGGEST",
    "MillingMonitor",
    "MillingMonitorConfig",
    "MillingMonitorResult",
    "MillingPeriodicMonitor",
    "MillingPeriodicMonitorConfig",
    "get_monitor_config",
    "get_monitors",
    "monitor_class_for",
    "register_monitor",
]

MONITOR_ENTRY_POINT_GROUP = "fibsem.milling.monitors"
BUILTIN_MONITORS: typing.Dict[str, typing.Type[MillingMonitor[typing.Any]]] = {}

MONITOR_PLUGINS: PluginRegistry[typing.Type[MillingMonitor[typing.Any]]] = (
    PluginRegistry(
        group=MONITOR_ENTRY_POINT_GROUP,
        kind="monitor",
        resolve=subclass_of(MillingMonitor, lambda cls: cls.name),
        builtins=BUILTIN_MONITORS,
    )
)


def get_monitors() -> typing.Dict[str, typing.Type[MillingMonitor[typing.Any]]]:
    """Every monitor by name: built-in, then registered, then plugin on a clash."""
    return MONITOR_PLUGINS.all()


def register_monitor(monitor_cls: typing.Type[MillingMonitor[typing.Any]]) -> None:
    MONITOR_PLUGINS.register(monitor_cls.name, monitor_cls)


def get_monitor_plugin_records() -> typing.Tuple[PluginRecord, ...]:
    """Every ``fibsem.milling.monitors`` entry point and what became of it."""
    return MONITOR_PLUGINS.plugin_records()


def monitor_class_for(
    config_class: typing.Type[MillingMonitorConfig],
) -> typing.Type[MillingMonitor[typing.Any]]:
    """The monitor whose config class *config_class* is."""
    for monitor in get_monitors().values():
        if monitor.config_class is config_class:
            return monitor
    raise ValueError(
        f"No milling monitor is configured by {config_class.__name__}; "
        "register the monitor it belongs to."
    )


def get_monitor_config(
    saved: typing.Dict[str, typing.Any],
) -> typing.Optional[MillingMonitorConfig]:
    """The config of a saved ``{name, config}`` monitor; None, with a warning,
    for a monitor not installed here (it is dropped from the stage)."""
    name = saved.get("name")
    monitor = get_monitors().get(name)
    if monitor is None:
        logging.warning(
            f"No milling monitor is called {name!r} (a plugin not installed "
            "here?); the stage mills without it."
        )
        return None
    return monitor.config_class.from_dict(saved.get("config") or {}).made_for(monitor)
