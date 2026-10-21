"""What a milling monitor is configured by, held on the stage (FIB-1481).

Outside the monitors package so ``fibsem.milling.base`` can name it on the
stage without importing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Type, TypeVar

from fibsem.codec import Tagged, decode, encode
from fibsem.structures import Configurable, param

if TYPE_CHECKING:
    from fibsem.milling.monitors.base import MillingMonitor

TMillingMonitorConfig = TypeVar("TMillingMonitorConfig", bound="MillingMonitorConfig")


@dataclass
class MillingMonitorConfig(Tagged, Configurable):
    """What a monitor is configured by, held on the stage. Which monitor is the
    one whose ``config_class`` this is."""

    # Saved as `{name, config}`, like a strategy's: a patch names the fields
    # `stages.0.monitors.0.config.<field>`.
    TYPE_KEY: ClassVar[str] = "name"
    SAVED_UNDER: ClassVar[str] = "config"

    max_pause: float = param(
        60.0,
        unit="s",
        doc="The longest this monitor may pause the mill; longer fails the step.",
    )

    @property
    def monitor_class(self) -> Type["MillingMonitor[Any]"]:
        """The monitor this configures: the one it was made for, which a monitor
        sharing another's config class needs, else the one whose config it is."""
        from fibsem.milling.monitors import monitor_class_for

        return self.__dict__.get("_monitor_class") or monitor_class_for(type(self))

    def made_for(self, monitor: Type["MillingMonitor[Any]"]) -> "MillingMonitorConfig":
        """This config, marked as *monitor*'s. Not a field: not saved or compared."""
        object.__setattr__(self, "_monitor_class", monitor)
        return self

    @property
    def monitor_name(self) -> str:
        return self.monitor_class.name

    def tagged_name(self) -> str:
        return self.monitor_name

    @classmethod
    def tagged_variants(cls) -> Dict[str, type]:
        from fibsem.milling.monitors import get_monitors

        return {name: m.config_class for name, m in get_monitors().items()}

    @classmethod
    def read_tagged(cls, name: Optional[str], data: Any, strict: bool) -> Any:
        """The config of the monitor called ``name``; None, with a warning, for a
        monitor not installed here (the stage mills without it)."""
        from fibsem.milling.monitors import get_monitors

        monitor = get_monitors().get(name)
        if monitor is None:
            logging.warning(
                f"No milling monitor is called {name!r} (a plugin not installed "
                "here?); the stage mills without it."
            )
            return None
        return decode(monitor.config_class, data, strict=strict).made_for(monitor)

    def to_dict(self) -> Dict[str, Any]:
        return encode(self)

    @classmethod
    def from_dict(
        cls: Type[TMillingMonitorConfig], d: Dict[str, Any]
    ) -> TMillingMonitorConfig:
        return decode(cls, d)
