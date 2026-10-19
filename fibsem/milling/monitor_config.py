"""What a milling monitor is configured by, held on the stage (FIB-1481).

Outside the monitors package so ``fibsem.milling.base`` can name it on the
stage without importing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, TypeVar

from fibsem.structures import Configurable, param

if TYPE_CHECKING:
    from fibsem.milling.monitors.base import MillingMonitor

TMillingMonitorConfig = TypeVar("TMillingMonitorConfig", bound="MillingMonitorConfig")


@dataclass
class MillingMonitorConfig(Configurable):
    """What a monitor is configured by, held on the stage. Which monitor is the
    one whose ``config_class`` this is."""

    # Saved as `{name, config}`, like a strategy's: a patch names the fields
    # `stages.0.monitors.0.config.<field>`.
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

    def to_dict(self) -> Dict[str, Any]:
        from fibsem.codec import encode

        return encode(self)

    @classmethod
    def from_dict(
        cls: Type[TMillingMonitorConfig], d: Dict[str, Any]
    ) -> TMillingMonitorConfig:
        from fibsem.codec import decode

        return decode(cls, d)
