import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Generic

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.fm.structures import ChannelSettings
    from fibsem.structures import FibsemRectangle

TAutoFocusStrategyConfig = TypeVar("TAutoFocusStrategyConfig", bound="AutoFocusStrategyConfig")


class AutoFocusStrategyConfig(BaseModel):
    """Abstract base class for autofocus strategy configurations.

    Each subclass declares a ``name`` ClassVar that identifies it in the registry
    and is embedded in ``to_dict()`` so serialised configs are self-describing.
    Field metadata for UI rendering is stored in ``Field(json_schema_extra={...})``.
    """
    name: ClassVar[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["name"] = self.name
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    @classmethod
    def from_dict(cls: Type[TAutoFocusStrategyConfig], d: Dict[str, Any]) -> TAutoFocusStrategyConfig:
        return cls.model_validate({k: v for k, v in d.items() if k != "name"})

    @property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(self.__class__.model_fields.keys())

    @property
    def advanced_attributes(self) -> Tuple[str, ...]:
        return tuple(
            name for name, finfo in self.__class__.model_fields.items()
            if (finfo.json_schema_extra or {}).get("advanced", False)
        )

    @property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        return tuple(
            name for name, finfo in self.__class__.model_fields.items()
            if (finfo.json_schema_extra or {}).get("hidden", False)
        )

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        from fibsem.structures import DEFAULT_FIELD_METADATA
        result = {}
        for fname, finfo in self.__class__.model_fields.items():
            meta = dict(DEFAULT_FIELD_METADATA)
            meta.update(finfo.json_schema_extra or {})
            result[fname] = meta
        return result


class AutoFocusStrategy(ABC, Generic[TAutoFocusStrategyConfig]):
    """Abstract base class for autofocus strategies.

    Behaviour-only: holds a config instance and exposes run().
    Serialisation is handled entirely by the config class.
    The strategy name is owned by the config class.
    """
    config_class: ClassVar[Type[TAutoFocusStrategyConfig]]

    def __init__(self, config: Optional[TAutoFocusStrategyConfig] = None) -> None:
        self.config: TAutoFocusStrategyConfig = config or self.config_class()

    @property
    def name(self) -> str:
        return self.config_class.name

    def summary(self) -> str:
        from fibsem.utils import format_value
        lines = [f"    Strategy: {self.name}"]
        for attr in self.config.required_attributes:
            if attr in self.config._hidden_attributes or attr in self.config.advanced_attributes:
                continue
            val = getattr(self.config, attr)
            meta = self.config.field_metadata.get(attr, {})
            unit = meta.get("unit", None)
            label = meta.get("label") or attr.replace("_", " ").title()
            if isinstance(val, float) and unit:
                val_str = format_value(val, unit=unit, precision=1)
            else:
                val_str = val.name if hasattr(val, "name") and not isinstance(val, float) else str(val)
            lines.append(f"        {label}: {val_str}")
        return "\n".join(lines)

    @abstractmethod
    def run(
        self,
        microscope: "FluorescenceMicroscope",
        channel_settings: Optional["ChannelSettings"] = None,
        roi: Optional["FibsemRectangle"] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[float]:
        """Run autofocus and return the best focus z-position in metres, or None if cancelled."""
        pass


def get_autofocus_strategy(
    name: str = "Sweep", config: Optional[Dict[str, Any]] = None
) -> "AutoFocusStrategy[Any]":
    """Instantiate a strategy by name, optionally restoring config from a dict.

    To serialise a strategy:
        strategy.config.to_dict()           # includes "name" key
    To restore:
        get_autofocus_strategy(d["name"], d)
    """
    from fibsem.fm.strategy import AUTOFOCUS_STRATEGIES, DEFAULT_STRATEGY

    strategy_cls = AUTOFOCUS_STRATEGIES.get(name, DEFAULT_STRATEGY)
    config_obj = strategy_cls.config_class.from_dict(config or {})
    return strategy_cls(config=config_obj)


def get_strategy_from_config(config: AutoFocusStrategyConfig) -> "AutoFocusStrategy[Any]":
    """Instantiate the matching strategy class for a given config instance."""
    from fibsem.fm.strategy import AUTOFOCUS_STRATEGIES, DEFAULT_STRATEGY

    strategy_cls = AUTOFOCUS_STRATEGIES.get(config.name, DEFAULT_STRATEGY)
    return strategy_cls(config=config)


def run_autofocus_strategy(
    microscope: "FluorescenceMicroscope",
    config: AutoFocusStrategyConfig,
    channel_settings: Optional["ChannelSettings"] = None,
    roi: Optional["FibsemRectangle"] = None,
    stop_event: Optional[threading.Event] = None,
) -> Optional[float]:
    """Run an autofocus strategy from a config instance.

    Args:
        microscope: The fluorescence microscope to focus.
        config: Strategy config that determines which strategy is used and with what parameters.
        channel_settings: Optional channel to use during autofocus.
        roi: Optional region of interest (0–1 relative coords) for focus evaluation.
        stop_event: Optional threading event to cancel mid-run.

    Returns:
        Best focus z-position in metres, or None if cancelled.
    """
    return get_strategy_from_config(config).run(
        microscope=microscope,
        channel_settings=channel_settings,
        roi=roi,
        stop_event=stop_event,
    )
