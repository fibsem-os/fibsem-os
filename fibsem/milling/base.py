# Deferred annotations, so `FibsemMicroscope` below can be a type-checking-only
# import. See the note beside it.
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import AfterValidator

from fibsem.codec import NotSaved, Tagged, decode, encode, saved
from fibsem.milling.config import MILLING_SPUTTER_RATE
from fibsem.milling.monitor_config import MillingMonitorConfig
from fibsem.milling.patterning import DEFAULT_MILLING_PATTERN
from fibsem.milling.patterning.base import BasePattern
from fibsem.structures import (
    Configurable,
    CrossSectionPattern,
    FibsemImage,
    FibsemMillingSettings,
    FibsemPatternSettings,
    ImageSettings,
    MillingAlignment,
    Parameter,
    field_specs,
    flagged_fields,
    param,
)

if TYPE_CHECKING:
    # Type-checking only, so `fibsem.milling` stays microscope-free.
    # `fibsem/milling/__init__.py` imports this module, so a runtime import here
    # means importing *anything* under `fibsem.milling` pulls in
    # `fibsem.microscope` -- and `fibsem.microscope` needs to import the milling
    # progress contract, which makes that a cycle: the microscope module is only
    # half-built when this line runs, and `FibsemMicroscope` is not bound yet.
    #
    # The same fix `fibsem.imaging` already carries; see tests/test_import_cycles.py.
    # `from __future__ import annotations` is on at the top of this file, so the two
    # annotations below need no runtime object.
    from fibsem.microscope import FibsemMicroscope

TMillingStrategyConfig = TypeVar(
    "TMillingStrategyConfig", bound="MillingStrategyConfig"
)
TMillingStrategy = TypeVar("TMillingStrategy", bound="MillingStrategy")


@dataclass
class MillingStrategyConfig(Tagged, Configurable, ABC):
    """What a milling strategy is configured by, held on the stage it mills.

    The stage keeps this, not the strategy: the strategy is made from it when the
    stage mills (`FibsemMillingStage.get_strategy`). Which strategy is the one whose
    `config_class` this is, so a config knows its strategy's name.
    """

    # A stage saves its strategy as `{name, config}`, these fields under `config`.
    # A patch walks the saved shape (`stages.0.strategy.config.supervised`), so
    # `fibsem.server.config_patch` reads this segment as the config itself.
    TYPE_KEY: ClassVar[str] = "name"
    SAVED_UNDER: ClassVar[str] = "config"

    @property
    def strategy_class(self) -> Type["MillingStrategy[Any]"]:
        """The strategy this configures: the one it was made for, which a strategy
        sharing another's config class needs, else the one whose config class it is."""
        return self.__dict__.get("_strategy_class") or strategy_class_for(type(self))

    @property
    def strategy_name(self) -> str:
        """The name of the strategy this configures, as a stage saves it."""
        return self.strategy_class.name

    def tagged_name(self) -> str:
        return self.strategy_name

    @classmethod
    def tagged_variants(cls) -> Dict[str, type]:
        from fibsem.milling.strategy import get_strategies

        return {name: s.config_class for name, s in get_strategies().items()}

    @classmethod
    def read_tagged(cls, name: Optional[str], data: Any, strict: bool) -> Any:
        """The config of the strategy called ``name`` (Standard when absent).

        A name no strategy has reads as Standard, with a warning, and drops the
        saved settings, which are another strategy's (`_strategy_named`)."""
        strategy = _strategy_named(name or "Standard")
        config = data if strategy.name == (name or "Standard") else {}
        return _made_for(strategy, decode(strategy.config_class, config, strict=strict))

    def to_dict(self) -> Dict[str, Any]:
        return encode(self)

    @classmethod
    def from_dict(
        cls: Type[TMillingStrategyConfig], d: Dict[str, Any]
    ) -> TMillingStrategyConfig:
        return decode(cls, d)

    @cached_property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @cached_property
    def advanced_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are marked as advanced in the metadata."""
        return flagged_fields(self.__class__, "advanced")

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return flagged_fields(self.__class__, "hidden")


@dataclass
class MillingStrategyResult:
    """What a strategy says about its run, for the experiment's record: why it
    stopped, a few facts, a figure it saved. Returned by ``MillingStrategy.run``
    and recorded on the stage's line in the ``milling`` event (FIB-1257).

    ``facts`` are keyed by a label a person reads, with the unit in the label
    (``"Mill time (s)": 250``), and hold plain values: the record is JSON.
    """

    end_reason: Optional[str] = None  # "intensity drop", "reached the time limit"
    facts: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    figure: Optional[str] = None  # the path of a figure the run saved

    def to_dict(self) -> Dict[str, Any]:
        """As the record holds it: numbers as plain numbers, anything else as
        text, so a numpy value cannot stop the event being written."""
        facts: Dict[str, Union[str, int, float, bool]] = {}
        for label, value in self.facts.items():
            if isinstance(value, (bool, str)):
                facts[str(label)] = value
            elif isinstance(value, int):
                facts[str(label)] = int(value)
            else:
                try:
                    facts[str(label)] = float(value)
                except (TypeError, ValueError):
                    facts[str(label)] = str(value)
        record: Dict[str, Any] = {"facts": facts}
        if self.end_reason:
            record["end_reason"] = str(self.end_reason)
        if self.figure:
            record["figure"] = str(self.figure)
        return record


class MillingStrategy(ABC, Generic[TMillingStrategyConfig]):
    """Abstract base class for different milling strategies"""

    name: str = "Milling Strategy"
    config_class: Type[TMillingStrategyConfig]
    selectable: bool = True

    def __init__(self, config: Optional[TMillingStrategyConfig] = None) -> None:
        self.config: TMillingStrategyConfig = config or self.config_class()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls: Type[TMillingStrategy], d: Dict[str, Any]) -> TMillingStrategy:
        config = cls.config_class.from_dict(d.get("config", {}))
        return cls(config=config)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the strategy and its config."""
        from fibsem.utils import format_value

        lines = [f"    Strategy: {self.name}"]
        for attr in self.config.required_attributes:
            if (
                attr in self.config._hidden_attributes
                or attr in self.config.advanced_attributes
            ):
                continue
            val = getattr(self.config, attr)
            spec = self.config.params.get(attr)
            unit = spec.unit if spec else None
            label = (spec and spec.label) or attr.replace("_", " ").title()
            if isinstance(val, float) and unit:
                val_str = format_value(val, unit=unit, precision=1)
            else:
                val_str = (
                    val.name
                    if hasattr(val, "name") and not isinstance(val, float)
                    else str(val)
                )
            lines.append(f"        {label}: {val_str}")
        return "\n".join(lines)

    @abstractmethod
    def run(
        self,
        microscope: FibsemMicroscope,
        stage: "FibsemMillingStage",
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[MillingStrategyResult]:
        """Mill ``stage``. May return what it has to say about the run, which
        the milling task records on the stage; None says nothing."""


def _strategy_named(name: str) -> Type[MillingStrategy[Any]]:
    """The strategy called *name*; Standard, with a warning, for a name none has.

    A name none has is a plugin strategy not installed here, or a misspelling. The
    stage still loads, as Standard; saved again, it is saved as Standard, so the
    warning names what is lost.
    """
    from fibsem.milling.strategy import DEFAULT_STRATEGY, get_strategies

    strategy = get_strategies().get(name)
    if strategy is None:
        logging.warning(
            f"No milling strategy is called {name!r} (a plugin not installed here?); "
            f"the stage uses {DEFAULT_STRATEGY.name}, and its {name} settings are "
            "not kept."
        )
        return DEFAULT_STRATEGY
    return strategy


def get_strategy(
    name: str = "Standard", config: Optional[Dict[str, Any]] = None
) -> MillingStrategy[Any]:
    """The strategy called *name*, made from a saved `{name, config}` document."""
    strategy = _strategy_named(name)
    return strategy.from_dict((config or {}) if strategy.name == name else {})


def get_strategy_config(
    name: str = "Standard", config: Optional[Dict[str, Any]] = None
) -> MillingStrategyConfig:
    """The config of the strategy called *name*, read from a saved `{name, config}`
    document."""
    return decode(MillingStrategyConfig, {**(config or {}), "name": name})


def strategy_class_for(
    config_class: Type[MillingStrategyConfig],
) -> Type[MillingStrategy[Any]]:
    """The strategy whose config class *config_class* is."""
    from fibsem.milling.strategy import get_strategies

    for strategy in get_strategies().values():
        if strategy.config_class is config_class:
            return strategy
    raise ValueError(
        f"No milling strategy is configured by {config_class.__name__}; "
        "register the strategy it belongs to."
    )


def _made_for(
    strategy: Type[MillingStrategy[Any]], config: TMillingStrategyConfig
) -> TMillingStrategyConfig:
    """*config*, marked as *strategy*'s. Not a field: not saved or compared, but
    copied with the config."""
    object.__setattr__(config, "_strategy_class", strategy)
    return config


def _installed(
    monitors: List[Optional[MillingMonitorConfig]],
) -> List[MillingMonitorConfig]:
    """A stage's saved monitors, without those not installed here (read as None)."""
    return [monitor for monitor in monitors if monitor is not None]


def _default_strategy_config() -> MillingStrategyConfig:
    from fibsem.milling.strategy import DEFAULT_STRATEGY

    return _made_for(DEFAULT_STRATEGY, DEFAULT_STRATEGY.config_class())


def _same_strategy(a: MillingStrategyConfig, b: MillingStrategyConfig) -> bool:
    return a.strategy_name == b.strategy_name and a == b


# A stage saved without its alignment block reads it as a saved alignment would
# without its keys: off (`MillingAlignment`), not a new stage's default.
@saved(required=("name", "milling", "pattern"), absent={"alignment": {}})
@dataclass
class FibsemMillingStage(Configurable):
    name: str = "Milling Stage"
    num: int = 0
    enabled: bool = True
    milling: FibsemMillingSettings = field(default_factory=FibsemMillingSettings)
    pattern: BasePattern = field(default_factory=DEFAULT_MILLING_PATTERN)
    patterns: Annotated[Optional[List[BasePattern]], NotSaved] = param(
        None, not_used="a stage mills its one pattern; nothing reads patterns"
    )
    # What the strategy is configured by; the strategy itself is made from it when the
    # stage mills (`get_strategy`). Given a strategy, the stage keeps its config.
    strategy: MillingStrategyConfig = field(default_factory=_default_strategy_config)
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    imaging: ImageSettings = field(
        default_factory=ImageSettings
    )  # settings for post-milling acquisition
    # What runs during each mill step of this stage (FIB-1481); each step makes
    # its monitors afresh from these.
    monitors: Annotated[List[MillingMonitorConfig], AfterValidator(_installed)] = field(
        default_factory=list
    )
    reference_image: Annotated[Optional[FibsemImage], NotSaved] = None
    # The whole-frame reference, for a stage that aligns full frame at a low
    # current (``alignment.full_frame_below``). Set with reference_image, never
    # saved with the stage.
    reference_image_full_frame: Annotated[Optional[FibsemImage], NotSaved] = None
    # Where this stage's run writes: its alignment reference, the drift
    # correction, anything a strategy saves. Set by FibsemMillingTask before the
    # strategy runs, like reference_image, and never saved with the stage -- it
    # belongs to one run, not to the protocol. ``imaging.path`` is an imaging
    # setting; read this to know where to save.
    output_dir: Annotated[Optional[str], NotSaved] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "strategy" and isinstance(value, MillingStrategy):
            # A strategy given whole: keep its config, and the strategy as the one
            # this stage mills with, so whatever holds it sees the run.
            object.__setattr__(self, "_strategy", value)
            value = _made_for(type(value), value.config)
        super().__setattr__(name, value)

    def get_strategy(self) -> MillingStrategy[Any]:
        """The strategy that mills this stage, made from its config.

        The same one until the config is replaced, so a viewer can connect to it
        before the stage mills and read how the run ended after.
        """
        strategy = self.__dict__.get("_strategy")
        if strategy is None or strategy.config is not self.strategy:
            strategy = self.strategy.strategy_class(config=self.strategy)
            object.__setattr__(self, "_strategy", strategy)
        return strategy

    def __post_init__(self):

        if self.imaging.resolution is None:
            self.imaging.resolution = [1536, 1024]  # default resolution for imaging
        if self.imaging.hfw is None:
            self.imaging.hfw = 150e-6
        if self.imaging.dwell_time is None:
            self.imaging.dwell_time = 1e-6
        if self.imaging.autocontrast is None:
            self.imaging.autocontrast = False
        if self.imaging.save is None:
            self.imaging.save = False

    def to_dict(self, short: bool = False) -> Dict[str, Any]:
        """As a protocol saves it. ``short`` leaves out what a milling task sets on
        the stage itself (`TASK_SETS_ON_STAGE`)."""
        ddict = encode(self)
        if short:
            ddict.pop("alignment")
            ddict.pop("imaging")
            ddict.pop("num")
            ddict["milling"].pop("acquire_images")
        return ddict

    @classmethod
    def from_dict(cls, data: dict):
        return decode(cls, data)

    @property
    def estimated_time(self) -> float:
        return estimate_stage_milling_time(self)

    def run(
        self,
        microscope: FibsemMicroscope,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Run the milling stage strategy on the given microscope."""
        self.get_strategy().run(
            microscope=microscope, stage=self, stop_event=stop_event
        )

    @property
    def summary(self) -> str:
        """Return a multi-line human-readable summary of the milling stage parameters."""
        return "\n".join(
            [
                self.name,
                self.milling.summary(),
                self.pattern.summary(),
                self.get_strategy().summary(),
            ]
        )

    @property
    def pretty_name(self) -> str:
        """Return a pretty name for the milling stage, including the milling current."""
        from fibsem.utils import format_value

        milling_current = self.milling.milling_current
        mc = format_value(val=milling_current, unit="A", precision=1)

        dp = ""
        if hasattr(self.pattern, "depth") and self.pattern.depth is not None:
            depth = self.pattern.depth
            dp = format_value(val=depth, unit="m", precision=1)
        txt = f"{self.name} - {self.pattern.name} ({dp}, {mc})"
        return txt

    def define_patterns(self) -> List[FibsemPatternSettings]:
        """Define the patterns for the milling stage."""
        shapes = []
        if self.patterns is not None:
            for p in self.patterns:
                shapes.extend(p.define())
        else:
            shapes = self.pattern.define()
        return shapes

    def __eq__(self, other: object) -> bool:
        # Two strategies can share a config class, so the config alone doesn't say
        # which strategy a stage mills with.
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _same_strategy(self.strategy, other.strategy) and all(
            getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
            if f.compare and f.name != "strategy"
        )

    def is_compatible_with(self, other: "FibsemMillingStage") -> bool:
        """Return True when both stages share milling settings and strategy. Compatible stages can be grouped together for milling."""
        if not isinstance(other, FibsemMillingStage):
            return False

        if self.milling != other.milling:
            return False

        return (
            _same_strategy(self.strategy, other.strategy)
            and self.monitors == other.monitors
        )


def get_milling_stages(
    key: str, protocol: Dict[str, List[Dict[str, Any]]]
) -> List[FibsemMillingStage]:
    """Get the milling stages for specific key from the protocol.
    Args:
        key: the key to get the milling stages for
        protocol: the protocol to get the milling stages from
    Returns:
        List[FibsemMillingStage]: the milling stages for the given key"""
    if key not in protocol:
        raise ValueError(
            f"Key {key} not found in protocol. Available keys: {list(protocol.keys())}"
        )

    stages = []
    for stage_config in protocol[key]:
        stage = FibsemMillingStage.from_dict(stage_config)
        stages.append(stage)
    return stages


def get_protocol_from_stages(
    stages: Union[FibsemMillingStage, List[FibsemMillingStage]],
) -> List[Dict[str, Any]]:
    """Convert a list of milling stages to a protocol dictionary.
    Args:
        stages: the list of milling stages to convert
    Returns:
        List[Dict[str, Any]]: the protocol dictionary"""
    if not isinstance(stages, list):
        stages = [stages]

    return deepcopy([stage.to_dict() for stage in stages])


# The connected backend's milling-time model, if it has its own. The planning stack
# (task ETAs, confirmation dialogs) reaches estimates through the
# FibsemMillingStage.estimated_time property, which has no microscope in scope, so
# the model is session-level: utils.setup_session installs the connected
# microscope's milling service's estimate_stage here on connect. It is kept after a
# disconnect (ETAs quoted then still describe the instrument this session plans
# for); connecting another backend replaces it. None means every stage uses the
# shared sputter-rate table.
MillingTimeEstimator = Callable[["FibsemMillingStage"], Optional[float]]
_milling_time_estimator: Optional[MillingTimeEstimator] = None


def set_milling_time_estimator(estimator: Optional[MillingTimeEstimator]) -> None:
    """Install the connected backend's milling-time estimator (None: the table)."""
    global _milling_time_estimator
    _milling_time_estimator = estimator


def estimate_stage_milling_time(stage: "FibsemMillingStage") -> float:
    """Estimated milling time for one stage.

    The installed backend estimator answers first; when there is none, or it
    returns None for this stage, the shared sputter-rate table is used.
    """
    if _milling_time_estimator is not None:
        estimate = _milling_time_estimator(stage)
        if estimate is not None:
            return estimate
    return estimate_milling_time(stage.pattern, stage.milling.milling_current)


def estimate_milling_time(pattern: BasePattern, milling_current: float) -> float:
    """Estimate the milling time for a given pattern and milling current.
    The time is calculated as the volume of the pattern divided by the sputter rate at the given current.
    The sputter rate is taken from the microscope application files.
    This is a rough estimate, as the actual milling time is calculated at milling time.

    Args:
        pattern (BasePattern): the milling pattern
        milling_current (float): the milling current in A

    Returns:
        float: the estimated milling time in seconds
    """
    if hasattr(pattern, "time") and pattern.time != 0:
        return pattern.time

    volume = pattern.volume  # m3

    if (
        hasattr(pattern, "cross_section")
        and pattern.cross_section is CrossSectionPattern.CleaningCrossSection
    ):
        volume *= 0.66  # ccs is approx 2/3 of the volume of a rectangle

    return estimate_volume_time(volume, milling_current)


def estimate_volume_time(volume: float, milling_current: float) -> float:
    """Seconds to mill *volume* (m3) at *milling_current* (A), from the sputter
    rate table."""
    # get the key that is closest to the milling current
    sp_keys = list(MILLING_SPUTTER_RATE.keys())
    sp_keys.sort(key=lambda x: abs(x - milling_current))

    # get the sputter rate for the closest key
    sputter_rate = MILLING_SPUTTER_RATE[sp_keys[0]]  # um3/s

    # scale the sputter rate based on the expected current
    sputter_rate = sputter_rate * (milling_current / sp_keys[0])

    time = (volume * 1e6**3) / sputter_rate
    return time * 0.75  # QUERY: accuracy of this estimate?
