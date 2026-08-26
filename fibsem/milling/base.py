import re
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.config import MILLING_SPUTTER_RATE
from fibsem.milling.patterning import DEFAULT_MILLING_PATTERN, get_pattern
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import (
    CrossSectionPattern,
    FibsemImage,
    FibsemMillingSettings,
    FibsemPatternSettings,
    ImageSettings,
    MillingAlignment,
    get_fields_with_metadata,
)

TMillingStrategyConfig = TypeVar(
    "TMillingStrategyConfig", bound="MillingStrategyConfig"
)
TMillingStrategy = TypeVar("TMillingStrategy", bound="MillingStrategy")


@dataclass
class MillingStrategyConfig(ABC):
    """Abstract base class for milling strategy configurations"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls: Type[TMillingStrategyConfig], d: Dict[str, Any]
    ) -> TMillingStrategyConfig:
        return cls(**d)

    @cached_property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @cached_property
    def advanced_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are marked as advanced in the metadata."""
        return tuple(f.name for f in fields(self) if f.metadata.get("advanced", False))

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return tuple(f.name for f in fields(self) if f.metadata.get("hidden", False))

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return dataclass fields with metadata, filling any missing keys with defaults."""
        return get_fields_with_metadata(self.__class__)


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
            meta = self.config.field_metadata.get(attr, {})
            unit = meta.get("unit", None)
            label = meta.get("label", attr.replace("_", " ").title())
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
        asynch: bool = False,
        parent_ui=None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        pass


def get_strategy(
    name: str = "Standard", config: Optional[Dict[str, Any]] = None
) -> MillingStrategy[Any]:
    from fibsem.milling.strategy import DEFAULT_STRATEGY, get_strategies

    if config is None:
        config = {}

    strategies = get_strategies()
    return strategies.get(name, DEFAULT_STRATEGY).from_dict(config)


@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    enabled: bool = True
    milling: FibsemMillingSettings = field(default_factory=FibsemMillingSettings)
    pattern: BasePattern = field(default_factory=DEFAULT_MILLING_PATTERN)
    patterns: Optional[List[BasePattern]] = None  # unused
    strategy: MillingStrategy[Any] = field(default_factory=get_strategy)
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    imaging: ImageSettings = field(
        default_factory=ImageSettings
    )  # settings for post-milling acquisition
    reference_image: Optional[FibsemImage] = None

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
        ddict = {
            "name": self.name,
            "num": self.num,
            "enabled": self.enabled,
            "milling": self.milling.to_dict(),
            "pattern": self.pattern.to_dict(),
            "strategy": self.strategy.to_dict(),
            "alignment": self.alignment.to_dict(),
            "imaging": self.imaging.to_dict(),
        }
        if short:
            ddict.pop("alignment")
            ddict.pop("imaging")
            ddict.pop("num")
            ddict["milling"].pop("acquire_images")
        return ddict

    @classmethod
    def from_dict(cls, data: dict):
        strategy_config = data.get("strategy", {})
        strategy_name = strategy_config.get("name", "Standard")
        pattern_name = data["pattern"]["name"]
        alignment = data.get("alignment", {})
        imaging: dict = data.get("imaging", {})
        if imaging == {} or imaging.get("path", None) is None:
            imaging["path"] = None  # set to None if not explicitly set
        return cls(
            name=data["name"],
            num=data.get("num", 0),
            enabled=data.get("enabled", True),
            milling=FibsemMillingSettings.from_dict(data["milling"]),
            pattern=get_pattern(pattern_name, data["pattern"]),
            strategy=get_strategy(strategy_name, config=strategy_config),
            alignment=MillingAlignment.from_dict(alignment),
            imaging=ImageSettings.from_dict(imaging),
        )

    @property
    def estimated_time(self) -> float:
        return estimate_stage_milling_time(self)

    def run(
        self, microscope: FibsemMicroscope, asynch: bool = False, parent_ui=None
    ) -> None:
        """Run the milling stage strategy on the given microscope."""
        self.strategy.run(
            microscope=microscope, stage=self, asynch=asynch, parent_ui=parent_ui
        )

    @property
    def summary(self) -> str:
        """Return a multi-line human-readable summary of the milling stage parameters."""
        return "\n".join(
            [
                self.name,
                self.milling.summary(),
                self.pattern.summary(),
                self.strategy.summary(),
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

    def is_compatible_with(self, other: "FibsemMillingStage") -> bool:
        """Return True when both stages share milling settings and strategy. Compatible stages can be grouped together for milling."""
        if not isinstance(other, FibsemMillingStage):
            return False

        if self.milling != other.milling:
            return False

        if type(self.strategy) is not type(other.strategy):
            return False

        return self.strategy.config == other.strategy.config


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


# Whether estimated_time uses the preset-driven dose model (TESCAN) instead of the
# legacy sputter-rate table. The planning stack (task ETAs, confirmation dialogs)
# reaches estimates through the FibsemMillingStage.estimated_time property, which
# has no microscope in scope — so the connected backend registers its model here.
# Only the TESCAN driver flips this (construction: True, disconnect: False); every
# other backend keeps the legacy table by never touching it. Keying off the stage's
# own fields instead would not work: protocols saved before the preset default
# became None carry a real-looking string ("30 keV; 2nA") on every backend.
_PRESET_DRIVEN_ESTIMATION = False


def set_preset_driven_estimation(enabled: bool) -> None:
    """Register whether the connected backend's milling is preset-driven (TESCAN)."""
    global _PRESET_DRIVEN_ESTIMATION
    _PRESET_DRIVEN_ESTIMATION = bool(enabled)


# A current token inside a free-form TESCAN preset name, e.g. "30 keV; 100 pA" or
# "30 keV; 2nA; my cool preset". Only prefixed units (pA/nA/uA/µA): a bare "A" in an
# arbitrary name (e.g. "slot 2A") is far more likely noise than a beam current.
_PRESET_CURRENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([pnuµ])A(?![a-zA-Z])")
_SI_CURRENT_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6}


def parse_current_from_preset(preset: Optional[str]) -> Optional[float]:
    """Parse the beam current (in A) out of a TESCAN preset name, or None.

    Preset names are free-form on the instrument, but conventionally embed the
    beam conditions ("30 keV; 100 pA"). The first current-looking token wins.
    """
    if not preset:
        return None
    match = _PRESET_CURRENT_RE.search(preset)
    if match is None:
        return None
    return float(match.group(1)) * _SI_CURRENT_PREFIX[match.group(2)]


def _estimate_preset_driven_milling_time(
    stage: "FibsemMillingStage",
) -> Optional[float]:
    """Dose-model estimate t = volume / (rate × current) for a preset-driven stage.

    The same inputs DrawBeam computes the real exposure from: the stage's own
    (per-material) etch rate and the current embedded in the preset name — the
    legacy sputter-rate table is a silicon calibration keyed on a current field
    the preset-driven backend ignores. Returns None (caller falls back to the
    legacy model) when the preset carries no parseable current or the rate is
    unusable.
    """
    pattern_time = getattr(stage.pattern, "time", 0)
    if pattern_time:
        return pattern_time

    current = parse_current_from_preset(stage.milling.preset)
    rate = stage.milling.rate  # m³/A/s
    if current is None or current <= 0 or not rate or rate <= 0:
        return None

    volume = stage.pattern.volume  # m³
    if (
        hasattr(stage.pattern, "cross_section")
        and stage.pattern.cross_section is CrossSectionPattern.CleaningCrossSection
    ):
        volume *= 0.66  # ccs is approx 2/3 of the volume of a rectangle
    return volume / (rate * current)


def estimate_stage_milling_time(stage: "FibsemMillingStage") -> float:
    """Estimated milling time for one stage, per the registered estimation model.

    Preset-driven backends (TESCAN, registered via set_preset_driven_estimation)
    get the dose model; everywhere else — and any preset the model cannot read —
    falls through to the legacy sputter-rate table, unchanged.
    """
    if _PRESET_DRIVEN_ESTIMATION:
        estimate = _estimate_preset_driven_milling_time(stage)
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

    # get the key that is closest to the milling current
    sp_keys = list(MILLING_SPUTTER_RATE.keys())
    sp_keys.sort(key=lambda x: abs(x - milling_current))

    # get the sputter rate for the closest key
    sputter_rate = MILLING_SPUTTER_RATE[sp_keys[0]]  # um3/s

    # scale the sputter rate based on the expected current
    sputter_rate = sputter_rate * (milling_current / sp_keys[0])
    volume = pattern.volume  # m3

    if (
        hasattr(pattern, "cross_section")
        and pattern.cross_section is CrossSectionPattern.CleaningCrossSection
    ):
        volume *= 0.66  # ccs is approx 2/3 of the volume of a rectangle

    time = (volume * 1e6**3) / sputter_rate
    return time * 0.75  # QUERY: accuracy of this estimate?


def estimate_total_milling_time(stages: List[FibsemMillingStage]) -> float:
    """Estimate the total milling time for a list of milling stages"""
    if not isinstance(stages, list):
        stages = [stages]
    return sum([estimate_stage_milling_time(stage) for stage in stages])
