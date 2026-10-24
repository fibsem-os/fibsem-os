"""The base class every milling pattern, built-in or plugin, subclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from fibsem.codec import NotSaved, Tagged, decode, encode
from fibsem.milling.properties import distance_param
from fibsem.structures.params import (
    Configurable,
    Display,
    RangeLimit,
    flagged_fields,
    param,
)
from fibsem.structures.settings import Point, TFibsemPatternSettings

TPattern = TypeVar("TPattern", bound="BasePattern")


@dataclass
class BasePattern(Tagged, Configurable, ABC, Generic[TFibsemPatternSettings]):
    """A milling pattern, saved with its ``name`` beside its fields."""

    TYPE_KEY: ClassVar[str] = "name"
    name: ClassVar[str] = field(init=False)
    # Its type, Point, comes from the annotation: the form dispatches the
    # two-spinbox row on it. An offset from the image centre, so signed.
    point: Point = param(
        parameter=distance_param(
            "Point",
            "Point coordinates for the milling pattern.",
            limits=RangeLimit(-1e-3, 1e-3),
            advanced=True,
        ),
        default_factory=Point,
    )
    # Computed by define(), never edited: the form skips it on `hidden` rather
    # than on a hardcoded field name, so a plugin can hide its own fields too.
    shapes: Annotated[Optional[List[TFibsemPatternSettings]], NotSaved] = param(
        None, display=Display(hidden=True), init=False
    )

    @abstractmethod
    def define(self) -> List[TFibsemPatternSettings]:
        pass

    @classmethod
    def type_name(cls) -> str:
        return cls.name

    @classmethod
    def subclass_named(cls, name: str) -> Type["BasePattern"]:
        """The registered pattern called ``name``, in any case.

        Raises:
            ValueError: no pattern is called that.
        """
        from fibsem.milling.patterning import get_patterns

        patterns = get_patterns()
        folded = {key.casefold(): pattern for key, pattern in patterns.items()}
        pattern = patterns.get(name) or folded.get(str(name).casefold())
        if pattern is None:
            raise ValueError(
                f"No milling pattern named '{name}'. "
                f"Available patterns: {', '.join(patterns)}"
            )
        return pattern

    @classmethod
    def tagged_variants(cls) -> Dict[str, type]:
        from fibsem.milling.patterning import get_patterns

        return dict(get_patterns())

    def to_dict(self) -> Dict[str, Any]:
        return encode(self, BasePattern)

    @classmethod
    def from_dict(cls: Type[TPattern], ddict: Dict[str, Any]) -> TPattern:
        """A pattern of this class, or of the one ``ddict`` names when this is
        the base."""
        return decode(BasePattern if cls is BasePattern else cls, ddict)

    @cached_property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(
            f.name
            for f in fields(self)
            if f.name not in self._hidden_attributes and f not in fields(BasePattern)
        )

    @cached_property
    def advanced_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are marked as advanced in the metadata."""
        return flagged_fields(self.__class__, "advanced")

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return flagged_fields(self.__class__, "hidden")

    @property
    def volume(self) -> float:
        # calculate the total volume of the milling pattern (sum of all shapes)
        return sum([shape.volume for shape in self.define()])

    def summary(self) -> str:
        """Return a multi-line human-readable summary of key pattern parameters."""
        from fibsem.utils import format_value

        lines = [f"    Pattern: {self.name}"]
        for attr in self.required_attributes:
            if attr in self.advanced_attributes:
                continue
            val = getattr(self, attr)
            spec = self.params.get(attr)
            unit = spec.unit if spec else None
            label = (spec and spec.label) or attr.replace("_", " ").title()
            if isinstance(val, float) and unit:
                val_str = format_value(val, unit=unit, precision=1)
            elif hasattr(val, "name"):  # Enum
                val_str = val.name
            else:
                val_str = str(val)
            lines.append(f"        {label}: {val_str}")
        return "\n".join(lines)
