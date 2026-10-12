"""The base class every milling pattern, built-in or plugin, subclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from typing import (
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

import numpy as np

from fibsem.milling.properties import (
    DEFAULT_DISTANCE_METADATA,
)
from fibsem.structures import (
    CrossSectionPattern,
    Point,
    TFibsemPatternSettings,
    field_meta,
    get_fields_with_metadata,
)

TPattern = TypeVar("TPattern", bound="BasePattern")


@dataclass
class BasePattern(ABC, Generic[TFibsemPatternSettings]):
    name: ClassVar[str] = field(init=False)
    # `type` has to be Point, and has to win over the base: the form dispatches
    # the two-spinbox row on it. The dict literal this replaces declared
    # `"type": Point` *before* spreading DEFAULT_DISTANCE_METADATA, so the
    # spread's `float` silently won and the declaration was a lie -- which is
    # why the row had to be special-cased on the field being named "point".
    point: Point = field(
        default_factory=Point,
        metadata=field_meta(
            DEFAULT_DISTANCE_METADATA,
            label="Point",
            type=Point,
            minimum=-1000.0,
            maximum=1000.0,
            tooltip="Point coordinates for the milling pattern.",
            advanced=True,
        ),
    )
    # Computed by define(), never edited: the form skips it on `hidden` rather
    # than on a hardcoded field name, so a plugin can hide its own fields too.
    shapes: Optional[List[TFibsemPatternSettings]] = field(
        default=None, init=False, metadata=field_meta(hidden=True)
    )

    @abstractmethod
    def define(self) -> List[TFibsemPatternSettings]:
        pass

    def to_dict(self) -> Dict[str, Any]:
        ddict = asdict(self)
        # Handle any special cases
        if "cross_section" in ddict:
            ddict["cross_section"] = ddict["cross_section"].name
        if "vertices" in ddict:
            ddict["vertices"] = ddict["vertices"].tolist()
        ddict["name"] = self.name
        del ddict["shapes"]
        return ddict

    @classmethod
    def from_dict(cls: Type[TPattern], ddict: Dict[str, Any]) -> TPattern:
        kwargs = {}
        for f in fields(cls):
            if f.name in ddict:
                kwargs[f.name] = ddict[f.name]

        # Construct objects
        point = kwargs.pop("point", None)
        if point is not None:
            kwargs["point"] = Point.from_dict(point)

        cross_section = kwargs.pop("cross_section", None)
        if cross_section is not None:
            kwargs["cross_section"] = CrossSectionPattern[cross_section]

        vertices = kwargs.pop("vertices", None)
        if vertices is not None:
            kwargs["vertices"] = np.array(vertices)

        return cls(**kwargs)

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
        return tuple(f.name for f in fields(self) if f.metadata.get("advanced", False))

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return tuple(f.name for f in fields(self) if f.metadata.get("hidden", False))

    @property
    def volume(self) -> float:
        # calculate the total volume of the milling pattern (sum of all shapes)
        return sum([shape.volume for shape in self.define()])

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return dataclass fields with metadata, filling any missing keys with defaults."""
        return get_fields_with_metadata(self.__class__)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of key pattern parameters."""
        from fibsem.utils import format_value

        lines = [f"    Pattern: {self.name}"]
        for attr in self.required_attributes:
            if attr in self.advanced_attributes:
                continue
            val = getattr(self, attr)
            meta = self.field_metadata.get(attr, {})
            unit = meta.get("unit", None)
            label = meta.get("label", attr.replace("_", " ").title())
            if isinstance(val, float) and unit:
                val_str = format_value(val, unit=unit, precision=1)
            elif hasattr(val, "name"):  # Enum
                val_str = val.name
            else:
                val_str = str(val)
            lines.append(f"        {label}: {val_str}")
        return "\n".join(lines)
