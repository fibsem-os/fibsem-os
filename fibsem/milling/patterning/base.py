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

from fibsem.display import Display
from fibsem.milling.properties import distance_metadata
from fibsem.structures import (
    CrossSectionPattern,
    Point,
    RangeLimit,
    TFibsemPatternSettings,
    field_meta,
    get_fields_with_metadata,
)

TPattern = TypeVar("TPattern", bound="BasePattern")


@dataclass
class BasePattern(ABC, Generic[TFibsemPatternSettings]):
    name: ClassVar[str] = field(init=False)
    # Its type, Point, comes from the annotation: the form dispatches the
    # two-spinbox row on it. An offset from the image centre, so signed.
    point: Point = field(
        default_factory=Point,
        metadata=distance_metadata(
            "Point",
            "Point coordinates for the milling pattern.",
            limits=RangeLimit(-1e-3, 1e-3),
            advanced=True,
        ),
    )
    # Computed by define(), never edited: the form skips it on `hidden` rather
    # than on a hardcoded field name, so a plugin can hide its own fields too.
    shapes: Optional[List[TFibsemPatternSettings]] = field(
        default=None, init=False, metadata=field_meta(display=Display(hidden=True))
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
        return tuple(
            name for name, meta in self.field_metadata.items() if meta["advanced"]
        )

    @cached_property
    def _hidden_attributes(self) -> Tuple[str, ...]:
        """Return attributes that are hidden from the UI."""
        return tuple(
            name for name, meta in self.field_metadata.items() if meta["hidden"]
        )

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
