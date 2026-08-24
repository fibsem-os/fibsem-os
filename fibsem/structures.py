# fibsem structures
from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import InitVar, asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import cv2
import numpy as np
import tifffile as tff
from numpy.typing import NDArray

import fibsem
from fibsem.config import (
    METADATA_VERSION,
    SUPPORTED_COORDINATE_SYSTEMS,
    UNVERSIONED_METADATA,
)
from fibsem.manufacturers import normalize_manufacturer
from fibsem.versioning import get_revision

TFibsemPatternSettings = TypeVar(
    "TFibsemPatternSettings", bound="FibsemPatternSettings"
)

DEFAULT_FIELD_METADATA: Dict[str, Any] = {
    "label": None,  # the display label for the field
    "type": None,  # the data type of the field
    "unit": None,  # the display unit for the field (after scaling)
    "tooltip": None,  # the tooltip/help text for the field
    "scale": None,  # scale factor for display (e.g., 1e6 for metres to microns)
    "dimensions": None,  # for complex dimensions, e.g. areas or volumes
    "default": None,  # default value for the field
    "minimum": None,  # minimum value for numeric fields
    "maximum": None,  # maximum value for numeric fields
    "step": None,  # step size for numeric fields
    "decimals": None,  # number of decimal places for numeric fields
    "items": None,  # for lists/enums, the possible items. items specified as 'dynamic' are fetched from the microscope via the 'microscope_parameter' key
    "hidden": False,  # whether the field is hidden from the UI
    "advanced": False,  # whether the field is considered advanced in the UI
    "manufacturer": None,  # manufacturer specific parameter (ThermoFisher, Tescan, etc.). Common parameters have None
    "microscope_parameter": None,  # the corresponding microscope parameter name, if applicable (via get/set)
    "format_fn": None,  # function to format the value for display
    "format_fn_kwargs": None,  # kwargs for the format function # NOTE: unused yet
    "filepath": None,  # render a string field as a file picker rather than a line edit
}

# Superseded spelling -> the key that replaced it. Nothing resolves these at
# runtime, so a field still declaring one renders without a tooltip or a unit
# suffix; this map exists only so the diagnostics below say something useful.
#
# The AutoLamella task configs used to spell two keys differently from the rest
# of the codebase, and the form that rendered them read only those spellings. All
# in-tree declarations were converted (FIB-384); this map exists so an
# out-of-tree config that has not been converted is told exactly what to change,
# rather than getting the generic "no form reads this" line.
RENAMED_METADATA_KEYS: Dict[str, str] = {
    "help": "tooltip",
    "units": "unit",
}


def field_meta(
    base: Optional[Mapping[str, Any]] = None,
    *,
    label: Optional[str] = None,
    type: Optional[Any] = None,
    unit: Optional[str] = None,
    tooltip: Optional[str] = None,
    scale: Optional[float] = None,
    dimensions: Optional[int] = None,
    default: Optional[Any] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,
    decimals: Optional[int] = None,
    items: Optional[Union[str, Sequence[Any]]] = None,
    hidden: Optional[bool] = None,
    advanced: Optional[bool] = None,
    manufacturer: Optional[str] = None,
    microscope_parameter: Optional[str] = None,
    format_fn: Optional[Callable[..., str]] = None,
    format_fn_kwargs: Optional[Dict[str, Any]] = None,
    filepath: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a form-metadata dict, checking the keys at import time.

        width: float = field(default=10e-6, metadata=field_meta(unit="m", scale=1e6))

    Every key is spelled out as a keyword argument, so a misspelling is a
    TypeError when the module is imported rather than a form that silently
    renders without its tooltip or its unit suffix. That failure mode is not
    hypothetical: two spellings of two keys coexisted across the codebase for a
    long time and neither side ever got an error (FIB-384).

    `base` extends a shared metadata dict, the way `dict(base, key=value)` does,
    which is how most patterns and strategies are declared:

        overtilt: float = field(metadata=field_meta(DEFAULT_ANGLE_METADATA, label="Overtilt"))

    Keywords override the base, matching the `{**BASE, "label": ...}` literal it
    replaces. The base's keys are checked too, so this cannot be used to launder
    a raw dict past the check it exists to perform.

    Passing the base by `**` instead is stricter, since Python rejects a keyword
    the base already supplies:

        field_meta(**DEFAULT_DISTANCE_METADATA, type=Point)   # TypeError

    Worth preferring where nothing needs overriding, because a dict literal
    silently resolves that collision instead -- `patterns2.BasePattern.point`
    declares `"type": Point` and renders as a float because the spread that
    follows wins.

    This returns the plain dict every form already reads, so raw-dict
    declarations keep working and files convert as they are touched. Arguments
    left unset are omitted rather than returned as None, which is what lets the
    base, a struct-level DEFAULT_METADATA, or `get_fields_with_metadata`'s own
    defaults supply them instead.
    """
    # `locals()` here is exactly the parameters, which is the point: listing them
    # again would let a newly added one be silently dropped -- the same class of
    # bug this function exists to prevent.
    declared = dict(locals())
    inherited = declared.pop("base") or {}
    for key in inherited:
        if key not in DEFAULT_FIELD_METADATA:
            renamed = RENAMED_METADATA_KEYS.get(key)
            hint = f", which was renamed to {renamed!r}" if renamed else ""
            raise TypeError(
                f"field_meta() base declares unknown metadata key {key!r}{hint}"
            )
    return {
        **inherited,
        **{key: value for key, value in declared.items() if value is not None},
    }


_warned_metadata_keys: set = set()


def _warn_unknown_metadata_keys(
    struct_cls: Type[Any], field_name: str, metadata: Mapping[str, Any]
) -> None:
    """Log once for a metadata key nothing will ever read.

    A mis-keyed field is otherwise silent: the form renders, the value is right,
    and the label or suffix is simply missing. Plugin authors have no way to
    discover the vocabulary, so a typo costs an afternoon. Warned once per
    (class, field, key) because form metadata is re-read on every rebuild.
    """
    for key in metadata:
        if key in DEFAULT_FIELD_METADATA:
            continue
        marker = (struct_cls.__qualname__, field_name, key)
        if marker in _warned_metadata_keys:
            continue
        _warned_metadata_keys.add(marker)
        replacement = RENAMED_METADATA_KEYS.get(key)
        if replacement is not None:
            logging.warning(
                f"{struct_cls.__name__}.{field_name} declares metadata key {key!r}, which was "
                f"renamed to {replacement!r} and is no longer read. The field will render "
                f"without it until the declaration is updated."
            )
        else:
            logging.warning(
                f"{struct_cls.__name__}.{field_name} declares metadata key {key!r}, which no "
                f"form reads. Known keys: {', '.join(sorted(DEFAULT_FIELD_METADATA))}."
            )


def get_fields_with_metadata(struct_cls: Type[Any]) -> Dict[str, Dict[str, Any]]:
    """Return dataclass fields with metadata, filling any missing keys with defaults."""
    # Prefer a struct-level DEFAULT_METADATA if provided, layering on top of the
    # module-wide defaults so any missing keys are still populated.
    default_metadata = {
        **DEFAULT_FIELD_METADATA,
        **getattr(struct_cls, "DEFAULT_METADATA", {}),
    }
    field_metadata: Dict[str, Dict[str, Any]] = {}
    for f in fields(struct_cls):
        declared = dict(f.metadata)
        _warn_unknown_metadata_keys(struct_cls, f.name, declared)
        merged_metadata = {**default_metadata, **declared}
        field_metadata[f.name] = merged_metadata
    return field_metadata


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @staticmethod
    def from_dict(d: dict) -> "Point":
        x = float(d["x"])
        y = float(d["y"])
        return Point(x, y)

    def to_list(self) -> list:
        return [self.x, self.y]

    @staticmethod
    def from_list(l: list) -> "Point":
        x = float(l[0])
        y = float(l[1])
        return Point(x, y)

    def __add__(self, other) -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def _to_metres(self, pixel_size: float) -> "Point":
        return Point(self.x * pixel_size, self.y * pixel_size)

    def _to_pixels(self, pixel_size: float) -> "Point":
        return Point(self.x / pixel_size, self.y / pixel_size)

    def distance(self, other: "Point") -> "Point":
        """Calculate the distance between two points. (other - self)"""
        return Point(x=(other.x - self.x), y=(other.y - self.y))

    def euclidean(self, other: "Point") -> float:
        """Calculate the euclidean distance between two points."""
        return float(np.linalg.norm(self.distance(other).to_list()))


# TODO: convert these to match autoscript...
class BeamType(Enum):
    """Enumerator Class for Beam Type
    1: Electron Beam
    2: Ion Beam

    """

    ELECTRON = 1  # Electron
    ION = 2  # Ion
    # CCD_CAM = 3
    # NavCam = 4 # see enumerations/ImagingDevice


class ImagingState(Enum):
    IDLE = 0
    RUNNING = 1
    STOPPING = 2
    PAUSED = 3
    ERROR = 4


class MillingState(Enum):
    IDLE = 0
    RUNNING = 1
    STOPPING = 2
    PAUSED = 3
    ERROR = 4


ACTIVE_MILLING_STATES = [
    MillingState.RUNNING,
    MillingState.STOPPING,
    MillingState.PAUSED,
]


class ManipulatorState(Enum):
    RETRACTED = 0
    INSERTED = 1
    MOVING = 2


class AutoFocusMode(Enum):
    """When to run autofocus during a tiled acquisition.

    One vocabulary for both tilers. There used to be two enums of this name --
    this one, and an identical set of concepts in `fibsem.fm.structures` -- which
    meant `AutoFocusMode.NONE is AutoFocusMode.NONE` was False across the two
    import paths, with no type error to catch it. `fibsem.fm.structures` now
    re-exports this one.

    Every spelling either enum was ever written in still resolves: the `EVERY_*`
    aliases below (this side persisted by name), the lowercase values (the FM side
    persisted by value), and the integers 0-3 that used to be this enum's values.
    """

    NONE = "none"
    ONCE = "once"
    EACH_ROW = "each_row"
    EACH_TILE = "each_tile"

    # Aliases, not new members: `AutoFocusMode.EVERY_ROW is AutoFocusMode.EACH_ROW`,
    # `AutoFocusMode["EVERY_ROW"]` resolves, and `list(AutoFocusMode)` still yields
    # exactly the four modes above -- which is what the mode combo boxes iterate.
    EVERY_ROW = "each_row"
    EVERY_TILE = "each_tile"

    @classmethod
    def _missing_(cls, value):
        """Resolve the older spellings, so nothing already written stops loading."""
        if isinstance(value, str):
            # Member names ("EVERY_ROW", "NONE"). Values are lowercase and names are
            # uppercase, so this cannot collide with a real value.
            return cls.__members__.get(value.upper())
        # The integers this enum used to be, in declaration order. bool is excluded
        # deliberately: True == 1 would otherwise silently resolve to ONCE.
        if isinstance(value, int) and not isinstance(value, bool):
            order = (cls.NONE, cls.ONCE, cls.EACH_ROW, cls.EACH_TILE)
            return order[value] if 0 <= value < len(order) else None
        return None


class TileOrderStrategy(Enum):
    TYPEWRITER = "typewriter"  # rows always left-to-right
    SERPENTINE = "serpentine"  # alternating: row 0 L→R, row 1 R→L, ...
    SPIRAL = "spiral"  # outward clockwise spiral from centre tile


@dataclass
class FibsemStagePosition:
    """Data class for storing stage position data.

    Attributes:
        x (float): The X position of the stage in meters.
        y (float): The Y position of the stage in meters.
        z (float): The Z position of the stage in meters.
        r (float): The Rotation of the stage in radians.
        t (float): The Tilt of the stage in radians.
        coordinate_system (str): The coordinate system used for the stage position.

    Methods:
        to_dict(): Convert the stage position object to a dictionary.
        from_dict(data: dict): Create a new stage position object from a dictionary.
        See fibsem.microscopes.autoscript for AutoScript conversion utilities (stage_position_to_autoscript, stage_position_from_autoscript).
    """

    name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    r: Optional[float] = None
    t: Optional[float] = None
    coordinate_system: Optional[str] = None

    def to_dict(self) -> dict:
        position_dict = {}

        position_dict["name"] = self.name if self.name is not None else None
        position_dict["x"] = float(self.x) if self.x is not None else None
        position_dict["y"] = float(self.y) if self.y is not None else None
        position_dict["z"] = float(self.z) if self.z is not None else None
        position_dict["r"] = float(self.r) if self.r is not None else None
        position_dict["t"] = float(self.t) if self.t is not None else None
        position_dict["coordinate_system"] = self.coordinate_system

        return position_dict

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemStagePosition":
        items = ["x", "y", "z", "r", "t"]

        for item in items:
            value = data[item]

            assert isinstance(value, float) or isinstance(value, int) or value is None

        return cls(
            name=data.get("name", None),
            x=data["x"],
            y=data["y"],
            z=data["z"],
            r=data["r"],
            t=data["t"],
            coordinate_system=data["coordinate_system"],
        )

    def __add__(self, other: "FibsemStagePosition") -> "FibsemStagePosition":
        return FibsemStagePosition(
            x=self.x + other.x if other.x is not None else self.x,
            y=self.y + other.y if other.y is not None else self.y,
            z=self.z + other.z if other.z is not None else self.z,
            r=self.r + other.r if other.r is not None else self.r,
            t=self.t + other.t if other.t is not None else self.t,
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: "FibsemStagePosition") -> "FibsemStagePosition":
        return FibsemStagePosition(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            r=self.r - other.r,
            t=self.t - other.t,
            coordinate_system=self.coordinate_system,
        )

    def _scale_repr(self, scale: float, precision: int = 2):
        return f"x:{self.x * scale:.{precision}f}, y:{self.y * scale:.{precision}f}, z:{self.z * scale:.{precision}f}"

    def is_close(self, pos2: "FibsemStagePosition", tol: float = 1e-6) -> bool:
        """Check if two positions are close to each other."""
        return (
            (abs(self.x - pos2.x) < tol)
            and (abs(self.y - pos2.y) < tol)
            and (abs(self.z - pos2.z) < tol)
            and (abs(self.t - pos2.t) < tol)
            and (abs(self.r - pos2.r) < tol)
        )

    def is_close2(
        self,
        pos2: "FibsemStagePosition",
        tol: float = 1e-6,
        axes: Optional[List[str]] = None,
    ) -> bool:
        """Check if two positions are close to each other."""
        VALID_AXES = ["x", "y", "z", "t", "r"]
        if axes is None:
            axes = VALID_AXES

        if any(axis not in VALID_AXES for axis in axes):
            raise ValueError(f"Invalid axes: {axes}. Must be one of: {VALID_AXES}")
        for axis in axes:
            pos1_val = getattr(self, axis)
            pos2_val = getattr(pos2, axis)

            if pos1_val is None or pos2_val is None:
                return False

            if abs(pos1_val - pos2_val) >= tol:
                return False

        return True

    def is_within_limits(
        self, limits: Dict[str, "RangeLimit"], axes: Optional[List[str]] = None
    ) -> bool:
        """Check if the position is within the specified limits.

        Args:
            limits: Dictionary mapping axis names to RangeLimit objects.
            axes: List of axes to check. If None, checks all axes present in limits.

        Returns:
            True if position is within limits for all specified axes, False otherwise.
        """
        if axes is None:
            axes = list(limits.keys())

        for axis in axes:
            if axis not in limits:
                continue

            pos_val = getattr(self, axis, None)
            if pos_val is None:
                continue

            limit = limits[axis]
            if pos_val < limit.min or pos_val > limit.max:
                return False

        return True

    @property
    def pretty_string(self) -> str:
        """Returns a pretty string representation of the stage position."""
        from fibsem import constants

        xstr = (
            f"X:{self.x * constants.METRE_TO_MILLIMETRE:.2f}"
            if self.x is not None
            else "X:None"
        )
        ystr = (
            f"Y:{self.y * constants.METRE_TO_MILLIMETRE:.2f}"
            if self.y is not None
            else "Y:None"
        )
        zstr = (
            f"Z:{self.z * constants.METRE_TO_MILLIMETRE:.2f}"
            if self.z is not None
            else "Z:None"
        )
        rstr = (
            f"R:{self.r * constants.RADIANS_TO_DEGREES:.1f}"
            if self.r is not None
            else "R:None"
        )
        tstr = (
            f"T:{self.t * constants.RADIANS_TO_DEGREES:.1f}"
            if self.t is not None
            else "T:None"
        )
        return f"{xstr}, {ystr}, {zstr}, {rstr}, {tstr}"

    @property
    def pretty_orientation(self) -> str:
        """Returns a pretty string representation of the stage orientation."""
        from fibsem import constants

        rstr = (
            f"R:{self.r * constants.RADIANS_TO_DEGREES:.1f}"
            if self.r is not None
            else "R:None"
        )
        tstr = (
            f"T:{self.t * constants.RADIANS_TO_DEGREES:.1f}"
            if self.t is not None
            else "T:None"
        )
        return f"{rstr}, {tstr}"

    @property
    def pretty(self) -> str:
        """Returns a pretty string representation of the stage position including units."""
        from fibsem import constants

        xstr = (
            f"X:{self.x * constants.METRE_TO_MILLIMETRE:.2f}mm"
            if self.x is not None
            else "X:None"
        )
        ystr = (
            f"Y:{self.y * constants.METRE_TO_MILLIMETRE:.2f}mm"
            if self.y is not None
            else "Y:None"
        )
        zstr = (
            f"Z:{self.z * constants.METRE_TO_MILLIMETRE:.2f}mm"
            if self.z is not None
            else "Z:None"
        )
        rstr = (
            f"R:{self.r * constants.RADIANS_TO_DEGREES:.1f}°"
            if self.r is not None
            else "R:None"
        )
        tstr = (
            f"T:{self.t * constants.RADIANS_TO_DEGREES:.1f}°"
            if self.t is not None
            else "T:None"
        )
        return f"{xstr}, {ystr}, {zstr}, {rstr}, {tstr}"

    def euclidean_distance(self, other: "FibsemStagePosition") -> float:
        """Calculate the euclidean distance between two stage positions."""
        dx = (self.x - other.x) if self.x is not None and other.x is not None else 0.0
        dy = (self.y - other.y) if self.y is not None and other.y is not None else 0.0
        dz = (self.z - other.z) if self.z is not None and other.z is not None else 0.0
        return float(np.linalg.norm([dx, dy, dz]))


@dataclass
class FibsemManipulatorPosition:
    """Data class for storing manipulator position data.

    Attributes:
        x (float): The X position of the manipulator in meters.
        y (float): The Y position of the manipulator in meters.
        z (float): The Z position of the manipulator in meters.
        r (float): The Rotation of the manipulator in radians.
        t (float): The Tilt of the manipulator in radians.
        coordinate_system (str): The coordinate system used for the manipulator position.

    Methods:
        to_dict(): Convert the manipulator position object to a dictionary.
        from_dict(data: dict): Create a new manipulator position object from a dictionary.
        See fibsem.microscopes.autoscript for AutoScript conversion utilities (manipulator_position_to_autoscript, manipulator_position_from_autoscript).
        to_tescan_position(): Convert the manipulator position to a format that is compatible with Tescan.
        from_tescan_position(): Create a new FibsemManipulatorPosition object from a Tescan-compatible manipulator position.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    t: float = 0.0
    coordinate_system: str = "RAW"

    def __post_init__(self):
        assert (
            isinstance(self.coordinate_system, str) or self.coordinate_system is None
        ), f"unsupported type {type(self.coordinate_system)} for coorindate system"
        assert (
            self.coordinate_system in SUPPORTED_COORDINATE_SYSTEMS
            or self.coordinate_system is None
        ), (
            f"coordinate system value {self.coordinate_system} is unsupported or invalid syntax. Must be RAW or SPECIMEN"
        )

    def to_dict(self) -> dict:
        position_dict = {}
        position_dict["x"] = self.x
        position_dict["y"] = self.y
        position_dict["z"] = self.z
        position_dict["r"] = self.r
        position_dict["t"] = self.t
        position_dict["coordinate_system"] = self.coordinate_system.upper()

        return position_dict

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemManipulatorPosition":
        items = ["x", "y", "z", "r", "t"]

        for item in items:
            value = data[item]

            assert isinstance(value, float) or isinstance(value, int) or value is None

        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            r=data["r"],
            t=data["t"],
            coordinate_system=data["coordinate_system"],
        )

    def __add__(
        self, other: "FibsemManipulatorPosition"
    ) -> "FibsemManipulatorPosition":
        return FibsemManipulatorPosition(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.r + other.r,
            self.t + other.t,
            self.coordinate_system,
        )


@dataclass
class FibsemRectangle:
    """Universal Rectangle class used for ReducedArea"""

    left: float = 0.0
    top: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self):
        assert isinstance(self.left, float) or isinstance(self.left, int), (
            f"type {type(self.left)} is unsupported for left, must be int or floar"
        )
        assert isinstance(self.top, float) or isinstance(self.top, int), (
            f"type {type(self.top)} is unsupported for top, must be int or floar"
        )
        assert isinstance(self.width, float) or isinstance(self.width, int), (
            f"type {type(self.width)} is unsupported for width, must be int or floar"
        )
        assert isinstance(self.height, float) or isinstance(self.height, int), (
            f"type {type(self.height)} is unsupported for height, must be int or floar"
        )

    @classmethod
    def from_dict(cls, settings: dict) -> "FibsemRectangle":
        if settings is None:
            return None
        points = ["left", "top", "width", "height"]

        for point in points:
            value = settings[point]

            assert isinstance(value, float) or isinstance(value, int) or value is None

        return FibsemRectangle(
            left=settings["left"],
            top=settings["top"],
            width=settings["width"],
            height=settings["height"],
        )

    def to_dict(self) -> dict:
        return {
            "left": float(self.left),
            "top": float(self.top),
            "width": float(self.width),
            "height": float(self.height),
        }

    @property
    def is_valid_reduced_area(self) -> bool:
        return _is_valid_reduced_area(self)

    @property
    def pretty_string(self) -> str:
        """Returns a pretty string representation of the rectangle."""
        return f"Left: {self.left:.2f}, Top: {self.top:.2f}, Width: {self.width:.2f}, Height: {self.height:.2f}"

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Convert FibsemRectangle (normalized coordinates 0-1) to image pixel coordinates.

        Args:
            image_shape: (height, width) tuple of the image shape

        Returns:
            Tuple of (x, y, width, height) in pixel coordinates where:
            - x, y are the top-left corner pixel coordinates
            - width, height are the dimensions in pixels
        """
        height, width = image_shape

        # Convert normalized coordinates to pixel coordinates
        x = int(self.left * width)
        y = int(self.top * height)
        pixel_width = int(self.width * width)
        pixel_height = int(self.height * height)

        return (x, y, pixel_width, pixel_height)


def _is_valid_reduced_area(reduced_area: FibsemRectangle) -> bool:
    """Check whether the reduced area is valid.
    Left and top must be between 0 and 1, and width and height must be between 0 and 1.
    Must not exceed the boundaries of the image 0 - 1
    """
    # if left or top is less than 0, or width or height is greater than 1, return False
    if (
        reduced_area.left < 0
        or reduced_area.top < 0
        or reduced_area.width > 1
        or reduced_area.height > 1
    ):
        return False
    if (
        reduced_area.left + reduced_area.width > 1
        or reduced_area.top + reduced_area.height > 1
    ):
        return False
    # no negative values
    if (
        reduced_area.left < 0
        or reduced_area.top < 0
        or reduced_area.width <= 0
        or reduced_area.height <= 0
    ):
        return False
    return True


@dataclass
class ImageSettings:
    """A data class representing the settings for an image acquisition.

    Attributes:
        resolution (list of int): The resolution of the acquired image in pixels, [x, y].
        dwell_time (float): The time spent per pixel during image acquisition, in seconds.
        hfw (float): The horizontal field width of the acquired image, in microns.
        autocontrast (bool): Whether or not to apply automatic contrast enhancement to the acquired image.
        beam_type (BeamType): The type of beam to use for image acquisition.
        save (bool): Whether or not to save the acquired image to disk.
        filename (str): The filename to use when saving the acquired image.
        path (Path): The path to the directory where the acquired image should be saved.
        reduced_area (FibsemRectangle): The rectangular region of interest within the acquired image, if any.

    Methods:
        from_dict(settings: dict) -> ImageSettings:
            Converts a dictionary of image settings to an ImageSettings object.
        to_dict() -> dict:
            Converts the ImageSettings object to a dictionary of image settings.
    """

    # There was an `autogamma` flag here, which applied gamma correction to the pixels
    # after acquisition. It was removed (FIB-505): unlike `autocontrast`, which
    # configures the detector *before* the image exists, gamma is a display correction
    # applied to the array afterwards -- and baking it into the stored data is
    # destructive and unrecoverable. The canvas's ContrastGammaControl does it at
    # display time instead, where it is adjustable and reversible.
    resolution: Tuple[int, int] = (1536, 1024)
    dwell_time: float = 1e-6
    hfw: float = 150e-6
    autocontrast: bool = False
    beam_type: BeamType = BeamType.ELECTRON
    save: bool = False
    filename: str = "default_image"
    path: Optional[Union[Path, str]] = None
    reduced_area: Optional[FibsemRectangle] = None
    line_integration: Optional[int] = None  # (int32) 2 - 255
    scan_interlacing: Optional[int] = None  # (int32) 2 - 8
    frame_integration: Optional[int] = None  # (int32) 2 - 512
    drift_correction: bool = False  # (bool) # requires frame_integration > 1

    def __post_init__(self):
        assert isinstance(self.resolution, (list, tuple)) or self.resolution is None, (
            f"resolution must be a list, currently is {type(self.resolution)}"
        )
        assert isinstance(self.dwell_time, float) or self.dwell_time is None, (
            f"dwell time must be of type float, currently is {type(self.dwell_time)}"
        )
        assert (
            isinstance(self.hfw, float) or isinstance(self.hfw, int) or self.hfw is None
        ), f"hfw must be int or float, currently is {type(self.hfw)}"
        assert isinstance(self.autocontrast, bool) or self.autocontrast is None, (
            f"autocontrast setting must be bool, currently is {type(self.autocontrast)}"
        )
        assert isinstance(self.beam_type, BeamType) or self.beam_type is None, (
            f"beam type must be a BeamType object, currently is {type(self.beam_type)}"
        )
        assert isinstance(self.save, bool) or self.save is None, (
            f"save option must be a bool, currently is {type(self.save)}"
        )
        assert isinstance(self.filename, str) or self.filename is None, (
            f"filename must b str, currently is {type(self.filename)}"
        )
        assert isinstance(self.path, (Path, str)) or self.path is None, (
            f"save path must be Path or str, currently is {type(self.path)}"
        )
        assert (
            isinstance(self.reduced_area, FibsemRectangle) or self.reduced_area is None
        ), (
            f"reduced area must be a fibsemRectangle object, currently is {type(self.reduced_area)}"
        )

    @property
    def scan_time(self) -> float:
        """Seconds of beam-on time for one frame at these settings.

        Dwell time per pixel, over every pixel, times the passes each one gets. Line and
        frame integration both re-scan: `line_integration` sweeps each line N times,
        `frame_integration` acquires and averages N frames. `scan_interlacing` changes
        the order lines are visited in, not how many are visited, so it does not appear.

        Scan time, not run time: it excludes flyback, autocontrast, saving, and -- for a
        tileset -- the stage, which dominates. `OverviewAcquisitionSettings.scan_time`
        says more about why that last one is left out rather than guessed at.
        """
        width, height = self.resolution
        passes = (self.line_integration or 1) * (self.frame_integration or 1)
        return self.dwell_time * width * height * passes

    @staticmethod
    def from_dict(settings: dict) -> "ImageSettings":
        if "reduced_area" in settings and settings["reduced_area"] is not None:
            reduced_area = FibsemRectangle.from_dict(settings["reduced_area"])
        else:
            reduced_area = None

        # default to Electron if not specified
        beam_name = settings.get("beam_type", "Electron")
        if beam_name is None:
            beam_name = "Electron"

        image_settings = ImageSettings(
            resolution=settings.get("resolution", (1536, 1024)),
            dwell_time=settings.get("dwell_time", 1.0e-6),
            hfw=settings.get("hfw", 150e-6),
            autocontrast=settings.get("autocontrast", False),
            beam_type=BeamType[beam_name.upper()],
            save=settings.get("save", False),
            path=settings.get("path", os.getcwd()),
            filename=settings.get("filename", "default_image"),
            reduced_area=reduced_area,
            line_integration=settings.get("line_integration", None),
            scan_interlacing=settings.get("scan_interlacing", None),
            frame_integration=settings.get("frame_integration", None),
            drift_correction=settings.get("drift_correction", False),
        )

        return image_settings

    def to_dict(self) -> dict:
        settings_dict = {
            "beam_type": self.beam_type.name if self.beam_type is not None else None,
            "resolution": list(self.resolution)
            if self.resolution is not None
            else None,
            "dwell_time": self.dwell_time if self.dwell_time is not None else None,
            "hfw": self.hfw if self.hfw is not None else None,
            "autocontrast": self.autocontrast
            if self.autocontrast is not None
            else None,
            "save": self.save if self.save is not None else None,
            "path": str(self.path) if self.path is not None else None,
            "filename": self.filename if self.filename is not None else None,
            "reduced_area": {
                "left": self.reduced_area.left,
                "top": self.reduced_area.top,
                "width": self.reduced_area.width,
                "height": self.reduced_area.height,
            }
            if self.reduced_area is not None
            else None,
            "line_integration": self.line_integration,
            "scan_interlacing": self.scan_interlacing,
            "frame_integration": self.frame_integration,
            "drift_correction": self.drift_correction,
        }

        return settings_dict

    @property
    def field_of_view(self) -> float:
        """Calculate the field of view based on the horizontal field width (hfw)."""
        return self.hfw

    @field_of_view.setter
    def field_of_view(self, value: float):
        """Set the horizontal field width (hfw) based on the desired field of view."""
        self.hfw = value

    @property
    def estimated_time(self) -> float:
        """Estimated acquisition time for a single image in seconds."""
        pixel_time = self.resolution[0] * self.resolution[1] * self.dwell_time
        return pixel_time * (self.frame_integration or 1) * (self.line_integration or 1)

    @staticmethod
    def fromFibsemImage(image: "FibsemImage") -> "ImageSettings":
        """Returns the image settings for a FibsemImage object.

        Args:
            image (FibsemImage): The FibsemImage object to get the image settings from.

        Returns:
            ImageSettings: The image settings for the given FibsemImage object.
        """
        from copy import deepcopy

        from fibsem import utils

        image_settings = deepcopy(image.metadata.image_settings)
        if image_settings.filename is None:
            image_settings.filename = utils.current_timestamp()
        image_settings.save = True

        return image_settings


@dataclass
class FocusStackSettings:
    """Settings for focus-stack acquisition in tiled overview acquisition.

    Attributes:
        enabled: Whether to use focus stacking for each tile.
        n_steps: Number of vertical strips to divide each tile into.
        auto_focus: Whether to run autofocus for each strip.
    """

    enabled: bool = False
    n_steps: int = 3
    auto_focus: bool = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_steps": self.n_steps,
            "auto_focus": self.auto_focus,
        }

    @staticmethod
    def from_dict(d: dict) -> "FocusStackSettings":
        return FocusStackSettings(
            enabled=d.get("enabled", False),
            n_steps=d.get("n_steps", 3),
            auto_focus=d.get("auto_focus", True),
        )


@dataclass
class AutoFocusSettings:
    """Settings for autofocus in tiled overview acquisition.

    Attributes:
        mode: When to apply autofocus (NONE, ONCE, EACH_ROW, EACH_TILE).
              beam_type and reduced_area are taken from image_settings at acquisition time.
    """

    mode: AutoFocusMode = AutoFocusMode.NONE

    def to_dict(self) -> dict:
        # By value, matching how the fluorescence side has always written this mode.
        # Files that stored the old member name ("EVERY_ROW") still load.
        return {"mode": self.mode.value}

    @staticmethod
    def from_dict(d: dict) -> "AutoFocusSettings":
        return AutoFocusSettings(
            mode=AutoFocusMode(d.get("mode", AutoFocusMode.NONE.value))
        )


@dataclass
class OverviewAcquisitionSettings:
    """Settings for a tiled overview acquisition.

    Attributes:
        image_settings: Per-tile image settings (hfw = tile FOV, beam_type, resolution, etc.)
        nrows: Number of tile rows in the grid.
        ncols: Number of tile columns in the grid.
        overlap: Fractional overlap between adjacent tiles (0.0 = no overlap).
            Honoured by `TiledAcquisitionRunner` and by the shared geometry core,
            which step by `fov * (1 - overlap)`; the docstring said otherwise long
            after it stopped being true.
        tile_mask: Optional per-tile enable mask, `tile_mask[row][col]`. None acquires
            every tile. Disabled tiles are skipped but keep their place: the mosaic is
            still the full grid size and acquired tiles land at the same canvas
            coordinates they would have in a dense overview.
    """

    image_settings: ImageSettings = field(default_factory=ImageSettings)
    nrows: int = 3
    ncols: int = 3
    overlap: float = 0.1
    focus_stack_settings: FocusStackSettings = field(default_factory=FocusStackSettings)
    autofocus_settings: AutoFocusSettings = field(default_factory=AutoFocusSettings)
    tile_order: TileOrderStrategy = TileOrderStrategy.TYPEWRITER
    tile_mask: Optional[List[List[bool]]] = None

    @property
    def n_enabled_tiles(self) -> int:
        """How many tiles a run would actually acquire.

        The number every progress readout has to count towards. `nrows * ncols` is the
        grid's *shape*, which is still what the mosaic is sized from -- but a masked run
        that reported it would stop at "6 / 9" and read as a failure.
        """
        if self.tile_mask is None:
            return self.nrows * self.ncols
        return sum(1 for row in self.tile_mask for enabled in row if enabled)

    @property
    def scan_time(self) -> float:
        """Seconds of beam-on time for the whole overview.

        Counts the tiles a run would actually acquire, not the grid's shape: a masked
        overview scans only what is enabled, and reporting the full grid would overstate
        a typical sparse selection roughly threefold.

        Deliberately **scan time only**, and not an estimate of how long a run will take.
        The two differ by a lot: a 3 x 3 tileset of 1024 x 1024 pixels at 1 us scans for
        about 9 seconds and takes minutes, because the stage has to move and settle
        eight times in between. That missing term is not something to guess at --
        `fibsem/fm/timing.py` assumes 5 seconds per stage move, which nobody has
        measured, and a total built on it would be mostly that assumption quoted as
        though it were arithmetic. This is arithmetic, so it is reported under its own
        name, and answers what the number is consulted for: whether the dwell time and
        resolution just chosen make a run of seconds or of hours.
        """
        return self.image_settings.scan_time * self.n_enabled_tiles

    @property
    def total_fov_x(self) -> float:
        """Total horizontal FOV in meters, accounting for overlap."""
        hfw = self.image_settings.hfw
        dx = hfw * (1 - self.overlap)
        return (self.ncols - 1) * dx + hfw

    @property
    def total_fov_y(self) -> float:
        """Total vertical FOV in meters, accounting for overlap and tile aspect ratio."""
        w, h = self.image_settings.resolution
        hfw = self.image_settings.hfw
        tile_fov_y = hfw * (h / w) if w > 0 else hfw
        dy = tile_fov_y * (1 - self.overlap)
        return (self.nrows - 1) * dy + tile_fov_y

    @property
    def total_fov(self) -> float:
        """Total horizontal FOV in meters (alias for total_fov_x)."""
        return self.total_fov_x

    @staticmethod
    def from_dict(d: dict) -> "OverviewAcquisitionSettings":
        # backward compat: old configs had a bare use_focus_stack bool
        if "focus_stack_settings" not in d and "use_focus_stack" in d:
            fss = FocusStackSettings(enabled=d["use_focus_stack"])
        else:
            fss = FocusStackSettings.from_dict(d.get("focus_stack_settings", {}))
        mask = d.get("tile_mask")
        return OverviewAcquisitionSettings(
            image_settings=ImageSettings.from_dict(d.get("image_settings", {})),
            nrows=d.get("nrows", 3),
            ncols=d.get("ncols", 3),
            overlap=d.get("overlap", 0.1),
            focus_stack_settings=fss,
            autofocus_settings=AutoFocusSettings.from_dict(
                d.get("autofocus_settings", {})
            ),
            tile_order=TileOrderStrategy(
                d.get("tile_order", TileOrderStrategy.TYPEWRITER.value)
            ),
            tile_mask=None
            if mask is None
            else [[bool(v) for v in row] for row in mask],
        )

    def to_dict(self) -> dict:
        return {
            "image_settings": self.image_settings.to_dict(),
            "nrows": self.nrows,
            "ncols": self.ncols,
            "overlap": self.overlap,
            "focus_stack_settings": self.focus_stack_settings.to_dict(),
            "autofocus_settings": self.autofocus_settings.to_dict(),
            "tile_order": self.tile_order.value,
            # plain bools: np.bool_ does not survive yaml.safe_dump, and a mask arriving
            # from a numpy grid is exactly how one gets here.
            "tile_mask": None
            if self.tile_mask is None
            else [[bool(v) for v in row] for row in self.tile_mask],
        }


@dataclass
class BeamSettings:
    """
    Dataclass representing the beam settings for an imaging session.

    Attributes:
        beam_type (BeamType): The type of beam to use for imaging.
        working_distance (float): The working distance for the microscope, in meters.
        beam_current (float): The beam current for the microscope, in amps.
        hfw (float): The horizontal field width for the microscope, in meters.
        resolution (list): The desired resolution for the image.
        dwell_time (float): The dwell time for the microscope.
        stigmation (Point): The point for stigmation correction.
        shift (Point): The point for shift correction.

    Methods:
        to_dict(): Returns a dictionary representation of the object.
        from_dict(state_dict: dict) -> BeamSettings: Returns a new BeamSettings object created from a dictionary.

    """

    beam_type: BeamType
    working_distance: Optional[float] = None
    beam_current: Optional[float] = None
    voltage: Optional[float] = None
    hfw: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    dwell_time: Optional[float] = None
    stigmation: Optional[Point] = None
    shift: Optional[Point] = None
    scan_rotation: Optional[float] = None
    preset: Optional[str] = None

    def __post_init__(self):
        assert (
            self.beam_type in [BeamType.ELECTRON, BeamType.ION]
            or self.beam_type is None
        ), f"beam_type must be instance of BeamType, currently {type(self.beam_type)}"
        assert (
            isinstance(self.working_distance, (float, int))
            or self.working_distance is None
        ), (
            f"Working distance must be float or int, currently is {type(self.working_distance)}"
        )
        assert (
            isinstance(self.beam_current, (float, int)) or self.beam_current is None
        ), f"beam current must be float or int, currently is {type(self.beam_current)}"
        assert isinstance(self.voltage, (float, int)) or self.voltage is None, (
            f"voltage must be float or int, currently is {type(self.voltage)}"
        )
        assert isinstance(self.hfw, (float, int)) or self.hfw is None, (
            f"horizontal field width (HFW) must be float or int, currently is {type(self.hfw)}"
        )
        assert isinstance(self.resolution, (list, tuple)) or self.resolution is None, (
            f"resolution must be a list or tuple, currently is {type(self.resolution)}"
        )
        assert isinstance(self.dwell_time, (float, int)) or self.dwell_time is None, (
            f"dwell_time must be float or int, currently is {type(self.dwell_time)}"
        )
        assert isinstance(self.stigmation, Point) or self.stigmation is None, (
            f"stigmation must be a Point instance, currently is {type(self.stigmation)}"
        )
        assert isinstance(self.shift, Point) or self.shift is None, (
            f"shift must be a Point instance, currently is {type(self.shift)}"
        )
        assert (
            isinstance(self.scan_rotation, (float, int)) or self.scan_rotation is None
        ), (
            f"scan rotation must be float or int, currently is {type(self.scan_rotation)}"
        )
        assert isinstance(self.preset, str) or self.preset is None, (
            f"preset must be str, currently is {type(self.preset)}"
        )

    def to_dict(self) -> dict:
        state_dict = {
            "beam_type": self.beam_type.name,
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "voltage": self.voltage,
            "hfw": self.hfw,
            "resolution": list(self.resolution)
            if self.resolution is not None
            else None,
            "dwell_time": self.dwell_time,
            "stigmation": self.stigmation.to_dict()
            if self.stigmation is not None
            else None,
            "shift": self.shift.to_dict() if self.shift is not None else None,
            "scan_rotation": self.scan_rotation,
            "preset": self.preset,
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "BeamSettings":
        if "stigmation" in state_dict and state_dict["stigmation"] is not None:
            stigmation = Point.from_dict(state_dict["stigmation"])
        else:
            stigmation = Point()
        if "shift" in state_dict and state_dict["shift"] is not None:
            shift = Point.from_dict(state_dict["shift"])
        else:
            shift = Point()

        wd = state_dict.get(
            "working_distance", state_dict.get("eucentric_height", None)
        )
        current = state_dict.get("beam_current", state_dict.get("current", None))

        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"].upper()],
            working_distance=wd,
            beam_current=current,
            voltage=state_dict["voltage"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            stigmation=stigmation,
            shift=shift,
            scan_rotation=state_dict.get("scan_rotation", 0.0),
            preset=state_dict.get("preset", None),
        )

        return beam_settings


@dataclass
class FibsemDetectorSettings:
    type: str = "Unknown"
    mode: str = "Unknown"
    brightness: float = 0.5
    contrast: float = 0.5

    def __post_init__(self):
        assert isinstance(self.type, str) or self.type is None, (
            f"type must be input as str, currently is {type(self.type)}"
        )
        assert isinstance(self.mode, str) or self.mode is None, (
            f"mode must be input as str, currently is {type(self.mode)}"
        )
        assert isinstance(self.brightness, (float, int)) or self.brightness is None, (
            f"brightness must be int or float value, currently is {type(self.brightness)}"
        )
        assert isinstance(self.contrast, (float, int)) or self.contrast is None, (
            f"contrast must be int or float value, currently is {type(self.contrast)}"
        )

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "type": self.type,
            "mode": self.mode,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemDetectorSettings":
        """Converts from a dictionary."""
        return FibsemDetectorSettings(
            type=settings.get("type", "Unknown"),
            mode=settings.get("mode", "Unknown"),
            brightness=settings.get("brightness", 0.0),
            contrast=settings.get("contrast", 0.0),
        )


@dataclass
class MicroscopeState:
    """Data Class representing the state of a microscope with various parameters.

    Attributes:

        timestamp (float): A float representing the timestamp at which the state of the microscope was recorded. Defaults to the timestamp of the current datetime.
        stage_position (FibsemStagePosition): An instance of FibsemStagePosition representing the current absolute position of the stage. Defaults to an empty instance of FibsemStagePosition.
        electron_beam (BeamSettings): An instance of BeamSettings representing the electron beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ELECTRON.
        ion_beam (BeamSettings): An instance of BeamSettings representing the ion beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ION.

    Methods:

        to_dict(self) -> dict: Converts the current state of the Microscope to a dictionary and returns it.
        from_dict(state_dict: dict) -> "MicroscopeState": Returns a new instance of MicroscopeState with attributes created from the passed dictionary.
    """

    timestamp: float = datetime.timestamp(datetime.now())
    stage_position: Optional[FibsemStagePosition] = field(
        default_factory=FibsemStagePosition
    )
    electron_beam: Optional[BeamSettings] = field(
        default_factory=lambda: BeamSettings(beam_type=BeamType.ELECTRON)
    )
    ion_beam: Optional[BeamSettings] = field(
        default_factory=lambda: BeamSettings(beam_type=BeamType.ION)
    )
    electron_detector: Optional[FibsemDetectorSettings] = field(
        default_factory=FibsemDetectorSettings
    )
    ion_detector: Optional[FibsemDetectorSettings] = field(
        default_factory=FibsemDetectorSettings
    )
    objective_position: Optional[float] = None  # in meters

    def __post_init__(self):
        assert (
            isinstance(self.stage_position, FibsemStagePosition)
            or self.stage_position is None
        ), (
            f"absolute position must be of type FibsemStagePosition, currently is {type(self.stage_position)}"
        )
        assert (
            isinstance(self.electron_beam, BeamSettings) or self.electron_beam is None
        ), (
            f"electron_beam must be of type BeamSettings, currently is {type(self.electron_beam)}"
        )
        assert isinstance(self.ion_beam, BeamSettings) or self.ion_beam is None, (
            f"ion_beam must be of type BeamSettings, currently us {type(self.ion_beam)}"
        )
        assert (
            isinstance(self.electron_detector, FibsemDetectorSettings)
            or self.electron_detector is None
        ), (
            f"electron_detector must be of type FibsemDetectorSettings, currently is {type(self.electron_detector)}"
        )
        assert (
            isinstance(self.ion_detector, FibsemDetectorSettings)
            or self.ion_detector is None
        ), (
            f"ion_detector must be of type FibsemDetectorSettings, currently is {type(self.ion_detector)}"
        )

    def to_dict(self) -> dict:
        state_dict = {
            "timestamp": self.timestamp,
            "stage_position": self.stage_position.to_dict()
            if self.stage_position is not None
            else None,
            "electron_beam": self.electron_beam.to_dict()
            if self.electron_beam is not None
            else None,
            "ion_beam": self.ion_beam.to_dict() if self.ion_beam is not None else None,
            "electron_detector": self.electron_detector.to_dict()
            if self.electron_detector is not None
            else None,
            "ion_detector": self.ion_detector.to_dict()
            if self.ion_detector is not None
            else None,
            "objective_position": self.objective_position,
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "MicroscopeState":

        # beam, and detector settings are now optional
        electron_beam, electron_detector = None, None
        ion_beam, ion_detector = None, None

        if state_dict.get("electron_beam", None) is not None:
            electron_beam = BeamSettings.from_dict(state_dict["electron_beam"])
        if state_dict.get("ion_beam", None) is not None:
            ion_beam = BeamSettings.from_dict(state_dict["ion_beam"])
        if state_dict.get("electron_detector", None) is not None:
            electron_detector = FibsemDetectorSettings.from_dict(
                state_dict["electron_detector"]
            )
        if state_dict.get("ion_detector", None) is not None:
            ion_detector = FibsemDetectorSettings.from_dict(state_dict["ion_detector"])

        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            stage_position=FibsemStagePosition.from_dict(state_dict["stage_position"]),
            electron_beam=electron_beam,
            ion_beam=ion_beam,
            electron_detector=electron_detector,
            ion_detector=ion_detector,
            objective_position=state_dict.get("objective_position", None),
        )

        return microscope_state


########### Base Pattern Settings
@dataclass
class FibsemPatternSettings(ABC):
    def to_dict(self) -> Dict[str, Any]:
        ddict = asdict(self)
        # Handle any special cases
        if "cross_section" in ddict:
            ddict["cross_section"] = ddict["cross_section"].name
        return ddict

    @classmethod
    def from_dict(
        cls: Type[TFibsemPatternSettings], data: Dict[str, Any]
    ) -> TFibsemPatternSettings:
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]

        # Construct objects
        cross_section = kwargs.pop("cross_section", None)
        if cross_section is not None:
            kwargs["cross_section"] = CrossSectionPattern[cross_section]

        return cls(**kwargs)

    @property
    @abstractmethod
    def volume(self) -> float:
        pass


class CrossSectionPattern(Enum):
    Rectangle = auto()
    RegularCrossSection = auto()
    CleaningCrossSection = auto()


@dataclass
class FibsemRectangleSettings(FibsemPatternSettings):
    width: float
    height: float
    depth: float
    centre_x: float
    centre_y: float
    rotation: float = 0
    cleaning_cross_section: bool = False
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    passes: int = 0
    time: float = 0.0
    is_exclusion: bool = False

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth


@dataclass
class FibsemLineSettings(FibsemPatternSettings):
    start_x: float
    end_x: float
    start_y: float
    end_y: float
    depth: float

    @property
    def volume(self) -> float:
        return (
            np.sqrt((self.end_x - self.start_x) ** 2 + (self.end_y - self.start_y) ** 2)
            * self.depth
        )


@dataclass
class FibsemCircleSettings(FibsemPatternSettings):
    radius: float
    depth: float
    centre_x: float
    centre_y: float
    thickness: float = 0
    start_angle: float = 0.0
    end_angle: float = 360.0
    rotation: float = 0.0  # annulus -> thickness !=0
    is_exclusion: bool = False

    @property
    def volume(self) -> float:
        return np.pi * self.radius**2 * self.depth


@dataclass
class FibsemBitmapSettings(FibsemPatternSettings):
    width: float
    height: float
    depth: float
    centre_x: float
    centre_y: float
    rotation: float = 0
    scan_direction: str = "TopToBottom"
    passes: int = 0
    time: float = 0.0
    is_exclusion: bool = False
    flip_y: bool = False
    path: InitVar[Optional[Union[str, os.PathLike]]] = None
    array: InitVar[Optional[NDArray[Any]]] = None
    bitmap: Optional[NDArray[Any]] = field(init=False)
    interpolate: Optional[Literal["nearest", "bicubic", "bilinear"]] = None

    def __post_init__(
        self, path: Optional[Union[str, os.PathLike]], array: Optional[NDArray[Any]]
    ) -> None:
        if array is None:
            if path is None:
                # Fallback on empty array
                array = None
            else:
                from PIL import Image

                array = np.asarray(Image.open(path), dtype=np.uint8)

        if array is not None:
            if array.dtype == np.uint8:
                # Convert bitmap image to bitmap points - simpler if it's handled here and consistent after
                from fibsem.milling.patterning.utils import bitmap_image_to_points

                array = bitmap_image_to_points(array)
            else:
                array = array.copy()

        self.bitmap = array

    @property
    def volume(self) -> float:
        if self.bitmap is None:
            return 0
        return self.width * self.height * self.depth * self.bitmap[:, :, 0].mean()


@dataclass
class FibsemPolygonSettings(FibsemPatternSettings):
    vertices: np.ndarray[float]  # n[x, y]
    depth: float
    is_exclusion: bool = False

    def to_dict(self) -> dict:
        return {
            "vertices": self.vertices.tolist(),
            "depth": self.depth,
            "is_exclusion": self.is_exclusion,
        }

    @staticmethod
    def from_dict(data: dict) -> "FibsemPolygonSettings":
        return FibsemPolygonSettings(
            vertices=np.asarray(data["vertices"], dtype=float),
            depth=data["depth"],
            is_exclusion=data.get("is_exclusion", False),
        )

    @property
    def volume(self) -> float:
        # NOTE: this is a VERY rough estimate, assuming a convex polygon
        # calculate polygon area as rectangle area
        if len(self.vertices) == 0:
            return 100e-6
        xmin = min(v[0] for v in self.vertices)
        xmax = max(v[0] for v in self.vertices)
        ymin = min(v[1] for v in self.vertices)
        ymax = max(v[1] for v in self.vertices)
        width = xmax - xmin
        height = ymax - ymin
        # volume is area * depth
        return width * height * self.depth


@dataclass
class FibsemMillingSettings:
    """
    This class is used to store and retrieve settings for FIBSEM milling.

    Attributes:
    milling_current (float): The current used in the FIBSEM milling process. Default value is 20.0e-12 A.
    spot_size (float): The size of the beam spot used in the FIBSEM milling process. Default value is 5.0e-8 m.
    rate (float): The milling rate of the FIBSEM process. Default value is 3.0e-3 m^3/A/s.
    dwell_time (float): The dwell time of the beam at each point during the FIBSEM milling process. Default value is 1.0e-6 s.
    hfw (float): The high voltage field width used in the FIBSEM milling process. Default value is 150e-6 m.

    Methods:
    to_dict(): Converts the object attributes into a dictionary.
    from_dict(settings: dict) -> "FibsemMillingSettings": Creates a FibsemMillingSettings object from a dictionary of settings.
    """

    milling_current: float = field(
        default=20.0e-12,
        metadata={
            "unit": "A",
            "label": "Milling Current",
            "type": float,
            "items": "dynamic",
            "microscope_parameter": "current",
            "tooltip": "The current used for milling. Higher currents mill faster but with less precision and more damage.",
            "manufacturer": "ThermoFisher",
        },
    )
    milling_voltage: float = field(
        default=30e3,
        metadata={
            "unit": "V",
            "label": "Milling Voltage",
            "type": float,
            "items": "dynamic",
            "microscope_parameter": "voltage",
            "advanced": True,
            "tooltip": "The voltage used for milling. Higher voltages provide higher energy ions for milling.",
            "manufacturer": "ThermoFisher",
        },
    )
    application_file: str = field(
        default="Si",
        metadata={
            "label": "Application File",
            "type": str,
            "items": "dynamic",
            "microscope_parameter": "application_file",
            "advanced": True,
            "tooltip": "The application file used for milling. Note: this can be changed at runtime depending on the pattern and other parameters.",
            "manufacturer": "ThermoFisher",
        },
    )
    patterning_mode: str = field(
        default="Serial",
        metadata={
            "label": "Patterning Mode",
            "type": str,
            "advanced": True,
            "items": ["Serial", "Parallel"],
            "advanced": True,
            "tooltip": "The patterning mode used for milling. 'Serial' mills the entire pattern in one pass, 'Parallel' mills multiple pattern simultaneously.",
        },
    )
    hfw: float = field(
        default=150e-6,
        metadata={
            "label": "Field of View",
            "type": float,
            "unit": "m",
            "scale": 1e6,
            "default": 150.0,
            "minimum": 20.0,
            "maximum": 950.0,
            "step": 10.0,
            "decimals": 2,
            "microscope_parameter": "hfw",
            "hidden": True,
            "tooltip": "The horizontal field width used for milling. Patterns must fit within this field of view.",
        },
    )
    preset: str = field(
        default="30 keV; 2nA",
        metadata={
            "label": "Preset",
            "type": str,
            "items": "dynamic",
            "microscope_parameter": "preset",
            "tooltip": "The preset used for milling. Presets define the beam settings for different milling conditions.",
            "manufacturer": "Tescan",
        },
    )
    # 1 µm is the value the cryo lamella milling was validated at on hardware
    # (2026-07-22). Protocols do not set spot_size, so this default is what every
    # milling stage actually uses -- the `milling:` block in the system config is
    # not consulted for it (only milling_current is read from there).
    spot_size: float = field(
        default=1.0e-6,
        metadata={
            "label": "Spot Size",
            "type": float,
            "unit": "m",
            "scale": 1e6,
            # bounds are in the DISPLAY unit (µm). Real spot sizes are
            # tens of nm -- the TESCAN default is 50 nm -- so a 1.0 µm
            # minimum silently clamped every real value up to 1 µm.
            "minimum": 0.001,
            "maximum": 100.0,
            "step": 0.01,
            "decimals": 3,
            "tooltip": "The spot size for the ion beam during milling.",
            "manufacturer": "Tescan",
        },
    )
    rate: float = field(
        default=1.3e-8,
        metadata={
            "label": "Rate",
            "type": float,
            # unit must stay a BASE unit: the display suffix is built by
            # prefixing it from `scale` (1e3 -> "m"), giving "mm³/A/s".
            # mm³/A/s and TESCAN's own µm³/nA/s are numerically identical.
            "unit": "m³/A/s",
            "scale": 1e3,
            "dimensions": 3,
            "tooltip": "Ion etching rate — how much material one amp removes per "
            "second. Equivalently µm³/nA/s, which is how TESCAN quotes "
            "it. Default is the cryo lamella value; silicon is 0.3.",
            "manufacturer": "Tescan",
        },
    )
    dwell_time: float = field(
        default=1.0e-6,
        metadata={
            "label": "Dwell Time",
            "type": float,
            "unit": "s",
            "scale": 1e6,
            "tooltip": "The dwell time for the ion beam during milling (µs).",
            "manufacturer": "Tescan",
        },
    )
    spacing: float = field(
        default=0.005,
        metadata={
            "label": "Spacing",
            "type": float,
            "minimum": 0.0,
            "maximum": 100.0,
            "step": 0.001,
            "decimals": 4,
            "tooltip": "Exposition mesh spacing — how finely the pattern is filled "
            "with exposure points. Dimensionless; the TESCAN default is "
            "1.0 and smaller values mill more finely and take longer.",
            "manufacturer": "Tescan",
        },
    )
    milling_channel: BeamType = field(
        default=BeamType.ION,
        metadata={
            "label": "Milling Channel",
            "type": BeamType,
            "items": [BeamType.ION, BeamType.ELECTRON],
            "tooltip": "The beam channel used for milling.",
            "hidden": True,
        },
    )
    acquire_images: bool = field(
        default=False,
        metadata={
            "label": "Acquire Images",
            "type": bool,
            "tooltip": "Whether to acquire images after milling.",
            "hidden": True,
        },
    )

    # Parameter mapping for different manufacturers
    _SUPPORTED_MANUFACTURERS = {"ThermoFisher", "Tescan"}

    def __post_init__(self):
        assert isinstance(self.milling_current, (float, int)), (
            f"invalid type for milling_current, must be int or float, currently {type(self.milling_current)}"
        )
        assert isinstance(self.spot_size, (float, int)), (
            f"invalid type for spot_size, must be int or float, currently {type(self.spot_size)}"
        )
        assert isinstance(self.rate, (float, int)), (
            f"invalid type for rate, must be int or float, currently {type(self.rate)}"
        )
        assert isinstance(self.dwell_time, (float, int)), (
            f"invalid type for dwell_time, must be int or float, currently {type(self.dwell_time)}"
        )
        assert isinstance(self.hfw, (float, int)), (
            f"invalid type for hfw, must be int or float, currently {type(self.hfw)}"
        )
        assert isinstance(self.patterning_mode, str), (
            f"invalid type for value for patterning_mode, must be str, currently {type(self.patterning_mode)}"
        )
        assert isinstance(self.application_file, (str)), (
            f"invalid type for value for application_file, must be str, currently {type(self.application_file)}"
        )
        assert isinstance(self.spacing, (float, int)), (
            f"invalid type for value for spacing, must be int or float, currently {type(self.spacing)}"
        )
        # assert isinstance(self.preset,(str)), f"invalid type for value for preset, must be str, currently {type(self.preset)}"

    def to_dict(self) -> dict:
        settings_dict = {
            "milling_current": self.milling_current,
            "spot_size": self.spot_size,
            "rate": self.rate,
            "dwell_time": self.dwell_time,
            "hfw": self.hfw,
            "patterning_mode": self.patterning_mode,
            "application_file": self.application_file,
            "preset": self.preset,
            "spacing": self.spacing,
            "milling_voltage": self.milling_voltage,
            "milling_channel": self.milling_channel.name,
            "acquire_images": self.acquire_images,
        }

        return settings_dict

    @staticmethod
    def from_dict(settings: dict) -> "FibsemMillingSettings":
        # fall back to the dataclass field defaults rather than repeating them here,
        # so there is a single source of truth for every default
        defaults = FibsemMillingSettings()
        milling_settings = FibsemMillingSettings(
            milling_current=settings.get("milling_current", defaults.milling_current),
            spot_size=settings.get("spot_size", defaults.spot_size),
            rate=settings.get("rate", defaults.rate),
            dwell_time=settings.get("dwell_time", defaults.dwell_time),
            hfw=float(settings.get("hfw", defaults.hfw)),
            patterning_mode=settings.get("patterning_mode", defaults.patterning_mode),
            application_file=settings.get(
                "application_file", defaults.application_file
            ),
            preset=settings.get("preset", defaults.preset),
            spacing=settings.get("spacing", defaults.spacing),
            milling_voltage=settings.get("milling_voltage", defaults.milling_voltage),
            milling_channel=BeamType[
                settings.get("milling_channel", defaults.milling_channel.name)
            ],
            acquire_images=settings.get("acquire_images", defaults.acquire_images),
        )

        return milling_settings

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return dataclass fields with metadata, filling any missing keys with defaults."""
        return get_fields_with_metadata(self.__class__)

    @property
    def advanced_attributes(self) -> Set[str]:
        """Return a set of advanced attribute names."""
        fields_with_metadata = self.field_metadata
        return {
            field_name
            for field_name, metadata in fields_with_metadata.items()
            if metadata.get("advanced", False)
        }

    def get_parameters_for_manufacturer(self, manufacturer: str) -> tuple[str, ...]:
        """Get all parameter names for a specific manufacturer (any known spelling)."""
        manufacturer = normalize_manufacturer(manufacturer)
        if manufacturer not in self._SUPPORTED_MANUFACTURERS:
            raise ValueError(
                f"Manufacturer must be one of: {', '.join(self._SUPPORTED_MANUFACTURERS)}"
            )

        # use the field metadata to determine manufacturer-specific parameters
        fields_with_metadata = self.field_metadata
        required_params = []

        for field_name, metadata in fields_with_metadata.items():
            param_manufacturer = metadata.get("manufacturer", None)
            if param_manufacturer == manufacturer or param_manufacturer is None:
                required_params.append(field_name)
        return tuple(sorted(set(required_params)))

    def get_parameters(self, manufacturer: str) -> Dict[str, Any]:
        """Get parameter values for a specific manufacturer."""
        required_params = self.get_parameters_for_manufacturer(manufacturer)
        return {param: getattr(self, param) for param in required_params}

    def summary(self) -> str:
        from fibsem.utils import format_value

        mc = format_value(self.milling_current, unit="A", precision=1)
        mv = format_value(self.milling_voltage, unit="V", precision=1)
        lines = [
            "    Milling:",
            f"        Current: {mc}",
            f"        Voltage: {mv}",
            f"        Patterning Mode: {self.patterning_mode}",
        ]
        return "\n".join(lines)


@dataclass
class StageSystemSettings:
    rotation_reference: float
    rotation_180: float
    shuttle_pre_tilt: float
    manipulator_height_limit: float
    enabled: bool = True
    rotation: bool = True
    tilt: bool = True
    milling_angle: float = 15

    def to_dict(self):
        return {
            "rotation_reference": self.rotation_reference,
            "rotation_180": self.rotation_180,
            "shuttle_pre_tilt": self.shuttle_pre_tilt,
            "manipulator_height_limit": self.manipulator_height_limit,
            "enabled": self.enabled,
            "rotation": self.rotation,
            "tilt": self.tilt,
            "milling_angle": self.milling_angle,
        }

    @staticmethod
    def from_dict(settings: dict):
        return StageSystemSettings(
            rotation_reference=settings["rotation_reference"],
            rotation_180=settings["rotation_180"],
            shuttle_pre_tilt=settings["shuttle_pre_tilt"],
            manipulator_height_limit=settings["manipulator_height_limit"],
            enabled=settings.get("enabled", True),
            rotation=settings.get("rotation", True),
            tilt=settings.get("tilt", True),
            milling_angle=settings.get("milling_angle", 15.0),
        )


@dataclass
class BeamSystemSettings:
    beam_type: BeamType
    enabled: bool
    beam: BeamSettings
    detector: FibsemDetectorSettings
    eucentric_height: float
    column_tilt: float
    plasma: bool = False
    plasma_gas: Optional[str] = None

    def to_dict(self):
        ddict = {
            "beam_type": self.beam_type.value,
            "enabled": self.enabled,
            "eucentric_height": self.eucentric_height,
            "column_tilt": self.column_tilt,
            "plasma": self.plasma,
            "plasma_gas": self.plasma_gas,
        }
        ddict.update(self.beam.to_dict())
        ddict.update(self.detector.to_dict())

        # rename keys to match config
        ddict["detector_mode"] = ddict.pop("mode")
        ddict["detector_type"] = ddict.pop("type")
        ddict["detector_brightness"] = ddict.pop("brightness")
        ddict["detector_contrast"] = ddict.pop("contrast")
        ddict["current"] = ddict.pop("beam_current")

        return ddict

    @staticmethod
    def from_dict(settings: dict) -> "BeamSystemSettings":
        return BeamSystemSettings(
            beam_type=BeamType[settings["beam_type"]],
            enabled=settings["enabled"],
            beam=BeamSettings.from_dict(settings),
            detector=FibsemDetectorSettings.from_dict(settings),
            eucentric_height=settings["eucentric_height"],
            column_tilt=settings["column_tilt"],
            plasma=settings.get("plasma", False),
            plasma_gas=settings.get("plasma_gas", None),
        )


@dataclass
class ManipulatorSystemSettings:
    enabled: bool
    rotation: bool
    tilt: bool

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "rotation": self.rotation,
            "tilt": self.tilt,
        }

    @staticmethod
    def from_dict(settings: dict):
        return ManipulatorSystemSettings(
            enabled=settings["enabled"],
            rotation=settings["rotation"],
            tilt=settings["tilt"],
        )


@dataclass
class GISSystemSettings:
    enabled: bool
    multichem: bool
    sputter_coater: bool
    inserted: bool = False

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "multichem": self.multichem,
            "sputter_coater": self.sputter_coater,
        }

    @staticmethod
    def from_dict(settings: dict):
        return GISSystemSettings(
            enabled=settings["enabled"],
            multichem=settings["multichem"],
            sputter_coater=settings["sputter_coater"],
        )


@dataclass
class SystemInfo:
    """Which instrument, and what software is running on it. Provenance.

    Owns both facts, and is the only place either belongs (FIB-445 D1).
    ``serial_number`` is the instrument identity -- the key any per-instrument
    aggregation joins on, and the only field that distinguishes two of the same
    model in one facility.

    The version fields were duplicated on ``FibsemExperimentRef`` until v5, which
    removed them from there (FIB-448). What software is running is a property of
    the running system, not of an experiment.

    An experiment spanning a software upgrade will therefore disagree with its own
    images -- the experiment record captured v0.5.1 at creation, day-2 images say
    v0.5.2 here. That is two true facts, not a duplication bug: it says the run
    spanned an upgrade, which is worth knowing.

    There is deliberately no ``application_version``. It existed up to v4 and was
    never once populated: an application shipped inside fibsem has no version of its
    own, and ``fibsem_revision`` already pins the exact commit doing the work. An
    out-of-tree application wanting to stamp its own version needs a public
    registration API first -- the field can come back alongside one.
    """

    name: str
    ip_address: str
    manufacturer: str
    model: str
    serial_number: str
    hardware_version: str
    software_version: str
    fibsem_version: str = fibsem.__version__
    application: Optional[str] = None
    # The commit actually running, when installed from a source checkout. None
    # for a wheel install. default_factory, not a plain default, so the lookup
    # happens on first use rather than at import of this module.
    fibsem_revision: Optional[str] = field(default_factory=get_revision)

    def to_dict(self):
        return {
            "name": self.name,
            "ip_address": self.ip_address,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "hardware_version": self.hardware_version,
            "software_version": self.software_version,
            "fibsem_version": self.fibsem_version,
            "application": self.application,
            "fibsem_revision": self.fibsem_revision,
        }

    @staticmethod
    def from_dict(settings: dict):
        return SystemInfo(
            name=settings.get("name", "Unknown"),
            ip_address=settings.get("ip_address", "Unknown"),
            # normalise on read: configs and old experiments carry "Thermo"/"TESCAN"
            # etc.; everything downstream compares against the canonical spellings
            manufacturer=normalize_manufacturer(
                settings.get("manufacturer", "Unknown")
            ),
            model=settings.get("model", "Unknown"),
            serial_number=settings.get("serial_number", "Unknown"),
            hardware_version=settings.get("hardware_version", "Unknown"),
            software_version=settings.get("software_version", "Unknown"),
            fibsem_version=settings.get("fibsem_version", fibsem.__version__),
            application=settings.get("application", None),
            # `application_version` is not read: files up to v4 carry it, always null.
            # `or`, not a .get() default: settings.get(k, get_revision()) would
            # evaluate the lookup eagerly on every call.
            fibsem_revision=settings.get("fibsem_revision") or get_revision(),
        )


@dataclass
class SystemSettings:
    stage: StageSystemSettings
    electron: BeamSystemSettings
    ion: BeamSystemSettings
    manipulator: ManipulatorSystemSettings
    gis: GISSystemSettings
    info: SystemInfo
    sim: Dict[str, Union[str, bool]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "stage": self.stage.to_dict(),
            "electron": self.electron.to_dict(),
            "ion": self.ion.to_dict(),
            "manipulator": self.manipulator.to_dict(),
            "gis": self.gis.to_dict(),
            "info": self.info.to_dict(),
            "sim": self.sim,
        }

    @staticmethod
    def from_dict(settings: dict):

        # TODO: remove this once the settings are updated
        settings["electron"]["beam_type"] = BeamType.ELECTRON.name
        settings["ion"]["beam_type"] = BeamType.ION.name

        return SystemSettings(
            stage=StageSystemSettings.from_dict(settings["stage"]),
            electron=BeamSystemSettings.from_dict(settings["electron"]),
            ion=BeamSystemSettings.from_dict(settings["ion"]),
            manipulator=ManipulatorSystemSettings.from_dict(settings["manipulator"]),
            gis=GISSystemSettings.from_dict(settings["gis"]),
            info=SystemInfo.from_dict(settings["info"]),
            sim=settings.get("sim", {}),
        )


class CameraImageTransform(Enum):
    """Image transformations for aligning fluorescence images with SEM/FIB coordinate systems.

    Flips only. Any fixed rotation between the sensor and the stage belongs to the
    mount, not to user preference, and is corrected inside the driver
    (``FluorescenceMicroscope.mount_transform``) before this is applied.

    Restricting the set to flips makes it the Klein four-group: every member is its
    own inverse and composition is order-independent, so mapping a displacement
    between the displayed image and the stage is two sign flips with no axis swap
    and no inverse to get backwards. Flips also preserve the array shape, so the
    image and its geometry metadata always describe the same frame.

    Lives here rather than in ``fibsem.fm.structures`` only because
    ``FibsemHardwareGeometry`` needs it and core cannot import from the FM package --
    ``fm.structures`` imports from this module, so the reverse is a cycle. Re-exported
    there, which is where every consumer of it still is.
    """

    NONE = None
    FLIP_X = "flip-x"
    FLIP_Y = "flip-y"
    FLIP_XY = "flip-xy"

    def apply_to_delta(self, dx: float, dy: float) -> Tuple[float, float]:
        """Map a displacement between the raw and displayed frames.

        Every member is its own inverse, so this maps in both directions: use it to
        take a delta measured in the displayed image back to the underlying frame,
        and vice versa.
        """
        flip_x = self in (CameraImageTransform.FLIP_X, CameraImageTransform.FLIP_XY)
        flip_y = self in (CameraImageTransform.FLIP_Y, CameraImageTransform.FLIP_XY)
        return (-dx if flip_x else dx, -dy if flip_y else dy)


# Transforms that stored configurations may still hold. A half turn is the same
# element as flipping both axes, so it maps across without losing the setting; the
# quarter turns describe a mount, which the driver now corrects, and have no
# equivalent here.
_LEGACY_IMAGE_TRANSFORMS = {"rotate-180": CameraImageTransform.FLIP_XY}


def _parse_image_transform(value: Any) -> CameraImageTransform:
    """Read a stored transform, tolerating values that are no longer members.

    Rotations were removed once mount rotation moved into the driver. A half turn is
    migrated to the equivalent flip so the setting survives; anything else falls back
    to no transform with a warning rather than raising.
    """
    if value is None:
        return CameraImageTransform.NONE
    try:
        return CameraImageTransform(value)
    except ValueError:
        pass

    migrated = _LEGACY_IMAGE_TRANSFORMS.get(value)
    if migrated is not None:
        logging.info(
            f"Camera image transform {value!r} is now {migrated.value!r}; migrated."
        )
        return migrated

    logging.warning(
        f"Unsupported camera image transform {value!r}; falling back to none. "
        "Rotations are now applied as a fixed mount correction inside the driver."
    )
    return CameraImageTransform.NONE


@dataclass
class FibsemHardwareGeometry:
    """The instrument's fixed physical arrangement: column tilts, stage reference angles.

    Recorded on an image so a stage position can be projected onto it without a live
    microscope -- and, more to the point, without *assuming* the live microscope still
    matches. Reprojecting a saved image against the current pose is silently wrong the
    moment the stage has moved or the instrument has been reconfigured.

    Distinct from its two neighbours. ``MicroscopeState`` is dynamic observation, what
    the instrument was *doing*; ``SystemSettings`` is the config file, the whole of it.
    This is what the instrument *is*, in the terms a projection actually needs.

    **One record for both modalities.** The beam and fluorescence paths were given
    separate structures that named the same six terms identically, free to drift with
    nothing to catch it. ``camera_tilt`` and ``transform`` describe the fluorescence
    camera and stay at their defaults on a beam image; that is two dormant fields on a
    SEM picture, accepted deliberately so there is exactly one definition of how this
    instrument is arranged rather than two that merely agree today (FIB-481).

    Angles are in degrees, matching ``SystemSettings``. Both reprojection paths convert
    at the point of use.

    ``is_compustage`` is stored rather than derived. The live value is ground truth --
    ThermoFisher reads it from ``connection.specimen.compustage.is_installed`` -- but
    an image had no field for it, so the beam path inferred it by matching the model
    name against "Arctis", with a TODO against the line. A capability is not a name.

    Note this is deliberately *not* grouped this way in ``SystemSettings`` itself: that
    maps onto ``microscope-configuration.yaml``, a user-facing file, and regrouping it
    would mean migrating every site's config for a cosmetic gain. The scatter stays
    there; ``from_system_settings`` is the one place that knows about it.
    """

    column_tilt: float = 0.0  # electron column
    fib_column_tilt: float = 52.0  # ion column; fixes the compustage FIB pose
    shuttle_pre_tilt: float = 0.0
    rotation_reference: float = 0.0
    rotation_180: float = 180.0
    is_compustage: bool = False
    # Fluorescence only; left at these defaults for a beam image.
    camera_tilt: float = 0.0  # viewing axis, from the electron column
    transform: CameraImageTransform = CameraImageTransform.NONE

    @classmethod
    def from_system_settings(
        cls, system: SystemSettings, is_compustage: bool = False
    ) -> "FibsemHardwareGeometry":
        """Gather the geometry terms out of a full system configuration.

        ``is_compustage`` is a parameter because ``SystemSettings`` does not carry it:
        it is a property of the installed hardware, which only the connected
        microscope knows. Callers holding one should pass ``microscope.stage_is_compustage``.
        """
        return cls(
            column_tilt=system.electron.column_tilt,
            fib_column_tilt=system.ion.column_tilt,
            shuttle_pre_tilt=system.stage.shuttle_pre_tilt,
            rotation_reference=system.stage.rotation_reference,
            rotation_180=system.stage.rotation_180,
            is_compustage=is_compustage,
        )

    def to_dict(self) -> dict:
        return {
            "column_tilt": self.column_tilt,
            "fib_column_tilt": self.fib_column_tilt,
            "shuttle_pre_tilt": self.shuttle_pre_tilt,
            "rotation_reference": self.rotation_reference,
            "rotation_180": self.rotation_180,
            "is_compustage": self.is_compustage,
            "camera_tilt": self.camera_tilt,
            "transform": self.transform.value,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "FibsemHardwareGeometry":
        # `.get` throughout, with the field defaults repeated rather than referenced:
        # a file written before this record existed has none of these keys, and the
        # point of the record is that such a file still loads.
        return cls(
            column_tilt=ddict.get("column_tilt", 0.0),
            fib_column_tilt=ddict.get("fib_column_tilt", 52.0),
            shuttle_pre_tilt=ddict.get("shuttle_pre_tilt", 0.0),
            rotation_reference=ddict.get("rotation_reference", 0.0),
            rotation_180=ddict.get("rotation_180", 180.0),
            is_compustage=ddict.get("is_compustage", False),
            camera_tilt=ddict.get("camera_tilt", 0.0),
            # Not a bare CameraImageTransform(...): stored configurations may hold a
            # rotation that is no longer a member, which the parser migrates.
            transform=_parse_image_transform(ddict.get("transform")),
        )


@dataclass
class MicroscopeSettings:
    """
    A data class representing the settings for a microscope system.

    Attributes:
        system (SystemSettings): An instance of the `SystemSettings` class that holds the system settings.
        image (ImageSettings): An instance of the `ImageSettings` class that holds the image settings.
        milling (FibsemMillingSettings): An instance of the `FibsemMillingSettings` class that holds the fibsem milling settings..
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to None.

    Methods:
        to_dict(): Returns a dictionary representation of the `MicroscopeSettings` object.
        from_dict(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """

    system: SystemSettings
    image: ImageSettings
    milling: FibsemMillingSettings
    protocol: Optional[dict] = None
    fm: Optional["FluorescenceConfiguration"] = None

    def to_dict(self) -> dict:
        settings_dict = {
            "imaging": self.image.to_dict(),
            "protocol": self.protocol,
            "milling": self.milling.to_dict(),
        }
        settings_dict.update(self.system.to_dict())

        return settings_dict

    @staticmethod
    def from_dict(
        settings: dict, protocol: Optional[dict] = None
    ) -> "MicroscopeSettings":

        if protocol is None:
            protocol = settings.get("protocol", {"name": "demo"})

        fm_config = None
        fm_config_path = settings.get("fm", {}).get("config", None)
        if fm_config_path is not None and isinstance(fm_config_path, str):
            from fibsem.fm.structures import FluorescenceConfiguration

            fm_config = FluorescenceConfiguration.load(fm_config_path)
        else:
            # fall back to the auto-persisted FM working state (survives restarts)
            from fibsem.fm.config import load_fm_configuration

            fm_config = load_fm_configuration()

        return MicroscopeSettings(
            system=SystemSettings.from_dict(settings),
            image=ImageSettings.from_dict(settings["imaging"]),
            protocol=protocol,
            milling=FibsemMillingSettings.from_dict(settings["milling"]),
            fm=fm_config,
        )


@dataclass
class FibsemExperimentRef:
    """A reference to the experiment an image was acquired for. Provenance.

    Not an experiment. The experiment itself is the application's own record --
    AutoLamella's ``Experiment``, with positions, protocol and history -- and this
    is a denormalised pointer to it, embedded in every image so the file can say
    what produced it without the surrounding directory. Named ``...Ref`` because
    the two types are otherwise a single import apart and easily confused.

    **Defers to the record.** Authoritative only in the absence of the experiment
    record, which wins on conflict (FIB-445 D2). That rule needs a stable key: with
    only a name, a renamed experiment is indistinguishable from a different one, so
    there is no conflict to detect, just two strings that disagree.

    The deference rule applies here because a richer record exists to defer to. It
    does not apply to every embedded copy -- see ``FibsemUser``.

    **Identity only.** Which experiment, which item, which task, and when. What
    software was running is a property of the running system, not of an experiment,
    so it lives on ``SystemInfo`` and only there (FIB-445 D1, FIB-448). This carried
    duplicates of it until v5.

    **Two write rates, one record.** The experiment is set once at registration; the
    item and task change as a run progresses and are written and cleared around each
    one (FIB-466). They are kept together because they are one answer to one question
    -- *what produced this image* -- and a reader wants "experiment X, lamella Y, task
    Z" in one place rather than assembled from two.

    That only works because every image gets its **own copy**: `_set_additional_metadata`
    deepcopies this. It used to be shared by reference, which was harmless while
    nothing mutated it, and would have silently rewritten the item on every
    already-acquired image the moment one did.
    """

    # Experiment.id, a UUID -- stable, the join key. Held the experiment *name* up to
    # and including v0.5.2; a file whose `name` is absent is from that era and its `id`
    # is a name. See FIB-446.
    id: Optional[str] = None
    name: Optional[str] = None
    # default_factory, not a plain default: a plain default is evaluated once at
    # class definition, so every experiment recorded the interpreter's import
    # time rather than its own creation time.
    date: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))

    # Where in the run. None outside a workflow -- the minimap, a manual acquisition,
    # a script -- which is a real answer rather than missing information.
    #
    # "item" rather than "lamella": the core library has no reason to know what an
    # application works through one at a time, and ``HookContext`` settled on the same
    # word for the same reason. IDs *and* names because they answer different
    # questions -- the name is what a person reads, the id is what a reader joins on
    # and it survives a rename. Recording only names is the mistake FIB-446 fixed for
    # the experiment itself.
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None

    def set_workflow_metadata(
        self,
        item_id: Optional[str] = None,
        item_name: Optional[str] = None,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> None:
        """Record what subsequent images should say about where in the run they were
        taken, leaving experiment identity alone.

        Named ``..._metadata`` because that is all it does. It does not start, select
        or configure a workflow -- several widgets have a ``set_workflow*`` that does
        something along those lines, and this is not one of them.
        """
        self.item_id = item_id
        self.item_name = item_name
        self.task_id = task_id
        self.task_name = task_name

    def clear_workflow_metadata(self) -> None:
        """Stop stamping an item and task, keeping the experiment.

        A method rather than four assignments at the call site: this has to run on
        every path out of a task, and the one thing it must never do is take the
        experiment's own identity with it.
        """
        self.set_workflow_metadata()

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "date": self.date,
            "item_id": self.item_id,
            "item_name": self.item_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemExperimentRef":
        """Converts from a dictionary.

        Files written before v5 also carry `application`, `application_version`,
        `fibsem_version`, `fibsem_revision` and `method` here. None are read.
        `application`, `fibsem_version` and `fibsem_revision` live on ``SystemInfo``,
        which is where a reader should look; `application_version` was never
        populated anywhere; `method` never held anything but the string "null". The
        values are still in those files if anyone needs to dig them out by hand.
        """
        return FibsemExperimentRef(
            id=settings.get("id", "Unknown"),
            # Absent in files written before v0.5.3, where `id` holds the name. Left
            # as None rather than backfilled from `id`, so a reader can tell the two
            # eras apart instead of being handed a name that claims to be an ID.
            name=settings.get("name"),
            date=settings.get("date", "Unknown"),
            # Absent before v8, and in any image acquired outside a workflow.
            item_id=settings.get("item_id"),
            item_name=settings.get("item_name"),
            task_id=settings.get("task_id"),
            task_name=settings.get("task_name"),
        )


@dataclass
class FibsemUser:
    """Who acquired an image. Provenance, and the only user model there is.

    Unlike ``FibsemExperimentRef``, this is not a reference to anything: there is
    no richer user record for it to defer to, so the "the record wins on conflict"
    rule (FIB-445 D2) does not apply -- this *is* the record, and it happens to be
    stored embedded in every image.

    That would change if a user table ever lands (the DB work models one), at which
    point this becomes a snapshot of it and the deference rule starts to apply.
    Worth knowing before treating the two structures as the same kind of thing.

    Practical consequence of being embedded: it cannot be corrected retroactively.
    A wrong name here is wrong in every file already written.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    organization: Optional[str] = None
    # Which machine, not which person -- host identity sitting on the user record.
    # Left here rather than moved, because moving it changes the serialised shape.
    hostname: Optional[str] = None
    # TODO: add host_ip_address

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "organization": self.organization,
            "hostname": self.hostname,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemUser":
        """Converts from a dictionary."""
        return FibsemUser(
            name=settings.get("name", "Unknown"),
            email=settings.get("email", "Unknown"),
            organization=settings.get("organization", "Unknown"),
            hostname=settings.get("hostname", "Unknown"),
        )

    @staticmethod
    def from_environment() -> "FibsemUser":
        import getpass
        import platform
        import socket

        # getpass covers every platform: USERNAME is Windows-only, so reading it
        # directly meant Linux and macOS fell through to the literal string
        # "username" -- wrong rather than absent, and silently so. That affects
        # Odemis/METEOR sites, which run on Linux. See FIB-447.
        try:
            username = getpass.getuser()
        except Exception:
            # getpass raises if it cannot resolve a name (no passwd entry, no
            # environment). Nothing here is worth failing an acquisition over.
            username = "unknown"

        if platform.system() == "Windows":
            hostname = os.environ.get("COMPUTERNAME", "hostname")
        elif platform.system() in ["Linux", "Darwin"]:
            hostname = socket.gethostname()
        else:
            hostname = "hostname"

        user = FibsemUser(
            name=username, email="null", organization="null", hostname=hostname
        )

        return user


@dataclass
class SessionInfo:
    """Which instrument, whose account, and what software worked on an experiment.

    A session is what ``setup_session`` establishes: a connected instrument and the
    configuration around it. ``SystemInfo`` answers the instrument half and rides in
    every image; this is that plus who is at the keyboard and which plugins are
    installed, recorded once on the experiment itself.

    An experiment directory otherwise says what it contains and nothing about what
    made it, so "was this before or after the upgrade", "which machine was this on"
    and "which build of my plugin ran" are all unanswerable from the record. Every
    part of this was already collected per-image; this promotes it (FIB-451).

    **The latest session, not the one that created the experiment.**
    ``Experiment.create()`` has no microscope -- the dialog that calls it never
    holds one -- so the instrument is simply unknown until a session adopts the
    experiment. The latest is the fact that can actually be established, and it has
    the useful property of backfilling onto experiments that predate this.

    Overwriting loses less than it looks like. Every image already carries its own
    ``SystemInfo``, so an experiment spanning an upgrade shows v0.5.1 on day-one
    images and v0.5.2 here -- two true facts, which is the same reasoning that put
    the version fields on ``SystemInfo`` in the first place (FIB-445 D1). Keeping
    every session rather than the latest is FIB-452's shape, not this record's.
    """

    recorded_at: float = field(
        default_factory=lambda: datetime.timestamp(datetime.now())
    )
    system: Optional[SystemInfo] = None
    user: Optional[FibsemUser] = None
    # Distribution name -> version for every installed extension; see
    # `installed_plugin_versions`. A plain dict so it survives `yaml.safe_dump`,
    # which is how an experiment is written.
    plugins: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def collect(
        cls, microscope: "FibsemMicroscope", user: Optional[FibsemUser] = None
    ) -> "SessionInfo":
        """Snapshot what is running right now.

        ``user`` overrides the environment's, for a caller that knows better: on a
        shared facility login the OS account names the workstation rather than the
        operator, so a name somebody actually typed is the stronger evidence.

        The system info is copied. It is a live object on the microscope -- the
        application field is set on it during registration, and a driver may update
        it -- and a record that quietly changes after the fact is not a record.
        """
        from copy import deepcopy

        # Lazy: `report` imports all three registries, so importing it here would
        # cycle straight back through this module.
        from fibsem.plugins.report import installed_plugin_versions

        info = getattr(getattr(microscope, "system", None), "info", None)
        return cls(
            system=deepcopy(info) if info is not None else None,
            user=user if user is not None else FibsemUser.from_environment(),
            plugins=installed_plugin_versions(),
        )

    def to_dict(self) -> dict:
        return {
            "recorded_at": self.recorded_at,
            "system": self.system.to_dict() if self.system is not None else None,
            "user": self.user.to_dict() if self.user is not None else None,
            "plugins": dict(self.plugins),
        }

    @staticmethod
    def from_dict(ddict: dict) -> "SessionInfo":
        system = ddict.get("system")
        user = ddict.get("user")
        return SessionInfo(
            recorded_at=ddict.get("recorded_at"),
            system=SystemInfo.from_dict(system) if system else None,
            user=FibsemUser.from_dict(user) if user else None,
            plugins=dict(ddict.get("plugins") or {}),
        )


@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage.

    Three kinds of claim live here, and they are easy to mistake for each other
    (FIB-445 D1). Anything added should be placed deliberately in one of them:

    **Provenance** -- what produced this image. ``system_info`` (which instrument),
    ``user`` (who), ``experiment`` (which run). Constant for a run. The same
    question at a finer grain -- which lamella, which task -- is not recorded yet;
    see FIB-466. It varies *within* a run, which changes the mechanism that writes
    it, but not the kind of fact it is.

    **Configuration** -- what the instrument *is*. ``hardware_geometry``: the fixed
    physical arrangement a projection needs. Up to v5 this was the entire
    ``SystemSettings``, 1683 bytes of it, to deliver six numbers -- and it carried
    the manipulator configuration and the simulator flags into every picture. See
    FIB-481.

    **Observation** -- what the instrument was doing. ``microscope_state``,
    ``pixel_size``. Measured at acquisition.

    **Request** -- what was asked for. ``image_settings``. Note this is *intent*,
    not outcome: if autocontrast ran, or the requested hfw was clamped, nothing
    here records that the request was not honoured. See FIB-482.

    The rule for provenance is one writer per fact, denormalised deliberately at
    this boundary: an image is embedded in a file that may be copied, emailed or
    read years later, so it carries enough to identify its source without the
    surrounding directory. Whether an embedded copy *defers* to a richer record is
    a separate question, answered per-structure -- see ``FibsemExperimentRef``
    (defers) and ``FibsemUser`` (does not, because there is nothing to defer to).
    """

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    # Both replaced `system: Optional[SystemSettings]` in v6 (FIB-481). Optional
    # because a file written before v6 may carry neither in a recoverable form, and
    # because a FibsemImage can be constructed without a microscope at all.
    system_info: Optional[SystemInfo] = None
    hardware_geometry: Optional[FibsemHardwareGeometry] = None
    version: str = METADATA_VERSION
    user: FibsemUser = field(default_factory=lambda: FibsemUser())
    experiment: FibsemExperimentRef = field(
        default_factory=lambda: FibsemExperimentRef()
    )

    @property
    def beam_type(self) -> BeamType:
        return self.image_settings.beam_type

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.microscope_state.stage_position

    @property
    def acquisition_date(self) -> datetime:
        return datetime.fromtimestamp(self.microscope_state.timestamp)

    def to_dict(self) -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        settings_dict = {}
        if self.image_settings is not None:
            settings_dict["image"] = self.image_settings.to_dict()
        if self.version is not None:
            settings_dict["version"] = self.version
        if self.pixel_size is not None:
            settings_dict["pixel_size"] = self.pixel_size.to_dict()
        if self.microscope_state is not None:
            settings_dict["microscope_state"] = self.microscope_state.to_dict()
        # Not nested under the microscope_state guard: who acquired an image is
        # unrelated to whether the instrument's state was captured, and nesting it
        # dropped the user silently -- from_dict defaults it back, so a reload
        # produced a plausible empty FibsemUser rather than an error. See FIB-486.
        settings_dict["user"] = self.user.to_dict()
        settings_dict["experiment"] = self.experiment.to_dict()
        settings_dict["system_info"] = (
            self.system_info.to_dict() if self.system_info is not None else {}
        )
        settings_dict["hardware_geometry"] = (
            self.hardware_geometry.to_dict()
            if self.hardware_geometry is not None
            else {}
        )

        return settings_dict

    @staticmethod
    def _geometry_from_legacy_system(system: dict) -> Optional[FibsemHardwareGeometry]:
        """Recover the geometry from a pre-v6 `system` blob.

        Read with `.get()` chains rather than by building a `SystemSettings` first.
        That constructor is bracket-indexed throughout -- a blob missing any of
        `stage`, `electron`, `ion`, `manipulator`, `gis` or `info` raises KeyError --
        and inheriting that here would break exactly the old files this exists to
        load. See `tests/test_metadata_fixtures.py`.

        Compustage is recovered the way the reprojection used to detect it, by model
        name, falling back to the simulator flag. That match is wrong -- a capability
        inferred from a name -- which is why v6 records it instead. It survives here
        only because a pre-v6 file has nothing better in it.
        """
        if not system:
            return None

        stage = system.get("stage") or {}
        electron = system.get("electron") or {}
        ion = system.get("ion") or {}
        info = system.get("info") or {}
        sim = system.get("sim") or {}

        model = info.get("model") or ""
        is_compustage = "Arctis" in model or bool(sim.get("is_compustage", False))

        # Field defaults where a key is absent, so a partial blob degrades to the
        # same values a freshly-constructed record would have.
        default = FibsemHardwareGeometry()
        return FibsemHardwareGeometry(
            column_tilt=electron.get("column_tilt", default.column_tilt),
            fib_column_tilt=ion.get("column_tilt", default.fib_column_tilt),
            shuttle_pre_tilt=stage.get("shuttle_pre_tilt", default.shuttle_pre_tilt),
            rotation_reference=stage.get(
                "rotation_reference", default.rotation_reference
            ),
            rotation_180=stage.get("rotation_180", default.rotation_180),
            is_compustage=is_compustage,
        )

    @staticmethod
    def from_dict(settings: dict) -> "FibsemImageMetadata":
        """Converts a dictionary to metadata."""

        image_settings = ImageSettings.from_dict(settings["image"])
        version = settings.get("version", UNVERSIONED_METADATA)
        if settings["pixel_size"] is not None:
            pixel_size = Point.from_dict(settings["pixel_size"])
        if settings["microscope_state"] is not None:
            microscope_state = MicroscopeState.from_dict(settings["microscope_state"])

        # Presence-detection, not a version switch (FIB-445 D3): v6 writes
        # `system_info` and `hardware_geometry`, everything before it wrote a whole
        # `system`. Both are optional -- an image may be built without a microscope.
        legacy_system = settings.get("system") or {}

        info_dict = settings.get("system_info") or legacy_system.get("info") or {}
        system_info = SystemInfo.from_dict(info_dict) if info_dict else None

        geometry_dict = settings.get("hardware_geometry") or {}
        if geometry_dict:
            hardware_geometry = FibsemHardwareGeometry.from_dict(geometry_dict)
        else:
            hardware_geometry = FibsemImageMetadata._geometry_from_legacy_system(
                legacy_system
            )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
            user=FibsemUser.from_dict(settings.get("user", {})),
            experiment=FibsemExperimentRef.from_dict(settings.get("experiment", {})),
            system_info=system_info,
            hardware_geometry=hardware_geometry,
        )
        return metadata


@dataclass
class ImageStats:
    """Histogram statistics for a FibsemImage.

    All intensity values are normalised to [0, 1] relative to the dtype maximum.
    """

    mean: float
    std: float
    p01: float  # 1st percentile
    p99: float  # 99th percentile
    saturation_lo: float  # fraction of pixels at dtype min
    saturation_hi: float  # fraction of pixels at dtype max
    contrast_ratio: float  # coefficient of variation: std / mean
    range_utilisation: float  # p99 - p01
    median: float  # normalised median (robust alternative to mean)
    snr: float  # mean / std
    entropy: float  # Shannon entropy of the normalised histogram (bits)

    def __str__(self) -> str:
        return (
            f"mean={self.mean:.3f}, median={self.median:.3f}, std={self.std:.3f}, "
            f"p01={self.p01:.3f}, p99={self.p99:.3f}, "
            f"sat_lo={self.saturation_lo:.4f}, sat_hi={self.saturation_hi:.4f}, "
            f"CV={self.contrast_ratio:.3f}, SNR={self.snr:.2f}, "
            f"range={self.range_utilisation:.3f}, entropy={self.entropy:.2f}b"
        )

    def converged(
        self, mean_target: float, mean_tolerance: float, saturation_limit: float
    ) -> bool:
        """Return True when mean and saturation hard criteria are both satisfied."""
        return (
            abs(self.mean - mean_target) <= mean_tolerance
            and self.saturation_hi <= saturation_limit
        )


class FibsemImage:
    """
    Class representing a FibsemImage and its associated metadata.
    Has in built methods to deal with image types of TESCAN and ThermoFisher API

    Args:
        data (np.ndarray): The image data stored in a numpy array.
        metadata (FibsemImageMetadata, optional): The metadata associated with the image. Defaults to None.

    Methods:
        load(cls, tiff_path: str) -> "FibsemImage":
            Loads a FibsemImage from a tiff file.

            Args:
                tiff_path (path): path to the tif* file

            Returns:
                FibsemImage: instance of FibsemImage

        save(self, path: Path) -> str:
            Saves a FibsemImage to a tiff file.

            Inputs:
                path (path): path to save directory and filename

            Returns:
                str: the resolved path written to

    Attributes:
        filepath (Optional[str]): the file this image is associated with on disk, set by
            save() and load(). None for an image that has never been written or read.
    """

    def __init__(
        self, data: np.ndarray, metadata: Optional[FibsemImageMetadata] = None
    ):
        if check_data_format(data):
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]
            self.data = data  # setter also populates _filtered_data
        else:
            raise Exception("Invalid Data format for Fibsem Image")
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None
        # the file this image is associated with on disk, set by save() and load().
        # not serialised: it describes where the image lives, not what it contains.
        self.filepath: Optional[str] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the image data."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the image data."""
        return self.data.dtype

    @property
    def data(self) -> NDArray:
        """Returns the image data as a numpy array."""
        return self._data

    @data.setter
    def data(self, value: NDArray) -> None:
        if check_data_format(value):
            self._data = value
            self._filtered_data = self._filter_data(value)
        else:
            raise Exception("Invalid Data format for Fibsem Image")

    @property
    def filtered_data(self) -> NDArray:
        """Returns a median filtered version of the image data. Typically used for display purposes."""
        return self._filtered_data

    def _filter_data(self, data, size: int = 3, sigma: float = 1) -> NDArray:
        """Returns a filtered version of the image data using a median filter followed by a gaussian filter. Can be used for display or processing purposes."""
        # opencv rather than scipy: the same two filters, ~300x faster at 4096x4096. the
        # setter runs this on every assignment, including the one inside load(), so a
        # 385 MB overview took ~30 s to load on scipy and ~0.2 s here. data is 2D uint8 or
        # uint16 (check_data_format), exactly what medianBlur(ksize=3) accepts.
        # the kernel size and border mode match scipy, not opencv's defaults, which would be
        # a 7x7 kernel and reflect-101 -- 16x further from the output this replaces.
        radius = int(4.0 * sigma + 0.5)  # scipy's gaussian_filter default truncate=4.0
        ksize = 2 * radius + 1
        filtered = cv2.GaussianBlur(
            cv2.medianBlur(data, size),
            (ksize, ksize),
            sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        # opencv drops a trailing length-1 axis. (H, W, 1) can reach the setter unsqueezed
        # -- only __init__ squeezes it -- and filtered_data has always matched data's shape.
        return filtered.reshape(data.shape)

    @classmethod
    def load(cls, tiff_path: str) -> "FibsemImage":
        """Loads a FibsemImage from a tiff file.

        Args:
            tiff_path (path): path to the tif* file

        Returns:
            FibsemImage: instance of FibsemImage
        """
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                metadata = json.loads(
                    tiff_image.pages[0].tags["ImageDescription"].value
                )
                metadata = FibsemImageMetadata.from_dict(metadata)
            except Exception as e:
                metadata = None
                # print(f"Error: {e}")
                # import traceback
                # traceback.print_exc()
        image = cls(data=data, metadata=metadata)
        image.filepath = str(tiff_path)
        return image

    def save(self, path: Optional[Union[Path, str]] = None) -> str:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            path (path): path to save directory and filename

        Returns:
            str: the resolved path the image was written to (also set on self.filepath)
        """

        if path is None:
            if self.metadata is None:
                raise ValueError(
                    "No metadata provided, cannot determine save path. Please provide a path."
                )
            filename = self.metadata.image_settings.filename
            directory = self.metadata.image_settings.path
            if filename is None:
                raise ValueError(
                    "No filename provided in metadata, cannot determine save path. Please provide a path."
                )
            if directory is None:
                raise ValueError(
                    "No path provided in metadata, cannot determine save path. Please provide a path."
                )
            # The recorded path is an absolute directory on whichever machine acquired
            # the image, and it travels inside the file. Creating it would mean loading
            # a colleague's image and re-saving it silently reconstructs their directory
            # tree here -- `D:\SharedData\<their name>\...` and all. A path the caller
            # passes in is theirs to create; one that arrived in a file is not.
            if not os.path.isdir(directory):
                raise ValueError(
                    f"The directory recorded in this image's metadata does not exist: "
                    f"{directory}. It is from the machine that acquired the image. "
                    f"Please provide a path."
                )
            path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = Path(path).with_suffix(".tif")

        if self.metadata is not None:
            metadata_dict = self.metadata.to_dict()
        else:
            metadata_dict = None
        tff.imwrite(
            path,
            self.data,
            metadata=metadata_dict,
        )
        # set only after a successful write, so a recorded path is always a path that exists
        self.filepath = str(path)
        return self.filepath

    ### EXPERIMENTAL START ####

    def _save_ome_tiff(self, path: str, filename: str) -> None:
        from ome_types import OME
        from ome_types.model import (
            Channel,
            Image,
            Instrument,
            ManufacturerSpec,
            MapAnnotation,
            Microscope,
            Pixels,
            Plane,
            StructuredAnnotations,
            TiffData,
        )
        from ome_types.model.simple_types import UnitsLength

        md = self.metadata
        microscope = Microscope(
            # md.system_info since FIB-481; this was md.system.info, which no longer
            # exists. Nothing calls this -- see FIB-485 for whether it should live at
            # all -- but leaving a known-broken reference is worse than fixing it.
            manufacturer=md.system_info.manufacturer,
            model=md.system_info.model,
            serial_number=md.system_info.serial_number,
        )
        instrument = Instrument(microscope=microscope)

        size_y = self.data.shape[0]
        size_x = self.data.shape[1]
        # TODO: use sample plane projection position for this pos
        stage_position = md.microscope_state.stage_position
        pos_x = stage_position.x
        pos_y = stage_position.y
        pos_z = stage_position.z

        plane = Plane(
            the_c=0,
            the_z=0,
            the_t=0,
            position_x=pos_x,
            position_y=pos_y,
            position_z=pos_z,
            position_x_unit=UnitsLength.METER,
            position_y_unit=UnitsLength.METER,
            position_z_unit=UnitsLength.METER,
        )
        tiff_data = TiffData(ifd=0)

        ch = Channel(
            id="Channel:0",
            name="SEM" if md.image_settings.beam_type is BeamType.ELECTRON else "FIB",
            samples_per_pixel=1,
        )

        pixels = Pixels(
            id="Pixels:0",
            dimension_order="XYZTC",
            size_x=size_x,
            size_y=size_y,
            size_c=1,
            size_t=1,
            size_z=1,
            type=self.data.dtype.name,
            physical_size_x=md.pixel_size.x,
            physical_size_y=md.pixel_size.y,
            physical_size_x_unit=UnitsLength.METER,
            physical_size_y_unit=UnitsLength.METER,
            channels=[ch],
            planes=[plane],
            tiff_data_blocks=[tiff_data],
        )

        sa = StructuredAnnotations()
        mapAnnotation = [
            MapAnnotation(
                id="Annotation:0", value={"fibsemOS": json.dumps(md.to_dict())}
            )
        ]
        sa.map_annotations = mapAnnotation

        ome_image = Image(
            id="Image:0",
            name=md.image_settings.filename,
            acquisition_date=md.microscope_state.timestamp,
            pixels=pixels,
        )

        ome = OME()
        ome.images.append(ome_image)
        ome.instruments.append(instrument)
        ome.structured_annotations = sa

        assert tff.OmeXml.validate(ome.to_xml()), "OME-XML validation failed"

        # TODO: check for a unique filename
        path = os.path.join(path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # add suffix if not present
        OME_TIFF_SUFFIXES = (".ome.tiff", ".ome.tif", ".tif", ".tiff")
        if not path.endswith(OME_TIFF_SUFFIXES):
            # Note: with_suffix doesn't work correctly with double extensions, .ome.tiff
            path = Path(path).with_suffix(".ome.tiff")

        with tff.TiffWriter(path) as tif:
            tif.write(self.data, contiguous=True)
            tif.overwrite_description(ome.to_xml())

    @classmethod
    def _load_from_ome_tiff(cls, path: str) -> "FibsemImage":
        import ome_types

        # read ome-xml, extract fibsemOS metadata
        try:
            ome = ome_types.from_tiff(path)
            fibsemos_md = json.loads(
                ome.structured_annotations.map_annotations[0].value["fibsemOS"]
            )

            # parse metadata to struct
            md = FibsemImageMetadata.from_dict(fibsemos_md)
        except Exception as e:
            import logging

            logging.warning(f"Failing to load metadata from OME-TIFF: {e}")
            md = None

        # load image data
        with tff.TiffFile(path) as tif:
            data = tif.pages[0].asarray()
        return cls(data=data, metadata=md)

    ### EXPERIMENTAL END ####

    def extract_region(self, rect: "FibsemRectangle") -> "FibsemImage":
        """Extract a sub-region of the image and return a new FibsemImage with valid metadata.

        The returned image has the same resolution/hfw as the original (metadata describes
        the full scan), with reduced_area updated to reflect the extracted region.

        Args:
            rect (FibsemRectangle): Normalized rectangle (0–1 coordinates) defining the region to extract.

        Returns:
            FibsemImage: A new FibsemImage containing the cropped data and updated metadata.

        Raises:
            ValueError: If metadata is None or if rect coordinates are invalid / out of bounds.
        """
        if self.metadata is None:
            raise ValueError("Cannot extract region from FibsemImage without metadata.")

        if not rect.is_valid_reduced_area:
            raise ValueError(
                f"Invalid rectangle: {rect.pretty_string}. "
                "left/top must be >= 0, width/height > 0, and region must not exceed image bounds."
            )

        # Convert normalized coords to pixel indices using existing helper
        x, y, pw, ph = rect.to_pixel_coordinates(
            self.data.shape
        )  # (x, y, width, height)
        cropped = self.data[y : y + ph, x : x + pw].copy()

        # Clone metadata; only update reduced_area — resolution/hfw/pixel_size unchanged
        from copy import deepcopy

        new_metadata = deepcopy(self.metadata)
        new_metadata.image_settings.reduced_area = rect

        return FibsemImage(data=cropped, metadata=new_metadata)

    def resize(self, resolution: Tuple[int, int]) -> "FibsemImage":
        """Resize the image to the given resolution and return a new FibsemImage with updated metadata.

        HFW is preserved; pixel_size is recalculated to match the new pixel dimensions.

        Args:
            resolution (Tuple[int, int]): Target resolution as (width, height) in pixels.

        Returns:
            FibsemImage: A new FibsemImage with resized data and updated metadata.

        Raises:
            ValueError: If metadata is None.
        """
        if self.metadata is None:
            raise ValueError("Cannot resize FibsemImage without metadata.")

        from skimage.transform import resize as skimage_resize

        new_width, new_height = resolution
        resized = skimage_resize(
            self.data,
            output_shape=(new_height, new_width),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(self.data.dtype)

        from copy import deepcopy

        new_metadata = deepcopy(self.metadata)
        new_metadata.image_settings.resolution = resolution
        # pixel size scales inversely with resolution at fixed HFW
        orig_height, orig_width = self.data.shape
        new_metadata.pixel_size = Point(
            x=self.metadata.pixel_size.x * (orig_width / new_width),
            y=self.metadata.pixel_size.y * (orig_height / new_height),
        )

        return FibsemImage(data=resized, metadata=new_metadata)

    def apply_gamma(self, gamma: float) -> "FibsemImage":
        """Return a copy of the image with the given gamma correction applied.

        Args:
            gamma (float): Gamma value to apply. Must be > 0.
                Values < 1 brighten the image; values > 1 darken it.

        Returns:
            FibsemImage: New image with gamma-corrected data and the same metadata.

        Raises:
            ValueError: If gamma is not positive.
        """
        from copy import deepcopy

        from fibsem.autofunctions.gamma import apply_gamma as _apply_gamma

        return FibsemImage(
            data=_apply_gamma(self.data, gamma), metadata=deepcopy(self.metadata)
        )

    def auto_contrast_brightness(
        self,
        clip_percentile_lo: float = 0.5,
        clip_percentile_hi: float = 99.5,
    ) -> "FibsemImage":
        """Return a copy of the image with a percentile stretch applied.

        Pixel values are clipped to [p_lo, p_hi] and linearly rescaled to fill
        the full dtype range.

        Args:
            clip_percentile_lo: Lower clip percentile (default 0.5).
            clip_percentile_hi: Upper clip percentile (default 99.5).

        Returns:
            FibsemImage: New image with stretched data and the same metadata.
        """
        from copy import deepcopy

        from fibsem.imaging.utils import percentile_stretch

        stretched = percentile_stretch(
            self.data, clip_percentile_lo, clip_percentile_hi
        )
        return FibsemImage(data=stretched, metadata=deepcopy(self.metadata))

    def compute_stats(self) -> "ImageStats":
        """Compute histogram statistics for this image.

        Returns:
            ImageStats with all metrics normalised to [0, 1].
        """
        data = self.filtered_data.astype(np.float64)
        if np.issubdtype(self.data.dtype, np.floating):
            dtype_max = 1.0
        else:
            dtype_max = float(np.iinfo(self.data.dtype).max)

        norm = data / dtype_max
        mean = float(np.mean(norm))
        std = float(np.std(norm))
        p01 = float(np.percentile(norm, 1))
        p99 = float(np.percentile(norm, 99))
        sat_lo = float(np.mean(norm <= 0.0))
        sat_hi = float(np.mean(norm >= 1.0))
        cv = std / mean if mean > 0 else 0.0
        median = float(np.median(norm))
        snr = mean / std if std > 0 else 0.0

        counts, _ = np.histogram(norm, bins=256, range=(0.0, 1.0))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        return ImageStats(
            mean=mean,
            std=std,
            p01=p01,
            p99=p99,
            saturation_lo=sat_lo,
            saturation_hi=sat_hi,
            contrast_ratio=cv,
            range_utilisation=p99 - p01,
            median=median,
            snr=snr,
            entropy=entropy,
        )

    @property
    def brightness(self) -> float:
        """Mean pixel intensity of the image."""
        return float(np.mean(self.data))

    @staticmethod
    def generate_blank_image(
        resolution: Tuple[int, int] = (1536, 1024),
        hfw: float = 100e-6,
        pixel_size: Optional[Point] = None,
        random: bool = False,
        dtype: np.dtype = np.uint8,
    ) -> "FibsemImage":
        """Generate a blank image with a given resolution and field of view.
        Args:
            resolution: List[int]: Resolution of the image.
            hfw: float: Horizontal field width of the image.
            pixel_size: Point: Pixel size of the image.
            random: bool: If True, generate a random (noise) image.
            dtype: np.dtype: Data type of the image. Defaults to np.uint8.
        Returns:
            FibsemImage: Blank image with valid metadata from display.
        """
        # need at least one of hfw, pixelsize
        if pixel_size is None and hfw is None:
            raise ValueError("Need to specify either hfw or pixelsize")

        if pixel_size is None:
            vfw = hfw * resolution[1] / resolution[0]
            pixel_size = Point(hfw / resolution[0], vfw / resolution[1])

        shape = (resolution[1], resolution[0])
        if random:
            arr = np.random.randint(0, 255, size=shape, dtype=dtype)
        else:
            arr = np.zeros(shape=shape, dtype=dtype)

        image = FibsemImage(
            data=arr,
            metadata=FibsemImageMetadata(
                image_settings=ImageSettings(hfw=hfw, resolution=resolution),
                microscope_state=None,
                pixel_size=pixel_size,
            ),
        )
        return image


@dataclass
class ReferenceImages:
    low_res_eb: FibsemImage
    high_res_eb: FibsemImage
    low_res_ib: FibsemImage
    high_res_ib: FibsemImage

    def __iter__(self) -> List[FibsemImage]:
        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


def check_data_format(data: np.ndarray) -> bool:
    """Checks that data is in the correct format."""
    # assert data.ndim == 2  # or data.ndim == 3
    # assert data.dtype in [np.uint8, np.uint16]
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    return data.ndim == 2 and data.dtype in [np.uint8, np.uint16]


def save_tiff(data: np.ndarray, path: Union[str, Path]) -> str:
    """Write a raw image array to a TIFF file.

    Args:
        data: Image data to write.
        path: Destination path (``.tif`` appended if no suffix given).

    Returns:
        The path written to, as a string.
    """
    path = str(path)
    if not path.lower().endswith((".tif", ".tiff")):
        path += ".tif"
    tff.imwrite(path, data)
    return path


def load_tiff(path: Union[str, Path]) -> np.ndarray:
    """Read a raw image array from a TIFF file."""
    return tff.imread(str(path))


@dataclass
class FibsemGasInjectionSettings:
    port: str
    gas: str
    duration: float
    insert_position: Optional[str] = None  # multichem only

    @staticmethod
    def from_dict(d: dict):
        return FibsemGasInjectionSettings(
            port=d["port"],
            gas=d["gas"],
            duration=d["duration"],
            insert_position=d.get("insert_position", None),
        )

    def to_dict(self):
        return {
            "port": self.port,
            "gas": self.gas,
            "duration": self.duration,
            "insert_position": self.insert_position,
        }


def calculate_fiducial_area_v2(
    image: FibsemImage, fiducial_centre: Point, fiducial_length: float
) -> Tuple[FibsemRectangle, bool]:

    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError("Image metadata or pixel size is not set.")

    from fibsem import conversions

    pixelsize = image.metadata.pixel_size.x

    fiducial_centre.y = -fiducial_centre.y
    fiducial_centre_px = conversions.convert_point_from_metres_to_pixel(
        fiducial_centre, pixelsize
    )

    rcx = fiducial_centre_px.x / image.metadata.image_settings.resolution[0] + 0.5
    rcy = fiducial_centre_px.y / image.metadata.image_settings.resolution[1] + 0.5

    fiducial_length_px = (
        conversions.convert_metres_to_pixels(fiducial_length, pixelsize)
        * 1.5  # SCALE_FACTOR
    )
    h_offset = fiducial_length_px / image.metadata.image_settings.resolution[0] / 2
    v_offset = fiducial_length_px / image.metadata.image_settings.resolution[1] / 2

    left = rcx - h_offset
    top = rcy - v_offset
    width = 2 * h_offset
    height = 2 * v_offset

    if left < 0 or (left + width) > 1 or top < 0 or (top + height) > 1:
        flag = True
    else:
        flag = False

    alignment_area = FibsemRectangle(left, top, width, height)

    return alignment_area, flag


DEFAULT_ALIGNMENT_AREA = {"left": 0.7, "top": 0.3, "width": 0.25, "height": 0.4}


@dataclass
class MillingAlignment:
    """Drift correction settings for milling"""

    enabled: bool = True
    interval_enabled: bool = False
    interval: int = 30  # seconds
    rect: FibsemRectangle = field(
        default_factory=lambda: FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
    )
    use_autocontrast: bool = True
    use_autofocus: bool = False
    steps: int = 3
    imaging: ImageSettings = field(default_factory=ImageSettings)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "interval_enabled": self.interval_enabled,
            "interval": self.interval,
            "rect": self.rect.to_dict(),
            "use_autocontrast": self.use_autocontrast,
            "use_autofocus": self.use_autofocus,
            "steps": self.steps,
            "imaging": self.imaging.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "MillingAlignment":
        return MillingAlignment(
            enabled=d.get("enabled", False),
            interval_enabled=d.get("interval_enabled", False),
            interval=d.get("interval", 30),
            rect=FibsemRectangle.from_dict(
                d.get("rect", DEFAULT_ALIGNMENT_AREA),
            ),
            use_autocontrast=d.get("use_autocontrast", True),
            use_autofocus=d.get("use_autofocus", False),
            steps=d.get("steps", 3),
            imaging=ImageSettings.from_dict(d.get("imaging", {})),
        )


@dataclass
class RangeLimit:
    min: float
    max: float

    def clamp(self, value: float) -> float:
        return max(self.min, min(self.max, value))

    def to_dict(self) -> dict:
        return {"min": self.min, "max": self.max}

    @staticmethod
    def from_dict(d: dict) -> "RangeLimit":
        return RangeLimit(min=d["min"], max=d["max"])


@dataclass
class ReferenceImageParameters:
    imaging: ImageSettings = field(default_factory=ImageSettings)
    field_of_view1: float = field(
        default=100e-6, metadata={"tooltip": "Field of view for first reference image"}
    )
    field_of_view2: float = field(
        default=150e-6, metadata={"tooltip": "Field of view for second reference image"}
    )
    acquire_sem: bool = field(
        default=True, metadata={"tooltip": "Whether to acquire SEM reference images"}
    )
    acquire_fib: bool = field(
        default=True, metadata={"tooltip": "Whether to acquire FIB reference images"}
    )
    acquire_image1: bool = field(
        default=True, metadata={"tooltip": "Whether to acquire first reference image"}
    )
    acquire_image2: bool = field(
        default=True, metadata={"tooltip": "Whether to acquire second reference image"}
    )

    def to_dict(self) -> dict:
        return {
            "imaging": self.imaging.to_dict(),
            "field_of_view1": self.field_of_view1,
            "field_of_view2": self.field_of_view2,
            "acquire_sem": self.acquire_sem,
            "acquire_fib": self.acquire_fib,
            "acquire_image1": self.acquire_image1,
            "acquire_image2": self.acquire_image2,
        }

    @staticmethod
    def from_dict(settings: dict) -> "ReferenceImageParameters":
        imaging = ImageSettings.from_dict(settings.get("imaging", {}))
        return ReferenceImageParameters(
            imaging=imaging,
            field_of_view1=settings.get("field_of_view1", 100e-6),
            field_of_view2=settings.get("field_of_view2", 150e-6),
            acquire_sem=settings.get("acquire_sem", True),
            acquire_fib=settings.get("acquire_fib", True),
            acquire_image1=settings.get("acquire_image1", True),
            acquire_image2=settings.get("acquire_image2", True),
        )

    @property
    def field_of_views(self) -> Tuple[float, ...]:
        """Returns a tuple of the selected field of views, sorted from largest to smallest."""
        fovs = []
        if self.acquire_image1:
            fovs.append(self.field_of_view1)
        if self.acquire_image2:
            fovs.append(self.field_of_view2)
        return tuple(sorted(fovs, reverse=True))  # largest to smallest

    @property
    def estimated_time(self) -> float:
        n_fovs = sum([self.acquire_image1, self.acquire_image2])
        n_beams = sum([self.acquire_sem, self.acquire_fib])
        return self.imaging.estimated_time * n_fovs * n_beams
