# fibsem structures
from __future__ import annotations

import json
import logging
import os
import weakref
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass, field, fields
from datetime import datetime, tzinfo
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
    NamedTuple,
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
import yaml
from numpy.typing import NDArray

import fibsem
from fibsem.config import (
    METADATA_VERSION,
    SUPPORTED_COORDINATE_SYSTEMS,
    UNVERSIONED_METADATA,
)
from fibsem.manufacturers import DEMO, normalize_manufacturer
from fibsem.util.timestamps import (
    acquisition_datetime_of,
    now,
    to_aware,
    to_datetime,
    to_iso,
    utc_offset_of,
    zone_from_offset,
)
from fibsem.versioning import get_revision

if TYPE_CHECKING:
    from fibsem.autofunctions.autofocus import AutoFocusSettings
    from fibsem.fm.structures import FluorescenceConfiguration
    from fibsem.microscope import FibsemMicroscope

TFibsemPatternSettings = TypeVar(
    "TFibsemPatternSettings", bound="FibsemPatternSettings"
)

DEFAULT_FIELD_METADATA: Dict[str, Any] = {
    "label": None,  # the display label for the field
    "type": None,  # the data type of the field
    "unit": None,  # the field's unit; shown with the scale's SI prefix (m + 1e6 -> µm)
    "display_unit": None,  # the shown unit, when the scale isn't an SI prefix (° , %)
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
    display_unit: Optional[str] = None,
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
        merged_metadata = {
            **default_metadata,
            **_beam_parameter_display(declared.get("microscope_parameter")),
            **declared,
        }
        field_metadata[f.name] = merged_metadata
    return field_metadata


def _beam_parameter_display(name: Optional[str]) -> Dict[str, Any]:
    """How the beam shows its parameter ``name``, for a field bound to it: the unit,
    scale, step and decimals a field leaves out are the beam's, so a milling current
    reads like the beam panel's current. Empty for a name the beam doesn't declare.
    """
    if not name:
        return {}
    from fibsem.devices.beam import Beam
    from fibsem.devices.core import Parameter

    parameter = getattr(Beam, name, None)
    if not isinstance(parameter, Parameter) or parameter.display is None:
        return {}
    metadata = parameter.display.as_field_metadata(parameter.unit)
    # the field names it in its own context, and says where its form shows it
    metadata.pop("label", None)
    metadata.pop("advanced", None)
    return metadata


class Resolution(NamedTuple):
    """An image size in pixels. A tuple, so it compares equal to a plain
    ``(width, height)`` and goes wherever one does; it adds the "WxH" spelling the
    microscopes and the widgets use."""

    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @classmethod
    def parse(cls, text: str) -> "Resolution":
        """Read "1536x1024" (spaces and a capital X allowed)."""
        try:
            width, height = text.lower().replace(" ", "").split("x")
            return cls(int(width), int(height))
        except (AttributeError, ValueError):
            raise ValueError(
                f"A resolution is WxH, e.g. '1536x1024', got {text!r}"
            ) from None

    @property
    def aspect(self) -> float:
        """Width over height: 1.5 for 1536x1024."""
        return self.width / self.height


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
    # The state could not be read. Not an error and not a guess -- a producer saying so
    # deliberately, because on ThermoFisher `get_milling_state()` is a getter that
    # *sets the active view* as a side effect, and a caller that must not disturb the
    # view has no way to ask. The coincidence milling strategy is exactly that caller:
    # it runs a fluorescence acquisition that holds the active view for the whole
    # strategy, so polling the milling state mid-strategy would yank the view out from
    # under it.
    #
    # A real member rather than `None` or the string `"UNKNOWN"`, so a consumer can
    # render or ignore it deliberately instead of pattern-matching a magic value.
    UNKNOWN = 5


# Milling is under way, in the sense that the caller should keep waiting. Read as a loop
# condition -- `while get_milling_state() in ACTIVE_MILLING_STATES` guards the milling
# waits and TESCAN's spot-burn poll.
#
# `UNKNOWN` is deliberately **not** in here, and both classifications have a failure
# mode: excluded, a genuinely-running mill that reported `UNKNOWN` makes the loop exit
# early and the caller proceeds as though milling finished; included, a mill that has
# actually stopped spins forever. Exiting early is recoverable and bounded. Spinning is
# not.
ACTIVE_MILLING_STATES = [
    MillingState.RUNNING,
    MillingState.STOPPING,
    MillingState.PAUSED,
]


class ManipulatorState(Enum):
    RETRACTED = 0
    INSERTED = 1
    MOVING = 2


class InsertableDeviceState(Enum):
    """Where a device that goes in and out is: the FM objective, the manipulator. A
    driver that only knows in or out reports those two."""

    RETRACTED = "retracted"
    INSERTED = "inserted"
    MOVING = "moving"
    ERROR = "error"
    UNKNOWN = "unknown"


class ChamberState(Enum):
    """The chamber's vacuum. A vendor state with no match here reads as UNKNOWN."""

    PUMPED = "pumped"
    VENTED = "vented"
    PUMPING = "pumping"
    VENTING = "venting"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_name(cls, name: str) -> "ChamberState":
        """The state an instrument names ("Pumped", "VENTED"), else UNKNOWN."""
        try:
            return cls(str(name).lower())
        except ValueError:
            return cls.UNKNOWN


class ScanMode(Enum):
    """What a beam scans. The values are the ``scanning_mode`` key's."""

    FULL_FRAME = "full_frame"
    REDUCED_AREA = "reduced_area"
    SPOT = "spot"


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


class AutoContrastMode(Enum):
    """When to set the contrast during a tiled acquisition.

    ONCE is one detector setting for the whole mosaic, taken at the grid centre
    before the first tile, so the tiles stitch without seams and a brightness
    difference between tiles is real. EACH_TILE is what `ImageSettings.autocontrast`
    means for any single image, before every tile: it flattens a gradient across
    the grid at the cost of seams where neighbours were scaled differently.
    """

    NONE = "none"
    ONCE = "once"
    EACH_TILE = "each_tile"


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
        See fibsem.drivers.autoscript.microscope for AutoScript conversion utilities (stage_position_to_autoscript, stage_position_from_autoscript).
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
        See fibsem.drivers.autoscript.microscope for AutoScript conversion utilities (manipulator_position_to_autoscript, manipulator_position_from_autoscript).
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
    # How a form shows them. The beam-bound fields take their unit, scale, step and
    # decimals from the beam's parameter, and an instrument's limits win over these.
    resolution: Tuple[int, int] = field(
        default=(1536, 1024),
        metadata=field_meta(label="Resolution", microscope_parameter="resolution"),
    )
    dwell_time: float = field(
        default=1e-6,
        metadata=field_meta(label="Dwell Time", microscope_parameter="dwell_time"),
    )
    hfw: float = field(
        default=150e-6,
        metadata=field_meta(label="Field of View", microscope_parameter="hfw"),
    )
    autocontrast: bool = False
    beam_type: BeamType = BeamType.ELECTRON
    save: bool = False
    filename: str = "default_image"
    path: Optional[Union[Path, str]] = None
    reduced_area: Optional[FibsemRectangle] = None
    # None is off; a form shows it as 1.
    line_integration: Optional[int] = field(
        default=None,
        metadata=field_meta(label="Line Integration", minimum=1, maximum=255),
    )
    scan_interlacing: Optional[int] = field(
        default=None,
        metadata=field_meta(label="Scan Interlacing", minimum=1, maximum=8),
    )
    frame_integration: Optional[int] = field(
        default=None,
        metadata=field_meta(label="Frame Integration", minimum=1, maximum=512),
    )
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


def _default_autofocus_settings() -> "AutoFocusSettings":
    """The overview's focus sweep, which is simply the library default.

    Deliberately not a pinned copy of the values. `AutoFocusSettings()` is two passes --
    50 um at 5 um, then 10 um at 1 um -- and an overview wanting exactly that means it
    should *inherit* it, so a later improvement to the default reaches overviews too
    (FIB-646).

    Imported here rather than at module scope because `autofunctions.autofocus` imports
    this module for `BeamType`, so the arrow only points one way. `structures` already
    reaches downstream this way for `autofunctions.gamma` and `fm.structures`.
    """
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    return AutoFocusSettings()


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
        autofocus_mode: *When* to focus during the traversal (NONE, ONCE, EACH_ROW,
            EACH_TILE).
        autocontrast_mode: *When* to set the contrast (NONE, ONCE at the grid centre,
            EACH_TILE). The runner drives `image_settings.autocontrast` from it, so
            the per-image flag is not the switch here.
        autofocus_settings: *How* to focus -- sweep passes, method and probe frame.
            Two separate questions, and they used to be conflated: this field held a
            `fibsem.structures.AutoFocusSettings` whose only member was the mode, a
            second class sharing a name with the real sweep config in
            `autofunctions.autofocus`. `AutoFocusMode` had already been through exactly
            that (two identical enums, so `NONE is NONE` was False across the two import
            paths, with no type error to catch it), so the duplicate name is gone rather
            than left to bite twice.
    """

    image_settings: ImageSettings = field(default_factory=ImageSettings)
    nrows: int = 3
    ncols: int = 3
    overlap: float = 0.1
    focus_stack_settings: FocusStackSettings = field(default_factory=FocusStackSettings)
    autofocus_mode: AutoFocusMode = AutoFocusMode.NONE
    autofocus_settings: "AutoFocusSettings" = field(
        default_factory=_default_autofocus_settings
    )
    autocontrast_mode: AutoContrastMode = AutoContrastMode.NONE
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

        # Two shapes of the same two facts. Before FIB-646 the mode lived *inside*
        # `autofocus_settings`, in a class that held nothing else:
        #
        #     {"autofocus_settings": {"mode": "EACH_ROW"}}          <- old
        #     {"autofocus_mode": "EACH_ROW",                        <- new
        #      "autofocus_settings": {"method": ..., "passes": [...]}}
        #
        # `"mode"` is a safe discriminator and not a heuristic: the real sweep config
        # writes method / passes / probe_resolution / probe_dwell_time / reduced_area /
        # use_autocontrast / channel_name, and never a "mode" key. So an old file is
        # recognised by the one key the new shape cannot produce.
        #
        # An old file carries no sweep at all, so it gets the default -- which is the
        # right answer rather than a fallback: it is exactly what it was running under,
        # since the mode was the only thing it could configure.
        raw_autofocus = d.get("autofocus_settings") or {}
        if "mode" in raw_autofocus:
            mode = AutoFocusMode(raw_autofocus["mode"])
            sweep = _default_autofocus_settings()
        else:
            mode = AutoFocusMode(d.get("autofocus_mode", AutoFocusMode.NONE.value))
            if raw_autofocus:
                from fibsem.autofunctions.autofocus import AutoFocusSettings

                sweep = AutoFocusSettings.from_dict(raw_autofocus)
            else:
                sweep = _default_autofocus_settings()

        image_settings = ImageSettings.from_dict(d.get("image_settings", {}))
        # A file from before the mode existed said "auto contrast" with the
        # per-image flag alone. It meant the mosaic, not each tile: ONCE.
        if "autocontrast_mode" in d:
            autocontrast_mode = AutoContrastMode(d["autocontrast_mode"])
        elif image_settings.autocontrast:
            autocontrast_mode = AutoContrastMode.ONCE
        else:
            autocontrast_mode = AutoContrastMode.NONE

        return OverviewAcquisitionSettings(
            image_settings=image_settings,
            nrows=d.get("nrows", 3),
            ncols=d.get("ncols", 3),
            overlap=d.get("overlap", 0.1),
            focus_stack_settings=fss,
            autofocus_mode=mode,
            autofocus_settings=sweep,
            autocontrast_mode=autocontrast_mode,
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
            "autofocus_mode": self.autofocus_mode.value,
            "autofocus_settings": self.autofocus_settings.to_dict(),
            "autocontrast_mode": self.autocontrast_mode.value,
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
            beam_type=BeamType[state_dict.get("beam_type", "ELECTRON").upper()],
            working_distance=wd,
            beam_current=current,
            voltage=state_dict.get("voltage"),
            hfw=state_dict.get("hfw"),
            resolution=state_dict.get("resolution"),
            dwell_time=state_dict.get("dwell_time"),
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
            "label": "Milling Current",
            "type": float,
            "items": "dynamic",
            "microscope_parameter": "current",
            "tooltip": "The current used for milling. Higher currents mill faster but with less precision and more damage.",
        },
    )
    milling_voltage: float = field(
        default=30e3,
        metadata={
            "label": "Milling Voltage",
            "type": float,
            "items": "dynamic",
            "microscope_parameter": "voltage",
            "advanced": True,
            "tooltip": "The voltage used for milling. Higher voltages provide higher energy ions for milling.",
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
            "default": 150.0,
            "minimum": 20.0,
            "maximum": 950.0,
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


# The axes a device is allowed to constrain.
#
# Linear only, and that is the model rather than a shortcut: a device is a *place*,
# and the pose the sample is held in once the stage is there is the orientation, which
# is the other axis of this model entirely. A device that also fixed r or t would be
# the same conflation this is here to undo. It also sidesteps a units question --
# x/y/z are metres in the configuration, while `rotation_reference` and
# `shuttle_pre_tilt` are degrees, converted at use.
DEVICE_AXES = ("x", "y", "z")

# The named orientations the microscope derives poses for -- see
# `_update_orientations`. The poses themselves are computed from physical parameters
# (pre-tilt, column tilt, milling angle); these are the only names a device's
# `available_orientations` may reference.
KNOWN_ORIENTATIONS = ("SEM", "FIB", "MILLING", "FM")

# The orientations the beams image from, which is also a device's default when it
# states none and its type has no default of its own.
BEAM_ORIENTATIONS = ("SEM", "FIB", "MILLING")


class DeviceImagingState(Enum):
    """Can this device see the sample from where the stage is -- and if not, why not.

    The answer to `FibsemMicroscope.get_device_imaging_state`, and a state rather than
    a bool because the *reason* is the useful part: each failing value names the remedy
    a caller should offer, and on either mounting the geometry makes the right one fall
    out of the same two questions (FIB-839).

    Callers act on it by policy, not uniformly. Acquisition gates refuse `NO_DEVICE`
    and the travel states but permit `NEEDS_REPOSE` -- acquiring from a "wrong" pose is
    a harmless watch when the device is here, and a different place in the chamber when
    it is not. That single policy is what preserves the compustage's acquire-anywhere
    behaviour (it can never need travel) while refusing at the beams on an offset mount
    (it always does) -- with no flag and no per-mounting branch. Planning and
    move-prompt sites require `READY` strictly.
    """

    # The instrument behind the device is absent: nothing to travel to, nothing to
    # offer. Terminal -- distinct from every other value, which all mean "it exists
    # and here is how to reach it".
    NO_DEVICE = "no_device"

    READY = "ready"

    # Right pose, wrong place: traverse to the device.
    NEEDS_TRAVEL = "needs_travel"

    # Right place, wrong pose: re-pose. On an offset mount the route is longer --
    # back to the beams, re-pose there, travel out again -- because rotating while
    # parked at the device is refused (FIB-841).
    NEEDS_REPOSE = "needs_repose"

    # Wrong on both axes. Re-pose first, then travel: the bracketing order, so the
    # rotation happens at the beams and never under an objective.
    NEEDS_REPOSE_THEN_TRAVEL = "needs_repose_then_travel"

    @property
    def allows_acquisition(self) -> bool:
        """The acquisition-gate policy, in one place.

        Acquiring in place from a "wrong" pose is a harmless watch when the device is
        here -- an Arctis user turning the light on at a beam pose -- so a re-pose
        does not refuse. Travel states do: from a different place in the chamber the
        device is not looking at the sample at all. `NO_DEVICE` refuses, terminally.

        Sites that *drive the stage* through an FM-built frame, or *write the pose
        down* (marking a lamella, a tileset walking a grid), require `READY` and do
        not use this: from a pose the device cannot image from, nothing has checked
        the frame their numbers go through.
        """
        return self in (DeviceImagingState.READY, DeviceImagingState.NEEDS_REPOSE)


def device_axes_to_dict(position: FibsemStagePosition) -> dict:
    """The device axes a partial position sets, as a plain dict. Absent axes are absent."""
    return {
        axis: getattr(position, axis)
        for axis in DEVICE_AXES
        if getattr(position, axis) is not None
    }


def device_axes_from_dict(axes: dict, what: str) -> FibsemStagePosition:
    """A partial position from device axes, refusing anything that is not one."""
    axes = axes or {}
    unknown = set(axes) - set(DEVICE_AXES)
    if unknown:
        raise ValueError(
            f"Unsupported {what} axes: {sorted(unknown)}. Supported: {list(DEVICE_AXES)}."
        )
    return FibsemStagePosition(**{axis: float(value) for axis, value in axes.items()})


@dataclass
class StageDeviceSettings:
    """Where the stage travels for one instrument to see the sample.

    An offset fluorescence microscope is not under the grid; the stage travels to it.
    So the FM is a *place* on the stage rather than a pose the stage is held in, and
    `origin` is that place, in stage coordinates.

    It is a reference point, not a destination. The stage does not land on it: it
    travels by the difference between two origins and arrives wherever that puts it.
    Partial, too -- an offset FM is an x location and leaves y, z, r and t free, so an
    axis that is absent does not decide anything.

    A device is therefore described along **both** axes: `origin` says where the stage
    goes, `available_orientations` says which poses the instrument can see the sample
    in once it is there. Neither answers on its own -- see `FibsemMicroscope` and
    FIB-839 -- and which of the two does the discriminating is a fact about the
    mounting rather than about the code:

    | | origin | available_orientations |
    | -- | -- | -- |
    | compustage | shared with the beams, so the term is true everywhere | `["FM"]` -- carries it |
    | offset mount | 48.8 mm away -- carries it | `["FIB"]`, true wherever the objective reaches |

    So the same conjunction discriminates on both, and the term that *fails* names the
    remedy: a wrong place means travel, a wrong pose means re-pose.
    """

    origin: FibsemStagePosition

    # Named orientations, not poses. The poses themselves stay derived in code from
    # physical parameters -- pre-tilt, column tilt, milling angle -- so a site cannot
    # write one that contradicts its own geometry (Patrick, 2026-08-31: "just ship with
    # code only orientations"). Which *named* orientations an instrument can image
    # from is a different kind of fact: it is how the device is bolted on, nothing
    # derives it, and a list of names can only reference the derived poses, never
    # disagree with them.
    #
    # Always stated, never empty: an empty list is refused where configuration enters
    # (`from_dict`), because "this device can image from nowhere" would turn a
    # forgotten key into a silently dead instrument. A device that does not say gets
    # its type's default (`DEFAULT_AVAILABLE_ORIENTATIONS`) -- the beams image from
    # SEM, FIB and MILLING.
    available_orientations: List[str] = field(
        default_factory=lambda: list(BEAM_ORIENTATIONS)
    )

    # The region belonging to this device, as a half-width per axis from its origin.
    # `None` is unbounded: the beams, which the stage is at wherever no other device
    # claims it. A configured device that states none gets 1 mm on each axis its
    # origin sets (`default_device_range`).
    range: Optional[FibsemStagePosition] = None

    def contains(self, stage_position: FibsemStagePosition) -> bool:
        """Is `stage_position` within this device's `range` of its origin?

        Each device has its own range, so a traverse is no longer invertible by
        construction: the stage travels by the *difference* between two origins and
        keeps its offset, so a grid position 15 mm along at the beams arrives 15 mm
        along at the FM, which may not count that as arrived. So a traverse checks
        both ends -- the source's range at the start, the target's for where it will
        arrive -- before it moves (`FibsemMicroscope.move_to_device`).

        Not the same question as whether the device usefully *covers* the sample
        here -- an objective's field, a knife's approach -- which genuinely does vary
        per device. Nothing needs that yet; see FIB-839.

        Devices may overlap, and on a compustage they fully do: the objective is under
        the grid, so the beams and the FM are the same place reached by flipping. The
        device axis is degenerate there and this question is not the one to ask -- see
        `FibsemMicroscope.get_current_device`.

        A device whose origin constrains an axis that `range` says nothing
        about answers **no**. The caller is "have I already arrived", and the two costs are
        not symmetric: a wrong `False` costs a move that was not needed, a wrong
        `True` skips one that was.
        """
        constrained = [
            axis for axis in DEVICE_AXES if getattr(self.origin, axis) is not None
        ]
        if not constrained:
            return False

        for axis in constrained:
            value = getattr(stage_position, axis)
            if value is None:
                return False
            if self.range is None:
                continue
            extent = getattr(self.range, axis)
            if extent is None:
                return False
            if abs(value - getattr(self.origin, axis)) > extent:
                return False
        return True

    def to_dict(self) -> dict:
        """The keys a device entry carries for its stage position."""
        ddict = {
            "origin": device_axes_to_dict(self.origin),
            "available_orientations": list(self.available_orientations),
        }
        if self.range is not None:
            ddict["range"] = device_axes_to_dict(self.range)
        return ddict

    @staticmethod
    def from_dict(
        ddict: dict,
        default_orientations: Sequence[str] = (),
        default_range: Optional[FibsemStagePosition] = None,
    ) -> "StageDeviceSettings":
        """Read a device's stage position.

        `available_orientations` missing is the device type's default; stated empty is
        an error. The configuration version 1 spelling, `acquisition_orientations`, is
        read too, and there an empty list meant "any orientation", so it reads as all
        of them. `range` missing is *default_range*, or 1 mm on each axis the origin
        sets.
        """
        origin = device_axes_from_dict(ddict.get("origin"), "device origin")
        if "available_orientations" in ddict:
            orientations = [str(o) for o in ddict["available_orientations"] or []]
            if not orientations:
                raise ValueError(
                    "available_orientations is empty: a device that can image from no "
                    f"orientation. Name at least one of {list(KNOWN_ORIENTATIONS)}, "
                    "or leave the key out for the device's default."
                )
        elif "acquisition_orientations" in ddict:
            orientations = [str(o) for o in ddict["acquisition_orientations"] or []]
            orientations = orientations or list(KNOWN_ORIENTATIONS)
        else:
            orientations = list(default_orientations) or list(BEAM_ORIENTATIONS)
        # Validated here, at the one place configuration enters, because a typo would
        # otherwise be perfectly quiet: the conjunction that reads this list would
        # simply never be true, and the instrument would be dead with no error.
        unknown = [o for o in orientations if o not in KNOWN_ORIENTATIONS]
        if unknown:
            raise ValueError(
                f"Unknown orientation(s) {unknown}. "
                f"Known orientations: {list(KNOWN_ORIENTATIONS)}"
            )
        if ddict.get("range"):
            range_ = device_axes_from_dict(ddict["range"], "device range")
        elif default_range is not None:
            range_ = deepcopy(default_range)
        else:
            range_ = default_device_range(origin)
        # A range covers every axis the origin sets; one it leaves out gets the
        # default. Otherwise `contains` answers no on that axis, and the device -- a
        # version 1 `{x: 20 mm}` copied onto an origin that also sets y -- could never
        # be arrived at.
        for axis in DEVICE_AXES:
            if getattr(origin, axis) is not None and getattr(range_, axis) is None:
                setattr(range_, axis, DEFAULT_DEVICE_RANGE_EXTENT)
        return StageDeviceSettings(
            origin=origin, available_orientations=orientations, range=range_
        )


# What a configured device's range is when it states none: 1 mm on each axis its origin
# sets.
DEFAULT_DEVICE_RANGE_EXTENT: float = 1.0e-3


def default_device_range(origin: FibsemStagePosition) -> FibsemStagePosition:
    """1 mm on each axis *origin* constrains, and nothing on the rest."""
    return FibsemStagePosition(
        **{
            axis: DEFAULT_DEVICE_RANGE_EXTENT
            for axis in DEVICE_AXES
            if getattr(origin, axis) is not None
        }
    )


# The range every device shared in configuration version 1 when the file stated none.
# A version 1 file is read with its `device_range` (or this) copied onto each device,
# so an existing site keeps the window it had; the 1 mm default is for new entries.
VERSION_1_DEVICE_RANGE = FibsemStagePosition(x=20.0e-3)

# The ion column's angle from the electron column, in degrees. A property of the
# instrument rather than a preference, and the same on every dual-beam this supports,
# which is why a file that omits it can still be read correctly. Declared once because
# it is read in two places -- the config reader and the geometry recorded on an image
# -- and they must not be able to disagree about it.
DEFAULT_FIB_COLUMN_TILT: float = 52.0

# The version of the microscope configuration file format. Written by
# `MicroscopeSettings.to_dict` and stated by every shipped file. Nothing branches on
# it yet: it exists because a format change cannot migrate a file that does not say
# what format it is, and the field cannot be added retrospectively -- a file without
# it is indistinguishable from one written before it existed.
#
# 2: `hardware:` is a list of devices (`hardware.devices`) instead of one block per
# device. Version 1 files load unchanged; the reader takes both.
CONFIGURATION_VERSION: int = 2

# **The default is the objective under the grid**: the FM shares the beams' origin, and
# is told apart by the pose the sample is held in. A site whose objective is offset --
# piescope, METEOR, iFLM, all in the TFS SDB chamber -- declares the traverse instead,
# as `sim-iflm-configuration.yaml` does.
#
# The default used to be the other way round, with the FM 48.8 mm along x, because
# these entries were lifted from `move_to_microscope`'s inlined `TRANSLATION_DX` and
# that function only ever ran on an offset mount. Nothing declares a `devices:` block
# except the offset simulator, so every other configuration -- Aquilos, Hydra, Arctis,
# Tescan, Odemis, several with no fluorescence microscope at all -- inherited a phantom
# FM 48.8 mm away, somewhere their stage never goes. It was invisible because
# `_device_translation` used to short-circuit on a compustage; `contains` did not, and
# read these origins literally. A compustage FM's origin is now travelled by after the
# flip: the offset of the objective from the beams' coincidence point, at the FM pose.
#
# Getting this the right way round is what lets one question be asked of both mountings
# instead of each caller branching on the stage type (FIB-839).
# The beams' origin: the stage's zero on x and y. z is left free -- the stage sits at
# its working distance there, not at zero -- so a device origin's z is checked but
# never travelled along until the beams' working z is configured too. An axis a
# device's origin leaves unset is the beams'.
BEAMS_ORIGIN = FibsemStagePosition(x=0.0, y=0.0)

DEFAULT_STAGE_DEVICES: Dict[str, StageDeviceSettings] = {
    # The beams: the implicit zero, imaging from SEM, FIB and MILLING, and unbounded --
    # the stage is at the beams wherever no other device claims it. Never configured;
    # no device entry describes them.
    "FIBSEM": StageDeviceSettings(origin=deepcopy(BEAMS_ORIGIN)),
    "FM": StageDeviceSettings(
        origin=FibsemStagePosition(x=0.0),
        available_orientations=["FM"],
        range=default_device_range(FibsemStagePosition(x=0.0)),
    ),
}

# The stage-device key for the beams, which no configuration entry describes.
BEAMS_STAGE_DEVICE = "FIBSEM"

# A device type's orientations when its entry states none. Any other type gets the
# beams' (`BEAM_ORIENTATIONS`).
DEFAULT_AVAILABLE_ORIENTATIONS: Dict[str, Tuple[str, ...]] = {"fm": ("FM",)}


def stage_device_key(entry_name: str) -> str:
    """The `StageSystemSettings.devices` key for a device entry's stage position.

    The FM was keyed `FM` before devices had entries, and `move_to_device("FM")` and
    its callers still use that; any other device is keyed by its entry name.
    """
    return "FM" if entry_name == "fm" else entry_name


def stage_device_entry_name(key: str) -> str:
    """The device entry a `StageSystemSettings.devices` key is written onto."""
    return "fm" if key == "FM" else key


# Where a half turn of the stage is centred, in raw stage coordinates (x, y), metres:
# a position p recorded on one side of the stage is at 2c - p on the other. The centre
# reprojection used before drivers reported one, calibrated on one ThermoFisher
# instrument (FIB-655); kept for images that recorded no centre. ThermoFisher now
# reports xT's centre plus `rotation_centre_correction`.
LEGACY_ROTATION_CENTRE: Tuple[float, float] = (
    -0.0005127403888932854 + 25e-6,
    0.0007937916666666666 + 12.5e-6,
)


# The frame a stage's positions are in (FIB-1114). Every stage reports fibsem's frame
# except Tescan's, which still reports its own: x and y run opposite the image, y rides
# on the tilt module and z is chamber-vertical, +z down.
STAGE_FRAME_FIBSEM = "fibsem"
STAGE_FRAME_TESCAN = "tescan_native"


def _parse_rotation_centre(value) -> Optional[Tuple[float, float]]:
    """A stored rotation centre as an (x, y) tuple of floats, or None."""
    if value is None:
        return None
    x, y = value
    return (float(x), float(y))


@dataclass
class StageSystemSettings:
    rotation_reference: float
    # Accepted as a constructor keyword, held as `_shuttle_pre_tilt`, and read back
    # through the property below. An `InitVar` rather than a field because the value
    # a caller passes is a *fallback* -- the answer comes from the active holder when
    # there is one.
    shuttle_pre_tilt: InitVar[float] = 0.0
    # The fallback the property reads while no holder answers. A real field rather
    # than a bare attribute set in `__post_init__`, so that `__eq__` and `__repr__`
    # see it: two stages at 0 and 35 degrees must not compare equal, and a
    # round-trip test that compares records must be able to notice a pre-tilt
    # that was dropped on the way through the file.
    _shuttle_pre_tilt: float = field(init=False, default=0.0)
    enabled: bool = True
    # Whether the stage has a rotation axis. Load-bearing: it is what `rotation_180`
    # below is derived from, so it describes the geometry and not merely a permission.
    #
    # **Written by the instrument, not by a file.** No configuration states it any
    # more: `FibsemMicroscope._read_stage_capabilities` fills it at connect from the
    # stage's own axes, and again after `apply_configuration` replaces this record.
    # The default here is what an unconnected `SystemSettings` holds, and it is the
    # rotating case because that is the commoner stage -- a compustage that never
    # reached a microscope has no orientations to get wrong.
    #
    # There was a `tilt` beside it and there is not any more. Nothing read it, and
    # every shipped file said `true` because every file always would have:
    # `_get_axis_limits` returns a `t` axis on every backend, compustage included.
    # A flag with one reachable value is not a capability, it is a place for a typo
    # to sit -- which is exactly what `rotation` had become in the simulator's Arctis
    # configuration before it was read by anything.
    rotation: bool = True
    milling_angle: float = 15
    # Where the stage travels for each instrument to see the sample. Keyed by device
    # name -- "FIBSEM" and "FM" today -- and separate from `orientations`, which says
    # what pose the sample is held in once the stage is there.
    #
    # Configured on each device's entry (`origin`, `available_orientations`, `range`)
    # rather than on the stage; `SystemSettings.from_dict` fills this from them. Each
    # device's `range` is how far from it the stage can be and still count as having
    # travelled to it. Not to be confused with `microscope._stage.limits`, which is
    # how far the axes can physically move.
    devices: Dict[str, StageDeviceSettings] = field(
        default_factory=lambda: deepcopy(DEFAULT_STAGE_DEVICES)
    )
    # The holders this system has, and which one is on the stage. A keyed map with a
    # selection rather than a single holder, because a site that swaps a flat shuttle
    # for a pre-tilted one should select the other entry rather than re-enter its
    # geometry -- and, once pre-tilt moves onto the holder, re-calibrate for it.
    #
    # Empty by default. `_create_sample_stage` fills it, importing a `sample-holder.yaml`
    # if the site has one; nothing here reads that file, so a `SystemSettings` built
    # from a dict stays a pure function of that dict.
    holders: Dict[str, "SampleHolder"] = field(default_factory=dict)
    active_holder: str = ""
    # How far the true centre of a half turn sits from where the vendor's compucentric
    # rotation puts it, in raw stage coordinates (x, y), metres (FIB-655). Zero means
    # the vendor's rotation is taken as perfect. Measured per instrument: centre a
    # feature, rotate, centre it again, and the midpoint of the two positions less the
    # vendor's centre is this value. Stage moves and image reprojection both add it.
    rotation_centre_correction: Tuple[float, float] = (0.0, 0.0)

    @property
    def rotation_180(self) -> float:
        """Where the stage sits to face the ion beam, in degrees.

        Derived, not configured. It used to be a field, and every shipped file gave it
        `(rotation_reference + 180) % 360` -- Tescan included, whose reference of 180
        is what makes the modulo load-bearing rather than decorative. The two
        exceptions were the compustages, which set it *equal* to the reference to say
        "this stage does not turn round". That is a boolean's job, and `rotation` is
        the boolean -- already on this class, and already `false` on the real Arctis
        (FIB-834).

        So a value that was never chosen is now computed, and the one case it could not
        express without a coincidence -- a stage with no rotation axis -- is asked of
        the field named for it.

        Readers move to the whole FIB pose (FIB-1101): a rotation alone does not say
        where a stage faces the ion beam, which a stage can reach by tilting instead.
        ``microscope.get_orientation("FIB")``, or an image's
        ``hardware_geometry.declared_poses()["FIB"]``.
        """
        return self._fib_rotation()

    def _fib_rotation(self) -> float:
        """The FIB pose's rotation by the configured rule, in degrees.

        What images stamp as ``rotation_180``, which old readers of an image still use.
        """
        if not self.rotation:
            return self.rotation_reference
        return (self.rotation_reference + 180) % 360

    def __post_init__(self, shuttle_pre_tilt: float) -> None:
        self._shuttle_pre_tilt = float(shuttle_pre_tilt)

    @property
    def shuttle_pre_tilt(self) -> float:
        """The pre-tilt of the shuttle on the stage, in degrees.

        The holder answers when there is one that says. Physically correct: the
        pre-tilt is a property of the shuttle, not of the stage it sits on, so
        swapping a 35 degree shuttle for a flat one should change it -- and today
        that means editing the stage block by hand, where forgetting silently wrongs
        every projection.

        The fallback is not decoration. A `StageSystemSettings` built from a
        configuration has no holder until `_create_sample_stage` resolves one, and
        every holder file written before this carries no pre-tilt. Returning 0.0 in
        either case would turn a 35 degree site flat, which is the one outcome this
        change must not produce. So the configured value stands until a holder
        states otherwise.
        """
        holder = self.holders.get(self.active_holder)
        if holder is not None:
            return holder.pre_tilt
        return self._shuttle_pre_tilt

    @shuttle_pre_tilt.setter
    def shuttle_pre_tilt(self, value: float) -> None:
        """Setting it sets the active holder's, which is what it means.

        A setter rather than a read-only property because around twenty-five test
        files use `microscope.system.stage.shuttle_pre_tilt = 35` as their setup
        idiom, and because it reads correctly: the stage's pre-tilt *is* whatever
        holder is on it, so changing one is changing the other. The fallback is
        written too, so the two cannot drift apart through this path.
        """
        value = float(value)
        self._shuttle_pre_tilt = value
        holder = self.holders.get(self.active_holder)
        if holder is not None:
            holder.pre_tilt = value

    def to_dict(self):
        ddict = {
            "rotation_reference": self.rotation_reference,
            "enabled": self.enabled,
            "rotation": self.rotation,
            "milling_angle": self.milling_angle,
            # `include_grids=False`: which grid is in which slot is session state and
            # has its own file. Writing it here would make the configuration go stale
            # every time someone swapped a grid.
            "holders": {
                name: holder.to_dict(include_grids=False)
                for name, holder in self.holders.items()
            },
            "active_holder": self.active_holder,
        }
        # Written only once measured, so a file that never calibrated it stays as it was.
        if any(self.rotation_centre_correction):
            ddict["rotation_centre_correction"] = list(self.rotation_centre_correction)
        # The pre-tilt has one home in the file. Once a holder is named it lives on
        # the holder, and writing it here as well would be a second copy that a hand
        # edit could put out of step -- silently, in the term every projection uses.
        # Until then (a record loaded from an old file and not yet connected) the
        # stage-level key is the only place the value has, so it is kept.
        if not self.holders:
            ddict["shuttle_pre_tilt"] = self.shuttle_pre_tilt
        return ddict

    @staticmethod
    def from_dict(settings: dict):
        # `rotation_180` is deliberately not read. A file written before FIB-834 still
        # carries the key and still loads -- the value is simply ignored, because it is
        # now derived from the two fields that decide it. Ignoring beats honouring: a
        # stored value that disagrees with the derivation is a value someone typed
        # wrong, and reading it back would preserve the mistake.
        holders = {
            name: _configured_holder_from(name, holder)
            for name, holder in (settings.get("holders") or {}).items()
        }
        active_holder = settings.get("active_holder", "")
        # Once a holder is named the file states the pre-tilt only on the holder,
        # so the stage's fallback is seeded from it. Left at 0.0 it changed nothing
        # anyone reads -- the holder answers -- but a record no longer equalled its
        # own round trip, which is exactly what the round-trip tests compare.
        active = holders.get(active_holder)
        fallback = settings.get(
            "shuttle_pre_tilt", active.pre_tilt if active is not None else 0.0
        )
        return StageSystemSettings(
            rotation_reference=settings.get("rotation_reference", 0.0),
            shuttle_pre_tilt=fallback,
            enabled=settings.get("enabled", True),
            rotation=settings.get("rotation", True),
            milling_angle=settings.get("milling_angle", 15.0),
            devices=_version_1_stage_devices(settings),
            holders=holders,
            active_holder=active_holder,
            rotation_centre_correction=_parse_rotation_centre(
                settings.get("rotation_centre_correction")
            )
            or (0.0, 0.0),
        )


def _version_1_stage_devices(stage: dict) -> Dict[str, StageDeviceSettings]:
    """The device positions a version 1 `stage:` block declares, or the defaults.

    Version 1 kept them under `stage.devices`, keyed `FIBSEM` and `FM`, with one
    `device_range` for all of them. Each declared device gets that range (or the 20 mm
    version 1 used when the file stated none), so an existing site keeps the window it
    had. The beams are the implicit zero and unbounded whatever the file said; a
    version 1 origin for them away from zero is not kept, and is warned about.
    """
    devices = stage.get("devices")
    result = deepcopy(DEFAULT_STAGE_DEVICES)
    if not devices:
        return result
    range_ = (
        device_axes_from_dict(stage["device_range"], "device range")
        if stage.get("device_range")
        else deepcopy(VERSION_1_DEVICE_RANGE)
    )
    for key, device in devices.items():
        if key == BEAMS_STAGE_DEVICE:
            origin = device_axes_from_dict(
                (device or {}).get("origin"), "device origin"
            )
            if any(getattr(origin, axis) for axis in DEVICE_AXES):
                logging.warning(
                    f"stage.devices.{key}.origin {device_axes_to_dict(origin)} is not "
                    "kept: the beams are the stage's zero."
                )
            continue
        result[key] = StageDeviceSettings.from_dict(
            device or {},
            default_orientations=DEFAULT_AVAILABLE_ORIENTATIONS.get(
                stage_device_entry_name(key), ()
            ),
            default_range=range_,
        )
    # A version 1 file that declared devices declared all of them: one it left out
    # did not exist there.
    for key in list(result):
        if key != BEAMS_STAGE_DEVICE and key not in devices:
            del result[key]
    return result


def _detector_block_from(settings: dict) -> dict:
    """The detector keys of a beam block, in the names `FibsemDetectorSettings` reads.

    The prefixed spelling wins when both are present, because it is the one the
    writer produces and the one every shipped file uses.
    """
    block = {}
    for name in ("type", "mode", "brightness", "contrast"):
        if f"detector_{name}" in settings:
            block[name] = settings[f"detector_{name}"]
        elif name in settings:
            block[name] = settings[name]
    return block


# Written by `BeamSettings` / `FibsemDetectorSettings` and not defaults: the column's
# alignment (working distance, stigmation, beam shift) and what the last autocontrast
# left on the detector. A configuration does not record them, and Apply does not set
# them (`FibsemMicroscope.set_beam_system_settings`).
NOT_BEAM_DEFAULTS = (
    "working_distance",
    "stigmation",
    "shift",
    "detector_brightness",
    "detector_contrast",
)

# Written by `ImageSettings` and not defaults: where this session saves its images,
# and the reduced area of the last acquisition.
NOT_IMAGING_DEFAULTS = ("path", "filename", "reduced_area")


def _split_defaults(beam: dict) -> dict:
    """Move the session defaults out of a written beam block, in place.

    Returns the keys that went. What stays is the hardware description. Alignment
    state is dropped from both.
    """
    moved = {
        k: beam.pop(k) for k in list(beam) if k not in SystemSettings.HARDWARE_BEAM_KEYS
    }
    for key in NOT_BEAM_DEFAULTS:
        moved.pop(key, None)
    return moved


def _configured_holder_from(name: str, data: dict) -> "SampleHolder":
    """A holder entry in `stage.holders`, which must state its pre-tilt.

    `SampleHolder.from_dict` reads a silent file as 0.0, and that is safe for a
    `sample-holder.yaml` because `_resolve_configured_holder` overwrites it with the
    configured value before use. A holder *in the configuration* gets no such
    overwrite -- it is the configured value -- so silence here would turn a 35
    degree shuttle flat with nothing to report. It is an error instead.
    """
    if (data or {}).get("pre_tilt") is None:
        raise ValueError(
            f"stage.holders.{name} states no pre_tilt. Every holder in the "
            "configuration must say its pre-tilt in degrees (0 for a flat shuttle)."
        )
    return SampleHolder.from_dict(data)


@dataclass
class BeamSystemSettings:
    beam_type: BeamType
    enabled: bool
    beam: BeamSettings
    detector: FibsemDetectorSettings
    eucentric_height: float
    column_tilt: float
    # The plasma source's gas, or None for a column with no plasma source. One value
    # for one fact: there used to be a `plasma: bool` beside this, and the pair could
    # disagree -- `plasma: true` with no gas, or a gas with `plasma: false` -- and
    # every shipped file spelt "no gas" as the YAML *string* "None".
    plasma_gas: Optional[str] = None

    @property
    def plasma(self) -> bool:
        """Whether this is a plasma column. Derived: a plasma column has a gas."""
        return self.plasma_gas is not None

    def to_dict(self):
        ddict = {
            "beam_type": self.beam_type.value,
            "enabled": self.enabled,
            "eucentric_height": self.eucentric_height,
            "column_tilt": self.column_tilt,
        }
        # Written for the ion column only. One class serves both columns, so the
        # fields exist on the electron one too, and writing them there put
        # `electron.plasma` into any configuration saved from the application --
        # a key no shipped file has ever carried and nothing reads. An electron
        # column has no plasma source.
        if self.beam_type is BeamType.ION:
            ddict["plasma_gas"] = self.plasma_gas
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
        beam_type = BeamType[settings.get("beam_type", "ELECTRON")]

        # The default depends on which column this is, so it cannot be a single
        # number: an absent electron column tilt is 0, an absent ion column tilt is
        # not. Taken from `FibsemHardwareGeometry`, which already declares both --
        # the alternative is two independent defaults for one physical constant,
        # which is how a config missing its `ion:` block came to load with a 52
        # degree column recorded as 0.
        default_column_tilt = (
            DEFAULT_FIB_COLUMN_TILT if beam_type is BeamType.ION else 0.0
        )

        return BeamSystemSettings(
            beam_type=beam_type,
            enabled=settings.get("enabled", True),
            beam=BeamSettings.from_dict(settings),
            # The file spells the detector keys with a `detector_` prefix -- that is
            # what `to_dict` writes -- and `FibsemDetectorSettings.from_dict` reads
            # the bare names, so for as long as both existed every shipped
            # `detector_type: ETD` loaded as "Unknown", and a saved file lost its
            # detector on the next load. Mapped here, at the one seam where the
            # prefixed spelling meets the record.
            detector=FibsemDetectorSettings.from_dict(_detector_block_from(settings)),
            eucentric_height=settings.get("eucentric_height", 0.0),
            column_tilt=settings.get("column_tilt", default_column_tilt),
            plasma_gas=_plasma_gas_from(settings),
        )


def _plasma_gas_from(settings: dict) -> Optional[str]:
    """The plasma gas a block states, or None for a column without one.

    Reads the old two-key spelling as well as the new one. `plasma: false` means no
    plasma source whatever the gas key says, because the flag was the one the
    drivers consulted. And the shipped files wrote "no gas" as `plasma_gas: None`,
    which YAML reads as the *string* "None", so that spelling (and its lower-case
    and empty cousins) is read as None too.
    """
    if settings.get("plasma") is False:
        return None
    gas = settings.get("plasma_gas")
    if gas is None or str(gas).strip().lower() in ("", "none", "null"):
        if settings.get("plasma") is True:
            # The old flag without a gas. Not a plasma column until the instrument
            # names its gas at connect (`FibsemMicroscope._read_plasma_source`).
            logging.info(
                "The configuration says `plasma: true` but names no plasma gas; the "
                "gas is read from the instrument at connect."
            )
        return None
    return str(gas)


@dataclass
class ManipulatorSystemSettings:
    enabled: bool = True
    # Whether the arm can rotate and tilt. Not in the configuration file: every
    # shipped file said `false`, which is the default, and the axes an arm has are
    # the instrument's to report, not a site's to state. Kept as fields because
    # `is_available("manipulator_rotation")` reads them and a backend that can ask
    # the instrument may set them at connect. Neither written nor read by the
    # file readers below.
    rotation: bool = False
    tilt: bool = False

    def to_dict(self):
        return {"enabled": self.enabled}

    @staticmethod
    def from_dict(settings: dict):
        return ManipulatorSystemSettings(enabled=settings.get("enabled", True))


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
    # The port the driver connects on. None uses the port its driver registers
    # (``fibsem.drivers.registry``): 7520 for ThermoFisher, 8300 for Tescan.
    port: Optional[int] = None

    def to_dict(self):
        ddict = {
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
        # Written only when set: unset is the driver's registered port, and a file
        # that says `port: null` reads as though it chose one.
        if self.port is not None:
            ddict["port"] = self.port
        return ddict

    @staticmethod
    def from_dict(settings: dict):
        return SystemInfo(
            name=settings.get("name", "Unknown"),
            ip_address=settings.get("ip_address", "Unknown"),
            port=settings.get("port"),
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


# The one FM driver a configuration names today; see `FluorescenceSystemSettings.driver`.
FM_DRIVER_REMOTE = "remote"


@dataclass
class FluorescenceSystemSettings:
    """Whether this site's instrument has a fluorescence microscope.

    A hardware fact about the site, so it belongs beside `manipulator.enabled` in the
    microscope configuration rather than in user preferences: a changed preference
    default reaches only fresh installs, and an operator toggling one would be
    asserting their instrument has hardware it may not have.

    **Default off, and it must stay off.** Where the objective is offset, support for
    it is incomplete -- an Aquilos or Helios with an iFLM fitted must not find half
    of it appearing in the UI on upgrade. Detecting the hardware is the failure mode
    here, not the goal: the flag decides, and the driver's own probe only confirms the
    hardware is really there once a site has said it should be.

    **Absent is not false.** `enabled` is `None` when the configuration does not say,
    and then each backend keeps the answer it always gave (`_fluorescence_default`):
    off on an offset mount, on for a compustage and for the Odemis stack, whose
    configurations have never carried the key. An explicit `false` means no FM on
    every backend -- a device switched off in the configuration is never built.
    So never test the raw flag for truth: ask `_fluorescence_is_configured()`.

    The key already existed in the file format and was read by nothing; `config` is
    the only part of the block anything consumed.
    """

    enabled: Optional[bool] = None

    # Which driver the FM comes from. `None` follows the microscope's own driver --
    # the iFLM and the Arctis FM are on the AutoScript connection, the Odemis stack
    # drives its own -- which is every site today. `FM_DRIVER_REMOTE` is an FM on its
    # own PC, reached at `address`:`port` (a METEOR beside natively driven beams,
    # FIB-835). The address belongs to the driver, so it is read only with one.
    driver: Optional[str] = None
    address: Optional[str] = None
    port: Optional[int] = None

    # A remote FM whose server isn't answering at connect is built offline and comes
    # online by itself (FIB-1086), so the beams are never held up by the FM's PC.
    # `required: true` makes the connect fail instead, for a site where an FM that is
    # quietly missing would be worse than no session.
    required: Optional[bool] = None

    # The objective's calibration, in metres: where it is in focus, and how far it
    # may be inserted. Measured at this instrument, so it is written under
    # `calibration.objective` by `SystemSettings.to_dict` rather than in this
    # block. `None` means the configuration does not state one, and the working
    # state file (`fm-configuration.yaml`) still answers, exactly as before -- so
    # a site that has not pressed "Save as Calibration" sees no change.
    focus_position: Optional[float] = None
    limit_position: Optional[float] = None

    # The flip that puts the camera's frames into the stage's axes, from how it is
    # mounted (`none`, `flip-x`, `flip-y`, `flip-xy`). A fact about this instrument,
    # found by watching which way a feature moves in the FM view as the stage moves.
    # Absent is none, as every FM has been. An FM on its own PC states its own on
    # its server (`--mount-transform`), so this is not read for `driver: remote`.
    mount_transform: "CameraImageTransform" = None  # type: ignore[assignment]

    # Tilt of the camera's optical axis from the SEM column, in degrees: what the
    # stage moves project a displacement in the FM image through (FIB-335). Absent,
    # it follows from the mount: 180 where the stage turns the grid over to the FM (a
    # compustage), the ion column's tilt for an FM beside the FIB column.
    camera_tilt: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mount_transform is None:
            self.mount_transform = CameraImageTransform.NONE

    def to_dict(self) -> dict:
        """The fm entry's own keys, as the configuration writes them and as every FM
        binder receives them (``config``)."""
        settings = {
            "enabled": self.enabled,
            "driver": self.driver,
            "address": self.address,
            "port": self.port,
            "required": self.required,
        }
        # Written only when stated, so a file that never named it is saved unchanged.
        if self.mount_transform is not CameraImageTransform.NONE:
            settings["mount_transform"] = self.mount_transform.value
        if self.camera_tilt is not None:
            settings["camera_tilt"] = self.camera_tilt
        return settings

    def objective_to_dict(self) -> dict:
        return {
            "focus_position": self.focus_position,
            "limit_position": self.limit_position,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FluorescenceSystemSettings":
        settings = settings or {}
        port = settings.get("port")
        return FluorescenceSystemSettings(
            enabled=(
                bool(settings["enabled"])
                if settings.get("enabled") is not None
                else None
            ),
            driver=settings.get("driver"),
            address=settings.get("address"),
            port=int(port) if port is not None else None,
            required=(
                bool(settings["required"])
                if settings.get("required") is not None
                else None
            ),
            mount_transform=_mount_transform(settings.get("mount_transform")),
            camera_tilt=(
                float(settings["camera_tilt"])
                if settings.get("camera_tilt") is not None
                else None
            ),
        )


def _mount_transform(name: Optional[str]) -> "CameraImageTransform":
    from fibsem.devices.fm import mount_transform_from_name

    try:
        return mount_transform_from_name(name)
    except ValueError as e:
        raise ValueError(f"hardware.devices: fm: {e}") from None


# The devices configuration v1 had a block for, and their type. Each has its own record
# on `SystemSettings` (`system.stage`, `system.electron`, ...), which is what every
# reader uses; their entries in `hardware.devices` fill those records.
CONFIGURED_DEVICES: Dict[str, str] = {
    "stage": "stage",
    "electron": "beam",
    "ion": "beam",
    "fm": "fm",
}

# Types a backend builds a device of by itself, so an entry named after one of them
# needs no `type:` -- `name: chamber` is the chamber. A plugin may configure a type not
# listed here; such an entry states its `type`.
DEVICE_TYPES: Tuple[str, ...] = (
    "beam",
    "stage",
    "chamber",
    "manipulator",
    "fm",
    "sample_loader",
)


# The devices with records of their own whose entries also keep `required:`, and a
# `driver:` with that driver's own keys, which the records have no place for
# (`SystemSettings.device_entry_keys`). The FM's record has its own.
DEVICES_KEEPING_DRIVER_KEYS: Tuple[str, ...] = ("stage", "electron", "ion")


class UnknownDeviceType(ValueError):
    """A `hardware.devices` entry states no type, and its name is no device type."""


@dataclass
class DeviceEntry:
    """One entry of `hardware.devices`: a device the configuration says something about.

    **The list is an overlay.** A backend builds the devices it always has; an entry
    only changes one (switches it off, gives it another driver, gives it keys), or adds
    one the backend cannot find for itself, such as an FM on its own PC. A device the
    file does not name is built exactly as before.

    `name` is unique within the file and is how the device is found; it defaults to the
    `type`, so a site with one manipulator writes `type: manipulator` and nothing else.
    `type` may be left out where the name says it (`name: fm`). Two devices of one
    type need two names. A beam is named for its column, `electron` or `ion`, as
    `beams[BeamType]` keys it.

    `enabled` has three states, as `fm.enabled` always has: absent is the backend's
    default, `false` means never built and its driver never touches it. `driver`
    absent is the driver for `info.manufacturer`. `roles` binds a role this device has
    to another entry by name (`{scanner: scan_generator}`); the Demo binds it
    (`fibsem.devices.entries.bind_device_roles`), and every backend keeps it on save.

    Every other key is the entry's own and sits beside these in the file: the device's
    facts (`column_tilt`, `rotation_reference`) and its driver's keys (`address`,
    `port`). They are kept in `options` and written back flat.
    """

    name: str
    type: str
    enabled: Optional[bool] = None
    driver: Optional[str] = None
    required: Optional[bool] = None
    roles: Optional[Dict[str, str]] = None
    options: Dict[str, Any] = field(default_factory=dict)

    # Which entry it is, rather than anything about the device.
    IDENTITY_KEYS = ("name", "type")

    def to_dict(self) -> dict:
        entry: Dict[str, Any] = {"name": self.name, "type": self.type}
        for key in ("enabled", "driver", "required", "roles"):
            value = getattr(self, key)
            if value is not None:
                entry[key] = value
        entry.update(self.options)
        return entry

    def as_block(self) -> dict:
        """The entry as the version 1 block for its device, which the records read."""
        block = dict(self.options)
        for key in ("enabled", "driver", "required"):
            value = getattr(self, key)
            if value is not None:
                block[key] = value
        return block

    @staticmethod
    def from_dict(data: dict, name: Optional[str] = None) -> "DeviceEntry":
        """Read one entry. *name* is the block's key, for a version 1 block."""
        data = dict(data or {})
        name = data.pop("name", None) or name
        type_ = data.pop("type", None)
        if name is None and type_ is None:
            raise ValueError(
                f"A device in hardware.devices names neither a name nor a type: {data}"
            )
        if type_ is None:
            type_ = CONFIGURED_DEVICES.get(name) or (
                name if name in DEVICE_TYPES else None
            )
            if type_ is None:
                raise UnknownDeviceType(
                    f"hardware.devices: '{name}' states no type, and no device type "
                    "is called that. A device the backend does not build itself "
                    "needs a `type:`."
                )
        if name is None:
            name = type_
        if type_ == "beam" and name not in ("electron", "ion"):
            raise ValueError(
                f"hardware.devices: a beam is named `electron` or `ion`, not '{name}'."
            )
        reserved = {
            key: data.pop(key, None)
            for key in ("enabled", "driver", "required", "roles")
        }
        return DeviceEntry(
            name=str(name),
            type=str(type_),
            enabled=reserved["enabled"],
            driver=reserved["driver"],
            required=reserved["required"],
            roles=reserved["roles"],
            options=data,
        )


# The keys a device entry carries for where the stage travels for it to see the sample.
STAGE_POSITION_KEYS = ("origin", "available_orientations", "range")


def read_device_entries(settings: dict) -> Dict[str, DeviceEntry]:
    """Every device a configuration dict names, by name, in file order.

    Reads all three shapes a file can be in: the blocks at the top level from before the
    sections (`stage:`), the version 1 blocks under `hardware:` (`hardware.stage:`) --
    both only for the devices that had one, `CONFIGURED_DEVICES` -- and the version 2
    list (`hardware.devices:`). A later shape wins key by key, so a file
    half way between two reads as it says. Two list entries with one name are an error:
    which of them the file meant cannot be known. A list entry with no type whose name
    is no device type is ignored with a warning, so a file naming a device this version
    no longer has still loads.
    """
    settings = settings or {}
    hardware = settings.get("hardware") or {}
    entries: Dict[str, DeviceEntry] = {}

    def merge(entry: DeviceEntry) -> None:
        old = entries.get(entry.name)
        if old is None:
            entries[entry.name] = entry
            return
        entries[entry.name] = DeviceEntry.from_dict(
            {**old.to_dict(), **entry.to_dict()}
        )

    for name in CONFIGURED_DEVICES:
        if isinstance(settings.get(name), dict):
            merge(DeviceEntry.from_dict(settings[name], name=name))
    # Only the blocks version 1 wrote. Any other `hardware.<name>` block was never read,
    # and reading it now would build a device from a key a site left there for nothing.
    for name in CONFIGURED_DEVICES:
        if isinstance(hardware.get(name), dict):
            merge(DeviceEntry.from_dict(hardware[name], name=name))

    listed = hardware.get("devices") or []
    if not isinstance(listed, list):
        raise ValueError("hardware.devices must be a list of devices.")
    seen: Set[str] = set()
    for item in listed:
        try:
            entry = DeviceEntry.from_dict(item)
        except UnknownDeviceType as e:
            # such as a device this version no longer has (`name: gis`)
            logging.warning(f"{e} It is ignored.")
            continue
        if entry.name in seen:
            raise ValueError(
                f"hardware.devices names '{entry.name}' twice. Give each device its "
                "own name."
            )
        seen.add(entry.name)
        merge(entry)
    return entries


@dataclass
class SystemSettings:
    stage: StageSystemSettings
    electron: BeamSystemSettings
    ion: BeamSystemSettings
    manipulator: ManipulatorSystemSettings
    info: SystemInfo
    sim: Dict[str, Union[str, bool]] = field(default_factory=dict)
    fm: FluorescenceSystemSettings = field(default_factory=FluorescenceSystemSettings)
    # Whether `defaults:` is pushed to the instrument at connect
    # (`utils.setup_session`, `FibsemMicroscope.apply_defaults`). Off unless the
    # file says so: pushing a kV to a shared instrument at connect is opted into.
    apply_defaults_on_connect: bool = False
    # Whether each column is turned on at connect (`turn_beams_on`), before the
    # defaults are applied. Only ever on; off unless the file says so.
    beams_on_at_connect: bool = False
    # The devices `hardware.devices` names beyond the four with records of their own
    # (`CONFIGURED_DEVICES`), as the file states them, and written back as they were.
    other_devices: List[DeviceEntry] = field(default_factory=list)
    # The `roles:` the file gives those four, by device name. Their records have no
    # place for it, so it is kept here and written back onto their entries.
    device_roles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # The entry keys the file gives the stage and the beams that their records have no
    # place for, by device name: `required:`, and a `driver:` with that driver's own
    # keys (`driver: remote` with its `address` and `port`). Kept here and written
    # back onto their entries. A key no driver is named for is not kept, as before.
    device_entry_keys: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    #: What a column *is*: the keys that stay in `electron:` / `ion:`. Everything
    #: else a `BeamSystemSettings` writes -- voltage, current, hfw, detector, the
    #: lot -- is a default a session starts from and goes under `defaults:`.
    HARDWARE_BEAM_KEYS = (
        "beam_type",
        "enabled",
        "column_tilt",
        "eucentric_height",
        "plasma_gas",
    )

    def to_dict(self):
        """Three sections, by what kind of thing a value is.

        `hardware:` is what the instrument is and cannot be asked. `calibration:` is
        what was measured at this instrument -- the holders, and the pre-tilt while no
        holder is named -- written by a calibration action, never by an autosave.
        `defaults:` is what a session starts from. The records underneath are the
        same ones as before; the sections exist so a person opening the file can tell
        which numbers are safe to touch.
        """
        stage = self.stage.to_dict()
        calibration = {
            "holders": stage.pop("holders"),
            "active_holder": stage.pop("active_holder"),
        }
        if "shuttle_pre_tilt" in stage:
            calibration["shuttle_pre_tilt"] = stage.pop("shuttle_pre_tilt")
        if "rotation_centre_correction" in stage:
            calibration["rotation_centre_correction"] = stage.pop(
                "rotation_centre_correction"
            )
        calibration["objective"] = self.fm.objective_to_dict()

        electron = self.electron.to_dict()
        ion = self.ion.to_dict()
        defaults = {
            "apply_on_connect": self.apply_defaults_on_connect,
            "beams_on_at_connect": self.beams_on_at_connect,
            "electron": _split_defaults(electron),
            "ion": _split_defaults(ion),
        }
        electron.pop("beam_type", None)  # the entry's name says which column
        ion.pop("beam_type", None)
        devices = [
            {"name": "stage", "type": "stage", **stage},
            {"name": "electron", "type": "beam", **electron},
            {"name": "ion", "type": "beam", **ion},
            {"name": "fm", "type": "fm", **self.fm.to_dict()},
        ]
        for entry in devices:
            if entry["name"] in self.device_roles:
                entry["roles"] = self.device_roles[entry["name"]]
            entry.update(self.device_entry_keys.get(entry["name"], {}))
        devices.extend(entry.to_dict() for entry in self.other_devices)
        self._write_stage_positions(devices)
        return {
            "info": self.info.to_dict(),
            # No manipulator unless the file named one. What is fitted is the
            # instrument's to report (or the backend's, where it cannot be asked), not
            # a file's to state: `hardware.devices` is an overlay on what the backend
            # builds, so a device it does not name is built exactly as before.
            "hardware": {"devices": devices},
            "calibration": calibration,
            "defaults": defaults,
            "sim": self.sim,
        }

    def _write_stage_positions(self, devices: List[dict]) -> None:
        """Write each device's stage position onto its entry in *devices*.

        The beams have none to write: they are the implicit zero. The FM's is written
        only when it is not the default -- the objective under the grid -- so a site
        with nothing to say about it says nothing.
        """
        by_name = {entry["name"]: entry for entry in devices}
        for key, position in self.stage.devices.items():
            if key == BEAMS_STAGE_DEVICE:
                continue
            if key in DEFAULT_STAGE_DEVICES and position == DEFAULT_STAGE_DEVICES[key]:
                continue
            name = stage_device_entry_name(key)
            entry = by_name.get(name)
            if entry is None:
                entry = by_name[name] = {"name": name, "type": name}
                devices.append(entry)
            entry.update(position.to_dict())

    @staticmethod
    def from_dict(settings: dict):

        # A missing *section* defaults like a missing field. A configuration that
        # drops a block it does not need -- no manipulator -- is a configuration,
        # not a corrupt file, and this is what lets a key be removed from the
        # shipped files without every existing one raising `KeyError` at load.
        #
        # `defaults:` names what a session starts from; `electron:` / `ion:` describe
        # what the column *is*. Merged back together here because nothing downstream
        # cares about the split -- the records are unchanged, and the readers of
        # `system.electron.beam` and `system.ion.detector` do not move. `defaults:`
        # wins a collision: every file written before the split states the keys in
        # the flat block only, which is why the merge is in this direction and why
        # those files load unchanged.
        # Every file written before the sections existed has its blocks at the top
        # level, so each block is read from there first and from `hardware:` over
        # it; `calibration:` folds into the stage record it belongs to.
        #
        # Version 2 lists the devices (`hardware.devices`); `read_device_entries`
        # reads the list and both older shapes, and gives each device back as the
        # block its record has always read.
        calibration = settings.get("calibration") or {}
        defaults = settings.get("defaults") or {}
        entries = read_device_entries(settings)

        def block(name: str) -> dict:
            entry = entries.get(name)
            return entry.as_block() if entry is not None else {}

        stage = block("stage")
        for key in (
            "holders",
            "active_holder",
            "shuttle_pre_tilt",
            "rotation_centre_correction",
        ):
            if key in calibration:
                stage[key] = calibration[key]
        electron = {**block("electron"), **(defaults.get("electron") or {})}
        ion = {**block("ion"), **(defaults.get("ion") or {})}
        electron["beam_type"] = BeamType.ELECTRON.name
        ion["beam_type"] = BeamType.ION.name

        # Device positions are on each device's entry (`origin`,
        # `available_orientations`, `range`); version 1 kept them under
        # `stage.devices`, which `StageSystemSettings.from_dict` reads. An entry's
        # wins.
        stage_settings = StageSystemSettings.from_dict(stage)
        for name, entry in entries.items():
            position = {
                key: entry.options.pop(key)
                for key in STAGE_POSITION_KEYS
                if key in entry.options
            }
            if not position or name == BEAMS_STAGE_DEVICE:
                continue
            if "origin" not in position:
                raise ValueError(
                    f"hardware.devices: '{name}' states "
                    f"{sorted(position)} but no origin."
                )
            stage_settings.devices[stage_device_key(name)] = (
                StageDeviceSettings.from_dict(
                    position,
                    default_orientations=DEFAULT_AVAILABLE_ORIENTATIONS.get(
                        entry.type, ()
                    ),
                )
            )

        fm = FluorescenceSystemSettings.from_dict(block("fm"))
        objective = calibration.get("objective") or {}
        fm.focus_position = objective.get("focus_position")
        fm.limit_position = objective.get("limit_position")

        electron_settings = BeamSystemSettings.from_dict(electron)
        ion_settings = BeamSystemSettings.from_dict(ion)
        records = {
            "stage": set(stage_settings.to_dict()) | set(calibration),
            "electron": set(electron_settings.to_dict()),
            "ion": set(ion_settings.to_dict()),
        }
        device_entry_keys = {}
        for name in DEVICES_KEEPING_DRIVER_KEYS:
            record_keys = records[name]
            entry = entries.get(name)
            if entry is None:
                continue
            kept: Dict[str, Any] = {}
            if entry.driver is not None:
                kept["driver"] = entry.driver
            if entry.required is not None:
                kept["required"] = entry.required
            if entry.driver is not None:
                kept.update(
                    (key, value)
                    for key, value in entry.options.items()
                    if key not in record_keys
                )
            if kept:
                device_entry_keys[name] = kept

        return SystemSettings(
            apply_defaults_on_connect=bool(defaults.get("apply_on_connect", False)),
            beams_on_at_connect=bool(defaults.get("beams_on_at_connect", False)),
            stage=stage_settings,
            electron=electron_settings,
            ion=ion_settings,
            # Not read from the file: filled in at connect by the backend.
            manipulator=ManipulatorSystemSettings(),
            info=SystemInfo.from_dict(settings.get("info") or {}),
            sim=settings.get("sim", {}),
            fm=fm,
            other_devices=[
                entry
                for name, entry in entries.items()
                if name not in CONFIGURED_DEVICES
            ],
            device_roles={
                name: entry.roles
                for name, entry in entries.items()
                if name in CONFIGURED_DEVICES and entry.roles is not None
            },
            device_entry_keys=device_entry_keys,
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
    # ion column; fixes the compustage FIB pose
    fib_column_tilt: float = DEFAULT_FIB_COLUMN_TILT
    shuttle_pre_tilt: float = 0.0
    rotation_reference: float = 0.0
    rotation_180: float = 180.0
    is_compustage: bool = False
    # Where a half turn of the stage is centred, raw (x, y) in metres; used to draw a
    # position recorded on the other side of the stage. Defaults to the value every
    # image was reprojected with before this field existed, so an image saved without
    # it draws exactly as it did (FIB-1081).
    rotation_centre: Tuple[float, float] = LEGACY_ROTATION_CENTRE
    # The pose the stage declared for each orientation name (SEM, FIB, and FM on a
    # compustage), as (r, t) in degrees (FIB-1101). Empty on an image written before
    # the stage declared its poses; `declared_poses` rebuilds them for those.
    poses: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # The frame the image's stage position is in: STAGE_FRAME_FIBSEM, or
    # STAGE_FRAME_TESCAN from a Tescan without a stage device (FIB-1114). None on an
    # image written before this was stamped, whose position is in the frame its backend
    # reported: Tescan's own for a Tescan image, fibsem's for any other. Nothing reads
    # Tescan's frame any more; old Tescan images are not converted.
    stage_frame: Optional[str] = None
    # Fluorescence only; left at these defaults for a beam image.
    camera_tilt: float = 0.0  # viewing axis, from the electron column
    transform: CameraImageTransform = CameraImageTransform.NONE

    def declared_poses(self) -> Dict[str, "FibsemStagePosition"]:
        """The stage's pose for each orientation name when this was recorded, in radians.

        The stamped poses when there are any. An image from before the stage declared
        them gets them rebuilt the way its readers interpreted it: SEM at the reference
        rotation and the pre-tilt; FIB at the stamped ``rotation_180``, tilted to the
        ion column less the pre-tilt, and turned over by 180 on a compustage, which
        also has FM with the grid turned fully over.
        """
        poses = self.poses
        if not poses:
            ref, pre_tilt = self.rotation_reference, self.shuttle_pre_tilt
            fib_tilt = self.fib_column_tilt - pre_tilt
            poses = {"SEM": (ref, pre_tilt), "FIB": (self.rotation_180, fib_tilt)}
            if self.is_compustage:
                poses["FIB"] = (self.rotation_180, fib_tilt - 180)
                poses["FM"] = (ref, -180.0)
        return {
            name: FibsemStagePosition(r=np.radians(r), t=np.radians(t))
            for name, (r, t) in poses.items()
        }

    @classmethod
    def from_system_settings(
        cls,
        system: SystemSettings,
        is_compustage: bool = False,
        rotation_centre: Optional[Tuple[float, float]] = None,
        poses: Optional[Mapping[str, "FibsemStagePosition"]] = None,
        stage_frame: Optional[str] = None,
    ) -> "FibsemHardwareGeometry":
        """Gather the geometry terms out of a full system configuration.

        ``is_compustage`` is a parameter because ``SystemSettings`` does not carry it:
        it is a property of the installed hardware, which only the connected
        microscope knows. Callers holding one should pass ``microscope._fm_is_a_pose()``.
        ``rotation_centre`` likewise comes from the driver (``microscope.rotation_centre``);
        None records LEGACY_ROTATION_CENTRE. ``poses`` are the stage's declared poses
        (``microscope._stage_poses()``), in radians; None records none.
        ``stage_frame`` is the frame the stage reports (``microscope.stage_frame``).
        """
        return cls(
            column_tilt=system.electron.column_tilt,
            fib_column_tilt=system.ion.column_tilt,
            shuttle_pre_tilt=system.stage.shuttle_pre_tilt,
            rotation_reference=system.stage.rotation_reference,
            rotation_180=system.stage._fib_rotation(),
            is_compustage=is_compustage,
            rotation_centre=(
                rotation_centre
                if rotation_centre is not None
                else LEGACY_ROTATION_CENTRE
            ),
            poses={
                name: (float(np.degrees(pose.r)), float(np.degrees(pose.t)))
                for name, pose in (poses or {}).items()
            },
            stage_frame=stage_frame,
        )

    def to_dict(self) -> dict:
        return {
            "column_tilt": self.column_tilt,
            "fib_column_tilt": self.fib_column_tilt,
            "shuttle_pre_tilt": self.shuttle_pre_tilt,
            "rotation_reference": self.rotation_reference,
            "rotation_180": self.rotation_180,
            "is_compustage": self.is_compustage,
            "rotation_centre": list(self.rotation_centre),
            "poses": {name: list(pose) for name, pose in self.poses.items()},
            "stage_frame": self.stage_frame,
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
            fib_column_tilt=ddict.get("fib_column_tilt", DEFAULT_FIB_COLUMN_TILT),
            shuttle_pre_tilt=ddict.get("shuttle_pre_tilt", 0.0),
            rotation_reference=ddict.get("rotation_reference", 0.0),
            rotation_180=ddict.get("rotation_180", 180.0),
            is_compustage=ddict.get("is_compustage", False),
            rotation_centre=(
                _parse_rotation_centre(ddict.get("rotation_centre"))
                or LEGACY_ROTATION_CENTRE
            ),
            poses={
                name: (float(pose[0]), float(pose[1]))
                for name, pose in (ddict.get("poses") or {}).items()
            },
            stage_frame=ddict.get("stage_frame"),
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
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to None.

    There is no `milling` here. A `milling:` block existed in the configuration and
    was read into a `FibsemMillingSettings`, but milling parameters belong to a
    milling stage -- `FibsemMillingStage.milling`, chosen per pattern from the
    protocol -- and nothing in the application consulted the configuration-level one.
    Removed in the schema v1 work; the identically-named `stage.milling` is a
    different object and is unaffected.

    Methods:
        to_dict(): Returns a dictionary representation of the `MicroscopeSettings` object.
        from_dict(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """

    system: SystemSettings
    image: ImageSettings
    protocol: Optional[dict] = None
    fm: Optional["FluorescenceConfiguration"] = None

    def to_dict(self) -> dict:
        settings_dict = {"version": CONFIGURATION_VERSION, "protocol": self.protocol}
        settings_dict.update(self.system.to_dict())
        # Into the `defaults:` block `SystemSettings.to_dict` just created, beside the
        # beams: the acquire tab's opening state is the same kind of thing as the
        # voltage a session begins at.
        imaging = self.image.to_dict()
        for key in NOT_IMAGING_DEFAULTS:
            imaging.pop(key, None)
        settings_dict["defaults"]["imaging"] = imaging

        return settings_dict

    @staticmethod
    def from_dict(
        settings: dict, protocol: Optional[dict] = None
    ) -> "MicroscopeSettings":

        if protocol is None:
            protocol = settings.get("protocol", {"name": "demo"})

        # The FM working state is session state, per instrument configuration, so
        # it is loaded by `utils.load_microscope_configuration`, which knows which
        # configuration this is. Reading it here made this a function of the disk
        # rather than of the dict it was given.
        fm_config = None

        return MicroscopeSettings(
            system=SystemSettings.from_dict(settings),
            # `defaults.imaging` first, the old top-level `imaging:` after it.
            image=ImageSettings.from_dict(
                (settings.get("defaults") or {}).get("imaging")
                or settings.get("imaging")
                or {}
            ),
            protocol=protocol,
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
    # When the reference was made, which is when a session adopted the experiment.
    # Aware, written as ISO 8601 with its offset (FIB-1197); older images hold a
    # POSIX float, which reads as this machine's zone. default_factory, not a plain
    # default: a plain default is evaluated once at class definition, so every
    # experiment recorded the interpreter's import time rather than its own.
    date: Optional[datetime] = field(default_factory=now)

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
            "date": to_iso(self.date),
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
            # None when absent or unreadable; files before FIB-1197 hold a float
            date=to_datetime(settings.get("date")),
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

    **The instrument's zone** (FIB-1196): ``utc_offset`` and ``zone_name``, so a time
    that carries no offset of its own -- a POSIX float, which every task time and
    most beam images before FIB-1190 are -- can be read on the instrument's clock
    on any machine (``wall_time_of(value, session.zone)``). New times carry their own
    offset and do not need it. Three limits:

    - An offset belongs to a moment, not a zone. A run that crosses a daylight
      saving change reads an hour out on the side the session did not record.
    - ``zone_name`` is what the OS calls the zone, as given: ``AEST`` on macOS and
      Linux, ``AUS Eastern Standard Time`` on Windows. It labels, it is not parsed.
    - A Demo session records no zone of its own and keeps the last one recorded, so
      opening an experiment on another machine does not replace the instrument's.
    """

    # Aware, written as ISO 8601 with its offset (FIB-1197); older experiments hold
    # a POSIX float, which reads as this machine's zone.
    recorded_at: Optional[datetime] = field(default_factory=now)
    system: Optional[SystemInfo] = None
    user: Optional[FibsemUser] = None
    # Distribution name -> version for every installed extension; see
    # `installed_plugin_versions`. A plain dict so it survives `yaml.safe_dump`,
    # which is how an experiment is written.
    plugins: Dict[str, str] = field(default_factory=dict)
    utc_offset: Optional[str] = None  # "+10:00"
    zone_name: Optional[str] = None

    @property
    def zone(self) -> Optional[tzinfo]:
        """The instrument's zone, fixed at its recorded offset; None when unknown."""
        return zone_from_offset(self.utc_offset, self.zone_name)

    @classmethod
    def collect(
        cls,
        microscope: "FibsemMicroscope",
        user: Optional[FibsemUser] = None,
        previous: Optional["SessionInfo"] = None,
    ) -> "SessionInfo":
        """Snapshot what is running right now.

        ``user`` overrides the environment's, for a caller that knows better: on a
        shared facility login the OS account names the workstation rather than the
        operator, so a name somebody actually typed is the stronger evidence.

        The system info is copied. It is a live object on the microscope -- the
        application field is set on it during registration, and a driver may update
        it -- and a record that quietly changes after the fact is not a record.

        ``previous`` is the session this one replaces: a Demo session keeps its zone.
        """
        from copy import deepcopy

        # Lazy: `report` imports all three registries, so importing it here would
        # cycle straight back through this module.
        from fibsem.plugins.report import installed_plugin_versions

        info = getattr(getattr(microscope, "system", None), "info", None)
        if normalize_manufacturer(getattr(info, "manufacturer", None)) == DEMO:
            utc_offset = previous.utc_offset if previous is not None else None
            zone_name = previous.zone_name if previous is not None else None
        else:
            here = now()
            utc_offset, zone_name = utc_offset_of(here), here.tzname()
        return cls(
            system=deepcopy(info) if info is not None else None,
            user=user if user is not None else FibsemUser.from_environment(),
            plugins=installed_plugin_versions(),
            utc_offset=utc_offset,
            zone_name=zone_name,
        )

    def to_dict(self) -> dict:
        return {
            "recorded_at": to_iso(self.recorded_at),
            "system": self.system.to_dict() if self.system is not None else None,
            "user": self.user.to_dict() if self.user is not None else None,
            "plugins": dict(self.plugins),
            "utc_offset": self.utc_offset,
            "zone_name": self.zone_name,
        }

    @staticmethod
    def from_dict(ddict: dict) -> "SessionInfo":
        system = ddict.get("system")
        user = ddict.get("user")
        return SessionInfo(
            recorded_at=to_datetime(ddict.get("recorded_at")),
            system=SystemInfo.from_dict(system) if system else None,
            user=FibsemUser.from_dict(user) if user else None,
            plugins=dict(ddict.get("plugins") or {}),
            utc_offset=ddict.get("utc_offset"),
            zone_name=ddict.get("zone_name"),
        )


@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage.

    Three kinds of claim live here, and they are easy to mistake for each other
    (FIB-445 D1). Anything added should be placed deliberately in one of them:

    **Provenance** -- what produced this image. ``system_info`` (which instrument),
    ``user`` (who), ``experiment`` (which run). Constant for a run. The same
    question at a finer grain -- which item (a lamella, or a grid), which task --
    is answered by ``experiment.item_name`` and ``experiment.task_name`` (FIB-466).
    That varies *within* a run, which changes the mechanism that writes it, but not
    the kind of fact it is.

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
    # When the image was acquired: an aware datetime, written as ISO 8601 with its
    # UTC offset, `2026-09-13T21:19:40.974286-06:00` (FIB-1190). An observation, like
    # microscope_state, but a separate field: the state's timestamp says when the
    # state was read, which is not the same moment. `Beam.acquire` stamps it just
    # before the driver runs; a driver with a vendor time sets its own. None on
    # files from before v11 and on images built rather than acquired -- read the
    # time through `acquisition_datetime_of`, which falls back for those.
    acquisition_datetime: Optional[datetime] = None

    @property
    def beam_type(self) -> BeamType:
        return self.image_settings.beam_type

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.microscope_state.stage_position

    @property
    def acquisition_date(self) -> Optional[datetime]:
        """When the image was acquired: aware, with the offset it was recorded with,
        when the file says which zone.

        Naive when it does not: a ThermoFisher image from before v11 records the
        instrument PC's clock time alone. See `acquisition_datetime_of`.
        """
        return acquisition_datetime_of(self)

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
        if self.acquisition_datetime is not None:
            settings_dict["acquisition_datetime"] = (
                self.acquisition_datetime.isoformat()
            )
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
        That constructor used to be bracket-indexed and raise on a blob missing any
        block, which would have broken exactly the old files this exists to load; it
        now defaults instead, so the two routes agree and the choice is no longer
        load-bearing. The agreement is not free, though -- it holds because both
        declare the FIB column tilt from one constant -- and
        `tests/test_metadata_fixtures.py` pins it.

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
            acquisition_datetime=to_aware(settings.get("acquisition_datetime")),
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
            acquisition_date=acquisition_datetime_of(md),
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


# ---------------------------------------------------------------------------
# The sample holder
#
# Moved here from `fibsem/microscopes/_stage.py`, unchanged, so that
# `SystemSettings` can hold one. `_stage.py` imports from this module, so a
# holder field on `SystemSettings` would have closed an import loop; a class this
# far down the dependency order belongs below the loop rather than behind a
# deferred import that hides it. `_stage.py` re-exports these four names, so
# every existing importer is unaffected.
# ---------------------------------------------------------------------------

GRID_RADIUS = 1e-3  # 1mm


@dataclass
class SampleGrid:
    """A physical TEM grid or sample that can be loaded into a GridSlot."""

    name: str
    description: str = ""
    radius: float = field(
        default=GRID_RADIUS,
        metadata={"unit": "mm", "tooltip": "Radius of the sample grid", "scale": 1e3},
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleGrid":
        return SampleGrid(
            name=data.get("name", ""),
            description=data.get("description", ""),
            radius=data.get("radius", GRID_RADIUS),
        )


@dataclass
class SlotCalibration:
    """The proof that a slot position was captured properly, and what it was captured against.

    Written only by the calibration wizard. A position without one, which is every
    holder file in the field before this existed, is not trusted: it was captured at
    an unknown orientation with a button that took whatever the stage said. A position
    whose ``pre_tilt`` or ``rotation_reference`` no longer match the system
    configuration is not trusted either, since the geometry it was captured against
    has moved. Both cases read as "not calibrated" and the wizard is the way back.
    """

    orientation: str
    pre_tilt: float
    rotation_reference: float
    # When the wizard captured it: aware, written as ISO 8601 with its offset
    # (FIB-1190, FIB-1197). Records before FIB-1190 hold a naive string, which stays
    # naive. None for a position nobody captured.
    captured_at: Optional[datetime] = None
    fibsem_version: str = ""

    def __post_init__(self) -> None:
        # the string a holder file or a caller holds
        self.captured_at = to_datetime(self.captured_at) if self.captured_at else None

    @classmethod
    def builtin(cls, pre_tilt: float, rotation_reference: float) -> "SlotCalibration":
        """A position nobody captured: the compustage working slot's nominal one.

        The autoloader puts every grid at the same place, nominally the stage
        origin, so the slot starts there with this record. A real instrument puts
        it a fixed distance off, so a position captured with the calibration wizard
        replaces it once there is one (FIB-1144).
        """
        return cls(
            orientation="SEM",
            pre_tilt=pre_tilt,
            rotation_reference=rotation_reference,
            captured_at=None,
            fibsem_version="built-in",
        )

    @property
    def is_builtin(self) -> bool:
        return self.fibsem_version == "built-in" and not self.captured_at

    def matches(self, pre_tilt: float, rotation_reference: float) -> bool:
        return (
            abs(self.pre_tilt - pre_tilt) < 1e-3
            and abs(self.rotation_reference - rotation_reference) < 1e-3
        )

    def to_dict(self) -> dict:
        return {
            "orientation": self.orientation,
            "pre_tilt": self.pre_tilt,
            "rotation_reference": self.rotation_reference,
            "captured_at": to_iso(self.captured_at) or "",
            "fibsem_version": self.fibsem_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SlotCalibration":
        return SlotCalibration(
            orientation=str(data.get("orientation", "")),
            pre_tilt=float(data.get("pre_tilt", 0.0)),
            rotation_reference=float(data.get("rotation_reference", 0.0)),
            captured_at=data.get("captured_at") or None,
            fibsem_version=str(data.get("fibsem_version", "")),
        )


@dataclass
class GridSlot:
    """A slot that may hold one SampleGrid.

    A holder *working* slot has a stage ``position`` once it has been calibrated, and
    a ``calibration`` record saying so; until then ``position`` is None and nothing
    will move to it. A loader *magazine* slot is storage and never has a position.
    """

    name: str
    index: int
    position: Optional[FibsemStagePosition] = None
    loaded_grid: Optional[SampleGrid] = None
    calibration: Optional[SlotCalibration] = None

    @property
    def is_calibrated(self) -> bool:
        return self.position is not None and self.calibration is not None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "position": self.position.to_dict() if self.position is not None else None,
            "loaded_grid": self.loaded_grid.to_dict()
            if self.loaded_grid is not None
            else None,
            "calibration": self.calibration.to_dict()
            if self.calibration is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridSlot":
        loaded_grid_data = data.get("loaded_grid")
        loaded_grid = (
            SampleGrid.from_dict(loaded_grid_data)
            if loaded_grid_data is not None
            else None
        )
        position_data = data.get("position")
        position = (
            FibsemStagePosition(**position_data) if position_data is not None else None
        )
        calibration_data = data.get("calibration")
        calibration = (
            SlotCalibration.from_dict(calibration_data)
            if calibration_data is not None
            else None
        )
        slot = GridSlot(
            name=data.get("name", ""),
            index=data.get("index", 0),
            position=position,
            loaded_grid=loaded_grid,
            calibration=calibration,
        )
        if slot.position is not None:
            slot.position.name = slot.name
        return slot


@dataclass
class SampleHolder:
    # First, and with no default, so it cannot be left out. The pre-tilt is a property
    # of *this holder* -- swap a 35 degree shuttle for a flat one and it changes with
    # the shuttle, which is why it stopped being a field on the stage.
    #
    # Required because the alternatives both fail quietly. A default of 0.0 turns a
    # construction site that forgot into a flat shuttle and wrongs every projection
    # made from it, with nothing to report; a `None` sentinel spreads its own handling
    # into every reader, and one reader that formats it instead becomes a hard abort
    # (PyQt5 turns an exception in a slot into `qFatal`). Ordered first because a
    # dataclass cannot put a non-default field after defaulted ones, and
    # `@dataclass(kw_only=True)` is 3.10+ while this package supports 3.8.
    #
    # Absence in a *file* is a different question and is not this field's to answer:
    # `from_dict` supplies the configured value, and `_resolve_configured_holder`
    # seeds it at connect.
    #
    # This used to be a property reading *back* from
    # `_parent.system.stage.shuttle_pre_tilt`. That direction is now reversed, and
    # both cannot exist: the stage reads the holder, so a holder that read the stage
    # would recurse until the interpreter gave up.
    pre_tilt: float = field(
        metadata={"unit": "°", "tooltip": "Pre-tilt of this holder, in degrees"}
    )
    name: str = field(
        default="Sample Holder", metadata={"tooltip": "Name of the sample holder"}
    )
    description: str = field(
        default="", metadata={"tooltip": "Description of the sample holder"}
    )
    capacity: int = field(
        default=2,
        metadata={
            "minimum": 1,
            "maximum": 12,
            "tooltip": "Number of grid slots on this holder",
        },
    )
    slots: dict[str, GridSlot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._parent_ref: Optional["weakref.ReferenceType"] = None

    # The microscope this holder is mounted on, held WEAKLY. A holder now lives in
    # `system.stage.holders`, a plain settings record, and whoever keeps that record
    # -- `microscope, settings = setup_session(...)` keeps it for as long as the
    # caller's frame lives -- would otherwise keep the whole microscope alive through
    # this back-reference, and with it every recorder and window subscribed to its
    # signals. The holder is a description; it must not own the instrument.
    @property
    def _parent(self) -> Optional["FibsemMicroscope"]:
        return self._parent_ref() if self._parent_ref is not None else None

    @_parent.setter
    def _parent(self, microscope: Optional["FibsemMicroscope"]) -> None:
        self._parent_ref = weakref.ref(microscope) if microscope is not None else None

    @property
    def reference_rotation(self) -> float:
        if self._parent is not None:
            return self._parent.system.stage.rotation_reference
        return 0.0

    def find_slot_for_grid(self, grid: "SampleGrid") -> Optional["GridSlot"]:
        """Return the slot that has this SampleGrid loaded, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid.name:
                return slot
        return None

    def find_slot_by_grid_name(self, grid_name: str) -> Optional["GridSlot"]:
        """Return the slot whose loaded grid matches the given name, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid_name:
                return slot
        return None

    @property
    def occupied_slots(self) -> List["GridSlot"]:
        """The working slots that hold a grid: what is loaded right now."""
        return [
            s
            for s in sorted(self.slots.values(), key=lambda s: s.index)
            if s.loaded_grid is not None
        ]

    @property
    def calibrated_slots(self) -> List["GridSlot"]:
        """The slots with a trusted position: the ones the stage can be sent to."""
        return [
            s
            for s in sorted(self.slots.values(), key=lambda s: s.index)
            if s.is_calibrated
        ]

    def discard_untrusted_positions(
        self, pre_tilt: float, rotation_reference: float
    ) -> List[str]:
        """Drop every slot position that was not calibrated against this geometry.

        A position with no calibration record was captured by the old per-slot
        button at an unknown orientation; one whose record disagrees with the current
        pre-tilt or reference rotation was captured against a stage that has since
        been reconfigured. Neither can be trusted, so both become "not calibrated"
        in memory. The file is left alone: the wizard rewrites it when someone
        recalibrates, and never before. Returns one line per discarded slot, for
        the log and the UI.
        """
        notes: List[str] = []
        for slot in sorted(self.slots.values(), key=lambda s: s.index):
            if slot.position is None:
                continue
            if slot.calibration is None:
                reason = "it has no calibration record"
            elif not slot.calibration.matches(pre_tilt, rotation_reference):
                reason = (
                    f"it was calibrated at pre-tilt {slot.calibration.pre_tilt:g}°, "
                    f"reference rotation {slot.calibration.rotation_reference:g}°, "
                    f"but the system is configured for {pre_tilt:g}° / "
                    f"{rotation_reference:g}°"
                )
            else:
                continue
            slot.position = None
            slot.calibration = None
            notes.append(f"{slot.name}: position discarded because {reason}")
        return notes

    def _ensure_slots(self) -> None:
        """Ensure exactly `capacity` slots exist; add empty ones for missing indices."""
        for i in range(self.capacity):
            name = f"Slot-{i + 1:02d}"
            if name not in self.slots:
                # A new slot has no position until it is calibrated; inventing one
                # at the origin was a number that looked like a measurement.
                self.slots[name] = GridSlot(name=name, index=i, position=None)
        for name in [
            n for n, s in list(self.slots.items()) if s.index >= self.capacity
        ]:
            del self.slots[name]

    def to_dict(self, include_grids: bool = True) -> dict:
        slots = {}
        for name, slot in self.slots.items():
            data = slot.to_dict()
            if not include_grids:
                data["loaded_grid"] = None
            slots[name] = data
        return {
            "name": self.name,
            "capacity": self.capacity,
            "slots": slots,
            "description": self.description,
            "pre_tilt": self.pre_tilt,
        }

    # -- occupancy: which grid is in which slot, kept apart from the calibration --

    def occupancy_to_dict(self) -> dict:
        """Slot name -> grid, for the slots that hold one."""
        return {
            name: slot.loaded_grid.to_dict()
            for name, slot in self.slots.items()
            if slot.loaded_grid is not None
        }

    def apply_occupancy(self, data: dict) -> None:
        """Put the recorded grids back into their slots; unlisted slots are emptied."""
        for name, slot in self.slots.items():
            grid_data = (data or {}).get(name)
            slot.loaded_grid = (
                SampleGrid.from_dict(grid_data) if grid_data is not None else None
            )

    @classmethod
    def from_dict(cls, data: dict) -> "SampleHolder":
        slots = {
            name: GridSlot.from_dict(slot_data)
            for name, slot_data in data.get("slots", {}).items()
        }
        # A file that does not state one reads as 0.0 here, and that is safe only
        # because of what happens next: `_resolve_configured_holder` overwrites it
        # with the configured pre-tilt before the holder is used. The required field
        # constrains *constructions in code*, which is where a forgotten pre-tilt has
        # nothing else to catch it; a silent file is caught at connect instead.
        holder = SampleHolder(
            pre_tilt=float(data.get("pre_tilt") or 0.0),
            name=data.get("name", "Sample Holder"),
            capacity=data.get("capacity", max(len(slots), 1)),
            slots=slots,
            description=data.get("description", ""),
        )
        holder._ensure_slots()
        return holder

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SampleHolder":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sample holder config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        """Write the holder's geometry and calibration. Not the grids in it: those
        are session state (``fibsem.microscopes._stage.save_holder_occupancy``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.to_dict(include_grids=False),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


# The holder a system starts with when nothing else describes one: no `stage.holders`
# in the configuration, and no `sample-holder.yaml` beside it.
#
# This used to be `default-sample-holder.yaml`, a shipped file. Once the holder moved
# into the microscope configuration the file was down to four fields and two empty
# slot stubs -- everything else in it was null -- and its name, "Pre-Tilted 35deg
# Shuttle", had become a claim the object could contradict: a flat system loaded a
# holder called that carrying a pre-tilt of 0, and the widget printed both. A default
# with no calibration in it is a code default, next to `DEFAULT_STAGE_DEVICES` and
# `DEFAULT_DEVICE_RANGE`, which are already here.
#
# A function rather than a module constant because a holder is mutable and gets a
# `_parent` bound to it; one shared instance would be handed to every microscope.
def default_sample_holder(pre_tilt: float) -> "SampleHolder":
    """A two-slot shuttle with nothing calibrated on it.

    `pre_tilt` is required for the same reason it is required on the holder: this is
    the one caller that has to decide, and the configured value is what it passes.
    The name deliberately describes the slot count rather than a geometry, so it
    cannot disagree with the number beside it.
    """
    holder = SampleHolder(
        pre_tilt=pre_tilt,
        name="Default Shuttle",
        capacity=2,
        description="Two grid slots, uncalibrated. Replace or calibrate before use.",
    )
    holder._ensure_slots()
    return holder
