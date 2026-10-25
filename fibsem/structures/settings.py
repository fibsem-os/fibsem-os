"""Positions, beams, imaging and milling settings, and the sample holder."""

from __future__ import annotations

import math
import numbers
import os
import weakref
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BeforeValidator

from fibsem import constants
from fibsem.codec import (
    Array,
    Blocks,
    NamedEnum,
    Tagged,
    Timestamp,
    decode,
    encode,
    saved,
)
from fibsem.codec.upgrades import holder as holder_upgrades
from fibsem.codec.upgrades.overview import acquisition_settings
from fibsem.config import DEFAULT_STANDARD_RESOLUTION_LIST
from fibsem.geometry.angles import angle_difference
from fibsem.structures.params import (
    Configurable,
    Display,
    RangeLimit,
    field_specs,
    flagged_fields,
    param,
)
from fibsem.util.timestamps import to_datetime

if TYPE_CHECKING:
    from fibsem.autofunctions.autofocus import AutoFocusSettings
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures.image import FibsemImage


def _plain_number(value: Any) -> Any:
    """A numpy scalar as the plain Python number it is (a float stays a float,
    an integer an int, so pixel coordinates still index); anything else (None,
    a Python number, an array) as is.

    Geometry built from numpy maths carries numpy scalars, which the YAML
    writers refuse; one in a saved experiment made it unsaveable.
    """
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


@saved(required=("x", "y"))
@dataclass
class Point(Configurable):
    x: float = param(
        0.0,
        doc="The x value. Its unit is the context's: metres for a shift or a position in an image, none for a stigmation.",
    )
    y: float = param(0.0, doc="The y value, in the same unit as x.")
    name: Optional[str] = param(None, doc="A name for the point.")

    # numpy would otherwise read a Point as the sequence [x, y] and answer
    # np.float64(2) * point with an array instead of calling __rmul__.
    __array_ufunc__ = None

    def __post_init__(self) -> None:
        self.x = _plain_number(self.x)
        self.y = _plain_number(self.y)

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(d: dict) -> "Point":
        return decode(Point, d)

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

    def __mul__(self, scale: float) -> "Point":
        """Both coordinates times a number, e.g. pixels times a pixel size. Only a
        number: the point carries no unit, so the result is in whatever unit the
        number makes it."""
        if not isinstance(scale, numbers.Real):
            return NotImplemented
        return Point(self.x * scale, self.y * scale)

    __rmul__ = __mul__

    def __truediv__(self, scale: float) -> "Point":
        """Both coordinates divided by a number, e.g. metres by a pixel size."""
        if not isinstance(scale, numbers.Real):
            return NotImplemented
        return Point(self.x / scale, self.y / scale)

    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def distance(self, other: "Point") -> "Point":
        """Calculate the distance between two points. (other - self)"""
        return Point(x=(other.x - self.x), y=(other.y - self.y))

    def euclidean(self, other: "Point") -> float:
        """Calculate the euclidean distance between two points."""
        return float(np.linalg.norm(self.distance(other).to_list()))


_DISTANCE_METRICS = ("euclidean", "manhattan", "max")


@dataclass
class FibsemAxesPosition(Configurable):
    """The maths shared by stage and manipulator positions.

    A subclass declares its axes as dataclass fields and lists them in ``AXES``; the
    ones in ``ANGULAR_AXES`` are angles in radians, the rest lengths in metres. An axis
    that is None is unknown, or "don't move" in an absolute move. Serialisation stays
    on each subclass, so the dict formats do not change.
    """

    AXES: ClassVar[Tuple[str, ...]] = ()
    LINEAR_AXES: ClassVar[Tuple[str, ...]] = ()
    ANGULAR_AXES: ClassVar[Tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        for axis in self.AXES:
            setattr(self, axis, _plain_number(getattr(self, axis)))

    def _require_same_type(self, other: object) -> None:
        if type(other) is not type(self):
            raise TypeError(
                f"{type(self).__name__} cannot be combined with {type(other).__name__}"
            )

    def _validated_axes(self, axes: Optional[Sequence[str]]) -> Tuple[str, ...]:
        if axes is None:
            return tuple(self.AXES)
        unknown = [axis for axis in axes if axis not in self.AXES]
        if unknown:
            raise ValueError(
                f"Invalid axes: {list(axes)}. Must be one of: {list(self.AXES)}"
            )
        return tuple(axes)

    def __add__(self, other: "FibsemAxesPosition") -> "FibsemAxesPosition":
        """Each axis of *other* added to this one; an axis None on *other* keeps this
        one's value, and an axis None here stays None."""
        self._require_same_type(other)
        values = {}
        for axis in self.AXES:
            mine, theirs = getattr(self, axis), getattr(other, axis)
            values[axis] = mine if mine is None or theirs is None else mine + theirs
        return type(self)(**values, coordinate_system=self.coordinate_system)

    def __sub__(self, other: "FibsemAxesPosition") -> "FibsemAxesPosition":
        """Each axis of *other* taken from this one; None where either is None."""
        self._require_same_type(other)
        values = {}
        for axis in self.AXES:
            mine, theirs = getattr(self, axis), getattr(other, axis)
            values[axis] = None if mine is None or theirs is None else mine - theirs
        return type(self)(**values, coordinate_system=self.coordinate_system)

    def axis_differences(
        self, other: "FibsemAxesPosition", axes: Optional[Sequence[str]] = None
    ) -> Dict[str, Optional[float]]:
        """The unsigned difference on each axis, angles wrapped; None where either
        side is None."""
        self._require_same_type(other)
        differences: Dict[str, Optional[float]] = {}
        for axis in self._validated_axes(axes):
            mine, theirs = getattr(self, axis), getattr(other, axis)
            if mine is None or theirs is None:
                differences[axis] = None
            elif axis in self.ANGULAR_AXES:
                differences[axis] = float(angle_difference(mine, theirs))
            else:
                differences[axis] = float(abs(mine - theirs))
        return differences

    def distance(
        self,
        other: "FibsemAxesPosition",
        axes: Optional[Sequence[str]] = None,
        metric: str = "euclidean",
    ) -> float:
        """The distance to *other* over *axes* (default: the linear axes).

        Axes that are None on either side are skipped. Angular axes are compared
        wrapped, so 359.9 and 0.1 degrees are 0.2 degrees apart. Mixing linear and
        angular axes adds metres to radians, so pick one kind unless that is meant.
        *metric* is "euclidean", "manhattan" (sum) or "max" (largest single axis).
        """
        if metric not in _DISTANCE_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {_DISTANCE_METRICS}"
            )
        if axes is None:
            axes = self.LINEAR_AXES
        known = [
            d for d in self.axis_differences(other, axes).values() if d is not None
        ]
        if not known:
            return 0.0
        if metric == "manhattan":
            return float(sum(known))
        if metric == "max":
            return float(max(known))
        return float(np.linalg.norm(known))

    def euclidean_distance(self, other: "FibsemAxesPosition") -> float:
        """The straight-line distance over the linear axes, skipping None axes."""
        return self.distance(other)

    def is_close2(
        self,
        pos2: "FibsemAxesPosition",
        tol: float = 1e-6,
        axes: Optional[List[str]] = None,
    ) -> bool:
        """Whether every axis in *axes* (default all) is within *tol* of *pos2*.

        Angles are compared wrapped. An axis None on either side is not close.
        """
        differences = self.axis_differences(pos2, axes)
        if any(d is None for d in differences.values()):
            return False
        return self.distance(pos2, axes=list(differences), metric="max") < tol

    def is_close(self, pos2: "FibsemAxesPosition", tol: float = 1e-6) -> bool:
        """Whether every axis is within *tol* of *pos2*."""
        return self.is_close2(pos2, tol=tol)

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

    def _pretty_axes(self, axes: Sequence[str], units: bool) -> str:
        parts = []
        for axis in axes:
            value = getattr(self, axis)
            label = axis.upper()
            if value is None:
                parts.append(f"{label}:None")
            elif axis in self.ANGULAR_AXES:
                unit = "°" if units else ""
                parts.append(
                    f"{label}:{value * constants.RADIANS_TO_DEGREES:.1f}{unit}"
                )
            else:
                unit = "mm" if units else ""
                parts.append(
                    f"{label}:{value * constants.METRE_TO_MILLIMETRE:.2f}{unit}"
                )
        return ", ".join(parts)

    @property
    def pretty_string(self) -> str:
        """Returns a pretty string representation of the position."""
        return self._pretty_axes(self.AXES, units=False)

    @property
    def pretty_orientation(self) -> str:
        """Returns a pretty string representation of the orientation."""
        return self._pretty_axes(self.ANGULAR_AXES, units=False)

    @property
    def pretty(self) -> str:
        """Returns a pretty string representation of the position including units."""
        return self._pretty_axes(self.AXES, units=True)


@saved(required=("x", "y", "z", "r", "t", "coordinate_system"))
@dataclass
class FibsemStagePosition(FibsemAxesPosition):
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
        Each driver converts to and from its SDK's own position type, in its module under fibsem.drivers.
    """

    AXES: ClassVar[Tuple[str, ...]] = ("x", "y", "z", "r", "t")
    LINEAR_AXES: ClassVar[Tuple[str, ...]] = ("x", "y", "z")
    ANGULAR_AXES: ClassVar[Tuple[str, ...]] = ("r", "t")

    name: Optional[str] = param(
        None, doc="A name for the position, such as a saved position's."
    )
    x: Optional[float] = param(
        None,
        unit="m",
        doc="The stage's x position.",
        display=Display("X", scale=1e3, step=0.001, decimals=5),
    )
    y: Optional[float] = param(
        None,
        unit="m",
        doc="The stage's y position.",
        display=Display("Y", scale=1e3, step=0.001, decimals=5),
    )
    z: Optional[float] = param(
        None,
        unit="m",
        doc="The stage's z position.",
        display=Display("Z", scale=1e3, step=0.001, decimals=5),
    )
    r: Optional[float] = param(
        None,
        unit="rad",
        doc="The stage's rotation.",
        display=Display("Rotation", scale=180 / math.pi, unit="°"),
    )
    t: Optional[float] = param(
        None,
        unit="rad",
        doc="The stage's tilt.",
        display=Display("Tilt", scale=180 / math.pi, unit="°"),
    )
    coordinate_system: Optional[str] = param(
        None,
        doc="The frame the coordinates are in, as the driver names it, such as RAW.",
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemStagePosition":
        return decode(cls, data)


@saved(required=("x", "y", "z", "r", "t"))
@dataclass
class FibsemManipulatorPosition(FibsemAxesPosition):
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
        Each driver converts to and from its SDK's own position type, in its module under fibsem.drivers.
    """

    AXES: ClassVar[Tuple[str, ...]] = ("x", "y", "z", "r", "t")
    LINEAR_AXES: ClassVar[Tuple[str, ...]] = ("x", "y", "z")
    ANGULAR_AXES: ClassVar[Tuple[str, ...]] = ("r", "t")

    x: float = param(
        0.0,
        unit="m",
        doc="The needle's x position.",
        display=Display("X", scale=1e3, step=0.001, decimals=5),
    )
    y: float = param(
        0.0,
        unit="m",
        doc="The needle's y position.",
        display=Display("Y", scale=1e3, step=0.001, decimals=5),
    )
    z: float = param(
        0.0,
        unit="m",
        doc="The needle's z position.",
        display=Display("Z", scale=1e3, step=0.001, decimals=5),
    )
    r: float = param(
        0.0,
        unit="rad",
        doc="The needle's rotation.",
        display=Display("Rotation", scale=180 / math.pi, unit="°"),
    )
    t: float = param(
        0.0,
        unit="rad",
        doc="The needle's tilt.",
        display=Display("Tilt", scale=180 / math.pi, unit="°"),
    )
    coordinate_system: Optional[str] = param(
        None, doc="The frame the coordinates are in, as the driver names it."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # in capitals, as it has always been saved
        if self.coordinate_system:
            self.coordinate_system = self.coordinate_system.upper()

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemManipulatorPosition":
        return decode(cls, data)


@saved(required=("left", "top", "width", "height"))
@dataclass
class FibsemRectangle(Configurable):
    """Universal Rectangle class used for ReducedArea"""

    left: float = param(
        0.0,
        limits=RangeLimit(0.0, 1.0),
        doc="The left edge, from the frame's left, as a fraction of the full frame.",
        display=Display("Left", step=0.01, decimals=2),
    )
    top: float = param(
        0.0,
        limits=RangeLimit(0.0, 1.0),
        doc="The top edge, from the frame's top, as a fraction of the full frame.",
        display=Display("Top", step=0.01, decimals=2),
    )
    width: float = param(
        1.0,
        limits=RangeLimit(0.0, 1.0),
        doc="The width, as a fraction of the full frame.",
        display=Display("Width", step=0.01, decimals=2),
    )
    height: float = param(
        1.0,
        limits=RangeLimit(0.0, 1.0),
        doc="The height, as a fraction of the full frame.",
        display=Display("Height", step=0.01, decimals=2),
    )

    def __post_init__(self):
        self.left = _plain_number(self.left)
        self.top = _plain_number(self.top)
        self.width = _plain_number(self.width)
        self.height = _plain_number(self.height)
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
    def from_dict(cls, settings: Optional[dict]) -> Optional["FibsemRectangle"]:
        """None for None: a reduced area that isn't set."""
        return None if settings is None else decode(cls, settings)

    def to_dict(self) -> dict:
        return encode(self)

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


class BeamType(NamedEnum, Enum):
    """Enumerator Class for Beam Type
    1: Electron Beam
    2: Ion Beam

    """

    ELECTRON = 1  # Electron
    ION = 2  # Ion

    # CCD_CAM = 3
    # NavCam = 4 # see enumerations/ImagingDevice


class ImagingState(NamedEnum, Enum):
    IDLE = 0
    RUNNING = 1
    STOPPING = 2
    PAUSED = 3
    ERROR = 4


class ScanMode(Enum):
    """What a beam scans. The values are the ``scanning_mode`` key's."""

    FULL_FRAME = "full_frame"
    REDUCED_AREA = "reduced_area"
    SPOT = "spot"


# `current` and `eucentric_height` are how a configuration's beam defaults spell
# them. A missing scan rotation is 0, as it has always read.
@saved(
    reads={"beam_current": "current", "working_distance": "eucentric_height"},
    absent={"beam_type": BeamType.ELECTRON, "scan_rotation": 0.0},
)
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
        return encode(self)

    @staticmethod
    def from_dict(state_dict: dict) -> "BeamSettings":
        return decode(BeamSettings, state_dict)


@dataclass
class FibsemDetectorSettings:
    type: Optional[str] = "Unknown"
    mode: Optional[str] = "Unknown"
    brightness: Optional[float] = 0.5
    contrast: Optional[float] = 0.5

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
        return encode(self)

    @staticmethod
    def from_dict(settings: dict) -> "FibsemDetectorSettings":
        return decode(FibsemDetectorSettings, settings)


@dataclass
class MicroscopeState:
    """Data Class representing the state of a microscope with various parameters.

    Attributes:

        timestamp (float): When the state was read, as a POSIX time. Defaults to the moment it is constructed.
        stage_position (FibsemStagePosition): An instance of FibsemStagePosition representing the current absolute position of the stage. Defaults to an empty instance of FibsemStagePosition.
        electron_beam (BeamSettings): An instance of BeamSettings representing the electron beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ELECTRON.
        ion_beam (BeamSettings): An instance of BeamSettings representing the ion beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ION.

    Methods:

        to_dict(self) -> dict: Converts the current state of the Microscope to a dictionary and returns it.
        from_dict(state_dict: dict) -> "MicroscopeState": Returns a new instance of MicroscopeState with attributes created from the passed dictionary.
    """

    # When the state was read, as POSIX: not when a frame was taken, which is the
    # image's acquisition_datetime. default_factory, not a plain default, which is
    # evaluated once when this module is imported.
    # A string only in a ThermoFisher image from before v11, which wrote the PC's
    # clock time as month/day/year; `acquisition_datetime_of` reads either.
    timestamp: Union[float, str] = field(
        default_factory=lambda: datetime.now().timestamp()
    )
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
        return encode(self)

    @staticmethod
    def from_dict(state_dict: dict) -> "MicroscopeState":
        return decode(MicroscopeState, state_dict)


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


class AutoFocusMode(NamedEnum, Enum):
    """When to run autofocus during a tiled acquisition.

    One vocabulary for both tilers; `fibsem.fm.structures` re-exports it.

    Every stored spelling resolves: the `EVERY_*` aliases below (saved by name), the
    lowercase values (saved by value), and the integers 0-3 of older files.
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
        # Older files' integers, in declaration order. bool is excluded
        # deliberately: True == 1 would otherwise silently resolve to ONCE.
        if isinstance(value, int) and not isinstance(value, bool):
            order = (cls.NONE, cls.ONCE, cls.EACH_ROW, cls.EACH_TILE)
            return order[value] if 0 <= value < len(order) else None
        return None


class TileOrderStrategy(NamedEnum, Enum):
    TYPEWRITER = "typewriter"  # rows always left-to-right
    SERPENTINE = "serpentine"  # alternating: row 0 L→R, row 1 R→L, ...
    SPIRAL = "spiral"  # outward clockwise spiral from centre tile


class AutoContrastMode(NamedEnum, Enum):
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
class ImageSettings(Configurable):
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

    # No gamma here: gamma is a display correction (the canvas's ContrastGammaControl),
    # and baking it into stored pixels cannot be undone.
    # The beam-bound fields take their unit, limits, choices, scale, step and decimals
    # from the beam's parameter, and an instrument's limits win over these.
    resolution: Tuple[int, int] = param(
        (1536, 1024), device_parameter="beam.resolution", display=Display("Resolution")
    )
    dwell_time: float = param(
        1e-6, device_parameter="beam.dwell_time", display=Display("Dwell Time")
    )
    hfw: float = param(
        150e-6, device_parameter="beam.hfw", display=Display("Field of View")
    )
    autocontrast: bool = param(
        False,
        doc="Set the detector's contrast and brightness automatically before the image is acquired.",
        display=Display("Auto Contrast"),
    )
    beam_type: BeamType = param(
        BeamType.ELECTRON,
        doc="The beam that acquires the image.",
        display=Display("Beam Type"),
    )
    save: bool = param(
        False, doc="Save the image after it is acquired.", display=Display("Save Image")
    )
    filename: str = param(
        "default_image",
        doc="The name the image is saved under, without its extension.",
        display=Display("Filename"),
    )
    path: Optional[Union[Path, str]] = param(
        None, doc="The folder the image is saved in.", display=Display("Path")
    )
    reduced_area: Optional[FibsemRectangle] = param(
        None,
        doc="Acquire only this part of the frame. Empty acquires the full frame.",
        display=Display("Reduced Area", advanced=True),
    )
    # None is off; a form shows it as 1.
    line_integration: Optional[int] = param(
        None, limits=RangeLimit(1, 255), display=Display("Line Integration")
    )
    scan_interlacing: Optional[int] = param(
        None, limits=RangeLimit(1, 8), display=Display("Scan Interlacing")
    )
    frame_integration: Optional[int] = param(
        None, limits=RangeLimit(1, 512), display=Display("Frame Integration")
    )
    drift_correction: bool = param(
        False,
        doc="Correct drift between integrated frames. Needs frame integration above 1.",
        display=Display("Drift Correction", advanced=True),
    )  # (bool) # requires frame_integration > 1

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
        return decode(ImageSettings, settings)

    def to_dict(self) -> dict:
        return encode(self)

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

        from fibsem import utils

        image_settings = deepcopy(image.metadata.image_settings)
        if image_settings.filename is None:
            image_settings.filename = utils.current_timestamp()
        image_settings.save = True

        return image_settings


@dataclass
class FocusStackSettings(Configurable):
    """Settings for focus-stack acquisition in tiled overview acquisition.

    Attributes:
        enabled: Whether to use focus stacking for each tile.
        n_steps: Number of vertical strips to divide each tile into.
        auto_focus: Whether to run autofocus for each strip.
    """

    enabled: bool = param(
        False,
        doc="Build each tile from strips focused one by one, for a tilted sample.",
        display=Display("Focus Stack"),
    )
    n_steps: int = param(
        3,
        limits=RangeLimit(1, 10),
        doc="The number of vertical strips each tile is divided into.",
        display=Display("Strips"),
    )
    auto_focus: bool = param(
        True, doc="Autofocus on each strip.", display=Display("Autofocus Each Strip")
    )

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(d: dict) -> "FocusStackSettings":
        return decode(FocusStackSettings, d)


def _default_autofocus_settings() -> "AutoFocusSettings":
    """The overview's focus sweep, which is simply the library default.

    Deliberately not a pinned copy of the values. `AutoFocusSettings()` is two passes --
    50 um at 5 um, then 10 um at 1 um -- and an overview wanting exactly that means it
    should *inherit* it, so a later improvement to the default reaches overviews too.


    Imported here rather than at module scope because `autofunctions.autofocus` imports
    this module for `BeamType`, so the arrow only points one way. `structures` already
    reaches downstream this way for `autofunctions.gamma` and `fm.structures`.
    """
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    return AutoFocusSettings()


# What an overview run looks like before anyone touches it. House defaults rather
# than physics: a 500 um tile and a 1 us dwell are what the overview has always
# opened with. The name decides where every overview is written:
# `TiledAcquisitionRunner._setup` makes the tile sub-folder from it.
OVERVIEW_TILE_HFW = 500e-6  # m


OVERVIEW_TILE_DWELL_TIME = 1e-6  # s


DEFAULT_OVERVIEW_FILENAME = "overview-image"


def _default_overview_tile_settings() -> ImageSettings:
    """The tile settings a new overview starts from.

    The standard 1536x1024 resolution, not a square one: `hfw` is the horizontal
    field, so the same 500 um tile at 1024x1024 is 0.49 um/px against 0.33 and half
    again as tall, and overviews worth putting side by side should not differ by it.
    A `tuple`, not the config list itself, so no two settings share one mutable list.
    """
    return ImageSettings(
        resolution=tuple(DEFAULT_STANDARD_RESOLUTION_LIST),
        hfw=OVERVIEW_TILE_HFW,
        dwell_time=OVERVIEW_TILE_DWELL_TIME,
        beam_type=BeamType.ELECTRON,
        save=True,
        path=None,  # whoever owns the experiment fills this in
        filename=DEFAULT_OVERVIEW_FILENAME,
    )


@dataclass
class OverviewAcquisitionSettings(Configurable):
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
        register_tiles: Experimental. Place each tile where its picture says it is,
            measured against its neighbours, rather than where the grid planned it.
            Only when the tiles match the stage steps (the layout check raises no
            warning) and at least three pairs match; otherwise the planned mosaic is
            kept. Size, centre and pixel size are unchanged either way.
        autofocus_mode: *When* to focus during the traversal (NONE, ONCE, EACH_ROW,
            EACH_TILE).
        autocontrast_mode: *When* to set the contrast (NONE, ONCE at the grid centre,
            EACH_TILE). The runner drives `image_settings.autocontrast` from it, so
            the per-image flag is not the switch here.
        autofocus_settings: *How* to focus -- sweep passes, method and probe frame.
    """

    image_settings: ImageSettings = param(
        doc="The settings each tile is acquired with; its field of view is the tile's.",
        default_factory=_default_overview_tile_settings,
    )
    nrows: int = param(
        3,
        limits=RangeLimit(1, 100),
        doc="The number of tile rows.",
        display=Display("Rows"),
    )
    ncols: int = param(
        3,
        limits=RangeLimit(1, 100),
        doc="The number of tile columns.",
        display=Display("Columns"),
    )
    overlap: float = param(
        0.1,
        limits=RangeLimit(0.0, 0.9),
        doc="How much neighbouring tiles overlap, as a fraction of a tile.",
        display=Display("Overlap", scale=100, unit="%", step=5.0, decimals=0),
    )
    focus_stack_settings: FocusStackSettings = param(
        doc="Focus stacking within each tile.", default_factory=FocusStackSettings
    )
    autofocus_mode: AutoFocusMode = param(
        AutoFocusMode.NONE,
        doc="When to autofocus during the run: never, once, at each row or at each tile.",
        display=Display("Autofocus"),
    )
    autofocus_settings: "AutoFocusSettings" = param(
        doc="How to autofocus: the sweep passes, the method and the probe frame.",
        default_factory=_default_autofocus_settings,
    )
    # One contrast for the whole mosaic by default; the runner sets the per-image
    # flag from this.
    autocontrast_mode: AutoContrastMode = param(
        AutoContrastMode.ONCE,
        doc="When to set the contrast: never, once at the grid centre, or at each tile. It decides image_settings.autocontrast for the run.",
        display=Display("Auto Contrast"),
    )
    tile_order: TileOrderStrategy = param(
        TileOrderStrategy.TYPEWRITER,
        doc="The order the tiles are visited in.",
        display=Display("Tile Order", advanced=True),
    )
    tile_mask: Optional[List[List[bool]]] = param(
        None,
        doc="Which tiles to acquire, as tile_mask[row][col]. Empty acquires every tile; a skipped tile keeps its place in the mosaic.",
        display=Display(hidden=True),
    )
    register_tiles: bool = param(
        False,
        doc="Experimental. Place each tile where its picture says it is, measured against its neighbours, rather than where the grid planned it. The planned mosaic is kept when the tiles don't match well enough.",
        display=Display("Register Tiles", advanced=True),
    )

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

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        # autofocus imports this module, so its class is named here only while the
        # schema is built
        from fibsem.autofunctions.autofocus import AutoFocusSettings

        globals()["AutoFocusSettings"] = AutoFocusSettings
        try:
            return handler(source)
        finally:
            del globals()["AutoFocusSettings"]

    @classmethod
    def from_dict(cls, d: dict) -> "OverviewAcquisitionSettings":
        """Read settings saved by any version (`fibsem.codec.upgrades.overview`)."""
        return decode(cls, acquisition_settings(d))

    def to_dict(self) -> dict:
        return encode(self)


# ---------------------------------------------------------------------------
# The sample holder
#
# Here rather than with the sample handling service (`fibsem.services.sample_handling`)
# so that `SystemSettings` can hold one: the service imports from this module, so a
# holder field on `SystemSettings` would close an import loop. The inventory rows
# (`GridSlotState`, `GridInventoryEntry`) sit here with them.
# ---------------------------------------------------------------------------

GRID_RADIUS = 1e-3  # 1mm


@dataclass
class SampleGrid(Configurable):
    """A physical TEM grid or sample that can be loaded into a GridSlot."""

    name: str
    description: str = ""
    radius: float = param(
        GRID_RADIUS,
        unit="m",
        doc="Radius of the sample grid",
        display=Display(scale=1e3),
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleGrid":
        return decode(cls, data)


def _blank_is_none(value: Any) -> Any:
    # older holder files wrote "" for a time nobody captured
    return value or None


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
    # When the wizard captured it: aware, written as ISO 8601 with its offset. Older
    # records hold a naive string, which stays naive. None for a position nobody
    # captured.
    captured_at: Annotated[Optional[Timestamp], BeforeValidator(_blank_is_none)] = None
    fibsem_version: str = ""

    def __post_init__(self) -> None:
        # the string a caller holds
        self.captured_at = to_datetime(self.captured_at) if self.captured_at else None

    @classmethod
    def builtin(cls, pre_tilt: float, rotation_reference: float) -> "SlotCalibration":
        """A position nobody captured: the compustage working slot's nominal one.

        The autoloader puts every grid at the same place, nominally the stage
        origin, so the slot starts there with this record. A real instrument puts
        it a fixed distance off, so a position captured with the calibration wizard
        replaces it once there is one.
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
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SlotCalibration":
        return decode(cls, data)


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

    def __post_init__(self) -> None:
        if self.position is not None:
            self.position.name = self.name

    @property
    def is_calibrated(self) -> bool:
        return self.position is not None and self.calibration is not None

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GridSlot":
        """Read a slot saved by any version (`fibsem.codec.upgrades.holder`)."""
        return decode(cls, holder_upgrades.slot(data))


@dataclass
class SampleHolder(Configurable):
    # First, and with no default, so it cannot be left out. The pre-tilt is a property
    # of *this holder* -- swap a 35 degree shuttle for a flat one and it changes with
    # the shuttle, which is why it stopped being a field on the stage.
    #
    # Required because the alternatives both fail quietly. A default of 0.0 turns a
    # construction site that forgot into a flat shuttle and wrongs every projection
    # made from it, with nothing to report; a `None` sentinel spreads its own handling
    # into every reader, and one reader that formats it instead becomes a hard abort
    # (PyQt5 turns an exception in a slot into `qFatal`). Ordered first because a
    # dataclass cannot put a non-default field after defaulted ones.
    #
    # Absence in a *file* is a different question and is not this field's to answer:
    # `from_dict` supplies the configured value, and `_resolve_configured_holder`
    # seeds it at connect.
    # The stage reads the holder, so the holder never reads the stage.
    pre_tilt: float = param(
        unit="°",
        limits=RangeLimit(-90.0, 90.0),
        doc="Pre-tilt of this holder, in degrees",
        display=Display(decimals=1),
    )
    name: str = param("Sample Holder", doc="Name of the sample holder")
    description: str = param("", doc="Description of the sample holder")
    capacity: int = param(
        2, limits=RangeLimit(1, 12), doc="Number of grid slots on this holder"
    )
    slots: dict[str, GridSlot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._parent_ref: Optional["weakref.ReferenceType"] = None

    # The microscope this holder is mounted on, held WEAKLY. A holder lives in
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

    def discard_untrusted_positions(
        self, pre_tilt: float, rotation_reference: float
    ) -> List[str]:
        """Drop every slot position that was not calibrated against this geometry.

        A position with no calibration record was captured at an unknown
        orientation; one whose record disagrees with the current
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
        # By index, not name: a slot may have a name of its own (the built-in
        # shuttle's), and is still the slot at its index.
        present = {slot.index for slot in self.slots.values()}
        for i in range(self.capacity):
            name = f"Slot-{i + 1:02d}"
            if i not in present and name not in self.slots:
                # A new slot has no position until it is calibrated; inventing one
                # at the origin was a number that looked like a measurement.
                self.slots[name] = GridSlot(name=name, index=i, position=None)
        for name in [
            n for n, s in list(self.slots.items()) if s.index >= self.capacity
        ]:
            del self.slots[name]

    def to_dict(self, include_grids: bool = True) -> dict:
        data = encode(self)
        if not include_grids:
            for slot in data["slots"].values():
                slot["loaded_grid"] = None
        return data

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
        # A file that does not state a pre-tilt reads as 0.0, which is safe only
        # because `_resolve_configured_holder` overwrites it with the configured one
        # before the holder is used. A file with no capacity holds the slots it lists.
        data = holder_upgrades.holder(data)
        holder = decode(
            cls,
            {
                "capacity": max(len(data.get("slots") or {}), 1),
                **data,
                "pre_tilt": float(data.get("pre_tilt") or 0.0),
            },
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
        are session state (``fibsem.services.sample_handling.save_holder_occupancy``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.to_dict(include_grids=False),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


class GridSlotState(str, Enum):
    """Where a grid is, as one word: the state of an inventory slot.

    ``UNKNOWN`` -- no inventory has answered for this slot yet; ``EMPTY`` -- the
    slot holds no grid; ``OCCUPIED`` -- a grid is in the slot and not on the
    microscope; ``LOADED`` -- the grid is in the microscope's sample holder. On a
    fixed holder a named slot is LOADED outright: its grid is already in the
    holder, and reaching it is only a stage move.
    """

    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    LOADED = "loaded"


@dataclass
class GridInventoryEntry:
    """One row of ``SampleHandling.grid_inventory()``: a slot, and what it holds.

    ``source`` says which hardware answered -- ``"magazine"`` on a system with a
    loader, ``"holder"`` otherwise. ``state`` is the one stored fact; ``present``
    (the grid is available to the system) and ``loaded`` (it is in the holder
    right now) are read off it. Kept apart on purpose: *present* is what a run
    can select from, *loaded* is what it can act on without an exchange.
    """

    slot_name: str
    index: int
    source: str
    name: Optional[str]
    state: GridSlotState

    @property
    def present(self) -> bool:
        return self.state in (GridSlotState.OCCUPIED, GridSlotState.LOADED)

    @property
    def loaded(self) -> bool:
        return self.state is GridSlotState.LOADED


# The holder a system starts with when nothing else describes one: no `stage.holders`
# in the configuration, and no `sample-holder.yaml` beside it. A function rather than a
# module constant because a holder is mutable and gets a `_parent` bound to it; one
# shared instance would be handed to every microscope.
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


TFibsemPatternSettings = TypeVar(
    "TFibsemPatternSettings", bound="FibsemPatternSettings"
)


class MillingState(NamedEnum, Enum):
    IDLE = 0
    RUNNING = 1
    STOPPING = 2
    PAUSED = 3
    ERROR = 4
    # The state could not be read. Not an error and not a guess -- a producer saying so
    # deliberately, because on ThermoFisher reading the milling state is a getter that
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
# condition -- `while milling.state.get_value() in ACTIVE_MILLING_STATES` guards the milling
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


########### Base Pattern Settings
@dataclass
class FibsemPatternSettings(Tagged, ABC):
    # Keys a dict may carry beyond the fields: the shape it names, which
    # `from_dict` reads to pick the class (`fibsem.devices.wire` accepts them).
    FROM_DICT_KEYS: ClassVar[Tuple[str, ...]] = ("type",)

    def to_dict(self) -> Dict[str, Any]:
        return encode(self)

    @classmethod
    def from_dict(
        cls: Type[TFibsemPatternSettings], data: Dict[str, Any]
    ) -> TFibsemPatternSettings:
        """Read a pattern; read as the base class, the dict names its ``type``."""
        return decode(cls, data)

    @classmethod
    def subclass_named(cls, name: str) -> Type[FibsemPatternSettings]:
        """The shape a pattern dict names in its ``type``, as the App Server's
        ``draw_patterns`` takes it: ``"Rectangle"`` (or ``"FibsemRectangleSettings"``)."""
        shapes: Dict[str, Type[FibsemPatternSettings]] = {}
        pending = list(cls.__subclasses__())
        while pending:
            sub = pending.pop()
            pending += sub.__subclasses__()
            shapes[sub.__name__] = sub
            shapes[sub.__name__.replace("Fibsem", "").replace("Settings", "")] = sub
        if name not in shapes:
            available = sorted(name for name in shapes if not name.startswith("Fibsem"))
            raise ValueError(f"Unknown pattern type: {name!r}. Available: {available}")
        return shapes[name]

    @property
    @abstractmethod
    def volume(self) -> float:
        pass


class CrossSectionPattern(NamedEnum, Enum):
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
    array: InitVar[Optional[Array]] = None
    bitmap: Optional[Array] = field(init=False)
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
    vertices: Array  # n[x, y]
    depth: float
    is_exclusion: bool = False

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)

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
class DriverMillingSettings(Configurable):
    """A driver's own milling settings: those no other driver means the same by
    (Tescan's preset and rate). A driver's registry entry names its class
    (``DriverEntry.milling_settings``), and a recipe keeps them in its ``driver``
    block under the driver's name. A value outside a field's limits is refused."""

    def __post_init__(self) -> None:
        for name, spec in field_specs(type(self)).items():
            value = getattr(self, name)
            limits = spec.limits
            if limits is None or not isinstance(value, (int, float)):
                continue
            if not limits.min <= value <= limits.max:
                raise ValueError(
                    f"{name} {value!r} is outside {limits.min!r} to {limits.max!r}"
                )


def _driver_milling_classes() -> Dict[str, Type[DriverMillingSettings]]:
    from fibsem.drivers.registry import milling_settings_classes

    return milling_settings_classes()


def _driver_name(name: str) -> str:
    from fibsem.drivers.registry import normalize_name

    return normalize_name(name)


# Each driver's own block in a recipe, by driver name.
_DRIVER_BLOCKS = Blocks(_driver_milling_classes, _driver_name)


@dataclass
class FibsemMillingSettings(Configurable):
    """A milling recipe, as a stage saves it.

    Its fields are the ones fibsem defines once: the channel, the field of view and
    the settings more than one driver mills with. A driver lists those it uses
    (``Milling.setting_names``); a driver's own settings are in ``driver``, a block
    per driver by its registry name (``{"Tescan": {"rate": ...}}``), each read as
    the class the driver's registry entry names. A recipe saved on one instrument
    loads, and saves back unchanged, on another.
    """

    # The fields bound to a microscope parameter take its unit, limits, choices and
    # display from the milling beam, and an instrument's own win over these.
    milling_current: float = param(
        20.0e-12,
        device_parameter="beam.current",
        doc="The current used for milling. Higher currents mill faster but with less precision and more damage.",
        display=Display("Milling Current"),
    )
    milling_voltage: float = param(
        30e3,
        device_parameter="beam.voltage",
        doc="The voltage used for milling. Higher voltages provide higher energy ions for milling.",
        display=Display("Milling Voltage", advanced=True),
    )
    application_file: str = param(
        "Si",
        device_parameter="beam.application_file",
        doc="The application file used for milling, on Thermo Fisher instruments only. Note: this can be changed at runtime depending on the pattern and other parameters.",
        display=Display("Application File", advanced=True),
    )
    patterning_mode: str = param(
        "Serial",
        choices=["Serial", "Parallel"],
        doc="The patterning mode used for milling. 'Serial' mills the entire pattern in one pass, 'Parallel' mills multiple pattern simultaneously.",
        display=Display("Patterning Mode", advanced=True),
    )
    hfw: float = param(
        150e-6,
        limits=RangeLimit(20e-6, 950e-6),
        device_parameter="beam.hfw",
        doc="The horizontal field width used for milling. Patterns must fit within this field of view.",
        display=Display("Field of View", hidden=True),
    )
    milling_channel: BeamType = param(
        BeamType.ION,
        choices=[BeamType.ION, BeamType.ELECTRON],
        doc="The beam channel used for milling.",
        display=Display("Milling Channel", hidden=True),
    )
    acquire_images: bool = param(
        False,
        doc="Whether to acquire images after milling.",
        display=Display("Acquire Images", hidden=True),
    )
    driver: Annotated[Dict[str, Any], _DRIVER_BLOCKS] = param(
        default_factory=_DRIVER_BLOCKS.defaults,
        doc="Each driver's own settings, a block per driver by its name (Tescan). "
        "An instrument reads its driver's block and keeps the others.",
        display=Display("Driver Settings", hidden=True),
    )

    def __post_init__(self):
        for name in ("milling_current", "milling_voltage", "hfw"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)):
                raise TypeError(
                    f"{name} must be a number, currently {type(value).__name__}"
                )
        for name in ("patterning_mode", "application_file"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a string, currently {type(value).__name__}"
                )

    def setting(self, name: str) -> Any:
        """A field by its name, or a driver's own by its path (``driver.Tescan.preset``,
        as ``Milling.supported_settings`` names it)."""
        if name.startswith("driver."):
            _, block, field_name = name.split(".", 2)
            return getattr(self.driver[block], field_name)
        return getattr(self, name)

    def set_setting(self, name: str, value: Any) -> None:
        """Set a field by its name, or a driver's own by its path."""
        if name.startswith("driver."):
            _, block, field_name = name.split(".", 2)
            self.driver[block] = replace(self.driver[block], **{field_name: value})
        else:
            setattr(self, name, value)

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, settings: dict) -> "FibsemMillingSettings":
        return decode(cls, settings)

    @property
    def advanced_attributes(self) -> Set[str]:
        """Return a set of advanced attribute names."""
        return set(flagged_fields(self.__class__, "advanced"))

    def summary(self) -> str:
        # Here, not at the top: fibsem.utils imports fibsem.structures.system.
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


DEFAULT_ALIGNMENT_AREA = {"left": 0.7, "top": 0.3, "width": 0.25, "height": 0.4}


@saved(absent={"enabled": False})
@dataclass
class MillingAlignment(Configurable):
    """Drift correction settings for milling"""

    enabled: bool = param(
        True,
        doc="Align between the imaging and milling currents before milling starts.",
        display=Display("Enable Initial Alignment"),
    )
    interval_enabled: bool = param(
        False,
        display=Display("Re-align during milling"),
        not_used="drift correction during a mill is not built yet (FIB-13); "
        "the task aligns before each stage only",
    )
    interval: int = param(
        30,
        unit="s",
        display=Display("Re-align every"),
        not_used="drift correction during a mill is not built yet (FIB-13)",
    )
    rect: FibsemRectangle = field(
        default_factory=lambda: FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
    )
    use_autocontrast: bool = param(
        True,
        doc="Auto contrast before the alignment image is acquired.",
        display=Display("Use Auto Contrast"),
    )
    use_autofocus: bool = param(
        False,
        doc="Auto focus before the alignment image is acquired.",
        display=Display("Use Auto Focus"),
    )
    steps: int = param(
        3,
        limits=RangeLimit(1, 9),
        doc="How many times the alignment is repeated.",
        display=Display("Alignment Steps"),
    )
    imaging: ImageSettings = field(default_factory=ImageSettings)
    # At polishing currents a full frame costs about the dose of an ordinary
    # image and gives the match far more to lock onto than a small area does.
    full_frame_below: float = param(
        0.0,
        unit="A",
        limits=RangeLimit(0.0, 100e-9),
        doc="Align on the whole frame instead of the alignment area when the "
        "milling current is at or below this. Off at 0.",
        display=Display("Full Frame At Or Below", scale=1e12, step=10, decimals=0),
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MillingAlignment":
        return decode(cls, data)


@dataclass
class ReferenceImageParameters(Configurable):
    imaging: ImageSettings = param(
        default_factory=ImageSettings,
        doc="The settings the reference images are acquired with; each image's field of view is field_of_view1 or field_of_view2.",
    )
    field_of_view1: float = param(100e-6, doc="Field of view for first reference image")
    field_of_view2: float = param(
        150e-6, doc="Field of view for second reference image"
    )
    acquire_sem: bool = param(True, doc="Whether to acquire SEM reference images")
    acquire_fib: bool = param(True, doc="Whether to acquire FIB reference images")
    acquire_image1: bool = param(True, doc="Whether to acquire first reference image")
    acquire_image2: bool = param(True, doc="Whether to acquire second reference image")

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceImageParameters":
        return decode(cls, data)

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
