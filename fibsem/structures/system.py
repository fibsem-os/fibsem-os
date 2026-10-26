"""The microscope's configuration and devices, and who is working in which session."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from datetime import tzinfo
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import fibsem
from fibsem.codec import NamedEnum, Timestamp, decode, encode, saved
from fibsem.structures.params import Configurable, Display, RangeLimit, param
from fibsem.structures.settings import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemStagePosition,
    ImageSettings,
    SampleHolder,
)
from fibsem.util.timestamps import now, utc_offset_of, zone_from_offset
from fibsem.versioning import get_revision

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceConfiguration
    from fibsem.microscope import FibsemMicroscope


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
# `StageMovement.update_orientations`. The poses themselves are computed from physical parameters
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
    out of the same two questions.

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
    # parked at the device is refused.
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
    and which of the two does the discriminating is a fact about the
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
    # write one that contradicts its own geometry. Which *named* orientations an
    # instrument can image from is a different kind of fact: it is how the device is
    # bolted on, nothing derives it, and a list of names can only reference the derived
    # poses, never disagree with them.
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

    # The pose an offset FM images in, r and t in radians, each left out taking the
    # FIB pose's: the one orientation a site states rather than derives, because how
    # the objective is bolted on decides it and nothing else does. Stated, it is the
    # "FM" orientation and the camera's tilt follows from it
    # (`StageMovement.camera_tilt`). None is the FIB pose, as every offset FM has been.
    pose: Optional[FibsemStagePosition] = None

    def contains(self, stage_position: FibsemStagePosition) -> bool:
        """Is `stage_position` within this device's `range` of its origin?

        Each device has its own range, so a traverse is not invertible by
        construction: the stage travels by the *difference* between two origins and
        keeps its offset, so a grid position 15 mm along at the beams arrives 15 mm
        along at the FM, which may not count that as arrived. So a traverse checks
        both ends -- the source's range at the start, the target's for where it will
        arrive -- before it moves (`FibsemMicroscope.move_to_device`).

        Not the same question as whether the device usefully *covers* the sample
        here -- an objective's field, a knife's approach -- which genuinely does vary
        per device. Nothing needs that yet.

        Devices may overlap, and on a compustage they fully do: the objective is under
        the grid, so the beams and the FM are the same place reached by flipping. The
        device axis is degenerate there and this question is not the one to ask -- see
        `StageMovement.current_device`.

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
        if self.pose is not None:
            ddict["pose"] = {
                axis: float(np.degrees(getattr(self.pose, axis)))
                for axis in ("r", "t")
                if getattr(self.pose, axis) is not None
            }
        return ddict

    @staticmethod
    def from_dict(
        ddict: dict,
        default_orientations: Sequence[str] = (),
    ) -> "StageDeviceSettings":
        """Read a device's stage position.

        `available_orientations` missing is the device type's default; stated empty is
        an error. `range` missing is 1 mm on each axis the origin sets.
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
        elif "pose" in ddict:
            orientations = ["FM"]
        else:
            orientations = list(default_orientations) or list(BEAM_ORIENTATIONS)
        pose = _device_pose_from_dict(ddict.get("pose"))
        if pose is not None and "FM" not in orientations:
            raise ValueError(
                f"pose is the FM orientation, but available_orientations is "
                f"{orientations}: list FM there, or leave it out."
            )
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
        else:
            range_ = default_device_range(origin)
        # A range covers every axis the origin sets; one it leaves out gets the
        # default. Otherwise `contains` answers no on that axis, and the device -- a
        # `{x: 20 mm}` range on an origin that also sets y -- could never be arrived at.
        for axis in DEVICE_AXES:
            if getattr(origin, axis) is not None and getattr(range_, axis) is None:
                setattr(range_, axis, DEFAULT_DEVICE_RANGE_EXTENT)
        return StageDeviceSettings(
            origin=origin, available_orientations=orientations, range=range_, pose=pose
        )


def _device_pose_from_dict(ddict: Optional[dict]) -> Optional[FibsemStagePosition]:
    """A device entry's ``pose``: r and t in degrees, read into radians."""
    if ddict is None:
        return None
    if not isinstance(ddict, dict) or not set(ddict) <= {"r", "t"} or not ddict:
        raise ValueError(f"pose takes r and t, in degrees: {ddict!r}")
    return FibsemStagePosition(
        **{axis: float(np.radians(float(value))) for axis, value in ddict.items()}
    )


def _stage_position_to_dict(position: FibsemStagePosition) -> Dict[str, float]:
    return {
        axis: float(getattr(position, axis))
        for axis in ("x", "y", "z", "r", "t")
        if getattr(position, axis) is not None
    }


@dataclass
class DeviceCalibration:
    """Where the stage travels between the beams and an offset FM, measured.

    Both origins are the centre of the holder's slot 1: ``fibsem`` with it at the
    beams' coincidence point, in the pose the FM images from, and ``fm`` with it
    under the objective, read at the FM. The stage travels by ``fm - fibsem``, in
    place of the FM entry's nominal ``origin``, which stays for the calibration's
    own first trip. ``holder`` is the holder it was measured on (None: any, as
    the simulator's file states); ``sites`` are each site's measurements.
    """

    fibsem: FibsemStagePosition
    fm: FibsemStagePosition
    holder: Optional[str] = None
    when: Optional[str] = None
    sites: List[Dict[str, Any]] = field(default_factory=list)

    def translation(self) -> FibsemStagePosition:
        """The stage move from the beams to the FM, on the device axes."""
        return FibsemStagePosition(
            **{
                axis: getattr(self.fm, axis) - getattr(self.fibsem, axis)
                for axis in DEVICE_AXES
                if getattr(self.fm, axis) is not None
                and getattr(self.fibsem, axis) is not None
            }
        )

    def holds_for(self, holder: str) -> bool:
        """Was it measured on *holder*, or on no holder in particular?"""
        return self.holder is None or self.holder == holder

    def to_dict(self) -> dict:
        ddict: Dict[str, Any] = {
            "fibsem": {"origin": _stage_position_to_dict(self.fibsem)},
            "fm": {"origin": _stage_position_to_dict(self.fm)},
            "holder": self.holder,
        }
        if self.when is not None:
            ddict["when"] = self.when
        if self.sites:
            ddict["sites"] = list(self.sites)
        return ddict

    @staticmethod
    def from_dict(ddict: dict) -> "DeviceCalibration":
        def origin(name: str) -> FibsemStagePosition:
            values = (ddict.get(name) or {}).get("origin")
            if not values:
                raise ValueError(f"calibration.devices.{name} states no origin.")
            return FibsemStagePosition(**{k: float(v) for k, v in values.items()})

        return DeviceCalibration(
            fibsem=origin("fibsem"),
            fm=origin("fm"),
            holder=ddict.get("holder"),
            when=ddict.get("when"),
            sites=list(ddict.get("sites") or []),
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
# A compustage FM's origin is travelled by after the flip: the offset of the objective
# from the beams' coincidence point, at the FM pose. One question is asked of both
# mountings instead of each caller branching on the stage type.
#
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
# reprojection used before drivers reported one, calibrated on one instrument.
# Only for reading images saved before the centre was recorded, all of
# which were drawn with it; a driver that reports no centre records None. A driver
# now reports its instrument's own centre plus `rotation_centre_correction`.
LEGACY_ROTATION_CENTRE: Tuple[float, float] = (
    -0.0005127403888932854 + 25e-6,
    0.0007937916666666666 + 12.5e-6,
)


# The frame a stage's positions are in: fibsem's, which every stage device
# reports. A driver that records positions in its instrument's own frame names that
# frame itself, in its own module; the core keeps such a name as an opaque string.
STAGE_FRAME_FIBSEM = "fibsem"


def _parse_rotation_centre(value) -> Optional[Tuple[float, float]]:
    """A stored rotation centre as an (x, y) tuple of floats, or None."""
    if value is None:
        return None
    x, y = value
    return (float(x), float(y))


@dataclass
class StageSystemSettings(Configurable):
    rotation_reference: float = param(
        unit="°",
        limits=RangeLimit(-360.0, 360.0),
        doc="The stage rotation the SEM orientation is defined at; on a stage that "
        "rotates, the FIB orientation is 180° from it.",
        display=Display("Reference rotation", decimals=2),
    )
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
    # stage's own axes.
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
    milling_angle: float = param(
        15,
        unit="°",
        limits=RangeLimit(0.0, 45.0),
        doc="The milling angle is the difference between the stage and the fib "
        "viewing angle.",
        display=Display("Milling Angle", step=1.0, decimals=1),
    )
    # Where the stage travels for each instrument to see the sample. Keyed by device
    # name -- "FIBSEM" and "FM" today -- and separate from `orientations`, which says
    # what pose the sample is held in once the stage is there.
    #
    # Configured on each device's entry (`origin`, `available_orientations`, `range`)
    # rather than on the stage; `SystemSettings.from_dict` fills this from them. Each
    # device's `range` is how far from it the stage can be and still count as having
    # travelled to it. Not to be confused with the stage device's `position.limits`, which is
    # how far the axes can physically move.
    devices: Dict[str, StageDeviceSettings] = field(
        default_factory=lambda: deepcopy(DEFAULT_STAGE_DEVICES)
    )
    # The holders this system has, and which one is on the stage. A keyed map with a
    # selection rather than a single holder, because a site that swaps a flat shuttle
    # for a pre-tilted one should select the other entry rather than re-enter its
    # geometry -- and, once pre-tilt moves onto the holder, re-calibrate for it.
    #
    # Empty by default. The sample handling service fills it, importing a `sample-holder.yaml`
    # if the site has one; nothing here reads that file, so a `SystemSettings` built
    # from a dict stays a pure function of that dict.
    holders: Dict[str, "SampleHolder"] = field(default_factory=dict)
    active_holder: str = ""
    # How far the true centre of a half turn sits from where the instrument's own
    # compucentric rotation puts it, in raw stage coordinates (x, y), metres.
    # Zero means the instrument's rotation is taken as perfect. Measured per
    # instrument: centre a feature, rotate, centre it again, and the midpoint of the
    # two positions less the instrument's centre is this value. Stage moves and image
    # reprojection both add it.
    rotation_centre_correction: Tuple[float, float] = (0.0, 0.0)
    # The SEM to FM calibration of an offset FM (`calibration.devices`), or None
    # until it is run.
    device_calibration: Optional[DeviceCalibration] = None

    def fm_calibrated(self) -> bool:
        """Is there an SEM to FM calibration that holds for the fitted holder?"""
        calibration = self.device_calibration
        return calibration is not None and calibration.holds_for(self.active_holder)

    def calibrated_device(self, name: str) -> StageDeviceSettings:
        """The device *name*, with the FM's origin from the SEM to FM calibration
        when it holds: the beams' origin plus the measured travel, on the axes the
        FM's entry sets. Raises KeyError for a device there is not."""
        device = self.devices[name]
        if name != "FM" or not self.fm_calibrated():
            return device
        beams = self.devices.get(BEAMS_STAGE_DEVICE)
        travel = self.device_calibration.translation()
        origin = deepcopy(device.origin)
        for axis in DEVICE_AXES:
            start = None if beams is None else getattr(beams.origin, axis)
            if getattr(origin, axis) is None or getattr(travel, axis) is None:
                continue
            setattr(origin, axis, (start or 0.0) + getattr(travel, axis))
        return replace(device, origin=origin)

    @property
    def rotation_180(self) -> float:
        """Where the stage sits to face the ion beam, in degrees.

        Derived, not configured: `(rotation_reference + 180) % 360` (Tescan's reference
        of 180 is why the modulo matters). A stage that does not turn round says so
        with `rotation`, which is `false` on the Arctis.

        Readers move to the whole FIB pose: a rotation alone does not say
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
        configuration has no holder until the sample handling service resolves one, and
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

    def to_dict(self) -> dict:
        data = encode(self)
        del data["_shuttle_pre_tilt"]  # written as shuttle_pre_tilt, below
        # Each device's position is written on its own entry (`SystemSettings`).
        del data["devices"]
        # The configuration names the holders; what is loaded is the holder file's.
        for holder in data["holders"].values():
            for slot in holder["slots"].values():
                slot["loaded_grid"] = None
        if not any(self.rotation_centre_correction):
            del data["rotation_centre_correction"]
        # Written under `calibration.devices` (`SystemSettings`).
        if self.device_calibration is None:
            del data["device_calibration"]
        else:
            data["device_calibration"] = self.device_calibration.to_dict()
        # The pre-tilt has one home: the holder once one is named, this key until then.
        if not self.holders:
            data["shuttle_pre_tilt"] = self.shuttle_pre_tilt
        return data

    @classmethod
    def from_dict(cls, settings: dict) -> "StageSystemSettings":
        holders = {
            name: _configured_holder_from(name, holder)
            for name, holder in (settings.get("holders") or {}).items()
        }
        active = holders.get(settings.get("active_holder", ""))
        settings = dict(settings)
        calibrated = settings.pop("device_calibration", None)
        stage = decode(
            cls,
            {
                "rotation_reference": 0.0,
                "shuttle_pre_tilt": active.pre_tilt if active is not None else 0.0,
                **settings,
                "devices": deepcopy(DEFAULT_STAGE_DEVICES),
                "holders": holders,
                "rotation_centre_correction": _parse_rotation_centre(
                    settings.get("rotation_centre_correction")
                )
                or (0.0, 0.0),
            },
        )
        if calibrated:
            stage.device_calibration = DeviceCalibration.from_dict(calibrated)
        return stage


def _detector_block_from(settings: dict) -> dict:
    """The detector keys of a beam block (``detector_type``), in the names
    `FibsemDetectorSettings` reads (``type``)."""
    return {
        name: settings[f"detector_{name}"]
        for name in ("type", "mode", "brightness", "contrast")
        if f"detector_{name}" in settings
    }


# Written by `BeamSettings` / `FibsemDetectorSettings` and not defaults: the column's
# alignment (working distance, stigmation, beam shift) and what the last autocontrast
# left on the detector. A configuration does not record them.
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
    # The plasma source's gas, or None for a column with no plasma source.
    plasma_gas: Optional[str] = None

    @property
    def plasma(self) -> bool:
        """Whether this is a plasma column. Derived: a plasma column has a gas."""
        return self.plasma_gas is not None

    def to_dict(self) -> dict:
        """The configuration's flat beam block: the beam and detector settings sit
        beside the column's own keys, the detector's with a ``detector_`` prefix."""
        data = encode(self)
        data.update(data.pop("beam"))
        data["current"] = data.pop("beam_current")
        data.update({f"detector_{k}": v for k, v in data.pop("detector").items()})
        # An electron column has no plasma source; no shipped file carries the key.
        if self.beam_type is not BeamType.ION:
            del data["plasma_gas"]
        return data

    @classmethod
    def from_dict(cls, settings: dict) -> "BeamSystemSettings":
        beam_type = BeamType[settings.get("beam_type", "ELECTRON")]
        # An absent column tilt depends on the column: `FibsemHardwareGeometry`'s
        # 52 degrees for the ion column, 0 for the electron one.
        column_tilt = DEFAULT_FIB_COLUMN_TILT if beam_type is BeamType.ION else 0.0
        return decode(
            cls,
            {
                "enabled": True,
                "eucentric_height": 0.0,
                "column_tilt": column_tilt,
                **settings,
                "beam_type": beam_type,
                "beam": settings,
                "detector": _detector_block_from(settings),
            },
        )


@dataclass
class ManipulatorSystemSettings:
    # Whether a manipulator is fitted, as `FibsemClient` reads it from the server. A
    # microscope answers from its devices (`is_available("manipulator")`), and the
    # axes the arm has from its device too, so neither is set here.
    enabled: bool = True

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ManipulatorSystemSettings":
        return decode(cls, data)


@dataclass
class SystemInfo:
    """Which instrument, and what software is running on it. Provenance.

    Owns both facts, and is the only place either belongs.
    ``serial_number`` is the instrument identity -- the key any per-instrument
    aggregation joins on, and the only field that distinguishes two of the same
    model in one facility.

    The version fields were duplicated on ``FibsemExperimentRef`` until v5, which
    removed them from there. What software is running is a property of
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

    name: str = "Unknown"
    ip_address: str = "Unknown"
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str = "Unknown"
    hardware_version: str = "Unknown"
    software_version: str = "Unknown"
    fibsem_version: str = fibsem.__version__
    application: Optional[str] = None
    # The commit actually running, when installed from a source checkout. None
    # for a wheel install. default_factory, not a plain default, so the lookup
    # happens on first use rather than at import of this module. A lambda, so the
    # lookup goes through this module's name (tests pin it).
    fibsem_revision: Optional[str] = field(default_factory=lambda: get_revision())
    # The port the driver connects on. None uses the port its driver registers
    # (``fibsem.drivers.registry``): 7520 for ThermoFisher, 8300 for Tescan.
    port: Optional[int] = None

    def __post_init__(self) -> None:
        # Here, not at the top: the client imports this module and no driver code.
        from fibsem.drivers.registry import normalize_name

        # Configurations and old experiments carry "Thermo", "TESCAN" and so on;
        # everything downstream compares against the canonical spellings.
        self.manufacturer = normalize_name(self.manufacturer)

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(settings: dict) -> "SystemInfo":
        return decode(SystemInfo, settings)


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

    # Which driver the FM comes from. `None` follows the microscope's own driver -- the
    # iFLM and the Arctis FM are on the AutoScript connection, the Odemis stack drives
    # its own -- which is every site today. `FM_DRIVER_REMOTE` is an FM on its own PC,
    # reached at `address`:`port` (a METEOR beside natively driven beams). The address
    # belongs to the driver, so it is read only with one.
    driver: Optional[str] = None
    address: Optional[str] = None
    port: Optional[int] = None

    # A remote FM whose server isn't answering at connect is built offline and comes
    # online by itself, so the beams are never held up by the FM's PC.
    # `required: true` makes the connect fail instead, for a site where an FM that is
    # quietly missing would be worse than no session.
    required: Optional[bool] = None

    # The objective's calibration, in metres: where it is in focus, and how far it
    # may be inserted. Measured at this instrument, so it is written under
    # `calibration.objective` by `SystemSettings.to_dict` rather than in this
    # block. `None` means the configuration does not state one, and the FM
    # working state (`fm.working` in the session state) answers instead -- so
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
    # stage moves project a displacement in the FM image through. Absent,
    # it follows from the mount: 180 where the stage turns the grid over to the FM (a
    # compustage), the ion column's tilt for an FM beside the FIB column.
    camera_tilt: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mount_transform is None:
            self.mount_transform = CameraImageTransform.NONE

    def to_dict(self) -> dict:
        """The fm entry's own keys, as the configuration writes them and as every FM
        binder receives them (``config``). The objective positions are written under
        ``calibration.objective`` (`objective_to_dict`)."""
        data = encode(self)
        del data["focus_position"], data["limit_position"]
        # By value (`flip-x`), the spelling Odemis' --mount-transform takes too.
        if self.mount_transform is CameraImageTransform.NONE:
            del data["mount_transform"]
        else:
            data["mount_transform"] = self.mount_transform.value
        if self.camera_tilt is None:
            del data["camera_tilt"]
        return data

    def objective_to_dict(self) -> dict:
        return {
            "focus_position": self.focus_position,
            "limit_position": self.limit_position,
        }

    @classmethod
    def from_dict(cls, settings: dict) -> "FluorescenceSystemSettings":
        settings = dict(settings or {})
        settings["mount_transform"] = _mount_transform(settings.get("mount_transform"))
        return decode(cls, settings)


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
    "gis",
    "sputter_coater",
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
    def from_dict(data: dict) -> "DeviceEntry":
        """Read one entry."""
        data = dict(data or {})
        name = data.pop("name", None)
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
STAGE_POSITION_KEYS = ("origin", "available_orientations", "range", "pose")


def read_device_entries(settings: dict) -> Dict[str, DeviceEntry]:
    """Every device a configuration dict names, by name, in file order.

    Reads the list (`hardware.devices:`); a file in an older shape is upgraded first
    (`fibsem.codec.upgrades.configuration`). Two entries with one name are an error:
    which of them the file meant cannot be known. A list entry with no type whose name
    is no device type is ignored with a warning, so a file naming a device this version
    no longer has still loads.
    """
    settings = settings or {}
    hardware = settings.get("hardware") or {}
    entries: Dict[str, DeviceEntry] = {}
    listed = hardware.get("devices") or []
    if not isinstance(listed, list):
        raise ValueError("hardware.devices must be a list of devices.")
    for item in listed:
        try:
            entry = DeviceEntry.from_dict(item)
        except UnknownDeviceType as e:
            # such as a device this version no longer has
            logging.warning(f"{e} It is ignored.")
            continue
        if entry.name in entries:
            raise ValueError(
                f"hardware.devices names '{entry.name}' twice. Give each device its "
                "own name."
            )
        entries[entry.name] = entry
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
        if "device_calibration" in stage:
            calibration["devices"] = stage.pop("device_calibration")
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
        # `defaults:` names what a session starts from; a column's entry describes
        # what the column *is*. Merged back together here because nothing downstream
        # cares about the split -- the readers of `system.electron.beam` and
        # `system.ion.detector` do not move. `defaults:` wins a collision.
        # `calibration:` folds into the stage record it belongs to.
        #
        # The current version only: a file in an older shape is upgraded before it
        # gets here (`fibsem.codec.upgrades.configuration`).
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
        if calibration.get("devices"):
            stage["device_calibration"] = calibration["devices"]
        electron = {**block("electron"), **(defaults.get("electron") or {})}
        ion = {**block("ion"), **(defaults.get("ion") or {})}
        electron["beam_type"] = BeamType.ELECTRON.name
        ion["beam_type"] = BeamType.ION.name

        # Device positions are on each device's entry (`origin`,
        # `available_orientations`, `range`), over the defaults.
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
        fm_device = stage_settings.devices.get("FM")
        if fm.camera_tilt is not None and fm_device is not None and fm_device.pose:
            # One fact, stated once: the camera sees the sample face on in the pose.
            raise ValueError(
                "hardware.devices: 'fm' states both pose and camera_tilt. The camera's "
                "tilt follows from the pose; leave camera_tilt out."
            )
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


class CameraImageTransform(NamedEnum, Enum):
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

    @classmethod
    def _codec_member(cls, value: Any) -> "CameraImageTransform":
        """By name or value, as any enum; a stored value that is no longer a member
        through `_parse_image_transform`, which migrates it or falls back to none."""
        try:
            return super()._codec_member(value)
        except ValueError:
            return _parse_image_transform(value)


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
    instrument is arranged rather than two that merely agree today.

    Angles are in degrees, matching ``SystemSettings``. Both reprojection paths convert
    at the point of use.

    ``fm_is_pose`` and ``mirrors_back_view`` are stored rather than derived, stamped
    from what the driver reports when the image is taken. ``fm_is_pose``: the stage
    reaches the fluorescence objective by turning the grid over rather than by
    travelling, so the FM is one of its poses. ``mirrors_back_view``: the instrument
    mirrors an image a beam takes of the back of the grid, so it reads as if seen from
    the front. An image written before these had ``is_compustage`` for both, and older
    still nothing at all, so the model name was matched against a product; both are
    read by `fibsem.codec.upgrades.hardware_geometry` and
    `fibsem.codec.upgrades.image_metadata`.

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
    fm_is_pose: bool = False
    mirrors_back_view: bool = False
    # Where a half turn of the stage is centred, raw (x, y) in metres; used to draw a
    # position recorded on the other side of the stage. None when the driver reported
    # none: reprojection then turns about the stage origin. An image saved before this
    # field existed reads LEGACY_ROTATION_CENTRE, so it draws exactly as it did.
    #
    rotation_centre: Optional[Tuple[float, float]] = None
    # The pose the stage declared for each orientation name (SEM, FIB, and FM where
    # the FM is a pose), as (r, t) in degrees. Empty on an image written before
    # the stage declared its poses; `declared_poses` rebuilds them for those.
    poses: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # The frame the image's stage position is in: STAGE_FRAME_FIBSEM, or a
    # driver's name for its instrument's own frame, kept as an opaque string. None on
    # an image written before this was stamped, whose position is in whatever frame its
    # driver reported then. Nothing converts between frames on read.
    stage_frame: Optional[str] = None
    # Fluorescence only; left at these defaults for a beam image.
    camera_tilt: float = 0.0  # viewing axis, from the electron column
    transform: CameraImageTransform = CameraImageTransform.NONE

    def declared_poses(self) -> Dict[str, "FibsemStagePosition"]:
        """The stage's pose for each orientation name when this was recorded, in radians.

        The stamped poses when there are any. An image from before the stage declared
        them gets them rebuilt the way its readers interpreted it: SEM at the reference
        rotation and the pre-tilt; FIB at the stamped ``rotation_180``, tilted to the
        ion column less the pre-tilt, and turned over by 180 where the FM is a pose,
        which also has FM with the grid turned fully over.
        """
        poses = self.poses
        if not poses:
            ref, pre_tilt = self.rotation_reference, self.shuttle_pre_tilt
            fib_tilt = self.fib_column_tilt - pre_tilt
            poses = {"SEM": (ref, pre_tilt), "FIB": (self.rotation_180, fib_tilt)}
            if self.fm_is_pose:
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
        fm_is_pose: bool = False,
        mirrors_back_view: bool = False,
        rotation_centre: Optional[Tuple[float, float]] = None,
        poses: Optional[Mapping[str, "FibsemStagePosition"]] = None,
        stage_frame: Optional[str] = None,
    ) -> "FibsemHardwareGeometry":
        """Gather the geometry terms out of a full system configuration.

        ``fm_is_pose`` and ``mirrors_back_view`` are parameters because
        ``SystemSettings`` does not carry them: they are properties of the installed
        hardware, which only the connected microscope knows
        (``microscope.hardware_geometry`` stamps them).
        ``rotation_centre`` likewise comes from the driver (``microscope.rotation_centre``);
        None records None. ``poses`` are the stage's declared poses
        (``microscope._stage_poses()``), in radians; None records none.
        ``stage_frame`` is the frame the stage reports (``microscope.stage_frame``).
        """
        return cls(
            column_tilt=system.electron.column_tilt,
            fib_column_tilt=system.ion.column_tilt,
            shuttle_pre_tilt=system.stage.shuttle_pre_tilt,
            rotation_reference=system.stage.rotation_reference,
            rotation_180=system.stage._fib_rotation(),
            fm_is_pose=fm_is_pose,
            mirrors_back_view=mirrors_back_view,
            rotation_centre=rotation_centre,
            poses={
                name: (float(np.degrees(pose.r)), float(np.degrees(pose.t)))
                for name, pose in (poses or {}).items()
            },
            stage_frame=stage_frame,
        )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "FibsemHardwareGeometry":
        from fibsem.codec.upgrades.hardware_geometry import upgrade

        return decode(cls, upgrade(ddict))


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
            image=ImageSettings.from_dict(
                (settings.get("defaults") or {}).get("imaging") or {}
            ),
            protocol=protocol,
            fm=fm_config,
        )


# A file with no date doesn't say when the experiment was: unknown, not now.
@saved(absent={"date": None})
@dataclass
class FibsemExperimentRef:
    """A reference to the experiment an image was acquired for. Provenance.

    Not an experiment. The experiment itself is the application's own record --
    AutoLamella's ``Experiment``, with positions, protocol and history -- and this
    is a denormalised pointer to it, embedded in every image so the file can say
    what produced it without the surrounding directory. Named ``...Ref`` because
    the two types are otherwise a single import apart and easily confused.

    **Defers to the record.** Authoritative only in the absence of the experiment
    record, which wins on conflict. That rule needs a stable key: with
    only a name, a renamed experiment is indistinguishable from a different one, so
    there is no conflict to detect, just two strings that disagree.

    The deference rule applies here because a richer record exists to defer to. It
    does not apply to every embedded copy -- see ``FibsemUser``.

    **Identity only.** Which experiment, which item, which task, and when. What
    software was running is a property of the running system, not of an experiment,
    so it lives on ``SystemInfo`` and only there. This carried
    duplicates of it until v5.

    **Two write rates, one record.** The experiment is set once at registration; the
    item and task change as a run progresses and are written and cleared around each
    one. They are kept together because they are one answer to one question
    -- *what produced this image* -- and a reader wants "experiment X, lamella Y, task
    Z" in one place rather than assembled from two.

    That only works because every image gets its **own copy**: `Session.stamp`
    deepcopies this, so changing one never rewrites the item on images already
    acquired.
    """

    # Experiment.id, a UUID -- stable, the join key. Held the experiment *name* up to
    # and including v0.5.2; a file whose `name` is absent is from that era and its `id`
    # is a name.
    id: Optional[str] = None
    name: Optional[str] = None
    # When the reference was made, which is when a session adopted the experiment.
    # Aware, written as ISO 8601 with its offset; older images hold a
    # POSIX float, which reads as this machine's zone. default_factory, not a plain
    # default: a plain default is evaluated once at class definition, so every
    # experiment recorded the interpreter's import time rather than its own.
    date: Optional[Timestamp] = field(default_factory=now)

    # Where in the run. None outside a workflow -- the minimap, a manual acquisition,
    # a script -- which is a real answer rather than missing information.
    #
    # "item" rather than "lamella": the core library has no reason to know what an
    # application works through one at a time, and ``HookContext`` settled on the same
    # word for the same reason. IDs *and* names because they answer different
    # questions -- the name is what a person reads, the id is what a reader joins on
    # and it survives a rename.
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    # Which iteration of the item's tasks, for an item that repeats them (a volume
    # position's slices); None for one that does not.
    iteration: Optional[int] = None

    def set_workflow_metadata(
        self,
        item_id: Optional[str] = None,
        item_name: Optional[str] = None,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        iteration: Optional[int] = None,
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
        self.iteration = iteration

    def clear_workflow_metadata(self) -> None:
        """Stop stamping an item and task, keeping the experiment.

        A method rather than four assignments at the call site: this has to run on
        every path out of a task, and the one thing it must never do is take the
        experiment's own identity with it.
        """
        self.set_workflow_metadata()

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(settings: dict) -> "FibsemExperimentRef":
        return decode(FibsemExperimentRef, settings)


@dataclass
class FibsemUser:
    """Who acquired an image. Provenance, and the only user model there is.

    Unlike ``FibsemExperimentRef``, this is not a reference to anything: there is
    no richer user record for it to defer to, so the "the record wins on conflict"
    rule does not apply -- this *is* the record, and it happens to be
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

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(settings: dict) -> "FibsemUser":
        return decode(FibsemUser, settings)

    @staticmethod
    def from_environment() -> "FibsemUser":
        import getpass
        import platform
        import socket

        # getpass covers every platform; USERNAME is Windows-only, and Odemis/METEOR
        # sites run on Linux.
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


@saved(absent={"recorded_at": None})
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
    part of this was already collected per-image; this promotes it.

    **The latest session, not the one that created the experiment.**
    ``Experiment.create()`` has no microscope -- the dialog that calls it never
    holds one -- so the instrument is simply unknown until a session adopts the
    experiment. The latest is the fact that can actually be established, and it has
    the useful property of backfilling onto experiments that predate this.

    Overwriting loses less than it looks like. Every image already carries its own
    ``SystemInfo``, so an experiment spanning an upgrade shows v0.5.1 on day-one
    images and v0.5.2 here -- two true facts, which is the same reasoning that put
    the version fields on ``SystemInfo`` in the first place.

    **The instrument's zone**: ``utc_offset`` and ``zone_name``, so a time
    that carries no offset of its own -- a POSIX float, which every task time and
    older beam images are -- can be read on the instrument's clock
    on any machine (``wall_time_of(value, session.zone)``). New times carry their own
    offset and do not need it. Three limits:

    - An offset belongs to a moment, not a zone. A run that crosses a daylight
      saving change reads an hour out on the side the session did not record.
    - ``zone_name`` is what the OS calls the zone, as given: ``AEST`` on macOS and
      Linux, ``AUS Eastern Standard Time`` on Windows. It labels, it is not parsed.
    - A Demo session records no zone of its own and keeps the last one recorded, so
      opening an experiment on another machine does not replace the instrument's.
    """

    # Aware, written as ISO 8601 with its offset; older experiments hold
    # a POSIX float, which reads as this machine's zone.
    recorded_at: Optional[Timestamp] = field(default_factory=now)
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
        # Here, not at the top: the client imports this module and no driver code.
        from fibsem.drivers.registry import normalize_name
        from fibsem.plugins.report import installed_plugin_versions

        info = getattr(getattr(microscope, "system", None), "info", None)
        if normalize_name(getattr(info, "manufacturer", None)) == "Demo":
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
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionInfo":
        return decode(cls, data)
