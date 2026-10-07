"""Where a lamella's reference images were taken, compared with where it is now.

The lamella editor offers every FIB/SEM image in the lamella's folder and picks one by
filename. A lamella moves after its images are taken -- a pose re-recorded, a position
dragged on the overview, a later task -- and patterns edited over an image of another
place land somewhere else. Every image records the stage position it was taken at, so
the editor can say so.

Pure: no Qt, no microscope. The image's position is read from the TIFF header alone,
which is about a tenth of the cost of loading the image, so a whole lamella folder can
be read on every selection.

An image that records no position is *unknown*, never *moved*: older images may carry
no metadata at all, and "this image was taken elsewhere" is a claim only a recorded
position can support.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

import tifffile as tff

from fibsem.structures import FibsemStagePosition
from fibsem.util.timestamps import to_datetime

# Far enough apart to be a different place. Images taken at a lamella's milling pose sit
# within 0.12 um of it on real experiments; images taken before the position was set,
# 60 um away. 5 um is the tolerance the fluorescence pose check already uses, and leaves
# room for a stage that does not return exactly where it was sent.
POSITION_TOLERANCE_M = 5e-6
ANGLE_TOLERANCE_RAD = math.radians(0.5)


class PositionMatch(Enum):
    SAME = "same"
    MOVED = "moved"
    UNKNOWN = "unknown"


@dataclass
class ImagePosition:
    """One image file's stage position, read without its pixels."""

    filename: str
    stage_position: Optional[FibsemStagePosition]
    # When the image was taken, in epoch seconds: the metadata's timestamp, else the
    # file's mtime.
    timestamp: float


@dataclass
class PositionComparison:
    """How far an image's position is from a pose."""

    match: PositionMatch
    distance_m: Optional[float] = None
    tilt_rad: Optional[float] = None
    rotation_rad: Optional[float] = None

    def describe(self) -> str:
        """The difference as a phrase: ``~42 µm``, ``~42 µm, tilted 2.0°``."""
        parts: List[str] = []
        if self.distance_m is not None and self.distance_m > POSITION_TOLERANCE_M:
            um = self.distance_m * 1e6
            parts.append(f"~{um:.0f} µm" if um >= 10 else f"~{um:.1f} µm")
        if self.tilt_rad is not None and abs(self.tilt_rad) > ANGLE_TOLERANCE_RAD:
            parts.append(f"tilted {abs(math.degrees(self.tilt_rad)):.1f}°")
        if (
            self.rotation_rad is not None
            and abs(self.rotation_rad) > ANGLE_TOLERANCE_RAD
        ):
            parts.append(f"rotated {abs(math.degrees(self.rotation_rad)):.1f}°")
        return ", ".join(parts)


def read_image_position(path: str) -> ImagePosition:
    """The stage position *path* was taken at, from its TIFF header. Never raises.

    Reads only ``microscope_state`` from the metadata rather than building the whole
    ``FibsemImageMetadata``, so an image whose other metadata an older version wrote
    differently still gives its position.
    """
    filename = os.path.basename(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    try:
        with tff.TiffFile(path) as tiff:
            description = tiff.pages[0].tags["ImageDescription"].value
        state = json.loads(description).get("microscope_state") or {}
    except Exception as e:  # noqa: BLE001 - no metadata is an answer: unknown
        logging.debug(f"No readable metadata on {path}: {e}")
        return ImagePosition(filename, None, mtime)

    stage_position = None
    try:
        if state.get("stage_position"):
            stage_position = FibsemStagePosition.from_dict(state["stage_position"])
    except Exception as e:  # noqa: BLE001
        logging.debug(f"No readable stage position on {path}: {e}")
    # The file's mtime is the fallback, and a poor one: copying an experiment resets it.
    recorded = to_datetime(state.get("timestamp"))
    return ImagePosition(
        filename, stage_position, mtime if recorded is None else recorded.timestamp()
    )


def _wrap(angle: float) -> float:
    """*angle* in [-pi, pi], so 359.9 degrees is 0.1 degrees from 0."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compare_positions(
    image_position: Optional[FibsemStagePosition],
    pose_position: Optional[FibsemStagePosition],
) -> PositionComparison:
    """Whether an image was taken at *pose_position*.

    UNKNOWN when either side has no x and y, or when both name their stage frame and
    the frames differ: a RAW and a SPECIMEN position are not comparable numbers. A
    frame missing on either side is compared anyway, as older files did not record it.

    Translation is the distance over x, y and z, with z only when both have it. Tilt and
    rotation are compared on their own, each only when both have it.
    """
    if image_position is None or pose_position is None:
        return PositionComparison(PositionMatch.UNKNOWN)
    for position in (image_position, pose_position):
        if position.x is None or position.y is None:
            return PositionComparison(PositionMatch.UNKNOWN)
    frames = (image_position.coordinate_system, pose_position.coordinate_system)
    if all(frames) and frames[0].upper() != frames[1].upper():
        return PositionComparison(PositionMatch.UNKNOWN)

    deltas = [
        image_position.x - pose_position.x,
        image_position.y - pose_position.y,
    ]
    if image_position.z is not None and pose_position.z is not None:
        deltas.append(image_position.z - pose_position.z)
    distance = math.sqrt(sum(d * d for d in deltas))

    tilt = None
    if image_position.t is not None and pose_position.t is not None:
        tilt = _wrap(image_position.t - pose_position.t)
    rotation = None
    if image_position.r is not None and pose_position.r is not None:
        rotation = _wrap(image_position.r - pose_position.r)

    moved = (
        distance > POSITION_TOLERANCE_M
        or (tilt is not None and abs(tilt) > ANGLE_TOLERANCE_RAD)
        or (rotation is not None and abs(rotation) > ANGLE_TOLERANCE_RAD)
    )
    return PositionComparison(
        PositionMatch.MOVED if moved else PositionMatch.SAME,
        distance_m=distance,
        tilt_rad=tilt,
        rotation_rad=rotation,
    )


def latest_image_at(
    images: Iterable[ImagePosition], pose_position: Optional[FibsemStagePosition]
) -> Optional[ImagePosition]:
    """The most recent of *images* taken at *pose_position*, or None if none was."""
    here = [
        image
        for image in images
        if compare_positions(image.stage_position, pose_position).match
        is PositionMatch.SAME
    ]
    if not here:
        return None
    return max(here, key=lambda image: (image.timestamp, image.filename))
