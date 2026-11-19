"""What the image tools answer, as codec structures: a saved image's
metadata, a point on it as a stage position and back, a drawing of it, and a
measurement on it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fibsem.structures.image import FibsemImageMetadata
from fibsem.structures.settings import FibsemStagePosition


@dataclass
class SavedImageMetadata:
    path: str
    metadata: FibsemImageMetadata


@dataclass
class SavedImagePosition:
    path: str
    stage_position: FibsemStagePosition


@dataclass
class PixelPoint:
    """Where a stage position falls on an image, in source pixels; ``reason``
    when the image cannot place it."""

    px: Optional[List[float]] = None
    inside: Optional[bool] = None
    reason: Optional[str] = None


@dataclass
class SavedImagePixels:
    path: str
    # the image's scale block (`fibsem.server.images.image_metadata`)
    image: Dict[str, Any]
    points: List[PixelPoint]


@dataclass
class ImageDrawn:
    image: str
    # The picture saved beside the image: drawn/<image>.png.
    path: str
    # The marks as drawn: measured lines labelled with their length, each
    # place with its stage position when the image says where it was taken.
    marks: Optional[Dict[str, Any]]
    plot: Dict[str, str]
    # Per layer drawn (a channel, or SEM/FIB): the look it was drawn with, its
    # auto limits as numbers, and its stats (`fibsem.imaging.looks.layer_stats`).
    layers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagePairDrawn:
    before: str
    after: str
    path: str
    marks: Optional[Dict[str, Any]]
    plot: Dict[str, str]
    # Each half's layers, as `ImageDrawn.layers`.
    before_layers: Dict[str, Any] = field(default_factory=dict)
    after_layers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageMeasured:
    """An analysis tool's answer (`fibsem.analysis.measure`) on a saved image."""

    image: str
    # intensity, profile or quality
    measurement: str
    result: Dict[str, Any]
    # The figure saved beside the image (drawn/<image>-<measurement>.png) and
    # its preview, when the measurement draws one.
    path: Optional[str] = None
    plot: Optional[Dict[str, str]] = None
