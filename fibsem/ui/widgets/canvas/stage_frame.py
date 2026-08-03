"""Stage space on a real-space canvas.

The canvas draws in pixels around an origin; the instrument works in stage coordinates.
Everything drawn from a stage position -- a marker, the travel limits, a planned
acquisition grid -- needs the same mapping between the two, and every click needs its
inverse. Left to the callers, that is five places agreeing to do the same arithmetic,
and they agree right up until one of them is changed.

:class:`StageFrame` is that mapping, composed from two halves neither of which is
enough alone:

* a :class:`StageProjection`, mapping stage coordinates into the displayed plane in
  metres. This is the half that differs by modality -- a fluorescence camera projects
  through its own axis tilt and image transform, a beam through scan rotation and a
  column tilt -- so it is supplied rather than assumed.
* the canvas's own scale, in metres per canvas pixel.

The frame is anchored at an *origin*: the stage position it is built around. Whoever
owns the frame fixes that once and keeps it, because re-deriving it from whatever
arrived last would shift the whole scene each time something did.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np

from fibsem.fm.reprojection import project_image_point, project_stage_position
from fibsem.structures import FibsemStagePosition, Point

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.fm.structures import FMImageGeometry
    from fibsem.microscope import FibsemMicroscope
    from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas


class StageProjection(Protocol):
    """Stage coordinates to the displayed plane and back, both in metres.

    Metres rather than pixels on purpose. The underlying projections answer in pixels
    of a hypothetical image, but pixels are a property of a detector, not of the
    geometry, and a real-space canvas has its own scale -- so a projection that spoke
    in pixels would have to be told which pixels, and the answer would cancel out
    immediately afterwards.
    """

    def to_plane(
        self, position: FibsemStagePosition, base: FibsemStagePosition
    ) -> Tuple[float, float]:
        """Where *position* falls relative to *base*, in metres in the displayed plane."""
        ...

    def from_plane(
        self, dx: float, dy: float, base: FibsemStagePosition
    ) -> FibsemStagePosition:
        """The stage position a displayed-plane offset from *base* corresponds to."""
        ...


@dataclass(frozen=True)
class FMStageProjection:
    """The fluorescence projection: camera axis tilt, image transform, sample tilt.

    `pixel_size` and `shape` are arguments to functions that measure in pixels, not
    inputs to the geometry -- they cancel between the projection and the conversion
    back to metres, verified identical across a 27,000x range of pixel sizes. They are
    carried here so the two directions cannot be handed different ones.
    """

    geometry: "FMImageGeometry"
    pixel_size: float
    shape: Tuple[int, int]  # (height, width)

    @classmethod
    def from_microscope(
        cls, microscope: "FibsemMicroscope"
    ) -> Optional["FMStageProjection"]:
        """Read the projection off the live instrument, or None if it cannot be read.

        Live rather than from a displayed image, for two reasons. `FMImageGeometry` is
        system configuration -- camera tilt, shuttle pre-tilt, the camera flip,
        compustage or not -- and not pose; the pose enters through the frame's origin,
        as its rotation and tilt. So a recorded geometry and the live one agree by
        construction.

        And, at present, an acquired tile has no geometry to read: the metadata
        constructors that combine channels and z-planes rebuild it field by field, and
        a stitched mosaic inherits the gap (FIB-416). Taking it from the displayed
        image made everything drawn from a stage position vanish the moment an overview
        was acquired.
        """
        if microscope.fm is None:
            return None
        try:
            width, height = microscope.fm.camera.resolution
            pixel_size = microscope.fm.camera.pixel_size[0]
            if not pixel_size:
                return None
            return cls(
                geometry=microscope.fm_image_geometry(),
                pixel_size=pixel_size,
                shape=(height, width),
            )
        except Exception as e:
            logging.debug(f"Could not read the live FM geometry: {e}")
            return None

    @classmethod
    def from_image(cls, image) -> Optional["FMStageProjection"]:
        """Read the projection off an image's own metadata, or None if it has none.

        For placing an image rather than drawing on top of one, and the reason the
        projection is not simply always taken live: an image may have been acquired
        under a configuration the instrument is no longer in -- a different camera
        transform, or loaded from disk entirely -- and it has to be placed as it was
        taken, not as things stand now.
        """
        metadata = getattr(image, "metadata", None)
        geometry = getattr(metadata, "geometry", None)
        pixel_size = getattr(metadata, "pixel_size_x", None)
        if geometry is None or not pixel_size:
            return None
        shape = np.asarray(image.data).shape[-2:]
        return cls(geometry=geometry, pixel_size=pixel_size, shape=shape)

    def to_plane(
        self, position: FibsemStagePosition, base: FibsemStagePosition
    ) -> Tuple[float, float]:
        point = project_stage_position(
            position, base, self.pixel_size, self.shape, self.geometry
        )
        # `project_stage_position` answers in pixels measured from the corner; the
        # frame wants metres measured from the centre.
        return (
            (point.x - self.shape[1] / 2) * self.pixel_size,
            (point.y - self.shape[0] / 2) * self.pixel_size,
        )

    def from_plane(
        self, dx: float, dy: float, base: FibsemStagePosition
    ) -> FibsemStagePosition:
        # `project_image_point` is the exact inverse of `project_stage_position`,
        # sharing its sign terms, so a click on a marker resolves to the position that
        # marker was drawn from rather than to somewhere plausibly near it.
        point = Point(
            x=self.shape[1] / 2 + dx / self.pixel_size,
            y=self.shape[0] / 2 + dy / self.pixel_size,
        )
        return project_image_point(
            point, base, self.pixel_size, self.shape, self.geometry
        )


class StageFrame:
    """Stage positions and canvas coordinates, in both directions, about an origin.

    Cheap to build and holds no state of its own, so a caller can make one per use
    rather than keeping one in sync with a microscope that moves underneath it.
    """

    def __init__(
        self,
        canvas: "FibsemRealSpaceCanvas",
        origin: FibsemStagePosition,
        projection: StageProjection,
    ) -> None:
        self._canvas = canvas
        self.origin = origin
        self.projection = projection

    def to_canvas(self, position: FibsemStagePosition) -> Tuple[float, float]:
        """Where a stage position falls on the canvas."""
        return self._canvas.metres_to_canvas(*self.offset(position))

    def to_stage(self, x: float, y: float) -> FibsemStagePosition:
        """The stage position under a canvas point."""
        return self.projection.from_plane(
            *self._canvas.canvas_to_metres(x, y), self.origin
        )

    def offset(self, position: FibsemStagePosition) -> Tuple[float, float]:
        """Where a stage position sits relative to the origin, in metres.

        The placement a canvas wants for an image acquired there, as distinct from the
        canvas coordinates :meth:`to_canvas` gives for drawing on top of one.
        """
        return self.projection.to_plane(position, self.origin)

    def length(self, metres: float) -> float:
        """A length in metres as a length in canvas pixels.

        Through the canvas rather than divided by its pixel size directly, so a
        measured length and a placed position cannot end up on different scales.
        """
        return self._canvas.metres_to_canvas(metres, 0.0)[0]
