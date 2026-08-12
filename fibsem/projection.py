"""Stage positions and the plane a canvas shows them in, in both directions.

The instrument works in stage coordinates; a display works in the plane it is showing.
Everything drawn from a stage position -- a marker, the travel limits, a planned
acquisition grid -- needs the same mapping between the two, and every click needs its
inverse.

A :class:`StageProjection` is that mapping, in metres, for one way of looking at the
sample. Which one differs by modality: a fluorescence camera projects through its own
axis tilt and image transform, a beam through scan rotation and a column tilt. Both are
here, so the display layer can be handed one and stay free of microscope geometry --
see :class:`fibsem.ui.widgets.canvas.stage_frame.StageFrame`, which composes a
projection with a canvas scale.

Deliberately outside ``fibsem/ui``. This is arithmetic over recorded geometry with no
Qt in it, and importing ``fibsem.ui`` pulls in the whole widget package -- which CI
does not install, so a projection living there could not be tested where it matters.

Microscope-free where it can be. Each class has a ``from_image`` reading the geometry
an image recorded, and a ``from_microscope`` reading the live instrument. They are not
interchangeable: ``from_image`` is what a saved overview needs, because it must project
as it was taken rather than as things stand now.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np

from fibsem.fm.reprojection import project_image_point, project_stage_position
from fibsem.imaging.tiling.reprojection import (
    inverse_y_corrected_stage_movement_tescan_from_geometry,
    y_corrected_stage_movement_tescan_from_geometry,
)
from fibsem.structures import BeamType, FibsemStagePosition, Point
from fibsem.transformations import (
    inverse_view_corrected_dy,
    view_corrected_stage_movement,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.fm.structures import FibsemHardwareGeometry
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage


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

    geometry: "FibsemHardwareGeometry"
    pixel_size: float
    shape: Tuple[int, int]  # (height, width)

    @classmethod
    def from_microscope(
        cls, microscope: "FibsemMicroscope"
    ) -> Optional["FMStageProjection"]:
        """Read the projection off the live instrument, or None if it cannot be read.

        Live rather than from a displayed image, for two reasons. `FibsemHardwareGeometry` is
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


@dataclass(frozen=True)
class BeamStageProjection:
    """The beam projection: column tilt, shuttle pre-tilt, scan rotation.

    The FIB/SEM counterpart of :class:`FMStageProjection`. Both directions run one
    derivation, which is the point: today they are two. Where a stage position lands on
    an overview comes from
    :func:`~fibsem.imaging.tiling.reprojection.calculate_reprojected_stage_position2`,
    working from image metadata; where a click sends the stage comes from
    :meth:`FibsemMicroscope.project_stable_move`, working from the live instrument. They
    are meant to be inverses and are written separately, so nothing holds them together
    -- and a disagreement shows up as markers that are subtly, consistently in the wrong
    place, which looks like a calibration problem rather than a bug.

    Two consequences of taking the projection off the instrument, both wanted:

    * **No hardware read per click.** ``project_stable_move`` calls ``get_scan_rotation``
      every time, which on a TFS system is a set-then-read on the shared imaging channel
      -- from the GUI thread, on a mouse event. Here the scan rotation is read once, when
      the projection is built, and carried.
    * **A saved overview projects as it was taken.** Clicking one after the stage has
      moved, or after the scan rotation has changed, still answers about that image.

    ``scan_rotation`` is in radians and only its proximity to pi matters: both directions
    flip x and y together at 180 degrees and do nothing otherwise, matching the live path.
    Intermediate rotations are not modelled by either -- that is pre-existing, and this
    class is deliberately not the place to start.
    """

    geometry: "FibsemHardwareGeometry"
    beam_type: BeamType
    scan_rotation: float  # radians
    # Tescan's stage geometry is different enough that it overrides the projection
    # rather than passing a different view tilt -- its z-axis sits below the tilt axis,
    # so the decomposition uses the full chamber-frame sample inclination. It also
    # inverts stage x against image x. Carried as a flag rather than re-derived from a
    # manufacturer string at each use (FIB-300 is the string check itself).
    is_tescan: bool = False

    @classmethod
    def from_microscope(
        cls, microscope: "FibsemMicroscope", beam_type: BeamType
    ) -> Optional["BeamStageProjection"]:
        """Read the projection off the live instrument, or None if it cannot be read.

        The one hardware read is ``get_scan_rotation``, done here so it is not done per
        click. A projection built this way describes the instrument as it is *now*, so
        it is the right one for drawing on a live view and the wrong one for a saved
        image -- see :meth:`from_image`.
        """
        try:
            return cls(
                geometry=microscope.hardware_geometry(),
                beam_type=beam_type,
                scan_rotation=microscope.get_scan_rotation(beam_type=beam_type),
                is_tescan=microscope.system.info.manufacturer.upper() == "TESCAN",
            )
        except Exception as e:
            logging.debug(f"Could not read the live beam geometry: {e}")
            return None

    @classmethod
    def from_image(cls, image: "FibsemImage") -> Optional["BeamStageProjection"]:
        """Read the projection off an image's own metadata, or None if it has none.

        For placing or clicking an image that was acquired under a configuration the
        instrument may no longer be in -- a different scan rotation, or loaded from disk
        entirely. It has to be projected as it was taken, not as things stand now.
        """
        metadata = getattr(image, "metadata", None)
        geometry = getattr(metadata, "hardware_geometry", None)
        if metadata is None or geometry is None:
            return None
        try:
            beam_type = metadata.image_settings.beam_type
            state = metadata.microscope_state
            beam = (
                state.electron_beam
                if beam_type is BeamType.ELECTRON
                else state.ion_beam
            )
            if beam is None or beam.scan_rotation is None:
                return None
            info = metadata.system_info
            return cls(
                geometry=geometry,
                beam_type=beam_type,
                scan_rotation=beam.scan_rotation,
                is_tescan=(
                    info is not None and info.manufacturer.upper() == "TESCAN"
                ),
            )
        except Exception as e:
            logging.debug(f"Could not read the beam geometry from the image: {e}")
            return None

    # ── the protocol ──────────────────────────────────────────────────────

    def to_plane(
        self, position: FibsemStagePosition, base: FibsemStagePosition
    ) -> Tuple[float, float]:
        delta = position - base
        dx = -delta.x if self.is_tescan else delta.x
        expected_y = self._expected_y(delta.y or 0.0, delta.z or 0.0, base)
        # The plane's y runs *down*, image-fashion, while `expected_y` is the
        # microscope's image-space y, which runs up. `FMStageProjection` inherits the
        # same flip from `project_stage_position`.
        return self._scan_rotated(dx or 0.0, -expected_y)

    def from_plane(
        self, dx: float, dy: float, base: FibsemStagePosition
    ) -> FibsemStagePosition:
        # Undone in the reverse order to `to_plane`, though both terms are sign flips
        # and so commute -- written this way because an inverse that reads as one is
        # easier to keep correct than one that relies on the commuting.
        dx, dy = self._scan_rotated(dx, dy)
        expected_y = -dy
        if self.is_tescan:
            dx = -dx

        position = deepcopy(base)
        if self.is_tescan:
            y_chamber, z_chamber = y_corrected_stage_movement_tescan_from_geometry(
                geometry=self.geometry,
                stage_position=base,
                expected_y=expected_y,
                beam_type=self.beam_type,
            )
            # `stable_move` inverts stage y against the chamber frame and leaves z; the
            # Tescan inverse undoes that same inversion on its way in.
            y_move, z_move = -y_chamber, z_chamber
        else:
            y_move, z_move = view_corrected_stage_movement(
                expected_y=expected_y,
                view_tilt=self._view_tilt(),
                geometry=self.geometry,
                stage_rotation=base.r or 0.0,
                stage_tilt=base.t or 0.0,
            )

        position.x = (position.x or 0.0) + dx
        position.y = (position.y or 0.0) + y_move
        position.z = (position.z or 0.0) + z_move
        return position

    # ── internals ─────────────────────────────────────────────────────────

    def _view_tilt(self) -> float:
        """How far this beam's axis is tilted from the electron column, in radians.

        The same quantity :meth:`FibsemMicroscope._beam_view_tilt` returns, and the same
        one the fluorescence path passes as the camera tilt -- which is what lets all
        three share one projection.
        """
        if self.beam_type is BeamType.ELECTRON:
            return 0.0
        return np.deg2rad(self.geometry.fib_column_tilt)

    def _expected_y(
        self, dy: float, dz: float, base: FibsemStagePosition
    ) -> float:
        """The image-space y-displacement a y/z stage movement corresponds to."""
        if self.is_tescan:
            return inverse_y_corrected_stage_movement_tescan_from_geometry(
                geometry=self.geometry,
                stage_position=base,
                dy=dy,
                dz=dz,
                beam_type=self.beam_type,
            )
        return inverse_view_corrected_dy(
            dy=dy,
            dz=dz,
            view_tilt=self._view_tilt(),
            geometry=self.geometry,
            stage_rotation=base.r or 0.0,
            stage_tilt=base.t or 0.0,
        )

    def _scan_rotated(self, x: float, y: float) -> Tuple[float, float]:
        """Apply a 180-degree scan rotation, which is its own inverse."""
        if np.isclose(self.scan_rotation, np.pi):
            return -x, -y
        return x, y
