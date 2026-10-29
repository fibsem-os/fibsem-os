"""The stage as rotations in the chamber, and what a view sees of it.

One model instead of rules per stage type. The chamber frame has x along the tilt axis,
y horizontal and z up. The stage's translation axes tilt with it about x; the rotation
axis and the shuttle sit on top of them, so ``r`` turns the shuttle (and its pre-tilt)
but not the x/y axes a move is commanded in. A view is fixed in the chamber: its image
y-axis is the chamber y-axis tilted by the view's tilt from the electron column.

From that, the rules the readers carry today are consequences, not branches:

- the pre-tilt sign is the shuttle turning with ``r`` (the surface normal's y component
  changes sign half a turn from the reference);
- a chamber-vertical move, in stage axes, is the inverse tilt applied to "up";
- a stage with the same rotation at SEM and FIB but a different tilt (JEOL) needs nothing.

Rotations in radians throughout. Nothing here reads a microscope: the same answer serves
a live move and a saved image.

Whether a view sees the front or the back of the sample is geometry too
(`views_back`). Some instruments mirror a back view so it reads as if from the front;
that is a property of the instrument, not the model, and the caller applies it. What
the model does not yet cover is the Tescan stage in its native frame, which keeps its
own maths until its frame conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fibsem.geometry.angles import wrap_angle


def rotate_x(angle: float) -> np.ndarray:
    """Rotation about the chamber x-axis (the tilt axis)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotate_z(angle: float) -> np.ndarray:
    """Rotation about the z-axis (the stage rotation axis, before tilt)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class StageModel:
    """A stage's kinematics: where its axes and the sample surface point in the chamber.

    Args:
        shuttle_pre_tilt: the shuttle's pre-tilt, in radians.
        rotation_reference: the stage rotation at which the pre-tilt leans towards the
            electron column's side (the SEM pose's rotation), in radians.
    """

    shuttle_pre_tilt: float = 0.0
    rotation_reference: float = 0.0

    @classmethod
    def from_geometry(cls, geometry) -> "StageModel":
        """The model for an image's stamped `FibsemHardwareGeometry` (degrees there)."""
        return cls(
            shuttle_pre_tilt=float(np.deg2rad(geometry.shuttle_pre_tilt)),
            rotation_reference=float(np.deg2rad(geometry.rotation_reference)),
        )

    # -- frames ----------------------------------------------------------------------

    def stage_axes(self, t: float) -> np.ndarray:
        """The stage's x, y and z translation axes in the chamber, as columns."""
        return rotate_x(t)

    def shuttle(self, r: float) -> np.ndarray:
        """The sample's axes on the untilted stage, as columns: rotated, then pre-tilted."""
        return rotate_z(r - self.rotation_reference) @ rotate_x(-self.shuttle_pre_tilt)

    def sample_frame(self, r: float, t: float) -> np.ndarray:
        """The sample's axes in the chamber, as columns: the shuttle on the stage."""
        return self.stage_axes(t) @ self.shuttle(r)

    def surface_normal(self, r: float, t: float) -> np.ndarray:
        """The sample surface normal in the chamber."""
        return self.sample_frame(r, t) @ UP

    # -- moves -----------------------------------------------------------------------

    def in_plane_direction(self, r: float) -> Tuple[float, float]:
        """The stage (y, z) direction that slides the sample along its own surface.

        Where the surface meets the stage's y-z plane: a move with no x that keeps the
        sample in its plane. Independent of tilt, which turns the plane and the axes
        together.
        """
        normal = self.shuttle(r) @ UP
        direction = np.array([normal[2], -normal[1]])
        return float(direction[0]), float(direction[1])

    def surface_slope(self, r: float) -> float:
        """The in-plane slide's angle below the stage y-axis, in radians.

        The pre-tilt at the reference rotation, minus the pre-tilt half a turn round,
        and in between whatever the shuttle's lean gives.
        """
        direction_y, direction_z = self.in_plane_direction(r)
        return float(np.arctan2(-direction_z, direction_y))

    def turned_round(self, r: float) -> bool:
        """Whether the shuttle is nearer half a turn from the reference than the reference."""
        from_reference = abs(wrap_angle(r - self.rotation_reference))
        return from_reference > np.pi / 2

    def chamber_vertical(self, t: float) -> Tuple[float, float]:
        """The stage (y, z) move that goes straight up the chamber by one unit."""
        up = self.stage_axes(t).T @ UP
        return float(up[1]), float(up[2])


def view_axis(view_tilt: float) -> np.ndarray:
    """A view's axis in the chamber, pointing from the sample back towards the view.

    The electron column looks down (axis up); a camera under the grid, at a half turn,
    looks up (axis down).
    """
    return rotate_x(view_tilt) @ UP


def views_back(model: StageModel, view_tilt: float, r: float, t: float) -> bool:
    """Whether the view sees the back of the sample: its surface faces away from it."""
    return bool(model.surface_normal(r, t) @ view_axis(view_tilt) < 0)


def image_y_axis(view_tilt: float) -> np.ndarray:
    """A view's image y-axis in the chamber, for a view tilted from the electron column."""
    return rotate_x(view_tilt) @ np.array([0.0, 1.0, 0.0])


def image_y_shift(
    model: StageModel, dy: float, dz: float, view_tilt: float, t: float
) -> float:
    """How far a stage (y, z) move shifts the image along y, orthographically."""
    move = model.stage_axes(t) @ np.array([0.0, dy, dz])
    return float(image_y_axis(view_tilt) @ move)


def in_plane_move(
    model: StageModel, expected_y: float, view_tilt: float, r: float, t: float
) -> Tuple[float, float]:
    """The stage (y, z) move along the sample surface that shifts the image by expected_y.

    The geometric form of `fibsem.geometry.movement.view_corrected_stage_movement`.
    """
    direction_y, direction_z = model.in_plane_direction(r)
    per_unit = image_y_shift(model, direction_y, direction_z, view_tilt, t)
    length = expected_y / per_unit
    return float(length * direction_y), float(length * direction_z)


def vertical_move(
    model: StageModel, fib_dy: float, fib_column_tilt: float, t: float
) -> Tuple[float, float]:
    """The stage (y, z) move straight up the chamber that cancels fib_dy in the FIB view.

    The geometric form of `fibsem.geometry.movement.vertical_move_delta`, without the
    reversal it applies where the stage reports the sample turned over.
    """
    height = fib_dy / np.sin(fib_column_tilt)
    up_y, up_z = model.chamber_vertical(t)
    return float(height * up_y), float(height * up_z)


def fm_camera_tilt(fm_is_pose: bool, ion_column_tilt: float) -> float:
    """Tilt of the FM camera's optical axis from the electron column, in degrees, for
    an instrument that states none.

    An FM the stage reaches by turning the grid over (the FM is a pose) looks up at it
    from the opposite side to the electron column: a half turn. An FM the stage
    travels to sits parallel to the ion column and shares its tilt.
    """
    return 180.0 if fm_is_pose else float(ion_column_tilt)


def pose_camera_tilt(model: StageModel, r: float, t: float) -> float:
    """Tilt from the electron column, in degrees, of a camera that sees the sample
    face on at the pose (r, t) in radians: the surface normal's tilt there. An FM's
    declared pose and its camera's tilt are one fact, read this way round."""
    normal = model.surface_normal(r, t)
    return float(np.degrees(np.arctan2(-normal[1], normal[2])))
