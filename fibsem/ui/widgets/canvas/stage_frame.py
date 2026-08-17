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

The projections themselves live in :mod:`fibsem.projection`, outside ``fibsem/ui``:
they are arithmetic over recorded geometry with no Qt in them, and importing
``fibsem.ui`` pulls in the whole widget package, which CI does not install. They are
re-exported here because this is where callers expect to find them alongside the frame
they are handed to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

# Re-exported for the callers that reach for a projection and a frame together.
from fibsem.projection import (  # noqa: F401
    BeamStageProjection,
    FMStageProjection,
    StageProjection,
    surface_foreshortening,
)
from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas


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

    def surface_foreshortening(self) -> float:
        """How much of a length along the *sample surface* survives into this view.

        1.0 looking down the surface normal, `cos(theta)` from `theta` away. What a
        shape lying on the sample needs -- a grid's boundary, say -- as opposed to a
        shape defined in the image, which needs nothing.
        """
        return surface_foreshortening(self.projection, self.origin)
