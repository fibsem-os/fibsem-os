"""The point-of-interest marker on the FIB view: placed, dragged, read back.

One marker, for Setup's point-of-interest proposal. The marker is
held in the view's pixels and converted against the frame the FIB view shows,
so placing, previewing and reading back always use the one frame the operator
sees; when the view takes another frame the marker moves to the
same point on it (``sync``).
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from fibsem import conversions
from fibsem.structures import BeamType, Point
from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

if TYPE_CHECKING:
    from fibsem.structures import FibsemImage

__all__ = ["PoiMarker", "poi_pixel", "poi_point"]

OVERLAY_ID = "poi"


def poi_pixel(frame: "FibsemImage", point: "Point") -> Tuple[float, float]:
    """A point of interest (metres from the frame's centre) as *frame*'s
    (col, row) pixel."""
    px = conversions.microscope_image_to_image_coordinates(
        point, frame.data.shape, frame.metadata.pixel_size.x
    )
    return (px.x, px.y)


def poi_point(frame: "FibsemImage", pixel: Tuple[float, float]) -> "Point":
    """The inverse of :func:`poi_pixel`."""
    col, row = pixel
    return conversions.image_to_microscope_image_coordinates(
        Point(x=col, y=row), frame.data, frame.metadata.pixel_size.x
    )


class PoiMarker:
    """The draggable marker on the main window's FIB view, for one point at a
    time. The image widget shows the frame and the quad view draws the marker;
    both are getters, None when the host has none."""

    def __init__(
        self,
        *,
        image_widget: Callable[[], Any],
        view_controller: Callable[[], Any],
    ) -> None:
        self._image_widget = image_widget
        self._view_controller = view_controller
        # The frame the marker was placed on, and the frame to use when the
        # view shows none; None while no marker is up.
        self.frame: Optional["FibsemImage"] = None
        self._fallback: Optional["FibsemImage"] = None

    def controller(self):
        """The main window's quad-view controller, or None standalone."""
        return self._view_controller()

    @property
    def shown(self) -> bool:
        return self.frame is not None

    def view_frame(self, fallback: Optional["FibsemImage"] = None):
        """The FIB frame the view shows; *fallback* (else the one the marker
        was shown with) when it shows none."""
        shown = getattr(self._image_widget(), "ib_image", None)
        if shown is not None:
            return shown
        return fallback if fallback is not None else self._fallback

    def show(
        self,
        poi: Optional["Point"],
        fallback: Optional["FibsemImage"] = None,
    ) -> bool:
        """Put the marker on the FIB view at *poi* (the centre for None),
        armed so the FIB canvas drags it. False when there is no view, or no
        frame to put it on."""
        controller = self.controller()
        if controller is None:
            return False
        self._fallback = fallback
        image = self.view_frame(fallback)
        if image is None:
            return False

        self.frame = image
        if poi is not None:
            col, row = poi_pixel(image, poi)
        else:
            row = image.data.shape[0] / 2
            col = image.data.shape[1] / 2
        controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id=OVERLAY_ID,
                points=[(col, row)],
                # Matches the protocol editor's POI, down to the legend entry: same
                # concept, same marker. Left to the defaults this drew at size 18 with
                # PointOverlay's 2.0 edge, a cross visibly fatter than the thin one the
                # editor and the config preview draw, and absent from the legend.
                # 1.2 keeps it reading like the centre crosshair.
                color="magenta",
                selected_color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                legend_label="Point of Interest",
                add_on_right_click=False,
                removable=False,
            ),
        )
        # POI owns FIB-canvas input: stage-move + milling menu stand down. The toolbar
        # toggle lets the user drop to Move and back. (See active-overlay model.)
        controller.arm_overlay(
            BeamType.ION, OVERLAY_ID, label="POI", icon="mdi:map-marker"
        )
        controller.fib_canvas.set_hint("drag to move")
        return True

    def sync(self) -> bool:
        """Keep the marker on the frame the FIB view shows: moved to the same
        point when the view has taken another frame. True when it moved."""
        old = self.frame
        if old is None:
            return False
        frame = self.view_frame()
        if frame is old:
            return False
        self.frame = frame
        controller = self.controller()
        pts = (
            controller.overlay_points(BeamType.ION, OVERLAY_ID) if controller else None
        )
        if pts:
            controller.set_points(
                BeamType.ION, OVERLAY_ID, [poi_pixel(frame, poi_point(old, pts[0]))]
            )
        return True

    def place(self, poi: "Point") -> None:
        """Move the marker to *poi*, on the frame the view shows."""
        controller = self.controller()
        if controller is None:
            raise RuntimeError("No canvas to place the point of interest on.")
        self.sync()
        self.frame = self.view_frame()
        controller.set_points(BeamType.ION, OVERLAY_ID, [poi_pixel(self.frame, poi)])

    def read(self) -> Optional["Point"]:
        """Where the marker is, as a point of interest; None without one."""
        controller = self.controller()
        if controller is None or self.frame is None:
            return None
        self.sync()
        pts = controller.overlay_points(BeamType.ION, OVERLAY_ID)
        if not pts:
            return None
        return poi_point(self.view_frame(), pts[0])

    def remove(self) -> None:
        """Take the marker down and give the FIB canvas back to Move."""
        self.frame = None
        self._fallback = None
        controller = self.controller()
        if controller is None:
            return
        controller.arm_overlay(BeamType.ION, None)  # restore Move
        controller.remove_overlay(BeamType.ION, OVERLAY_ID)
        controller.fib_canvas.set_hint(None)
