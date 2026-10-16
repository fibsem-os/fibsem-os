from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
    Union,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    distance_param,
)
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemRectangleSettings,
    Point,
    param,
)


@dataclass
class CircularSuspensionPattern(
    BasePattern[Union[FibsemCircleSettings, FibsemRectangleSettings]]
):
    """A pair of discs, each left suspended on two anchors.

    Per side: a solid disc of ``external_radius`` is milled away, with an annulus
    (``inner_radius`` .. ``outer_radius``) and two anchor rectangles marked as
    exclusions, so the material under them survives. The anchors are asymmetric by
    design — the left and right sides swap ``inner_anchor_height`` and
    ``outer_anchor_height``, keeping the thicker anchor on the outside of the pair.

    Ported from the circular-suspension plugin.
    """

    inner_anchor_height: float = param(
        2.0e-6,
        distance_param(
            "Inner Anchor Height",
            "Height of the anchor on the inner side of each disc.",
        ),
    )
    outer_anchor_height: float = param(
        2.0e-6,
        distance_param(
            "Outer Anchor Height",
            "Height of the anchor on the outer side of each disc.",
        ),
    )
    outer_radius: float = param(
        2.0e-6, distance_param("Outer Radius", "Outer radius of the suspended ring.")
    )
    inner_radius: float = param(
        1.0e-6, distance_param("Inner Radius", "Inner radius of the suspended ring.")
    )
    external_radius: float = param(
        2.5e-6,
        distance_param(
            "External Radius", "Radius of the disc milled around each suspension."
        ),
    )
    depth: float = param(1.0e-6, distance_param("Depth", "Depth of the pattern."))
    spacing: float = param(
        10.0e-6, distance_param("Spacing", "Gap between the two discs, edge to edge.")
    )

    name: ClassVar[str] = "CircularSuspension"

    def define(self) -> List[Union[FibsemRectangleSettings, FibsemCircleSettings]]:
        point = self.point
        depth = self.depth
        external_radius = self.external_radius
        # `spacing` is the gap between the discs, so the centres are one disc apart
        # on top of it.
        centre_spacing = self.spacing + (external_radius * 2)
        thickness = self.outer_radius - self.inner_radius
        anchor_width = (external_radius - self.outer_radius) * 1.5

        shapes: List[Union[FibsemRectangleSettings, FibsemCircleSettings]] = []
        centres = {
            "left": Point(point.x - centre_spacing / 2, point.y),
            "right": Point(point.x + centre_spacing / 2, point.y),
        }
        for side, centre in centres.items():
            # the disc that gets milled away
            shapes.append(
                FibsemCircleSettings(
                    radius=external_radius,
                    thickness=0,
                    depth=depth,
                    centre_x=centre.x,
                    centre_y=centre.y,
                    is_exclusion=False,
                )
            )
            # the ring that survives it
            shapes.append(
                FibsemCircleSettings(
                    radius=self.outer_radius,
                    thickness=thickness,
                    depth=depth,
                    centre_x=centre.x,
                    centre_y=centre.y,
                    is_exclusion=True,
                )
            )
            # the two anchors holding the ring, thicker one facing outwards
            inner_first = side == "right"
            shapes.append(
                FibsemRectangleSettings(
                    width=anchor_width,
                    height=self.inner_anchor_height
                    if inner_first
                    else self.outer_anchor_height,
                    depth=depth,
                    centre_x=centre.x - external_radius + anchor_width / 2,
                    centre_y=centre.y,
                    is_exclusion=True,
                )
            )
            shapes.append(
                FibsemRectangleSettings(
                    width=anchor_width,
                    height=self.outer_anchor_height
                    if inner_first
                    else self.inner_anchor_height,
                    depth=depth,
                    centre_x=centre.x + external_radius - anchor_width / 2,
                    centre_y=centre.y,
                    is_exclusion=True,
                )
            )

        # Exclusions first. Each shape becomes a microscope pattern in this order and
        # an exclusion is set as an exclusion *zone* on the one it is created against,
        # so the zone has to exist before the disc it protects. Kept as the plugin
        # ordered it; note `create_pattern_mask` sorts the other way, subtracting
        # exclusions last, which reaches the same picture by a different route.
        shapes.sort(key=lambda s: getattr(s, "is_exclusion", False), reverse=True)

        self.shapes = shapes
        return self.shapes
