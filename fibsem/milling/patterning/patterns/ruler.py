from __future__ import annotations

from dataclasses import dataclass
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    cross_section_param,
    distance_param,
    passes_param,
    scan_direction_param,
)
from fibsem.structures import (
    CrossSectionPattern,
    Display,
    FibsemRectangleSettings,
    RangeLimit,
    param,
)


@dataclass
class RulerPattern(BasePattern[FibsemRectangleSettings]):
    """A graduated scale milled into the sample front face, read in fluorescence.

    Used by the fib-view imaging method (Poge et al., eLife reviewed preprint
    106455), which acquires int-FLM with the stage in the lamella milling
    orientation, looking at the front face from the shallow angle the lamella is
    thinned at. The notches give that fluorescence image an axial scale: read where
    the target's signal falls against them, and you know how far down to thin.

    Being read optically rather than in the ion image is what sets the dimensions.
    `pitch` has to clear the light microscope's resolution or neighbouring notches
    blur into a single smear, and `depth` is chosen for what shows up in FM rather
    than for what mills quickest -- so the defaults here are microns where an
    ion-beam pattern would reach for something finer.
    """

    n_ticks: int = param(
        20,
        limits=RangeLimit(1, 500),
        doc="Number of notches, counted down from the origin.",
        display=Display("Number of Ticks", step=1, decimals=0),
    )
    pitch: float = param(
        2.0e-6,
        distance_param(
            "Pitch",
            "Spacing between notches. Must stay above the fluorescence "
            "resolution, or neighbouring notches cannot be told apart.",
        ),
    )
    tick_length: float = param(
        2.0e-6, distance_param("Tick Length", "Length of a minor notch.")
    )
    major_length: float = param(
        4.0e-6,
        distance_param(
            "Major Tick Length",
            "Length of a major notch, so position can be counted at a glance.",
        ),
    )
    major_every: int = param(
        5,
        limits=RangeLimit(0, 100),
        doc="Draw every Nth notch at the major length. 0 for a uniform comb.",
        display=Display("Major Tick Every", step=1, decimals=0),
    )
    tick_thickness: float = param(
        0.5e-6,
        distance_param("Tick Thickness", "Thickness of each notch, across the ruler."),
    )
    depth: float = param(
        1.0e-6,
        distance_param(
            "Depth", "Milling depth. Deep enough to show up in fluorescence."
        ),
    )
    scan_direction: str = param("TopToBottom", scan_direction_param())
    cross_section: CrossSectionPattern = param(
        CrossSectionPattern.Rectangle, cross_section_param()
    )
    passes: int = param(0, passes_param())

    name: ClassVar[str] = "Ruler"

    def summary(self) -> str:
        from fibsem.utils import format_value

        return "\n".join(
            [
                f"    Pattern: {self.name}",
                f"        Ticks: {int(self.n_ticks)} at "
                f"{format_value(self.pitch, unit='m', precision=1)} pitch",
                f"        Depth: {format_value(self.depth, unit='m', precision=1)}",
            ]
        )

    def define(self) -> List[FibsemRectangleSettings]:
        point = self.point
        n_ticks = int(self.n_ticks)
        major_every = int(self.major_every)

        self.shapes = []
        for i in range(n_ticks):
            is_major = major_every > 0 and i % major_every == 0
            length = self.major_length if is_major else self.tick_length

            # `point` is the ruler's origin -- the left end of the first notch --
            # not its centre, which is how every other pattern here places itself.
            # The origin is the thing lined up with the surface edge, and "three
            # notches down" only converts into a depth if it stays put while the
            # tick count and lengths change around it. So notches hang to the right
            # of point.x and run downwards from point.y (+y is up, as in
            # TrenchPattern's upper/lower offsets).
            self.shapes.append(
                FibsemRectangleSettings(
                    width=length,
                    height=self.tick_thickness,
                    depth=self.depth,
                    centre_x=point.x + length / 2,
                    centre_y=point.y - i * self.pitch,
                    scan_direction=self.scan_direction,
                    cross_section=self.cross_section,
                    passes=int(self.passes),
                )
            )
        return self.shapes
