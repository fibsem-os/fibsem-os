from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    List,
)

from fibsem.milling.patterning.base import BasePattern
from fibsem.milling.properties import (
    cross_section_metadata,
    distance_metadata,
)
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangleSettings,
)


@dataclass
class UndercutPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=5.0e-6,
        metadata=distance_metadata("Width", "Width of the undercut pattern."),
    )
    height: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Height", "Height of the undercut pattern."),
    )
    depth: float = field(
        default=10.0e-6,
        metadata=distance_metadata("Depth", "Depth of the undercut pattern."),
    )
    trench_width: float = field(
        default=2.0e-6,
        metadata=distance_metadata(
            "Trench Width", "Width of the trench in the undercut pattern."
        ),
    )
    rhs_height: float = field(
        default=10.0e-6,
        metadata=distance_metadata(
            "RHS Height", "Height of the right-hand side of the undercut pattern."
        ),
    )
    h_offset: float = field(
        default=5.0e-6,
        metadata=distance_metadata(
            "Horizontal Offset", "Horizontal offset of the undercut pattern."
        ),
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle, metadata=cross_section_metadata()
    )

    name: ClassVar[str] = "Undercut"

    def define(self) -> List[FibsemRectangleSettings]:

        point = self.point
        jcut_rhs_height = self.rhs_height
        jcut_lamella_height = self.height
        jcut_width = self.width
        jcut_trench_thickness = self.trench_width
        jcut_depth = self.depth
        jcut_h_offset = self.h_offset
        cross_section = self.cross_section

        # top_jcut
        jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
        jcut_top_centre_y = point.y + jcut_lamella_height
        jcut_top_width = jcut_width
        jcut_top_height = jcut_trench_thickness
        jcut_top_depth = jcut_depth

        top_pattern = FibsemRectangleSettings(
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
            centre_x=point.x,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )
        # rhs jcut
        jcut_rhs_centre_x = point.x + (jcut_width / 2) - jcut_trench_thickness / 2
        jcut_rhs_centre_y = point.y - (jcut_rhs_height / 2) + jcut_trench_thickness / 2
        jcut_rhs_width = jcut_trench_thickness
        jcut_rhs_height = jcut_rhs_height
        jcut_rhs_depth = jcut_depth

        rhs_pattern = FibsemRectangleSettings(
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
            centre_x=jcut_rhs_centre_x,
            centre_y=jcut_rhs_centre_y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
        )

        self.shapes = [top_pattern, rhs_pattern]
        return self.shapes
