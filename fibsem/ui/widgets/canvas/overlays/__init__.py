"""Canvas overlays for :class:`fibsem.ui.widgets.canvas.image_canvas.FibsemImageCanvas`.

Each overlay implements the duck-typed overlay protocol (``attach`` / ``detach`` /
``on_content_changed``); most subclass :class:`CanvasOverlay`. Import overlay classes
from this package, e.g. ``from fibsem.ui.widgets.canvas.overlays import MillingPatternOverlay``.
"""

from fibsem.ui.widgets.canvas.overlays.alignment_overlay import AlignmentAreaOverlay
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay
from fibsem.ui.widgets.canvas.overlays.mask_overlay import MaskOverlay
from fibsem.ui.widgets.canvas.overlays.milling_overlay import MillingPatternOverlay
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import (
    FieldOfViewOverlay,
    PointOverlay,
    PointsOverlay,
)
from fibsem.ui.widgets.canvas.overlays.rect_overlay import RectOverlay
from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay
from fibsem.ui.widgets.canvas.overlays.scan_direction_overlay import (
    ScanDirectionArrowOverlay,
)

__all__ = [
    "CanvasOverlay",
    "PointsOverlay",
    "FieldOfViewOverlay",
    "PointOverlay",
    "RectOverlay",
    "RulerOverlay",
    "ScanDirectionArrowOverlay",
    "AlignmentAreaOverlay",
    "MaskOverlay",
    "MillingPatternOverlay",
    "MinimapShapesOverlay",
    "ShapeSpec",
]
