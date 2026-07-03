"""Canvas overlays for :class:`fibsem.ui.widgets.image_canvas.FibsemImageCanvas`.

Each overlay implements the duck-typed overlay protocol (``attach`` / ``detach`` /
``on_image_changed``); most subclass :class:`CanvasOverlay`. Import overlay classes
from this package, e.g. ``from fibsem.ui.widgets.overlays import MillingPatternOverlay``.
"""

from fibsem.ui.widgets.overlays.base import CanvasOverlay
from fibsem.ui.widgets.overlays.point_overlay import PointOverlay, PointsOverlay
from fibsem.ui.widgets.overlays.rect_overlay import RectOverlay
from fibsem.ui.widgets.overlays.ruler_overlay import RulerOverlay
from fibsem.ui.widgets.overlays.pattern_overlay import (
    PatternOverlay,
    ScanDirectionArrowOverlay,
)
from fibsem.ui.widgets.overlays.alignment_overlay import AlignmentAreaOverlay
from fibsem.ui.widgets.overlays.mask_overlay import MaskOverlay
from fibsem.ui.widgets.overlays.milling_overlay import MillingPatternOverlay
from fibsem.ui.widgets.overlays.minimap_overlays import MinimapShapesOverlay, ShapeSpec

__all__ = [
    "CanvasOverlay",
    "PointsOverlay",
    "PointOverlay",
    "RectOverlay",
    "RulerOverlay",
    "PatternOverlay",
    "ScanDirectionArrowOverlay",
    "AlignmentAreaOverlay",
    "MaskOverlay",
    "MillingPatternOverlay",
    "MinimapShapesOverlay",
    "ShapeSpec",
]
