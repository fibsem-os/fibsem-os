"""Canvas subsystem: matplotlib-based image/FM canvases, overlays, and view controllers.

This package groups the (napari-free) canvas widgets and their supporting state:

- :mod:`~fibsem.ui.widgets.canvas.image_canvas` — the core :class:`FibsemImageCanvas`.
- :mod:`~fibsem.ui.widgets.canvas.fm_canvas` — fluorescence-microscopy canvas.
- :mod:`~fibsem.ui.widgets.canvas.fm_composite` — FM layer compositing helpers.
- :mod:`~fibsem.ui.widgets.canvas.quad_view` — quad-view widget, lamella editor, and
  :class:`MicroscopeViewController`.
- :mod:`~fibsem.ui.widgets.canvas.canvas_state` — declarative scene/overlay specs.
- :mod:`~fibsem.ui.widgets.canvas.overlays` — the overlay classes drawn on the canvases.

The public API is re-exported here so callers can write, e.g.::

    from fibsem.ui.widgets.canvas import FibsemImageCanvas, MicroscopeViewController

Imports are ordered leaf-first to avoid partial-initialisation issues.
"""

# Leaf modules first (canvas_state, fm_composite, contrast_gamma_control have no
# intra-cluster deps), then overlays, then image_canvas, fm_canvas, quad_view.
from fibsem.ui.widgets.canvas.canvas_state import (
    AlignmentSpec,
    CanvasState,
    MaskSpec,
    MillingSpec,
    OverlaySpec,
    PointsSpec,
    SceneModel,
)
from fibsem.ui.widgets.canvas.fm_composite import FMLayer, composite_fm_layers
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
    QuadViewWidget,
)

__all__ = [
    # image_canvas
    "FibsemImageCanvas",
    # fm_canvas
    "FMCanvasWidget",
    # fm_composite
    "FMLayer",
    "composite_fm_layers",
    # quad_view
    "QuadViewWidget",
    "LamellaEditorView",
    "MicroscopeViewController",
    # canvas_state
    "SceneModel",
    "CanvasState",
    "OverlaySpec",
    "MillingSpec",
    "PointsSpec",
    "MaskSpec",
    "AlignmentSpec",
]
