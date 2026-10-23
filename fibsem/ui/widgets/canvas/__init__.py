"""Canvas subsystem: matplotlib-based image/FM canvases, overlays, and view controllers.

This package groups the canvas widgets and their supporting state:

- :mod:`~fibsem.ui.widgets.canvas.canvas_base` — :class:`FibsemCanvasBase`, the
  content-agnostic navigation / overlay / toolbar / chrome half of the canvas.
- :mod:`~fibsem.ui.widgets.canvas.image_canvas` — the core :class:`FibsemImageCanvas`.
- :mod:`~fibsem.ui.widgets.canvas.real_space_canvas` — :class:`FibsemRealSpaceCanvas`,
  many images drawn at the positions they were acquired.
- :mod:`~fibsem.ui.widgets.canvas.fm_canvas` — fluorescence-microscopy canvas.
- :mod:`~fibsem.ui.widgets.canvas.fm_composite` — FM layer compositing helpers.
- :mod:`~fibsem.ui.widgets.canvas.quad_view` — quad-view widget, item editor, and
  :class:`MicroscopeViewController`.
- :mod:`~fibsem.ui.widgets.canvas.canvas_state` — declarative scene/overlay specs.
- :mod:`~fibsem.ui.widgets.canvas.overlays` — the overlay classes drawn on the canvases.

Importing the package imports none of them; import each from its module.
"""
