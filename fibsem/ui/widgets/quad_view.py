"""Quad-view microscope display — reusable 2x2 grid of FibsemImageCanvas.

Replaces the single napari viewer in the main microscope tab. Four cells:

    SEM (electron) | FIB (ion)
    FM (fluorescence) | "No Data" placeholder

``MicroscopeViewController`` wraps the widget and is the object handed to the
control widgets in place of the napari ``Viewer``. Its surface is intentionally
small in Phase 0 (image routing only) and grows in later phases (overlays,
per-beam click signals).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QSplitter, QVBoxLayout, QWidget

from fibsem.structures import BeamType, FibsemImage
from fibsem.ui.widgets.canvas_state import (
    CanvasState,
    MillingSpec,
    OverlaySpec,
    SceneModel,
)
from fibsem.ui.widgets.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.ui.widgets.image_canvas import CanvasOverlay

_logger = logging.getLogger(__name__)

_BG = "#1e2124"
_TITLE_STYLE = (
    "color: #888; font-size: 11px; padding: 2px 6px; background: #1e2124;"
)
_PLACEHOLDER_STYLE = "color: #777; font-size: 12px;"


def _titled(title: str, inner: QWidget) -> QWidget:
    """Wrap *inner* with a small title label above it."""
    w = QWidget()
    w.setStyleSheet(f"background: {_BG};")
    lbl = QLabel(title, alignment=Qt.AlignLeft)
    lbl.setStyleSheet(_TITLE_STYLE)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(lbl)
    lay.addWidget(inner)
    return w


def _splitter(orientation, *widgets) -> QSplitter:
    s = QSplitter(orientation)
    s.setChildrenCollapsible(False)
    for w in widgets:
        s.addWidget(w)
    s.setSizes([1000] * len(widgets))
    return s


class PlaceholderPanel(QFrame):
    """Inert 'No Data' panel for the 4th quad-view cell (no canvas, no toolbar)."""

    def __init__(self, text: str = "No Data") -> None:
        super().__init__()
        self.setStyleSheet(f"background: {_BG};")
        lbl = QLabel(text, alignment=Qt.AlignCenter)
        lbl.setStyleSheet(_PLACEHOLDER_STYLE)
        lay = QVBoxLayout(self)
        lay.addWidget(lbl)


class QuadViewWidget(QWidget):
    """2x2 grid: SEM | FIB over FM | placeholder, each a titled panel.

    The SEM/FIB cells are :class:`FibsemImageCanvas` instances (so they inherit
    the reset / scalebar / crosshair / contrast toolbar); the FM cell is an
    :class:`FMCanvasWidget` (multi-channel composite + per-channel controls), and
    the 4th is an inert placeholder. ``fm_canvas`` aliases the FM widget's inner
    canvas so overlays attach the same way they do on SEM/FIB.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sem_canvas = FibsemImageCanvas()
        self.fib_canvas = FibsemImageCanvas()
        self.fm_widget = FMCanvasWidget()
        self.fm_canvas = self.fm_widget.canvas
        self.placeholder = PlaceholderPanel("No Data")

        left = _splitter(
            Qt.Vertical, _titled("SEM", self.sem_canvas), _titled("FM", self.fm_widget)
        )
        right = _splitter(
            Qt.Vertical,
            _titled("FIB", self.fib_canvas),
            _titled("Placeholder", self.placeholder),
        )
        root = _splitter(Qt.Horizontal, left, right)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(root)


class MicroscopeViewController(QObject):
    """Seam object that replaces the napari ``Viewer`` in the main microscope tab.

    Holds the :class:`QuadViewWidget`, owns a declarative :class:`SceneModel`
    (image + overlays + info per canvas), and renders it onto the per-beam canvases
    in a single debounced pass. Producers mutate the model *only* through the reducer
    API (``set_image`` / ``set_overlay`` / ``remove_overlay``); they never touch the
    canvases or overlay objects directly. See
    ``docs/design/canvas-overlay-state-model.md``.
    """

    # Emitted (queued) when a canvas needs re-rendering; the queued connection
    # marshals the render onto the GUI thread, so the reducer API is safe to call
    # from worker threads.
    _render_requested = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._widget = QuadViewWidget()
        self._canvases: Dict[BeamType, FibsemImageCanvas] = {
            BeamType.ELECTRON: self._widget.sem_canvas,
            BeamType.ION: self._widget.fib_canvas,
        }

        # ── declarative state model (one CanvasState per canvas) ──────────
        self._scene = SceneModel()
        self._states: Dict[FibsemImageCanvas, CanvasState] = {
            self._widget.sem_canvas: self._scene.sem,
            self._widget.fib_canvas: self._scene.fib,
            self._widget.fm_canvas: self._scene.fm,
        }
        # reducer-owned overlay objects (id → overlay) per canvas; their lifetime
        # matches the canvas, so producers never attach or tear them down.
        self._overlay_objs: Dict[FibsemImageCanvas, Dict[str, "CanvasOverlay"]] = {
            canvas: {} for canvas in self._states
        }
        self._dirty: Set[FibsemImageCanvas] = set()
        self._render_scheduled = False
        self._render_requested.connect(self._do_render, Qt.QueuedConnection)

    @property
    def widget(self) -> QuadViewWidget:
        return self._widget

    @property
    def sem_canvas(self) -> FibsemImageCanvas:
        return self._widget.sem_canvas

    @property
    def fib_canvas(self) -> FibsemImageCanvas:
        return self._widget.fib_canvas

    @property
    def fm_widget(self) -> FMCanvasWidget:
        """The multi-channel FM widget (composite + per-channel controls)."""
        return self._widget.fm_widget

    @property
    def fm_canvas(self) -> FibsemImageCanvas:
        """The FM widget's inner canvas — for overlays / scalebar, like SEM/FIB."""
        return self._widget.fm_canvas

    def get_canvas(self, beam: BeamType) -> Optional[FibsemImageCanvas]:
        """Return the canvas for a charged-particle beam (ELECTRON / ION)."""
        return self._canvases.get(beam)

    def set_image(self, beam: BeamType, image: FibsemImage) -> None:
        """Display *image* on the canvas for *beam* and stash it on the canvas state
        (no-op for unknown beams).

        The stashed image is what the reducer injects into image-dependent overlays
        (e.g. milling), so a bare image swap re-renders them against the new image.
        """
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        canvas.set_image(image)
        self._states[canvas].image = image
        self._mark_dirty(canvas)

    def set_fm_channel(self, name: str, data, color: Optional[str] = None) -> None:
        """Upsert one fluorescence channel into the FM composite (by *name*)."""
        self._widget.fm_widget.set_channel(name, data, color)

    def set_fm_pixel_size(self, pixel_size: Optional[float]) -> None:
        """Set the FM canvas pixel size (metres/px) for the scalebar."""
        self._widget.fm_widget.set_pixel_size(pixel_size)

    def set_fm_image(self, image: "FluorescenceImage") -> None:
        """Composite an acquired ``FluorescenceImage`` (all channels) onto the FM
        canvas. See :meth:`FMCanvasWidget.set_fm_image`."""
        self._widget.fm_widget.set_fm_image(image)

    # ── overlay reducer ───────────────────────────────────────────────────
    def set_overlay(self, beam: BeamType, spec: OverlaySpec) -> None:
        """Upsert an overlay *spec* (keyed by ``spec.id``) on the canvas for *beam*;
        rendered on the next debounced pass. No-op for unknown beams."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        self._states[canvas].overlays[spec.id] = spec
        self._mark_dirty(canvas)

    def remove_overlay(self, beam: BeamType, overlay_id: str) -> None:
        """Remove the overlay with *overlay_id* from the canvas for *beam*."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        if self._states[canvas].overlays.pop(overlay_id, None) is not None:
            self._mark_dirty(canvas)

    # ── render loop ───────────────────────────────────────────────────────
    def _mark_dirty(self, canvas: FibsemImageCanvas) -> None:
        self._dirty.add(canvas)
        if not self._render_scheduled:
            self._render_scheduled = True
            self._render_requested.emit()  # queued → coalesced render on the GUI thread

    def _do_render(self) -> None:
        self._render_scheduled = False
        dirty = list(self._dirty)  # snapshot (GIL-safe vs. concurrent producers)
        self._dirty.clear()
        for canvas in dirty:
            try:
                self._reconcile(canvas)
            except Exception:
                _logger.exception("MicroscopeViewController: reconcile failed")

    def _reconcile(self, canvas: FibsemImageCanvas) -> None:
        """Diff a canvas's specs against its owned overlay objects: create / drive /
        remove so the rendered overlays match the model."""
        state = self._states[canvas]
        objs = self._overlay_objs[canvas]
        image = state.image
        # drop overlays whose spec is gone
        for oid in list(objs.keys()):
            if oid not in state.overlays:
                canvas.remove_overlay(objs.pop(oid))
        # create + drive overlays from their specs
        for oid, spec in list(state.overlays.items()):
            obj = objs.get(oid)
            if obj is None:
                obj = self._make_overlay(spec)
                if obj is None:
                    continue
                objs[oid] = obj
                canvas.add_overlay(obj)
            self._drive_overlay(obj, spec, image)

    def _make_overlay(self, spec: OverlaySpec):
        """Construct the overlay object for *spec* (one branch per migrated slice)."""
        if isinstance(spec, MillingSpec):
            from fibsem.ui.widgets.milling_overlay import MillingPatternOverlay

            return MillingPatternOverlay()
        _logger.warning(
            "MicroscopeViewController: no renderer for spec %r", type(spec).__name__
        )
        return None

    def _drive_overlay(self, obj, spec: OverlaySpec, image: Optional[FibsemImage]) -> None:
        """Push *spec* data into its overlay object, injecting the canvas image."""
        if isinstance(spec, MillingSpec):
            if image is None:
                obj.clear()
                return
            obj.set_stages(
                spec.stages,
                image,
                background_stages=spec.background_stages,
                selected_index=spec.selected_index,
            )

    def clear(self) -> None:
        """Clear all canvases (images + reducer-owned overlays) back to placeholders."""
        for canvas, state in self._states.items():
            for obj in list(self._overlay_objs[canvas].values()):
                canvas.remove_overlay(obj)
            self._overlay_objs[canvas].clear()
            state.image = None
            state.overlays.clear()
            state.info.clear()
            state.armed_overlay = None
        self.sem_canvas.clear()
        self.fib_canvas.clear()
        self._widget.fm_widget.clear()
