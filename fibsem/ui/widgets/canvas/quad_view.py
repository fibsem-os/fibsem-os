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

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.structures import BeamType, FibsemImage
from fibsem.ui.widgets.canvas.canvas_state import (
    AlignmentSpec,
    CanvasState,
    MaskSpec,
    MillingSpec,
    OverlaySpec,
    PointsSpec,
    SceneModel,
)
from fibsem.ui.stylesheets import CANVAS_BG as _BG, PRIMARY_ACCENT as _SELECT_ACCENT
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.structures import FibsemRectangle
    from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

_logger = logging.getLogger(__name__)

_TITLE_STYLE = (
    "color: #888; font-size: 11px; padding: 2px 6px; background: #1e2124;"
)
_PLACEHOLDER_STYLE = "color: #777; font-size: 12px;"
# Selected-view border: the primary accent (matches PRIMARY_BUTTON_STYLESHEET), kept subtle.
# A transparent border of the same width is always present so selection causes no layout shift,
# and it's scoped via the #viewPanel object name so it never cascades onto the title / canvas.
_PANEL_QSS = "#viewPanel {{ background: {bg}; border: 2px solid {border}; }}"
# Live-acquisition border: green, and it takes priority over the blue selected border so a
# live view is obvious even when you've clicked away to another cell.
_LIVE_ACCENT = "#4caf50"


def _titled(title: str, inner: QWidget) -> QFrame:
    """Wrap *inner* in a selectable panel frame with a small title label above it."""
    frame = QFrame()
    frame.setObjectName("viewPanel")
    frame.setAttribute(Qt.WA_StyledBackground, True)
    frame.setStyleSheet(_PANEL_QSS.format(bg=_BG, border="transparent"))
    lbl = QLabel(title, alignment=Qt.AlignLeft)
    lbl.setStyleSheet(_TITLE_STYLE)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(lbl)
    lay.addWidget(inner)
    return frame


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
    """2x2 grid: SEM | FIB over FM | placeholder, each a selectable titled panel.

    The SEM/FIB cells are :class:`FibsemImageCanvas` instances (so they inherit
    the reset / scalebar / crosshair / contrast toolbar); the FM cell is an
    :class:`FMCanvasWidget` (multi-channel composite + per-channel controls), and
    the 4th is an inert placeholder. ``fm_canvas`` aliases the FM widget's inner
    canvas so overlays attach the same way they do on SEM/FIB.

    The most-recently-clicked canvas is the *selected* view: its panel gets a subtle
    primary border, only its canvas toolbar is shown, and it is the target for
    view-scoped hotkeys. All canvases stay fully interactive regardless of selection.
    """

    # Emitted when the selected view changes; payload is the view key
    # (BeamType.ELECTRON / BeamType.ION for SEM / FIB, or the string "fm").
    view_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sem_canvas = FibsemImageCanvas()
        self.fib_canvas = FibsemImageCanvas()
        self.fm_widget = FMCanvasWidget()
        self.fm_canvas = self.fm_widget.canvas
        self.placeholder = PlaceholderPanel("No Data")

        sem_panel = _titled("SEM", self.sem_canvas)
        fm_panel = _titled("FM", self.fm_widget)
        fib_panel = _titled("FIB", self.fib_canvas)
        placeholder_panel = _titled("Placeholder", self.placeholder)

        left = _splitter(Qt.Vertical, sem_panel, fm_panel)
        right = _splitter(Qt.Vertical, fib_panel, placeholder_panel)
        root = _splitter(Qt.Horizontal, left, right)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(root)

        # ── full-screen state ─────────────────────────────────────────────
        # Full screen hides a cell's vertical-splitter sibling + the other vertical splitter,
        # leaving one cell filling the quad. Reversible: saved sizes restore the grid.
        self._root_splitter = root
        self._splitters = {"left": left, "right": right}
        self._all_panels = [sem_panel, fm_panel, fib_panel, placeholder_panel]
        self._panel_splitter = {
            sem_panel: left, fm_panel: left, fib_panel: right, placeholder_panel: right,
        }
        self._placeholder_panel = placeholder_panel
        self._fullscreen: Optional[object] = None
        self._saved_sizes: dict = {}

        # ── selected-view state ───────────────────────────────────────────
        self._panels: Dict[object, QFrame] = {
            BeamType.ELECTRON: sem_panel,
            BeamType.ION: fib_panel,
            "fm": fm_panel,
        }
        self._canvas_keys: Dict[QWidget, object] = {
            self.sem_canvas: BeamType.ELECTRON,
            self.fib_canvas: BeamType.ION,
            self.fm_canvas: "fm",
        }
        self._key_canvas: Dict[object, QWidget] = {v: k for k, v in self._canvas_keys.items()}
        self._live: set = set()  # view keys currently live-acquiring (green border + LIVE badge)
        self._selected: Optional[object] = None
        for canvas in self._canvas_keys:
            canvas.installEventFilter(self)
        # default: SEM selected, so exactly one view always has the border + toolbar
        self.set_selected(BeamType.ELECTRON)

    @property
    def selected(self) -> Optional[object]:
        """The selected view key (``BeamType.ELECTRON`` / ``BeamType.ION`` / ``"fm"``)."""
        return self._selected

    def set_selected(self, key: object) -> None:
        """Select the view *key*: highlight its panel, show only its canvas toolbar, and
        emit :attr:`view_selected`. No-op for unknown keys or re-selecting the current view."""
        if key not in self._panels or key == self._selected:
            return
        self._selected = key
        self._refresh_borders()
        for canvas, k in self._canvas_keys.items():
            canvas.set_toolbar_visible(k == key)
        self.view_selected.emit(key)

    def _panel_border(self, key: object) -> str:
        """Border colour for a panel: green while live (wins), else blue if selected, else none."""
        if key in self._live:
            return _LIVE_ACCENT
        if key == self._selected:
            return _SELECT_ACCENT
        return "transparent"

    def _refresh_borders(self) -> None:
        for k, frame in self._panels.items():
            frame.setStyleSheet(_PANEL_QSS.format(bg=_BG, border=self._panel_border(k)))

    def set_live(self, key: object, live: bool) -> None:
        """Mark view *key* (a ``BeamType``) as live-acquiring: green border + a "LIVE" badge on
        its canvas, independent of which view is selected. The border wins over selection."""
        if key not in self._panels:
            return
        if live:
            self._live.add(key)
        else:
            self._live.discard(key)
        self._refresh_borders()
        canvas = self._key_canvas.get(key)
        if canvas is not None and hasattr(canvas, "set_live_badge"):
            canvas.set_live_badge(live)

    def set_fm_visible(self, visible: bool) -> None:
        """Show/hide the FM cell (plus the inert 'No Data' placeholder that balances the 2x2
        grid), so a system with no fluorescence microscope shows a clean SEM | FIB layout.
        If FM was the selected view and is being hidden, selection falls back to SEM."""
        self._panels["fm"].setVisible(visible)
        self._placeholder_panel.setVisible(visible)
        if not visible and self._selected == "fm":
            self.set_selected(BeamType.ELECTRON)

    def eventFilter(self, obj, event):
        """Select the view whose canvas was pressed (any mouse button). The event is not
        consumed, so clicks still pan / move the stage / open menus exactly as before."""
        if event.type() == QEvent.MouseButtonPress:
            key = self._canvas_keys.get(obj)
            if key is not None:
                self.set_selected(key)
        return super().eventFilter(obj, event)

    # ── full screen ─────────────────────────────────────────────────────────
    @property
    def fullscreen(self) -> Optional[object]:
        """The full-screened view key, or ``None`` when showing the 2x2 grid."""
        return self._fullscreen

    def toggle_fullscreen(self) -> None:
        """Toggle full screen for the selected view (no-op if nothing is selected)."""
        if self._fullscreen is not None:
            self.set_fullscreen(None)
        elif self._selected is not None:
            self.set_fullscreen(self._selected)

    def set_fullscreen(self, key: Optional[object]) -> None:
        """Full-screen the view *key* (``BeamType.ELECTRON`` / ``BeamType.ION`` / ``"fm"``),
        hiding the other cells; pass ``None`` to restore the 2x2 grid. No-op for unknown keys
        or re-requesting the current state.

        The full-screened view is also made the selected view, so the one visible cell always
        shows its toolbar (selection drives toolbar visibility)."""
        if key == self._fullscreen:
            return
        if key is None:
            self._restore_grid()
            return
        if key not in self._panels:
            return
        self.set_selected(key)  # invariant: the full-screened cell is the selected one
        if self._fullscreen is None:
            # entering from the grid: remember the grid's splitter sizes for restore
            self._saved_sizes = {
                "root": self._root_splitter.sizes(),
                "left": self._splitters["left"].sizes(),
                "right": self._splitters["right"].sizes(),
            }
        target = self._panels[key]
        keep_splitter = self._panel_splitter[target]
        for panel in self._all_panels:
            panel.setVisible(panel is target)
        for spl in self._splitters.values():
            spl.setVisible(spl is keep_splitter)
        self._fullscreen = key

    def _restore_grid(self) -> None:
        """Show all cells + both splitters and restore the saved grid sizes."""
        for panel in self._all_panels:
            panel.setVisible(True)
        for spl in self._splitters.values():
            spl.setVisible(True)
        if self._saved_sizes:
            self._root_splitter.setSizes(self._saved_sizes["root"])
            self._splitters["left"].setSizes(self._saved_sizes["left"])
            self._splitters["right"].setSizes(self._saved_sizes["right"])
        self._fullscreen = None


class LamellaEditorView(QWidget):
    """Task-driven stacked view backing the lamella protocol editor.

    Unlike :class:`QuadViewWidget` (a static 2x2 grid), this shows *one* page at a
    time, chosen by the selected task type — mirroring the old editor's per-task
    napari layer-visibility toggle:

        * "beams"        — the FIB canvas, with the SEM canvas beside it (hidden
                           unless toggled on). Milling / POI / alignment / spot-burn
                           overlays all live on the FIB canvas.
        * "fluorescence" — the multi-channel FM widget.

    Exposes ``sem_canvas`` / ``fib_canvas`` / ``fm_widget`` / ``fm_canvas`` so a
    :class:`MicroscopeViewController` can drive it exactly like ``QuadViewWidget``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sem_canvas = FibsemImageCanvas()
        self.fib_canvas = FibsemImageCanvas()
        self.fm_widget = FMCanvasWidget()
        self.fm_canvas = self.fm_widget.canvas

        self._sem_panel = _titled("SEM", self.sem_canvas)
        self._sem_panel.setVisible(False)  # shown on demand via set_sem_visible()
        self._beams_page = _splitter(
            Qt.Horizontal, _titled("FIB", self.fib_canvas), self._sem_panel
        )

        self._stack = QStackedWidget()
        self._stack.addWidget(self._beams_page)  # index 0: beams
        self._stack.addWidget(_titled("Fluorescence", self.fm_widget))  # index 1: FM

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._stack)

    def show_beams(self) -> None:
        """Show the FIB (+ optional SEM) page."""
        self._stack.setCurrentIndex(0)

    def show_fluorescence(self) -> None:
        """Show the multi-channel FM page."""
        self._stack.setCurrentIndex(1)

    def set_sem_visible(self, visible: bool) -> None:
        """Show/hide the SEM canvas beside FIB on the beams page."""
        self._sem_panel.setVisible(visible)


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

    # Emitted when the user commits an edit on an interactive overlay:
    # (beam, overlay_id, value). Producers subscribe and filter by overlay_id.
    overlay_edited = pyqtSignal(object, str, object)

    # Emitted when the user selects a point on a PointsSpec overlay:
    # (beam, overlay_id, index). Producers subscribe to mirror selection (e.g. a table).
    overlay_point_selected = pyqtSignal(object, str, int)

    # Forwarded from the view when the selected view changes (BeamType or "fm"). Only the
    # quad view emits this; the lamella editor view has no selection concept.
    view_selected = pyqtSignal(object)

    def __init__(
        self, parent: Optional[QObject] = None, view: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        # The view widget must expose sem_canvas / fib_canvas / fm_widget / fm_canvas.
        # Defaults to the 2x2 quad; the lamella editor passes a LamellaEditorView.
        self._widget = view if view is not None else QuadViewWidget()
        # Forward the view's selection signal when it supports selection (quad view does;
        # the lamella editor view does not).
        if hasattr(self._widget, "view_selected"):
            self._widget.view_selected.connect(self.view_selected)
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
        # reverse map (canvas → beam) for tagging edit signals; armed-mode applied
        # state (canvas → overlay object) so arming only toggles on change.
        self._beams: Dict[FibsemImageCanvas, BeamType] = {
            canvas: beam for beam, canvas in self._canvases.items()
        }
        self._armed_applied: Dict[FibsemImageCanvas, Optional["CanvasOverlay"]] = {
            canvas: None for canvas in self._states
        }
        self._dirty: Set[FibsemImageCanvas] = set()
        self._render_scheduled = False
        self._render_requested.connect(self._do_render, Qt.QueuedConnection)

    @property
    def widget(self) -> QWidget:
        """The view widget (``QuadViewWidget`` or a ``LamellaEditorView``)."""
        return self._widget

    @property
    def selected_view(self):
        """The selected view key (``BeamType.ELECTRON`` / ``BeamType.ION`` / ``"fm"``), or
        ``None`` for views (e.g. the lamella editor) that don't support selection."""
        return getattr(self._widget, "selected", None)

    @property
    def fullscreen(self):
        """The full-screened view key, or ``None`` (grid, or a view without full screen)."""
        return getattr(self._widget, "fullscreen", None)

    def set_fullscreen(self, key) -> None:
        """Full-screen a view (``BeamType`` or ``"fm"``), or restore the grid with ``None``.
        No-op on views that don't support it (e.g. the lamella editor)."""
        fn = getattr(self._widget, "set_fullscreen", None)
        if callable(fn):
            fn(key)

    def toggle_fullscreen(self) -> None:
        """Toggle full screen for the selected view; no-op on unsupported views."""
        fn = getattr(self._widget, "toggle_fullscreen", None)
        if callable(fn):
            fn()

    def set_fm_visible(self, visible: bool) -> None:
        """Show/hide the FM cell (call with ``microscope.fm is not None`` on connect) so a
        system without a fluorescence microscope shows a clean SEM | FIB layout. No-op on
        views that don't support it."""
        fn = getattr(self._widget, "set_fm_visible", None)
        if callable(fn):
            fn(visible)

    def set_live(self, key: BeamType, live: bool) -> None:
        """Mark a beam view as live-acquiring (green border + LIVE badge). No-op on views
        that don't support it."""
        fn = getattr(self._widget, "set_live", None)
        if callable(fn):
            fn(key, live)

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

    def clear_fm(self) -> None:
        """Drop all composited FM channels. Used on a lamella/task swap so a prior
        selection's channels (esp. differently-named ones) don't linger."""
        self._widget.fm_widget.clear()

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

    def arm_overlay(
        self,
        beam: BeamType,
        overlay_id: Optional[str],
        label: str = "",
        icon: str = "mdi:cursor-default-click",
    ) -> None:
        """Arm *overlay_id* for edit input on the canvas for *beam* (single arbiter;
        ``None`` returns to Move). The reducer drives the canvas's
        ``enter_overlay_mode`` / ``exit_overlay_mode`` (incl. the toolbar toggle)."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        state = self._states[canvas]
        state.armed_overlay = overlay_id
        state.armed_label = label
        state.armed_icon = icon
        self._mark_dirty(canvas)

    def set_overlay_visible(self, beam: BeamType, overlay_id: str, visible: bool) -> None:
        """Toggle an overlay's visibility without replacing its spec (keeps points /
        value). Applies to specs with a ``visible`` field (e.g. PointsSpec)."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        spec = self._states[canvas].overlays.get(overlay_id)
        if spec is not None and hasattr(spec, "visible"):
            spec.visible = visible
            self._mark_dirty(canvas)

    def set_points(self, beam: BeamType, overlay_id: str, points) -> None:
        """Replace a PointsSpec overlay's points without touching its config / visibility."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        spec = self._states[canvas].overlays.get(overlay_id)
        if isinstance(spec, PointsSpec):
            spec.points = list(points)
            self._mark_dirty(canvas)

    def set_selected_point(
        self, beam: BeamType, overlay_id: str, index: Optional[int]
    ) -> None:
        """Select a point on a ``PointsSpec`` overlay (e.g. mirroring a table row).

        Persists the selection on the spec (so it survives the next reconcile / a live
        frame re-render) and applies it to the live object if it exists. Silent — no
        echo — so producers can drive it from a table without a feedback loop."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        spec = self._states[canvas].overlays.get(overlay_id)
        if isinstance(spec, PointsSpec):
            spec.selected = index
        obj = self._overlay_objs.get(canvas, {}).get(overlay_id)
        if obj is not None and hasattr(obj, "set_selected"):
            obj.set_selected(index)

    def set_alignment_edit(
        self, beam: BeamType, rect: Optional["FibsemRectangle"], editing: bool
    ) -> None:
        """Image widget: start/stop editing the alignment area. Editing wins over the
        milling read-only display and owns input (armed). The spec is kept when
        editing stops (hidden if nothing else wants it) so the value survives the
        workflow's clear-then-read."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        spec = self._alignment_spec(canvas)
        spec.editing = editing
        if rect is not None:
            spec.rect = rect
        self.arm_overlay(
            beam,
            "alignment" if editing else None,
            label="Alignment",
            icon="mdi:vector-rectangle",
        )
        self._mark_dirty(canvas)

    def set_alignment_display(
        self, beam: BeamType, rect: Optional["FibsemRectangle"], show: bool
    ) -> None:
        """Milling viewer: request the alignment area shown read-only. Yields to an
        active edit and never clobbers an in-progress edit's rect."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return
        spec = self._states[canvas].overlays.get("alignment")
        if not isinstance(spec, AlignmentSpec):
            if not show:
                return
            spec = self._alignment_spec(canvas)
        spec.display = show
        if rect is not None and not spec.editing:
            spec.rect = rect
        self._mark_dirty(canvas)

    def _alignment_spec(self, canvas: FibsemImageCanvas) -> AlignmentSpec:
        spec = self._states[canvas].overlays.get("alignment")
        if not isinstance(spec, AlignmentSpec):
            spec = AlignmentSpec()
            self._states[canvas].overlays["alignment"] = spec
        return spec

    def alignment_area(self, beam: BeamType) -> Optional["FibsemRectangle"]:
        """The current alignment-area rect for *beam* — the model's authoritative
        value (updated on every user edit) — or ``None``."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return None
        spec = self._states[canvas].overlays.get("alignment")
        return getattr(spec, "rect", None)

    def overlay_points(self, beam: BeamType, overlay_id: str) -> list:
        """Current points (x, y) for a ``PointsSpec`` overlay — the model's
        authoritative value (updated on every add / move / remove) — or ``[]``."""
        canvas = self._canvases.get(beam)
        if canvas is None:
            return []
        spec = self._states[canvas].overlays.get(overlay_id)
        pts = getattr(spec, "points", None)
        return list(pts) if pts else []

    # ── info bar ──────────────────────────────────────────────────────────
    def set_info(self, beam: BeamType, key: str, text: Optional[str]) -> None:
        """Upsert an info-bar field on the canvas for *beam* (drop when text empty)."""
        canvas = self._canvases.get(beam)
        if canvas is not None:
            self._set_info_on(canvas, key, text)

    def set_fm_info(self, key: str, text: Optional[str]) -> None:
        """Upsert an info-bar field on the FM canvas (FM isn't a BeamType)."""
        self._set_info_on(self._widget.fm_canvas, key, text)

    def _set_info_on(self, canvas: FibsemImageCanvas, key: str, text: Optional[str]) -> None:
        info = self._states[canvas].info
        for i, (k, _) in enumerate(info):
            if k == key:
                if text:
                    info[i] = (key, text)
                else:
                    del info[i]
                self._mark_dirty(canvas)
                return
        if text:
            info.append((key, text))
            self._mark_dirty(canvas)

    def update_info(self, microscope, stage_position=None, objective_position=None) -> None:
        """Refresh the info bar from microscope state (mirrors napari
        ``update_text_overlay``): STAGE on SEM+FIB, MILLING ANGLE on FIB, OBJECTIVE on
        FM. It goes through the model + debounced render, so it is safe to call from an
        ``@ensure_main_thread`` ``update_ui`` — there is no synchronous draw to re-enter
        (which is what froze the original info bar)."""
        try:
            if type(microscope).__name__ == "TescanMicroscope":
                return  # no stage-position display yet
            if stage_position is None:
                stage_position = microscope._stage_position
            orientation = microscope.get_stage_orientation(stage_position=stage_position)
            grid = microscope.current_grid
            milling_angle = microscope.get_current_milling_angle(stage_position=stage_position)
            stage_txt = f"STAGE: {stage_position.pretty_string} [{orientation}] [{grid}]"
            self.set_info(BeamType.ELECTRON, "stage", stage_txt)
            self.set_info(BeamType.ION, "stage", stage_txt)
            self.set_fm_info("stage", stage_txt)  # universal context (before objective)
            self.set_info(BeamType.ION, "milling", f"MILLING ANGLE: {milling_angle:.1f}°")
            if microscope.fm is not None:
                if objective_position is None:
                    objective_position = microscope.fm.objective.position
                self.set_fm_info(
                    "objective",
                    f"OBJECTIVE: {objective_position * constants.METRE_TO_MICRON:.1f} µm",
                )
        except Exception:
            _logger.warning("MicroscopeViewController.update_info failed", exc_info=True)

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
                obj = self._create_overlay(canvas, oid, spec)
                if obj is None:
                    continue
                objs[oid] = obj
                canvas.add_overlay(obj)
            self._drive_overlay(obj, spec, image)
        self._apply_arming(canvas, state, objs)
        info_text = "\n".join(text for _, text in state.info if text)
        if (canvas._info_text or "") != info_text:
            canvas.set_info_text(info_text)

    def _create_overlay(self, canvas: FibsemImageCanvas, overlay_id: str, spec: OverlaySpec):
        """Construct the overlay object for *spec* and wire its edit signal (if any)
        back to :attr:`overlay_edited` (one branch per migrated slice)."""
        if isinstance(spec, MillingSpec):
            from fibsem.ui.widgets.canvas.overlays.milling_overlay import MillingPatternOverlay

            return MillingPatternOverlay()
        if isinstance(spec, MaskSpec):
            from fibsem.ui.widgets.canvas.overlays.mask_overlay import MaskOverlay

            return MaskOverlay()
        if isinstance(spec, AlignmentSpec):
            from fibsem.ui.widgets.canvas.overlays.alignment_overlay import AlignmentAreaOverlay

            obj = AlignmentAreaOverlay(editable=spec.editing)
            beam = self._beams.get(canvas)
            obj.alignment_area_changed.connect(
                lambda value, b=beam, i=overlay_id: self._on_overlay_edited(b, i, value)
            )
            return obj
        if isinstance(spec, PointsSpec):
            from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

            obj = PointOverlay(
                color=spec.color,
                selected_color=spec.selected_color,
                marker=spec.marker,
                size=spec.size,
                label_prefix=spec.label_prefix,
                add_on_right_click=spec.add_on_right_click,
                removable=spec.removable,
                modal=spec.modal,
                edge_width=spec.edge_width,
                legend_label=spec.legend_label,
                numbered=spec.numbered,
            )
            beam = self._beams.get(canvas)
            for sig in (obj.point_added, obj.point_moved, obj.point_removed):
                sig.connect(
                    lambda *a, b=beam, i=overlay_id, o=obj: self._on_overlay_edited(
                        b, i, o.get_points()
                    )
                )
            obj.point_selected.connect(
                lambda idx, x, y, b=beam, i=overlay_id: self.overlay_point_selected.emit(
                    b, i, idx
                )
            )
            return obj
        _logger.warning(
            "MicroscopeViewController: no renderer for spec %r", type(spec).__name__
        )
        return None

    def _drive_overlay(self, obj, spec: OverlaySpec, image: Optional[FibsemImage]) -> None:
        """Push *spec* data into its overlay object, injecting the canvas image."""
        if isinstance(spec, MillingSpec):
            if image is None or not spec.visible:
                obj.clear()
                return
            obj.set_stages(
                spec.stages,
                image,
                background_stages=spec.background_stages,
                selected_index=spec.selected_index,
            )
        elif isinstance(spec, AlignmentSpec):
            if spec.rect is not None:
                obj.set_area(spec.rect)
            obj.set_editable(spec.editing)
            obj.set_visible(spec.editing or spec.display)
        elif isinstance(spec, PointsSpec):
            obj.set_points(list(spec.points), colors=spec.colors, labels=spec.labels)
            obj.set_visible(spec.visible)
            obj.set_selected(spec.selected)  # re-apply: set_points nulls the selection
        elif isinstance(spec, MaskSpec):
            obj.set_mask(spec.mask, colors=spec.colors)

    def _apply_arming(self, canvas: FibsemImageCanvas, state: CanvasState, objs) -> None:
        """Make the model's armed overlay the canvas's active input mode (single
        arbiter). Toggles only on change; the exit is scoped to the overlay we armed,
        so it never tears down a mode another (still-direct) consumer owns."""
        desired_id = state.armed_overlay if state.armed_overlay in objs else None
        desired = objs.get(desired_id) if desired_id else None
        prev = self._armed_applied.get(canvas)
        if prev is desired:
            return
        if desired is not None:
            canvas.enter_overlay_mode(
                desired, state.armed_label, icon=state.armed_icon or "mdi:cursor-default-click"
            )
        elif prev is not None:
            canvas.exit_overlay_mode(prev)  # scoped: no-op unless prev still owns the mode
        self._armed_applied[canvas] = desired

    def _on_overlay_edited(self, beam, overlay_id: str, value) -> None:
        """Fold a committed overlay edit back into the model and re-emit it for
        producers. No re-render — the overlay already shows the edited value."""
        canvas = self._canvases.get(beam)
        if canvas is not None:
            spec = self._states[canvas].overlays.get(overlay_id)
            if isinstance(spec, AlignmentSpec):
                spec.rect = value
            elif isinstance(spec, PointsSpec):
                spec.points = value
        self.overlay_edited.emit(beam, overlay_id, value)

    def clear(self) -> None:
        """Clear all canvases (images + reducer-owned overlays) back to placeholders."""
        for canvas, state in self._states.items():
            for obj in list(self._overlay_objs[canvas].values()):
                canvas.remove_overlay(obj)
            self._overlay_objs[canvas].clear()
            self._armed_applied[canvas] = None
            state.image = None
            state.overlays.clear()
            state.info.clear()
            state.armed_overlay = None
            state.armed_label = ""
            state.armed_icon = ""
        self.sem_canvas.clear()
        self.fib_canvas.clear()
        self._widget.fm_widget.clear()
