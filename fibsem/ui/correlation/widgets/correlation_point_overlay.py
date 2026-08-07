"""Correlation points on the shared canvas stack (FIB-535).

``CorrelationPointOverlay`` subclasses :class:`PointOverlay` and swaps its point
model for ``Coordinate`` — which is the whole reason this class exists.

Correlation points are not interchangeable: a ``Coordinate`` carries a
``PointType``, and the tab widget's registry keys exclusivity and lifecycle off
that identity (every handler's first line is ``self._point_specs[...]``, and one
removal path filters with ``c is not coord`` — object identity, not equality).
An index cannot stand in for that.

Two things follow, and they are why a subclass beats widening ``PointOverlay``:

* **Identity travels in the signals.** Base signals stay index-based for their
  five existing consumers; the identity-carrying ones live here, where they are
  the API rather than dead weight.
* **One surface shows several ``PointType``s at once.** ``PointOverlay`` is a
  single flat list with one style; holding ``List[Coordinate]`` here makes that
  a non-problem instead of something to translate.

``_coords`` is index-aligned with the base's ``_points``. That alignment is safe
because the base mutates ``_points`` structurally in exactly four public methods
(``set_points`` / ``add_point`` / ``remove_point`` / ``clear_points``), all
overridden below; its fifth mutation is the drag, which moves a point without
reordering. Index bookkeeping on removal — including shifting ``_selected`` —
stays in the base, which is precisely the part a translation layer would have
had to reimplement.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.patheffects as pe
from PyQt5.QtCore import pyqtSignal

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui.tokens import ORANGE_COLOR
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

# Saturated primaries, chosen to stay legible against arbitrary image content
# rather than to match the app palette — the same reasoning that keeps the
# NEUTRAL_* ramp out of the tinted tokens.
POINT_COLORS: Dict[PointType, str] = {
    PointType.FIB:        "#00ff00",
    PointType.FM:         "#00e5ff",
    PointType.POI:        "#ff00ff",
    PointType.SURFACE:    ORANGE_COLOR,
    PointType.SURFACE_FM: "#ffea00",
}
POINT_MARKERS: Dict[PointType, str] = {
    PointType.FIB:        "o",
    PointType.FM:         "o",
    PointType.POI:        "o",
    PointType.SURFACE:    "+",
    PointType.SURFACE_FM: "+",
}
MARKER_SIZE = 5.0  # the base draws the selected marker at size * 1.4 = 7.0

# Every label colour above is a bright saturated hue, so a dark stroke is the
# complement of all of them: it rescues SURFACE and FM-SURFACE against a bright
# image, where they otherwise disappear entirely. A *white* stroke is the wrong
# choice for the same reason -- it merges into a bright background, which is
# exactly where the outline has to work. 0.5 matches the old canvas; heavier
# starts eating the glyph at this font size.
LABEL_OUTLINE = [pe.withStroke(linewidth=0.5, foreground="black")]


def generate_names(coordinates: List[Coordinate]) -> List[str]:
    """Per-type 1-based names: ``FIB 1``, ``FIB 2``, ``POI 1`` ...

    Numbered within a type rather than across the list, so a point's label does
    not change when an unrelated type gains or loses one.
    """
    counters: Dict[PointType, int] = {}
    names = []
    for c in coordinates:
        counters[c.point_type] = counters.get(c.point_type, 0) + 1
        names.append(f"{c.point_type.value} {counters[c.point_type]}")
    return names


class CorrelationPointOverlay(PointOverlay):
    """Interactive correlation points, addressed by ``Coordinate`` identity."""

    # Identity-carrying counterparts of the base's index-based signals. The base
    # ones still fire; consumers here should use these.
    coordinate_selected = pyqtSignal(object)  # Coordinate
    coordinate_moved = pyqtSignal(object)  # Coordinate (drag finished)
    coordinate_removed = pyqtSignal(object)  # Coordinate (before removal)
    # The view decides which PointType a new point gets (it owns the menu), so
    # the overlay only reports where the user asked for one.
    add_requested = pyqtSignal(float, float)  # x, y

    def __init__(self, parent=None) -> None:
        super().__init__(size=MARKER_SIZE, parent=parent)
        self._coords: List[Coordinate] = []
        self._names: List[str] = []
        self._legend_visible = True
        self._labels_visible = True
        self.point_selected.connect(self._emit_selected)
        self.point_moved.connect(self._emit_moved)
        self.point_removed.connect(self._emit_removed)

    # ── model ─────────────────────────────────────────────────────────────

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        """Replace the displayed set. Coordinates are held by reference, so the
        objects handed back by the signals are the caller's own."""
        self._coords = list(coords)
        self._names = generate_names(self._coords)
        self.set_points([(c.point.x, c.point.y) for c in self._coords])

    def add_coordinate(self, coord: Coordinate) -> int:
        self._coords.append(coord)
        self._names = generate_names(self._coords)
        idx = super().add_point(coord.point.x, coord.point.y)
        self._refresh_chrome()
        return idx

    def remove_coordinate(self, coord: Coordinate) -> None:
        idx = self.index_of(coord)
        if idx is not None:
            self.remove_point(idx)

    def coordinates(self) -> List[Coordinate]:
        return list(self._coords)

    def index_of(self, coord: Optional[Coordinate]) -> Optional[int]:
        """Index of *coord* by object identity, not equality — two coordinates
        can hold equal values and still be different points."""
        for i, c in enumerate(self._coords):
            if c is coord:
                return i
        return None

    def selected_coordinate(self) -> Optional[Coordinate]:
        idx = self._selected
        return self._coords[idx] if idx is not None and idx < len(self._coords) else None

    def set_selected_coordinate(self, coord: Optional[Coordinate]) -> None:
        """Select by identity. Silent, like the base's ``set_selected``."""
        self.set_selected(self.index_of(coord))

    def refresh_coordinate(self, coord: Coordinate) -> None:
        """Re-read a coordinate's position after an external edit."""
        idx = self.index_of(coord)
        if idx is None:
            return
        self._points[idx] = [float(coord.point.x), float(coord.point.y)]
        self._update_artist_position(idx)
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── base overrides keeping _coords aligned ────────────────────────────

    def add_point(self, x: float, y: float) -> int:
        """Not available: a correlation point needs a PointType, which only the
        view can supply. Raises rather than appending, because a point without a
        matching entry in ``_coords`` desynchronises the two lists silently."""
        raise NotImplementedError("use add_coordinate(); a bare point has no PointType")

    def remove_point(self, index: int) -> None:
        if index < 0 or index >= len(self._coords):
            return
        # super() emits point_removed(index) before it pops, so _emit_removed can
        # still read _coords[index]; pop only once it returns.
        super().remove_point(index)
        self._coords.pop(index)
        self._names = generate_names(self._coords)
        self._refresh_chrome()

    def clear_points(self) -> None:
        super().clear_points()
        self._coords.clear()
        self._names = []

    # ── per-point style ───────────────────────────────────────────────────

    def _point_color(self, idx: int, selected: bool) -> str:
        """Colour by type. A selected point keeps its own colour — the rim and
        size carry the selection instead, so the type stays readable."""
        if idx >= len(self._coords):
            return super()._point_color(idx, selected)
        return POINT_COLORS.get(self._coords[idx].point_type, "white")

    def _point_marker(self, idx: int) -> str:
        if idx >= len(self._coords):
            return super()._point_marker(idx)
        return POINT_MARKERS.get(self._coords[idx].point_type, "o")

    def _point_label(self, idx: int) -> Optional[str]:
        return self._names[idx] if idx < len(self._names) else None

    def _marker_edge(self, idx: int, color: str, selected: bool):
        """Filled markers show selection as a white rim; unfilled ones ("+") are
        drawn entirely by their edge, so a white edge would replace the type
        colour and "none" would erase the marker — they thicken instead."""
        from matplotlib.lines import Line2D

        if self._point_marker(idx) in Line2D.filled_markers:
            return ("white" if selected else "none"), 2.0
        return color, (3.0 if selected else 2.0)

    # ── legend and labels ─────────────────────────────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = visible
        self._draw_legend()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_labels_visible(self, visible: bool) -> None:
        """Show or hide the per-point names without discarding the points.

        Separate from ``set_visible``, which hides the markers too: an operator
        clearing labels off a crowded image still wants to see the points.
        """
        self._labels_visible = visible
        self._apply_label_visibility()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_visible(self, visible: bool) -> None:
        super().set_visible(visible)
        # the base turns every annotation back on with the markers; labels the
        # operator switched off should stay off
        self._apply_label_visibility()

    def _apply_label_visibility(self) -> None:
        for ann in self._anns:
            if ann is not None:
                ann.set_visible(self._labels_visible and self._visible)

    def _append_artist(self, idx: int) -> None:
        super()._append_artist(idx)
        ann = self._anns[-1] if self._anns else None
        if ann is not None:
            # a point added while labels are off must not arrive showing its name
            ann.set_visible(self._labels_visible and self._visible)
            ann.set_path_effects(LABEL_OUTLINE)

    def _legend_entries(self):
        """One swatch per PointType currently on screen.

        The base draws a single entry because its points are homogeneous. Here
        they are not, and the colour is the only thing distinguishing a FIB point
        from a POI — so the legend is what makes the display readable, not
        decoration. Iterating ``PointType`` rather than the coordinates keeps the
        order stable as points come and go.
        """
        from matplotlib.lines import Line2D

        if not self._legend_visible:
            return [], []
        handles, labels = [], []
        for pt in PointType:
            if any(c.point_type is pt for c in self._coords):
                handles.append(self._legend_handle(pt))
                labels.append(pt.value)
        return handles, labels

    def _legend_handle(self, point_type: PointType) -> "Line2D":
        """A proxy artist matching how the point is actually drawn — same rule as
        `_marker_edge`, so an unfilled "+" keeps its colour instead of going white."""
        from matplotlib.lines import Line2D

        color = POINT_COLORS.get(point_type, "white")
        marker = POINT_MARKERS.get(point_type, "o")
        unfilled = marker not in Line2D.filled_markers
        return Line2D(
            [], [],
            marker=marker,
            markersize=7,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color if unfilled else "white",
            markeredgewidth=1.5 if unfilled else 0.8,
            linestyle="none",
            label=point_type.value,
        )

    # ── interaction ───────────────────────────────────────────────────────

    def _on_right_click(self, x: float, y: float) -> None:
        """Report the request and add nothing: the view has to ask which
        PointType before a Coordinate can exist."""
        self.add_requested.emit(x, y)

    # ── index → identity ──────────────────────────────────────────────────

    def _emit_selected(self, idx: int, x: float, y: float) -> None:
        if idx < len(self._coords):
            self.coordinate_selected.emit(self._coords[idx])

    def _emit_moved(self, idx: int, x: float, y: float) -> None:
        if idx >= len(self._coords):
            return
        coord = self._coords[idx]
        coord.point.x, coord.point.y = float(x), float(y)
        self.coordinate_moved.emit(coord)

    def _emit_removed(self, idx: int) -> None:
        if idx < len(self._coords):
            self.coordinate_removed.emit(self._coords[idx])

    def _refresh_chrome(self) -> None:
        """Re-derive the labels and the legend after the coordinate set changes.

        ``set_points`` redraws both via ``_draw_all``, but ``add_point`` and
        ``remove_point`` only touch one artist -- and both can change which
        PointTypes are on screen, which is exactly what the legend reports.
        """
        if self._anns:
            self._refresh_ann_text()
        self._apply_label_visibility()
        self._draw_legend()
