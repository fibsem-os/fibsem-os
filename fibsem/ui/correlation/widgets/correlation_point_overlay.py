"""Correlation points on the shared canvas stack (FIB-535).

Two overlays, the same split the base module makes between ``PointsOverlay`` and
``PointOverlay``:

* :class:`CorrelationPointOverlay` — the interactive picked points, plus the
  dashed datum row marking the FIB surface height.
* :class:`CorrelationResultOverlay` — the static markers a correlation *result*
  reprojects onto the FIB image. Not pickable, not ``Coordinate``s, and cleared
  whenever the image beneath them changes.

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib.patheffects as pe
from PyQt5.QtCore import pyqtSignal

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui.tokens import ORANGE_COLOR
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

if TYPE_CHECKING:
    from matplotlib.lines import Line2D

    from fibsem.ui.widgets.canvas.canvas_base import ContentRect

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


LABEL_FONTSIZE = 8
LABEL_OFFSET = (7, 5)  # points, from the marker


def marker_edge(color: str, marker: str, hollow: bool) -> Tuple[str, float]:
    """Edge colour and width for a marker.

    An unfilled marker ("+") *is* its edge, and a hollow ring is drawn by its edge
    too, so both keep their own colour there — a white edge would swallow the
    colour and leave nothing carrying it. Filled markers keep the thin white
    outline that makes them legible against a bright image.

    Shared by the drawn marker and its legend swatch so the two cannot disagree.
    """
    from matplotlib.lines import Line2D

    if hollow or marker not in Line2D.filled_markers:
        return color, 1.5
    return "white", 0.8


def legend_handle(
    color: str, marker: str = "o", hollow: bool = False, alpha: float = 1.0
) -> "Line2D":
    """Proxy artist for one legend entry, styled like the marker it stands for."""
    from matplotlib.lines import Line2D

    edge_color, edge_width = marker_edge(color, marker, hollow)
    return Line2D(
        [], [],
        marker=marker,
        markersize=7,
        color=color,
        markerfacecolor="none" if hollow else color,
        markeredgecolor=edge_color,
        markeredgewidth=edge_width,
        alpha=alpha,
        linestyle="none",
    )


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
        # entries contributed by CorrelationResultOverlay; see set_extra_legend_entries
        self._extra_legend: Tuple[List["Line2D"], List[str]] = ([], [])
        self._surface_line = None  # dashed datum row; see _draw_surface_line
        self._surface_coord: Optional[Coordinate] = None
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
        if self._surface_line is not None:
            self._surface_line.set_visible(visible)

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

        The result overlay's entries are appended, so hiding the legend hides all
        of it and the picked-point types always read first.
        """
        if not self._legend_visible:
            return [], []
        handles, labels = [], []
        for pt in PointType:
            if any(c.point_type is pt for c in self._coords):
                handles.append(self._legend_handle(pt))
                labels.append(pt.value)
        extra_handles, extra_labels = self._extra_legend
        return handles + extra_handles, labels + extra_labels

    def _legend_handle(self, point_type: PointType) -> "Line2D":
        """A proxy artist matching how the point is actually drawn — same rule as
        `_marker_edge`, so an unfilled "+" keeps its colour instead of going white."""
        return legend_handle(
            POINT_COLORS.get(point_type, "white"),
            POINT_MARKERS.get(point_type, "o"),
        )

    def set_extra_legend_entries(
        self, handles: Sequence["Line2D"], labels: Sequence[str]
    ) -> None:
        """Adopt legend entries owned by another overlay on the same axes.

        Correlation's result markers live on :class:`CorrelationResultOverlay`, but
        a second ``Legend`` would land in the same corner as this one and sit on
        top of it. They describe the same thing anyway — what a marker on this
        image means — so there is one box, and this overlay draws it.
        """
        self._extra_legend = (list(handles), list(labels))
        self._draw_legend()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── the surface datum line ────────────────────────────────────────────

    def _draw_all(self) -> None:
        super()._draw_all()
        self._draw_surface_line()

    def _remove_all_artists(self) -> None:
        super()._remove_all_artists()
        self._remove_surface_line()

    def _draw_surface_line(self) -> None:
        """A dashed row across the image at the FIB surface height.

        The FIB surface is a datum an operator reads other features against, and a
        single marker on a large image is a poor way to read a height off. An FM
        surface is a z-plane instead, so no in-plane line applies to it — which is
        why only ``SURFACE`` gets one.

        Only the first is drawn: the tab widget's registry makes SURFACE exclusive,
        so a second one means the caller is mid-edit, not that two datums exist.
        """
        self._remove_surface_line()
        if self._ax is None:
            return
        surf = next(
            (c for c in self._coords if c.point_type is PointType.SURFACE), None
        )
        if surf is None:
            return
        self._surface_coord = surf
        self._surface_line = self._ax.axhline(
            surf.point.y,
            color=POINT_COLORS[PointType.SURFACE],
            linestyle="--",
            zorder=4,  # under the markers, which sit at 8
            visible=self._visible,
        )
        self._style_surface_line()

    def _remove_surface_line(self) -> None:
        if self._surface_line is not None:
            try:
                self._surface_line.remove()
            except Exception:
                pass  # already gone with ax.cla()
            self._surface_line = None
        self._surface_coord = None

    def _style_surface_line(self) -> None:
        """Selection reads as a heavier, brighter line — the colour stays the type's,
        the same rule the markers follow."""
        if self._surface_line is None:
            return
        selected = self._surface_coord is self.selected_coordinate()
        self._surface_line.set_linewidth(1.5 if selected else 1.0)
        self._surface_line.set_alpha(0.9 if selected else 0.6)

    def _update_artist_appearance(self, idx: int) -> None:
        super()._update_artist_appearance(idx)
        self._style_surface_line()

    def _update_artist_position(self, idx: int) -> None:
        super()._update_artist_position(idx)
        if self._surface_line is None or idx >= len(self._coords):
            return
        if self._coords[idx] is self._surface_coord:
            y = self._points[idx][1]
            self._surface_line.set_ydata([y, y])

    def _blit_artists(self) -> List:
        artists = super()._blit_artists()
        if self._surface_line is not None:
            artists.insert(0, self._surface_line)  # under the dragged marker
        return artists

    def _start_drag(self, idx: int, event) -> None:
        # Animated *before* super() captures the background, or the line is baked
        # into it at its old y and the blit paints a second one at the new y.
        if self._surface_line is not None:
            self._surface_line.set_animated(True)
        super()._start_drag(idx, event)

    def _on_release(self, event) -> None:
        if self._surface_line is not None:
            self._surface_line.set_animated(False)
        super()._on_release(event)

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
        PointTypes are on screen, which is exactly what the legend reports, and
        whether a SURFACE point exists to hang the datum line off.
        """
        if self._anns:
            self._refresh_ann_text()
        self._apply_label_visibility()
        self._draw_legend()
        self._draw_surface_line()


@dataclass
class _ResultGroup:
    """One styled batch of result markers, exactly as ``add_points`` received it.

    Kept rather than only drawn, because the axes is cleared and rebuilt on every
    image change and the group has to be redrawable from its own description.
    """

    points: List[Tuple[float, float]] = field(default_factory=list)
    color: str = "#ff0000"
    label_prefix: str = ""
    size: float = 7.0
    marker: str = "o"
    alpha: float = 1.0
    show_labels: bool = True
    hollow: bool = False
    legend_label: Optional[str] = None


class CorrelationResultOverlay(CanvasOverlay):
    """Static markers reporting a correlation result — reprojected fiducials, POI.

    Separate from :class:`CorrelationPointOverlay` because these are a different
    kind of thing: they are computed output, not picked input. Nothing here is
    draggable, deletable or addressed by ``Coordinate`` identity, and keeping them
    off the interactive overlay is what makes that true by construction — the
    interactive overlay hit-tests ``_points``, which these never enter.

    Groups accumulate: call :meth:`clear` before the first :meth:`add_points` when
    replacing a previous result set.

    Parameters
    ----------
    legend_host :
        Overlay that draws the shared legend, i.e. the ``CorrelationPointOverlay``
        on the same canvas. See
        :meth:`CorrelationPointOverlay.set_extra_legend_entries` for why there is
        one box rather than two.
    """

    def __init__(self, legend_host: Optional[CorrelationPointOverlay] = None) -> None:
        self._legend_host = legend_host
        self._groups: List[_ResultGroup] = []
        self._labels_visible = True
        self._ax = None
        self._canvas = None
        self._artists: list = []
        self._label_artists: list = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_content_changed(self, rect: "ContentRect") -> None:
        """A new image discards the result rather than redrawing it.

        The markers describe a correlation against the image that was underneath
        them; over a different one they are wrong, not merely stale. The old
        canvas made the same choice in ``set_image``, after "POI (P)" was left in
        the legend of an image that had no POI (FIB-321).
        """
        self.clear()

    # ── public API ────────────────────────────────────────────────────────

    def add_points(
        self,
        points: Sequence[Tuple[float, float]],
        *,
        color: str = "#ff0000",
        label_prefix: str = "",
        size: float = 7.0,
        marker: str = "o",
        alpha: float = 1.0,
        show_labels: bool = True,
        hollow: bool = False,
        legend_label: Optional[str] = None,
    ) -> None:
        """Append a styled group of markers, numbered ``label_prefix`` + 1, 2, ...

        ``hollow`` draws an unfilled ring, so the marker survives another one
        landing on top of it — which is the normal case for the uncorrected-POI
        ghost. ``legend_label`` adds a matching entry to the canvas legend.
        """
        group = _ResultGroup(
            points=[(float(x), float(y)) for x, y in points],
            color=color,
            label_prefix=label_prefix,
            size=size,
            marker=marker,
            alpha=alpha,
            show_labels=show_labels,
            hollow=hollow,
            legend_label=legend_label,
        )
        self._groups.append(group)
        self._draw_group(group)
        self._push_legend()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove every group added via :meth:`add_points`."""
        self._remove_artists()
        self._groups.clear()
        self._push_legend()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_labels_visible(self, visible: bool) -> None:
        """Show or hide the per-marker numbers, leaving the markers themselves.

        Driven by the same toggle as the picked points' labels: on a crowded
        result the numbers are the first thing an operator wants gone.
        """
        self._labels_visible = visible
        for ann in self._label_artists:
            ann.set_visible(visible)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def legend_entries(self) -> Tuple[List["Line2D"], List[str]]:
        """``(handles, labels)`` for the groups that asked for a legend entry."""
        handles, labels = [], []
        for group in self._groups:
            if not group.legend_label:
                continue
            handles.append(
                legend_handle(group.color, group.marker, group.hollow, group.alpha)
            )
            labels.append(group.legend_label)
        return handles, labels

    # ── drawing ───────────────────────────────────────────────────────────

    def _push_legend(self) -> None:
        if self._legend_host is not None:
            self._legend_host.set_extra_legend_entries(*self.legend_entries())

    def _remove_artists(self) -> None:
        for artist in self._artists + self._label_artists:
            try:
                artist.remove()
            except Exception:
                pass  # already gone with ax.cla()
        self._artists.clear()
        self._label_artists.clear()

    def _draw_group(self, group: _ResultGroup) -> None:
        if self._ax is None:
            return
        edge_color, edge_width = marker_edge(group.color, group.marker, group.hollow)
        for i, (x, y) in enumerate(group.points, start=1):
            (line,) = self._ax.plot(
                x, y,
                marker=group.marker,
                markersize=group.size,
                color=group.color,
                markerfacecolor="none" if group.hollow else group.color,
                markeredgecolor=edge_color,
                markeredgewidth=edge_width,
                linestyle="none",
                alpha=group.alpha,
                zorder=8,
            )
            self._artists.append(line)

            if not group.show_labels:
                continue
            label = f"{group.label_prefix}{i}" if group.label_prefix else str(i)
            ann = self._ax.annotate(
                label,
                xy=(x, y),
                xytext=LABEL_OFFSET,
                textcoords="offset points",
                color=group.color,
                fontsize=LABEL_FONTSIZE,
                alpha=group.alpha,
                zorder=9,
                path_effects=LABEL_OUTLINE,
            )
            ann.set_visible(self._labels_visible)
            self._label_artists.append(ann)
