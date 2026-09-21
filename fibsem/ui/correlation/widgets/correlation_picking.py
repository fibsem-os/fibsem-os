"""Correlation point picking over any `FibsemImageCanvas` (FIB-535).

**Goal: one implementation of "pick correlation points on a canvas",** so the FIB
and FM surfaces cannot drift apart. Both are moving onto the shared canvas stack
— `CorrelationCanvasWidget` for FIB, `FMCanvasWidget` for FM — and both need the
identical picking behaviour on top of a completely different image pipeline.

**Explicit non-goal: sharing image display.** The two hosts genuinely differ:
`set_image(ndarray)` on one, `set_fm_image(FluorescenceImage)` plus z-slice, MIP
and per-channel layers on the other. Conflating "shows an image" with "lets you
pick points on it" is what produced two 600–1000 line widgets in the first place.
This needs a canvas to hang overlays on and nothing else.

Composition rather than a mixin. A mixin would work — PyQt5 does pick up
`pyqtSignal` from a plain-object mixin — but its dependency on ``self.canvas``
would be implicit and unenforced: the attribute must exist, be a
`FibsemImageCanvas`, and be constructed *before* the mixin's own init runs. Here
all three are one constructor argument. It also matches the package, where
`.canvas` / `.points` / `.results` are already reach-through attributes and the
overlay system is composition end to end.

Hosts forward the four signals rather than making callers reach through
`.picking`, which keeps `correlation_tab_widget` and its `_CanvasAdapter`
untouched by the migration.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu, QWidget

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui.correlation.widgets.correlation_point_overlay import (
    CorrelationPointOverlay,
    CorrelationResultOverlay,
)


class CorrelationPicking(QObject):
    """Draggable correlation points and result markers on a shared canvas.

    Signal shapes are deliberately identical to ``ImagePointCanvas``: the tab
    widget's four handlers connect unchanged, via whichever host forwards them.
    """

    point_selected = pyqtSignal(object)  # Coordinate
    point_moved = pyqtSignal(object)  # Coordinate
    point_removed = pyqtSignal(object)  # Coordinate
    point_add_requested = pyqtSignal(float, float, object)  # x, y, PointType

    def __init__(
        self,
        canvas,
        *,
        allowed_point_types: Optional[List[PointType]] = None,
        menu_parent: Optional[QWidget] = None,
    ) -> None:
        """Attach picking to *canvas*.

        *menu_parent* is the widget the add-menu hangs from — a parameter rather
        than an assumption, because neither this nor the overlay is a QWidget and
        a QMenu needs one. *allowed_point_types*: None means "every type", an
        empty list means "adding is off" — the convention ``ImagePointCanvas``
        uses, kept so a read-only canvas cannot silently regain an add menu.
        """
        super().__init__(menu_parent)
        self._canvas = canvas
        self._menu_parent = menu_parent
        self._allowed_types = allowed_point_types

        self.points = CorrelationPointOverlay()
        canvas.add_overlay(self.points)
        # Result markers get their own overlay: they are computed output, never
        # picked, and keeping them off `points` is what makes them unpickable
        # rather than something to remember. Registered second so they draw over
        # the picked points, matching the old canvas's artist order.
        self.results = CorrelationResultOverlay(legend_host=self.points)
        canvas.add_overlay(self.results)

        # Identity signals straight through; only the add path needs translating,
        # because the overlay reports *where* and this decides *what*.
        self.points.coordinate_selected.connect(self.point_selected)
        self.points.coordinate_moved.connect(self.point_moved)
        self.points.coordinate_removed.connect(self.point_removed)
        self.points.add_requested.connect(self._show_add_menu)

        # Legend and label toggles. The shared toolbar has neither -- they are
        # correlation's chrome, not the canvas's -- but ImagePointCanvas carried
        # both, so without these the swap would quietly cost two controls that
        # exist today. Same icons and tooltips it used.
        self.btn_legend = canvas.add_toolbar_button(
            "mdi:format-list-bulleted",
            "Toggle legend",
            self.set_legend_visible,
            checkable=True,
        )
        self.btn_legend.setChecked(True)
        self.btn_labels = canvas.add_toolbar_button(
            "mdi:label-outline",
            "Toggle point labels",
            self.set_labels_visible,
            checkable=True,
        )
        self.btn_labels.setChecked(True)

    def dispose(self) -> None:
        """Unregister both overlays from the canvas.

        Not needed for teardown — the canvas and this object die together, and
        the overlay signals are ``pyqtSignal``, which Qt disconnects itself. It
        exists because the canvas, not this object, owns the overlay registry:
        dropping a `CorrelationPicking` leaves its overlays registered, so
        attaching a second one to the same canvas would silently stack a
        duplicate pair.
        """
        for overlay in (self.results, self.points):
            self._canvas.remove_overlay(overlay)

    # ── the _CanvasAdapter surface ────────────────────────────────────────

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        self.points.set_coordinates(coords)

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        self.points.set_selected_coordinate(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        self.points.refresh_coordinate(coord)

    # ── chrome ────────────────────────────────────────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the point-type legend (one swatch per type on screen)."""
        self.points.set_legend_visible(visible)
        self.btn_legend.setChecked(visible)

    def set_labels_visible(self, visible: bool) -> None:
        """Show or hide the per-point names. Independent of marker visibility --
        a crowded image often wants the points without the text.

        Covers the result markers' numbers too: one toggle, all the text, which is
        what the old canvas did and what makes the button mean something on an
        image showing a result."""
        self.points.set_labels_visible(visible)
        self.results.set_labels_visible(visible)
        self.btn_labels.setChecked(visible)

    # ── result markers ────────────────────────────────────────────────────

    def add_overlay_points(
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
        """Append a group of non-interactive result markers.

        Signature kept identical to ``ImagePointCanvas.add_overlay_points`` -- the
        tab widget's three calls move across untouched. Groups accumulate; call
        :meth:`clear_overlay` first when replacing a previous result.
        """
        self.results.add_points(
            points,
            color=color,
            label_prefix=label_prefix,
            size=size,
            marker=marker,
            alpha=alpha,
            show_labels=show_labels,
            hollow=hollow,
            legend_label=legend_label,
        )

    def clear_overlay(self) -> None:
        """Remove every marker added via :meth:`add_overlay_points`."""
        self.results.clear()

    # ── add menu ──────────────────────────────────────────────────────────

    def _show_add_menu(self, x: float, y: float) -> None:
        """Ask which PointType to add at *x*, *y*, then report it.

        The overlay cannot do this itself -- it is a QObject, not a widget, so it
        has no parent to hang a QMenu from, and the allowed-type list is per-side
        config the tab widget already owns. No stylesheet: QMenu is styled
        app-wide in napari_style, and ImagePointCanvas overriding that locally is
        the anomaly, not the rule.

        Emits nothing when adding is disabled (``allowed_point_types=[]``) or the
        menu is dismissed -- the point is created by the caller, not here, so the
        Coordinate still gets its z from the adapter as it does today.
        """
        types = (
            self._allowed_types if self._allowed_types is not None else list(PointType)
        )
        if not types:
            return

        menu = QMenu(self._menu_parent)
        for pt in types:
            action = QAction(f"Add {pt.value}", self._menu_parent)
            action.setData((x, y, pt))
            menu.addAction(action)

        chosen = menu.exec_(QCursor.pos())
        if chosen is not None:
            cx, cy, pt = chosen.data()
            self.point_add_requested.emit(cx, cy, pt)
