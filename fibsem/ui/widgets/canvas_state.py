"""Declarative canvas-state model for the quad-view microscope display.

Pure data: one :class:`CanvasState` per canvas (image + overlays + info + the
armed overlay), aggregated by :class:`SceneModel`. Producers mutate this model
*only* through ``MicroscopeViewController`` (the reducer), which renders it onto
the matplotlib canvases in a single debounced pass.

Overlay specs (:class:`OverlaySpec` subclasses) are data records the reducer maps
onto the existing ``CanvasOverlay`` objects — they hold no Qt / matplotlib state,
so they stay trivially testable and thread-safe to construct.

See ``docs/design/canvas-overlay-state-model.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from fibsem.structures import FibsemImage, FibsemRectangle


@dataclass
class OverlaySpec:
    """Base for declarative overlay descriptions; ``id`` keys the overlay on a canvas."""

    id: str


@dataclass
class MillingSpec(OverlaySpec):
    """Milling stage patterns for a canvas (rendered by ``MillingPatternOverlay``).

    Carries no image: the reducer injects the canvas's current image when driving
    the overlay (``set_stages`` needs pixel-size + shape).
    """

    id: str = "milling"
    stages: Sequence = ()
    background_stages: Sequence = ()
    selected_index: Optional[int] = None


@dataclass
class AlignmentSpec(OverlaySpec):
    """Image-alignment reduced area — one overlay, two producers.

    ``rect`` is a normalized ``FibsemRectangle``. ``display`` = the milling viewer
    wants it shown read-only; ``editing`` = the image widget is editing it (wins).
    The reducer resolves: editable = editing, visible = editing or display.
    """

    id: str = "alignment"
    rect: Optional["FibsemRectangle"] = None
    display: bool = False
    editing: bool = False


@dataclass
class PointsSpec(OverlaySpec):
    """Interactive points (rendered by ``PointOverlay``) — POI / spot burn / detection
    features. ``points`` are (x, y) pixel coords; the rest configure the overlay at
    creation. ``colors`` / ``labels`` are optional index-aligned per-point overrides.
    """

    id: str = "points"
    points: Sequence = ()
    color: str = "cyan"
    selected_color: str = "yellow"
    marker: str = "o"
    size: float = 10.0
    label_prefix: str = ""
    add_on_right_click: bool = False
    removable: bool = False
    modal: bool = False
    colors: Optional[Sequence] = None
    labels: Optional[Sequence] = None


@dataclass
class CanvasState:
    """Everything shown on one canvas; the reducer renders from this."""

    image: Optional["FibsemImage"] = None
    overlays: Dict[str, OverlaySpec] = field(default_factory=dict)
    info: List[Tuple[str, str]] = field(default_factory=list)
    armed_overlay: Optional[str] = None  # id that owns edit input (None = view/move)
    armed_label: str = ""                # toolbar-mode label for the armed overlay
    armed_icon: str = ""                 # toolbar-mode icon for the armed overlay


@dataclass
class SceneModel:
    """Per-canvas states for the quad view (SEM / FIB / FM)."""

    sem: CanvasState = field(default_factory=CanvasState)
    fib: CanvasState = field(default_factory=CanvasState)
    fm: CanvasState = field(default_factory=CanvasState)
