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
    from fibsem.structures import FibsemImage


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
class CanvasState:
    """Everything shown on one canvas; the reducer renders from this."""

    image: Optional["FibsemImage"] = None
    overlays: Dict[str, OverlaySpec] = field(default_factory=dict)
    info: List[Tuple[str, str]] = field(default_factory=list)
    armed_overlay: Optional[str] = None  # id that owns edit input (None = view/move)


@dataclass
class SceneModel:
    """Per-canvas states for the quad view (SEM / FIB / FM)."""

    sem: CanvasState = field(default_factory=CanvasState)
    fib: CanvasState = field(default_factory=CanvasState)
    fm: CanvasState = field(default_factory=CanvasState)
