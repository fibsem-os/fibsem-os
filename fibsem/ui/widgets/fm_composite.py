"""Multi-channel fluorescence compositing for the quad-view FM canvas.

Blends per-channel grayscale frames into one RGB image the way napari does on the
main tab — each channel tinted by its colour and **additively** summed — so the
matplotlib `FibsemImageCanvas` can show the colocalised multi-channel result.

Reusable + Qt-free: the coincidence viewer (which currently max-projects to
grayscale) could converge onto this later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.colors import to_rgb

# Canonical channel colours (mirrors channel_list_widget.AVAILABLE_COLORS).
AVAILABLE_COLORS = ["violet", "blue", "cyan", "green", "yellow", "red", "gray"]

_WHITE = (1.0, 1.0, 1.0)


@dataclass
class FMLayer:
    """One fluorescence channel's display state (separate from acquisition
    ``ChannelSettings``, which only carries ``name`` + ``color``)."""

    name: str
    data: Optional[np.ndarray] = None        # 2D (H, W)
    color: str = "gray"
    opacity: float = 1.0                      # 0..1
    clim: Optional[Tuple[float, float]] = None  # manual limits (used when not auto)
    visible: bool = True
    autocontrast: bool = True                 # recompute clim from data each composite
    gamma: float = 1.0                        # display = norm ** gamma (1 = linear)


def tint_rgb(color: str) -> Tuple[float, float, float]:
    """RGB tint for a channel colour. ``gray`` maps to white (black→white ramp,
    like napari's gray colormap); other names go through matplotlib."""
    if color == "gray":
        return _WHITE
    try:
        return to_rgb(color)
    except (ValueError, TypeError):
        return _WHITE


def auto_clim(data: np.ndarray) -> Tuple[float, float]:
    """Robust auto contrast limits (1st/99th percentile, min/max fallback)."""
    lo = float(np.percentile(data, 1.0))
    hi = float(np.percentile(data, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(data)), float(np.max(data))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def composite_fm_layers(
    layers: List[FMLayer], shape: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """Blend the visible *layers* into an ``(H, W, 3)`` uint8 RGB image.

    Each visible layer is contrast-normalised by its ``clim`` (or auto), tinted by
    ``color`` scaled by ``opacity``, and additively summed; the result is clipped.
    Returns ``None`` if there is nothing to show and *shape* is unknown.
    """
    visible = [l for l in layers if l.visible and l.data is not None]
    if not visible:
        return np.zeros((*shape, 3), dtype=np.uint8) if shape is not None else None
    H, W = shape if shape is not None else visible[0].data.shape[:2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for layer in visible:
        d = np.asarray(layer.data, dtype=np.float32)
        if d.shape[:2] != (H, W):
            continue  # ignore mismatched-shape channels
        if layer.autocontrast or layer.clim is None:
            lo, hi = auto_clim(d)
        else:
            lo, hi = layer.clim
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(d)
        if layer.gamma != 1.0:
            norm = np.power(norm, layer.gamma)
        tint = np.asarray(tint_rgb(layer.color), dtype=np.float32) * float(layer.opacity)
        rgb += norm[..., None] * tint
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
