"""Multi-channel fluorescence compositing for the quad-view FM canvas.

Re-exports :mod:`fibsem.fm.composite`, which is the implementation. This module was
a second, byte-identical copy of it: the compositor was written here for the
quad-view canvas on PR #111 and lifted into `fibsem.fm` so the review panel could
use it without importing the UI package. #111 has landed, so this becomes the
re-export its counterpart's docstring always said it should.

Kept as a module rather than deleted because the canvas package is where a caller
working on the FM canvas looks for it, and both import paths are in use. There is
now one `FMLayer` class rather than two that merely looked alike -- which is what
made this worth doing rather than leaving alone.
"""
from __future__ import annotations

from fibsem.fm.composite import (
    AVAILABLE_COLORS,
    FMLayer,
    auto_clim,
    composite_fm_layers,
    tint_rgb,
)

__all__ = [
    "AVAILABLE_COLORS",
    "FMLayer",
    "auto_clim",
    "composite_fm_layers",
    "tint_rgb",
]
