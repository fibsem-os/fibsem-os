"""A contrast range the user set by hand survives a z-slice change (FIB-959).

`FMCanvasWidget._apply_z_mode` runs on every z-slider move. Its MIP / single-plane
branch already honoured `layer.manual` (FIB-743); the z-scrub branch did not, and
unconditionally wrote the MIP-derived auto clim back. The correlation FM display
defaults to planes rather than MIP, so every scrub there reverted a manual contrast
edit. Gamma is not touched by `_apply_z_mode`, which is why the reset looked
intermittent to users: gamma-only edits persisted, contrast edits did not.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_fm_canvas_manual_contrast_survives_z.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.correlation.widgets.correlation_fm_canvas_widget import (
    CorrelationFMCanvasWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _make_widget():
    w = CorrelationFMCanvasWidget()
    rng = np.random.default_rng(0)
    stack = (rng.random((3, 64, 64)) * 1000).astype(np.uint16)
    w._stacks = {"GFP": stack}
    layer = w._upsert_layer("GFP", stack.max(axis=0), "green")
    w._max_projection = False
    w._z_index = 0
    w._apply_z_mode()
    return w, layer


def test_manual_clim_and_gamma_survive_z_change():
    w, layer = _make_widget()
    layer.manual = True
    layer.autocontrast = False
    layer.clim = (100.0, 200.0)
    layer.gamma = 0.5

    w._on_z_changed(1)
    assert layer.clim == (100.0, 200.0)
    assert layer.gamma == 0.5
    assert layer.manual is True

    w._on_z_changed(2)
    assert layer.clim == (100.0, 200.0)
    w.close()


def test_auto_clim_is_held_constant_across_planes():
    w, layer = _make_widget()
    assert layer.manual is False
    first = layer.clim
    assert first is not None

    w._on_z_changed(1)
    assert layer.clim == first
    w._on_z_changed(2)
    assert layer.clim == first
    w.close()
