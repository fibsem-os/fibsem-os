"""A contrast edit made in the layers panel survives a z-slice change (FIB-1173).

In planes mode the canvas holds one MIP-derived clim across planes, and did so by
turning `autocontrast` off. The panel read `autocontrast` for its Auto pill, so it
opened showing Auto off with the slider live; a drag wrote `clim` without marking
the channel `manual`, and the next z change wrote the held clim back over it. The
FIB-959 test set `manual` by hand and so never went through the panel.

These drive the panel's own controls and the z-slider only.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_fm_contrast_edit_survives_z_via_panel.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.ui.correlation.widgets.correlation_fm_canvas_widget import (
    CorrelationFMCanvasWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

_NZ = 5


def _image() -> FluorescenceImage:
    """One channel, five planes, each brighter than the last -- so a per-plane auto
    clim would differ from plane to plane and from the held MIP clim."""
    rng = np.random.default_rng(0)
    planes = rng.random((_NZ, 64, 64)) * 1000 * np.arange(1, _NZ + 1)[:, None, None]
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=470,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="green",
        objective_magnification=100,
        objective_numerical_aperture=0.85,
    )
    return FluorescenceImage(
        data=planes.astype(np.uint16)[None],
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-10-07T12:00:00",
            pixel_size_x=65e-9,
            pixel_size_y=65e-9,
            channels=[channel],
            z_positions=[i * 0.5e-6 for i in range(_NZ)],
        ),
    )


@pytest.fixture
def opened():
    """The correlation FM display with a stack loaded and the layers panel open."""
    w = CorrelationFMCanvasWidget()
    w.set_fm_image(_image())
    w._btn_layers.setChecked(True)
    w._toggle_layers_panel()
    yield w, w._panel, w.layers[0]
    w._close_layers_panel()
    w.close()


def _move_z(w) -> None:
    w._z_slider.setValue((w._z_slider.value() + 1) % (w._z_max + 1))


def test_panel_opens_on_auto_in_planes_mode(opened):
    w, p, layer = opened
    assert not w._max_projection
    assert p.autocontrast_cb.isChecked()
    assert not p.contrast.isEnabled()
    assert layer.manual is False


def test_contrast_drag_survives_z_change(opened):
    w, p, layer = opened
    p.autocontrast_cb.click()
    assert p.contrast.isEnabled()
    p.contrast.setValue((200, 400))
    dragged = layer.clim

    _move_z(w)
    assert layer.clim == dragged
    assert layer.manual is True
    _move_z(w)
    assert layer.clim == dragged


def test_a_contrast_write_marks_the_channel_manual(opened):
    """Every clim the panel writes is a user choice, whatever the pill says."""
    w, p, layer = opened
    p.contrast.setValue((200, 400))
    dragged = layer.clim
    assert layer.manual is True

    _move_z(w)
    assert layer.clim == dragged


def test_reset_returns_to_the_held_clim(opened):
    w, p, layer = opened
    held = layer.clim
    p.autocontrast_cb.click()
    p.contrast.setValue((200, 400))

    p.btn_reset.click()
    assert layer.manual is False
    assert p.autocontrast_cb.isChecked()
    assert layer.clim == held  # at once, not only after the next z move
    for _ in range(_NZ):
        _move_z(w)
        assert layer.clim == held


def test_auto_back_on_returns_to_the_held_clim_at_once(opened):
    w, p, layer = opened
    held = layer.clim
    p.autocontrast_cb.click()
    p.contrast.setValue((200, 400))

    p.autocontrast_cb.click()
    assert layer.manual is False
    assert layer.clim == held
