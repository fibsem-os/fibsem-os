"""FM z-slider + max-projection behaviour for ``FMCanvasWidget``.

Feeds a synthetic CZYX stack (each z-plane a distinct constant) and checks the plane
selection + contrast mode as the max-projection checkbox / z-slider are driven. The
displayed 2-D plane is asserted on each channel's ``FMLayer.data`` (independent of the
RGB composite).

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_fm_z_slider.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget


def _fake_fm(data: np.ndarray, colors):
    """Minimal stand-in for FluorescenceImage: CZYX data + metadata (channels, pixel size)."""
    channels = [SimpleNamespace(name=f"ch{i}", color=colors[i]) for i in range(data.shape[0])]
    return SimpleNamespace(data=data, metadata=SimpleNamespace(pixel_size_x=1e-7, channels=channels))


def _layer(fmw, name):
    return next(l for l in fmw._layers if l.name == name)


def test_fm_z_slider_and_max_projection():
    nc, nz, h, w = 2, 5, 6, 6
    data = np.zeros((nc, nz, h, w), dtype=np.uint16)
    for c in range(nc):
        for z in range(nz):
            data[c, z] = c * 100 + (z + 1)  # distinct constant per (channel, z)

    fmw = FMCanvasWidget()
    assert not fmw._btn_mip.isVisibleTo(fmw)  # no stack yet → no MIP toggle (live microscope canvas)
    fmw.set_fm_image(_fake_fm(data, ["green", "red"]))
    _QAPP.processEvents()

    # default = max projection: MIP toggle shown for the stack, layers hold stack.max(0), z-slider hidden
    assert fmw._max_projection is True
    assert fmw._btn_mip.isVisibleTo(fmw)
    assert not fmw._z_row.isVisibleTo(fmw)
    assert np.array_equal(_layer(fmw, "ch0").data, data[0].max(0))
    assert _layer(fmw, "ch0").data.flat[0] == 5  # max over z of ch0 == 5

    # turn off Max projection via the canvas toolbar toggle
    fmw._btn_mip.setChecked(False)
    fmw._on_mip_button()
    _QAPP.processEvents()
    assert fmw._max_projection is False
    assert fmw._z_row.isVisibleTo(fmw)
    assert np.array_equal(_layer(fmw, "ch0").data, data[0][0])  # plane 0
    assert _layer(fmw, "ch0").data.flat[0] == 1
    # fixed clim while scrubbing (decision A)
    assert _layer(fmw, "ch0").autocontrast is False and _layer(fmw, "ch0").clim is not None
    clim0 = _layer(fmw, "ch0").clim

    # scrub to z = 3
    fmw._z_slider.setValue(3)
    _QAPP.processEvents()
    assert np.array_equal(_layer(fmw, "ch0").data, data[0][3])
    assert _layer(fmw, "ch0").data.flat[0] == 4
    assert _layer(fmw, "ch1").data.flat[0] == 104  # second channel scrubs in lock-step
    assert fmw._z_label.text() == "4/5"
    assert _layer(fmw, "ch0").clim == clim0  # clim held fixed across planes

    # turn Max projection back on: MIP, z-slider hidden, auto-contrast restored
    fmw._btn_mip.setChecked(True)
    fmw._on_mip_button()
    _QAPP.processEvents()
    assert fmw._max_projection is True
    assert not fmw._z_row.isVisibleTo(fmw)
    assert _layer(fmw, "ch0").autocontrast is True and _layer(fmw, "ch0").clim is None
    assert _layer(fmw, "ch0").data.flat[0] == 5

    # single-plane stack: no z-slider even with max-projection off
    fmw._btn_mip.setChecked(False)
    fmw._on_mip_button()
    _QAPP.processEvents()
    single = np.full((1, 1, h, w), 7, dtype=np.uint16)
    fmw.set_fm_image(_fake_fm(single, ["green"]))
    _QAPP.processEvents()
    assert fmw._z_max == 0
    assert not fmw._z_row.isVisibleTo(fmw)
    assert not fmw._btn_mip.isVisibleTo(fmw)  # single frame → no MIP toggle
    assert _layer(fmw, "ch0").data.flat[0] == 7

    # a live 2-D channel has no stack and is not scrubbable
    fmw.set_channel("live", np.full((h, w), 9, dtype=np.uint16))
    _QAPP.processEvents()
    assert "live" not in fmw._stacks


def test_manual_contrast_survives_live_frames():
    """Regression: the live FM feed pushes each frame through set_fm_image as a single-plane
    stack. _apply_z_mode must not force a channel back to auto-contrast when the user has set
    a manual contrast, or the live feed reverts the user's clim on every frame."""
    h = w = 16
    fmw = FMCanvasWidget()
    fmw.set_fm_image(_fake_fm(np.full((1, 1, h, w), 3, dtype=np.uint16), ["green"]))
    _QAPP.processEvents()

    # user turns Auto off and dials in a manual contrast (mirrors _on_autocontrast(False))
    layer = _layer(fmw, "ch0")
    layer.autocontrast = False
    layer.manual = True
    layer.clim = (100.0, 4000.0)

    # the next live frame arrives (another single-plane set_fm_image)
    fmw.set_fm_image(_fake_fm(np.full((1, 1, h, w), 7, dtype=np.uint16), ["green"]))
    _QAPP.processEvents()

    layer = _layer(fmw, "ch0")
    assert layer.autocontrast is False, "live frame reverted the channel to auto-contrast"
    assert layer.clim == (100.0, 4000.0), "live frame clobbered the manual clim"

    # user turns Auto back on: the next frame re-derives the clim (fix doesn't over-preserve)
    layer.autocontrast = True
    layer.manual = False
    fmw.set_fm_image(_fake_fm(np.full((1, 1, h, w), 9, dtype=np.uint16), ["green"]))
    _QAPP.processEvents()
    assert _layer(fmw, "ch0").clim is None, "auto channel should re-derive its clim"


if __name__ == "__main__":
    test_fm_z_slider_and_max_projection()
    test_manual_contrast_survives_live_frames()
    print("PASS")
