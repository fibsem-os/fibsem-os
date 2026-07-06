"""Headless smoke for the minimap overview contrast/gamma routing.

The overview is shown as an RGB composite, so the canvas's grayscale contrast can't touch
it — FibsemMinimapWidget._on_overview_contrast_changed drives the overview FMLayer's
clim/gamma instead. Tested via the unbound method against a fake self (no microscope).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_overview_contrast.py
"""
import sys
import types

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.widgets.canvas.fm_composite import FMLayer

_app = QApplication.instance() or QApplication(sys.argv)


def _self(layer, ctrl, recomposited):
    return types.SimpleNamespace(
        _overview_layer=layer,
        canvas=types.SimpleNamespace(_contrast=ctrl),
        _recomposite=lambda: recomposited.append(True),
    )


def test_manual_contrast_sets_layer_clim_gamma():
    layer = FMLayer(name="overview", data=np.array([[0, 100], [200, 255]], dtype=np.uint8))
    ctrl = types.SimpleNamespace(
        is_default=lambda: False, contrast_min=0.25, contrast_max=0.75, gamma=1.5
    )
    recomposited = []
    FibsemMinimapWidget._on_overview_contrast_changed(_self(layer, ctrl, recomposited))
    assert layer.autocontrast is False
    assert layer.clim == (0.25 * 255, 0.75 * 255)  # data range 0..255
    assert layer.gamma == 1.5
    assert recomposited == [True]


def test_default_contrast_restores_autocontrast():
    layer = FMLayer(
        name="overview", data=np.zeros((2, 2), dtype=np.uint8),
        autocontrast=False, clim=(10.0, 20.0), gamma=2.0,
    )
    ctrl = types.SimpleNamespace(
        is_default=lambda: True, contrast_min=0.0, contrast_max=1.0, gamma=1.0
    )
    FibsemMinimapWidget._on_overview_contrast_changed(_self(layer, ctrl, []))
    assert layer.autocontrast is True
    assert layer.clim is None
    assert layer.gamma == 1.0


def test_noop_without_overview_layer():
    def _boom():
        raise AssertionError("recomposite must not run without a layer")
    s = types.SimpleNamespace(
        _overview_layer=None,
        canvas=types.SimpleNamespace(_contrast=None),
        _recomposite=_boom,
    )
    FibsemMinimapWidget._on_overview_contrast_changed(s)  # must not raise


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
