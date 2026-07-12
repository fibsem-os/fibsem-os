"""Offscreen tests for the reusable ContrastGammaControl popover."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("superqt")

from PyQt5.QtWidgets import QApplication, QWidget  # noqa: E402

from fibsem.ui.widgets.contrast_gamma_control import ContrastGammaControl  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_normalize_scales_to_unit_range(qapp):
    frame = np.array([[10, 20], [30, 50]], dtype=np.uint16)
    norm = ContrastGammaControl.normalize(frame)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_normalize_flat_frame_is_zeros(qapp):
    norm = ContrastGammaControl.normalize(np.full((4, 4), 7, dtype=np.uint8))
    assert np.all(norm == 0.0)


def test_is_default_and_apply(qapp):
    ctrl = ContrastGammaControl()
    assert ctrl.is_default()

    norm = np.linspace(0, 1, 100, dtype=np.float32)
    # gamma only
    ctrl.sld_gamma.setValue(0.5)
    assert not ctrl.is_default()
    out = ctrl.apply(norm)
    assert out.min() >= 0.0 and out.max() <= 1.0

    ctrl.reset()
    assert ctrl.is_default()


def test_min_max_anti_cross(qapp):
    ctrl = ContrastGammaControl()
    ctrl.sld_max.setValue(0.4)
    ctrl.sld_min.setValue(0.6)  # would cross max → clamped below it
    assert ctrl.contrast_min < ctrl.contrast_max


def test_changed_signal_fires(qapp):
    ctrl = ContrastGammaControl()
    fired = []
    ctrl.changed.connect(lambda: fired.append(True))
    ctrl.sld_gamma.setValue(1.5)
    assert fired


def test_popover_open_close(qapp):
    host = QWidget()
    ctrl = ContrastGammaControl(host)
    assert ctrl.isVisibleTo(host) is False
    ctrl.set_open(True)
    assert ctrl.isVisibleTo(host) is True
    ctrl.set_open(False)
    assert ctrl.isVisibleTo(host) is False
