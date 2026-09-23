"""The point-pair dialog behind "Fit from points" (FIB-1030).

The dialog does no geometry: it pairs the i-th point on each side, says how many pairs
there are, enables the fit at three, and hands the host's preview what the fit would
give. The host converts and applies.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.similarity import SimilarityFit
from fibsem.correlation.structures import PointType
from fibsem.ui.widgets.image_fit_dialog import MIN_PAIRS, ImageFitDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """The correlation canvases leave top-levels behind."""


def _dialog(preview=None):
    reference = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return ImageFitDialog(reference, image, preview=preview)


def test_pairs_are_the_ith_point_on_each_side(qapp):
    dialog = _dialog()
    dialog.add_pair((1.0, 2.0), (10.0, 20.0))
    dialog.add_pair((3.0, 4.0), (30.0, 40.0))
    assert dialog.pairs() == [(1.0, 2.0, 10.0, 20.0), (3.0, 4.0, 30.0, 40.0)]
    dialog.close()


def test_the_fit_is_offered_at_three_pairs_and_not_before(qapp):
    dialog = _dialog()
    for i in range(MIN_PAIRS - 1):
        dialog.add_pair((i, i), (i, i))
    assert not dialog.btn_fit.isEnabled()
    assert f"{MIN_PAIRS - 1} of {MIN_PAIRS} pairs" in dialog.label_status.text()

    dialog.add_pair((5.0, 5.0), (5.0, 5.0))

    assert dialog.btn_fit.isEnabled()
    dialog.close()


def test_the_preview_is_asked_and_its_answer_shown(qapp):
    asked = []

    def preview(pairs, fix_scale):
        asked.append((len(pairs), fix_scale))
        return SimilarityFit(
            scale=1.02, rotation=-3.5, translation=(0.0, 0.0), rms=1.25
        )

    reference = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    dialog = ImageFitDialog(
        reference,
        np.zeros((32, 32, 3), dtype=np.uint8),
        preview=preview,
        rms_text=lambda rms: f"RMS {rms * 0.5:.2f} um",
    )
    for i in range(3):
        dialog.add_pair((i * 10.0, 0.0), (i * 10.0 + 1.0, 0.0))
    text = dialog.label_status.text()
    # The host says what a canvas unit is; the dialog only repeats it.
    assert "RMS 0.62 um" in text and "-3.50°" in text and "×1.020" in text
    assert asked[-1] == (3, False)

    dialog.check_lock_scale.setChecked(True)

    assert asked[-1] == (3, True)
    assert dialog.fix_scale
    dialog.close()


def test_an_unpaired_extra_point_is_ignored_and_said(qapp):
    dialog = _dialog()
    for i in range(3):
        dialog.add_pair((i, i), (i, i))
    dialog._on_add_requested(9.0, 9.0, PointType.FIB)
    assert len(dialog.pairs()) == 3
    assert "extra is ignored" in dialog.label_status.text()
    dialog.close()
