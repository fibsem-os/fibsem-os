"""Offscreen tests for ImageSettingsWidget beam-type selector (opt-in)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.structures import BeamType, ImageSettings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {e}")
    return app


def test_beam_type_hidden_by_default_preserves_value(qapp):
    """A hidden beam-type selector must not clobber the stored beam_type — the
    widget only writes fields it exposes."""
    from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

    w = ImageSettingsWidget()  # show_beam_type defaults False
    assert w.beam_type_combo.isHidden() is True
    w.update_from_settings(ImageSettings(beam_type=BeamType.ION, hfw=150e-6))
    # edit an exposed field, then read back — beam_type stays ION
    w.hfw_spinbox.setValue(200.0)
    assert w.get_settings().beam_type == BeamType.ION


def test_beam_type_shown_roundtrips_and_emits(qapp):
    from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

    w = ImageSettingsWidget(show_beam_type=True)
    assert w.beam_type_combo.isHidden() is False
    seen = []
    w.settings_changed.connect(lambda s: seen.append(s.beam_type))

    w.update_from_settings(ImageSettings(beam_type=BeamType.ELECTRON))
    assert w.beam_type_combo.currentData() == BeamType.ELECTRON

    w.beam_type_combo.setCurrentIndex(w.beam_type_combo.findData(BeamType.ION))
    assert w.get_settings().beam_type == BeamType.ION
    assert seen and seen[-1] == BeamType.ION  # user change emitted


def test_set_show_beam_type_toggles_visibility(qapp):
    from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

    w = ImageSettingsWidget()
    assert w.beam_type_combo.isHidden() is True
    w.set_show_beam_type(True)
    assert w.beam_type_combo.isHidden() is False
