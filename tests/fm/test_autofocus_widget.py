"""Headless tests for the dynamic multi-pass AutofocusWidget.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, FocusMethod


@pytest.fixture
def channels():
    return [
        ChannelSettings(name="Reflection", excitation_wavelength=550, emission_wavelength=None),
        ChannelSettings(name="GFP", excitation_wavelength=488, emission_wavelength="FLUORESCENCE"),
    ]


def _widget(qapp, channels):
    from fibsem.ui.fm.widgets.autofocus_widget import AutofocusWidget
    return AutofocusWidget(channel_settings=channels)


def test_default_has_two_rows(qapp, channels):
    w = _widget(qapp, channels)
    assert len(w._rows) == 2
    assert len(w.autofocus_settings.passes) == 2


def test_add_remove_controls_hidden_by_default(qapp, channels):
    w = _widget(qapp, channels)
    assert not w._header.btn_add.isVisible()
    assert all(not row.btn_remove.isVisible() for row in w._rows)


def test_enable_editing_reveals_controls(qapp, channels):
    w = _widget(qapp, channels)
    w.show()
    w.set_pass_editing_enabled(True)
    assert w._header.btn_add.isVisible()
    assert all(row.btn_remove.isVisible() for row in w._rows)


def test_add_pass_appends_row(qapp, channels):
    w = _widget(qapp, channels)
    w._on_add_pass()
    assert len(w.autofocus_settings.passes) == 3
    assert len(w._rows) == 3


def test_remove_pass(qapp, channels):
    w = _widget(qapp, channels)
    w._on_remove_pass(w._rows[0])
    assert len(w.autofocus_settings.passes) == 1
    assert len(w._rows) == 1


def test_remove_last_pass_blocked(qapp, channels):
    w = _widget(qapp, channels)
    w._on_remove_pass(w._rows[0])
    w._on_remove_pass(w._rows[0])  # only one left — should be a no-op
    assert len(w.autofocus_settings.passes) == 1


def test_max_passes_enforced(qapp, channels):
    w = _widget(qapp, channels)
    for _ in range(10):
        w._on_add_pass()
    assert len(w.autofocus_settings.passes) == w.MAX_PASSES


def test_row_edit_updates_pass_and_emits(qapp, channels):
    w = _widget(qapp, channels)
    emitted = []
    w.settings_changed.connect(lambda s: emitted.append(s))
    row = w._rows[0]
    row.range_spin.setValue(123.0)  # µm
    assert row.sweep_pass.search_range == pytest.approx(123e-6)
    assert emitted


def test_set_get_round_trip(qapp, channels):
    w = _widget(qapp, channels)
    settings = AutoFocusSettings.from_coarse_fine(
        coarse_range=30e-6, coarse_step=6e-6,
        fine_range=8e-6, fine_step=2e-6,
        method=FocusMethod.SOBEL,
    )
    w.set_autofocus_settings(settings)
    assert len(w._rows) == 2
    assert w._rows[0].range_spin.value() == pytest.approx(30.0)
    assert w._rows[1].step_spin.value() == pytest.approx(2.0)
    got = w.get_autofocus_settings()
    assert got.method is FocusMethod.SOBEL
    assert got.passes[0].search_range == pytest.approx(30e-6)


def test_pass_steps_label(qapp, channels):
    w = _widget(qapp, channels)
    # default coarse pass: 20µm / 5µm → 4 steps; fine: 10µm / 1µm → 10 steps
    assert w._rows[0].steps_label.text() == "4"
    assert w._rows[1].steps_label.text() == "10"


def test_pass_steps_updates_on_edit(qapp, channels):
    w = _widget(qapp, channels)
    w._rows[0].step_spin.setValue(2.0)  # 20µm range / 2µm step → 10 steps
    assert w._rows[0].steps_label.text() == "10"


def test_disabled_pass_disables_spinboxes(qapp, channels):
    w = _widget(qapp, channels)
    row = w._rows[0]
    assert row.range_spin.isEnabled() and row.step_spin.isEnabled()
    row.checkbox.setChecked(False)
    assert not row.range_spin.isEnabled()
    assert not row.step_spin.isEnabled()
    row.checkbox.setChecked(True)
    assert row.range_spin.isEnabled()
    assert row.step_spin.isEnabled()


def test_disabled_pass_initial_state(qapp, channels):
    w = _widget(qapp, channels)
    w.autofocus_settings.passes[0].enabled = False
    w._rebuild_rows()
    assert not w._rows[0].range_spin.isEnabled()
    assert w._rows[1].range_spin.isEnabled()


def test_method_change(qapp, channels):
    w = _widget(qapp, channels)
    idx = w.comboBox_method.findData(FocusMethod.VARIANCE)
    w.comboBox_method.setCurrentIndex(idx)
    assert w.autofocus_settings.method is FocusMethod.VARIANCE


def test_channel_selection(qapp, channels):
    w = _widget(qapp, channels)
    w.comboBox_channel.setCurrentIndex(1)
    assert w.autofocus_settings.channel_name == "GFP"
    assert w.get_selected_channel().name == "GFP"


def test_update_channels_preserves_selection(qapp, channels):
    w = _widget(qapp, channels)
    w.set_selected_channel_by_name("GFP")
    new_channels = [channels[1], ChannelSettings(name="DAPI", excitation_wavelength=405, emission_wavelength="FLUORESCENCE")]
    w.update_channels(new_channels)
    assert w.autofocus_settings.channel_name == "GFP"
