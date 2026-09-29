"""The FIB/SEM overview settings column: the auto contrast mode, and the advanced
imaging rows (the scan's integration controls) behind the Imaging panel's toggle,
read and written as the settings spell them."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.structures import (
    AutoContrastMode,
    ImageSettings,
    OverviewAcquisitionSettings,
)
from fibsem.ui.widgets.fibsem_overview_settings_widget import (
    FibsemOverviewSettingsWidget,
)


@pytest.fixture
def widget(qapp):
    w = FibsemOverviewSettingsWidget()
    w.show()
    return w


def test_the_advanced_rows_hide_behind_the_toggle(widget):
    rows = widget._advanced_fields + widget._advanced_labels
    assert all(not w.isVisible() for w in rows)
    widget.btn_advanced.setChecked(True)
    assert all(w.isVisible() for w in rows)
    widget.btn_advanced.setChecked(False)
    assert all(not w.isVisible() for w in rows)


def test_one_is_off_and_reads_as_none(widget):
    """The Image tab's convention: a value of 1 is no integration, spelled None
    on the settings, and drift correction needs frame integration to mean
    anything."""
    image = widget.get_settings().image_settings
    assert (image.line_integration, image.scan_interlacing) == (None, None)
    assert image.frame_integration is None and image.drift_correction is False
    assert not widget.check_drift_correction.isEnabled()

    widget.spin_line_integration.setValue(4)
    widget.spin_scan_interlacing.setValue(2)
    widget.spin_frame_integration.setValue(8)
    assert widget.check_drift_correction.isEnabled()
    widget.check_drift_correction.setChecked(True)
    image = widget.get_settings().image_settings
    assert (image.line_integration, image.scan_interlacing) == (4, 2)
    assert image.frame_integration == 8 and image.drift_correction is True

    # Frame integration back to 1 drops drift correction with it.
    widget.spin_frame_integration.setValue(1)
    assert not widget.check_drift_correction.isEnabled()
    assert not widget.check_drift_correction.isChecked()
    assert widget.get_settings().image_settings.drift_correction is False


def test_the_settings_round_trip(widget):
    settings = OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            line_integration=3,
            scan_interlacing=4,
            frame_integration=16,
            drift_correction=True,
        )
    )
    widget.update_from_settings(settings)
    assert widget.spin_line_integration.value() == 3
    assert widget.spin_scan_interlacing.value() == 4
    assert widget.spin_frame_integration.value() == 16
    assert widget.check_drift_correction.isChecked()
    image = widget.get_settings().image_settings
    assert (image.line_integration, image.scan_interlacing) == (3, 4)
    assert image.frame_integration == 16 and image.drift_correction is True

    widget.update_from_settings(OverviewAcquisitionSettings())
    assert widget.spin_frame_integration.value() == 1
    assert not widget.check_drift_correction.isChecked()


def test_auto_contrast_is_a_mode_and_drives_the_per_image_flag(widget):
    """Once, at the grid centre, is the default; each tile is the per-image
    meaning of the flag. The flag on the settings follows the mode, so a file
    written here reads back as the same choice."""
    settings = widget.get_settings()
    assert settings.autocontrast_mode is AutoContrastMode.ONCE
    assert settings.image_settings.autocontrast is False

    widget.combo_autocontrast.set_value(AutoContrastMode.EACH_TILE)
    settings = widget.get_settings()
    assert settings.autocontrast_mode is AutoContrastMode.EACH_TILE
    assert settings.image_settings.autocontrast is True

    widget.update_from_settings(
        OverviewAcquisitionSettings(autocontrast_mode=AutoContrastMode.NONE)
    )
    assert widget.combo_autocontrast.value() is AutoContrastMode.NONE
    assert widget.get_settings().image_settings.autocontrast is False
