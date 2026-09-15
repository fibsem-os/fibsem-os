"""The FIB/SEM overview settings column."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.structures import AutoContrastMode, OverviewAcquisitionSettings
from fibsem.ui.widgets.fibsem_overview_settings_widget import (
    FibsemOverviewSettingsWidget,
)


@pytest.fixture
def widget(qapp):
    w = FibsemOverviewSettingsWidget()
    w.show()
    return w


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
