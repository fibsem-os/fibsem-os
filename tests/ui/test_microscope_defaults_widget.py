"""The Defaults panel: read from the instrument, edit, save to the configuration."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.structures import MicroscopeSettings  # noqa: E402
from fibsem.ui.widgets.microscope_defaults_widget import (  # noqa: E402
    MicroscopeDefaultsWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def toasts(monkeypatch):
    """Toasts go to a list, not to the notification service.

    The service is a module-level QObject owned by whichever QApplication first
    created it; a later test module's teardown can delete it, and the next toast
    raises "wrapped C/C++ object has been deleted" from inside the widget under
    test. The widgets' own behaviour is what these tests are about.
    """
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, notification_type="info": shown.append(
            (message, notification_type)
        ),
    )
    return shown


@pytest.fixture()
def microscope(tmp_path):
    import yaml

    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            utils.load_yaml(
                os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
            )
        )
    )
    microscope, _ = utils.setup_session(config_path=str(path), manufacturer="Demo")
    yield microscope
    microscope.disconnect()


@pytest.fixture()
def widget(qapp, microscope):
    w = MicroscopeDefaultsWidget()
    w.set_microscope(microscope)
    yield w
    w.close()
    w.deleteLater()


def test_disabled_until_there_is_a_microscope(qapp):
    w = MicroscopeDefaultsWidget()
    assert not w.isEnabled()
    w.deleteLater()


def test_a_loaded_form_has_no_changes(widget):
    assert not widget.is_modified()


def test_an_edit_is_a_change_and_saving_it_is_not(widget):
    changes = []
    widget.changed.connect(lambda: changes.append(widget.is_modified()))

    widget.electron.hfw.setValue(80.0)
    assert widget.is_modified()
    assert changes[-1] is True

    assert widget.save_to_configuration()
    assert not widget.is_modified()
    assert changes[-1] is False


def test_editing_back_to_the_saved_value_is_no_change(widget):
    saved = widget.electron.hfw.value()

    widget.electron.hfw.setValue(saved + 10)
    widget.electron.hfw.setValue(saved)

    assert not widget.is_modified()


def test_a_save_that_fails_keeps_the_change(widget, microscope, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(utils, "write_configuration", refuse)
    widget.electron.hfw.setValue(80.0)

    assert not widget.save_to_configuration()
    assert widget.is_modified()


def test_saved_values_are_what_was_typed(widget, microscope):
    """100 µm is written as 1e-4, not the 9.999999999999999e-05 the unit conversion
    leaves."""
    widget.electron.hfw.setValue(100.0)
    widget.ion.dwell_time.setValue(0.3)

    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]
    assert written["electron"]["hfw"] == 1e-4
    assert written["ion"]["dwell_time"] == 3e-7
    assert microscope.system.electron.beam.hfw == 1e-4


def test_beams_on_at_connect_is_a_change_and_is_saved(widget, microscope):
    assert widget.beams_on_at_connect.isEnabled()
    assert not widget.beams_on_at_connect.isChecked()  # the shipped file says false

    widget.beams_on_at_connect.setChecked(True)
    assert widget.is_modified()
    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]
    assert written["beams_on_at_connect"] is True
    assert written["apply_on_connect"] is False
    assert microscope.system.beams_on_at_connect is True
    assert not widget.is_modified()


def test_apply_on_connect_is_a_change_and_is_saved(widget, microscope):
    assert widget.apply_on_connect.isEnabled()
    assert not widget.apply_on_connect.isChecked()  # the shipped file says false

    widget.apply_on_connect.setChecked(True)
    assert widget.is_modified()
    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]
    assert written["apply_on_connect"] is True
    assert microscope.system.apply_defaults_on_connect is True
    assert not widget.is_modified()


def test_the_form_shows_the_configured_defaults(widget, microscope):
    assert widget.electron.voltage.value() == microscope.system.electron.beam.voltage
    # The shipped 1536x1024, not the first item in the list: a tuple did not
    # survive the combo box and the form showed 768x512 for every configuration.
    assert widget.electron.resolution.value() == "1536x1024"
    assert widget.electron.detector_type.value() == "ETD"
    assert widget.ion.voltage.value() == microscope.system.ion.beam.voltage
    assert widget.electron.hfw.value() == pytest.approx(
        microscope.system.electron.beam.hfw * 1e6
    )


def test_reading_from_the_microscope_takes_the_live_values(widget, microscope):
    live = microscope.get_microscope_state().electron_beam.voltage
    microscope.system.electron.beam.voltage = (live or 0) + 4321  # stale
    widget.show_system()
    assert widget.electron.voltage.value() != live

    widget.read_from_microscope()

    assert microscope.system.electron.beam.voltage == live
    assert widget.electron.voltage.value() == live


def test_saving_writes_the_defaults_section_and_nothing_else(widget, microscope):
    path = microscope.configuration_path
    before = utils.load_yaml(path)
    widget.ion.voltage.set_value(8000)
    widget.electron.hfw.setValue(80.0)

    widget.save_to_configuration()

    written = utils.load_yaml(path)
    assert written["defaults"]["ion"]["voltage"] == 8000
    assert written["defaults"]["electron"]["hfw"] == pytest.approx(80.0e-6)
    assert written["hardware"] == before["hardware"]
    assert written["calibration"] == before["calibration"]
    assert written["defaults"]["imaging"] == before["defaults"]["imaging"]
    assert written["defaults"]["apply_on_connect"] is False
    assert utils.unrecognised_configuration_keys(written) == []
    reloaded = MicroscopeSettings.from_dict(written)
    assert reloaded.system.ion.beam.voltage == 8000


def test_saving_writes_the_eight_keys_and_no_alignment_state(widget, microscope):
    """Not the beam shift, stigmation or working distance -- those are alignment
    state, and written here Apply would push them back."""
    microscope.system.electron.beam.working_distance = 0.0042

    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]["electron"]
    assert set(written) == {
        "voltage",
        "current",
        "hfw",
        "resolution",
        "dwell_time",
        "detector_type",
        "detector_mode",
        "scan_rotation",
    }


def test_saving_does_not_touch_the_column(widget, microscope):
    """Save decides what the defaults are; Apply is what pushes them."""
    live_before = microscope.get_beam_settings(microscope.system.ion.beam_type).voltage
    widget.ion.voltage.set_value(8000)

    widget.save_to_configuration()

    assert (
        microscope.get_beam_settings(microscope.system.ion.beam_type).voltage
        == live_before
    )


# ---------------------------------------------------------------------------
# Imaging: what the acquire tab opens with, `defaults.imaging`
# ---------------------------------------------------------------------------


@pytest.fixture()
def imaging_widget(qapp, microscope):
    settings = utils.load_microscope_configuration(microscope.configuration_path)
    w = MicroscopeDefaultsWidget()
    w.set_microscope(microscope, image_settings=settings.image)
    yield w, settings.image
    w.close()
    w.deleteLater()


def _acquire_tab_settings():
    from fibsem.structures import BeamType, ImageSettings

    return ImageSettings(
        beam_type=BeamType.ION,
        hfw=80e-6,
        resolution=(3072, 2048),
        dwell_time=0.5e-6,
        autocontrast=False,
    )


def test_the_imaging_form_shows_the_configured_defaults(imaging_widget):
    widget, image = imaging_widget
    assert widget.imaging.beam_type.value() == image.beam_type.name
    assert widget.imaging.hfw.value() == pytest.approx(image.hfw * 1e6)
    assert widget.imaging.resolution.value() == "1536x1024"
    assert widget.imaging.autocontrast.isChecked() is bool(image.autocontrast)


def test_reading_from_the_acquire_tab_fills_the_imaging_form(imaging_widget):
    widget, _ = imaging_widget
    assert not widget.button_read_imaging.isEnabled()  # no tab offered yet

    widget.set_current_imaging(_acquire_tab_settings)
    assert widget.button_read_imaging.isEnabled()
    widget.read_from_acquire_tab()

    assert widget.imaging.beam_type.value() == "ION"
    assert widget.imaging.hfw.value() == pytest.approx(80.0)
    assert widget.imaging.resolution.value() == "3072x2048"
    assert widget.imaging.dwell_time.value() == pytest.approx(0.5)
    assert widget.imaging.autocontrast.isChecked() is False


def test_saving_writes_the_imaging_defaults_and_keeps_save(imaging_widget, microscope):
    widget, image = imaging_widget
    before = utils.load_yaml(microscope.configuration_path)["defaults"]["imaging"]
    widget.set_current_imaging(_acquire_tab_settings)
    widget.read_from_acquire_tab()

    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)
    assert written["defaults"]["imaging"] == {
        **before,
        "beam_type": "ION",
        "hfw": pytest.approx(80e-6),
        "resolution": [3072, 2048],
        "dwell_time": pytest.approx(0.5e-6),
        "autocontrast": False,
    }
    assert utils.unrecognised_configuration_keys(written) == []
    reloaded = utils.load_microscope_configuration(microscope.configuration_path)
    assert reloaded.image.hfw == pytest.approx(80e-6)
    assert tuple(reloaded.image.resolution) == (3072, 2048)
    assert image.hfw == pytest.approx(80e-6)  # the session's record follows


def test_without_the_loaded_settings_imaging_is_left_alone(widget, microscope):
    """The panel was not told what the configuration was loaded with: showing the
    form's own starting values and saving them would overwrite the file."""
    before = utils.load_yaml(microscope.configuration_path)["defaults"]["imaging"]
    assert not widget.imaging.isEnabled()
    widget.set_current_imaging(_acquire_tab_settings)
    assert not widget.button_read_imaging.isEnabled()

    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]["imaging"]
    assert written == before


def test_disconnecting_forgets_the_acquire_tab(imaging_widget):
    widget, _ = imaging_widget
    widget.set_current_imaging(_acquire_tab_settings)

    widget.set_microscope(None)

    assert not widget.button_read_imaging.isEnabled()


def test_reading_one_beam_leaves_the_other_form_alone(widget, microscope):
    """Each beam's header reads that beam: an unsaved edit to the other survives."""
    widget.ion.voltage.set_value(8000)  # an edit not yet saved
    live = microscope.get_microscope_state().electron_beam.voltage
    microscope.system.electron.beam.voltage = (live or 0) + 4321  # stale

    widget.button_read_electron.click()

    assert widget.electron.voltage.value() == live
    assert widget.ion.voltage.value() == 8000


def test_scan_rotation_is_shown_in_degrees_and_saved_in_radians(widget, microscope):
    import math

    microscope.system.ion.beam.scan_rotation = math.pi
    widget.show_system()
    assert widget.ion.scan_rotation.value() == pytest.approx(180.0)

    widget.ion.scan_rotation.setValue(180.0)
    widget.save_to_configuration()

    written = utils.load_yaml(microscope.configuration_path)["defaults"]["ion"]
    assert written["scan_rotation"] == pytest.approx(math.pi)
    reloaded = utils.load_microscope_configuration(microscope.configuration_path)
    assert reloaded.system.ion.beam.scan_rotation == pytest.approx(math.pi)


def test_reading_a_beam_takes_its_scan_rotation(widget, microscope):
    import math

    from fibsem.structures import BeamType

    microscope.set_scan_rotation(math.pi, BeamType.ION)

    widget.button_read_ion.click()

    assert widget.ion.scan_rotation.value() == pytest.approx(180.0)
